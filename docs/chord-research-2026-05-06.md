# Chord-Detection Research Plan — 2026-05-06

**Author:** research agent
**Status:** plan only — no code changes proposed in this doc.
**Budget context:** $147 burned on chord R&D in the last month with little to show for it. $135 of that was Phi-3 (abandoned). Audit re-runs cost $0.06 × 17 = $1.02/pass.

---

## ⚠️ 2026-05-09 ADDENDUM — Anthropic chord corrector layer

**This plan was written before the Anthropic chord corrector was enabled in production.** The corrector was wired earlier but `ENABLE_ANTHROPIC_CORRECTION=true` + `ANTHROPIC_CORRECTION_MODE=full` flipped on at 2026-05-08 22:21 UTC. It post-processes the librosa output at `pipeline.py:546-549` (Claude Sonnet 4.5, ~$0.01-$0.02/song) — drops hallucinated chords, swaps chord names when key is wrong, relabels sections. Lyrics + bar timing never sent to Claude (legal posture per Passman ch.19).

**Re-audit on 2026-05-09 with full pipeline (librosa + corrector):**
- F1 avg = **0.77** on the 17-song rock set (vs May 5 raw 0.40, **Δ +37**)
- Precision 0.80 / Recall 0.76
- 3 perfect scores (Free Fallin', Hey Joe, House of the Rising Sun)
- 8 of 18 songs at A-grade (≥0.85)
- Major→minor cohort almost entirely fixed: Bad Company 0→91, Back In Black 33→86, Take On Me 60→91, Man in the Box 89
- Still failing: Sister Golden Hair (.40), Don't Let Me Be Misunderstood (.15)

**What this means for the experiments below:**
1. The TOP-3 (D / A / B) was scored against RAW librosa output. **The marginal value of further detector engineering is now smaller** because the corrector layer already cleans up most extension hallucination and many wrong-key cascades.
2. Experiment **D (Chordino feasibility gate)** is still worth running — but the bar is now "can Chordino + corrector beat librosa + corrector at 0.77?" not "can Chordino beat 0.40?"
3. Experiments **A (PC count gating)** and **B (PC energy m3-vs-M3)** target failure modes the corrector partially addresses. Validate against post-correction baseline before committing engineering time.
4. The 2 still-failing songs (Sister Golden Hair, Misunderstood) are the actual residual problem — investigate WHY the corrector doesn't recover those before designing new experiments.

**Operational note:** the May 9 audit took ~90 min wall time (vs ~5 min Modal separation time) due to post-separation semaphore contention — bottleneck is `pipeline.py:551-561`. See `docs/scaling-pipeline-2026-05-09.md`.

---

## 1. What I read (proof of context absorption)

- **Master state (`stemscriber_full_state.md`).** Project entered May 5 with the May 5 17-song rock audit landing at **40% avg, 2 songs at 0% (Bad Company, Man in the Box), 6 major→minor mislabels, 11 hallucinated chords, 13 missed chords**. The Apr 25 "8 of 10 at B+" result was Steely Dan / jazz fusion-heavy — the test set rewarded the detector's bias toward extensions. Power-chord rock exposes that same bias as the dominant failure mode. Strategic pivot: stop chasing UG-quality detection by June 20; ship "import your chart" (already live, `practice.html:2288`, `:7018`) and reposition product around stems + practice tools.
- **`stem_chord_detector.py`.** Active production path is `StemAwareChordDetector` (`processing/transcription.py:883-895`), with BTC/V8 hybrid as fallback only. Pipeline: Basic Pitch on each stem → onset-weighted PC voting (`:628`) → bass-root-first chord assembly (`:492`) → `_match_intervals_to_quality` template scoring (`:413`) → `_simplify_bleed_extensions` (`:1055`) → `_prune_outlier_chords` (`:964`) → K-K key detection (`:790`).
- **The May 5 0.93→0.95 experiment.** Documented inline at `:1190-1206`. **Bypassed** because `high_consistency_extended` (`:1147`) already fires for power-chord rock — the detector is so consistently wrong (every bar Amin7) that extension_rate > 0.75 + family_consistency ≥ 0.65 both pass, skipping the per-chord threshold gate entirely. Tweaking the threshold cannot fix the bypass logic.
- **Known root cause (per task brief, confirmed in audit log).** Distorted-guitar harmonics produce m3/b7 pitch-class evidence at attack time. In `_onset_weighted_pitch_classes_in_segment` (`:628`), the m3 PC arrives with a real onset and passes the gate; intervals `{0, 3, 7}` then exact-match the `min` template at `:455`, scoring `2.0 - priority_penalty` and dominating any major-triad subset bonus. The bug is upstream of `_match_intervals_to_quality` — by the time we score templates, the m3 is already in the set.
- **Already-built but partially used.** Tuning compensation exists at `chord_detector_v10.py:1178` (`librosa.estimate_tuning` + `pitch_shift`) — but **only in the BTC fallback path**. The primary stem-aware detector does NOT estimate or compensate tuning before Basic Pitch runs. This is consistent with the older `CHORD_DETECTION_ANALYSIS.md` (2026-03-09) recommending tuning compensation, getting half-implemented, and then forgotten when the stem-aware path became primary.
- **Key-detection cascade.** `detect_key_from_chords` (`:790`) uses chord-root histogram + K-K profile correlation, with a chord-quality polarity gate (`:881-886`) that hard-restricts to major or minor candidates based on whether ≥60% of chords are minor-family. When the detector floods a major-key song with min7s (Hells Bells: A major → 100% Am bars), this gate locks out the major candidates entirely, guaranteeing a minor key. Then `promote_diatonic_maj7` (`bass_root_extraction.py:411`) refuses to fire on minor keys, blocking the obvious recovery.
- **`_redetect_key_from_bargrid`** (`chart_formatter.py:379`). Re-runs key detection after `smooth_qualities` family-aware promotion, but feeds the same `detect_key_from_chords` — same K-K, same polarity gate, same lockout.

**Audit failure modes by root cause:**

| Pattern | Songs | Root cause |
|---|---|---|
| Major→minor mislabel | 6/13 (Hells Bells, Back In Black, Take On Me, Day After Day, Sister Golden Hair, Don't Let Me Be Misunderstood) | distorted-guitar m3 bleed → minor template exact-match → polarity gate locks key to minor |
| Wrong root entirely | 2/13 (Bad Company D#m vs G; Man in the Box D# vs E) | tuning offset OR Basic Pitch confusing distorted guitar fundamentals |
| Extension hallucination | 11/13 | `high_consistency_extended` bypass + bleed b7s on power-chord bars |
| Missed chords | 13/13 | every song misses 1-3 chords — secondary problem, downstream of the above |

---

## 2. Candidate experiments

### Experiment A — Pitch-class COUNT gating for sparse bars

- **Description.** In `notes_to_chord` (`:492`), before calling `_match_intervals_to_quality`, count `len(pitch_classes)`. If `≤ 3` PCs, restrict template choice to `{maj, min, sus2, sus4, 5, dim, aug}` — no 7ths, 9ths, 11ths, 13ths regardless of "exact match." Power-chord rock literally has 2-3 distinct PCs per bar; any extension detected on those bars is bleed by definition.
- **Hypothesis.** Eliminates extension hallucination on power-chord rock (Amin11 on Hells Bells, etc.) without touching jazz where bars routinely have 4-6 PCs.
- **Targets.** Failure modes #2 (extension hallucination) directly. Indirectly helps key detection because removing the m7 from `Amin7` reduces the polarity lock.
- **Implementation:** ~1-2 hours. Single guard in `notes_to_chord`.
- **Validation:** $1.02 audit + ~30 min eyeball. Steely Dan regression risk requires re-running the Apr 25 jazz songs alongside the rock 17 — call it a 17 + 5 = $1.32 / two-pass validation.
- **ROI: HIGH.** Cheap, targets the dominant failure mode, has a clear "don't touch jazz" safety property by construction.
- **Risk of no-op.** If `_simplify_bleed_extensions` later promotes triads back to extensions via `smooth_qualities` (`bass_root_extraction.py:336-356`) — the m3-detection priority pass triggers on ≥3 minor-family events. **Need to verify the simplification stack also respects PC count, otherwise the gate gets undone downstream.** Probably a 30-minute trace before committing.

---

### Experiment B — Pitch-class energy disambiguation for m3 vs M3 collision

- **Description.** Currently `_match_intervals_to_quality` handles "both 3 and 4 are present" (`:478-483`) with a tiny ±0.03 score adjustment based on which template explains the contradiction. That's symmetric — it doesn't actually distinguish. Replace with: when both intervals are present, look up the **energy** of pitch class `(root+3) % 12` vs `(root+4) % 12` from the discarded `pc_score` array (`:679-685`) and pick the higher-energy interval as "the real one." Strip the loser before template matching.
- **Hypothesis.** Distorted-guitar m3 bleed shows up at lower energy than the genuine major 3rd — the m3 is a 6th-harmonic artifact, the M3 is a fundamental. Energy ratio should reliably distinguish them.
- **Targets.** Failure mode #1 (major→minor mislabel) — the dominant audit failure. Direct attack on the known root cause.
- **Implementation:** 4-6 hours. Need to thread `pc_score` (currently a local in `_onset_weighted_pitch_classes_in_segment`) through to chord assembly. Means changing the return tuple of that function and threading through `detect_chords_from_stems:1378-1402`.
- **Validation:** $1.02 audit + 1-2 hours of per-song failure-mode tracing. Want to manually inspect 3 songs (Hells Bells, Back In Black, Take On Me) to confirm the m3-vs-M3 energy ratio actually splits the way the hypothesis says.
- **ROI: HIGH.** Targets the named root cause directly. Uses signal that's already computed and thrown away.
- **Risk of no-op.** Two scenarios: (a) the m3 from harmonics could carry comparable energy to the M3 fundamental on heavy-distortion bars (worst case: 6th-harmonic of A2 root is loud), in which case the energy ratio is noise; (b) Basic Pitch's frame-level activations may already be normalized in a way that crushes the energy difference before `pc_score` sees it. **Mitigate by spike-testing the hypothesis BEFORE doing the threading work** — 30-minute jupyter session: dump pc_score for 5 known-failing bars from Hells Bells and 5 known-correct bars from Aja, see if the m3 vs M3 energy ratio is meaningfully different.

---

### Experiment C — Tuning compensation in the stem-aware path

- **Description.** Port the `librosa.estimate_tuning` + `pitch_shift` block from `chord_detector_v10.py:1178-1183` into `stem_chord_detector.py`, applied to each stem audio file (or to a quick down-mix) BEFORE Basic Pitch runs. Optionally pass the offset to Basic Pitch instead of pitch-shifting the audio (would need to check Basic Pitch API).
- **Hypothesis.** The "Bad Company D#m vs G major" outlier (4-semitone-off, 0% audit score) and "Highway to Hell C# vs A" (also far off) are not random — they're systematic. The 2026-03-09 `CHORD_DETECTION_ANALYSIS.md` named tuning offset as the most likely culprit for "The Time Comes" F#m → G failure. We've never tested whether the same fix works on the stem-aware path.
- **Targets.** The 2 catastrophic 0% songs. Possibly other key-detection failures if the recordings happen to be ~50¢ flat.
- **Implementation:** 2-3 hours. The exact pattern is already written in `chord_detector_v10.py:1178-1183`. Decide whether to estimate per-stem or on a mix.
- **Validation:** $1.02 audit. Should specifically watch what happens on Bad Company and Man in the Box.
- **ROI: MEDIUM.** Could be a huge win on 2-3 outlier songs but probably won't move the average much. The named root cause (distorted guitar harmonics) isn't a tuning problem.
- **Risk of no-op.** Basic Pitch may internally do tuning estimation already (the model runs at fixed bin frequencies, so it's compensated by training augmentation). If so, this is a re-implementation of something already done, and the Bad Company failure has a different root cause that we'd misattribute to a tuning fix that wasn't actually doing anything. **Verify Basic Pitch's tuning behavior in the docs / source before implementing.**
- **Already partially built — flag.** Tuning code exists at `chord_detector_v10.py:1178` but never reaches the active stem-aware path. This is exactly the kind of half-implemented thing the brief warned about.

---

### Experiment D — Drop-in pre-trained chord baseline (Chordino / autochord) survey + bench

- **Description.** Run autochord (Bi-LSTM-CRF, 25 classes) and/or Sonic Annotator's Chordino plugin (NNLS-Chroma, well-tuned for guitar) against the 17-song audit set. Compare scores to current detector. If Chordino beats us on power-chord rock, use it as either (a) primary detector, (b) ensemble second-opinion (force-triad when Chordino + stem-aware disagree on quality), or (c) sanity check on key.
- **Hypothesis.** Chordino is 15+ years old, runs on mixed audio (no stem separation needed), and is famous for being decent on rock. We've never benchmarked against it. If it scores ≥60% on the 17-song audit — comparable to or better than our 40% — we have an ensemble lever, possibly a leapfrog.
- **Targets.** Whole-detector replacement OR ensemble disagreement signal for any failure mode.
- **Implementation:** Survey + install + run = 4-6 hours. autochord is `pip install autochord`. Chordino requires Vamp host; doable but more setup.
- **Validation:** Free — runs locally on existing audio files. No Modal cost.
- **ROI: HIGH.** Cheap to test, no engineering commitment until we know if it works. Tells us whether the ceiling on a free model is above or below where we are. Worth doing FIRST as a feasibility gate before building anything custom.
- **Risk of no-op.** If Chordino scores ~40% too, we've burned 4-6 hours and learned the negative result — but the negative result is itself important information (it tells us "no free lunch is going to fix this"). The likely outcome is somewhere in between: Chordino does better on some rock songs, worse on jazz. That outcome IS the ensemble lever.

---

### Experiment E — Fix the K-K minor-polarity lockout

- **Description.** The minor-polarity gate in `detect_key_from_chords:881-886` hard-restricts K-K candidates to minor when ≥60% of chord events are minor-family. This is a downstream cascade: if every bar is mislabeled Am, the gate locks minor; if the gate locks minor, no major key can win; if no major key wins, `promote_diatonic_maj7` (only fires for major) is dead. **Replace the hard gate with a soft penalty** — apply a 0.85 multiplier (or similar) to the disfavored polarity rather than excluding it. Genuine A-major songs labeled 100% Am can recover if the K-K profile correlation overwhelmingly favors A major.
- **Hypothesis.** The hard gate is what makes the major→minor failure unrecoverable. Softening it creates a path back even when the chord events are wrong.
- **Targets.** Failure mode #1 (major→minor mislabel) — but as a *recovery path*, not a *prevention path*. Works in conjunction with A or B (which prevent the mislabel in the first place).
- **Implementation:** 1-2 hours. Tiny change inside `detect_key_from_chords`.
- **Validation:** $1.02 audit. Watch for false major detections on genuine minor-key songs (Cosmic Girl, Aja, Alright) — REGRESSION risk if the soft penalty is too soft.
- **ROI: MEDIUM.** Direct attack on the cascade, but only matters if the underlying mislabel persists. If A or B fixes the mislabel, this is unnecessary. If neither lands, this is the band-aid that makes the symptom less catastrophic.
- **Risk of no-op.** If chord-root histogram alone (the K-K input) is too biased by the wrong roots, even relaxed K-K can't recover. Likely partial improvement at best.

---

### Experiment F — PC-class K-K key detection (audio-direct, not chord-derived)

- **Description.** Replace the chord-derived K-K input in `detect_key_from_chords` with a song-level pitch-class energy histogram aggregated directly from `_onset_weighted_pitch_classes_in_segment`'s `pc_score` arrays across all segments. Run K-K against THAT histogram. The current implementation (`:828-855`) builds the histogram from chord-implied PCs (root + 3rd + 5th of detected chords) — circular: wrong chords → wrong histogram → wrong key.
- **Hypothesis.** Aggregating raw PC energy across the whole song is more robust to per-bar chord mislabels. A song that's actually in A major spends most of its PC energy on `{A, C#, E, F#, G#, B, D}` — even if every bar gets called Am.
- **Targets.** All 6 major→minor mislabels + 2 catastrophic-key failures, AS A RECOVERY PATH (same as E, different mechanism).
- **Implementation:** 3-5 hours. Need to either (a) accumulate `pc_score` arrays during chord assembly and pass them to the key detector, or (b) re-extract them from saved frame data. (a) is cleaner, requires touching `detect_chords_from_stems:1364-1423`.
- **Validation:** $1.02 audit + jazz regression check.
- **ROI: MEDIUM.** Theoretically sounder than E (closer to the audio, less dependent on intermediate label noise) but also more invasive. Worth doing if E proves insufficient.
- **Risk of no-op.** Distorted guitar puts a LOT of PC energy on harmonic-overtone PCs — the m3 and b7 — which would still pull a major-key song toward its parallel minor. Could be a no-op for the same reason the chord assembly fails: the input PC distribution is contaminated by the same bleed.

---

### Experiment G — Spectral-distortion gate as a metadata signal

- **Description.** Compute `librosa.feature.spectral_flatness` over the source guitar stem. When flatness is high (distorted), tag those bars and either (a) discount Basic Pitch m3/b7 evidence by 0.5, or (b) force-triad those bars at chord assembly, or (c) just propagate "distortion = true" as metadata that downstream simplification respects.
- **Hypothesis.** Spectral flatness reliably distinguishes "clean stem with discrete pitches" from "distorted stem with smeared harmonic content." If we can detect distortion algorithmically, we can apply the right policy automatically — strict triads on distorted bars, full vocabulary on clean bars.
- **Targets.** Same as A (extension hallucination) but with a more principled discriminator.
- **Implementation:** 4-6 hours. New feature extraction + threading through to chord assembly.
- **Validation:** $1.02 audit + ground-truth distortion labels for spot-checking. Need to confirm that distorted-guitar bars and clean-guitar bars actually have separable spectral flatness in our stem outputs (BS-RoFormer's "guitar" stem may already smooth this).
- **ROI: MEDIUM.** Cleaner story than A (data-driven gate vs. PC-count heuristic), but more engineering work for what's likely a similar outcome. A is the cheap version; G is the sophisticated version. **Probably skip G unless A is insufficient.**
- **Risk of no-op.** BS-RoFormer's stem separation may itself smooth the spectral characteristics that distinguish distorted from clean — the "guitar" stem might look similar in flatness regardless of source distortion.

---

## 3. Recommended TOP-3, ordered

1. **D — Chordino / autochord baseline survey.** No code commitment, highest information gain per hour. Tells us whether the answer is "use a free model" instead of building anything. If a 15-year-old open-source model scores 60% on rock and we're at 40%, the right move is to ensemble or replace, not tune our own. **Do this first** — it's a feasibility gate.
2. **A — PC count gating.** If D doesn't immediately produce a leapfrog, A is the cheapest direct fix to the dominant failure mode. ~2 hours implementation, $1.32 validation, attacks extension hallucination by construction (sparse bars can't have extensions). Lowest implementation risk of the targeted experiments.
3. **B — PC energy m3-vs-M3 disambiguation.** If A drops the extension hallucinations but the major→minor mislabels persist, B is the next layer. Targets the named root cause directly using signal that's already computed. The 30-minute spike test on the energy ratio hypothesis (before any threading work) limits downside.

**Why this ordering:** D is information, A is the cheapest fix, B is the targeted fix. If D yields a Chordino ensemble, A and B may be reframed as "fix our own detector for jazz songs where Chordino loses, defer to Chordino on rock." That sequencing also matches Jeff's pivot — if `My Chart` paste is the practical answer for users, our detector job is to be "good enough by default, beatable by paste" — and Chordino-as-default may already meet that bar without us doing custom work.

---

## 4. Considered but rejected

- **Confidence threshold tuning in `_simplify_bleed_extensions`** — already proven no-op (May 5: 45% → 45%). The `high_consistency_extended` bypass (`:1147`) is the actual gate; threshold knob is dead code on power-chord rock.
- **Phi-3 / any LLM that produces chord names from audio** — abandoned, $135 burned, repetition-loop hallucination. Brief explicitly excludes.
- **Scraping UG / Songsterr / Chordify for ground truth** — lawyer killed (Apr 10, see `project_lawyer_apr10.md`).
- **Klangio API** — declined May 2 after a week of outreach. No going back.
- **Training a custom CRNN on power-chord rock** — would need ~months of data collection on a domain we have no labeled corpus for, and we have a launch in 6 weeks. Brief explicitly says "no fine-tuning that produces chord names from audio."
- **Adding more fallback levels (BTC v10 → V8 → V7 → basic)** — already tried via the existing fallback chain. The active path is stem-aware; fallbacks fire only on stem-aware failure, and stem-aware doesn't fail on rock — it just succeeds *wrongly*. Adding more fallbacks doesn't help when the primary returns wrong-with-confidence.
- **HMM/CRF temporal smoothing** — would help marginal cases but doesn't fix the underlying mislabel; same chord wrong for the whole song stays wrong after smoothing.
- **Beat-aligned chord-change snapping** — already implicit in `combine_with_detector_quality` + `smooth_qualities`. Marginal further gain unlikely to move the rock audit.
- **More training-data augmentation for BTC** — Apr 24's `btc_retrain_augmented.py` exists but the active path is stem-aware, not BTC. Re-training BTC fixes a detector that isn't running.

---

## 5. Next decision

**Jeff's call:** kick off D (Chordino survey, no Modal cost) and report back before committing to A/B/C. If D produces a free-model ensemble path, the conversation about A/B/C changes shape.

If D is unappealing: A is the cheapest standalone bet. ~2 hours + $1.32 to know whether it moves the audit.

Either way, do not re-tune `_simplify_bleed_extensions` thresholds.
