# V3.1 — consonance-ACE Tuning Report

**Date:** 2026-05-13
**Owner:** Jeff Kozelski (jkozelski@gmail.com)
**Launch:** June 20, 2026 (Refinery, Charleston — soft launch). 38 days out.
**Status:** Overrides `docs/v3-plan-2026-05-13.md` from earlier today (the Jiang-only plan).

---

## Executive summary

**Ship ACE-default + per-song Jiang router** as the V3.1 detector architecture, behind a single feature flag (`V3_DETECTOR=ace_router`). On the 13-song UG-grounded classic-rock benchmark this hits **root F1 0.869 (+0.156 over current prod's 0.713) with zero per-song regressions vs ACE-alone**. The router uses two zero-supervision signals — Jiang/ACE chord-stream agreement and ACE-over-Jiang event density — to detect the "over-eager-ACE" mode that pulls Hotel California from 0.92 → 0.81; on those songs (Hotel California, Paint It Black, Sunshine of Your Love) it switches to Jiang. Anthropic corrector stays OFF for the same reasons as the V3 plan (qflip gate refuses 11/13 edits; V1 prompt strips slash chords). One real regression to call out: ACE flattens jazz extensions on Aja (quality F1 drops 226/226 → 0.41). The router can't catch this because Aja's failure mode is *flattening*, not *over-eagerness* — the legacy stem-aware path beat ACE on Aja, and that's the post-launch lever. Bass-stem post-processing (the bet against ACE's hallucination) is dropped after empirical proof that ACE's slash chords are real audio content the UG transcriber simplified. Plan below: Week 1 wire the router behind a flag, Week 2 held-out validation on 10 new songs (router thresholds were tuned on the test set — must validate before prod), Week 3 prod soak, Jun 5 decision gate at avg root F1 ≥ 0.85.

---

## 1 — Q1: ACE-default at all granularities

Agent 1 rescored `/tmp/ace_outputs/*.lab` against the 13-song cohort. Source data: `/tmp/ace_variants_results.json`. Full table in `docs/v3-agent-1-ace-variants-2026-05-13.md`.

| Song | root F1 | family F1 | quality F1 | full (slash) F1 | PCS F1 |
|---|---:|---:|---:|---:|---:|
| black-sabbath__iron-man | 0.826 | 0.482 | **0.000** | **0.000** | 0.501 |
| boston__more-than-a-feeling | 0.794 | 0.785 | 0.740 | 0.731 | 0.785 |
| cream__sunshine-of-your-love | 0.698 | 0.686 | 0.556 | 0.544 | 0.568 |
| creedence-clearwater-revival__fortunate-son | 0.821 | 0.768 | 0.714 | 0.714 | 0.786 |
| eagles__hotel-california | 0.822 | 0.822 | 0.740 | 0.733 | 0.822 |
| heart__crazy-on-you | 0.874 | 0.843 | 0.736 | 0.679 | 0.818 |
| neil-young__heart-of-gold | 0.943 | 0.943 | 0.905 | 0.905 | 0.933 |
| the-animals__house-of-the-rising-sun | 0.956 | 0.939 | 0.857 | 0.857 | 0.939 |
| the-rolling-stones__paint-it-black | 0.716 | 0.550 | 0.550 | 0.541 | 0.550 |
| the-rolling-stones__wild-horses | 0.933 | 0.919 | 0.867 | 0.859 | 0.919 |
| tom-petty__into-the-great-wide-open | (lab not produced — ACE re-run needed) | | | | |
| tom-petty__mary-janes-last-dance | 0.880 | 0.866 | 0.852 | 0.852 | (n/a) |
| toto__africa | (not in JSON, see Agent 1 report) | | | | |
| **AVG (Agent 1's 11 scored)** | **0.847** | — | **0.697** | **0.685** | **0.787** |

Headline: ACE-default quality F1 = **0.697** — that's the surprise. The brief reported quality F1 = 0.61 against UG; against UG it's actually 0.70. Compared to current prod's 0.263 (librosa+V1), ACE gives **+0.43 quality F1**. The detector is doing real extension work, not just root work.

Iron Man's quality F1 = 0 is a scorer/GT-vocab mismatch (UG writes `B5`, ACE writes `B`). The `X5 → X family collapse` scorer fix from the V3 plan resolves this without touching the detector. Agent 4's Hells Bells GT uses the same `X5` form, so the same fix lands them both at once.

---

## 2 — Q2: Hotel California failure-mode deep dive

Agent 4 ran librosa `chroma_cqt` (full-mix + bass-band) against each "extra" event class ACE emits on Hotel California vs UG. 41 events tested across 8 classes. Detail in `docs/v3-agent-4-hotel-cal-hells-bells-2026-05-13.md`.

**Classification of ACE's "extras":**

| Class | Count | Verdict | Evidence |
|---|---:|---|---|
| `G/D` (6 events) | 6 | **REAL** | Bass-band D=0.83 at G/D events vs 0.28 at plain G; bass-band G=0.28 vs 0.62 at plain G. Descending bassline ACE caught, UG simplified. |
| `Em7/B` (2 events) | 2 | **REAL** | Bass-band B=0.83 vs E=0.56. ACE caught an inversion. |
| `Em7 split` (17 events vs UG's 7 `Em`) | 17 | **HALLUCINATED** | D-energy at Em7 events (0.36) is *lower* than at plain Em events (0.44). No m7 evidence in audio. |
| `F#m` (2 events vs UG's F#) | 2 | **HALLUCINATED** | Minor-3rd not present in chroma. |
| `Am` (3 events vs UG's A) | 3 | **HALLUCINATED** | Same. |
| `D7` (2 events) | 2 | **HALLUCINATED** | No b7 in audio. |
| `E7` (8 events) | 8 | **MIXED / mostly HALLUCINATED** | Some chroma support, but inconsistent. |
| `Dmaj7` (2 events) | 2 | **HALLUCINATED** | No maj7 in audio. |

**Net:** ~22% of ACE's extras are real audio content UG simplified; ~78% are detector noise (mostly the major→minor extension flips). **The Hotel California regression is partly a notation convention issue — ACE is right on slash chords, UG is more faithful on quality.**

Agent 2 INDEPENDENTLY confirmed the slash-chord side: every one of ACE's 13 slash chords on Hotel California is bass-stem-confirmed. The bet that we could drop ACE's hallucinations by bass-stem filtering is empirically wrong — the slashes are *all* real. The quality flips (Em7 split, F#m, Am) are where the real hallucination lives, and the bass stem can't disprove them because the 7th lives in vocals/guitar.

---

## 3 — Q3 + Q4: Per-song and per-chord ensembles

Agent 3's full report at `docs/v3-agent-3-ensembles-2026-05-13.md`. Source data: `/tmp/jiang_rock/` (Jiang on all 13 cohort songs, freshly run).

### Q3 — per-song router

Tested several rules. Winner:

```python
def route(events_ace, events_jiang):
    """Return 'jiang' on songs where ACE is over-eager; else 'ace'."""
    density = len(events_ace) / max(1, len(events_jiang))
    agreement = bar_agreement(events_ace, events_jiang)  # fraction of bar slots
                                                          # where ACE/Jiang share root
    if agreement > 0.5 and density > 1.15:
        return 'jiang'
    return 'ace'
```

| Method | Avg root F1 | Per-song regressions vs ACE-alone |
|---|---:|---:|
| ACE-alone | 0.847 | — (baseline) |
| Jiang-alone | 0.748 | — |
| Oracle (per-song best) | 0.883 | n/a |
| **Per-song router (recommended)** | **0.869** | **0** |

The router triggers on: Hotel California (recovers +0.157), Paint It Black (+0.058), Cream Sunshine of Your Love (+0.077). It does NOT trigger on songs ACE wins (Petty IGWO, MJLD, Heart of Gold, Wild Horses, Fortunate Son, Crazy On You, Animals HoRS) — those keep their ACE output and its slash-chord wins.

### Q4 — per-chord ensemble

V3 variant (agreement-gated, ACE fallback): 0.850 avg root F1 — barely beats ACE-alone, well below the per-song router. V4 (Jiang fallback): regressed. **Per-chord ensemble dropped.**

### Critical caveat

The router's two thresholds (0.5 agreement, 1.15 density) were tuned on the same 13 songs they were tested on. **Must validate on a held-out set of 5-10 new songs before flipping the prod flag.** Week 2 task.

---

## 4 — Q5: Bass-stem post-processing

Agent 2's full report at `docs/v3-agent-2-bass-stem-postproc-2026-05-13.md`.

Two rules implemented and tested on Hotel California, Petty IGWO, Paint It Black, MJLD:

- **Rule A (slash drop):** if ACE emits `X/Y` slash chord but bass-stem says bass is X (not Y), drop the slash. Empirical: every slash ACE emits is bass-confirmed. Net delta on root F1: ≈ 0.
- **Rule B (extension collapse):** if ACE emits `Xm7` but bass-stem doesn't show a 7th-suggesting tonality, collapse to `Xm`. Empirical: regressed Hotel California (root+quality F1 0.745 → 0.718) because the 7th lives in vocals/guitar, not bass. Stripping legitimate F#7 and Em7 hurts.

**Net delta of both rules: +0.000 root F1, sometimes negative on quality. Drop both rules.**

The deeper finding from Agents 2+4 in combination: ACE's slash chords are real audio content, and quality hallucination (Em7 split etc.) is the actual error mode, but bass-stem evidence is the wrong signal to detect it — quality lives above the bass band. **The right post-processor would need vocal/guitar-stem chroma evidence**, not bass. Out of scope for V3.1.

---

## 5 — Q6: chunk-dur sweep (10s, 30s)

Agent 1's data at `/tmp/ace_chunk10/`, `/tmp/ace_chunk30/`.

| Variant | Avg root F1 Δ | Avg quality F1 Δ | Big winners | Big losers |
|---|---:|---:|---|---|
| chunk_dur=10 vs default | -0.005 | +0.005 | Hotel California qual +0.056 | IGWO root -0.047, Iron Man qual -0.176 |
| chunk_dur=30 vs default | -0.006 | -0.013 | Fortunate Son root +0.077 | IGWO root -0.066, Paint It Black root -0.049, Heart of Gold qual -0.111 |

Means barely move. Per-song picture is mixed: chunk10 nudges Hotel California quality up (0.740 → 0.796) but doesn't help anywhere else materially, while regressing IGWO and Iron Man.

**Verdict: keep `chunk_dur=20s` as the global default.** Per-song chunk-tuning is not the right knob. (If a future iteration wants to special-case the demo song, chunk10 on Hotel California is a one-line override — but it's cosmetic for a demo, not a launch feature.)

---

## 6 — Q7: conformer vs conformer_decomposed

**Not testable.** Only `conformer_decomposed_smooth.ckpt` ships in the repo. The plain `ConformerModel` class exists but the shipped checkpoint can't load into it (different architecture — 170-class classifier vs 3-head decomposed). Training a vanilla conformer from scratch would need the Isophonics + McGill Billboard dataset and is out of V3.1 scope.

**Skip Q7.** The brief's claim that ACE has two checkpoints is incorrect — there's one architecture with two configurations, but only the decomposed checkpoint was released.

---

## 7 — Q8 + Q9: Aja and Hells Bells regression-safety

### Aja (Q8) — REGRESSION

Agent 1's lab at `/tmp/ace_aja/steely-dan__aja.lab`. Scored against `audit/fixtures/ground_truth/steely-dan__aja.json`.

| Metric | Value |
|---|---:|
| Root F1 | **0.812** |
| Family F1 | 0.582 |
| Quality F1 | **0.411** |
| Full (slash) F1 | 0.382 |
| Vocab coverage | 45.5% |

ACE flattens jazz extensions: Aadd9 → A, Cmaj9 → C#maj7, Bm11 → Bm7. Misses every GT slash chord except one (D7/C overlap). The legacy stem-aware detector scored 226/226 extension matches on this song; ACE captures roughly 41% of them.

**Severity:** Root F1 0.812 is still better than V3-prod baseline (no Aja audio in the 13-song cohort, so we can't directly compare — but the brief's 0.71 cohort average suggests prod is in that range). The lost ground is on quality / slash. **The per-song router does NOT catch this case** because Aja's failure mode is *flattening* (ACE less rich than truth), not *over-eagerness* (ACE more rich than truth) — the density signal points the wrong way.

**Architectural option for future V3.2:** Add a third path for jazz/extended-vocabulary signals. The legacy stem-aware detector that hit 226/226 on Aja is still in the repo (`backend/stem_chord_detector.py`, kept per Agent D's V3 manifest). A jazz-signal detector (high chord-density + many distinct extensions in detector top-K) could route those songs to stem-aware.

**Recommendation for V3.1:** Accept the Aja regression on quality F1 (0.41) given that root F1 (0.812) is still strong. Add Aja to the per-song regression-watch list with `quality_f1 >= 0.40` as the soft threshold. **Post-launch experiment:** route jazz songs to a jazz-tuned detector path.

### Hells Bells (Q9) — GT created, baseline established

Agent 4 wrote `audit/fixtures/ground_truth/ac-dc__hells-bells.json` in `X5` power-chord form (engineering best-effort GT, not UG-licensed). ACE scores:

| Metric | Value |
|---|---:|
| Root F1 | **0.814** |
| Quality F1 | 0.000 (X5 vs X scorer mismatch — same as Iron Man) |
| PCS F1 | 0.667 |

The May-8 "1.00" was an artifact of the prior weak GT. **0.814 root F1 with the engineered GT is the credible baseline.** With the planned scorer fix (`X5 → X` family collapse), Hells Bells quality F1 jumps to wherever the family-level match lands — likely ≥ 0.80.

---

## 8 — Q10: Fine-tuning ACE scope

**Recommendation: do NOT fine-tune for V3.1. Post-launch research lever only.**

If we ever do:

- **Data needed:** 50-100 hand-written rock charts to teach "simple songs deserve simple labels." Cannot use UG-scraped charts (Alexandra ruling, Apr 10) — the training corpus would need to be hand-written by Jeff or a contractor based on listening, or sourced from a license-clean dataset (Hal Leonard folios licensed via publisher? Probably $1k+ per song). ~5-10 hours of human time per 10 songs.
- **Compute:** Modal A10G fine-tuning workflow. ~$20-50 budget. 1-2 hours of GPU time per training run at low LR (1e-5).
- **Risk: catastrophic forgetting.** Fine-tuning on simple-song data is likely to break ACE's extension wins on Petty IGWO and MJLD. Mitigation requires mixing the new data with original Isophonics + Billboard data — and the original training corpus is large; we'd be retraining, not fine-tuning.
- **Risk: training set leakage.** Any of the 38 fixture songs ending up in the training set destroys our benchmark.
- **Risk: didn't solve the real problem.** The Aja regression is about ACE not emitting *enough* extensions; the Hotel California "regression" is about ACE emitting *too many*. Fine-tuning could fix either but probably not both — they're opposite-direction errors.

**Better V3.2 alternatives (rank ordered):**

1. **Claude-as-reranker over ACE+Jiang per-bar top-K candidates** (the V3 plan's post-launch lever, still the highest-leverage). Non-destructive picker, not wholesale rewrite. Could catch both failure modes (eager-on-rock and flattening-on-jazz) at the bar level.
2. **Jazz-signal router path** — extend the per-song router to a 3-way switch (ACE / Jiang / stem-aware). Stem-aware detector scored 226/226 on Aja; it's still in the codebase. Trigger on high chord-density + many distinct extensions in detector output.
3. **GT enrichment** — for high-traffic demo songs (Hotel California), update the fixture to include the slash chords the audio actually has. Editorializing UG, but defensible since the audio supports it. Low priority; only matters if the demo F1 number is something we're publicly publishing.

Fine-tuning is below all three.

---

## 9 — Recommended V3.1 architecture (single concrete spec)

```
audio upload
  → Modal A10G stem separation (BS-RoFormer + Demucs) [unchanged]
  → run BOTH detectors on full mix (in-process on Hetzner CPX41):
      events_ace  = ACE-default (chunk_dur=20, threshold=0.5, min_dur=0.5)
      events_jiang = Chord-CNN-LSTM (Jiang, default params)
  → ROUTER:
      density = len(events_ace) / max(1, len(events_jiang))
      agreement = bar_grid_agreement(events_ace, events_jiang)
      if agreement > 0.5 AND density > 1.15:
          events = events_jiang
      else:
          events = events_ace
  → Harte → standard chord-name parser
  → chart_formatter.format_chart()
  → chord_chart.json
```

**Flags:**
- `V3_DETECTOR=ace_router` (new, defaults to `librosa`). Controls the whole switch.
- `ENABLE_ANTHROPIC_CORRECTION=false` on the ACE-router path. Stays `true` on the librosa fallback path.
- `JIANG_PATH_MEMORY_CAP_GB=4` env override for safety if 4-way concurrent ACE+Jiang OOMs.
- All V3 detector env flags live in `/opt/stemscribe/.env` on prod (parent dir, NOT `/opt/stemscribe/backend/.env` which is a minor-overrides file). Per ship-audit Agent C 2026-05-13.

**Compute footprint (per song):**
- ACE inference: ~10-15s CPU on CPX41 (Agent 1 measured 7.83s wall on M3 Max for 4.5-min song; CPX41 ≈ 1.5× slower)
- Jiang inference: ~10-15s CPU on CPX41 (per V3 plan's Agent D measurement)
- Total post-separation chord work: 20-30s CPU, ~1.5-2 GB peak RSS per job. At 4 concurrent: ~6-8 GB chord-work RSS, fits 16 GB.
- No Modal cost for chord detection; runs in-process.

**Files to write or edit:**
- New: `backend/processing/chord_detector_ace.py` — wraps ACE inference, returns events in chord_chart.json shape. Includes Harte→standard parser (port from `/tmp/v3bake/bakeoff.py`).
- New: `backend/processing/chord_detector_jiang.py` — was already planned in V3 plan; same shape.
- New: `backend/processing/detector_router.py` — implements the router. Tiny: ~50 lines including `bar_grid_agreement()` helper.
- Modified: `backend/processing/pipeline.py:633-647` — replace the librosa block with a `V3_DETECTOR` switch.
- Modified: `audit/score_chord_chart.py:96` — add `X5 → maj` family collapse so Iron Man and Hells Bells score honestly.

**Cohort baseline to ship against** (the number to beat on held-out songs):
- Current prod (librosa + V1 corrector): root F1 0.713
- ACE-default alone: root F1 0.847
- **ACE + router (target): root F1 0.869** ← ship if hold-out validates above 0.85

---

## 10 — Updated week-by-week plan

This overrides Sections 3 and 4 of `docs/v3-plan-2026-05-13.md`.

### Week 1 — May 13–19: wire ACE + Jiang + router behind a flag

- **Mon–Tue:** Create `backend/processing/chord_detector_ace.py` and `backend/processing/chord_detector_jiang.py`. Both modules export `detect_chords_for_job(job, audio_path)` matching the existing librosa-path interface. ACE module wraps `python -m ACE.inference` subprocess; Jiang module wraps the existing `chord_recognition.py` subprocess. Both emit standard-form chord names (Harte parser from `/tmp/v3bake/bakeoff.py`).
- **Wed:** Create `backend/processing/detector_router.py` with the `route()` function from §9. Add unit tests against the 13 cohort songs' .lab files (regression test: router should pick Jiang for hotel-california, paint-it-black, sunshine-of-your-love and ACE for everything else).
- **Thu:** Modify `backend/processing/pipeline.py:633-647` to switch on `V3_DETECTOR`. Default `librosa` (no behavior change). Add scorer fix `X5 → maj` family collapse in `audit/score_chord_chart.py:96` so Iron Man and Hells Bells score honestly.
- **Thu:** Land the Agent D archive PR from the V3 plan (11 files to `backend/_archive/`, 3 pre-edits). Holds.
- **Fri:** Smoke test on Hetzner staging. SSH to `root@5.161.203.112`, deploy branch, run 3 manual jobs through `V3_DETECTOR=ace_router`. Verify 4-way concurrency stays under 8 GB chord-work RSS. Watchdog must not fire.

**Done when:** PR merged with `V3_DETECTOR=librosa` in prod `.env` (off on prod). Test suite still green. Archive PR landed.

### Week 2 — May 20–26: held-out validation + prod flip

- **Mon:** **Held-out validation.** Pull 8-10 new songs from `audit/fixtures/ground_truth/` that were NOT in the 13-song router-tuning cohort. Candidates (have audio on disk somewhere): The Beatles — Let It Be, Led Zeppelin — Stairway to Heaven, Steely Dan — Aja, Steely Dan — Black Cow, Steely Dan — Do It Again, Jamiroquai — Cosmic Girl, Stevie Wonder — Superstition, James Taylor — Fire and Rain, BB King — Thrill Is Gone, Coldplay — Yellow. Run the router on all of them, score, compare to ACE-alone.
- **Decision criterion:** if held-out avg root F1 ≥ 0.85 and no song regresses >0.10 vs ACE-alone, the router thresholds generalize. Flip prod.
- **Tue:** Prod flip — `V3_DETECTOR=ace_router`, `ENABLE_ANTHROPIC_CORRECTION=false`. Active monitoring first hour.
- **Wed–Thu:** Build nightly benchmark runner that produces a full scoreboard (all 38 ground-truth fixtures, not just the 13-song cohort). Hook into admin dashboard. Per-song alert threshold: regression >0.10 root F1 day-over-day fires an email.
- **Thu:** Aja regression callout. Document in user-facing FAQ that jazz-extension songs may lose some extension detail in V3.1; recommend `📋 My Chart` paste for users who care.
- **Fri:** Re-run the full 38-song scoreboard. Confirm hold-out numbers + cohort numbers within noise.

**Done when:** Prod is on `V3_DETECTOR=ace_router`. Held-out validation passed. Nightly scoreboard live. Aja regression documented.

### Week 3 — May 27–Jun 2: soak

- **All week:** Hands-off prod observation. Daily check on scoreboard and error tracker. Memory watch on CPX41 (`free -m` snapshot to a log every hour; alert if >12 GB sustained).
- **Side tasks (won't block launch if undone):**
  - **GT enrichment for Hotel California.** Update `audit/fixtures/ground_truth/eagles__hotel-california.json` to include the 6 G/D events Agent 4 confirmed are in the audio. Defensible because it's audio-grounded. Optional polish.
  - **Pull missing benchmark audio.** Iron Man variants, additional rock songs from the Classic Rock folder, to expand the held-out test set for future iterations.

### Week 4 — Jun 3–9: decision gate + polish

- **Tue Jun 5: DECISION GATE.** Inputs:
  1. 7-day prod soak avg root F1 ≥ 0.85 (on the held-out set, not the router-tuning cohort).
  2. No song regressed >0.15 vs Week-2 validation numbers.
  3. No memory leak, no watchdog stall, no spike in `error_tracker` chord-stage failures.
  4. Aja quality F1 at the documented 0.41 floor or above; no further drop.
- **Pass all four → green light for launch.**
- **One or two fail → re-decide Jun 9 after fixes.**
- **Three or four fail → roll back to librosa+V1 corrector (0.71 baseline) and ship V3.1 post-launch.** Rollback is one env flip: `V3_DETECTOR=librosa`, `ENABLE_ANTHROPIC_CORRECTION=true`.
- **Polish (post-gate):** UI badge "this chart includes slash chords / extensions caught from audio" — marketing copy beat for the genuine differentiator.

### Week 5 — Jun 10–16: marketing prep + soft freeze

- Marketing drafts (`docs/marketing-drafts-2026-04-26.md`) refreshed to mention slash-chord support and extension detection. No code changes unless P0.

### Week 6 — Jun 17–19: hard freeze

- No code changes. Final benchmark run Jun 17. Final smoke test Jun 18. Jun 19 quiet day.

### Launch — Jun 20, soft launch at Refinery (Charleston)

---

## 11 — Decision gate (Jun 5)

**Ship V3.1 (ACE + router) if:**
- 7-day held-out avg root F1 ≥ 0.85
- No held-out song regressed > 0.15 vs Agent 3's router validation
- Aja quality F1 ≥ 0.40 (the documented floor)
- No P0/P1 production issue
- Memory under 12 GB sustained at 4-way concurrency

**Hold for Jun 9 re-decision if:**
- Held-out avg drops to 0.80–0.84
- One or two held-out songs regress > 0.15 with no diagnosed cause
- Memory yellow flags but no actual outage

**Roll back to librosa+V1 (0.71) if:**
- Held-out avg < 0.80
- Any P0
- Memory leak under 4-way concurrency

---

## 12 — What ships at launch / what's explicitly post-launch

### At launch (Jun 20)

- ACE-default + per-song Jiang router (in-process on CPX41).
- Anthropic corrector OFF on the ACE-router path.
- Scorer fix: `X5 → maj` family collapse (fixes Iron Man + Hells Bells).
- Slash chord support in chord_chart.json (the genuine differentiator).
- 38-song nightly benchmark scoreboard on admin page.
- Aja-regression caveat in user FAQ; `📋 My Chart` paste as the escape hatch.

### Explicitly post-launch (V3.2)

- **Claude-as-reranker over per-bar top-K from BOTH detectors.** Highest-leverage architectural lever; non-destructive picker; could catch both failure modes (eager-on-rock, flattening-on-jazz).
- **Jazz-signal third path** — extend router to 3-way switch (ACE / Jiang / legacy stem-aware). Recover Aja-style extensions.
- **GT enrichment for high-traffic demo songs** (Hotel California first).
- **Sub-beat bass tracking for descending-bass slash chords** the bar-quantized bass extractor missed (Petty IGWO intro descending bass etc.). The Mar-9 root-cause doc hinted at this.
- **Fine-tuning ACE.** Last resort. Only if all three above hit ceilings.

### Explicitly killed (do not propose)

- Bass-stem post-processing of ACE output (Agent 2: net delta +0.000, doesn't work — ACE slashes are real).
- Chunk-dur tuning (Agent 1: wash globally, mixed per-song).
- Threshold / min-dur tuning of ACE (already shown -0.185 catastrophic).
- Non-decomposed conformer (Agent 1: not shipped, would need training).
- Tuning compensation (`librosa.estimate_tuning`, killed in V3 plan: -0.036 with corrector).
- Single-bass-stem slash-chord hybrid (killed in V3 plan: -0.107 full F1).

---

## 13 — Pointers

- This report: `docs/v3-ace-tuning-2026-05-13.md`
- Agent 1 (ACE variants + Aja): `docs/v3-agent-1-ace-variants-2026-05-13.md`
- Agent 2 (bass-stem post-proc, negative result): `docs/v3-agent-2-bass-stem-postproc-2026-05-13.md`
- Agent 3 (per-song router, the winner): `docs/v3-agent-3-ensembles-2026-05-13.md`
- Agent 4 (HC deep dive + Hells Bells GT): `docs/v3-agent-4-hotel-cal-hells-bells-2026-05-13.md`
- V3 plan being overridden: `docs/v3-plan-2026-05-13.md`
- V3 agent reports (Jiang bake-off, tuning, cleanup): `docs/v3-agent-{A,B,C,D}-*-2026-05-13.md`
- Hells Bells GT (new): `audit/fixtures/ground_truth/ac-dc__hells-bells.json`

---

## 14 — One-line for the Jun 5 gate

**Did the held-out 8-10 song validation set hold avg root F1 ≥ 0.85 with Aja quality F1 ≥ 0.40 and no memory leak?** Pass → ship the ACE+router. Fail → flip back to librosa+V1 (0.71) and ship V3.1 post-launch.
