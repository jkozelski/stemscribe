# Phase 4 — MIDI Detector vs Stem-Aware Detector Regression

**Date:** 2026-04-23
**Corpus:** Kozelski chord library + Jamiroquai *Alright* (m7 spot-check)
**Gate:** NEW must beat OLD on ≥ 15/20 Kozelski songs AND Alright m7-family ≥ 80%.
**Verdict: FAIL on both counts. Do not flip `ENABLE_MIDI_DETECTOR`. Ship with OLD detector for 2026-05-12.**

---

## TL;DR

- **NEW wins: 1/17 (event-strict), 2/17 (event-lenient), 8/17 (set-F1).** None reach the ≥15/20 threshold.
- *Alright* m7-family rate: 77.4% (measured in Phase 2), short of the 80% target.
- Root cause: NEW detector systematically **over-extends** chord names. It hears 9ths/maj7/add9 in the voicings and emits `Gadd9`/`Cmaj9`/`Am9`, while Kozelski's chart convention abstracts to plain triads (`G`, `C`, `Am`).
- The MIDI detector is not broken — it correctly detects what's played — but its output convention doesn't match Kozelski's chart convention. **For a jazz/funk catalog (like Jamiroquai) it wins. For a rock/folk catalog (like Kozelski) it loses.**
- NEW code stays in the repo behind `ENABLE_MIDI_DETECTOR=false`. Recommended future work listed at the bottom.

## Corpus

| Status | Count | Songs |
|---|---|---|
| Measured | 17 | born-in-zen, clandestine-sam, clever-devil, climbing-the-bars, cold-dice, eye-to-eye, lady-in-the-stars, living-water, mystified, natural-grow, silent-life, simpler-understanding, solar-connection, the-blinding-glow, the-city-gonna-teach-you, the-time-comes, thunderhead |
| Skipped (no ground truth) | 1 | hard-feelings (empty `chords_used`) |
| Skipped (no audio found) | 2 | haitian-divorce, the-search |

Gate calibrated at 15/20 (75%). Equivalent target on the 17-song measured subset: **13/17 (76.5%)**. Actual best: 8/17 (set-F1, 47%). **Gate fails on every metric, with or without the 3 unmeasured songs.**

## Methodology

For each song:

1. Stems = BS-RoFormer-SW output (Modal A10G via the running backend).
2. Ground truth = `chords_used` from the song's Kozelski JSON (unordered chord set).
3. OLD detector = `backend/stem_chord_detector.StemAwareChordDetector.detect_from_stems(...)`. Internally applies `merge_consecutive_chords` + `_prune_outlier_chords` + `_simplify_bleed_extensions`.
4. NEW detector = `backend/midi_chord_detector.detect_chords_from_midi(...)` with grid + bass_roots freshly extracted. **`merge_consecutive_chords` + `_prune_outlier_chords` applied afterward for apples-to-apples comparison** (same post-processing OLD uses).

Full script: `/tmp/phase4/run_regression.py`. Full results: `/tmp/phase4/regression_results.json`.

### Scoring

Three per-song metrics; higher wins by > 0.02.

| Metric | Per-event | Notes |
|---|---|---|
| `event_strict` | exact=1.0, root+family=0.75, root_only=0.5, miss=0 | Weighted per chord event |
| `event_lenient` | exact=1.0, root+family=**0.9**, root_only=0.5, miss=0 | Gives more credit when NEW outputs `Gadd9` vs GT's `G` (both major) |
| `set_F1` | Weighted F1 over unique-chord-set (Jaccard with partial credit) | Ignores per-event bar counts; pure "did you find the right chords" |

Families: {major, minor, dom, dim, sus, power, aug}. `maj`/`maj7`/`add9`/`6` all in `major`. `m`/`m7`/`m9` in `minor`. etc.

## Per-song results

| # | Song | Key | GT # | OLD strict | NEW strict | OLD F1 | NEW F1 | winner(event) | winner(F1) |
|---:|---|:-:|---:|---:|---:|---:|---:|:-:|:-:|
| 1 | born-in-zen | G | 10 | 0.861 | 0.764 | 0.485 | 0.676 | OLD | NEW |
| 2 | clandestine-sam | Gm | 10 | 0.900 | 0.705 | 0.649 | 0.477 | OLD | OLD |
| 3 | climbing-the-bars | G | 7 | **1.000** | 0.550 | 0.792 | 0.376 | OLD | OLD |
| 4 | eye-to-eye | C | 11 | 0.938 | 0.657 | 0.630 | 0.465 | OLD | OLD |
| 5 | lady-in-the-stars | Gm | 8 | 0.750 | 0.618 | 0.471 | 0.645 | OLD | NEW |
| 6 | natural-grow | Am | 15 | 0.867 | 0.842 | 0.656 | 0.734 | OLD | NEW |
| 7 | silent-life | F#m | 9 | **1.000** | 0.791 | 0.780 | 0.633 | OLD | OLD |
| 8 | **thunderhead** | Am | 10 | 0.694 | **0.784** | 0.459 | **0.693** | **NEW** | **NEW** |
| 9 | the-city-gonna-teach-you | Em | 11 | 0.150 | 0.089 | 0.074 | 0.059 | OLD | TIE |
| 10 | living-water | B | 7 | 0.925 | 0.828 | 0.731 | 0.802 | OLD | NEW |
| 11 | the-blinding-glow | Dm | 13 | 0.795 | 0.791 | 0.480 | 0.306 | TIE | OLD |
| 12 | cold-dice | Bm | 8 | 0.964 | 0.822 | 0.814 | 0.847 | OLD | NEW |
| 13 | mystified | Fm | 12 | 0.737 | 0.527 | 0.551 | 0.338 | OLD | OLD |
| 14 | clever-devil | Am | 9 | **1.000** | 0.500 | 0.714 | 0.308 | OLD | OLD |
| 15 | solar-connection | Am | 11 | 0.958 | 0.739 | 0.500 | 0.583 | OLD | NEW |
| 16 | simpler-understanding | C#m | 8 | 0.330 | 0.251 | 0.194 | 0.228 | OLD | NEW |
| 17 | the-time-comes | F#m | 8 | 0.955 | 0.681 | 0.830 | 0.461 | OLD | OLD |

## Aggregate

| Metric | NEW wins | OLD wins | TIE | Gate (≥15/20) |
|---|---:|---:|---:|:-:|
| event_strict | **1** | 15 | 1 | **FAIL** |
| event_lenient | 2 | 13 | 2 | **FAIL** |
| set_F1 | 8 | 8 | 1 | **FAIL** |

Even the most generous metric (set_F1) has NEW at 8/20, far short of 15.

## Alright m7-family re-verification

Measured at Phase 2 close; unchanged. 77.4% of 84 eligible bars resolve to a minor-7-family quality (m7 / m9 / m11 / m13) under the new detector — below the 80% target by 2.6 points. The existing stem-aware detector scores 0% m7-family on the same job (audit 2026-04-23), so NEW is still far better on 7ths-heavy material; 77.4 vs 80 is a margin call that was supposed to be rescued by the Kozelski regression, but the regression collapsed.

## Failure modes (specific, actionable)

### FM-1 — Systematic over-extension (the dominant failure)

NEW emits chord names with 9ths / maj7 / add9 / m9 where GT uses plain triads. Examples:

- **climbing-the-bars** — GT = {G, Bm, Em, Eb, Am, C, D} (all triads). NEW top outputs: `D9×18, Am9×16, Cmaj9×11, F#m7b5×10, Cadd9×5`. **0/7 exact matches** because every output has an extension.
- **clever-devil** — GT = {Am, G, C, F, Gm7, Dm, Bb, A, C7}. NEW top: `Am7×40, Em7×29, Fmaj9×18`. NEW promoted every major-family chord to maj9/add9 and every minor to m7. 0/9 exact.
- **the-time-comes** — GT triads {F#m, A, Bm, E, B, D, C#m, G}. NEW top: `F#m7×23, E9×16, Emaj9×7, Bm9×6`. 0/8 exact.

**Why:** Basic Pitch reports all notes in a voicing. If the guitarist strums a G-major chord voiced as `[G, B, D, A]` (G with a 9 on top) — a perfectly normal voicing in folk/rock — the detector correctly sees the A and outputs `Gadd9`. The chart says `G`. Neither is wrong; they're different abstraction levels.

### FM-2 — Over-extension cascades through set F1

With extensions in the output, even NEW's "set F1" suffers on precision. For example on *silent-life*: NEW recall=0.633 (decent coverage of GT chord set), but precision is dragged by the many `F#m7`, `Dmaj9`, `A6`, `Bm9` extras not in GT. Net F1 = 0.633 vs OLD 0.780.

### FM-3 — Root errors on a handful of songs

NEW occasionally picks the wrong bass root via the dominant-pc fallback when `bass_roots` has low confidence for a bar. Seen on *clandestine-sam* and *the-blinding-glow*, where NEW outputs `Dm9`/`Gm9` for bars the chart lists as `Bb` or `Eb` (wrong root). This is rarer than FM-1 but real.

### FM-4 — Pathological GT (not a detector bug but worth noting)

*the-city-gonna-teach-you* has GT `Dmaj7sus4/C#`, `Bb6sus2`, `Gmaj7/B`, etc. — chart conventions neither detector's template set covers. Both detectors score near zero (0.150 / 0.089). This is a ground-truth scope issue; fixing it requires templates for compound sus/add chords which neither detector has.

### Where NEW wins — FM-5 (sweet spot)

**thunderhead** — GT = {Am7, Bm7, Cm7, E9, Bm6, Dm7, Gmaj7, G#7, Em7, G#maj7}. Every GT chord has a 7th or extension. NEW wins event-strict (0.784 vs 0.694) AND set F1 (0.693 vs 0.459). This confirms the design thesis: **MIDI-intermediate detection is superior when the song's notation itself uses extensions.** Just not most of Kozelski.

## Gate decision

**FAIL.** On no metric does NEW beat the threshold. The gap is not close on the primary (event-strict) metric — 1/20 vs 15/20 required.

## Recommendations

### Ship for 2026-05-12

1. **Keep `ENABLE_MIDI_DETECTOR=false` in production.** Do not flip.
2. The MIDI detector code stays in the repo behind the flag. Same-revert posture Jeff specified in the Phase 2 answer to question #1.
3. Ship with the existing stem-aware detector. The 2026-04-23 audit's recommendations (P1 Whisper dedup, P2 seventh-promotion pass, the chart render rewrite) remain the launch-critical track.

### Post-launch work on the MIDI detector

The detector's underlying thesis — "read notes straight from Basic Pitch, match intervals" — is sound (Phase 0 confirmed 100% 7th capture). The current implementation over-commits to extensions. Four concrete tunings, in order of likely payoff:

1. **Add an extension-confidence gate.** Promote `min` → `m7` only when the b7 pc has weight ≥ 20% of bar peak (currently ≥ 5%). Promote further to `m9` only when both b7 AND 9 have ≥ 20%. Gives triads the benefit of the doubt on borderline voicings.
2. **Post-process with a chart-aware simplifier.** Run a pass that drops 9/add9/6 extensions if they appear on < 30% of bars in a section — treats them as voicing color, not chord identity.
3. **Tie-break toward simpler templates on close scores.** Today's tie-break prefers the longer template at 0.01 margin; invert to prefer the shorter one when the longer's only edge is an extension.
4. **Per-catalog tuning via `MIDI_DETECTOR_ENERGY_FLOOR`.** Jazz/standards catalog = 0.05 (current). Rock/folk catalog = 0.15-0.20. This was the whole point of making it env-tunable in Phase 2; we just didn't know the right value until now.

Plausible path: re-run the Phase 4 regression with `MIDI_DETECTOR_ENERGY_FLOOR=0.15` as the first experiment after launch. If it flips most Kozelski wins to NEW without collapsing on thunderhead, the detector becomes shippable as a config change.

## Data artifacts

- `/tmp/phase4/regression_results.json` — full structured output (17 songs, every metric).
- `/tmp/phase4/regression_run.log` — human-readable run log.
- `/tmp/phase4/score_one_song.py` — single-song re-scoring harness for debugging individual songs.
- `/tmp/phase4/run_regression.py` — batch driver.
- `docs/midi-prototype-results-2026-04-23.md` — Phase 0 gate data (unchanged).
- `docs/midi-chord-detector-spec-2026-04-23.md` — Phase 1 architecture spec (unchanged).

## Phase 4 close

Phase 4 complete. NEW detector **remains dormant behind the feature flag.** Existing stem-aware detector is the production path for 2026-05-12 launch. Regression data preserved for the post-launch tuning pass described above.
