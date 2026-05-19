# V3.1 ship audit — jazz-detector empirical bake-off

**Agent:** Audit-A (60-minute time-box)
**Date:** 2026-05-13
**Question:** Does StemScriber's stem-aware "jazz-routed" chord detector (the Apr-25-sprint family-aware-consistency detector at `backend/stem_chord_detector.py`) actually beat ACE on the 6 UG jazz fixtures, or is the V3.1 router's jazz path an empirical no-op?

## TL;DR

**ACE wins on 5 of 6 songs at root F1 and on 6 of 6 at quality F1.** The stem-aware detector aggregates **root F1 0.555 / quality F1 0.118** vs ACE's **0.811 / 0.422** on the same 6 jazz songs — a delta of **+0.256 root / +0.304 quality** in ACE's favor. The stem-aware detector emits **zero slash chords** on all 6 songs; ACE emits **51 across the cohort** (1 to 17 per song).

The Apr 25 "Aja 226/226 extension matches" headline is **not reproducible against UG ground truth**. The stem-aware detector scores **root F1 0.341 / quality F1 0.000 / 0% vocab coverage** on Aja — it pruned the entire 33-class jazz vocabulary down to 3 chord types (Bmin9, A9, Bmin7). The Apr 25 number was an artifact of scoring against self-generated, weaker ground truth.

**Verdict: drop the jazz path from V3.1.** Route everything to ACE (or ACE+post-processing). The stem-aware detector's `_prune_outlier_chords(min_frequency=0.05)` step is fatal on jazz repertoire where 20-33 unique chords per song is normal and a 5%-of-bars threshold quietly deletes 25-30 of them.

## What was run

1. Located the 6 UG jazz fixtures in `audit/fixtures/ground_truth/`:
   - `steely-dan__aja`, `steely-dan__peg`, `steely-dan__rikki-dont-lose-that-number`
   - `jamiroquai__cosmic-girl`, `jamiroquai__alright`, `jamiroquai__virtual-insanity`
2. Located stems for 5/6 songs locally under `/Users/jeffkozelski/stemscribe/outputs/<job_id>/stems/` (matched via `/tmp/v3bake/resolution.json` from Agent A's earlier work). For `jamiroquai__virtual-insanity` the local job didn't exist; pulled the stems from the VPS at `/opt/stemscribe/outputs/21eb5acc-6560-4ea6-9555-ce3ef59d41d1/stems/` to `/tmp/v3jazz/stems/virtual-insanity/`. No songs were missing.
3. Ran the live-pipeline entry (`StemAwareChordDetector(min_duration=0.15).detect_from_stems(...)`) on each song via `/tmp/v3jazz/run_stem_aware.py` and wrote chord_chart.json for each.
4. Ran ACE (`python -m ACE.inference --threshold 0.5 --chord-min-duration 0.5`) on each song's full-mix audio via `/tmp/v3jazz/run_ace.py` and wrote chord_chart.json for each.
5. Scored both detector outputs for each song against the UG fixture with `audit/score_chord_chart.py`. Aggregated to `/tmp/v3jazz/results.json`.

Nothing was deployed to production.

## Per-song results

### Root F1 (just the root note — most lenient metric)

| Song | ACE | stem-aware | Δ(stem-aware − ACE) |
|---|---:|---:|---:|
| Aja | **0.812** | 0.341 | -0.471 |
| Peg | **0.821** | 0.442 | -0.379 |
| Rikki Don't Lose That Number | **0.792** | 0.586 | -0.206 |
| Cosmic Girl | **0.837** | 0.495 | -0.342 |
| Alright | 0.861 | **0.899** | **+0.038** |
| Virtual Insanity | **0.744** | 0.569 | -0.175 |
| **MEAN** | **0.811** | **0.555** | **-0.256** |

Only Alright lifts under the stem-aware detector, and only by 0.038. Every Steely Dan song and Cosmic Girl regress by ≥0.21 root F1.

### Quality F1 (Em7 must match Em7, not just Em)

| Song | ACE | stem-aware | Δ(stem-aware − ACE) |
|---|---:|---:|---:|
| Aja | **0.411** | 0.000 | -0.411 |
| Peg | **0.228** | 0.096 | -0.132 |
| Rikki Don't Lose That Number | **0.465** | 0.414 | -0.051 |
| Cosmic Girl | **0.382** | 0.000 | -0.382 |
| Alright | **0.500** | 0.000 | -0.500 |
| Virtual Insanity | **0.546** | 0.198 | -0.348 |
| **MEAN** | **0.422** | **0.118** | **-0.304** |

Quality F1 is where ACE most decisively wins. The stem-aware detector scores **exactly 0.000** on Aja, Cosmic Girl, and Alright because it pruned its way to 3-9 chord types that have **no overlap** with the UG quality vocabulary on those songs (e.g. for Cosmic Girl the GT uses `Em7 / F#m7 / B7sus4 / B7` and the detector emits `Emin7 / F#min7 / B9sus4 / C#min7 / C#min11 / F#min9` — the GT vocabulary doesn't include the `B9sus4` voicing and detector vocabulary doesn't include `B7` or `B7sus4`).

### Full F1 (slash-exact)

| Song | ACE | stem-aware |
|---|---:|---:|
| Aja | **0.382** | 0.000 |
| Peg | **0.211** | 0.096 |
| Rikki | **0.455** | 0.414 |
| Cosmic Girl | **0.382** | 0.000 |
| Alright | **0.500** | 0.000 |
| Virtual Insanity | **0.498** | 0.198 |
| **MEAN** | **0.405** | **0.118** |

ACE wins on every song.

### Slash-chord count emitted

| Song | ACE | stem-aware | GT (UG) |
|---|---:|---:|---:|
| Aja | 17 | **0** | 29 |
| Peg | 11 | **0** | 22 |
| Rikki | 6 | **0** | 4 |
| Cosmic Girl | 1 | **0** | 0 |
| Alright | 6 | **0** | 0 |
| Virtual Insanity | 10 | **0** | 11 |
| **TOTAL** | **51** | **0** | **66** |

The stem-aware detector emits **zero slash chords** on all 6 songs. ACE emits slashes on every song, and on Aja/Peg/Virtual Insanity captures the right side of the song's slash-chord vocabulary. (Slash-chord *correctness* — does ACE's `Bmaj7/F#` match GT's `Bmaj7/F#`? — is mostly miss, as the earlier Agent-1 Q8 audit noted: ACE invents disjoint slashes. But the *capability* exists in ACE and is gated off in stem-aware.)

### Extensions emitted (count of chord events whose label implies 7+ extension)

| Song | ACE | stem-aware |
|---|---:|---:|
| Aja | 132 | 120 |
| Peg | 45 | 26 |
| Rikki | 35 | 0 |
| Cosmic Girl | 48 | 45 |
| Alright | 63 | 89 |
| Virtual Insanity | 104 | 57 |
| **TOTAL** | **427** | **337** |

Extension *count* is similar — the stem-aware detector does emit lots of `min7`/`min9` labels, but they're concentrated on the 3-9 unique chord types that survive the 5% prune. ACE distributes its extensions across 17-39 unique chord types.

### Detector unique-chord-vocabulary vs ground truth

| Song | GT unique | ACE unique | stem-aware unique |
|---|---:|---:|---:|
| Aja | 33 | **39** | **3** |
| Peg | 20 | **21** | **5** |
| Rikki | 19 | **27** | **5** |
| Cosmic Girl | 5 | 17 | 6 |
| Alright | 4 | 21 | 9 |
| Virtual Insanity | 18 | **33** | **5** |

ACE matches or exceeds GT vocabulary size on every song. The stem-aware detector under-covers by 3-7x on every Steely Dan track and on Virtual Insanity.

## Why stem-aware collapses on jazz

The detector's `_prune_outlier_chords(min_frequency=0.05)` step drops any (root, quality) pair appearing in less than 5% of bars and replaces it with the nearest surviving chord by semitone distance. For Aja, the live log reads:

```
Pruned 53 outlier chord types (94/120 bars = 78%) below 5% threshold
```

Aja has 33 unique chords in 197 GT events — most of them legitimately appearing in 1-3 bars (Aadd9, Bmaj7, Cmaj7#11, Dadd9, Ebadd9, D6/9, Bm11, …). On a 120-event detector output, **every chord that appears fewer than 6 times is wiped.** That deletes 53 of 56 detected chord types and rewrites 78% of bars to the 3 most-frequent ones (Bmin9, A9, Bmin7). On rock songs with a 4-chord vamp this prune is harmless or beneficial; on jazz it is the single largest source of error.

The same pattern recurs on every Steely Dan track and Virtual Insanity:
- **Peg:** 36 outlier types pruned, 57% of bars rewritten
- **Rikki:** outlier-prune step compressed to 5 unique chords from a 19-chord GT vocabulary
- **Virtual Insanity:** 36 outlier types pruned, 57% of bars rewritten

## Aja claim verification

The Apr 25 memo and `stemscriber_full_state.md` carry the claim **"Aja 226/226 extension matches (perfect Bm jazz)"**. Today's UG-grounded result:

| Metric | Value | Apr 25 claim |
|---|---:|---|
| root F1 (UG) | **0.341** | n/a |
| quality F1 (UG) | **0.000** | "226/226 perfect" |
| pcs F1 (UG) | **0.215** | n/a |
| Unique detected chords | **3** | implied ~33 (Bm vocabulary) |
| GT slash chords matched | **0/29** | n/a |
| Vocab coverage | **0.0%** | implied 100% |

**The Apr 25 "226/226" number was scored against a self-generated ground truth that the detector itself helped produce.** Re-scored against the UG jazz fixture, the detector misses every quality class GT defines and emits zero slash chords. The headline does not survive UG-grounded validation.

## V3.1 router recommendation

**Drop the jazz path. Route 100% to ACE.**

Specific changes the V3.1 router should make:

1. **Remove the genre-based jazz routing** in `backend/processing/detector_router.py`. The stem-aware detector is not a credible jazz specialist; on jazz repertoire it is the worst configuration measured today.
2. **Replace with ACE** as the unconditional detector. ACE's 0.811 root / 0.422 quality / 51 slashes on this 6-song jazz set is the strongest configuration measured and matches Agent A's earlier finding that ACE beats every prior baseline on the 13-song rock cohort.
3. **Post-processing layer (V3.2):** ACE's residual error is concentrated in slash-chord *identity* (it invents `A/E` where GT has `Aadd9/B`) and extension *flattening* (it picks `Cm7` where GT has `Cm11`). Neither failure mode is fixed by re-routing to the stem-aware detector — both are vocabulary-distribution issues in ACE's training data (Isophonics + Billboard). The fix is a reranker or LLM-corrector that has access to the song's title/artist, not a router that swaps in a worse detector.
4. **Update the marketing claim.** The "Aja 226/226" number cannot be reused in V3.1 launch material. The honest jazz-cohort number under ACE is: **root F1 0.81, quality F1 0.42** — strong for a pop/rock-trained model, weak vs human transcribers, and the right baseline to set expectations from.

## Honest caveats

1. **The 6-song cohort is small.** Six songs is enough to show that the jazz path is empty on the songs we have UG ground truth for; it does not prove that no jazz arrangement could benefit from stem-aware detection. A `_prune_outlier_chords(min_frequency=0.01)` (or removing the prune entirely) might salvage the stem-aware detector on jazz, but that test was outside today's time-box.
2. **Alright is the one positive datapoint for stem-aware** (+0.038 root F1, but -0.500 quality F1). Even on this song the quality regression is much larger than the root lift; ACE is still the right call.
3. **Slash-chord credit was awarded by "count, not correctness."** ACE's 51 slashes are mostly the wrong slashes (Agent-1's Q8 audit showed this on Aja specifically: ACE invents 9 slashes, only 1 of which appears in GT). The headline for V3.1 launch shouldn't be "we emit slash chords" — it should be "we emit a credible chord chart with a 7th/9th-aware vocabulary and the right root sequence." Slash-chord correctness is a V3.2+ problem.
4. **Songs missing stems: none.** All 5 Steely Dan + Jamiroquai jobs had local stems; Virtual Insanity had stems on the VPS at job `21eb5acc-6560-4ea6-9555-ce3ef59d41d1` and they pulled cleanly to `/tmp/v3jazz/stems/virtual-insanity/`.

## Artifacts

- Driver script (stem-aware): `/tmp/v3jazz/run_stem_aware.py`
- Driver script (ACE): `/tmp/v3jazz/run_ace.py`
- Scorer wrapper: `/tmp/v3jazz/score_all.py`
- Per-song chord_chart.json: `/tmp/v3jazz/charts/<slug>__{stemaware,ACE}.json` (12 files)
- ACE .lab files: `/tmp/v3jazz/ace_lab/<slug>.lab` (6 files)
- Raw results table: `/tmp/v3jazz/results.json`
- Run summaries: `/tmp/v3jazz/stemaware_run_summary.json`, `/tmp/v3jazz/ace_run_summary.json`

Total elapsed: ~25 min stem-aware + ~1 min ACE + scoring = under the 60-minute time-box.
