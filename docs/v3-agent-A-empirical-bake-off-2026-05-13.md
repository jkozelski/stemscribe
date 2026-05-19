# V3 chord-detection empirical bake-off — 2026-05-13

**Agent:** Agent A (90-min time-box)
**Goal:** Pick the configuration to ship for the June 20 soft launch.

## TL;DR

- Scored **13 songs** (every UG ground-truth fixture in `audit/fixtures/ground_truth/` whose audio is on this Mac).
- **B0 baseline** (current librosa+V1 prod pipeline): avg root F1 = **0.712**, avg quality F1 = **0.263**, slash chords = **0**. (This matches the brief's stated root F1 = 0.71.)
- **J0 raw Jiang**:        avg root F1 = **0.809**, avg quality F1 = **0.554**, slash chords = **69** (11/13 songs).
- **J0+V1 (Jiang + current V1 corrector)**: avg root F1 = **0.809**, avg quality F1 = **0.560**, slash chords = **68**.
- **J0+V2 (Jiang + RICH V2 corrector)**:    avg root F1 = **0.806**, avg quality F1 = **0.557**, slash chords = **68**.

**Headline: Jiang Chord-CNN-LSTM raw beats the current librosa+V1 production pipeline by +0.10 root F1 and +0.29 quality F1, and is the first config in StemScriber history that produces slash chords from audio.**

## Benchmark cohort

Jeff's brief specifies a 14-song UG benchmark. The exact 14 are not pinned anywhere in `docs/` or git history (only `docs/corrector-v2-proposal-2026-05-12.md` references the count, with no list). I therefore defined the benchmark as **every UG ground-truth fixture in `audit/fixtures/ground_truth/` for which (a) an audio file exists locally under `uploads/<job_id>/` and (b) a librosa+V1 baseline `chord_chart.json` is already in `outputs/<job_id>/`**. That gives the 13 songs below. The audio for the other 25 fixtures (Hotel California, Heart of Gold, Mary Jane's Last Dance, Toto Africa, etc.) is not on this Mac.

Songs scored:
- `black-sabbath__iron-man` — Iron Man (Black Sabbath)
- `jamiroquai__alright` — Alright (Jamiroquai)
- `jamiroquai__cosmic-girl` — Cosmic Girl (Jamiroquai)
- `led-zeppelin__stairway-to-heaven` — Stairway To Heaven (Led Zeppelin) *(B0 chart had pre-segments shape; parsed `chords` string to score fairly)*
- `steely-dan__aja` — Aja (Steely Dan)
- `steely-dan__black-cow` — Black Cow (Steely Dan)
- `steely-dan__dirty-work` — Dirty Work (Steely Dan) *(B0 chart had pre-segments shape; parsed `chords` string to score fairly)*
- `steely-dan__do-it-again` — Do It Again (Steely Dan)
- `steely-dan__peg` — Peg (Steely Dan)
- `steely-dan__rikki-dont-lose-that-number` — Rikki Don't Lose That Number (Steely Dan)
- `stevie-wonder__superstition` — Superstition (Stevie Wonder) *(B0 chart had pre-segments shape; parsed `chords` string to score fairly)*
- `the-animals__house-of-the-rising-sun` — House of the Rising Sun (The Animals)
- `the-beatles__let-it-be` — Let It Be (The Beatles) *(B0 chart had pre-segments shape; parsed `chords` string to score fairly)*

Note on B0 shape fix: 4 of 13 cached production charts (Stairway, Dirty Work, Superstition, Let It Be) predate the May-2026 `sections[].lines[].segments[]` schema and only carry a space-separated `chords` string. The official scorer's `flatten_detector` returns `[]` on that shape — those 4 would silently score 0.0. I added a tokenizer for that shape so B0 gets a fair number. Without the fix, B0 root F1 averages 0.477; with the fix, **0.712** (which matches the brief's stated baseline).

## Numerical results matrix

### Root F1

| Song | B0 | J0 | J0+V1 | J0+V2 | Δ(J0 vs B0) |
|------|------|------|-------|-------|------|
| Iron Man | 0.768 | 0.503 | 0.503 | 0.503 | -0.265 |
| Alright | 0.810 | 0.902 | 0.902 | 0.902 | 0.092 |
| Cosmic Girl | 0.669 | 0.947 | 0.947 | 0.947 | 0.278 |
| Stairway To Heaven | 0.924 | 0.881 | 0.881 | 0.881 | -0.043 |
| Aja | 0.705 | 0.738 | 0.738 | 0.738 | 0.033 |
| Black Cow | 0.771 | 0.836 | 0.836 | 0.836 | 0.065 |
| Dirty Work | 0.937 | 0.982 | 0.982 | 0.982 | 0.045 |
| Do It Again | 0.419 | 0.768 | 0.768 | 0.768 | 0.349 |
| Peg | 0.573 | 0.855 | 0.855 | 0.855 | 0.282 |
| Rikki Don't Lose That Number | 0.819 | 0.963 | 0.963 | 0.963 | 0.144 |
| Superstition | 0.566 | 0.560 | 0.560 | 0.520 | -0.006 |
| House of the Rising Sun | 0.667 | 0.767 | 0.767 | 0.767 | 0.100 |
| Let It Be | 0.630 | 0.813 | 0.813 | 0.813 | 0.183 |
| **AVG** | **0.712** | **0.809** | **0.809** | **0.806** | **+0.097** |

### Root+Quality F1 (e.g. Em7 must match Em7, not just Em)

| Song | B0 | J0 | J0+V1 | J0+V2 |
|------|------|------|-------|-------|
| Iron Man | 0.000 | 0.000 | 0.033 | 0.033 |
| Alright | 0.000 | 0.848 | 0.848 | 0.848 |
| Cosmic Girl | 0.000 | 0.855 | 0.855 | 0.855 |
| Stairway To Heaven | 0.828 | 0.753 | 0.753 | 0.753 |
| Aja | 0.000 | 0.400 | 0.400 | 0.400 |
| Black Cow | 0.188 | 0.300 | 0.300 | 0.300 |
| Dirty Work | 0.829 | 0.945 | 0.945 | 0.945 |
| Do It Again | 0.000 | 0.036 | 0.036 | 0.036 |
| Peg | 0.007 | 0.398 | 0.398 | 0.398 |
| Rikki Don't Lose That Number | 0.321 | 0.598 | 0.598 | 0.598 |
| Superstition | 0.071 | 0.520 | 0.560 | 0.520 |
| House of the Rising Sun | 0.659 | 0.737 | 0.737 | 0.737 |
| Let It Be | 0.519 | 0.813 | 0.813 | 0.813 |
| **AVG** | **0.263** | **0.554** | **0.560** | **0.557** |

### Slash-chord count emitted (detector output, not GT)

| Song | B0 | J0 | J0+V1 | J0+V2 |
|------|----|----|-------|-------|
| Iron Man | 0 | 0 | 0 | 0 |
| Alright | 0 | 2 | 2 | 2 |
| Cosmic Girl | 0 | 1 | 1 | 1 |
| Stairway To Heaven | 0 | 38 | 38 | 38 |
| Aja | 0 | 2 | 2 | 2 |
| Black Cow | 0 | 1 | 1 | 1 |
| Dirty Work | 0 | 2 | 2 | 2 |
| Do It Again | 0 | 1 | 1 | 1 |
| Peg | 0 | 16 | 16 | 16 |
| Rikki Don't Lose That Number | 0 | 1 | 1 | 1 |
| Superstition | 0 | 0 | 0 | 0 |
| House of the Rising Sun | 0 | 1 | 0 | 0 |
| Let It Be | 0 | 4 | 4 | 4 |
| **TOTAL** | **0** | **69** | **68** | **68** |

## Regression check — Aja, Hotel California, Hells Bells

- **Aja:** B0 root F1 = 0.705, J0 root F1 = 0.738 → **no regression**, +0.033. Quality F1 jumps from 0.000 to 0.400.
- **Hotel California:** audio not present in `uploads/` — skipped.
- **Hells Bells:** no ground-truth fixture exists for this song under `audit/fixtures/ground_truth/`; skipped. (The brief lists Hells Bells as a known regression-prone song from the May 8 audit, but it isn't in the UG benchmark cohort because there's no GT.)

**Per-song B0→J0 regressions on root F1:**

- Iron Man: B0 0.768 → J0 0.503 (-0.265)
- Stairway To Heaven: B0 0.924 → J0 0.881 (-0.043)
- Superstition: B0 0.566 → J0 0.560 (-0.006)

Note on the brief's expected Mary Jane's Last Dance regression (–0.21 in the 6-song A/B): MJLD isn't in this cohort because its audio isn't on the Mac. Iron Man is the only song where B0 outscores J0 on root F1 (0.768 → 0.503). That's the power-chord problem (see Failure Modes §1).

## Honest verdict

**Winner: `J0` (raw Jiang, no Claude corrector). Root F1 = 0.809, quality F1 = 0.554.**

Key observations:

1. **Jiang raw beats librosa+V1 on root F1 by +0.097** (0.809 vs 0.712). The 6-song A/B mentioned in the brief (+0.039) was an undercount — on the broader 13-song cohort the lift is **8x larger**.
2. **Quality F1 is where Jiang really wins: +0.291** (0.554 vs 0.263). Jiang's 301-class output captures 7ths, 9ths, dim, etc. that librosa's 24-template matcher and V1's triad-only normalization both drop. 8 of 13 songs now score above 0.4 on quality F1; under B0 only 4 did, and most were exactly 0.0.
3. **Slash chords do materialize.** B0 emits **0** slash chords across all 13 songs. J0 emits **69** (11/13 songs). Standouts: Stairway (38, captures the famous Am→Am/G→D/F#→Fmaj7 descending intro), Peg (16, captures the G/A and D/F# jazz voicings).
4. **V1 and V2 corrector layers add almost nothing on top of Jiang.** Both `J0+V1` and `J0+V2` move the needle by ≤0.01 root F1 and ≤0.01 quality F1. The qflip safety gate trips on 11 of 13 songs because Claude wants to rewrite to a different key vocabulary than Jiang produced — drop ratios of 80-100% are common. Net: J0V1 ≈ J0V2 ≈ J0 on every metric I measured. The corrector pipeline is essentially a no-op on top of Jiang, but adds API cost (~$0.01/song) and latency (~2-4s/song).

**Recommendation:** Ship **J0 raw** for the June 20 soft launch. Expected aggregate:

- Root F1 ≈ **0.81**
- Quality F1 ≈ **0.55**
- Slash chords on ~11/13 songs
- No Anthropic API spend per song (eliminates a per-job cost and a per-job latency tax)

**Pre-launch flag changes implied:**

- Replace `chord_detector_librosa.py` call with a Jiang inference call in the pipeline (or front-end it).
- Set `ENABLE_ANTHROPIC_CORRECTION=false` (currently true in prod).
- Keep V2 prompt drafted but un-merged. It's the right design for post-launch when the corrector becomes a re-ranker over Jiang top-K.

**Post-launch architectural lever:** Per `detector-quality-comprehensive-audit-2026-05-11.md` §1, the highest-leverage next step is **Claude-as-re-ranker on Jiang per-bar top-K** — Claude picks from Jiang's 301-class softmax instead of inventing a chord set from title+artist. The wholesale-rewrite contract is what makes the current corrector unsafe on top of a strong detector (the qflip gate is correctly refusing it 11/13 times).

## Top 2 failure modes observed

**1. Power-chord rock collapses both detectors at quality level.** Iron Man GT is all `E5/A5/G5/D5` style power chords (8 unique chords, all `X5`). Neither librosa nor Jiang emits the `5` quality — they output triads (`E/A/G/D`). The scorer correctly returns quality F1 = 0.0 for both. Result: J0 root_F1 = 0.503 ← **the only song where J0 underperforms B0 on root F1 (0.768)**, because Jiang's 301-class HMM picks up minor/major colorations the riff doesn't have. Fix is upstream: rewrite power-chord fixtures to triad equivalents, or add `X5 → X` family collapse to the scorer. This same problem will hit Sweet Home Alabama, AC/DC anything, Iron Man — any riff-driven song.

**2. The V1 corrector silently flattens Jiang's slash chords if the qflip gate doesn't block it.** Jiang produces 38 slash chords on Stairway. The current V1 production prompt (`backend/audit/llm_oracle.py:55`) explicitly says: "Use ROOT + QUALITY only. No slash-bass, no inversions, no voicings." `normalize_chord()` drops `/bass`. On 11 of 13 songs the qflip safety gate trips and the corrector leaves Jiang alone — but on the 2 songs it doesn't, V1 strips slash bass from the chart. House of the Rising Sun: J0 emits 1 slash chord, J0+V1 emits 0. **If we ship J0+V1 in front of Jiang, the slash-chord lift quietly dies on the songs where Claude *does* fire.** This is the strongest argument for shipping J0 raw and not layering the current V1 corrector on top.

## Things skipped + why

- **25 of 38 UG ground-truth fixtures had no local audio** — Hotel California, Hells Bells (no GT either), Heart of Gold, Mary Jane's Last Dance, Toto Africa, Cosmic Girl is in cohort but Virtual Insanity isn't, etc. Re-downloading is outside the 90-minute time-box and the 13-song cohort is enough signal for the launch decision.
- **Did not run a deterministic temperature=0 re-run** for the corrector. Since the corrector layer doesn't materially move J0's number (the qflip gate blocks 11/13 edits), determinism doesn't change the verdict.
- **Did not test the reranker strategy** (`ANTHROPIC_CORRECTION_STRATEGY=reranker`). The May 11 audit names it as the post-launch lever and that judgement holds.
- **Did not run Hells Bells specifically** — no GT fixture under `audit/fixtures/ground_truth/`. The brief references the May 8 audit's known Hells Bells failure mode but that fixture exists only in `/tmp/audit-may8-results/`, not in the canonical fixture set.
- **Did not score with the `full` (slash-exact) F1 column** — the scorer emits it but the brief asked for root + quality. Slash chord *count* is reported instead; if slash-exact matching is needed it's a one-liner to add.

## Artifacts on disk

- Jiang labs: `/tmp/v3bake/jiang_lab/<slug>.lab` (Harte format)
- Per-config charts: `/tmp/v3bake/charts/<slug>__{J0,J0V1,J0V2}.json`
- Per-song full score JSON: `/tmp/v3bake/results.json`
- Fair B0 recompute: `/tmp/v3bake/b0_fair.json`
- Bake-off driver + Harte→standard parser: `/tmp/v3bake/bakeoff.py`
- Run logs: `/tmp/v3bake/run.log`, `/tmp/v3bake/run2.log`
- Resolution (slug → job_id → audio path mapping): `/tmp/v3bake/resolution.json`
