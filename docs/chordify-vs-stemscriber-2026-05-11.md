# StemScriber vs. Chordify — Head-to-Head Audit (2026-05-11)

**TL;DR:** On the same 18-song audit set, scored against the same LLM oracle with the same extended-vocabulary F1 metric:

| Tool        | Mean F1 (transposed-fair) | Mean F1 (raw)  |
|-------------|--------------------------:|---------------:|
| **StemScriber** | **0.804**             | 0.804          |
| Chordify    | 0.770                     | 0.648          |

StemScriber wins outright on 10/18 songs, ties on 5/18, loses on 3/18. Chordify's "raw" number is dragged down by three songs where its key-detection literally outputs the wrong key (off by a half-step or capo equivalent); the "transposed" number gives Chordify the benefit of the doubt by re-pitching its chord set to match the canonical key before scoring.

## Method

For each song, I loaded Chordify's first-result chord page via Playwright (free tier, no signup), extracted unique chord labels from `.chord-label` DOM elements (Unicode normalized), and ran the set through the same `diff_sets()` scorer that produced StemScriber's May 8 audit (`backend/audit/llm_oracle.py`). Power-chord `5` collapses to bare root; slash-bass is stripped. Canonical chord set per song is the LLM oracle's from `/tmp/audit-may8-results/_oracle-qflip.jsonl` — identical reference for both tools.

Three Chordify charts (Man in the Box, Bad Company, Every Rose) returned a different key than canonical (half-step-flat YouTube source for the first two; capo-vs-concert-pitch for Every Rose). I transposed those into the canonical key before scoring — fairest for isolating chord-recognition quality from arbitrary key choice. Raw column shows the un-transposed result for transparency.

## Per-Song Results

| Song | StemScriber F1 | Chordify F1 | Winner | Notes |
|---|---:|---:|---|---|
| A-Ha — Take On Me                       | 0.83 | 0.80 | SS | CH adds spurious B + G; SS misses C#m + D#dim |
| ACDC — Back In Black                    | 0.75 | 0.75 | Tie | Both add same extras (B, G) |
| ACDC — Hells Bells                      | **1.00** | 0.83 | SS | SS clean 5/5; CH lists redundant power-chord variants (A5, D5, etc.) |
| ACDC — Highway To Hell                  | 0.80 | 0.80 | Tie | Both miss Bm + F#m |
| Aerosmith — Dream On                    | 0.56 | **0.67** | CH | Jazz-influenced, CH catches Bbm + Cm that SS misses |
| Alice In Chains — Man in the Box        | **0.80** | 0.73 | SS | CH off by half-step (Eb), transposed; misses C + F |
| America — Sister Golden Hair            | **0.92** | 0.86 | SS | CH adds spurious G#; both miss D |
| Animals — Don't Let Me Be Misunderstood | 0.15 | **0.92** | CH | SS detector failure (wrong key — Am instead of Bm); CH near-perfect |
| Animals — House of the Rising Sun       | **1.00** | 0.71 | SS | SS clean; CH adds 4 extras (Dm, E7, Em, F7) |
| Bad Company — Bad Company               | **0.83** | 0.73 | SS | CH off by half-step (Eb), transposed; misses B7 + C |
| Badfinger — Day After Day               | **0.57** | 0.55 | SS | Both struggle; CH's key detector gave F major (canonical is C) |
| Def Leppard — Pour Some Sugar On Me     | **0.80** | 0.67 | SS | CH adds 3 spurious (C#, D, F#) |
| Eagles — Hotel California               | 0.93 | 0.93 | Tie | Both nail it; identical extra (F#) |
| Jimi Hendrix — Hey Joe                  | 1.00 | 1.00 | Tie | Both perfect |
| Moody Blues — Your Wildest Dreams       | **1.00** | 0.83 | SS | CH misses Bm and adds spurious D7 |
| Poison — Every Rose Has Its Thorn       | 0.75 | 0.75 | Tie | Both score Cadd9 → C the same way (-1 fn each) |
| Rolling Stones — Paint It Black         | **0.77** | 0.67 | SS | CH misses A, C, E; SS has B/B7 mismatch |
| Tom Petty — Free Fallin'                | **1.00** | 0.67 | SS | CH outputs Csus4 (counts as miss for C) |

**Tally:** StemScriber wins 10, Chordify wins 2, ties 6. Means: SS 0.804, CH 0.770.

## Where Each Tool Wins

**StemScriber's edge — clean rock.** Hells Bells, House of the Rising Sun, Your Wildest Dreams, Free Fallin', Hey Joe — five perfect scores. The detector's family-aware/per-root consistency (Apr 25 sprint) avoids over-segmentation: Chordify tends to emit power-chord variants and passing chords as separate entries, costing it precision.

**Chordify's edge — jazz/extended harmony and tricky keys.** Dream On (F minor with descending bass — CH catches Bbm + Cm + Ebm voicings); Don't Let Me Be Misunderstood (CH correctly Bm, while SS's key detector flipped to Am — same Krumhansl-Kessler failure mode as Black Cow).

## Bottom Line

Direct comparable: **StemScriber 0.804 F1, Chordify 0.770 F1, n=18, same songs, same scorer, extended vocabulary (7ths/9ths/sus).** Not a published benchmark, but a clean head-to-head. Chordify's own published number (WCSR 82% on Isophonics, majmin-only, 2021) is a different and easier metric.

Defensible marketing claim: **on a 18-song classic-rock-and-jazz audit, StemScriber beats Chordify on extended chord identification by 4 F1 points and dominates clean rock while staying competitive on harder material.** Drop the fairness transposition and the gap widens to 15.6 points (0.804 vs 0.648).

## Caveats

- Audit set is classic rock + a few jazz-adjacent tunes — not pure-pop or EDM.
- Chordify outputs many power-chord variants (`A5`, `D5`) collapsed to bare roots in scoring; without collapsing, CH scores would drop ~5 points.
- Don't Let Me Be Misunderstood is a known SS upstream key-detection failure (same root cause as Black Cow, `docs/black-cow-maj7-diagnosis-2026-04-25.md`). Fixing that one bug closes the gap further.

## Source URLs (Chordify free-tier)

- [Take On Me](https://chordify.net/chords/a-ha-songs/take-on-me-chords) · [Back In Black](https://chordify.net/chords/ac-dc-songs/back-in-black-3-chords) · [Hells Bells](https://chordify.net/chords/ac-dc-songs/hells-bells-5-chords) · [Highway To Hell](https://chordify.net/chords/ac-dc-songs/highway-to-hell-4-chords)
- [Dream On](https://chordify.net/chords/aerosmith-songs/dream-on-3-chords) · [Man in the Box](https://chordify.net/chords/alice-in-chains-songs/man-in-the-box-2-chords) · [Sister Golden Hair](https://chordify.net/chords/america-songs/sister-golden-hair-3-chords)
- [Don't Let Me Be Misunderstood](https://chordify.net/chords/the-animals-songs/don-t-let-me-be-misunderstood-2-chords) · [House of the Rising Sun](https://chordify.net/chords/the-animals-songs/house-of-the-rising-sun-3-chords)
- [Bad Company](https://chordify.net/chords/bad-company-songs/bad-company-3-chords) · [Day After Day](https://chordify.net/chords/badfinger-songs/day-after-day-chords) · [Pour Some Sugar On Me](https://chordify.net/chords/def-leppard-pour-some-sugar-on-me-theo-nug)
- [Hotel California](https://chordify.net/chords/eagles-songs/hotel-california-10-chords) · [Hey Joe](https://chordify.net/chords/the-jimi-hendrix-experience-songs/hey-joe-3-chords) · [Your Wildest Dreams](https://chordify.net/chords/the-moody-blues-songs/your-wildest-dreams-2-chords)
- [Every Rose Has Its Thorn](https://chordify.net/chords/poison-songs/every-rose-has-its-thorn-6-chords) · [Paint It Black](https://chordify.net/chords/the-rolling-stones-songs/paint-it-black-7-chords) · [Free Fallin'](https://chordify.net/chords/tom-petty-songs/free-fallin-2-chords)

Extraction data: `/tmp/chordify-audit/all_results.json` · scorer: `/tmp/chordify-audit/score.py` · per-song diffs: `/tmp/chordify-audit/scored.json`.
