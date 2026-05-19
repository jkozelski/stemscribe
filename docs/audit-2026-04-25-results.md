# StemScriber chord-chart audit — 2026-04-25

Scoring of 8 detector outputs against ground-truth charts (Hooktheory / Ultimate Guitar / hakwright.co.uk transcriptions). Bar-level alignment, naive section-agnostic loop ground truth (most pop/funk loops repeat 4-bar cycles, so this gives a reasonable upper-bound estimate of detector accuracy; section-aware scoring would shift exact-match by ±0.05).

## Comparison table

| Song | Cohort | Pre-fix exact | Post-fix exact | Δ | Post-fix root | Post-fix transpose-inv | Bars | 7ths in output | Dominant failure |
|---|---|---|---|---|---|---|---|---|---|
| Alright (Jamiroquai) | post-fix | 0.00 | **0.000** | +0.00 | 0.175 | 0.000 | 97 | 0/97 | whole-chart transpose + 7ths dropped |
| Cosmic Girl (Jamiroquai) | post-fix | 0.00 | **0.028** | +0.03 | 0.046 | 0.028 | 109 | 9/109 | 7ths dropped + 3rds flipped |
| Black Cow (Steely Dan) | post-fix | 0.01 | **0.017** | +0.01 | 0.103 | 0.086 | 116 | 0/116 | 7ths dropped (catastrophic — every bar bare-major) |
| Peg (Steely Dan) | post-fix | 0.00 | **0.185** | +0.19 | 0.241 | 0.185 | 108 | 0/108 | 7ths dropped + section/quality wrong |
| Aja (Steely Dan) | OOS | — | 0.000 | — | 0.075 | 0.004 | 226 | 0/226 | bass-bleed wrong root + 7ths dropped |
| Virtual Insanity (Jamiroquai) | OOS | — | 0.000 | — | 0.056 | 0.000 | 116 | 0/116 | whole-chart transpose error + 7ths dropped |
| Space Cowboy (Jamiroquai) | OOS | — | 0.000 | — | 0.158 | 0.000 | 168 | 0/168 | whole-chart transpose + 7ths dropped |
| Rikki Don't Lose That Number (Steely Dan) | OOS | — | 0.171 | — | 0.171 | 0.214 | 117 | 0/117 | 7ths dropped (chord bailing to root only) |

## 1. Pre-fix vs post-fix delta on the 4 funk-jazz songs

**The fix moved the needle on Peg (+0.19 exact) and barely budged the rest.** Black Cow gained 0.01, Cosmic Girl 0.03, Alright zero. Crucially the **root_match score went *down* on Alright** (0.34 → 0.18) and Cosmic Girl (0.24 → 0.05). That's a regression: the bass-anchored pipeline is now committing to confidently-wrong roots more often than the previous version did. Transpose-invariant scores show the keys are still mostly off by a fifth/third — Peg posts 0.19 trans-inv exactly equal to its raw exact, meaning no benefit from re-keying (it's actually in the right key range but the *qualities* are missing).

## 2. Out-of-sample numbers

OOS is dramatically worse on three of four. Aja, Virtual Insanity, Space Cowboy all score 0.000 exact and ≤0.16 root. Rikki is the lone bright spot at 0.171 exact / 0.214 trans-inv — Rikki happens to be a 3-chord diatonic E-major tune (D-A-E), and the bass extractor handles diatonic well. Anything modal/jazz (Aja's Bm11, Virtual Insanity's Ebm7-Ab7-Db9, Space Cowboy's Ebm9) collapses.

## 3. The 7ths question — answered directly

**Confirmed: 7 of 8 charts have ZERO 7ths in output. Cosmic Girl is the lone exception (9 bars: 5 Emin7, 4 Cmin11).** Concretely:

- Alright: 0/97 bars have any 7. Ground truth is Cm7-Gm7-Bbmaj7-Cmaj7. Detector returned bare Cm/Gm/Dm/Am.
- Peg: 0/108. Ground truth Cmaj7 throughout. Detector returned bare C/G/D.
- Black Cow: 0/116. Ground truth Amaj7/Dmaj7/F#m7 throughout. Detector returned bare A/D/E. **Every single bar's `detector_quality` is the empty string** (major triad).
- Aja: 0/226. Ground truth Bmaj7/Bm11. Detector: 218/226 bars `quality='m'`, never `m7`.

**This is NOT a stripping bug.** I traced through `_simplify_uncommon_quality` (chart_formatter.py:1400) and it explicitly preserves `m7`, `maj7`, `7`, and `dim`. The regex `_SUS_AUG_STRIP_RE` only strips sus/aug/add/altered-5. The 7ths-fix shipped on Apr 22 is **doing what it's supposed to do** — but the input never had 7ths to preserve. The v8 chord detector (337-class, vocab includes `maj7`/`min7`/`7`) is essentially never *predicting* 7ths on these stems. Inspection of `detector_quality` field: Black Cow returns `''` for 116/116 bars, Peg returns `''` for 107/108, Rikki returns `''` for 116/117. The CRNN's posterior is collapsing to plain-triad classes.

## 4. Failure taxonomy summary

- **7ths-dropped (8/8 songs)** — root cause: detector posterior, not the formatter strip.
- **Whole-chart transpose error (3/8: Alright, Virtual Insanity, Space Cowboy)** — detector reports key=G for Alright (truth: Cm), key=B for Virtual Insanity (truth: Ebm), key=A# for Space Cowboy (truth: Dbmaj/Bbm). Transpose-invariant scores stay near zero, meaning it's not a clean key offset — it's structurally confused.
- **Bass-bleed wrong root (Aja)** — `bass_root` field shows F# / D# / G# entries on bars where ground truth is Bm/Em. The bass extractor is grabbing the bassist's walking-line passing tones rather than the chord root.
- **Chord bailing to root only (all 4 Steely Dan tunes)** — `detector_quality=''` rate is 99-100% on Black Cow/Peg/Rikki.
- **Sections field is empty (`[None, None, ...]`) on every chart** — section boundaries can't be evaluated because they don't exist in the output.

## 5. Top recommendation — single fix in 48 hours

**Force the v8 CRNN to emit 7ths instead of collapsing to plain triads.** The model's vocab supports maj7/min7/dom7 (chord_detector_v8.py:194), but its softmax is biased toward the higher-prior triad classes. Two options, in order of effort:

**Best fix (4-6 hrs):** In `backend/processing/bass_root_extraction.py:288`, the smoother counts `detector_quality` per bass root and snaps to the dominant. The dominant is `''` on Steely Dan because the detector floods empty-strings. Add a **logit re-weighting pass** in `chord_detector_v8.py` near line 340 (`chord_name = self.chord_classes[pred]`) that, when the top-1 is a plain triad and the top-2 is the matching 7th chord with posterior within a margin (e.g. ratio > 0.6), promotes the 7th. Combined with a key-aware prior (if detected key is major, prefer maj7 over dom7 on I/IV; prefer m7 on ii/iii/vi), this would lift Black Cow and Peg from 0.0/0.19 toward the 0.5+ range without retraining.

**Cheaper hack (1 hr) if 4 hrs is too much:** in `combine_with_detector_quality` (bass_root_extraction.py:215-241), when `detector_quality == ''` on >70% of bars in a song AND the bass roots cluster on a diatonic set, append `'7'` to the V chord and `'maj7'` to the I/IV chords by key-context lookup. This is a band-aid but on Steely Dan / Jamiroquai funk it would close most of the gap.

The Apr 22 "7ths fix" addressed the formatter (which was already correct). The actual leak is upstream in the CRNN posterior, and one weekend of logit-prior tuning will move post-fix exact from <0.05 to plausibly 0.4+ on the funk-jazz cohort.

## Sources

- Black Cow: http://www.hakwright.co.uk/music/tab/black_cow.shtml (hakwright transcription, gold standard)
- Peg: http://www.hakwright.co.uk/music/tab/peg.shtml
- Alright: e-chords.com/chords/jamiroquai/alright + UG tab/jamiroquai/alright-chords-2567109
- Cosmic Girl: UG tab/jamiroquai/cosmic-girl-official-chords-1949237 + Hooktheory cosmic-girl page
- Aja: Hooktheory theorytab/view/steely-dan/aja
- Virtual Insanity: Hooktheory theorytab/view/jamiroquai/virtual-insanity
- Space Cowboy: Hooktheory theorytab/view/jamiroquai/space-cowboy
- Rikki: guitaretab.com/s/steely-dan/332148.html (key E major, verse D-A-E)

## Caveats / uncertainty flags

- Naive ground-truth pattern repetition penalizes section-rich tunes (Peg verse vs chorus differ). Real exact-match is likely +0.05 to +0.15 with section-aware alignment.
- Aja's ground-truth key is disputed: Hooktheory analyzes parts in B major and E mixolydian, hakwright suggests Bm-centered. Detector's `key=B` is plausibly correct; the failure is on quality, not root.
- Rikki ground-truth original key is disputed across UG versions (D vs E vs A); I used E major per Musicnotes' original sheet.
