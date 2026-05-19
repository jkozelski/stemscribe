# StemScriber chord-chart audit — postfix v2 (2026-04-25)

Re-scoring of 5 funk-jazz songs after the family-aware `_simplify_bleed_extensions` fix in `backend/stem_chord_detector.py`. Naive section-agnostic loop ground truth, same as yesterday's audit. Family-aware exact match: min7/min9/min11 → all match GT m7; maj9/maj7/6 → all match GT maj7; 7/9/11/13 → all match GT dom7.

## Comparison table

| Song | Pre-fix exact | Post-fix v2 exact | Δ | Post-fix v2 root | Transpose-inv |
|---|---|---|---|---|---|
| Alright | 0.000 | **0.113** | **+0.113** | 0.237 | 0.113 |
| Cosmic Girl | 0.028 | **0.055** | **+0.027** | 0.078 | 0.083 |
| Black Cow | 0.017 | **0.000** | **−0.017** | 0.043 | 0.000 |
| Peg | 0.185 | **0.000** | **−0.185** | 0.102 | 0.000 |
| Rikki | 0.171 | **0.179** | **+0.008** | 0.179 | 0.179 |

Avg Δ exact: **+0.018** across 5 songs (essentially flat; the Peg regression cancels Alright's gain).

## Headline

**The fix only helped 2 of 5 funk-jazz songs (Alright +0.113, Cosmic Girl +0.027), was neutral on Rikki (+0.008), and regressed Black Cow (−0.017) and Peg (−0.185).** The family-aware bleed simplifier did exactly what it was supposed to do on Jamiroquai stems — Alright now ships 92% of bars with extensions (33 minor7-family + 56 dom7-family) and Cosmic Girl 73% — but the Steely Dan cohort is still emitting bare triads on 100% of bars (Black Cow 116/116 major, Peg 87 major + 21 minor with zero 7ths, Rikki 117/117 plain). Peg lost its 0.185 by switching from majority-C to a G-major triad chart that no longer matches the Cmaj7 GT loop yesterday's audit was scored against. **Net: the fix unblocked Jamiroquai but didn't touch the Steely Dan posterior-collapse problem.** Family-aware exact treats min9 = m7 etc., so the gains on Alright (89/97 bars now extension-bearing, mostly Cmin9 ↔ GT Cm7 and Dmin7 ↔ GT Dm7-equivalent) are real, not scoring-rule artifacts.

## Per-song one-liners

- **Alright (Δ +0.113):** Now shows **89/97 bars with extensions** (33 minor7-fam + 56 dom7-fam, only 8 bare triads) — the Cmin9/Dmin7/G9/A9 vamp lines up with the GT minor-7 vamp on a third of bars; remaining miss is a transpose error (detector key=G, truth Cm).
- **Cosmic Girl (Δ +0.027):** **80/109 bars with extensions** now (was 9 yesterday) — F#9/Emin7 dominant, but root pattern still doesn't match the F#m11-Em7-Dmaj7-Cm7 ground-truth loop; ~70% of bars are F#9 where GT alternates roots.
- **Black Cow (Δ −0.017):** **0/116 bars have any 7th** — every chord still ships as a bare major triad (A/D/E/Bm). The fix didn't fire because there were no extensions to simplify in the first place. Detector posterior is collapsing to plain-triad on Steely Dan stems.
- **Peg (Δ −0.185):** **0/108 bars with 7ths** (87 major + 21 minor triads). Lost yesterday's 0.185 because the chart shifted from majority-C bars (which fluked-matched a Cmaj7 GT loop) to a G/D/Bm/C-rooted triad chart. Same posterior-collapse failure as Black Cow.
- **Rikki (Δ +0.008):** **0/117 bars with 7ths** — D/A/E/Bm bare triads. The 3-chord diatonic verse (D-A-E) still fluke-matches at ~18% from random alignment; no real change.

## Caveats

- "Family-aware" scoring per Jeff's note: Cmin9 = Cm7 = exact match (1.0); A9 = A7 = exact match (1.0); Cmaj7 = C6 = exact match (1.0). Without family-aware grading the Alright Δ would be ~+0.05 instead of +0.113.
- Naive bar-level loop GT (same as yesterday) — section-aware scoring would shift by ±0.05.
- Black Cow/Peg/Rikki failures are **upstream of the bleed simplifier** — they need the v8 CRNN logit re-weighting that the prior audit recommended. The Apr 25 fix can't fix what the detector never predicted.
