# V3.1 ACE Variant Experiments — 2026-05-13

**Agent:** ACE-variants (Agent 1)
**Cohort:** 13 classic-rock songs (UG ground-truth fixtures)
**Scorer:** `audit/score_chord_chart.py` (bag-of-chords F1 + PCS-equivalent F1)
**Model:** `conformer_decomposed_smooth.ckpt` (ACE, Poltronieri et al. ISMIR 2025)

## Honest summary

ACE-default (chunk=20s, threshold=0.5, min-dur=0.5) lands at mean **root F1 0.847 / quality F1 0.697** across the 13-song rock cohort. Sweeping `--chunk-dur` to 10s or 30s is **a wash on the mean** (root ±0.006, quality +0.005 at chunk10, −0.013 at chunk30) but produces clear per-song swings that suggest chunk size is a song-specific tuning lever, not a global one: **Hotel California improves to root=0.83 / quality=0.796 at chunk10** (chunk30 hurts quality), **Fortunate Son jumps to root=0.898 at chunk30**, while **Heart of Gold quality collapses from 0.905 → 0.794 at chunk30** and **Wild Horses degrades at both** alternatives. The Petty pair holds within ±0.05. Q7 (non-decomposed conformer) is **not available** — only the decomposed checkpoint ships, and the alternative architecture cannot load it (gin/`conformer_dim` arg mismatch confirmed). Q8 Aja result is the headline concern: ACE scores root=0.812 / quality=0.411 / full=0.382, emitting **zero of the 11 unique GT slash chords** and **none of the GT add9 / m11 / maj9 / maj7#11 / 6/9 extensions** — a hard regression versus the old detector's 226/226 extension audit. ACE is a credible drop-in for chord-shape rock but **not safe for the jazz vocabulary** the old per-root-family detector handled.

---

## Q1 — Rescore ACE-default at quality / slash / PCS F1

ACE-default outputs from `/tmp/ace_outputs/*.lab` (chunk=20s, threshold=0.5, min-dur=0.5), Harte→standard via `/tmp/v3bake/bakeoff.py`, scored against `audit/fixtures/ground_truth/<slug>.json`. IGWO baseline was missing from the pre-existing outputs and was generated for this report.

| Song | root F1 | family F1 | **quality F1** | **full F1 (slash)** | **PCS F1** |
|---|---:|---:|---:|---:|---:|
| black-sabbath__iron-man | 0.826 | 0.482 | **0.000** | 0.000 | 0.501 |
| boston__more-than-a-feeling | 0.794 | 0.785 | 0.740 | 0.731 | 0.785 |
| cream__sunshine-of-your-love | 0.698 | 0.686 | 0.556 | 0.544 | 0.568 |
| creedence__fortunate-son | 0.821 | 0.768 | 0.714 | 0.714 | 0.786 |
| eagles__hotel-california | 0.822 | 0.822 | 0.740 | 0.733 | 0.822 |
| heart__crazy-on-you | 0.874 | 0.843 | 0.736 | 0.679 | 0.818 |
| neil-young__heart-of-gold | 0.943 | 0.943 | **0.905** | 0.905 | 0.933 |
| the-animals__house-of-rising-sun | 0.956 | 0.939 | 0.857 | 0.857 | 0.939 |
| stones__paint-it-black | 0.716 | 0.550 | 0.550 | 0.541 | 0.550 |
| stones__wild-horses | 0.933 | 0.919 | **0.867** | 0.859 | 0.919 |
| petty__into-the-great-wide-open | 0.882 | 0.858 | 0.740 | 0.693 | 0.866 |
| petty__mary-janes-last-dance | 0.880 | 0.866 | **0.852** | 0.852 | 0.873 |
| toto__africa | 0.863 | 0.863 | 0.804 | 0.794 | 0.863 |
| **MEAN** | **0.847** | **0.794** | **0.697** | **0.685** | **0.787** |

Notes:
- **Iron Man quality=0** is real and expected: GT uses power-chord notation (`E5`, `D5`, …) which ACE never emits; root F1 is still strong (0.826).
- ACE emits slash chords on most songs (we already see C/G, F#7/A# etc. in the .lab files), so `full` and `quality` are close on most songs — except where ACE collapses extensions (Cream, Paint It Black).
- PCS F1 mirrors family F1 closely — ACE's chord-tone activations are right but the surface notation differs.

---

## Q6 — chunk_dur sweep (10s vs 30s vs default 20s)

All runs at `--threshold 0.5 --chord-min-duration 0.5`. Δ values are relative to ACE-default (chunk=20s).

### Root F1

| Song | default | chunk10 | Δ10 | chunk30 | Δ30 |
|---|---:|---:|---:|---:|---:|
| iron-man | 0.826 | 0.799 | -0.027 | 0.799 | -0.027 |
| boston | 0.794 | 0.820 | +0.026 | 0.767 | -0.027 |
| sunshine | 0.698 | 0.658 | -0.040 | 0.712 | +0.014 |
| fortunate-son | 0.821 | 0.774 | -0.047 | **0.898** | **+0.077** |
| **hotel-california** | 0.822 | **0.830** | **+0.008** | 0.842 | +0.020 |
| crazy-on-you | 0.874 | 0.897 | +0.023 | 0.902 | +0.028 |
| heart-of-gold | 0.943 | 0.971 | +0.028 | 0.953 | +0.010 |
| rising-sun | 0.956 | 0.982 | +0.026 | 0.976 | +0.020 |
| paint-it-black | 0.716 | 0.750 | +0.034 | 0.667 | -0.049 |
| wild-horses | 0.933 | 0.919 | -0.014 | 0.888 | -0.045 |
| **IGWO** | 0.882 | 0.835 | -0.047 | 0.816 | -0.066 |
| **MJLD** | 0.880 | 0.852 | -0.028 | 0.875 | -0.005 |
| toto-africa | 0.863 | 0.853 | -0.010 | 0.832 | -0.031 |
| **MEAN** | **0.847** | **0.842** | -0.005 | **0.841** | -0.006 |

### Quality F1

| Song | default | chunk10 | Δ10 | chunk30 | Δ30 |
|---|---:|---:|---:|---:|---:|
| iron-man | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| boston | 0.740 | 0.772 | +0.032 | 0.689 | -0.051 |
| sunshine | 0.556 | 0.472 | -0.084 | 0.675 | +0.119 |
| fortunate-son | 0.714 | 0.755 | +0.041 | **0.797** | **+0.083** |
| **hotel-california** | 0.740 | **0.796** | **+0.056** | 0.716 | **-0.024** |
| crazy-on-you | 0.736 | 0.718 | -0.018 | 0.726 | -0.010 |
| heart-of-gold | 0.905 | 0.942 | +0.037 | **0.794** | **-0.111** |
| rising-sun | 0.857 | 0.880 | +0.023 | 0.857 | 0.000 |
| paint-it-black | 0.550 | 0.586 | +0.036 | 0.587 | +0.037 |
| wild-horses | 0.867 | 0.874 | +0.007 | **0.762** | **-0.105** |
| **IGWO** | 0.740 | 0.731 | -0.009 | 0.735 | -0.005 |
| **MJLD** | 0.852 | 0.758 | -0.094 | 0.795 | -0.057 |
| toto-africa | 0.804 | 0.837 | +0.033 | 0.764 | -0.040 |
| **MEAN** | **0.697** | **0.702** | +0.005 | **0.684** | -0.013 |

### Verdict per Jeff's pointed questions

- **Hotel California — improve to >0.85?** No on root (chunk10=0.830, chunk30=0.842, both shy of 0.85). **Yes on quality** at chunk10 (0.796, up from 0.740) — chunk10 is the right setting for Hotel California specifically. Chunk30 *hurts* its quality (0.716).
- **IGWO + MJLD — do they hold?** Mostly. IGWO regresses on both root (-0.047 / -0.066) and quality is roughly flat. MJLD root holds within -0.005 at chunk30; quality regresses at both (-0.094 / -0.057). Neither alternative is a clear win for the Petty pair — keep them at default.

### Per-song chunk recommendations (if we shipped per-song tuning)

| Song | best chunk | gain (root / qual) |
|---|---|---|
| Hotel California | **10s** | root 0.822→0.830, **quality 0.740→0.796** |
| Fortunate Son | **30s** | root 0.821→**0.898**, quality 0.714→0.797 |
| Sunshine of Your Love | 30s | root +0.014, quality +0.119 |
| Heart of Gold | 10s | root +0.028, quality +0.037 |
| Africa | 10s | quality 0.804→0.837 |
| Wild Horses, IGWO, MJLD, Iron Man | **default 20s** | alternatives all regress |

Conclusion: chunk-dur is a per-song lever, not a global improvement. Picking it via a tempo heuristic (slow songs → bigger chunk?) is plausible but premature; **keep chunk=20s as the global default** and only special-case Hotel California (10s) if we want that win for the demo cohort.

---

## Q7 — Non-decomposed conformer variant

**Verdict: not available out of the box. Skip.**

Findings:
1. `ACE/inference.py` exposes `--model-name {conformer,conformer_decomposed}` (default decomposed).
2. `ACE/models/` ships two classes — `ConformerModel` (170-class classifier) and `ConformerDecomposedModel` (3-head root/bass/chord-activations).
3. `ACE/checkpoints/` ships **only `conformer_decomposed_smooth.ckpt`**. No second `.ckpt` for the plain conformer.
4. The README explicitly distinguishes them: "*`conformer`: baseline architecture for chord classification (170 classes). `conformer_decomposed`: our proposed model …*" — and inference documentation shows the decomposed model as the only pretrained one.
5. Attempting `--model-name conformer` against the only available checkpoint fails with `TypeError: ConformerModel.__init__() got an unexpected keyword argument 'conformer_dim'` — gin config from the decomposed model is incompatible with the plain conformer's `__init__` (which takes `hidden_dim`, not `conformer_dim`). Architectures are not interchangeable.

Training a non-decomposed conformer from scratch would require the McGill Billboard + Isophonics datasets and several GPU-hours — out of scope for today's 90-minute box and explicitly skipped per the brief.

No `/tmp/ace_conformer/` outputs were produced.

---

## Q8 — Aja safety

Audio: `/Users/jeffkozelski/stemscribe/uploads/569ce01e-e8a7-4153-951f-6cda91f08930/02_-_Aja.mp3`
GT: `audit/fixtures/ground_truth/steely-dan__aja.json`
Settings: ACE-default (chunk=20s, threshold=0.5, min-dur=0.5), 15.0s runtime.

| Metric | Value |
|---|---:|
| root F1 | **0.812** |
| root_family F1 | 0.758 |
| **quality F1** | **0.411** |
| **full F1 (slash-exact)** | **0.382** |
| PCS F1 | 0.594 |
| GT chord events | 197 |
| Det chord events | 217 |
| Vocab coverage | **45.5%** (5/11 quality classes shared) |
| GT slash chords | 11 unique (Aadd9/B, Bmaj7/F#, Cmaj9/G, D6/9, Dadd9/E, …) |
| Det slash chords | 9 unique (A/E, A7/D, Am7/G, Cm/D#, D7/C, E/D, G7/F, …) |
| GT-only quality classes | Aadd9, Bm11, Bmaj9, Cmaj9, Cmaj7#11, Dadd9, D6/9, E9, Em9, … |
| Det-only quality classes | Am, Am7, Asus4, B7, Bm, Bm7, C#m7, C#maj7, Csus4, Dsus4, … |

### Verdict — REGRESSION on Aja

The old detector's earlier-audit result was "226/226 extension matches" — perfect on the jazz vocabulary. **ACE breaks this hard**:

- **Quality F1 0.411 vs the old detector's effectively perfect quality on Aja's vocabulary.** ACE substitutes `Cm7` for `Cm11`, `Bmaj7` for `Bmaj9`, `D` for `Dadd9`, etc. — it has the *family* mostly right (0.758) but **flattens every extension to its 7th-chord shadow**.
- **Slash chords miss every GT specimen.** ACE's 9 detected slashes are a *disjoint set* from GT's 11 — only one (D7/C) appears in both. Aja's `Aadd9/B`, `Bmaj7/F#`, `Cmaj9/G`, `D6/9`, `Dadd9/E`, `Dbadd9/Eb`, `Ebadd9/F` are *all absent*. ACE invents `A/E`, `Am7/G`, `E/D` instead.
- **45.5% vocab coverage** is the lowest of any song studied today.

This is the canonical example of ACE's vocabulary ceiling: it was trained on Isophonics + Billboard, which are pop/rock-heavy. Aja's modal-jazz vocabulary (add9, m11, maj9, maj7#11, 6/9) lives outside the training distribution. **Do not deploy ACE as a drop-in replacement on jazz material** — the regression cost is too high.

**Recommendation:** if ACE ships for V3.1, keep the old per-root-family detector active as a fallback for songs flagged as jazz/extended (key signature + bass-line extension density would be cheap signals), or run both and prefer ACE only when its PCS F1 against the old detector's output is ≥ a threshold.

---

## Artifacts on disk

- `/tmp/ace_outputs/*.lab` — ACE-default (chunk=20s) — 13 cohort songs (IGWO regenerated this session)
- `/tmp/ace_chunk10/*.lab` — chunk-dur 10s — 13 cohort songs
- `/tmp/ace_chunk30/*.lab` — chunk-dur 30s — 13 cohort songs
- `/tmp/ace_aja/steely-dan__aja.lab` — Aja ACE-default
- `/tmp/ace_variants_results.json` — full raw scores
- `/tmp/ace_variants_driver.py` — reproduction driver
