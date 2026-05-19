# Honest Chord-Chart Scorer (v2) — Rationale & Validation

**Status:** for human review. New scorer added alongside (NOT replacing) the
v1 bag-of-chords scorer. No production or deploy changes.
**Branch:** `feat/honest-chord-scorer-v2`
**Files:** `audit/score_chord_chart_v2.py`, `audit/validate_scorer_v2.py`,
`audit/tests/test_score_chord_chart_v2.py`, prod charts pinned in
`audit/fixtures/prod_charts/`.

---

## 1. Why v1 had to be replaced

`audit/score_chord_chart.py` (v1) is a **bag-of-chords** metric: it throws
every chord into a multiset and compares multisets. It ignores ORDER,
PLACEMENT, and — at its most-cited `root` level — FLAVOR.

Measured consequence (three prior forensic agents, reproduced here):

| Chart | v1 `root` F1 | v1 `root_quality` F1 | Musician verdict |
|---|---|---|---|
| "In My Life" served prod (job fb8a175e) | **0.919** | 0.685 | "basically root notes, missing the Dm / A7 / B7" — the **worst** chart of every config tested |

v1's 0.919 rewarded librosa's single best trait (it often gets the right
chord *names somewhere* in the song) while being blind to its worst traits
(wrong placement; dropped 7ths/min). That number made a genuinely ~25% better
detector (ACE/Jiang) look like a regression and has been misleading every
detector decision.

A musician needs **three** independent things from a chart. A single
composite hides exactly the failure that misled every prior decision, so v2
**always reports the per-axis breakdown**.

---

## 2. What each axis measures, and WHY

### AXIS 1 — Root correctness (order-aware, NOT a bag)
*What:* order-aware LCS F1 over the run-length-encoded root sequence (GT bar
sequence vs detector segment sequence).
*Why:* "is the right root called, in roughly the right place in the
progression". RLE removes hold-length noise so this is a clean
*vocabulary-in-order* signal. This deliberately replaces v1's bag `root`
level (the 0.919) — the same question, asked honestly.

### AXIS 2 — Placement correctness (three variants, all reported)
GT fixtures are **bar-indexed with no absolute timestamps**; detector output
is an ordered segment list with relative times + a `bars` hold count.
Placement is therefore a **sequence-alignment** problem, not a time problem.
Three reads separate the failure modes:

- **strict_bar** — position-for-position root match at zero offset, GT cell
  expanded to its bar and detector segments expanded by `bars`. The harshest,
  most literal "right chord in the right bar". This is what a player reading
  along a lyric sheet actually experiences.
- **best_offset** — strict, but search a global ±N-bar shift and keep the
  best. A large gap over `strict_bar` means the whole chart is merely
  *mis-anchored* (an intro-count mismatch — musically harmless and cheap to
  fix), NOT genuinely misplaced.
- **hold_invariant** — RLE both sides, then order-aware LCS. Removes "held
  N bars vs M bars" + tempo-scale noise; pure progression-order. Reported,
  but **deliberately NOT the composite representative** — using the most
  lenient placement variant is exactly the leniency that produced v1's
  misleading picture.

The composite uses **best_offset** for placement: it forgives the one
musically-harmless error (constant anchoring shift) and nothing else.

### AXIS 3 — Flavor / quality, broken out by class
Per-class accuracy, graded **only at root-aligned positions** (you cannot
grade the flavor of a chord the detector never root-identified — AXIS 1
already penalizes that):

| Class | Weight | Why |
|---|---|---|
| `maj_min` | 1.0 | maj↔min is the most fundamental flavor; always matters |
| `triad_dom7` | 1.0 | dominant-7 is the blues/funk character; structurally load-bearing |
| `triad_maj7_min7` | 0.8 | maj7/min7 color; important, slightly less than dom7 |
| `sus_add_ext` | 0.5 | sus/add/9/11/13 embellishment |
| `slash_inv` | **0.0** | slash/inversion is **directional only** — slash detection is a known gated-off gap; reported, never penalized |

**Aggregation principle (the anti-v1 fix):** the flavor composite is the
structural-weighted mean of per-class *accuracies* (each class counts once,
scaled by musical weight, independent of instance count), blended 60/40
toward the **weakest** structurally-weighted class. A *systematic* color
failure (e.g. 0/4 dom7 — the detector never hears the 7th) is what a musician
hears as "this chart is wrong" and must NOT be diluted by a large pile of
easy-correct plain majors. Count-weighted averaging (the v1-style pathology)
would let 34/41 correct majors bury 0/4 dom7. v2 does not.

### COMPOSITE
`0.6 * weighted_mean(root, placement, flavor) + 0.4 * weakest_axis`.
Weights `root 0.40 / placement 0.35 / flavor 0.25`. The **weakest-axis term
is the explicit anti-bag-of-chords safeguard**: a chart cannot earn a high
composite by acing one axis while another is broken — the exact pathology
that gave "In My Life" 0.919.

---

## 3. The anti-goalpost design rule

> **The scorer must agree with informed-musician judgment. If it scores a
> chart HIGH that a musician calls bad, the SCORER is wrong, not the
> musician.**

This file is validated *against that bar*, not tuned to flatter our
detector. Concretely: the v1→v2 changes were chosen because v1 disagreed with
the musician on "In My Life" (called it 0.919 / good; musician calls it bad),
and every v2 design choice (order-aware not bag; placement = best_offset not
hold_invariant; flavor = weakest-blend not count-mean; composite =
weakest-blend) makes the number move *toward* the musician verdict on
principle — none was reverse-fit to a target score.

---

## 4. Validation evidence

Run: `./venv311/bin/python audit/validate_scorer_v2.py`

### 4a. Mechanics floor (sanity)
GT-vs-GT (a perfect chart) scores **1.0 on every axis** for both songs and
all 8 diagnostic fixtures. A scorer that cannot give a perfect chart 1.0 is
broken; v2 passes.

### 4b. "In My Life" — v1 vs v2 (the whole point)

| Metric | v1 | v2 |
|---|---|---|
| root | **0.919** | AXIS1 root = 0.837 |
| root_quality | 0.685 | — |
| AXIS2 strict_bar | — | **0.314** |
| AXIS3 flavor (weighted) | — | **0.249** |
| **COMPOSITE** | (v1 had none; cited 0.919) | **0.428** |

v2 flavor breakdown shows the exact musician complaint: `triad_dom7 = 0.000
(0/4)` — **every A7 and B7 dropped to plain A/B**; plus `G→Gm`, `Dm→D`
flips. Placement strict 0.314: only 16/51 bars land in the right place.
**v2 rates "In My Life" LOW (composite 0.428), in deliberate contrast to v1's
0.919 — matching the human verdict that it is "basically root notes, missing
the Dm/7ths".** This is the pass criterion and it passes.

### 4c. Detector ranking (raw forensic event streams)

Expected from measured raw-signal truth: **oracle > ACE/Jiang > librosa**.
A scorer that ranks librosa top is broken.

| song | stream | root | pl_strict | flavor | composite |
|---|---|---|---|---|---|
| iml | librosa | 0.639 | 0.255 | 0.227 | **0.344** |
| iml | jiang | 0.829 | 0.333 | 0.288 | **0.440** |
| iml | ace | 0.838 | 0.157 | 0.508 | **0.505** |
| iml | oracle | 1.000 | 1.000 | 1.000 | **1.000** |
| pos | librosa | 0.495 | 0.267 | 0.981 | **0.479** |
| pos | jiang | 0.707 | 0.233 | 1.000 | **0.492** |
| pos | ace | 0.707 | 0.200 | 1.000 | **0.533** |
| pos | oracle | 1.000 | 1.000 | 1.000 | **1.000** |

Oracle tops both; **librosa is last on both** (composite). ACE/Jiang beat
librosa. Ranking is sane — the inversion v1 produced is corrected.

(Forensic streams reconstructed from `/tmp/forensic`, verified
byte-identical to prod via md5 during construction. If `/tmp` is wiped the
streams cannot be reproduced without re-fetching — the harness says so
honestly rather than fabricating numbers.)

### 4d. Diagnostic set discrimination

All 8 `DIAGNOSTIC_SET.md` fixtures flatten and score **1.0 GT-vs-GT**
(mechanics sound). Scoring real *detector* output needs each song's audio run
through the pipeline (out of scope: no prod/deploy) — stated honestly, not
faked. To prove the axes *discriminate*, a simulated "strips all extensions"
detector (the exact `DIAGNOSTIC_SET.md` headline-failure hypothesis) was run:

- **Don't Know Why** (extension axis): root **1.000**, strict **1.000**,
  flavor **0.194** (`triad_dom7 0/21`, `triad_maj7_min7 0/21`). v1's `root`
  would report 1.0 and hide the entire failure; v2 isolates it on AXIS 3 —
  exactly the dimension the old 2-song set could never move.
- **Folsom Prison Blues** (dom7 blues axis): root 1.000, flavor **0.300**
  (`triad_dom7 0/5`). Blues character loss surfaces cleanly.

---

## 5. Honest limitations

1. **No absolute GT timestamps.** Placement is sequence-aligned (relative
   order + hold), NOT time-aligned. `strict_bar` assumes 1 GT cell == 1 bar;
   songs whose GT cells span ≠1 bar will read low on `strict_bar` while
   `best_offset`/`hold_invariant` stay valid. *Proposed minimal GT extension
   (not silently assumed):* add an optional per-line `bars_per_cell` int (or
   a `cell_bars` list) to the fixture schema so `strict_bar` can expand GT
   cells to true bar widths. Until then `best_offset` is the composite
   placement representative precisely because it tolerates the anchoring
   ambiguity this limitation creates.
2. **Diagnostic set not scored against real detector output** — needs each
   song's audio through the live pipeline (out of scope here). v2 mechanics
   are validated GT-vs-GT and via simulated-failure injection instead.
3. **Forensic raw streams depend on `/tmp/forensic`.** They were md5-verified
   identical to prod when captured; the served prod charts themselves are now
   pinned in-repo (`audit/fixtures/prod_charts/`) so the headline contrast
   (4b) is fully reproducible without `/tmp`. Only the per-detector ranking
   (4c) needs the raw streams.
4. **Slash/inversion is directional only** (weight 0). This is a deliberate,
   documented choice reflecting the known gated-off slash gap — it is
   reported so regressions are visible but never drags the grade.

---

## 6. Does it pass the "agrees with the musician" bar?

**Yes.** The single concrete musician verdict on record — "In My Life" is
basically root notes, missing the Dm/7ths, the worst chart tested — was
*contradicted* by v1 (0.919) and is *confirmed* by v2 (composite 0.428,
flavor 0.249, every dom7 dropped, 16/51 bars correctly placed). The detector
ranking v1 inverted (librosa on top) is corrected (librosa last). Every
design choice moves the number toward the musician on principle, none was
reverse-fit. The honest caveat: only one fully-specified human verdict exists
to test against; broader confirmation needs a musician to grade more charts
the scorer rates, which is the recommended next validation step.
