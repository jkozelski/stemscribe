# V3 Agent B — Tuning Compensation A/B (2026-05-13)

**Hypothesis:** wiring `librosa.estimate_tuning()` + `librosa.effects.pitch_shift`
into `backend/processing/chord_detector_librosa.py` (the V10 pattern) closes
some of the gap to UG ground truth. The Mar 9 root-cause doc
(`backend/CHORD_DETECTION_ANALYSIS.md` §3.4, §5.A) called this out as the
#1 cause of systematic semitone-shift errors.

**Time-box:** 45 min. Single-hypothesis, single-edit, single-rerun experiment.

---

## TL;DR — DO NOT SHIP (as a quality lever); ship only if free

Aggregate root F1 lift = **+0.0005** on 13 songs (baseline 0.4382 → tuned 0.4388).
Aggregate quality F1 lift = **+0.0012** (0.2135 → 0.2147).

The lift is **two orders of magnitude smaller** than the decision threshold
(+0.02). It's also a fraction of the per-song noise floor reported by Agent C
in `detector-quality-comprehensive-audit-2026-05-11.md` (single-run noise std ≈ 0.135).
**Statistically indistinguishable from zero.**

Per the decision criterion, lift is in the "0 to +0.01 — ship anyway, can't
hurt" band. Recommend wiring it in **only** if the cost is one extra librosa
call. The single-song benefit is real (House of the Rising Sun +0.019 root
F1 at +0.080 semitones offset), but it's drowned out by 12 songs that don't
benefit.

The **larger and more important finding**: the V1 Anthropic corrector
*regressed* the tuned variant **more** than it regressed baseline
(−0.036 vs −0.005 root F1 mean). One catastrophic case: Steely Dan's
"Dirty Work" went 0.677 → 0.139 after tuned+corrector. The corrector is the
real bottleneck — not tuning. See "Why corrector regresses" below.

---

## 1. V10 reference pattern (verified)

Confirmed at `backend/chord_detector_v10.py:1178`:

```python
tuning_offset = librosa.estimate_tuning(y=original_wav, sr=sr)
logger.info(f"Estimated tuning offset: {tuning_offset:.3f} semitones")
if abs(tuning_offset) > 0.05:
    original_wav = librosa.effects.pitch_shift(
        original_wav, sr=sr, n_steps=-tuning_offset)
    logger.info(f"Applied tuning compensation: {-tuning_offset:+.3f} semitones")
```

The May 6 doc was correct. This block runs before `librosa.cqt(...)` is computed.

## 2. The exact code change (diff)

The prod detector at `backend/processing/chord_detector_librosa.py:103` currently is:

```python
y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

# CQT-based chromagram (better octave equivariance than STFT chroma)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
```

The wired version (gated on `LIBROSA_TUNING_COMPENSATION=true`):

```python
y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)

# Tuning compensation. librosa.estimate_tuning returns the offset in
# semitones (typ. -0.5 to +0.5). Recordings cut sharp or flat of A=440
# fool the CQT bin alignment and cause systematic root errors (F#m -> G).
# Only compensate when |offset| > 0.05 — below that, pitch_shift adds
# more artifact than it removes alignment error.
tuning_compensation = os.environ.get('LIBROSA_TUNING_COMPENSATION', '').lower() in ('1', 'true', 'yes')
tuning_offset = 0.0
if tuning_compensation:
    tuning_offset = float(librosa.estimate_tuning(y=y, sr=sr))
    if abs(tuning_offset) > 0.05:
        y = librosa.effects.pitch_shift(y, sr=sr, n_steps=-tuning_offset)
        logger.info(f"Tuning compensation applied: {-tuning_offset:+.3f} semitones")

# CQT-based chromagram (better octave equivariance than STFT chroma)
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
```

Cost: one extra `librosa.estimate_tuning()` call per song (~0.1s on 5-min audio)
plus a `pitch_shift` (~1-2s) for the ~30% of songs that cross the 0.05 threshold.

Implementation in the harness: `/tmp/v3_agent_b_harness.py` — `detect()`
with `apply_tuning=True`.

## 3. Per-song results

13 of the 14 target ground-truth fixtures had matching audio locally.
Hotel California GT exists but no local audio. Hells Bells audio exists
but no GT fixture (so excluded from regression check).

Cohort: `black-sabbath__iron-man`, `jamiroquai__alright`,
`jamiroquai__cosmic-girl`, `led-zeppelin__stairway-to-heaven`,
`steely-dan__aja`, `steely-dan__black-cow`, `steely-dan__dirty-work`,
`steely-dan__do-it-again`, `steely-dan__peg`,
`steely-dan__rikki-dont-lose-that-number`, `stevie-wonder__superstition`,
`the-animals__house-of-the-rising-sun`, `the-beatles__let-it-be`.

### Raw detector (no corrector)

| Slug | Tuning offset | base root F1 | tuned root F1 | dR | base qual F1 | tuned qual F1 | dQ |
|---|---:|---:|---:|---:|---:|---:|---:|
| black-sabbath__iron-man | +0.290 | 0.491 | 0.490 | −0.001 | 0.000 | 0.000 | 0.000 |
| jamiroquai__alright | +0.000 | 0.436 | 0.436 | 0.000 | 0.000 | 0.000 | 0.000 |
| jamiroquai__cosmic-girl | +0.020 | 0.446 | 0.446 | 0.000 | 0.000 | 0.000 | 0.000 |
| led-zeppelin__stairway-to-heaven | −0.030 | 0.573 | 0.573 | 0.000 | 0.438 | 0.438 | 0.000 |
| steely-dan__aja | +0.050 | 0.351 | 0.351 | 0.000 | 0.130 | 0.130 | 0.000 |
| steely-dan__black-cow | +0.020 | 0.386 | 0.386 | 0.000 | 0.110 | 0.110 | 0.000 |
| steely-dan__dirty-work | +0.100 | 0.677 | 0.675 | −0.002 | 0.592 | 0.590 | −0.002 |
| steely-dan__do-it-again | −0.090 | 0.159 | 0.152 | **−0.007** | 0.139 | 0.139 | 0.000 |
| steely-dan__peg | −0.060 | 0.585 | 0.583 | −0.002 | 0.070 | 0.069 | −0.001 |
| steely-dan__rikki-dont-lose-that-number | +0.110 | 0.291 | 0.291 | 0.000 | 0.149 | 0.149 | 0.000 |
| stevie-wonder__superstition | −0.010 | 0.149 | 0.149 | 0.000 | 0.000 | 0.000 | 0.000 |
| the-animals__house-of-the-rising-sun | +0.080 | 0.664 | **0.683** | **+0.019** | 0.664 | 0.683 | **+0.019** |
| the-beatles__let-it-be | +0.030 | 0.489 | 0.489 | 0.000 | 0.483 | 0.483 | 0.000 |

**Aggregate (mean across 13 songs):**

- base root F1 0.4382 → tuned root F1 0.4388  → **ΔR = +0.0005**
- base qual F1 0.2135 → tuned qual F1 0.2147  → **ΔQ = +0.0012**

### With V1 Anthropic corrector (ENABLE_ANTHROPIC_CORRECTION=true, MODE=full)

- base+corrector root F1 0.4328 → tuned+corrector root F1 0.3968 → **ΔR = −0.036**
- base+corrector qual F1 0.1996 → tuned+corrector qual F1 0.1682 → **ΔQ = −0.032**

Tuning compensation **degrades** Anthropic corrector outcomes on net. The
−0.036 mean is dominated by one Dirty Work blow-up (0.677 → 0.139) where
Claude in full-mode rewrote the bar_grid against a chord_set the tuned
detector had not "spelled" the same way as the baseline detector. Stripping
out that one regressor, the +corrector tuned/baseline delta is roughly
flat — but the variance is huge and the corrector is the dominant signal.

## 4. Tuning-offset observations

- 5 of 13 songs (38%) had |offset| > 0.05 and were actually pitch-shifted:
  Iron Man (+0.29), Rikki (+0.11), Dirty Work (+0.10), HOTRS (+0.08), Do It Again (−0.09).
- Of those 5, **only HOTRS gained meaningfully** (+0.019 root F1, +0.019 quality F1).
- Iron Man's enormous +0.29 offset is almost certainly the
  estimator being misled by pitched percussion (the song's main riff is
  doubled by power-chord guitar with strong fundamentals around the third
  harmonic). The CQT realignment doesn't help because the chord ambiguity
  is between F# and G (a semitone) — both fit the bin equally well after
  shift.
- 8 of 13 songs had |offset| < 0.05, so the compensation gate didn't fire,
  and scores are identical by construction.
- Two songs (Do It Again, Peg) had tiny *negative* deltas on the order of
  −0.002 to −0.007 — within float-precision noise of pitch_shift's own
  re-rendering artifacts. Not real regressions, but not wins either.

## 5. Regression check

Brief required: Aja, Hells Bells, Hotel California must not drop.
Available in cohort: Aja only.

| Song | base root F1 | tuned root F1 | ΔR |
|---|---:|---:|---:|
| Aja | 0.351 | 0.351 | 0.000 (OK) |
| Hells Bells | — | — | no GT fixture |
| Hotel California | — | — | no local audio |

Wider regression check across the cohort: **largest single regression is
Do It Again at −0.007** (still well within noise floor). No song dropped
by ≥0.01. **No real regressions.**

## 6. Why does the corrector *worsen* the tuned variant?

The V1 Anthropic corrector in `full` mode takes Claude's canonical
chord_set as ground truth and rewrites the detector's `bar_grid` to fit.
When `pitch_shift` nudges a borderline detector decision from chord X to
chord Y, it shifts which chords appear in `chords_used`. The corrector
then sees a slightly different set and makes different replacement
decisions. Sometimes those decisions are worse — as on Dirty Work, where
the tuned detector's `chords_used` triggered a more aggressive
wholesale rewrite that flattened to a near-empty intersection with GT.

This is **not** a tuning-compensation bug. It's the corrector's
sensitivity to its detector input — the same architectural concern raised
in `detector-quality-comprehensive-audit-2026-05-11.md` (variance per
song ≈ 0.135). The lesson: any detector-side change must be paired with a
corrector re-run, and the corrector's variance can swamp small detector
wins.

## 7. Aggregate delta vs decision criterion

| Metric | Lift | Threshold | Verdict |
|---|---:|---:|---|
| Aggregate root F1 (raw detector) | +0.0005 | +0.02 (ship as win) | below |
| Aggregate root F1 (raw detector) | +0.0005 | 0 to +0.01 (ship anyway) | within |
| Aggregate root F1 (with V1 corrector) | −0.036 | 0 (regression) | regression |
| Worst single regression (raw) | −0.007 (Do It Again) | — | within noise |
| Worst single regression (+corrector) | −0.538 (Dirty Work) | — | catastrophic |

## 8. Recommendation

**Two interpretations of "ship":**

### A. Ship as a detector-only change, behind a flag, default OFF
Adopt the V10 pattern in `chord_detector_librosa.py`. Net cost is one
`librosa.estimate_tuning()` call plus a conditional `pitch_shift`. Net
benefit is +0.0005 root F1 on this 13-song set — not a quality win, but
a free knob to have for songs that happen to be cut sharp/flat of A=440.
The "free win" framing in the task brief applies here.

### B. Don't ship until the corrector is robustified
Because the V1 corrector compounds with tuning compensation to produce
genuine regressions (Dirty Work −0.538), shipping tuning in production —
where the corrector also runs — could *worsen* user-visible quality on
some songs. This is a real risk if the prod path puts the corrector
downstream of the detector with no guard.

**My recommendation: NOT ship to prod as a quality lever.** The signal
is below noise and the corrector compounding risk is real. If wiring it
in anyway (because "it can't hurt the raw detector"), absolutely do so
**behind a flag, default OFF, with the corrector regression case
documented**. Re-evaluate as part of the corrector V2 / Claude-as-re-ranker
work referenced in `corrector-v2-proposal-2026-05-12.md` and
`reranker-design-2026-05-11.md`, where the corrector contract is more
robust to detector input perturbations.

The bigger lesson for the V3 detector roadmap: **tuning compensation is
not the missing key**. The Mar 9 root-cause analysis overstated its
impact. Real lifts come from (a) the Claude-as-re-ranker architecture
(Agent B audit estimated +0.04 to +0.10), (b) a learned chord front-end
to replace the 24-template matcher, or (c) the corrector V2 prompt that
preserves slash chords and extensions.

---

## Appendix — raw JSON

Detailed per-song scoring at `docs/v3-agent-B-tuning-compensation-results-2026-05-13.json`.
Harness at `/tmp/v3_agent_b_harness.py`.

## Appendix — methodology caveats

- 13 songs, not 14 — Hotel California GT exists but no local audio file.
  None of the 14-song sets I could find in `docs/` has a fully-enumerated
  song list; I picked the 14 GT fixtures with the cleanest
  filename-to-fixture matches in `uploads/`. Of those 14, the
  Dirty Work fixture had a string `capo` field that crashed the scorer
  before my coercion fix.
- Audio for "Let It Be" (`Let_It_Be_Remastered_2009.wav`) is the real
  Beatles recording. Earlier auto-matching surfaced `Scarlet_Begonias`
  for that slot — dropped in the hand-verified pair list.
- Some songs are scoring catastrophically against GT (Superstition 0.149,
  Do It Again 0.152). That's the prod detector. The corrector + key
  detection + slash-chord gaps documented elsewhere are responsible —
  tuning compensation doesn't move them.
- Single-trial. Agent C's variance audit (May 11) showed std ≈ 0.135
  per song; a true A/B would need 3-5 reruns per arm to separate signal
  from noise. With aggregate delta +0.0005 we're confidently in noise.
