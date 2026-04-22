# Tuning Detection Wiring Results

**Date:** 2026-03-16

## Problem

The tuning detector (`tuning_detector.py`) correctly identifies Eb tuning but was only wired into the job pipeline (`processing/transcription.py`), not into `ChordDetector.detect()` in `chord_detector_v10.py`. Any caller using `ChordDetector` directly got un-corrected chord names (Ebm instead of Em).

## Changes Made

### 1. `chord_detector_v10.py`

- Added `tuning_info` field to `ChordProgression` dataclass (default `None`)
- Added `_apply_tuning_correction()` method to `ChordDetector` class
- Restructured `detect()` to run tuning detection on **unconstrained** BTC output (before vocab constraint), then use the corrected result if a tuning offset is found
- Key insight: vocab constraint + tuning detection both try to fix the same problem (shifted chord names), but vocab constraint produces a confusing mix of shifted/unshifted names. Running tuning detection on the raw unconstrained output gives clean, consistent results.
- When tuning offset is detected, vocab-constrained mode is skipped (unnecessary since chords are already transposed to correct names)
- Both BTC and Essentia fallback paths include tuning correction

### 2. `processing/transcription.py`

- Removed duplicate tuning detection code (~30 lines)
- Now reads `progression.tuning_info` directly from the `ChordProgression` object returned by `detect()`
- Job metadata storage preserved (tuning_offset per chord, job.tuning_info)

## Little Wing Test Results

**Audio:** `/Users/jeffkozelski/stemscribe/outputs/a0db058f/stems/htdemucs_6s/Jimi_Hendrix_-_Little_wing_-_HQ/guitar.mp3`

**Tuning detected:** Half-step down (Eb), offset=+1, score=0.96, method=vocabulary

**Detected chords:** Am, Bm, C, D, Em, F, G

**All expected chords present:** Em, G, Am, Bm, C, F, D -- all 7 found.

**No un-corrected flat chords:** Ebm, D#m, F#, Gb, Abm, G#m -- none found.

**Progression sample (first ~60s):**
```
  1.1s  Em    ->  3.4s  G    ->  3.8s  C    ->  4.6s  G
  8.1s  Am    ->  9.4s  C    -> 11.7s  Em   -> 14.1s  D
 15.3s  Bm    -> 20.3s  C    -> 22.2s  G    -> 26.3s  C
 27.4s  Am    -> 28.1s  D    -> 34.9s  Em   -> 38.7s  G
 42.1s  Am    -> 44.0s  C    -> 45.6s  Em   -> 46.7s  G
 50.0s  Bm    -> 52.7s  Am   -> 53.8s  C    -> 55.8s  G
 57.6s  F     -> 59.3s  C    -> 62.0s  D
```

The real Little Wing progression (Em - G - Am - Em - Bm - Bbm - Am - C - G - F - C - D) is well-represented. The Bbm chromatic walk-down was not detected (it's extremely brief), but all major diatonic chords are correct.

## Test Suite

**138/138 tests passing** (no regressions)

## Architecture Notes

Tuning correction now lives in a single place: `ChordDetector._apply_tuning_correction()`. This ensures:
- Any caller of `ChordDetector.detect()` gets corrected chords automatically
- The job pipeline (`transcription.py`) no longer duplicates the logic
- `ChordProgression.tuning_info` carries metadata for downstream consumers
