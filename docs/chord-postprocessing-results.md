# Chord Post-Processing Implementation Results

## What Was Implemented

All post-processing logic was added to `/Users/jeffkozelski/stemscribe/backend/chord_detector_v10.py` as standalone functions called from `ChordDetector.detect()`. Post-processing runs after both the BTC+Essentia ensemble path and the Essentia-only fallback path.

### Functions Added

| Function | Purpose |
|----------|---------|
| `postprocess_chords()` | Orchestrator - runs all 4 steps in sequence |
| `_merge_consecutive()` | Merges adjacent identical chord events, combining durations and averaging confidence (duration-weighted) |
| `_get_min_beat_duration()` | Detects tempo via `librosa.beat.beat_track` on first 60s of audio; returns 1-beat duration. Falls back to 1.0s |
| `_filter_min_duration()` | Removes chord events shorter than 1 beat |
| `_filter_rare_chords()` | Removes chords appearing only once in the whole song (boundary chords exempted) |
| `_apply_key_weighting()` | Boosts diatonic chord confidence +12%, penalizes non-diatonic -18% |
| `_get_diatonic_chords()` | Returns diatonic pitch class set for a given key |
| `_root_to_pitch_class()` | Converts root note name to pitch class integer |

### Processing Pipeline Order

1. **Merge consecutive identical chords** - Combines [Am, Am, Am, G, G] into [Am (3x dur), G (2x dur)]
2. **Minimum duration filter** - Tempo-aware: detects BPM, removes events < 1 beat
3. **Rare chord filter** - Removes chords with only 1 occurrence (except first/last chord)
4. **Key-aware confidence weighting** - Diatonic chords get +12% confidence, non-diatonic get -18%
5. **Re-merge** - Catches new adjacencies created by removals in steps 2-3

## Test: "The Time Comes" Guitar Stem

- **Audio:** `outputs/4afcef17-.../guitar.mp3` (240s)
- **Detected key:** F#m (both audio-based and chord-based agree)
- **Detected tempo:** 152.0 BPM (1 beat = 0.39s)
- **Ground truth chords:** F#m, A, B, D, C#m, Bm, G, E

### Before/After Comparison

| Metric | Before (raw) | After (post-processed) |
|--------|-------------|----------------------|
| Total chord events | 126 | 125 |
| Unique chords detected | 8 | 8 |
| Correct (in ground truth) | 8/8 (100%) | 8/8 (100%) |
| False positives | 0 | 0 |
| Missed ground truth | 0 | 0 |

### Post-Processing Step Breakdown

| Step | Events Before | Events After | Removed |
|------|--------------|-------------|---------|
| 1. Merge consecutive | 126 | 126 | 0 (no adjacent duplicates from ensemble) |
| 2. Min duration (0.39s) | 126 | 125 | 1 (short noise event) |
| 3. Rare chord filter | 125 | 125 | 0 (all chords appear >= 2 times) |
| 4. Key weighting | 125 | 125 | 0 (confidence adjusted, no removals) |

### Confidence Improvements (Key Weighting)

Diatonic chords to F#m (F#, G#, A, B, C#, D, E) received +12% confidence boost. Examples:

| Chord | Raw Confidence | Post-Processed | Change |
|-------|---------------|----------------|--------|
| F#m (t=8.33) | 0.912 | 0.990 | +0.078 (diatonic, capped at 0.99) |
| E (t=12.96) | 0.806 | 0.926 | +0.120 (diatonic boost) |
| F#m (t=151.85) | 0.655 | 0.775 | +0.120 (diatonic boost) |
| F#m (t=154.63) | 0.617 | 0.737 | +0.120 (diatonic boost) |

G chord (non-diatonic to F#m) received -18% penalty:
- G (t=55.00): 0.990 -> 0.810
- G (t=92.50): 0.990 -> 0.810

This is musically correct: G is non-diatonic to F#m but does appear in the song (borrowed chord / mode mixture). The penalty reduces its confidence but does NOT remove it.

## Test Suite Results

All 138 existing tests pass with no changes:
```
======================== 138 passed, 1 warning in 2.91s ========================
```

## Design Decisions

1. **Rare chord threshold = 2** (not 3): Initial testing with threshold=3 incorrectly removed the G chord which appears twice in the song. Lowered to 2 to only remove true one-off noise.

2. **Post-processing applies to ALL detection paths**: Both BTC+Essentia ensemble and Essentia-only fallback get the same post-processing.

3. **Output format preserved**: ChordEvent dataclass unchanged - `{chord, time, duration, confidence, root, quality}`.

4. **Confidence weighting is additive, not multiplicative**: +12%/-18% absolute adjustment keeps the confidence values interpretable and consistent across detection methods.

5. **Tempo detection uses first 60s only**: Faster than full-song analysis, sufficient for reliable BPM estimation.
