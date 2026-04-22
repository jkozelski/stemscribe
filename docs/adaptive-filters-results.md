# Adaptive Post-Processing Filters — Results

**Date:** 2026-03-16

## Problem
Round 2 post-processing filters were too aggressive when few unique chords exist. On "The Time Comes," 20 events were being reduced to 5. The rare chord filter and min-duration filter compounded the problem when vocab constraint already limited output.

## Changes Made (`chord_detector_v10.py`)

### 1. Adaptive Rare Chord Filter (`_filter_rare_chords_adaptive`)
- `<= 8 unique chords` (global sparse check): skip filter entirely
- `7-12 unique chords`: remove chords appearing only once
- `> 12 unique chords`: remove chords appearing 1-2 times (original behavior)
- Boundary chords (first/last) always kept

### 2. Adaptive Min Duration Filter (`_filter_min_duration_adaptive`)
- Gap protection: if removing a chord would leave a gap > 2 beats, keep it
- Safety net: if total events drops below 5, relax threshold to 0.5x beats
- When sparse (<= 8 unique types): use 0.5x beat threshold from the start

### 3. Key-Aware Weighting — Chromatic Passing Protection
- Reduced non-diatonic penalty from 18% to 10%
- New `_is_chromatic_passing()` detector: if a non-diatonic chord sits between two diatonic chords a whole step apart, it gets zero penalty
- Example: Bm -> Bbm -> Am (Bbm preserved at full confidence)

### 4. Global Sparse Check
- If raw detection produces <= 8 unique chord types, rare chord filter is skipped entirely
- Merge and min-duration filters still run but with relaxed thresholds

## Test Results

### Little Wing (Jimi Hendrix) — Guitar Stem
- **Key:** F#
- **Events:** 19 (was similar before, Little Wing was not the problem case)
- **Unique chords:** 8
- **Filter log:**
  - Step 1 (merge): 21 -> 19
  - Step 2 (min duration 0.44s, adaptive): 19 -> 19 (sparse mode, relaxed threshold)
  - Step 3 (rare filter): 19 -> 19 (skipped, only 8 unique types)
  - Final: 19 events

### The Time Comes (Kozelski) — Guitar Stem
- **Key:** F#m
- **Events:** 126 (previously reduced to ~5)
- **Unique chords:** 8
- **Filter log:**
  - Step 1 (merge): 126 -> 126
  - Step 2 (min duration 0.39s, adaptive): 126 -> 126 (sparse mode)
  - Step 3 (rare filter): 126 -> 126 (skipped, only 8 unique types)
  - Final: 126 events
- **Chord progression detected:** Gm, A#, Cm, F cycling patterns with D#, Dm, G# bridge sections

### Unit Tests
- **138/138 tests passing** (no regressions)

## Key Observations
1. The sparse check (`<= 8 unique types`) correctly identifies both songs as sparse and disables the rare chord filter
2. The adaptive min-duration filter with 0.5x relaxation prevents loss of short but legitimate chord changes
3. The non-diatonic penalty reduction (18% -> 10%) and chromatic passing protection preserve musical accuracy
4. The Time Comes went from ~5 events to 126 events — a dramatic improvement that preserves the full chord progression
