# Chord Pattern Analysis

## Overview
`backend/chord_pattern_analyzer.py` detects song structure by finding repeating chord progressions. This is the primary signal for section detection -- more reliable than audio energy features alone.

## Algorithm

### 1. Parse chord events
Takes `[{chord, time, duration, confidence}]` and produces a clean, time-sorted event list.

### 2. Time-window segmentation
- Estimates window size from median chord duration (~6s windows, roughly 2 bars).
- Tries multiple window sizes (1x, 2x, 3x base) and picks the one that produces the best clustering score.
- Each window captures its chord set (unordered) as a fingerprint.

### 3. Cluster windows by Jaccard similarity
- Windows with Jaccard similarity > 0.55 are assigned to the same cluster.
- Greedy single-linkage: each window joins the best matching existing cluster or starts a new one.

### 4. Build sections from cluster runs
- Consecutive windows in the same cluster are merged into a section.
- Very short (1-window) sections with unique patterns are merged into neighbors.

### 5. Label by frequency and position
- Most common pattern (by total window count) = verse (A)
- Second most common = chorus (B)
- Patterns appearing only once = bridge (C, D, etc.)
- First section with a unique pattern = intro
- Last section matching intro pattern = outro

## API

```python
from chord_pattern_analyzer import analyze_chord_patterns

sections = analyze_chord_patterns(
    chords=[{"chord": "Am", "time": 0.0, "duration": 2.0, "confidence": 0.9}, ...],
    audio_path="/path/to/audio.mp3",  # Optional, for librosa beat tracking
    beats_per_bar=4,                   # Default 4/4 time
)
```

### Output format
```python
[
    {
        "pattern": "A",              # Letter label (A=most common)
        "start_time": 0.0,
        "end_time": 25.0,
        "chords": ["F#m", "A", "Bm", "E"],  # Unique chords in section
        "bar_count": 4,              # Window count (roughly bar pairs)
        "section_hint": "verse",     # Suggested: intro/verse/chorus/bridge/outro/section
    },
    ...
]
```

## Validation: "The Time Comes"

Tested against `outputs/4afcef17-c1ed-4bd1-8eb8-e43afe027ba5/ai_chords.json`.

| Detected Pattern | Time Range | Chords | Known Section |
|---|---|---|---|
| B (intro) | 8.3-20.9s | F#m, A, Bm, E | Intro (0-20.3s) |
| A (verse) | 20.9-46.1s | F#m, A, B, D | Verse 1 (20.3-44.3s) |
| D+E (section) | 46.1-58.7s | D, C#m, Bm, G | Chorus 1 (44.3-60.3s) |
| A (verse) | 58.7-83.9s | F#m, A, B, D | Verse 2 (60.3-80.3s) |
| D+E (section) | 83.9-96.5s | D, C#m, Bm, G | Chorus 2 (80.3-96.3s) |
| A (verse) | 96.5-121.7s | F#m, A, E, B, D | Verse 3 + Section |
| C (section) | 121.7-146.8s | D, Bm, F#m, A, E | Section + Chorus 3 |
| A (verse) | 146.8-197.2s | F#m, A, B, E | Bridge (144.3-204.3s) |
| B (outro) | 197.2-222.4s | A, Bm, E, F#m | Outro (204.3-240.1s) |

Key findings:
- Intro, verse, and outro patterns align well with known structure
- Chorus detection works (D+E pattern = D-C#m-Bm / D-C#m-G) but currently labeled as "section" rather than "chorus" since these sub-patterns (D and E) aren't the 2nd most common overall pattern
- Future improvement: merge consecutive related sub-patterns (D+E) and detect chorus as a compound pattern

## Tests
42 tests in `backend/tests/test_chord_pattern_analyzer.py`:
- Unit tests for all internal functions (parse, Jaccard, edit distance, clustering, labeling)
- Integration tests with synthetic data (simple repeating, SAMPLE_CHORDS)
- Real data tests against ai_chords.json with boundary tolerance checks

```bash
cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/test_chord_pattern_analyzer.py -v
```

## Future improvements
1. **Compound chorus detection**: Merge consecutive D+E-style sub-patterns into a single chorus section
2. **Audio-assisted tempo**: Use `audio_path` with librosa beat tracking for more accurate bar quantization
3. **Lyric-assisted labeling**: Integrate with lyrics timestamps to refine intro/verse/chorus labels
4. **Integration with structure_detector.py**: Feed pattern analysis into the existing structure detector as a primary signal
