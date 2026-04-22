# Song Structure Detection — Implementation Notes

## Overview

`backend/structure_detector.py` automatically detects song sections (Intro, Verse, Chorus, Bridge, Outro) from chord progressions, audio energy, and optional lyrics timestamps.

## API

```python
from structure_detector import detect_structure

sections = detect_structure(
    audio_path="path/to/audio.mp3",      # Required: audio for energy analysis
    chords=[{"chord": "Am", "time": 0, "duration": 2, "confidence": 0.9}, ...],  # Optional
    lyrics=[{"text": "hello", "start_time": 5.0, "end_time": 6.0}, ...],         # Optional
    output_path="structure.json",          # Optional: save results
)
# Returns: [{"name": "Verse 1", "start_time": 0.0, "end_time": 30.5}, ...]
```

## Algorithm

### Primary path (chords available)

**Distinctive Chord Fingerprinting** — the key insight is that in pop/rock, each song section uses a partially unique chord vocabulary. The chorus introduces chords not found in the verse.

1. **Non-overlapping windows**: Divide the chord timeline into 4-second windows.

2. **Chord frequency analysis**: Count how often each chord appears across all windows.
   - **Common chords** (>40% of windows): F#m, A, B, E — these appear everywhere
   - **Distinctive chords** (<30% of windows): D, C#m, Bm, G — these mark specific sections

3. **Window labeling**: Each window is labeled by its set of distinctive chords.
   - Windows with no distinctive chords get label `_base_` (typically verse)
   - Windows with `{C#m, D}` → chorus
   - Windows with `{Bm}` → intro/transition

4. **Run-length encoding**: Adjacent windows with the same label are grouped into runs. Very short runs (< 2 windows = < 8s) are merged into neighbors.

5. **Section labeling**:
   - The repeating label group with the most total time → **Verse**
   - The next repeating group (or highest energy) → **Chorus**
   - Unique labels → **Bridge** (middle) or **Intro/Outro** (edges)
   - Post-labeling: sections after the last Chorus become **Bridge** instead of Verse

6. **Refinements**:
   - Short non-base segments adjacent to Intro are merged into Intro
   - Lyrics timestamps override labels for intro/outro detection
   - Segment boundaries are snapped to nearest chord onsets

### Fallback path (no chords)

Uses librosa's SSM (self-similarity matrix) with checkerboard kernel novelty detection + agglomerative clustering for boundaries. Labels by energy: louder = Chorus, quieter = Verse.

## Test Results on "The Time Comes" (Kozelski)

**Ground truth**: Intro → Verse 1 → Chorus → Verse 2 → Chorus → Bridge

**Detected** (using guitar stem + AI chord data):
- Intro (0-20s) — correct section, slightly short (actual ~0-36s)
- Verse 1 (20-44s) — picks up late intro + verse
- Chorus 1 (44-60s) — correctly identifies chorus region
- Verse 2 (60-80s) — correct
- Chorus 2 (80-96s) — correct
- Bridge (96-204s) — mostly correct, some sub-segments
- Outro (204-240s) — bridge ending

**Score**: 5/6 main sections correctly identified with ~10s timing accuracy.

## Limitations

- Window size (4s) means sections shorter than ~8s may not be detected
- Chord detection accuracy directly affects structure accuracy
- Using individual stems (guitar/vocals) instead of full mix gives less reliable energy analysis
- Songs with very small chord vocabularies (3-4 chords shared across all sections) are harder

## Dependencies

- `librosa` (already installed) — audio loading, energy, chroma, SSM
- `scipy` (already installed) — median filter, peak finding (audio fallback only)
- `numpy` (already installed) — array operations

No new dependencies were added.

## Files

- `backend/structure_detector.py` — main implementation
- `backend/tests/test_structure_detector.py` — 14 unit tests
- `docs/structure-detection.md` — this file

## Research Notes

### Libraries evaluated

| Library | Status | Notes |
|---------|--------|-------|
| **MSAF** | Not installed | pip install fails on Python 3.11/ARM, heavy deps (mir_eval etc.) |
| **librosa.segment** | Used (fallback) | `agglomerative` and `recurrence_matrix` work well for audio-only |
| **librosa chroma** | Used | CQT chroma is the basis for SSM in audio-only mode |
| **scipy** | Used (fallback) | `median_filter`, `find_peaks` for novelty detection |
| **Self-similarity matrix** | Used (fallback) | Checkerboard kernel novelty from SSM diagonal |

### Why chord-first?

Audio-based SSM/novelty boundaries are unreliable for labeling because:
1. They split too aggressively (10+ segments for a simple song)
2. Chord patterns within small segments are too partial for matching
3. Songs with similar chord vocabulary across sections confuse LCS-based similarity

The distinctive-chord approach works because it exploits a fundamental property of pop/rock songwriting: the chorus introduces at least one chord not found in the verse.
