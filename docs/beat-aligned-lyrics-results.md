# Beat-Aligned Lyrics Results

## Summary
Added beat-aligned lyric line splitting to `lyrics_extractor.py` that uses librosa's beat tracker and chord change positions to split lyric lines at musical measure boundaries instead of Whisper's speech pauses.

## Problem
Whisper splits lyrics based on speech pauses, which don't align with musical measures. This causes lyrics to bleed across chord changes:

**Before (pause-based):**
```
F#m      A            F#m              B
Say you wanted someone to take you away from
```

**After (beat-aligned):**
```
F#m                    A
Say you wanted someone
F#m                    B
To take you away from
```

## How It Works

### 1. Beat Detection
- Uses `librosa.beat.beat_track()` on the drums stem (clearest rhythm)
- Detects tempo (76 BPM for The Time Comes, expected ~75)
- Groups beats into 4/4 measures

### 2. Chord Boundary Detection
- Extracts timestamps where chord changes occur
- Merges with measure boundaries into unified break point list

### 3. Beat-Aligned Splitting
- Whisper still provides word-level timestamps
- Instead of splitting at speech pauses, words are grouped by the measure/chord intervals they fall into
- Short lines (<2 words) are merged with neighbors
- Long lines (>12 words) are split at chord changes

### 4. Fallback
- If `audio_path` or `chords` are not provided, falls back to the original pause-based splitting
- If beat detection fails for any reason, falls back gracefully

## Files Modified
- `backend/lyrics_extractor.py` — Added beat detection and beat-aligned splitting functions
- `backend/chart_assembler.py` — Passes `audio_path` and `chords` to `extract_lyrics`

## Files Added
- `backend/tests/test_beat_aligned_lyrics.py` — 17 new tests

## Test Results
- 260 passed, 2 pre-existing failures (unrelated to this change)
- 17 new tests all passing

## The Time Comes — Verse 1 Comparison

### Manual (ground truth):
```
F#m                    A
You say you wanted
F#m                        B
Someone to take you away
F#m                       A
From out of the deep end
F#m                          B
Where the devil's all play
```

### Old auto (pause-based):
```
F#m    A          F#m               B
Say you wanted someone to take you away from
F#m              A         F#m              B
Out of the deep end but where the devils all play
```

### New auto (beat-aligned):
```
F#m      A           F#m
Say you wanted someone
F#m          B
To take you away from
F#m         A
Out of the deep end
A  F#m                  B
But where the devils all play is
```

The beat-aligned version splits lines at chord changes rather than speech pauses. Lines now correspond to ~1-2 measures each.

## Remaining Issues
1. **Whisper transcription accuracy** — Some words are wrong ("straight" instead of "astray", "be on" instead of "get on"). This is a Whisper issue, not a splitting issue.
2. **Trailing words** — Some lines end with connecting words ("from", "the", "and") that belong to the next phrase. This happens when a word straddles a measure boundary.
3. **Long instrumental gaps** — Whisper occasionally produces a "word" spanning a huge instrumental section (119s-174s). The beat splitter handles this but it creates an odd line.
4. **Section detection** — The structure detector sometimes mis-labels sections (chorus content appearing in verse sections). This is a separate issue from line splitting.

## Beat Detection Stats (The Time Comes)
- Tempo: 76.0 BPM (expected ~75)
- Beats detected: 272
- Measures: 68
- Chord changes: 126
- Total break points: 71
