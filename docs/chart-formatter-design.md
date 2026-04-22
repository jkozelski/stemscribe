# Chart Formatter Design

**Module:** `backend/chart_formatter.py`
**Status:** Built and tested locally
**Date:** 2026-04-04

## Problem

The chord detector (BTC v10) outputs raw timestamped chords:
```json
[{"time": 0.0, "duration": 2.5, "chord": "Am", "root": "A", "quality": "m", "confidence": 0.9}, ...]
```

Whisper outputs raw word timestamps:
```json
[{"word": "Hello", "start": 0.5, "end": 1.2}, {"word": "darkness", "start": 1.3, "end": 1.8}, ...]
```

We need Ultimate-Guitar-style output:
```
[Verse 1]
Am                C
Hello darkness my old friend
```

## Solution

`chart_formatter.py` takes both inputs and produces the chord library JSON format (same schema as `chord_library/kozelski/the-time-comes.json`).

## Pipeline

```
chord_detector_v10.detect_chords(audio)  ──┐
                                           ├──> chart_formatter.format_chart() ──> JSON
word_timestamps.get_word_timestamps(vocal) ┘
```

### Step-by-step:

1. **Split words into lyric lines** -- uses timing gaps between words (>0.4s) and max word count (12) to find natural line breaks
2. **Build raw sections** -- groups lyric lines by timing gaps (>4s = new section), creates Intro/Outro/Instrumental for chord-only regions
3. **Assign chords to lines** -- for vocal sections, places each chord above the word closest in time; for instrumental sections, groups chords into rows of 4
4. **Label section types** -- chord progression fingerprinting (relative intervals) to detect Verse vs Chorus vs Bridge patterns:
   - Most repeated pattern = Verse
   - Second most repeated = Chorus
   - Unique patterns = Bridge
   - Short sections between Verse and Chorus = Pre-Chorus
5. **Number repeated sections** -- Verse 1, Verse 2, etc.
6. **Output** -- chord library JSON format

## Chord-Over-Word Alignment

The key insight: chords are placed above lyrics by matching chord timestamps to word timestamps. For each chord event, we find the word whose `start` time is closest to the chord's `time`, then position the chord string at that word's character offset in the line.

```python
# Chord at time=5.5 -> word "friend" starts at 5.2 -> char position 22
# Result: "Am                    G"
#         "Hello darkness my old friend"
```

## Section Detection Algorithm

Uses chord progression fingerprinting rather than absolute chord names:
- Convert chord roots to pitch classes (C=0, C#=1, ..., B=11)
- Compute intervals between consecutive chords: `(next_root - prev_root) % 12`
- Sections with the same interval sequence = same type

This means transposed songs still match (Am-G-F-C has the same fingerprint as Dm-C-Bb-F).

## Output Format

Matches the existing chord library schema:
```json
{
  "title": "The Sound of Silence",
  "artist": "Simon & Garfunkel",
  "key": "Am",
  "capo": 0,
  "source": "StemScriber AI",
  "chords_used": ["Am", "G", "F", "C"],
  "sections": [
    {
      "name": "Verse 1",
      "lines": [
        {"chords": "Am                    G", "lyrics": "Hello darkness my old friend"},
        {"chords": "G                          Am", "lyrics": "I've come to talk with you again"}
      ]
    }
  ]
}
```

## Integration Points

- **Input from:** `chord_detector_v10.detect_chords()` and `word_timestamps.get_word_timestamps()`
- **Output to:** chord library JSON (same format as `routes/chord_sheet.py` serves to frontend)
- **Can replace:** the `_align_chords_to_lyrics()` function in `routes/chord_sheet.py` which currently does simpler alignment

## API

```python
from chart_formatter import format_chart, render_plain_text

chart = format_chart(
    chord_events=chords,        # from chord_detector_v10
    word_timestamps=words,      # from word_timestamps
    title="My Song",
    artist="My Band",
    key="Am",
)

# Get plain text for preview
text = render_plain_text(chart)

# Or save as JSON
import json
with open("output.json", "w") as f:
    json.dump(chart, f, indent=2)
```

## Tuning Parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `LINE_GAP_THRESHOLD` | 0.4s | Min gap between words to split lines |
| `SECTION_GAP_THRESHOLD` | 4.0s | Min gap between lines to start new section |
| `MAX_WORDS_PER_LINE` | 12 | Hard limit on words per line |
| `MIN_WORDS_PER_LINE` | 2 | Min words before allowing a line split |
| `CHORD_SNAP_TOLERANCE` | 0.8s | How close a chord must be to snap to a word |

## Test

Run the built-in test with sample "Sound of Silence" data:
```bash
cd ~/stemscribe && ./venv311/bin/python backend/chart_formatter.py
```

## Next Steps

- Wire into the processing pipeline (after chord detection + Whisper, before saving to DB)
- Replace `_align_chords_to_lyrics()` in `routes/chord_sheet.py` with `format_chart()`
- Add beat detection (librosa) to improve line splitting at measure boundaries
- Train section detection on chord library examples for better Verse/Chorus classification
