# Lyrics Pipeline — Implementation Notes

## Overview
`backend/lyrics_extractor.py` extracts timestamped lyrics from isolated vocal stems using faster-whisper. It produces line-level lyrics with start/end times for chord charts, karaoke, and display.

## Architecture

### Function Signature
```python
extract_lyrics(vocals_path: str, output_dir: str = None) -> list[dict]
```

Returns:
```python
[{"text": "When the time comes around", "start_time": 12.5, "end_time": 15.2}, ...]
```

Saves `lyrics.json` in `output_dir` when provided.

### Processing Pipeline
1. **Transcribe** — faster-whisper "base" model, word-level timestamps, VAD filter, English
2. **Split into lines** — uses pause detection (>1s gap), sentence boundaries, max 15 words, max 6s duration
3. **Merge short fragments** — single-word lines merged with neighbors if gap < 1.5s
4. **Deduplicate** — removes consecutive repeated/subset lines (Whisper hallucination)
5. **Clean text** — strips `[Music]`/`[Applause]` artifacts, hallucinations ("Thank you", "Subscribe"), capitalizes, removes trailing periods

### Model Configuration
- Model: `base` (matches `word_timestamps.py`)
- Device: `cpu`, compute_type: `int8`
- Shares model singleton with `word_timestamps.py` when possible
- VAD: min_silence_duration_ms=500, speech_pad_ms=200

## Integration Points
- **Existing Whisper fallback** in `routes/lyrics.py` (`_whisper_fallback`) — uses `word_timestamps.get_word_timestamps()` directly. Could be refactored to use `extract_lyrics()` instead.
- **Chord sheet** in `routes/chord_sheet.py` — uses `word_timestamps.py` for word-level placement.
- **Karaoke mode** — consumes synced lyrics from the lyrics API.

## Test Results — "The Time Comes" (Kozelski)

Tested on: `outputs/4afcef17-.../stems/htdemucs_6s/.../vocals.mp3`

### Accuracy
- **Overall word similarity**: 60% (against ground truth docx)
- **Lines extracted**: 25 (ground truth has 28 unique lines; Whisper also captured the bridge repeat)
- **Chorus**: Accurately transcribed ("Time comes to turn it around", "All the lines must be put down")
- **Verse 1**: Mostly correct structure, some word-level errors ("wanted" -> "want it", "astray" -> "straight")
- **Bridge**: Hardest section — significant misheard words ("Come to life" -> "from the lie high")

### What works well
- Timestamp accuracy is solid — lines start/end at the right times
- Line splitting produces natural-looking lyric lines (5-10 words each)
- Deduplication correctly handles Whisper's tendency to repeat
- Chorus is very well recognized

### Known limitations
- "base" model trades accuracy for speed; "small" or "medium" would improve word accuracy at ~2-4x cost
- Sung vocals (especially with effects/reverb) are harder than speech
- Bridge sections with layered harmonies are challenging
- Some lines get merged that should be separate (e.g., "Is where I found you when you were going straight")

## Future improvements
- Try "small" model for better accuracy on misheard words
- Post-processing with LLM to fix common mishearings
- Cross-reference with LRCLIB when available (use LRCLIB text + Whisper timestamps)
- Language detection for non-English songs
