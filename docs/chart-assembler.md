# Chart Assembler

Module: `backend/chart_assembler.py`
Tests: `backend/tests/test_chart_assembler.py` (19 tests)

## Purpose
Combines detected chords, lyrics, and song structure into a formatted chord chart JSON that the frontend's `renderManualChordChart()` can render directly.

## API

### `assemble_chart(chords, lyrics=None, sections=None, title="", artist="") -> dict`
- **chords**: `[{chord, time, duration, confidence}]` from chord detector
- **lyrics**: `[{text, start_time, end_time}]` from lyrics extractor (optional)
- **sections**: `[{name, start_time, end_time}]` from structure detector (optional)
- Returns dict matching `chord_chart.json` format

### `save_chart(chart, output_path) -> None`
Writes chart dict to JSON file, creating directories as needed.

## Output Format
Matches `chord_chart_manual.json` / served by `GET /api/chord-chart/<job_id>`:
```json
{
  "title": "Song Name",
  "artist": "Artist",
  "source": "auto",
  "sections": [
    {
      "name": "Verse 1",
      "lines": [
        {"chords": "F#m          A", "lyrics": "When the time comes around"}
      ]
    }
  ]
}
```

## Key Behaviors
- **Chord alignment**: Chord names placed at character positions proportional to their time offset within each lyric line (Ultimate Guitar style)
- **Instrumental sections**: Detects repeating chord patterns, uses `(x4)` notation
- **Chord carry-forward**: If no chord change occurs on a lyric line, the last active chord is shown
- **Auto-sectioning**: When no sections provided, groups lyrics into 8-line "Verse" sections
- **Graceful degradation**: Works with chords-only, lyrics-only, or any combination

## Integration
The output `chord_chart.json` is served by `routes/api.py` at `GET /api/chord-chart/<job_id>` and rendered by `renderManualChordChart()` in `frontend/practice.html`.
