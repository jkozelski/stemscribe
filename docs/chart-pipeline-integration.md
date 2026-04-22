# Chord Chart Pipeline Integration

## Overview

The auto chord chart generation pipeline runs automatically after chord detection
completes in the main processing pipeline. It combines three modules:

1. **Lyrics Extractor** (`backend/lyrics_extractor.py`) - faster-whisper on vocals stem
2. **Structure Detector** (`backend/structure_detector.py`) - chord + energy + lyrics analysis
3. **Chart Assembler** (`backend/chart_assembler.py`) - combines all into renderable JSON

## Pipeline Flow

```
process_audio()
  -> separate_stems()
  -> detect_chords_for_job()          # populates job.chord_progression
  -> generate_chord_chart_for_job()   # Step 2b in pipeline.py (lines 291-319)
       -> extract_lyrics(vocals_stem)
       -> detect_structure(audio, chords, lyrics)
       -> assemble_chart(chords, lyrics, sections)
       -> save_chart() -> chord_chart.json
  -> transcribe_to_midi()
  -> convert_midi_to_musicxml()
  -> Guitar Pro / Songsterr tabs
```

## Orchestration Function

`generate_chord_chart_for_job()` in `chart_assembler.py` (line 332) handles the full
sub-pipeline. It is called from `pipeline.py` line 304 with:

- `job_id` - for output directory resolution
- `chords` - from `job.chord_progression`
- `vocals_path` - prefers `vocals_lead` over `vocals` stem
- `audio_path` - original audio for structure detection
- `title` / `artist` - from `job.metadata`

## Fault Tolerance

Each step is independently wrapped in try/except:

| Step | Failure behavior |
|------|-----------------|
| Lyrics extraction fails | Chord-only chart (no lyrics) |
| Structure detection fails | Auto-sectioning (every 8 lines = 1 section) |
| Chart assembly fails | Returns None, pipeline continues |
| No chords detected | Skips chart generation entirely |
| Entire Step 2b fails | Logged as warning, pipeline continues to MIDI transcription |

## File Naming & Songsterr Coexistence

- Auto-generated chart: `chord_chart.json`
- If `chord_chart_manual.json` already exists: saves as `chord_chart_auto.json`
- Songsterr tabs use a separate endpoint (`/api/songsterr/chords/<song_id>`) and
  do not write `chord_chart_manual.json`, so there is no conflict.

## API Endpoint

`GET /api/chord-chart/<job_id>` in `routes/api.py`:
- First checks `chord_chart.json`
- Falls back to `chord_chart_auto.json`
- Returns 404 if neither exists
- `PUT` saves/overwrites `chord_chart.json`

## Tests

3 integration tests in `backend/tests/test_chart_assembler.py`:
- `TestGenerateChordChartForJob::test_generates_chart_file`
- `TestGenerateChordChartForJob::test_does_not_overwrite_manual_chart`
- `TestGenerateChordChartForJob::test_returns_none_on_empty_chords`

Run: `cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/test_chart_assembler.py -v`
