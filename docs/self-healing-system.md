# StemScribe Self-Healing & Learning System

## Overview

Four independent subsystems that make StemScribe learn from errors and user corrections:

1. **User Correction Feedback Loop** — captures chord/lyrics edits as training data
2. **Processing Failure Recovery** — watchdog detects stalled jobs and auto-retries
3. **Chord Accuracy Scoring** — compares AI chords against Songsterr ground truth
4. **Error Pattern Detection** — logs failures with context and finds patterns

Each feature works independently. If one fails, the others continue functioning.

---

## Feature 1: User Correction Feedback Loop

**Files:** `backend/routes/feedback.py`, `backend/feedback_data.json`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/feedback/chord-correction` | Save a chord correction |
| POST | `/api/feedback/lyrics-correction` | Save a lyrics correction |
| GET | `/api/feedback/corrections` | List all corrections |

### Chord Correction Payload
```json
{
  "job_id": "abc123",
  "original_chord": "Am",
  "corrected_chord": "Am7",
  "position": 12.5,
  "context": {"before": "G", "after": "F"}
}
```

### Lyrics Correction Payload
```json
{
  "job_id": "abc123",
  "original_line": "Is this the real life",
  "corrected_line": "Is this the real life?",
  "line_index": 0
}
```

### Query Filters
- `GET /api/feedback/corrections?type=chord` — chord corrections only
- `GET /api/feedback/corrections?type=lyrics` — lyrics corrections only
- `GET /api/feedback/corrections?job_id=abc123` — filter by job

### Storage
Corrections are stored in `backend/feedback_data.json` (append-only, thread-safe). Each entry includes timestamp, job_id, song title, original value, corrected value, and position.

---

## Feature 2: Processing Failure Recovery (Watchdog)

**Files:** `backend/processing/watchdog.py`, `backend/watchdog_log.json`

### How It Works
- Background daemon thread starts with the server
- Checks all active jobs every 60 seconds
- If a job hasn't progressed in 5 minutes, it's marked as **stalled**
- Stalled jobs are automatically retried (max 2 retries)
- After 2 failed retries, the job is marked as permanently failed
- All events are logged to `watchdog_log.json`

### Configuration (constants in watchdog.py)
| Constant | Default | Description |
|----------|---------|-------------|
| `STALL_THRESHOLD_SECONDS` | 300 | Time without progress before stall detection |
| `MAX_RETRIES` | 2 | Maximum retry attempts |
| `CHECK_INTERVAL_SECONDS` | 60 | How often the watchdog checks |

### Safety
- Only catches genuinely stalled jobs (no progress for 5+ minutes)
- Tracks retry count per job to prevent infinite loops
- Each retry resets the job to pending and re-runs the full pipeline
- The `watchdog_retry` metadata flag lets the pipeline know this is a retry

---

## Feature 3: Chord Accuracy Scoring

**Files:** `backend/chord_accuracy.py`, `backend/routes/accuracy.py`, `backend/accuracy_scores.json`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/accuracy/<job_id>` | Accuracy report for a specific job |
| GET | `/api/accuracy` | Summary of all scored jobs |

### Scoring Logic
- Compares AI chord names against Songsterr reference chords
- 1-second time tolerance for position matching
- Normalizes chord names (Cmaj=C, Amin=Am, etc.)
- Tracks: correct, wrong, missed (in reference but not AI), extra (in AI but not reference)

### Example Response
```json
{
  "job_id": "abc123",
  "accuracy_percent": 78.5,
  "total_reference_chords": 42,
  "correct": 33,
  "wrong": 5,
  "missed": 4,
  "extra": 2,
  "details": {
    "wrong": [{"time": 12.5, "ai_chord": "Em", "reference_chord": "Am"}]
  }
}
```

### Ground Truth
Place Songsterr chord data at `outputs/<job_id>/songsterr_chords.json` as a JSON array of `{"chord": "Am", "time": 5.0}` objects. The endpoint will auto-compute and cache the score.

---

## Feature 4: Error Pattern Detection

**Files:** `backend/error_tracker.py`, `backend/routes/accuracy.py`, `backend/error_log.json`

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/errors/patterns` | Pattern analysis across all logged errors |
| GET | `/api/errors/recent?limit=50` | Most recent errors |

### What Gets Logged
Every processing failure is automatically logged with:
- job_id, timestamp
- error_type (e.g., `separation_failed`, `download_failed`, `DurationLimitExceeded`)
- error_message
- song_duration (if known)
- source (`youtube`, `upload`, `archive`, `spotify`, etc.)
- processing_stage (where in the pipeline it failed)

### Pattern Analysis Response
```json
{
  "total_errors": 47,
  "by_type": {"separation_failed": 20, "download_failed": 15},
  "by_source": {"youtube": 30, "archive": 12, "upload": 5},
  "by_stage": {"separation": 20, "download": 15},
  "duration_buckets": {"0-3min": 5, "8-12min": 25},
  "duration_insights": ["70% of failures are on songs > 8 minutes"],
  "source_insights": ["archive songs fail 2.4x more than average"],
  "errors_last_7_days": {"2026-03-15": 5, "2026-03-16": 8},
  "top_error_messages": [{"message": "OOM error", "count": 12}]
}
```

### Integration Points
Error tracking is wired into `processing/pipeline.py` at:
- Duration limit exceeded
- Separation failure
- `process_audio()` top-level exception handler
- `process_url()` top-level exception handler

---

## Testing

```bash
cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/test_self_healing.py -v
```

31 tests covering all four features.
