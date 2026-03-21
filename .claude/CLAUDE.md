# StemScribe — Agent Context

## What Is This?
Audio-to-Guitar-Pro transcription app. Upload a song → stem separation → MIDI transcription → Guitar Pro (.gp5) file.

## Stack
- **Backend:** Python/Flask at `backend/app.py` (modularized into blueprints)
- **Frontend:** `frontend/index.html` (mixer UI) + `frontend/practice.html` (tab viewer)
- **Python:** 3.11, venv at `venv311/`
- **Server:** `http://localhost:5555`
- **GPU:** Apple M3 Max with MPS acceleration

## Architecture
- **Separation:** RoFormer (default) or Demucs htdemucs_6s (fallback) → 6 stems
- **Transcription:** Drums/Piano = custom CRNN models, Guitar/Bass = Basic Pitch + post-processing
- **Async:** Background threading with semaphore queue, DemucsRunner subprocess
- **Output:** MIDI → Guitar Pro (.gp5) via `midi_to_gp.py`

## Key Modules (post-refactor)
- `app.py` — Flask factory (~106 lines)
- `dependencies.py` — All imports, feature flags
- `routes/api.py` — Core API endpoints
- `routes/health.py` — Health check
- `auth/` — JWT auth (Flask-JWT-Extended)
- `billing/` — Stripe integration
- `storage/` — Cloudflare R2 integration
- `middleware/rate_limit.py` — Flask-Limiter + plan enforcement
- `processing/separation.py` — All stem separation modes
- `processing/transcription.py` — MIDI transcription
- `models/job.py` — ProcessingJob class

## Model Files (don't delete these)
- `backend/models/pretrained/best_drum_model.pt` (114MB)
- `backend/models/pretrained/best_piano_model.pt` (145MB)

## Tests
```bash
cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/ -v
```
138 tests, all passing.

## Deploy (not yet live — configs ready)
- Dockerfile + docker-compose.yml + railway.toml
- `.env.example` has all required vars

## Rules
- Don't break existing API interfaces (see memory/stemscribe.md for contracts)
- One agent at a time on app.py to prevent edit conflicts
- Auth/billing/storage are standalone blueprints — register via app factory
- Run tests after any backend changes

## Full Reference
- `~/.claude/projects/-Users-jeffkozelski/memory/stemscribe.md`
- `~/KOZELSKI_AGENT_TEAMS_BRIEFING.md`
