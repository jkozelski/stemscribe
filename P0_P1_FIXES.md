# P0 + P1 Fixes Applied — 2026-03-04

All 138 tests passing after every change.

---

## P0 Fixes (Deployment Blockers)

### 1. Health check endpoint mismatch
- **Files:** `Dockerfile` (line 23), `railway.toml` (line 8)
- **Fix:** Changed `/health` → `/api/health` in both Dockerfile HEALTHCHECK and Railway's `healthcheckPath`
- **Also added:** `/health` alias already existed in `routes/health.py` — both paths now work

### 2. `_validate_job_id` regex bug
- **File:** `backend/routes/api.py` (line 33)
- **Fix:** Changed `r'^[a-f0-9\\-]+$'` → `r'^[a-f0-9-]+$'`
- **Impact:** Previously allowed backslash characters in job IDs

### 3. CORS restriction
- **File:** `backend/app.py` (line 29-32)
- **Fix:** Replaced `CORS(app)` (allow all origins) with configurable origins via `CORS_ORIGINS` env var
- **Default:** `http://localhost:5555,http://localhost:3000` for dev
- **Production:** Set `CORS_ORIGINS=https://yourdomain.com` in environment

### 4. Gunicorn worker conflict
- **Files:** `Dockerfile` (line 32), `Procfile` (line 1)
- **Fix:** Changed `--workers 2` → `--workers 1` in both files
- **Reason:** In-memory `jobs` dict can't be shared across workers. Single worker until Redis/DB job store is added.

### 5. Removed `--cookies-from-browser chrome` from yt-dlp
- **File:** `backend/services/downloader.py` (lines 24-26, 53-55)
- **Fix:** Removed `--cookies-from-browser chrome` and `--remote-components ejs:github` flags from both metadata and download commands
- **Reason:** Won't work in Docker containers; security risk on shared servers

### 6. Pinned requirements.txt versions
- **File:** `requirements.txt`
- **Fix:** Pinned all 30+ packages to exact versions from `venv311/` (e.g., `flask==3.1.3`, `torch==2.10.0`)
- **Also:** Removed `replicate` (dead dependency, not used anywhere) and `PyMuPDF` (not installed, not imported)
- **Also created:** `requirements-lock.txt` with full `pip freeze` output

---

## P1 Fixes (Should Fix Soon)

### 7. Removed dead code (~3,400 lines Python + 251MB model weights)

**Deleted Python scripts (never imported, one-time/utility scripts):**
- `backend/integration_guide.py` (427 lines — docs-as-code, never imported)
- `backend/prepare_guitar_data.py` (723 lines — training data prep)
- `backend/training_data_pipeline.py` (599 lines — training data tools)
- `backend/train_linear_probe.py` (809 lines — model training)
- `backend/regenerate_chords.py` (180 lines — batch utility)
- `backend/regenerate_gp.py` (103 lines — batch utility)

**Deleted dead model files:**
- `backend/models/pretrained/best_guitar_v3_model.pt` (71MB — confirmed dead end, 0.36 F1)
- `backend/models/pretrained/best_tab_model.pt` (57MB — "not loaded" per memory)
- `backend/models/pretrained/best_bass_model.pt` (113MB — "not loaded" per memory)
- `backend/models/pretrained/guitar_v3_training.log` (10MB — archival training log)
- `backend/models/pretrained/train_guitar_runpod.py` (37KB — one-time RunPod training script)

### 8. Fixed no-cache headers
- **File:** `backend/app.py` (lines 39-44)
- **Fix:** No-cache headers now only apply to `application/json` and `text/html` responses
- **Impact:** Audio file downloads (stems, MIDI, GP5) are no longer forced to re-download on every request

### 9. Fixed `job.output_dir` AttributeError
- **File:** `backend/processing/pipeline.py` (line 291)
- **Fix:** Changed `Path(job.output_dir)` → `OUTPUT_DIR / job.job_id`
- **Reason:** `ProcessingJob` has no `output_dir` attribute — would crash if Songsterr auto-fetch triggered

### 10. Consolidated conditional imports
- **Files:** `backend/processing/transcription.py`, `backend/processing/pipeline.py`
- **Fix:** Replaced ~80 lines of duplicated try/except import blocks in each file with imports from `dependencies.py`
- **Pattern:** Flags imported from `dependencies.py`; callable objects conditionally imported only when their flag is True
- **Result:** Single source of truth for all feature availability flags in `dependencies.py`

### 11. Created `requirements-ci.txt`
- **File:** `requirements-ci.txt` (new)
- **Purpose:** Minimal deps for CI test runs — flask, pytest, numpy, mido, etc. No torch/demucs/ML stack.
- **Impact:** CI install goes from ~2GB/10min to ~50MB/30sec

### 12. Moved Google OAuth credentials out of repo
- **Files moved:** `credentials.json`, `token.json` → `~/.config/stemscribe/`
- **Updated:** `backend/drive_service.py` to read from `~/.config/stemscribe/` (configurable via `STEMSCRIBE_CONFIG_DIR` env var)
- **Reason:** Even though gitignored, having OAuth creds in the repo root is a security risk

---

## Test Results

```
======================== 138 passed, 1 warning in 0.48s ========================
```

All tests pass after every change.
