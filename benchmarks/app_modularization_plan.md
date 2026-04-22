# app.py Modularization Plan

## Date: 2026-02-27
## Author: Pipeline Agent
## Task: #12

---

## 1. Current State

`backend/app.py` is **4238 lines** — a monolith containing:
- Flask app creation and config
- 25+ try/except conditional imports
- ProcessingJob class and persistence
- 6 separation functions (2 dead code)
- Transcription pipeline (~500 lines)
- Smart extraction (~250 lines)
- URL resolution (Spotify/Apple Music/YouTube search)
- 30+ route handlers across 7 domains
- Shutdown handler, semaphore, runner tracking

This makes the file hard to test, review, modify safely, and reason about.

---

## 2. Proposed Module Structure

```
backend/
  app.py                    (~120 lines) Flask app factory, main entry point
  dependencies.py           (~200 lines) All try/except imports, feature flags

  models/
    __init__.py
    job.py                  (~220 lines) ProcessingJob class + persistence
    model_manager.py        (existing, unchanged)
    pretrained/             (existing, unchanged)

  processing/
    __init__.py
    pipeline.py             (~280 lines) process_audio, process_url orchestrators
    separation.py           (~550 lines) All separate_stems_* functions (minus dead code)
    transcription.py        (~560 lines) transcribe_to_midi, quantize_midi, chords, musicxml
    smart_extract.py        (~260 lines) smart_separate, _deep_extract_stem, helpers
    utils.py                (~65 lines)  check_stem_has_content, convert_wavs_to_mp3

  routes/
    __init__.py
    api.py                  (~350 lines) /api/upload, /api/url, /api/status, /api/health, etc.
    library.py              (~80 lines)  /api/library CRUD
    stems.py                (~110 lines) /api/split-stem, /api/analyze-stereo, /api/split-vocals
    tabs.py                 (~260 lines) /api/find-tabs, /api/download-pro-tabs
    archive.py              (~300 lines) /api/archive/* endpoints
    drive.py                (~65 lines)  /api/drive/* endpoints

  services/
    __init__.py
    url_resolver.py         (~110 lines) is_supported_url, Spotify/Apple Music/YT search
    downloader.py           (~90 lines)  download_from_url
```

### What stays in app.py (thin shell):
```python
"""StemScriber - Audio Stem Separation & Transcription API"""
from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024

    # Register error handlers
    from routes import register_error_handlers
    register_error_handlers(app)

    # Register blueprints
    from routes.api import api_bp
    from routes.library import library_bp
    from routes.stems import stems_bp
    from routes.tabs import tabs_bp
    from routes.archive import archive_bp
    from routes.drive import drive_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(library_bp)
    app.register_blueprint(stems_bp)
    app.register_blueprint(tabs_bp)
    app.register_blueprint(archive_bp)
    app.register_blueprint(drive_bp)

    # Load saved jobs
    from models.job import load_all_jobs_from_disk
    loaded = load_all_jobs_from_disk()

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5555)), debug=False)
```

---

## 3. Shared State / Globals

The biggest challenge is shared mutable state. Current globals in app.py:

| Global | Type | Used By | Migration Strategy |
|--------|------|---------|-------------------|
| `jobs` | dict | Routes, pipeline, persistence | Move to `models/job.py`, import everywhere |
| `_separation_semaphore` | Semaphore(1) | 2 separation functions | Move to `processing/separation.py` |
| `_active_runners` | list | 2 separation functions, shutdown | Move to `processing/separation.py` |
| `_active_runners_lock` | Lock | separation, shutdown | Move to `processing/separation.py` |
| `UPLOAD_DIR` / `OUTPUT_DIR` | Path | Routes, pipeline, persistence | Move to `config.py` or `dependencies.py` |
| `FRONTEND_DIR` | Path | Static file serving | Move to `app.py` |
| 25+ `*_AVAILABLE` flags | bool | Routes, pipeline | Move to `dependencies.py` |
| `_gpu_manager` | GPUManager | Ensemble separation | Move to `dependencies.py` |
| `_chord_theory_engine` | ChordTheoryEngine | Chord routes | Move to `dependencies.py` |

### Strategy: Module-level singletons
Each module owns its state. Other modules import what they need:

```python
# In processing/separation.py
_separation_semaphore = threading.Semaphore(1)
_active_runners = []
_active_runners_lock = threading.Lock()

# In models/job.py
jobs = {}

# In dependencies.py
ENHANCED_SEPARATOR_AVAILABLE = False
# ... all feature flags
```

---

## 4. Execution Order (Phased)

### Phase 1: Dead Code Removal (prerequisite, ~10 min)
- Delete `separate_stems_hq_vocals()` (lines 1327-1430)
- Delete `separate_stems_vocal_focus()` (lines 1431-1535)
- Remove `hq_vocals` and `vocal_focus` params from process_audio, process_url
- Remove parsing from upload/url/archive endpoints
- Remove from health endpoint's separation_modes
- **Net: -209 lines of dead separation code, -30 lines of param threading**

### Phase 2: Extract models/job.py (~15 min)
- Move ProcessingJob class (lines 718-772)
- Move save_job_to_disk, save_job_checkpoint, load_job_from_disk, load_all_jobs_from_disk (lines 445-610)
- Move get_job helper (lines 562-610)
- Move `jobs` dict
- Update all imports

### Phase 3: Extract dependencies.py (~15 min)
- Move all 25+ try/except import blocks (lines 146-413)
- Move all `*_AVAILABLE` flags
- Move `_gpu_manager`, `_chord_theory_engine` singletons
- Move `UPLOAD_DIR`, `OUTPUT_DIR`, `SCRIPT_DIR`
- Update all imports

### Phase 4: Extract processing/ modules (~30 min)
- `processing/utils.py`: check_stem_has_content, convert_wavs_to_mp3
- `processing/separation.py`: All separate_stems_* functions + semaphore + runners
- `processing/transcription.py`: transcribe_to_midi, quantize_midi, detect_chords_for_job, convert_midi_to_musicxml
- `processing/smart_extract.py`: smart_separate, _deep_extract_stem, _measure_stem_energy, _smart_rename_stem
- `processing/pipeline.py`: process_audio, process_url, apply_skills_to_job

### Phase 5: Extract services/ (~10 min)
- `services/url_resolver.py`: is_supported_url, is_streaming_url, get_spotify_track_info, get_apple_music_track_info, search_youtube_for_song
- `services/downloader.py`: download_from_url

### Phase 6: Extract routes/ as Blueprints (~30 min)
- Convert @app.route to Blueprint routes
- `routes/api.py`: upload, url, health, status, capabilities, download, cleanup, jobs
- `routes/library.py`: library GET, DELETE
- `routes/stems.py`: split-stem, analyze-stereo, split-vocals
- `routes/tabs.py`: find-tabs, download-pro-tabs, download-pro-tab-file
- `routes/archive.py`: archive/search, archive/collections, archive/show, archive/process, archive/batch
- `routes/drive.py`: drive/auth, drive/stats, drive/upload

### Phase 7: Slim down app.py (~10 min)
- Replace monolith with app factory pattern
- Register blueprints
- Keep static file serving (or move to its own blueprint)
- Keep error handlers (or extract to routes/__init__.py)

---

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| Circular imports | High | Strict dependency direction: dependencies -> models -> processing -> services -> routes -> app |
| Broken import paths | High | Run `python -c "from app import create_app; create_app()"` after each phase |
| Shared state bugs | Medium | Module-level singletons, no mutation from unexpected modules |
| Test breakage | Medium | Run existing tests after each phase |
| Blueprint URL prefix conflicts | Low | Use empty prefix, same URLs as before |
| Merge conflicts with other agents | High | **Confirm no one else is editing app.py before starting** |

---

## 6. Dependency Direction (No Circular Imports)

```
dependencies.py  (feature flags, shared imports)
       |
       v
  models/job.py  (ProcessingJob, persistence, jobs dict)
       |
       v
  processing/*   (separation, transcription, pipeline)
       |
       v
  services/*     (url_resolver, downloader)
       |
       v
  routes/*       (Flask blueprints, HTTP layer only)
       |
       v
  app.py         (app factory, blueprint registration)
```

No module imports from a module below it in this diagram.

---

## 7. What NOT To Change

- **External module files** (enhanced_separator.py, demucs_runner.py, etc.) — don't touch
- **separation/ package** (ensemble.py, gpu_manager.py, etc.) — don't touch
- **Frontend** — no changes needed
- **Tests** — update imports only, don't change test logic
- **API contract** — all URLs, request/response formats stay identical

---

## 8. Estimated Impact

| Metric | Before | After |
|--------|--------|-------|
| app.py lines | 4238 | ~120 |
| Total backend modules | 1 monolith | 13 focused modules |
| Dead code | 209 lines | 0 |
| Testable units | 1 (app.py) | 13 (each module independently) |

---

*Plan compiled: 2026-02-27*
