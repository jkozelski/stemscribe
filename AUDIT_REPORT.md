# StemScribe Comprehensive Audit Report

**Date:** 2026-03-04
**Audited by:** Claude Opus 4.6 (StemScribe Audit Team)
**Scope:** Full codebase — backend, frontend, tests, deployment, security

---

## Executive Summary

StemScribe is an ambitious audio-to-tablature pipeline with a solid core architecture. The stem separation and mixer UI work well. However, the project has accumulated significant technical debt from rapid prototyping: ~54K lines of backend Python, of which roughly 30% is dead/unused code; major features (auth, billing, storage) are scaffolded but not integrated into the frontend; transcription accuracy is poor by the project owner's own assessment; and several security issues would need resolution before any production deployment.

**Overall assessment: Not production-ready.** Estimated effort to launch: 3-4 focused weeks of cleanup + testing + deployment hardening.

---

## 1. Architecture Review

### What's Well-Structured

- **Flask blueprint organization** — Clean separation into 12 blueprints (`routes/api.py`, `routes/library.py`, `routes/songsterr.py`, etc.). Each has a focused responsibility.
- **Feature flag pattern** — `dependencies.py` uses try/except imports with boolean flags. This means the app gracefully degrades when optional modules are missing. Smart for a project with many optional ML dependencies.
- **Job persistence model** — `models/job.py` is clean: jobs serialize to JSON on disk, reload on startup, with checkpoint saves during processing. Good crash resilience.
- **GPU semaphore** — `processing/separation.py` uses a threading semaphore to ensure only one GPU separation runs at a time. Proper resource management.
- **Graceful shutdown** — `atexit` + `SIGINT`/`SIGTERM` handlers cancel active DemucsRunner processes. Prevents zombie subprocesses.
- **SSRF protection** — `services/url_resolver.py` has a blocklist for private/metadata IPs. Good baseline security.
- **Path traversal protection** — `_safe_path()` resolves paths and checks prefix. `secure_filename()` for uploads.

### What Needs Refactoring

- **Massive duplication in conditional imports.** The same try/except import blocks for chord_detector, drum_transcriber, guitar models, etc. appear THREE times: once in `dependencies.py`, once in `processing/transcription.py`, and once in `processing/pipeline.py`. This is a maintenance nightmare — change one and you must update all three. **Fix: Single source of truth in `dependencies.py`; other modules import from there.**

- **`dependencies.py` is 315 lines of try/except blocks.** It imports ~30 modules, each with its own flag. This file is itself a code smell. Consider a registry pattern or plugin loader.

- **`processing/transcription.py` is 838 lines of deeply nested if/elif/else fallback chains.** The transcription routing logic (neural drum -> OaF -> v2 spectral -> v1 -> Basic Pitch) is a 600-line function with 5+ levels of nesting. This should be refactored into a strategy/chain-of-responsibility pattern.

- **In-memory job registry (`jobs = {}`)** — Acknowledged in the known issues. This is fine for single-process development, but breaks with Gunicorn workers (Dockerfile specifies `--workers 2`). The Procfile also uses `--workers 2`. Either reduce to 1 worker or move job state to Redis/DB.

- **Monolithic frontend** — `practice.html` is 2,468 lines of HTML with all CSS and JS inline. `index.html` is 702 lines. The `frontend/js/` modules are good, but practice.html is a single-file monolith that's impossible to maintain.

- **No build system** — Frontend is raw HTML/CSS/JS served directly. No bundling, no minification, no tree-shaking. Fine for dev, but for production you'd want at least basic asset optimization.

---

## 2. Feature Completeness

### Working End-to-End

| Feature | Status | Notes |
|---------|--------|-------|
| File upload (MP3/WAV/FLAC/M4A/OGG) | WORKING | Up to 500MB, validation, secure_filename |
| URL download (YouTube/SoundCloud/Bandcamp) | WORKING | via yt-dlp with Chrome cookies |
| Spotify/Apple Music URL resolution | WORKING | Extracts track info, searches YouTube |
| Stem separation (RoFormer+Demucs) | WORKING | 6 stems, MPS acceleration, CPU fallback |
| Audio mixer (play/pause/seek/mute/solo) | WORKING | Web Audio API, VU meters, 6-stem sync |
| MIDI transcription (4 instrument types) | WORKING | But accuracy is questionable (see below) |
| Guitar Pro export (.gp5) | WORKING | via pyguitarpro |
| MusicXML notation export | WORKING | via music21 |
| Practice mode (AlphaTab rendering) | WORKING | Loads GP5, renders tabs |
| Songsterr tab search | WORKING | Search + GP5 conversion |
| Library (save/load/delete) | WORKING | Persists to disk across restarts |
| Archive.org live music browser | WORKING | Search, browse shows, process tracks |
| Keyboard shortcuts | WORKING | Space=play, arrows=seek |
| Light/dark theme toggle | WORKING | CSS variables |

### Half-Built / Scaffolded But Not Live

| Feature | Status | Notes |
|---------|--------|-------|
| JWT Authentication | SCAFFOLDED | Full auth blueprint with register/login/refresh/reset, but frontend has ZERO auth UI. No login page, no token storage, no protected routes. |
| Stripe Billing (3 tiers) | SCAFFOLDED | Plans defined ($0/$4.99/$14.99), webhook handlers written, checkout route exists. But no frontend pricing page, no checkout flow, no user dashboard. |
| Cloudflare R2 Storage | SCAFFOLDED | Full S3-compatible client with presigned URLs. But not connected to any user flow. |
| Rate Limiting | SCAFFOLDED | Flask-Limiter initialized, plan-based enforcement written. But meaningless without auth (everyone is "anonymous"). |
| Google Drive upload | SCAFFOLDED | `drive_service.py` exists with OAuth. But `credentials.json` and `token.json` are in the repo root (security issue). |
| Chord chart UI | MISSING | Backend API exists (`/api/chords/`, `/api/theory/`). No frontend rendering whatsoever. |
| Waveform display | DISABLED | WaveSurfer conflicts with Web Audio mixer. Currently loads the library but doesn't use it. |
| A-B loop (click-to-set) | PARTIAL | UI buttons exist, loop logic exists, but needs waveform for click-to-set-point. |
| Ensemble separation mode | UNTESTED | Code exists but references `EnsembleSeparator` from the `separation/` package which may not be importable without extra setup. |

### Broken / Known Bad

| Feature | Status | Notes |
|---------|--------|-------|
| Transcription accuracy | POOR | Owner reports "notes are all wrong" in practice mode. Guitar v3 NN model is a confirmed dead end (0.36 F1 vs Basic Pitch 0.79 F1). |
| Chord detection quality | POOR | v8 detector found only 2 chords in Little Wing (should find 20+). |
| Vocal split quality | UNKNOWN | Auto-splits vocals into lead/backing via BS-RoFormer, but no quality validation. |
| Guitar lead/rhythm split | UNKNOWN | Uses MelBand-RoFormer, but `GUITAR_SEPARATOR_AVAILABLE` depends on a trained model that may not exist. |

---

## 3. Code Quality

### Bugs

1. **`_validate_job_id` regex has escaped backslash** (`/Users/jeffkozelski/stemscribe/backend/routes/api.py`, line 33):
   ```python
   re.match(r'^[a-f0-9\\-]+$', job_id)
   ```
   The `\\-` matches a literal backslash OR hyphen. Should be `r'^[a-f0-9-]+$'`. This means IDs containing backslashes would pass validation.

2. **Dockerfile health check hits wrong endpoint** (`/Users/jeffkozelski/stemscribe/Dockerfile`, line 23):
   ```
   CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-5555}/health')"
   ```
   Should be `/api/health`. The `/health` route doesn't exist. Also: Railway's `healthcheckPath` in `railway.toml` is set to `/health` (line 8), not `/api/health`.

3. **`railway.toml` health check path mismatch** — `healthcheckPath = "/health"` but the actual endpoint is `/api/health`. Railway would mark the service as unhealthy.

4. **Pipeline references `job.output_dir`** (`/Users/jeffkozelski/stemscribe/backend/processing/pipeline.py`, line 291):
   ```python
   output_dir = Path(job.output_dir)
   ```
   `ProcessingJob` has no `output_dir` attribute. This would crash if Songsterr auto-fetch ever triggers. Dead code path in practice since it requires title metadata.

5. **No-cache headers on ALL responses** (`/Users/jeffkozelski/stemscribe/backend/app.py`, line 39-44):
   ```python
   @app.after_request
   def add_no_cache_headers(response):
       response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
   ```
   This applies to stem MP3 downloads too, forcing re-download every time. Performance killer for 320kbps audio files. Should only apply to API responses, not static/download content.

6. **`CORS(app)` allows all origins** (`/Users/jeffkozelski/stemscribe/backend/app.py`, line 29):
   Not a bug for local dev, but a security hole for production.

### Security Issues

1. **`credentials.json` and `token.json` in repo root** — These are Google OAuth credentials. While gitignored, they exist on disk in the project. If ever committed (even temporarily), they'd be in git history permanently. **Move to a secure location outside the repo, or use environment variables.**

2. **CORS allows all origins** — Must be restricted before production deployment.

3. **No Content-Security-Policy headers** — Frontend loads scripts from CDNs (unpkg.com, cdn.jsdelivr.net, cdnjs.cloudflare.com). Should add CSP headers.

4. **`--cookies-from-browser chrome`** in yt-dlp commands (`/Users/jeffkozelski/stemscribe/backend/services/downloader.py`, lines 25, 53) — This reads Chrome cookies from the host machine. Useful for bypassing YouTube bot detection locally, but: (a) won't work in Docker/Railway, (b) is a security risk if the server is ever shared, (c) violates YouTube ToS.

5. **No input sanitization on Songsterr search queries** — The query is URL-encoded but passed directly to an external API. While `urllib.parse.quote` handles basic encoding, it's worth noting there's no length limit on the search query.

6. **`docker-compose.yml` exposes database with hardcoded password** (`localdev`) — Fine for local dev, just don't deploy this compose file.

### Dead Code (Candidates for Removal)

| File | Lines | Purpose | Status |
|------|-------|---------|--------|
| `integration_guide.py` | 427 | Documentation as Python file | DEAD — never imported |
| `prepare_guitar_data.py` | 723 | Training data preparation | DEAD — one-time script |
| `training_data_pipeline.py` | 599 | More training data tools | DEAD — one-time script |
| `train_linear_probe.py` | 809 | Model training | DEAD — one-time script |
| `regenerate_chords.py` | 180 | Batch re-processing utility | DEAD — manual script |
| `regenerate_gp.py` | 103 | Batch GP regeneration | DEAD — manual script |
| `monitor_runpod.py` | ? | RunPod monitoring | DEAD — monitoring script, wrong directory |
| `band_database_extended.py` | 580 | Extended band metadata | PARTIALLY DEAD — imported by band_config |
| `chord_analysis.py` | 406 | Chord analysis utilities | UNKNOWN — may not be imported |
| `chord_detector.py` | ? | v1 basic chord detector | FALLBACK — only used if v7 and v8 both fail |
| `chord_detector_v7.py` | ? | v7 chord detector | FALLBACK — only used if v8 fails |
| `songsterr.py` (root) | ? | Old Songsterr API client | POSSIBLY DEAD — `routes/songsterr.py` is the active one |
| `models/pretrained/guitar_v3_training.log` | 10MB | Training log | DEAD — archival only |
| `models/pretrained/train_guitar_runpod.py` | ? | Training script | DEAD — one-time RunPod script |
| `models/pretrained/best_guitar_v3_model.pt` | 71MB | Dead end model | DEAD — confirmed by A/B test |
| `models/pretrained/best_tab_model.pt` | 57MB | Old model | DEAD — "not loaded" per memory |
| `models/pretrained/best_bass_model.pt` | 113MB | Old model | DEAD — "not loaded" per memory |

**Estimated removable dead code: ~3,400 lines of Python + ~241MB of unused model weights.**

### Missing Error Handling

- `download_from_url()` doesn't handle the case where yt-dlp succeeds but no audio file is found (falls off the end of the function with no return value — returns `None`, which the caller doesn't check).
- `process_audio()` catches the top-level exception but individual separation mode failures set `job.error` without proper cleanup of partial state.
- `load_job_from_disk()` doesn't handle corrupt JSON gracefully (will crash on malformed `job_metadata.json`).
- Frontend JS has minimal error handling — network failures during processing status polling would silently fail.

---

## 4. Test Coverage Gaps

### What's Tested (138 tests, all passing)

| Test File | Tests | Coverage Area |
|-----------|-------|---------------|
| `test_api.py` | ~20 | Health, upload validation, URL validation, download path traversal, status, SSRF blocking |
| `test_billing.py` | ~15 | Plan definitions, price ID lookups, webhook event handlers |
| `test_job.py` | ~10 | URL pattern matching, numpy type conversion, streaming URL detection |
| `test_midi_to_gp.py` | ~25 | Note-to-fret mapping, chord voicing, tuning systems, GP file generation |
| `test_rate_limit.py` | ~25 | Limiter initialization, plan enforcement, duration limits, usage recording |
| `test_demucs_runner.py` | ~15 | DemucsRunner interface (mocked) |

### What's NOT Tested

- **Actual audio processing pipeline** — No integration tests with real audio. All processing is mocked. This is understandable (ML models are heavy), but means pipeline bugs only show up at runtime.
- **Songsterr integration** — No tests for `routes/songsterr.py` or `songsterr_to_gp.py`. These are new files (2026-03-03) that haven't been committed yet.
- **Practice mode** — No tests. The entire 2,468-line practice.html is untested.
- **Frontend JS** — Zero frontend tests. No Jest, Playwright, or Cypress. 10,559 lines of JS with no test coverage.
- **Auth flow** — The auth blueprint has no HTTP-level tests (register, login, refresh, token validation).
- **Storage module** — R2 upload/download not tested.
- **Library operations** — Only one test (`test_library_returns_list`). No tests for delete, edge cases.
- **Job persistence** — No tests for save/load/checkpoint cycle. The `load_job_from_disk` path with corrupt files is untested.
- **Chord detection** — No tests for chord_detector_v8 accuracy or output format.
- **Smart extraction** — `processing/smart_extract.py` has no tests.
- **Downloader** — `services/downloader.py` has no tests (hard to test without mocking yt-dlp).

### Test Infrastructure Issues

- `conftest.py` mocks 30+ modules with `MagicMock`. While this enables testing, it means tests can pass even when actual imports would fail. If someone renames a function in `chord_detector_v8.py`, tests would still pass because the mock returns `MagicMock()` for any attribute access.
- No test for CI itself — the `ci.yml` installs full `requirements.txt` which includes ~2GB of ML libraries. This would be extremely slow and expensive in CI. The tests themselves mock everything, so they don't need these libraries.

---

## 5. Frontend UX Issues

### index.html (Main Mixer Page)

1. **Tour Bus progress animation is creative but distracting** — 200+ lines of SVG for city skylines (SF, Denver, Chicago, NYC) with King Kong and a biplane. Fun, but takes up significant screen real estate during the ~10-15 minute processing wait. Consider making it collapsible.

2. **WaveSurfer loaded but unused** — `index.html` loads WaveSurfer.js and its Regions plugin from CDN (2 HTTP requests), but waveforms are disabled. Remove these script tags until waveforms work.

3. **No loading indicator for Songsterr search in practice mode** — Search sends a request but there's no spinner or "searching..." feedback.

4. **Song card shows "Tap info for track info" even for uploaded files** — The info panel tries to fetch track info for local files, which may fail silently.

5. **Settings panel has "Google Drive" section** but Drive integration is not user-facing. This creates confusion.

6. **No responsive testing** — `responsive.css` exists but practice.html has no responsive styles. Mobile users would get a broken experience.

7. **Emoji overload in UI** — Every button and label has emoji. While visually distinctive, it reduces readability and feels unprofessional for a paid SaaS product.

### practice.html (Practice Mode)

1. **2,468 lines in a single HTML file** — All CSS (hundreds of lines), JS (hundreds more), and HTML in one file. This is unmaintainable and slow to load.

2. **No way to get back to mixer** — The only "back" navigation is clicking the logo, which links to `/`. Should have an explicit "Back to Mixer" button.

3. **Tab rendering depends on AlphaTab CDN** — If the CDN is down, practice mode is completely broken. Should have a fallback or at least an error message.

4. **No error state for failed tab loading** — If the GP5 file is corrupt or missing, the user sees a blank page.

5. **Speed slider goes to 25% minimum** — Useful for practice, but the UI doesn't show the current speed prominently enough.

6. **Songsterr search results don't show instrument tracks available** — The API returns track data (guitar, bass, drums) but the search results UI only shows song title and artist.

---

## 6. Deployment Readiness

### Blockers (Must Fix Before Launch)

1. **Health check endpoint mismatch** — Dockerfile uses `/health`, Railway uses `/health`, but the actual endpoint is `/api/health`. Deployment would immediately fail health checks.

2. **In-memory job dict with 2 Gunicorn workers** — Workers don't share memory. Job created in worker A would be invisible to worker B. Either use 1 worker (reduces throughput) or move to Redis/DB.

3. **CORS wildcard** — Must restrict to production domain.

4. **No `.env` for production** — `.env.example` exists, but no documentation on which env vars are actually required vs optional for a minimal deploy.

5. **`--cookies-from-browser chrome`** in yt-dlp — This won't work in a Docker container. Needs a fallback strategy (or just remove it and accept that some YouTube videos may be bot-blocked).

6. **No SSL/TLS consideration** — Flask serves HTTP. Railway handles TLS termination, but `JWT_COOKIE_SECURE` isn't set, `SESSION_COOKIE_SECURE` isn't set.

7. **500MB upload limit** — Combined with WAV conversion, a single job could use 1-2GB of disk. Railway's disk is ephemeral and limited. Need R2 storage integration or disk management.

8. **Model files (502MB) not in repo** — They're gitignored. Deployment needs a strategy to fetch models (download script, Docker build layer, R2, etc.).

### Warnings (Should Fix)

9. **No-cache on everything** — Kills performance for repeat visits and file downloads.

10. **requirements.txt has no version pins** — `flask`, `demucs`, `torch` etc. are all unpinned. A `pip install` today may give different versions than next week. Use `pip freeze > requirements-lock.txt`.

11. **CI installs full ML stack** — The `ci.yml` runs `pip install -r requirements.txt` which includes torch, demucs, etc. Tests mock everything, so CI could use a `requirements-test.txt` with just flask, pytest, numpy.

12. **Docker image size** — `python:3.11-slim` base + torch + demucs + all ML libs = likely 5-8GB image. Consider multi-stage build.

13. **No database migration runner** — `docker-compose.yml` references `backend/migrations/001_initial_schema.sql` for auto-init, but there's no migration runner for production (just raw SQL files).

14. **Replicate integration** — Listed in requirements but `replicate` doesn't appear to be used anywhere in the codebase. Dead dependency.

---

## 7. Priority Recommendations (Ranked)

### P0 — Must Fix Before Any Deployment

1. **Fix health check endpoints** — Change Dockerfile and railway.toml to use `/api/health`. (5 minutes)

2. **Fix `_validate_job_id` regex** — Change `\\-` to just `-`. (1 minute)

3. **Restrict CORS** — Set allowed origins to production domain. (5 minutes)

4. **Fix Gunicorn worker / in-memory job conflict** — Either set `--workers 1` or move job state to disk-based locking. (30 minutes for workers=1, days for Redis)

5. **Remove `--cookies-from-browser chrome` from yt-dlp** — Won't work in Docker. (5 minutes)

6. **Pin requirements.txt versions** — Run `pip freeze` and lock versions. (15 minutes)

### P1 — Should Fix Soon

7. **Remove dead code** — Delete unused training scripts, dead model files (guitar_v3, old tab/bass models), integration_guide.py. Saves ~241MB of model weight and ~3,400 lines of Python. (1 hour)

8. **Fix no-cache headers** — Only apply to API JSON responses, not downloads. (15 minutes)

9. **Fix `job.output_dir` AttributeError** in pipeline Songsterr auto-fetch. (5 minutes)

10. **Consolidate conditional imports** — Single source of truth in `dependencies.py`, remove duplicates from `transcription.py` and `pipeline.py`. (2 hours)

11. **Create `requirements-ci.txt`** — Minimal deps for running tests without ML libraries. Speed up CI from ~10 min to ~30 sec. (30 minutes)

12. **Move `credentials.json`/`token.json` out of repo** — Even though gitignored, they shouldn't be in the project directory. (10 minutes)

### P2 — Should Fix Before Launch

13. **Investigate transcription accuracy** — This is the core value prop. If "notes are all wrong," the product can't launch. Re-process a test song with current models, A/B against Songsterr tabs, document what's working. (1-2 days)

14. **Investigate chord detection** — v8 finding only 2 chords in Little Wing is a showstopper for the chord feature. Either fix or disable/hide chord UI. (1 day)

15. **Build chord chart UI** — Backend API is ready. Frontend is missing. This is a key differentiator from competitors. (2-3 days)

16. **Re-enable waveform display** — Either share AudioContext between WaveSurfer and mixer, or use peaks-only rendering. The mixer without waveforms feels incomplete. (1-2 days)

17. **Refactor practice.html** — Extract CSS/JS into separate files. Add responsive styles. (1 day)

18. **Add frontend error handling** — Loading states, error toasts, retry logic for failed API calls. (1 day)

19. **Model deployment strategy** — Build a script that downloads models from R2 or a CDN at deploy time. Add to Dockerfile. (4 hours)

20. **Remove WaveSurfer script tags** from index.html until waveforms are re-enabled. (2 minutes)

### P3 — Nice to Have

21. **Add frontend tests** — At minimum, Playwright tests for the happy path (upload, process, view results, open practice mode). (2-3 days)

22. **Add integration tests with real audio** — Even one 10-second test file exercising the full pipeline would catch regressions. (1 day)

23. **Activate auth + billing** — Build login/register UI, Stripe checkout flow, usage tracking. This is the monetization path. (1-2 weeks)

24. **Refactor `transcribe_to_midi()`** — Break the 838-line function into a strategy pattern with pluggable transcribers. (1 day)

25. **Add CSP headers** — Enumerate trusted CDN sources, add Content-Security-Policy. (2 hours)

26. **Docker multi-stage build** — Reduce image size from ~8GB to ~3GB. (4 hours)

---

## Appendix: File Inventory

### Backend (53,776 lines Python across ~80 files)

- **Core**: `app.py` (151), `dependencies.py` (315), `config.py` (164)
- **Processing**: `pipeline.py` (368), `separation.py` (627), `transcription.py` (838), `smart_extract.py` (~200)
- **Models**: `job.py` (245), `model_manager.py`
- **Routes**: 10 blueprint files (~1,500 total)
- **Auth/Billing/Storage**: ~1,500 lines across 8 files (all scaffolded, not live)
- **Transcribers**: 8 transcriber modules (~4,000 total)
- **Tests**: 1,730 lines across 6 test files (138 tests)
- **Dead/utility scripts**: ~3,400 lines

### Frontend (10,559 lines across 23 files)

- `index.html` (702), `practice.html` (2,468)
- `js/` 12 modules (~5,000 total)
- `css/` 8 stylesheets (~2,400 total)

### Deployment

- Dockerfile, docker-compose.yml, railway.toml, Procfile, requirements.txt, .env.example, ci.yml
- 1 SQL migration file

---

*End of audit report.*
