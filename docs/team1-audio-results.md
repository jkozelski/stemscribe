# Team 1 — Audio Playback & Processing Stability Results
**Date:** 2026-03-18

---

## Agent 1A — iOS Mobile Audio Fix

### Root Cause
On iOS Safari, `AudioContext` starts **suspended** and can only be resumed inside a user gesture (tap/click). Two problems:

1. **`AudioContext.resume()` was fire-and-forget** — returned a Promise but code didn't await it. `source.start()` was called while context was still suspended → silence.
2. **`init()` called outside user gestures** — in `loadStemAudio()` (practice.html) and `showResults()` (results.js), both called during page load/job completion, not user taps. iOS rejected the resume.

### Files Changed

**`frontend/practice.html`** (StemAudioEngine class):
- Split `init()` into `init()` (user gestures, calls `resume()`) and `_ensureContext()` (for decoding, no resume)
- Made `init()` safe to call multiple times (checks closed/suspended state)
- `play()` now **awaits** `ctx.resume()` before starting buffer sources
- Removed eager `audioEngine.init()` from `loadStemAudio()` — context created by `_ensureContext()` for decoding, then `init()` resumes it in user's play tap
- `togglePlayback()` always calls `init()` in user gesture call stack
- Added console logging throughout fetch/decode/play pipeline for iOS debugging

**`frontend/js/mixer.js`** (shared StemAudioEngine):
- `play()` now **awaits** `ctx.resume()` before starting sources
- Added console logging to init, fetch, decode, and play
- `togglePlayback()` already called `init()` in user gesture — added logging

**`frontend/js/results.js`**:
- Removed `SS.audioEngine.init()` from `showResults()` — was outside user gesture
- Context now created (suspended) by `loadStems()`, resumed by `togglePlayback()`

### Desktop Compatibility
On desktop Chrome/Firefox, AudioContext starts in 'running' state (no user gesture required). New code detects this and calls `startAllSources()` synchronously — no behavior change.

---

## Agent 1B — Processing Stability Fix

### Problem
Jobs stall at random percentages (40%, 59%). Separation process dies silently. Server restarts kill active jobs.

### Fix 1: MPS Memory Flush After Each Separation Pass
**File:** `backend/processing/separation.py`

Added `_flush_mps_memory()` + `gc.collect()` after each GPU separation pass in all four functions:
- `separate_stems()` — after Demucs run completes
- `separate_stems_roformer()` — added flush after Demucs instrument pass (already had flushes between other passes)
- `separate_stems_mdx()` — after Step 1 (MDX23C) and after Step 2 (htdemucs_6s) cleanup
- `separate_stems_ensemble()` — after ensemble separator finishes

Prevents MPS memory accumulation across passes → fixes silent OOM hangs at 40%/59%.

### Fix 2: ffmpeg Subprocess Timeout
**File:** `backend/processing/separation.py`

Added `timeout=300` to ffmpeg subprocess call in `separate_stems_ensemble()`. Other paths already used `convert_wavs_to_mp3()` from `utils.py` which had 180s timeout. Ensemble path was the only one calling ffmpeg directly without timeout.

### Fix 3: Watchdog Retry Fix
**File:** `backend/processing/watchdog.py`

**Root cause:** Stalled thread held `_separation_semaphore`. Retry thread blocked forever trying to acquire it → retries never ran.

**Fix:** Before spawning retry thread, watchdog now:
1. Cancels all active `DemucsRunner` instances (kills stalled subprocess)
2. Force-releases `_separation_semaphore` so retry thread can acquire it
3. Flushes GPU memory, then spawns retry

### Fix 4: Graceful Shutdown
**File:** `backend/app.py`

Replaced immediate `sys.exit(0)` signal handler with `_graceful_shutdown()`:
1. Checks if any separation jobs are active
2. Waits up to 60 seconds for natural completion
3. Force-cancels runners only after deadline
4. Logs outcome (clean completion vs forced cancellation)

Prevents "Shutdown: cancelled all active separation processes" when server restarts.

---

## Test Results
- **315 tests passing**
- 2 pre-existing billing test failures (unrelated)
- No regressions introduced
