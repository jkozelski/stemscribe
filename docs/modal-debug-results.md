# Modal Cloud GPU Debug Results

**Date:** 2026-03-19
**Status:** FIXED and VERIFIED

## Root Cause

The Modal function call in `separation.py` line 723-724 was:
```python
from modal_separator import separate_stems_gpu
stems_data = separate_stems_gpu.remote(audio_bytes, filename=audio_path.name)
```

This imports the function **definition** from the local file, but Modal functions need to be "hydrated" (connected to Modal's API and resolved to the deployed version) before `.remote()` works. Importing the function object from the local module gives you an un-hydrated function that can't run.

**Error message:**
```
modal.exception.ExecutionError: Function has not been hydrated with the metadata
it needs to run on Modal, because the App it is defined on is not running.
```

## The Fix

Changed to use `modal.Function.from_name()` which looks up the **deployed** function by app name and function name:

```python
separate_fn = modal.Function.from_name("stemscribe-separator", "separate_stems_gpu")
stems_data = separate_fn.remote(audio_bytes, filename=audio_path.name)
```

**File:** `backend/processing/separation.py`, lines 722-727

## Why the Debug Prints Were Invisible

The server stdout/stderr goes to `/private/tmp/stemscribe-server.log`, NOT `/tmp/stemscribe.log`. The debug prints WERE firing — Modal WAS being called — but the error was caught by the fallback handler, which silently fell back to local GPU (RoFormer).

## Verification

1. **Standalone test:** Sent 4MB audio file to Modal → received 6 stems in 68s
2. **Full pipeline test:** Uploaded `Angelina_Scratch.mp3` via API → Modal processed stems on cloud T4 GPU → transcription, chords, Guitar Pro files all generated → uploaded to Google Drive
3. **Job ID:** `3deb6720-1e4d-4523-96d5-983b7f29b375`
4. **All 316 existing tests pass** (2 pre-existing billing test failures unrelated)

## What Was NOT the Problem

- `MODAL_AVAILABLE` was True (modal imported fine)
- `MODAL_ENABLED` env var was loaded correctly
- The Modal app was deployed and working
- The code path was correct (Modal branch was taken)
- No `modal.py` shadowing
- No stale `.pyc` files

The ONLY issue was the function invocation method: `from_name()` vs direct import.
