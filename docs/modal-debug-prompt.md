# Modal Debug Team Prompt

```
Fix the Modal cloud GPU integration for StemScriber. Modal is deployed, authenticated, and the code is wired in — but the processing thread never uses Modal. It always falls back to local GPU.

## System
- Backend: ~/stemscribe/backend/
- Modal app: stemscribe-separator (deployed, live on Modal)
- Python 3.11, venv at ~/stemscribe/venv311/
- Modal 1.3.5 installed and authenticated

## The Problem
- MODAL_AVAILABLE is True when tested standalone
- MODAL_ENABLED is True in .env
- Pipeline code at processing/pipeline.py line 174 checks: if MODAL_AVAILABLE and not ensemble_mode and not mdx_model
- But the Modal branch NEVER executes — no log output, no cloud GPU usage
- Songs always process on local M3 GPU (RoFormer)
- A print() with flush=True at line 174 doesn't appear in /tmp/stemscribe.log

## What We've Tried
- Cleared all __pycache__
- Set MODAL_ENABLED=true via export, inline env, and .env file
- Changed is_modal_enabled() to runtime check
- Hardcoded the check to skip env var entirely
- Verified modal imports fine in standalone Python

## Suspected Causes
1. The server's nohup process isn't getting the env var
2. MODAL_AVAILABLE is False at import time in the server process (modal import fails silently)
3. The processing thread has a different module context
4. There's a DIFFERENT code path being used for separation that bypasses pipeline.py
5. Python bytecache is being regenerated from a stale source

## Files
- ~/stemscribe/backend/processing/pipeline.py — lines 170-191, separation mode selection
- ~/stemscribe/backend/processing/separation.py — MODAL_AVAILABLE, separate_stems_modal()
- ~/stemscribe/backend/modal_separator.py — the Modal function definition
- ~/stemscribe/backend/app.py — server startup, dotenv loading

## Task
1. Add VERBOSE logging to trace exactly which code path executes during separation
2. Figure out why MODAL_AVAILABLE or the Modal branch isn't reached
3. Fix it so Modal actually gets used
4. Test with a real song processing through Modal cloud GPU
5. Verify stems come back correctly
```
