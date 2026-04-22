# Modal Cloud GPU Deployment — Stem Separation

## Overview

StemScriber can offload stem separation to Modal's cloud GPUs (T4) instead of
running on the local M3 Max. This is useful for production where you want
consistent GPU availability without tying up the local machine.

## Architecture

```
User uploads audio
    |
    v
Flask backend (local)
    |
    +--> MODAL_ENABLED=true?
    |       YES --> Send audio bytes to Modal function (T4 GPU)
    |                  |
    |                  v
    |               htdemucs_6s runs on cloud GPU
    |                  |
    |                  v
    |               Returns 6 MP3 stems as bytes
    |                  |
    |                  v
    |               Save stems locally, continue pipeline
    |
    |       NO --> Run locally on M3 Max (existing behavior)
    |
    v
Transcription, chords, tabs (always local)
```

## Files

| File | Purpose |
|------|---------|
| `backend/modal_separator.py` | Modal app definition — GPU function + image + volume |
| `backend/processing/separation.py` | `separate_stems_modal()` — calls Modal, falls back to local |
| `backend/processing/pipeline.py` | Routes to Modal when `MODAL_ENABLED=true` |
| `deploy_modal.sh` | One-command deploy script |
| `.env` | `MODAL_ENABLED=false` (flip to `true` for production) |

## Setup

### Prerequisites

```bash
# Modal is already installed (v1.3.5) and authenticated
pip install modal
modal token new  # if not already authenticated
```

### Deploy

```bash
# From project root
./deploy_modal.sh

# Or manually
cd ~/stemscribe/backend
../venv311/bin/python -m modal deploy modal_separator.py
```

### Enable

Set in `.env`:
```
MODAL_ENABLED=true
```

Then restart the backend.

### Test

```bash
# Test with Modal's built-in runner (uses a file from uploads/)
cd ~/stemscribe/backend
../venv311/bin/python -m modal run modal_separator.py

# Or with a specific file
../venv311/bin/python -m modal run modal_separator.py -- /path/to/song.mp3
```

## How It Works

1. **Image**: Debian Slim with Python 3.11, PyTorch 2.1.2 (CUDA 12.1), Demucs 4.0.1, ffmpeg
2. **GPU**: NVIDIA T4 (sufficient for htdemucs_6s, cheapest option)
3. **Volume**: `stemscribe-model-cache` — persists model weights (~1 GB) across cold starts
4. **Timeout**: 600s per function call (songs take 2-3 min typically)
5. **Memory**: 8 GB RAM allocated
6. **Output**: Stems returned as MP3 bytes (320kbps) — not WAV, to minimize transfer size

## Cold Start

- First run: ~45-60s (downloads htdemucs_6s weights into Volume)
- Subsequent cold starts: ~15-25s (weights loaded from cached Volume)
- Warm container: <1s

## Fallback Behavior

If Modal fails for any reason (network, timeout, quota exceeded), the system
automatically falls back to local RoFormer+Demucs separation on the M3 Max.
No user intervention needed.

## Cost

- T4 GPU: ~$0.59/hr on Modal
- Average song: ~2-3 min = ~$0.02-0.03 per song
- Cold start compute is billed but minimal

## Modes

Modal is used as the **default** separation mode when `MODAL_ENABLED=true`.
Users can still force specific modes:

- `ensemble_mode=true` → uses local ensemble (bypasses Modal)
- `mdx_model=true` → uses local MDX hybrid (bypasses Modal)
- Default with Modal off → local RoFormer+Demucs (existing behavior)

## Troubleshooting

```bash
# Check Modal deployment status
modal app list

# View logs for a specific run
modal app logs stemscribe-separator

# Re-deploy after code changes
./deploy_modal.sh

# Test connectivity
python -c "import modal; print(modal.__version__)"
```
