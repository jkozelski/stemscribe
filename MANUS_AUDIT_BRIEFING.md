# StemScribe - Manus 1.7 Audit Briefing

## Project Overview
**StemScribe** is a self-hosted, AI-powered audio intelligence app that:
1. Separates songs into stems (vocals, drums, bass, guitar, piano, other)
2. Transcribes stems to MIDI/MusicXML
3. Converts to Guitar Pro tabs with playback-synced cursor

## Current Architecture

### Backend (Flask/Python 3.14)
- **app.py** (2436 lines) - Main Flask API server
- **separation/ensemble.py** (296 lines) - Multi-model separation orchestrator
- Multiple support modules: chord_detector, drum_transcriber, midi_to_gp, etc.

### Frontend
- **index.html** (6308 lines) - Main UI with stem mixer, upload, URL input
- **practice.html** (3017 lines) - Tab viewer with AlphaTab/OSMD, moving cursor, chord overlay

### Tech Stack
- Demucs v4 (htdemucs, htdemucs_ft, htdemucs_6s models)
- Apple M3 Max with MPS (Metal) GPU acceleration
- yt-dlp for YouTube downloads
- ffmpeg for audio conversion
- AlphaTab + OpenSheetMusicDisplay for notation rendering

---

## THE PROBLEM 🔴

### Demucs Separation Fails Mid-Process

**Symptoms:**
```
Failed: Demucs failed:
0%| | 0.0/444.59 [00:00<?, ?seconds/s]
1%| | 5.85/444.59 [00:02<02:33, 2.86seconds/s]
3%| | 11.7/444.59 [00:04<02:27, 2.93seconds/s]
...
5%| [then fails]
```

**Root Cause Analysis:**
1. Demucs runs **synchronously** inside HTTP request handler
2. No timeout configured, but browser/Flask times out waiting
3. Progress output (tqdm) goes to stderr, captured as "error"
4. Long songs (7+ minutes) = 2-3 minute processing = timeout

**Code Location (app.py lines ~470-510):**
```python
def separate_stems(job: ProcessingJob, audio_path: Path):
    # Runs demucs via subprocess.run() - BLOCKS until complete
    result = subprocess.run([
        'python3', '-m', 'demucs',
        '--out', str(output_path),
        '-n', 'htdemucs_6s',
        str(audio_path)
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        # THIS IS THE PROBLEM - stderr contains progress bars
        # which look like errors but aren't
        raise Exception(f"Demucs failed: {result.stderr[:500]}")
```

---

## Fixes Needed

### 1. Async Processing (Priority: HIGH)
Convert separation to background job:
```python
@app.route('/api/process', methods=['POST'])
def start_processing():
    job = create_job()
    # Start in background thread/process
    threading.Thread(target=process_async, args=(job,)).start()
    return jsonify({'job_id': job.job_id, 'status': 'processing'})
```

### 2. Fix Error Detection (Priority: HIGH)
Don't treat stderr as error - demucs writes progress there:
```python
# Bad:
if result.returncode != 0:
    raise Exception(f"Demucs failed: {result.stderr}")

# Good:
if result.returncode != 0:
    # Only actual errors, not progress bars
    error_lines = [l for l in result.stderr.split('\n') 
                   if 'error' in l.lower() or 'exception' in l.lower()]
    if error_lines:
        raise Exception('\n'.join(error_lines))
```

### 3. Progress Streaming (Priority: MEDIUM)
Use subprocess.Popen with real-time stdout/stderr reading to update job progress.

---

## What's Working ✅

- Server starts successfully
- YouTube download (yt-dlp) works
- GPU detection (M3 Max MPS) works  
- All Python modules load
- Frontend renders correctly
- Practice mode UI is built (untested due to no stems)

## What's Not Working ❌

- Stem separation times out / fails
- Cannot test practice mode (needs stems first)
- Cannot test tab viewer / cursor sync

---

## Key Files for Review

| File | Lines | Purpose |
|------|-------|---------|
| `backend/app.py` | 2436 | Flask API, separation orchestration |
| `backend/separation/ensemble.py` | 296 | Multi-model ensemble logic |
| `frontend/index.html` | 6308 | Main UI |
| `frontend/practice.html` | 3017 | Tab viewer with cursor |

## Environment

- macOS (Apple Silicon M3 Max)
- Python 3.14 (latest)
- Homebrew packages
- Port 5555

---

## Quick Test Command

```bash
cd /Users/jeffkozelski/stemscribe/backend
python3 app.py
# Then visit http://localhost:5555
# Try a SHORT song (2-3 min) - longer ones timeout
```

## Dependencies Fixed

- `lameenc` doesn't support Python 3.14 - patched demucs/audio.py to make it optional
- Using ffmpeg for WAV→MP3 conversion instead

---

*Generated for Manus 1.7 audit - Feb 2026*
