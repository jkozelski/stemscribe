# GPU Memory Management Analysis

## Date: 2026-02-27
## Author: Pipeline Agent

---

## 1. Current Memory Management State

### What EXISTS:
- `GPUManager` class in `separation/gpu_manager.py` with `clear_memory()`, `can_load_model()`, `get_memory_usage()`
- `DemucsWrapper.unload_model()` in `separation/models/demucs_wrapper.py` (gc.collect + cache clear)
- Manual `gc.collect()` in `separate_stems_roformer()` (line 937) and `separate_stems_mdx()` (lines 1085, 1121)
- `del separator` in some paths

### What's MISSING:
1. **No MPS cache clearing** in the main pipeline (only in GPUManager and DemucsWrapper)
2. **No model unloading between stages** -- transcription models stay in memory after use
3. **No memory check before loading** new models in the main pipeline
4. **Singleton transcriber instances** (`_transcriber` globals) are never cleaned up
5. **Piano model** (138MB) stays loaded via `_transcriber` global forever
6. **Drum CRNN** (109MB) stays loaded via module-level global
7. **Chord detector models** (v7: 0.3MB, v8: 1.8MB) stay loaded
8. **EnhancedSeparator** instances created per-call but inner `Separator` object isn't explicitly freed

---

## 2. Full Model Loading Sequence in `process_audio()`

Here's every model load during a standard processing run, in order:

| Stage | Model | Approx Memory | Loaded By | Freed? |
|-------|-------|---------------|-----------|--------|
| Separation Pass 1 | BS-RoFormer (audio-separator) | ~2GB | `EnhancedSeparator` | `del separator` + `gc.collect()` |
| Separation Pass 2 | Demucs htdemucs_6s | ~1.5GB | `DemucsRunner` (subprocess) | Subprocess exits |
| Vocal Split | UVR_MDXNET_KARA_2 (audio-separator) | ~500MB | New `EnhancedSeparator` | **NEVER freed** |
| Guitar Split | MelBand-RoFormer custom | ~800MB | `GuitarSeparator` | **NEVER freed** |
| Smart Separate | Various (recursive) | ~1-2GB | Multiple | Varies |
| Drum Transcription | Drum CRNN | ~109MB | `NeuralDrumTranscriber` | **NEVER freed** (global singleton) |
| Guitar Transcription | Basic Pitch model | ~15MB | `GuitarTabTranscriber` | **NEVER freed** (global singleton) |
| Bass Transcription | Basic Pitch model | ~15MB | `BassTranscriber` | **NEVER freed** (global singleton) |
| Piano Transcription | Piano CRNN | ~138MB | `PianoTranscriber` | **NEVER freed** (global singleton) |
| Melody Extraction | pyin/librosa | ~50MB | `MelodyExtractor` | **NEVER freed** |
| Chord Detection | Chord CRNN v8 | ~2MB | `ChordDetectorV8` | **NEVER freed** |
| GP Conversion | (CPU only) | ~0 | `midi_to_gp` | N/A |

**Peak estimated memory:** ~4-5GB during separation phase, dropping to ~500MB during transcription, but transcription models accumulate to ~300MB and are never freed.

---

## 3. Problem Areas

### 3.1 Separation Phase (Biggest Memory Consumer)

The RoFormer pass properly cleans up (`del separator`, `gc.collect()`). But:

- **Vocal split** (line 2439): Creates a new `EnhancedSeparator` that loads the Karaoke model. Never freed.
- **Guitar split** (line 2464-ish): Creates `GuitarSeparator` which loads MelBand-RoFormer. Never freed.
- Both run AFTER the main separation, so they pile on top.

```python
# Line 2439 - no cleanup after this
separator = EnhancedSeparator(output_dir=str(vocal_split_dir))
lead_path, backing_path = separator.split_lead_backing_vocals(job.stems['vocals'])
# separator goes out of scope but Python may not GC it immediately
```

### 3.2 Transcription Phase (Model Accumulation)

Every transcriber uses a module-level `_transcriber` singleton pattern:
```python
_transcriber: Optional[GuitarTabTranscriber] = None

def transcribe_guitar_tab(audio_path, output_dir, tempo_hint=None):
    global _transcriber
    if _transcriber is None:
        _transcriber = GuitarTabTranscriber()
    ...
```

This means:
- First call loads the model
- Model stays in memory for the life of the Flask process
- 4 transcribers x avg ~70MB = ~280MB permanently allocated
- Plus chord detector: ~2MB
- **On concurrent jobs, these are shared** (good for memory, but not thread-safe)

### 3.3 DemucsRunner (Subprocess - Actually Clean)

`DemucsRunner` runs Demucs as a subprocess (`python3 -m demucs`), so it naturally cleans up when the subprocess exits. This is actually the best memory management pattern in the codebase.

### 3.4 `separate_stems_hq_vocals` and `separate_stems_vocal_focus`

These use raw `subprocess.run` for Demucs (which is fine for memory, but they miss the progress-as-error fix from DemucsRunner).

---

## 4. Recommendations

### Priority 1: Add explicit cleanup between stages

Add a `clear_pipeline_memory()` helper function:

```python
def clear_pipeline_memory():
    """Clear GPU/MPS memory between pipeline stages."""
    import gc
    import torch

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
```

Call this between separation and transcription, and after vocal/guitar split.

### Priority 2: Add `del` + cleanup for post-separation models

In `process_audio()`, after vocal split and guitar split:

```python
# After vocal split
separator = EnhancedSeparator(output_dir=str(vocal_split_dir))
lead_path, backing_path = separator.split_lead_backing_vocals(...)
del separator  # ADD THIS
clear_pipeline_memory()  # ADD THIS

# After guitar split
guitar_sep = GuitarSeparator()
lead, rhythm = guitar_sep.separate(...)
del guitar_sep  # ADD THIS
clear_pipeline_memory()  # ADD THIS
```

### Priority 3: Add memory monitoring

Log memory usage at each pipeline stage using `GPUManager.get_memory_usage()`:

```python
gpu_mgr = GPUManager()

# Before separation
logger.info(f"Memory before separation: {gpu_mgr.get_memory_usage()}")

# Before transcription
logger.info(f"Memory before transcription: {gpu_mgr.get_memory_usage()}")
```

### Priority 4: Optional - Add transcriber cleanup

For the singleton transcribers, add cleanup functions:

```python
# In each transcriber module
def unload():
    """Free transcriber model from memory."""
    global _transcriber
    if _transcriber is not None:
        if hasattr(_transcriber, '_model') and _transcriber._model is not None:
            del _transcriber._model
        _transcriber = None
```

Call these after transcription is complete for all stems:

```python
# After all transcription is done
if GUITAR_TAB_MODEL_AVAILABLE:
    from guitar_tab_transcriber import unload as unload_guitar
    unload_guitar()
# etc.
```

### Priority 5: Use GPUManager in main pipeline

The `GPUManager` exists but is only used in the ensemble separation path. It should be the central memory coordinator:

```python
# In process_audio()
gpu_mgr = GPUManager()

# Before loading each heavy model
if not gpu_mgr.can_load_model(2.0):  # 2GB for RoFormer
    gpu_mgr.clear_memory()

# After each stage
gpu_mgr.clear_memory()
```

---

## 5. Concurrent Job Safety

The current singleton transcriber pattern is **not thread-safe**. If two jobs are processing simultaneously:

```python
# Thread 1: Processing guitar
_transcriber = GuitarTabTranscriber()  # loads model
result = _transcriber.transcribe(audio_A, ...)

# Thread 2: Processing guitar (same singleton!)
result = _transcriber.transcribe(audio_B, ...)  # Uses same instance
```

This works for Basic Pitch (stateless inference) but could cause issues with models that have internal state.

**Fix:** Use thread-local storage or per-job transcriber instances with explicit cleanup.

---

## 6. Impact Assessment

| Change | Effort | Memory Savings | Risk |
|--------|--------|---------------|------|
| clear_pipeline_memory() between stages | Low | ~500MB-2GB during peaks | None |
| del + cleanup for post-sep models | Low | ~1.3GB (Karaoke + GuitarSep) | None |
| Memory logging | Low | Diagnostic only | None |
| Transcriber unload functions | Medium | ~280MB after transcription | Low |
| GPUManager integration | Medium | Prevention of OOM | Low |
| Thread-safe transcribers | High | N/A (correctness) | Medium |

---

*Analysis compiled: 2026-02-27*
