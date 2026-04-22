# BTC Model Loader Fix — Results

**Date:** 2026-03-16
**File changed:** `backend/chord_detector_v10.py` (line 301)

## Problem

On PyTorch 2.6+, `torch.load` defaults to `weights_only=True`. The BTC model checkpoint contains numpy globals (`numpy.core.multiarray.scalar`) that aren't in the safe globals list. The existing fix attempted to allowlist `np._core.multiarray.scalar` (the new numpy 2.x internal path), but PyTorch's unpickler references the old `numpy.core.multiarray.scalar` path from the pickle file.

Error message:
```
Unsupported global: GLOBAL numpy.core.multiarray.scalar was not an allowed global by default.
```

## Fix Applied

Changed from attempting `weights_only=True` with safe globals allowlisting to `weights_only=False`, since this is a trusted local model checkpoint:

```python
# Before (broken):
torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.ndarray, np.dtype, np.dtypes.Float64DType])
checkpoint = torch.load(model_file, map_location=self._device, weights_only=True)

# After (working):
checkpoint = torch.load(model_file, map_location=self._device, weights_only=False)
```

## Verification

### Model Loading
- BTC model loads successfully (fine-tuned checkpoint)
- Chord DB loaded: 856 songs

### Chord Detection — Little Wing Guitar Stem
- File: `outputs/a0db058f/stems/htdemucs_6s/Jimi_Hendrix_-_Little_wing_-_HQ/guitar.mp3`
- Key detected: F#
- Chords detected: 21
- Confidence range: 0.620 to 0.950 (mean 0.756)
- Confidence scores are real and varied (NOT flat 0.60)

### Test Suite
- All 138 tests pass (`pytest backend/tests/ -v`)

## Other torch.load Calls

Checked all `torch.load` calls in the codebase:
- `btc_chord/test.py` — already uses `weights_only=False`
- `btc_chord/train.py`, `train_crf.py`, `audio_dataset.py`, `utils/pytorch_utils.py` — training scripts, no `weights_only` param (defaults vary by PyTorch version, but these are not used in production)
- `backend/btc_retrain_augmented.py` — already uses `weights_only=False`
- Other backend model loaders (`piano_transcriber.py`, `drum_nn_transcriber.py`, etc.) — use `weights_only=True` and work fine (pure PyTorch state dicts without numpy)
