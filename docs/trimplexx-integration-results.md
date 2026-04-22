# Trimplexx CRNN Integration Results

**Date:** 2026-04-13
**Status:** DEPLOYED to VPS and running in production

## What Was Done

### 1. Created trimplexx_transcriber.py
- Self-contained transcriber module at `backend/trimplexx_transcriber.py`
- Loads model from `backend/models/pretrained/trimplexx_guitar_model.pt` (78.9MB)
- Reads hyperparameters from `backend/models/pretrained/trimplexx_run_config.json`
- Full model architecture (TabCNN + GuitarTabCRNN) embedded in file -- no dependency on trimplexx training repo
- Outputs both MIDI (pipeline compat) and `tab_data` list with `{start_time, end_time, string, fret, pitch_midi}`

### 2. Wired into Pipeline
- `dependencies.py`: Added `TRIMPLEXX_MODEL_AVAILABLE` flag + import block
- `processing/transcription.py`: Trimplexx is now **highest priority** for guitar stems
- Fallback chain: **trimplexx -> guitar_nn (Kong v3) -> basic_pitch guitar_tab -> melody -> enhanced -> basic_pitch**
- Bass stems unchanged (trimplexx is guitar-only)
- Tab data stored on `job.tab_data[stem_name]` for downstream Guitar Pro generation

### 3. Architecture Details
| Parameter | Value |
|-----------|-------|
| CNN | 5 layers: [32, 64, 128, 128, 128] channels, MaxPool(2,1) |
| RNN | BiGRU, hidden=768, layers=2, dropout=0.5 |
| Input | CQT spectrogram: 22050 Hz, hop=512, 168 bins, 24 bins/oct, fmin=E2 |
| Output (onset) | 6 logits per frame (one per string) |
| Output (fret) | 6 x 22 logits per frame (frets 0-20 + silence) |
| Parameters | 20,673,354 |
| Onset threshold | 0.5 |

### 4. Test Results -- Thunderhead Guitar Stem (job 4a0965d1)

| Metric | Local (M3 Max) | VPS (4-core Hetzner) |
|--------|---------------|---------------------|
| Inference time | 3.6s | 13.3s |
| Notes detected | 890 | 890 |
| Quality score | 0.97 | 0.97 |
| Pitch range | MIDI 43-69 | MIDI 43-69 |
| Polyphony | 2.1 | 2.1 |
| Fret range | 0-8 | 0-8 |

**Notes per string:** {0: 136, 1: 54, 2: 289, 3: 259, 4: 132, 5: 20}

Sample output shows correct chord voicings (barred 8th fret shapes), proper string distribution across all 6 strings, and musically coherent fret positions.

### 5. Deployment
All files deployed to VPS at `/opt/stemscribe/`:
- `backend/trimplexx_transcriber.py`
- `backend/dependencies.py` (updated)
- `backend/processing/transcription.py` (updated)
- `backend/models/pretrained/trimplexx_guitar_model.pt`
- `backend/models/pretrained/trimplexx_run_config.json`

Service restarted: `systemctl restart stemscribe`

Startup logs confirm:
```
INFO:trimplexx_transcriber:Trimplexx guitar model found (82.7MB)
INFO:dependencies:Trimplexx guitar model available (CRNN, string+fret, TDR F1=0.857)
```

## Key Improvement Over Previous Guitar Pipeline

The Kong-style `guitar_nn_transcriber.py` outputs MIDI pitches only -- string/fret assignment requires the A*-Guitar FretMapper post-processor in `midi_to_gp.py`. Trimplexx outputs string+fret positions **directly** from the model, meaning:

1. No ambiguity in fret assignment (the model learned guitar ergonomics from GuitarSet)
2. A*-Guitar can be bypassed when trimplexx tab_data is available
3. 85% TDR F1 vs Kong model which has no string/fret awareness at all
