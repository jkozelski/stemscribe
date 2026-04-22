# Phase 1: AI Tab Training Pipeline - Data Prep Status

**Date:** 2026-04-04
**Cost:** $0 (all local CPU work)
**GPU used:** None
**Modal used:** None

---

## Task 1: Chord Vocabulary Index - COMPLETE

**File:** `backend/training_data/chord_vocabulary_index.json` (3.5 MB)

- Extracted from 15,416 chord charts in `backend/chord_library/`
- Fields per song: title, artist, key, chords_used (no lyrics)
- **2,364 unique chords** after cleaning (removed bare slash prefixes, whitespace)
- Structured as constrained decoding lookup table:
  ```json
  {
    "metadata": { "total_songs": 15416, "unique_chords": 2364 },
    "chord_vocabulary": ["A", "A#", "A#/A", ...],
    "songs": [{ "title": "...", "artist": "...", "key": "...", "chords_used": [...] }]
  }
  ```

---

## Task 2: trimplexx/music-transcription - COMPLETE

**Location:** `train_tab_model/trimplexx/`

- Cloned from https://github.com/trimplexx/music-transcription
- MIT License
- CRNN architecture (CNN encoder + Bidirectional GRU + dual heads)
- All Python deps already satisfied in `venv311/` (librosa, mirdata, torch, etc.)
- **CPU inference verified** with dummy input (see Task 5)

### Key Architecture Details:
| Component | Detail |
|-----------|--------|
| Input | CQT spectrogram: 168 bins (24 bins/octave, 7 octaves), fmin=E2 (82.4 Hz) |
| Sample rate | 22,050 Hz |
| Hop length | 512 samples (~23ms) |
| CNN | 5-layer Conv2D (32->64->128->128->128) + BatchNorm + ReLU + MaxPool |
| RNN | 2-layer Bidirectional GRU, hidden=768, dropout=0.5 |
| Onset head | Linear -> [batch, time, 6] (per-string onset probability) |
| Fret head | Linear -> [batch, time, 6, 22] (silence + frets 0-20) |
| Parameters | ~20.7M |
| Best score | 0.8736 MPE F1, 0.8569 TDR F1 (run_72) |

---

## Task 3: GuitarSet - IN PROGRESS

**Location:** `train_tab_model/trimplexx/python/_mir_datasets_storage/`

- **Annotations:** Downloaded (360 JAMS files, ~37 MB) - COMPLETE
- **Audio mix:** Downloading (~651 MB) - IN PROGRESS
- License: CC BY 4.0
- 360 recordings from 6 performers
- Styles: Rock, Jazz, Funk, Bossa Nova, Singer-Songwriter
- Format: WAV audio + JAMS annotations (onset, pitch, string, fret)
- trimplexx uses `audio_mix` (mono mix) as primary, `audio_mic` as fallback
- Hex pickup audio (3.36GB each for debleeded/original) skipped -- not needed for training

### What trimplexx expects:
- mirdata loads GuitarSet from `_mir_datasets_storage/`
- Preprocessor reads audio_mix WAV + JAMS annotations
- Generates CQT spectrograms + onset/fret label matrices
- Splits 80/10/10 train/val/test (2 problematic files excluded)

---

## Task 4: DadaGP - DOCUMENTED (not downloaded)

**GitHub:** https://github.com/dada-bots/dadaGP
**Paper:** "DadaGP: A Dataset of Tokenized GuitarPro Songs for Sequence Models" (ISMIR 2021)
**Authors:** Sarmento, Kumar, Carr, Zukowski, Barthet, Yang

| Property | Value |
|----------|-------|
| Size | 26,181 GuitarPro songs in 739 genres |
| Format | Tokenized .gp3/.gp4/.gp5 files (text token sequences) |
| License | MIT (encoder/decoder code); dataset requires email request |
| Download | Contact authors (not publicly downloadable) |
| Contains | Symbolic tab data ONLY (no audio) |

### Compatibility Assessment:
- **Not directly usable** for audio->tab training (no audio included)
- **Potential use cases:**
  1. Pre-train a tab language model for constrained decoding
  2. Generate synthetic training pairs via MIDI synthesis (render tabs to audio)
  3. Augment GuitarSet with synthetic data
- **Action needed:** Email authors to request dataset access for research
- The encoder/decoder tool on GitHub can convert between GuitarPro and token format

---

## Task 5: CPU Inference Verification - COMPLETE

Successfully ran trimplexx model on CPU with dummy data:

```
Input:  torch.Size([1, 168, 100])  # [batch, CQT_bins, time_frames]
Onset:  torch.Size([1, 100, 6])    # [batch, time, strings]
Fret:   torch.Size([1, 100, 6, 22])  # [batch, time, strings, fret_classes]
```

### Pipeline Summary:
1. **Input:** Raw audio WAV (any length)
2. **Preprocessing:** Resample to 22050 Hz -> CQT spectrogram (168 bins)
3. **Model:** CRNN forward pass (works on CPU, ~20.7M params)
4. **Output:** Per-frame onset probabilities + fret class logits for 6 strings
5. **Post-processing:** Threshold onsets -> argmax frets -> ASCII tab or MIDI export

### Verified:
- All imports work in venv311
- Model instantiates on CPU without errors
- Forward pass produces correct output shapes
- No CUDA/GPU dependency for inference

---

## Next Steps (Phase 2)

1. **Complete GuitarSet download** (~651 MB mix audio remaining)
2. **Run trimplexx preprocessing** on GuitarSet (CQT extraction + label generation)
3. **Train baseline model** on GuitarSet (CPU feasible but slow; ~300 epochs, batch=2)
4. **Evaluate baseline** MPE F1 and TDR F1 scores
5. **Integrate with StemScriber** inference pipeline
6. **Contact DadaGP authors** for dataset access (synthetic data augmentation)
7. **Consider:** Training on M3 Max MPS (Apple Silicon GPU) for faster local iteration

### Integration Path:
```
User uploads song
  -> Stem separation (RoFormer/Demucs)
  -> Guitar stem isolated
  -> CQT spectrogram (22050 Hz, 168 bins)
  -> trimplexx CRNN inference
  -> Onset + fret predictions
  -> Constrained by chord_vocabulary_index.json
  -> Guitar Pro (.gp5) export
```
