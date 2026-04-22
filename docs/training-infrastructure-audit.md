# Training Infrastructure Audit

**Date:** 2026-04-04
**Scope:** Guitar, bass, and chord training pipelines + current transcription routing

---

## 1. Guitar Training (`~/stemscribe/train_guitar_model/`)

### 1A. Lead/Rhythm Separation (MelBand-Roformer)

**Files:** `train.py`, `config.yaml`, `configs/`, `models/`, `utils/`

This is a **source separation** pipeline (NOT transcription). It's based on ZFTurbo's Music Source Separation Training framework.

- **Architecture:** MelBand-Roformer (mel_band_roformer)
  - dim=192, depth=8, 60 frequency bands, 8 attention heads
  - Stereo input/output, 44.1 kHz, STFT with 2048 FFT / 512 hop
  - Multi-STFT resolution loss
- **Purpose:** Separate a guitar stem into **lead_guitar** and **rhythm_guitar** sub-stems
- **Config:** `config.yaml` defines 2-stem separation (lead_guitar, rhythm_guitar)
- **Training:** Supports DDP multi-GPU, AMP, EMA, gradient accumulation, WandB logging
- **Pretrained checkpoint:** `models/last_mel_band_roformer.ckpt` (181 MB) -- this is a starting checkpoint for fine-tuning, NOT a trained guitar model
- **Dataset:** `dataset/` contains:
  - `downloads/` -- 20 full WAV songs (~1.3 GB): Allman Brothers, Dire Straits, Eagles, Foo Fighters, Grateful Dead, GNR, Led Zeppelin, etc.
  - `training/` -- 18 song directories, each with `lead_guitar.wav` + `rhythm_guitar.wav` (pre-separated stereo splits)
  - `manifest.json` -- metadata linking songs to their lead/rhythm WAVs with stereo analysis
- **Status:** NEVER TRAINED. The infrastructure is ready but no training has been run. No output model exists.

### 1B. Guitar Transcription (Kong CRNN Domain Adaptation)

**File:** `train_guitar_runpod.py` (standalone RunPod script)

This is the actual **note transcription** training -- separate from the separation model above.

- **Architecture:** Kong et al. (2021) CRNN replicated from ByteDance piano_transcription
  - 4 ConvBlocks (48->64->96->128 channels) -> FC(768) -> BiGRU(256) -> Linear
  - Four parallel acoustic models: frame, onset, offset, velocity
  - Onset conditioned on velocity; frame conditioned on onset + offset
  - Guitar-specific: 48 pitches (MIDI 40-88, E2-E6) instead of 88 piano keys
- **Training strategy:** Riley et al. (ICASSP 2024) domain adaptation recipe
  - Phase 1: Freeze CNN, train RNN + output heads (20 epochs, ~2h on A100, LR 5e-5)
  - Phase 2: Full fine-tuning (80 epochs, ~6h on A100, LR 1e-5)
- **Starting checkpoint:** Kong piano checkpoint from Zenodo (172 MB) -- downloaded at runtime
- **Dataset:** GuitarSet from Zenodo (657 MB audio + 39 MB JAMS annotations) -- downloaded at runtime
- **Input features:** 229-bin log-mel spectrogram at 16 kHz, 160 hop, 2048 FFT
- **Loss:** Focal BCE (onset pw=50, gamma=2) + BCE (frame pw=10) + MSE (velocity)
- **Augmentation:** Gain, noise injection, pitch shift, SpecAugment
- **Output:** `/workspace/guitar_results/best_guitar_model.pt` -> copy to `backend/models/pretrained/best_guitar_model.pt`
- **Status:** NEVER TRAINED. No `best_guitar_model.pt` exists anywhere in the project. The inference module `backend/guitar_nn_transcriber.py` exists and is wired into the pipeline but has no model to load.

### 1C. Also present

- `Guitar_Training_Colab.ipynb` -- Colab notebook version
- `Guitar_Lead_Rhythm_Training.ipynb` -- Older notebook
- `guitar_training_package.zip` (942 MB) -- packaged version for cloud upload
- `configs/config_guitar_lead_rhythm.yaml` -- dedicated Roformer config for guitar separation
- 44 other config YAMLs for various source separation architectures (MUSDB, Demucs, SCNet, etc.)

---

## 2. Bass Training (`~/stemscribe/train_bass_model/`)

### 2A. Bass Transcription v3 (Custom CNN+LSTM)

**Files:** `train_bass_v3.py`, `train_bass_runpod.py` (identical content, both standalone RunPod scripts)

- **Architecture:** Custom CNN + bidirectional LSTM
  - CNN: 3 conv layers (1->32->64->128) with BatchNorm, ReLU, MaxPool, Dropout
  - Onset LSTM: BiLSTM (hidden=256, 2 layers) on CNN features
  - Frame LSTM: BiLSTM conditioned on onset probabilities
  - Velocity head: sigmoid output
  - Output shape: (batch, 4 strings, 24 frets, time)
- **Purpose:** Bass note transcription with string/fret assignment
  - 4 strings, 24 frets, standard tuning [E1=28, A1=33, D2=38, G2=43]
- **Input features:** 84-bin CQT at 22.05 kHz, 256 hop, 5-second chunks
- **Dataset:** Slakh2100-redux from Zenodo (~97 GB) -- downloaded and filtered at runtime
  - Extracts bass stems (MIDI programs 32-39) with paired audio (FLAC) + MIDI
  - Train/validation/test split from Slakh's own splits
- **Loss:** Focal BCE (onset, pw=50, gamma=2) + BCE (frame, pw=10) + 0.5 * MSE (velocity)
- **Training:** 50 epochs, batch=16, AdamW, ReduceLROnPlateau, collapse detection (zero recall threshold)
- **Augmentation:** v2: gain jitter + noise injection. v3 docstring mentions pitch shift, time stretch, reverb, EQ, SpecAugment but the actual code only implements v2 augmentations.
- **Output:** `/workspace/bass_results/best_bass_model.pt` -> copy to `backend/models/pretrained/best_bass_model.pt`
- **Status:** NEVER TRAINED. No `best_bass_model.pt` exists anywhere. The inference module `backend/bass_nn_transcriber.py` exists and is wired in but has no model to load.

### 2B. Also present

- `Bass_Transcription_Training.ipynb` -- v1 notebook
- `Bass_Transcription_Training_v2.ipynb` -- v2 notebook with focal loss additions

---

## 3. Transcription Pipeline (`backend/processing/transcription.py`)

The main `transcribe_to_midi()` function routes each stem to the best available transcriber with a cascading fallback chain:

### Drums
1. **Neural CRNN** (`drum_nn_transcriber.py`, `best_drum_model.pt` 115 MB) -- AVAILABLE, TRAINED
2. OaF neural network
3. Spectral drum transcriber v2 (ghost notes, cymbals)
4. Drum transcriber v1

### Guitar
1. **Guitar v3 NN** (`guitar_nn_transcriber.py`) -- CODE EXISTS, NO TRAINED MODEL
2. **Guitar Tab** (Basic Pitch, `guitar_tab_transcriber.py`) -- AVAILABLE (uses Basic Pitch with guitar post-processing)
3. Chord-to-tab fallback (if chord progression detected)
4. Melody extractor (for lead/monophonic lines)
5. Enhanced transcriber (with articulation detection)
6. **Basic Pitch** (final fallback) -- ALWAYS AVAILABLE

### Bass
1. **Bass v3 NN** (`bass_nn_transcriber.py`) -- CODE EXISTS, NO TRAINED MODEL
2. **Bass model** (Basic Pitch, `bass_transcriber.py`) -- AVAILABLE (uses Basic Pitch with bass post-processing)
3. Melody extractor
4. Enhanced transcriber
5. **Basic Pitch** (final fallback)

### Piano
1. **Neural CRNN** (`piano_transcriber.py`, `best_piano_model.pt` 145 MB) -- AVAILABLE, TRAINED
2. Enhanced transcriber
3. Basic Pitch

### Current reality
- **Guitar and bass currently fall through to Basic Pitch** (with guitar_tab_transcriber / bass_transcriber wrappers) because no NN models are trained
- Drums and piano have trained CRNN models and work with their neural pipelines
- All transcription modes include MIDI quantization (16th note grid by default)

---

## 4. Training Data (`backend/training_data/`)

### 4A. Chord Training Data

| Directory | Size | Contents |
|-----------|------|----------|
| `billboard/` | 15 MB | McGill-Billboard dataset: ~890 songs with chord annotations (lab files + index CSV) |
| `btc_finetune/` | 1.3 GB | BTC fine-tuning data: 988 songs with audio (WAV symlinks), labels, pre-computed features, and 3 trained checkpoints (`btc_finetuned_best.pt`, backup, pre-augment) |
| `chord_db/` | 548 KB | `chord_database.json` (242 KB), `song_urls.json` (303 KB), `ug_test_songs.json` |
| `chords/` | 340 KB | `chord_training_data.npz` (330 KB) + `metadata.json` |
| `jaah/` | 1.0 GB | Jazz Audio-Aligned Harmony: 113 songs with JSON annotations, lab files, pre-computed features |
| `chord_vocabulary_index.json` | 3.5 MB | 228K-line chord vocabulary index |

### 4B. Guitar Separation Training Data (in `train_guitar_model/dataset/`)

- 20 full songs as WAV (1.3 GB total)
- 18 songs split into lead_guitar.wav + rhythm_guitar.wav pairs
- Songs: classic rock (Eagles, Allman Brothers, Grateful Dead, GNR, Led Zeppelin, Dire Straits, Foo Fighters)

### 4C. What's NOT present locally

- **No GuitarSet** (for guitar NN transcription) -- downloaded at runtime on RunPod
- **No Slakh2100** (for bass NN transcription) -- downloaded at runtime on RunPod (~97 GB)
- **No guitar transcription training data** with audio+MIDI pairs exists locally

---

## 5. Pretrained Models Inventory (`backend/models/pretrained/`)

| Model | Size | Status |
|-------|------|--------|
| `best_drum_model.pt` | 115 MB | TRAINED, IN USE |
| `best_piano_model.pt` | 145 MB | TRAINED, IN USE |
| `v7_chord_model.pt` | 312 KB | Trained |
| `v8_chord_model.pt` | 1.8 MB | Trained (+ `v8_classes.json`) |
| `v9_chord_model.pt` | 1.7 MB | Trained, latest (+ `v9_classes.json`) |
| `best_guitar_model.pt` | -- | DOES NOT EXIST |
| `best_bass_model.pt` | -- | DOES NOT EXIST |

---

## 6. Summary: What's Ready vs. What's Missing

### Ready to use
- Drum CRNN transcription (trained, deployed)
- Piano CRNN transcription (trained, deployed)
- Chord detection (v7/v8/v9 models trained, BTC fine-tuned)
- Basic Pitch fallback for guitar and bass (working today)
- Guitar tab transcriber wrapper around Basic Pitch (working today)
- Bass transcriber wrapper around Basic Pitch (working today)

### Code complete, needs training
- **Guitar NN transcriber** -- RunPod script ready, inference code wired in, needs ~8h on A100 ($3-5)
  - Downloads GuitarSet (~700 MB) and Kong checkpoint (~172 MB) automatically
  - Domain adaptation from piano CRNN to guitar
- **Bass NN transcriber** -- RunPod script ready, inference code wired in, needs ~4h on A100 ($2-3) + 97 GB disk for Slakh
  - Downloads Slakh2100-redux and extracts bass stems automatically

### Infrastructure present but unused
- Guitar lead/rhythm separation (MelBand-Roformer) -- 18 training pairs exist, training framework ready
  - This would enable splitting a "guitar" stem into lead and rhythm sub-stems before transcription
  - Requires GPU training (not costed or prioritized)

### Estimated cost to train guitar + bass NN models
- Guitar: ~8h on A100 at ~$0.60/hr = **$5**
- Bass: ~4h on A100 at ~$0.60/hr = **$3** (plus needs large disk pod for Slakh)
- Total: approximately **$8-10** on RunPod
