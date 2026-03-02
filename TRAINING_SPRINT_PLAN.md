# StemScribe Training Sprint Runbook
## Guitar & Bass Transcription Model Training

**Goal:** Replace Basic Pitch fallback with domain-adapted CRNN models achieving 85%+ onset F1 on GuitarSet.

**Approach:** Fine-tune Kong et al. piano transcription checkpoint on guitar/bass data using domain adaptation (Riley et al. ICASSP 2024 recipe).

**Timeline:** 2-3 days for guitar, 1-2 days for bass (assumes Colab Pro or RunPod A100).

---

## Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Guitar Model Training](#2-guitar-model-training)
3. [Bass Model Training](#3-bass-model-training)
4. [Colab vs RunPod](#4-colab-vs-runpod)
5. [Integration into StemScribe](#5-integration-into-stemscribe)
6. [Validation & Testing](#6-validation--testing)

---

## 1. Prerequisites

### Python Environment
```bash
pip install torch torchaudio librosa mir_eval pretty_midi numpy scipy
pip install audiomentations  # for data augmentation
```

### Hardware
- GPU: NVIDIA A100 (40GB) recommended. T4 (16GB) works with batch_size=2.
- VRAM usage: ~12GB for batch_size=4, ~6GB for batch_size=2
- Training time: ~4-8 hours on A100, ~12-20 hours on T4

---

## 2. Guitar Model Training

### Step 2.1: Download Kong Piano Checkpoint (Pre-trained Weights)

The Kong et al. (2021) piano transcription model serves as our pre-training source. It was trained on MAESTRO v3 (200+ hours of piano) and provides strong onset/frame/velocity detection that transfers well to guitar.

**Option A: Via pip (recommended)**
```bash
pip install piano_transcription_inference
# Checkpoint auto-downloads on first use (~172MB)
# Location: ~/.cache/piano_transcription_inference/
python -c "
from piano_transcription_inference import PianoTranscription
import torch
# This triggers the download
model = PianoTranscription(device='cpu')
print('Checkpoint downloaded successfully')
# The model weights are in model.model.state_dict()
"
```

**Option B: Direct Zenodo download**
```bash
# Zenodo record: https://zenodo.org/record/4034264
wget https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth -O kong_piano_checkpoint.pth
# Size: ~172MB, License: CC BY 4.0
```

**Architecture details (must match for weight loading):**
- Model: `Regress_onset_offset_frame_velocity_CRNN`
- Input: Mel spectrogram (229 bins, sr=16000, hop=160)
- CNN: 3 conv blocks with batch norm
- RNN: 2-layer BiGRU (hidden=256)
- Heads: onset, offset, frame, velocity (each 88 outputs for piano keys)
- Total params: ~20M

### Step 2.2: Download Guitar Datasets

**GuitarSet (required, 3 hours)**
```bash
# Standard benchmark dataset - 360 recordings, 6 guitarists, hex pickup + mic
# Direct download from Zenodo
mkdir -p datasets/guitarset
cd datasets/guitarset
wget https://zenodo.org/record/3371780/files/GuitarSet_audio.zip
wget https://zenodo.org/record/3371780/files/GuitarSet_annotation.zip
unzip GuitarSet_audio.zip
unzip GuitarSet_annotation.zip
# Audio: WAV files (mono, 44.1kHz)
# Annotations: JAMS format (onset times, MIDI pitches, string assignments)
```

**GAPS Dataset (optional, 14 hours classical guitar)**
```bash
# ISMIR 2024 - 14h of classical guitar with aligned MIDI
# CAVEAT: Audio is YouTube-linked, not directly included
mkdir -p datasets/gaps
cd datasets/gaps
wget https://zenodo.org/record/TODO/files/GAPS_v1.zip  # ~7MB (metadata only)
unzip GAPS_v1.zip
# Contains: MIDI annotations, YouTube URLs, alignment data
# You MUST download audio separately:
pip install yt-dlp
python download_gaps_audio.py  # Script included in GAPS zip
# Expected: ~14 hours of audio after download
# Some YouTube videos may be unavailable - expect 10-15% attrition
```

**GOAT Dataset (optional, 68 hours, ISMIR 2025)**
```bash
# Guitar-Only Audio Transcription dataset
# Synthetic + real recordings with detailed annotations
# Check availability at: https://github.com/xavriley/GOAT-AMT
```

**SynthTab (optional, 13,000+ hours synthetic)**
```bash
# ICASSP 2024 - Synthetic guitar audio from DadaGP tablatures
# Massive pre-training dataset generated with VST plugins
# Check: https://github.com/SonyCSLParis/SynthTab
# WARNING: Very large download, use only for extended pre-training
```

### Step 2.3: Data Preparation

```python
"""
Prepare GuitarSet for training.
Converts JAMS annotations to frame-level onset/frame/velocity matrices.
"""
import librosa
import jams
import numpy as np
from pathlib import Path

SAMPLE_RATE = 16000
HOP_LENGTH = 160  # Match Kong checkpoint
N_MELS = 229      # Match Kong checkpoint
GUITAR_MIDI_RANGE = (40, 88)  # E2 to E6 (48 pitches)

def load_guitarset_track(audio_path, jams_path):
    """Load a GuitarSet track and create training targets."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    # Compute mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=2048,
        hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Parse JAMS annotations
    jam = jams.load(jams_path)
    n_frames = mel_db.shape[1]
    n_pitches = GUITAR_MIDI_RANGE[1] - GUITAR_MIDI_RANGE[0]

    onsets = np.zeros((n_frames, n_pitches), dtype=np.float32)
    frames = np.zeros((n_frames, n_pitches), dtype=np.float32)
    velocities = np.zeros((n_frames, n_pitches), dtype=np.float32)

    for ann in jam.annotations:
        if ann.namespace == 'note_midi':
            for obs in ann.data:
                midi_pitch = int(round(obs.value))
                if GUITAR_MIDI_RANGE[0] <= midi_pitch < GUITAR_MIDI_RANGE[1]:
                    pitch_idx = midi_pitch - GUITAR_MIDI_RANGE[0]
                    onset_frame = int(obs.time * SAMPLE_RATE / HOP_LENGTH)
                    offset_frame = int((obs.time + obs.duration) * SAMPLE_RATE / HOP_LENGTH)

                    onset_frame = min(onset_frame, n_frames - 1)
                    offset_frame = min(offset_frame, n_frames - 1)

                    onsets[onset_frame, pitch_idx] = 1.0
                    frames[onset_frame:offset_frame+1, pitch_idx] = 1.0
                    vel = obs.confidence if obs.confidence else 0.8
                    velocities[onset_frame:offset_frame+1, pitch_idx] = vel

    return mel_db.T, onsets, frames, velocities  # (T, F), (T, P), (T, P), (T, P)
```

### Step 2.4: Training Script

```python
"""
Guitar transcription training via domain adaptation from Kong piano checkpoint.

Based on Riley et al. (ICASSP 2024) recipe:
- Pre-train: Kong piano checkpoint (MAESTRO)
- Fine-tune: GuitarSet with frozen CNN, then full model
- Augmentation: time stretch, pitch shift, EQ, reverb, noise
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---- Hyperparameters (Riley et al. recipe) ----
LEARNING_RATE = 1e-5          # Low LR for fine-tuning
BATCH_SIZE = 4                # Reduce to 2 for T4
MAX_STEPS = 100_000           # ~100K steps
WARMUP_STEPS = 5000
ONSET_POS_WEIGHT = 50.0       # Critical: class imbalance ~1:1835
FRAME_POS_WEIGHT = 10.0
FOCAL_GAMMA = 2.0             # Focal loss for onset head

# ---- Two-phase training ----
PHASE1_EPOCHS = 20            # Frozen CNN, train RNN + heads only
PHASE2_EPOCHS = 80            # Full model, lower LR

# ---- Augmentation (trimplexx recipe: 87.4% F1) ----
AUGMENTATIONS = {
    'time_stretch': (0.8, 1.2),      # Random time stretch
    'pitch_shift': (-2, 2),           # Semitones
    'gain_jitter_db': (-6, 6),        # Volume variation
    'gaussian_noise_snr': (20, 40),   # SNR in dB
    'reverb_probability': 0.3,
    'eq_probability': 0.3,
    'spec_augment_freq_masks': 2,     # SpecAugment
    'spec_augment_time_masks': 2,
}

class FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for extreme class imbalance in onset detection."""
    def __init__(self, pos_weight=50.0, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device),
            reduction='none'
        )
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()

def load_kong_weights(model, checkpoint_path, n_output_pitches=48):
    """
    Load Kong piano checkpoint weights into guitar model.

    Strategy:
    - CNN layers: load directly (frequency features transfer well)
    - RNN layers: load directly (temporal patterns transfer)
    - Output heads: reinitialize (88 piano keys -> 48 guitar pitches)
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    # Filter out output head weights (dimension mismatch: 88 vs 48)
    filtered = {}
    for k, v in state_dict.items():
        if any(head in k for head in ['onset_head', 'frame_head', 'velocity_head', 'offset_head']):
            continue  # Skip output heads
        filtered[k] = v

    # Load compatible weights
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)} weight tensors from Kong checkpoint")
    print(f"Missing (reinitialized): {len(missing)} tensors")
    print(f"Unexpected (skipped): {len(unexpected)} tensors")

    return model

def train_phase1(model, train_loader, val_loader, device, epochs=20):
    """Phase 1: Freeze CNN, train only RNN + heads."""
    # Freeze CNN layers
    for name, param in model.named_parameters():
        if 'conv' in name or 'bn' in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Phase 1: Training {trainable:,} / {total:,} parameters ({trainable/total*100:.1f}%)")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE * 5,  # Higher LR for heads only
        weight_decay=1e-4
    )

    onset_loss_fn = FocalBCEWithLogitsLoss(pos_weight=ONSET_POS_WEIGHT, gamma=FOCAL_GAMMA)
    frame_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(FRAME_POS_WEIGHT).to(device)
    )

    best_val_f1 = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            mel, onset_targets, frame_targets, vel_targets = [b.to(device) for b in batch]

            onset_logits, frame_logits, vel_pred = model(mel)

            loss = onset_loss_fn(onset_logits, onset_targets) + \
                   frame_loss_fn(frame_logits, frame_targets) + \
                   nn.functional.mse_loss(vel_pred * frame_targets, vel_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        # Validate
        val_f1 = validate(model, val_loader, device)
        print(f"Phase 1 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_guitar_model_phase1.pt')

    return best_val_f1

def train_phase2(model, train_loader, val_loader, device, epochs=80):
    """Phase 2: Unfreeze all layers, train full model with low LR."""
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    onset_loss_fn = FocalBCEWithLogitsLoss(pos_weight=ONSET_POS_WEIGHT, gamma=FOCAL_GAMMA)
    frame_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(FRAME_POS_WEIGHT).to(device)
    )

    best_val_f1 = 0.0
    collapse_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            mel, onset_targets, frame_targets, vel_targets = [b.to(device) for b in batch]

            onset_logits, frame_logits, vel_pred = model(mel)

            loss = onset_loss_fn(onset_logits, onset_targets) + \
                   frame_loss_fn(frame_logits, frame_targets) + \
                   nn.functional.mse_loss(vel_pred * frame_targets, vel_targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()

        # Validate
        val_f1 = validate(model, val_loader, device)
        print(f"Phase 2 Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Collapse detection
        if val_f1 < 0.01:
            collapse_counter += 1
            if collapse_counter >= 10:
                print("WARNING: Model collapsed to all-zeros. Stopping.")
                break
        else:
            collapse_counter = 0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), 'best_guitar_model.pt')
            print(f"  -> New best! Saved checkpoint.")

    return best_val_f1

def validate(model, val_loader, device):
    """Compute onset F1 on validation set using mir_eval."""
    import mir_eval
    model.eval()
    all_f1 = []

    with torch.no_grad():
        for batch in val_loader:
            mel, onset_targets, _, _ = [b.to(device) for b in batch]
            onset_logits, _, _ = model(mel)
            onset_probs = torch.sigmoid(onset_logits)

            # Per-sample F1
            for i in range(onset_probs.shape[0]):
                pred = (onset_probs[i] > 0.5).cpu().numpy()
                ref = onset_targets[i].cpu().numpy()

                # Convert to event lists for mir_eval
                pred_times = np.where(pred.any(axis=1))[0] * HOP_LENGTH / SAMPLE_RATE
                ref_times = np.where(ref.any(axis=1))[0] * HOP_LENGTH / SAMPLE_RATE

                if len(ref_times) > 0:
                    f1 = mir_eval.onset.f_measure(
                        ref_times, pred_times, window=0.05
                    )[2]  # F-measure
                    all_f1.append(f1)

    return np.mean(all_f1) if all_f1 else 0.0
```

### Step 2.5: GuitarSet Train/Val Split

Use **leave-one-player-out** cross-validation (standard for GuitarSet):
- 6 guitarists total
- Train on 5, validate on 1
- Report average across all 6 folds

```
Players: ['00', '01', '02', '03', '04', '05']
Fold 0: val=00, train=[01,02,03,04,05]
Fold 1: val=01, train=[00,02,03,04,05]
...
```

For a single training run (not full CV), use player '05' as validation.

### Step 2.6: Expected Results

| Stage | Onset F1 | Notes |
|-------|----------|-------|
| Basic Pitch (current) | 79% | No training needed |
| Kong + Phase 1 only | ~82% | 20 epochs, 1-2 hours |
| Kong + Phase 2 (GuitarSet only) | ~87% | 100 epochs, 4-8 hours |
| Kong + Phase 2 (GuitarSet + GAPS) | ~91% | If GAPS audio available |

---

## 3. Bass Model Training

### Step 3.1: Dataset - Slakh2100

The existing `train_bass_runpod.py` script already handles Slakh2100 download and bass stem extraction.

```bash
# Slakh2100: 2100 multi-track songs, ~145 hours total
# Bass stems: Programs 32-39 (MIDI), rendered audio
# Download from Zenodo (large: ~90GB full, but only need bass subset)
mkdir -p datasets/slakh2100
# The train_bass_runpod.py script handles this automatically
```

### Step 3.2: Use Existing Training Script

The v2 bass training script at `stemscribe/train_bass_model/train_bass_runpod.py` already includes all the fixes from the v1 failure analysis:

- FocalBCEWithLogitsLoss (pos_weight=50.0, gamma=2.0) for onsets
- BCEWithLogitsLoss (pos_weight=10.0) for frames
- BiLSTM architecture with onset conditioning on frame head
- Collapse detection (recall < 0.01 for 10 epochs)
- Augmentation: gain jitter, Gaussian noise

**Run on RunPod (recommended for bass due to large dataset):**
```bash
cd stemscribe/train_bass_model
python train_bass_runpod.py
# Outputs: best_bass_model.pt
```

**Run on Colab:**
Use `Bass_Transcription_Training_v2.ipynb` with Google Drive checkpointing.

### Step 3.3: Additional Improvement - Kong Pre-training for Bass

Apply the same domain adaptation strategy as guitar:

```python
# Same approach: load Kong piano -> modify output heads for bass range
# Bass MIDI range: E1 (28) to G4 (67) = 40 pitches
# 4-string standard tuning: [28, 33, 38, 43] (E1, A1, D2, G2)
BASS_MIDI_RANGE = (28, 68)  # 40 pitches
```

This should improve bass F1 from the current v2 results by providing strong pre-trained features.

---

## 4. Colab vs RunPod

### Colab Pro (Recommended for Guitar)
- **Cost:** ~$10/month for Pro
- **GPU:** A100 40GB (Pro) or T4 16GB (free tier)
- **Session limit:** 24 hours (Pro), 12 hours (free)
- **Pros:** Easy setup, Google Drive checkpointing, free tier available
- **Cons:** Session disconnects, need checkpointing discipline
- **Verdict:** Good for guitar (4-8h training fits in one session)

**Checkpointing strategy for Colab:**
```python
# Save to Google Drive every 5 epochs
import shutil
DRIVE_CHECKPOINT = '/content/drive/MyDrive/stemscribe_checkpoints/'
if epoch % 5 == 0:
    shutil.copy('best_guitar_model.pt', DRIVE_CHECKPOINT)
    print(f"Checkpoint saved to Drive")
```

### RunPod (Recommended for Bass)
- **Cost:** ~$0.80/hr for A100 40GB
- **Session limit:** None (persistent)
- **Pros:** No disconnects, persistent storage, multiple GPUs available
- **Cons:** Costs real money, setup overhead
- **Verdict:** Better for bass (Slakh2100 is large, training may take 12-20h)

### Decision Matrix

| Factor | Guitar Training | Bass Training |
|--------|----------------|---------------|
| Dataset size | 3h (GuitarSet) | 20-50h (Slakh2100 bass) |
| Training time | 4-8h (A100) | 12-20h (A100) |
| Recommended | Colab Pro | RunPod |
| Fallback | RunPod | Colab Pro + checkpointing |

---

## 5. Integration into StemScribe

### Step 5.1: Create `guitar_nn_transcriber.py`

Follow the pattern established by `drum_nn_transcriber.py`:

```
backend/
  guitar_nn_transcriber.py    # NEW - neural guitar transcription
  guitar_tab_transcriber.py   # EXISTING - Basic Pitch fallback
  drum_nn_transcriber.py      # REFERENCE - pattern to follow
  models/
    pretrained/
      best_tab_model.pt       # 59MB - current (untrained well)
      best_guitar_model.pt    # NEW - from training sprint
      best_bass_model.pt      # 119MB - retrain with Kong init
      best_drum_model.pt      # 115MB - working
      best_piano_model.pt     # 145MB - working
```

**Key integration pattern (from drum_nn_transcriber.py):**

```python
# guitar_nn_transcriber.py skeleton

CHECKPOINT_PATH = Path(__file__).parent / 'models' / 'pretrained' / 'best_guitar_model.pt'
MODEL_AVAILABLE = CHECKPOINT_PATH.exists()

# Constants MUST match training exactly
SAMPLE_RATE = 16000
HOP_LENGTH = 160      # Match Kong checkpoint
N_MELS = 229          # Match Kong checkpoint
GUITAR_MIDI_RANGE = (40, 88)  # E2 to E6, 48 pitches

class GuitarTranscriptionModel(nn.Module):
    """Must match training architecture exactly for checkpoint loading."""
    # ... CNN -> BiGRU -> onset/frame/velocity heads
    pass

class GuitarNNTranscriber:
    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if MODEL_AVAILABLE:
            self._load_model()

    def _load_model(self):
        self.model = GuitarTranscriptionModel().to(self.device)
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def transcribe(self, audio_path: str) -> List[dict]:
        """Returns list of {onset, offset, midi_pitch, velocity, string, fret}."""
        if self.model is None:
            # Fallback to Basic Pitch
            from guitar_tab_transcriber import GuitarTabTranscriber
            return GuitarTabTranscriber().transcribe(audio_path)

        # Neural transcription with post-processing
        # ... chunk audio, run model, decode onsets, assign strings/frets
        pass
```

### Step 5.2: Wire Up in `app.py`

```python
# In app.py, add preference for neural model with Basic Pitch fallback
from guitar_nn_transcriber import GuitarNNTranscriber, MODEL_AVAILABLE as GUITAR_NN_AVAILABLE

if GUITAR_NN_AVAILABLE:
    guitar_transcriber = GuitarNNTranscriber()
    logger.info("Using neural guitar transcriber")
else:
    guitar_transcriber = GuitarTabTranscriber()  # existing Basic Pitch
    logger.info("Neural guitar model not found, using Basic Pitch fallback")
```

### Step 5.3: Update `model_manager.py`

The model registry already has entries for `guitar_tab` and `bass_nn`. After training, the new checkpoint replaces `best_tab_model.pt` (or add a new entry `guitar_nn` pointing to `best_guitar_model.pt`).

### Step 5.4: Post-processing Pipeline

The neural model outputs raw onset/frame/velocity matrices. Apply post-processing from the existing `guitar_tab_transcriber.py`:

1. **Guitar range filter:** E2 (40) to E6 (88) MIDI
2. **Onset peak picking:** Local maxima in onset probability, threshold 0.5
3. **String/fret assignment:** Minimize fret distance, respect 6-string polyphony limit
4. **Short note cleanup:** Remove notes < 30ms
5. **Velocity quantization:** Map to 0-127 MIDI velocity

---

## 6. Validation & Testing

### Metrics
- **Primary:** Onset F1 at 50ms tolerance (mir_eval standard)
- **Secondary:** Note F1 (onset + offset + pitch), Frame F1
- **Practical:** A/B listening test vs Basic Pitch output

### GuitarSet Evaluation Script
```bash
python evaluate_guitar.py --checkpoint best_guitar_model.pt --dataset datasets/guitarset/
# Expected output:
#   Onset F1: 0.87 (vs 0.79 Basic Pitch)
#   Note F1: 0.72 (vs 0.61 Basic Pitch)
#   Frame F1: 0.81 (vs 0.70 Basic Pitch)
```

### Regression Tests
After integration, verify:
1. Guitar transcription produces valid MIDI output
2. String/fret assignments are physically playable
3. No regression in drum transcription (independent model)
4. Fallback to Basic Pitch works when checkpoint is missing
5. Memory usage < 2GB for inference

---

## Quick Start Checklist

- [ ] Download Kong piano checkpoint (172MB)
- [ ] Download GuitarSet from Zenodo (~1.5GB)
- [ ] Set up Colab Pro notebook with Drive checkpointing
- [ ] Run Phase 1 training (20 epochs, ~2 hours)
- [ ] Run Phase 2 training (80 epochs, ~6 hours)
- [ ] Evaluate: target onset F1 > 0.85
- [ ] Copy `best_guitar_model.pt` to `backend/models/pretrained/`
- [ ] Create `guitar_nn_transcriber.py` following drum_nn pattern
- [ ] Wire up in `app.py` with Basic Pitch fallback
- [ ] Run bass v2 training on RunPod with Slakh2100
- [ ] Copy `best_bass_model.pt` to `backend/models/pretrained/`
- [ ] Wire up bass neural model in `bass_transcriber.py`
- [ ] Run regression tests
- [ ] A/B test vs Basic Pitch on real songs
