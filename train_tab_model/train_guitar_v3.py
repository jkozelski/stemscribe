#!/usr/bin/env python3
"""
Guitar Transcription v3 Training — Transfer Learning from Piano CNN
====================================================================
Uses frozen piano CNN features + adapter + self-attention + BiLSTM to transcribe
guitar audio to flat pitch vectors (49 keys: E2=40 to E6=88).

Key changes from v1/v2:
  - Piano CNN (frozen) provides pre-trained mel features
  - 49-key flat pitch output (not 6x20 string/fret)
  - Focal BCE (pos_weight=20, not 500) + Soft Dice loss
  - Curriculum learning: monophonic -> low polyphony -> full
  - 3-phase training: adapter warmup, full training, CNN fine-tuning
  - Augmentation: gain, noise, pitch shift

Architecture:
  Mel(229) -> Piano CNN (frozen) -> Adapter(1792->1024) + Residual
  -> MultiheadAttention(1024, 8 heads)
  -> onset_lstm BiLSTM(1024, 192) -> onset_head Linear(384, 49)
  -> frame_lstm BiLSTM(1024+49, 192) -> frame_head Linear(384, 49)
  -> velocity_head Linear(384, 49) + Sigmoid

Spectrogram params MUST match piano: sr=16000, hop=256, n_mels=229, n_fft=2048, fmin=30, fmax=8000

Run on RunPod:
    pip install torch librosa jams tqdm
    python train_guitar_v3.py

Output: /workspace/guitar_v3_results/best_guitar_v3_model.pt
"""

import os
import sys
import json
import time
import random
import subprocess
import shutil
from pathlib import Path

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

def preflight():
    print("=" * 60)
    print("PRE-FLIGHT CHECKS — Guitar v3 Training")
    print("=" * 60)
    errors = []

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [OK] GPU: {name} ({mem:.1f} GB)")
            if mem < 6:
                errors.append(f"GPU VRAM too low: {mem:.1f} GB (need 6+)")
        else:
            errors.append("No GPU detected — training requires GPU")
    except ImportError:
        errors.append("PyTorch not installed")

    # Disk
    try:
        total, used, free = shutil.disk_usage("/workspace")
        free_gb = free / 1e9
        print(f"  [OK] Disk: {free_gb:.1f} GB free")
        if free_gb < 2:
            errors.append(f"Need at least 2 GB free, have {free_gb:.1f} GB")
    except Exception:
        pass

    # Deps
    deps = ["librosa", "jams", "tqdm", "soundfile"]
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing.append(dep)
    if missing:
        print(f"  [FIX] Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
    else:
        print("  [OK] Python deps present")

    # Piano checkpoint
    piano_ckpt = find_piano_checkpoint()
    if piano_ckpt:
        print(f"  [OK] Piano checkpoint: {piano_ckpt} ({piano_ckpt.stat().st_size / 1e6:.0f} MB)")
    else:
        errors.append("Piano checkpoint not found")

    # GuitarSet
    gs_dir = Path("/workspace/guitarset")
    if gs_dir.exists():
        import glob
        wavs = glob.glob(f"{gs_dir}/**/*.wav", recursive=True)
        print(f"  [OK] GuitarSet: {len(wavs)} WAV files")
        if len(wavs) < 10:
            errors.append("GuitarSet has too few files. Run prepare_guitar_data.py first.")
    else:
        errors.append("GuitarSet not found. Run prepare_guitar_data.py first.")

    print("=" * 60)
    if errors:
        for e in errors:
            print(f"  FATAL: {e}")
        print("PRE-FLIGHT: FAILED")
        sys.exit(1)
    else:
        print("PRE-FLIGHT: All checks passed")
    print("=" * 60)
    print()


def find_piano_checkpoint():
    for path in [
        Path("/workspace/best_piano_model.pt"),
        Path("/workspace/piano_model_results/best_piano_model.pt"),
        Path("backend/models/pretrained/best_piano_model.pt"),
    ]:
        if path.exists():
            return path
    return None


preflight()


# ============================================================================
# IMPORTS
# ============================================================================

import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import jams
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG — Mel params MUST match piano EXACTLY
# ============================================================================

SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_MELS = 229
N_FFT = 2048
FMIN = 30.0
FMAX = 8000.0
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)

# Guitar pitch range
GUITAR_MIN_MIDI = 40  # E2
GUITAR_MAX_MIDI = 88  # E6
NUM_KEYS = 49

# Training hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 60
ONSET_POS_WEIGHT = 20.0
FRAME_POS_WEIGHT = 5.0
FOCAL_GAMMA = 2.0
DICE_WEIGHT = 0.3

# Phase schedule
PHASE1_END = 10   # Adapter warmup
PHASE2_END = 40   # Full training
PHASE3_END = 60   # CNN fine-tuning

# Curriculum schedule
CURRICULUM_A_END = 15  # Monophonic (max_poly <= 2)
CURRICULUM_B_END = 35  # Low polyphony (max_poly <= 4)
# After 35: full polyphony

# Paths
GUITARSET_DIR = Path("/workspace/guitarset")
SAVE_DIR = Path("/workspace/guitar_v3_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
PIANO_CKPT = find_piano_checkpoint()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# PIANO MODEL (duplicated for RunPod portability)
# ============================================================================

class PianoTranscriptionModel(nn.Module):
    """Exact copy of piano_transcriber.py for loading checkpoint."""

    def __init__(self, n_mels=229, num_keys=88,
                 hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_keys = num_keys

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(96, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
        )

        cnn_output_dim = 128 * 14

        self.onset_lstm = nn.LSTM(
            input_size=cnn_output_dim, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.frame_lstm = nn.LSTM(
            input_size=cnn_output_dim + num_keys, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_size * 2
        self.onset_head = nn.Linear(lstm_output_dim, num_keys)
        self.frame_head = nn.Linear(lstm_output_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, num_keys), nn.Sigmoid(),
        )

    def forward(self, x):
        batch, n_mels, time = x.shape
        cnn_out = self.cnn(x.unsqueeze(1))
        cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)
        onset_out, _ = self.onset_lstm(cnn_features)
        onset_logits = self.onset_head(onset_out)
        onset_pred = torch.sigmoid(onset_logits)
        frame_input = torch.cat([cnn_features, onset_pred.detach()], dim=-1)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)
        frame_pred = torch.sigmoid(frame_logits)
        velocity_pred = self.velocity_head(onset_out)
        return (
            onset_pred.permute(0, 2, 1),
            frame_pred.permute(0, 2, 1),
            velocity_pred.permute(0, 2, 1),
        )


# ============================================================================
# V3 GUITAR MODEL
# ============================================================================

class GuitarTranscriptionModel_v3(nn.Module):
    """
    Transfer-learned guitar transcriber.

    Piano CNN (frozen) -> Adapter -> Self-Attention -> Guitar LSTM -> Pitch heads (49 keys)

    Input:  Mel spectrogram (B, 229, T)
    Output: onset_logits    (B, 49, T)
            frame_logits    (B, 49, T)
            velocity_pred   (B, 49, T)
    """

    def __init__(self, num_keys=49, hidden_size=192, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_keys = num_keys

        # === FROZEN: Piano CNN feature extractor ===
        # Loaded separately via load_piano_cnn()
        self.cnn = None  # Set externally

        # === TRAINABLE: Domain adaptation layer ===
        self.adapter = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adapter_residual = nn.Linear(1792, 1024)

        # === TRAINABLE: Multi-head self-attention ===
        self.attention = nn.MultiheadAttention(
            embed_dim=1024, num_heads=8, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(1024)

        # === TRAINABLE: Onset and Frame LSTMs ===
        self.onset_lstm = nn.LSTM(
            input_size=1024, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout,
        )
        self.frame_lstm = nn.LSTM(
            input_size=1024 + num_keys, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout,
        )

        lstm_dim = hidden_size * 2  # 384

        # === TRAINABLE: Output heads ===
        self.onset_head = nn.Linear(lstm_dim, num_keys)
        self.frame_head = nn.Linear(lstm_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_dim, num_keys),
            nn.Sigmoid(),
        )

        # Initialize onset head bias for sparse targets
        nn.init.constant_(self.onset_head.bias, -3.0)
        nn.init.constant_(self.frame_head.bias, -2.0)

    def forward(self, mel):
        batch, n_mels, time = mel.shape

        # Frozen CNN features
        with torch.no_grad():
            cnn_out = self.cnn(mel.unsqueeze(1))  # (B, 128, 14, T)

        # Flatten CNN output
        cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)  # (B, T, 1792)

        # Domain adapter with residual
        adapted = self.adapter(cnn_features) + self.adapter_residual(cnn_features)  # (B, T, 1024)

        # Self-attention for harmonic context
        attn_out, _ = self.attention(adapted, adapted, adapted)
        adapted = self.attn_norm(adapted + attn_out)  # (B, T, 1024)

        # Onset prediction
        onset_out, _ = self.onset_lstm(adapted)  # (B, T, 384)
        onset_logits = self.onset_head(onset_out)  # (B, T, 49)
        onset_pred = torch.sigmoid(onset_logits)

        # Frame prediction (onset-conditioned)
        frame_input = torch.cat([adapted, onset_pred.detach()], dim=-1)  # (B, T, 1073)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)  # (B, T, 49)

        # Velocity
        velocity_pred = self.velocity_head(onset_out)  # (B, T, 49)

        return (
            onset_logits.permute(0, 2, 1),   # (B, 49, T) — raw logits
            frame_logits.permute(0, 2, 1),   # (B, 49, T) — raw logits
            velocity_pred.permute(0, 2, 1),  # (B, 49, T) — [0, 1]
        )


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss + pos_weight for sparse binary classification."""

    def __init__(self, pos_weight=20.0, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class SoftDiceLoss(nn.Module):
    """Soft Dice Loss — class-imbalance invariant."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class v3OnsetLoss(nn.Module):
    """Combined Focal BCE + Soft Dice for onset detection."""

    def __init__(self, pos_weight=20.0, gamma=2.0, dice_weight=0.3):
        super().__init__()
        self.focal = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=gamma)
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal = self.focal(logits, targets)
        dice = self.dice(logits, targets)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice


# ============================================================================
# METRICS
# ============================================================================

def compute_onset_f1(logits, targets, threshold=0.5):
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.item(), precision.item(), recall.item()


# ============================================================================
# DATASET
# ============================================================================

def extract_track_id(wav_name):
    stem = wav_name
    for suffix in ["_mic", "_mix", "_hex_cln", "_hex"]:
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
            break
    for prefix in [
        "audio_mono-mic_", "audio_mono-mic-",
        "audio_mono-pickup_mix_", "audio_mono-pickup_mix-",
        "audio_hex-pickup_original_", "audio_hex-pickup_original-",
        "audio_hex-pickup_debleeded_", "audio_hex-pickup_debleeded-",
    ]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem


class GuitarV3Dataset(Dataset):
    """
    GuitarSet with piano-matching mel spectrograms and flat pitch targets.
    Supports curriculum filtering by max polyphony.
    """

    def __init__(self, data_dir, split="train", augment=False, max_polyphony_filter=None):
        self.augment = augment
        self.max_poly_filter = max_polyphony_filter

        data_dir = Path(data_dir)
        wav_files = sorted(data_dir.rglob("*.wav"))
        jams_files = sorted(data_dir.rglob("*.jams"))

        jams_lookup = {}
        for jp in jams_files:
            jams_lookup[jp.stem] = jp
            jams_lookup[jp.stem.lower()] = jp

        all_samples = []
        for wav_path in wav_files:
            track_id = extract_track_id(wav_path.stem)
            jams_path = jams_lookup.get(track_id) or jams_lookup.get(track_id.lower())
            if jams_path is not None:
                all_samples.append({"audio": str(wav_path), "jams": str(jams_path)})

        np.random.seed(42)
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)

        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        # Pre-compute polyphony info for curriculum filtering
        if max_polyphony_filter is not None:
            self._precompute_polyphony()

        print(f"  {split}: {len(self.samples)} samples" +
              (f" (max_poly<={max_polyphony_filter})" if max_polyphony_filter else ""))

    def _precompute_polyphony(self):
        """Tag each sample with its max polyphony for curriculum filtering."""
        filtered = []
        for sample in self.samples:
            try:
                jam = jams.load(sample["jams"])
                notes = []
                for annot in jam.annotations:
                    if annot.namespace == "note_midi":
                        for obs in annot.data:
                            notes.append((obs.time, obs.time + obs.duration))
                if not notes:
                    max_poly = 0
                else:
                    events = []
                    for onset, offset in notes:
                        events.append((onset, 1))
                        events.append((offset, -1))
                    events.sort()
                    current = 0
                    max_poly = 0
                    for _, delta in events:
                        current += delta
                        max_poly = max(max_poly, current)
                sample["max_poly"] = max_poly
                if max_poly <= self.max_poly_filter:
                    filtered.append(sample)
            except Exception:
                filtered.append(sample)
        self.samples = filtered

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        audio, sr = librosa.load(sample["audio"], sr=SAMPLE_RATE, mono=True)

        # Random chunk
        chunk_samples = CHUNK_SAMPLES
        if len(audio) > chunk_samples:
            start = np.random.randint(0, len(audio) - chunk_samples)
            audio = audio[start:start + chunk_samples]
            start_time = start / SAMPLE_RATE
        else:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))
            start_time = 0.0

        end_time = start_time + CHUNK_DURATION

        # Augmentation
        if self.augment:
            if random.random() < 0.5:
                audio = audio * random.uniform(0.7, 1.3)
            if random.random() < 0.3:
                noise = np.random.randn(len(audio)).astype(np.float32) * random.uniform(0.001, 0.005)
                audio = audio + noise

        # Mel spectrogram (MUST match piano)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        )
        mel = np.log(mel + 1e-8).astype(np.float32)
        num_frames = mel.shape[1]

        # Flat pitch targets
        onset_target = np.zeros((NUM_KEYS, num_frames), dtype=np.float32)
        frame_target = np.zeros((NUM_KEYS, num_frames), dtype=np.float32)
        vel_target = np.zeros((NUM_KEYS, num_frames), dtype=np.float32)

        frames_per_sec = SAMPLE_RATE / HOP_LENGTH

        try:
            jam = jams.load(sample["jams"])
            for annot in jam.annotations:
                if annot.namespace != "note_midi":
                    continue
                for obs in annot.data:
                    onset = obs.time
                    offset = obs.time + obs.duration
                    midi_note = int(round(obs.value))

                    if midi_note < GUITAR_MIN_MIDI or midi_note > GUITAR_MAX_MIDI:
                        continue
                    if offset <= start_time or onset >= end_time:
                        continue

                    key_idx = midi_note - GUITAR_MIN_MIDI

                    onset_frame = int((onset - start_time) * frames_per_sec)
                    offset_frame = int((offset - start_time) * frames_per_sec)

                    if 0 <= onset_frame < num_frames:
                        onset_target[key_idx, onset_frame] = 1.0
                        vel_target[key_idx, onset_frame] = 0.8  # Default velocity

                    for f in range(max(0, onset_frame), min(num_frames, offset_frame)):
                        frame_target[key_idx, f] = 1.0
        except Exception:
            pass

        # Pad/truncate to fixed frames
        if num_frames < CHUNK_FRAMES:
            pad = CHUNK_FRAMES - num_frames
            mel = np.pad(mel, ((0, 0), (0, pad)))
            onset_target = np.pad(onset_target, ((0, 0), (0, pad)))
            frame_target = np.pad(frame_target, ((0, 0), (0, pad)))
            vel_target = np.pad(vel_target, ((0, 0), (0, pad)))
        elif num_frames > CHUNK_FRAMES:
            mel = mel[:, :CHUNK_FRAMES]
            onset_target = onset_target[:, :CHUNK_FRAMES]
            frame_target = frame_target[:, :CHUNK_FRAMES]
            vel_target = vel_target[:, :CHUNK_FRAMES]

        return (
            torch.from_numpy(mel),
            torch.from_numpy(onset_target),
            torch.from_numpy(frame_target),
            torch.from_numpy(vel_target),
        )


# ============================================================================
# CNN LOADING & FREEZING
# ============================================================================

def load_piano_cnn(checkpoint_path, device):
    """Load piano model, extract CNN, freeze all params."""
    print(f"Loading piano CNN from {checkpoint_path}...")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    piano_model = PianoTranscriptionModel()
    piano_model.load_state_dict(checkpoint["model_state_dict"])
    cnn = piano_model.cnn

    for param in cnn.parameters():
        param.requires_grad = False
    cnn.eval()

    print(f"  Piano CNN loaded (epoch={checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?')})")
    print(f"  {sum(p.numel() for p in cnn.parameters()):,} params (all frozen)")
    return cnn


def unfreeze_cnn_top_blocks(model):
    """Unfreeze CNN blocks 3-4 (last 2 conv blocks) for Phase 3 fine-tuning."""
    # CNN structure: [Conv, BN, ReLU, Conv, BN, ReLU, Pool, Drop,  <- block 1-2 (stay frozen)
    #                 Conv, BN, ReLU, Pool, Drop,                   <- block 3
    #                 Conv, BN, ReLU, Pool, Drop,                   <- block 4
    #                 Conv, BN, ReLU, Pool, Drop,                   <- block 5
    #                 Conv, BN, ReLU, Pool, Drop]                   <- block 6

    # The Sequential has indices 0-19 (20 layers)
    # Block 1-2: indices 0-7   (keep frozen)
    # Block 3:   indices 8-12  (unfreeze)
    # Block 4:   indices 13-17 (unfreeze)
    # Block 5:   indices 18-22 if exists (unfreeze)

    cnn = model.cnn
    total_layers = len(list(cnn.children()))

    # Unfreeze the last ~10 layers (blocks 3-5)
    freeze_boundary = 8  # First 8 layers stay frozen
    unfrozen = 0
    for i, layer in enumerate(cnn.children()):
        if i >= freeze_boundary:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen += 1

    print(f"  Unfroze {unfrozen} CNN parameters (blocks 3+)")
    return unfrozen


# ============================================================================
# COSINE ANNEALING WITH WARMUP
# ============================================================================

class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            factor = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)


# ============================================================================
# TRAINING & VALIDATION
# ============================================================================

def train_epoch(model, loader, optimizer, onset_loss_fn, frame_loss_fn, vel_loss_fn, device):
    model.train()
    # Keep CNN in eval mode for frozen BN
    if hasattr(model, "cnn") and model.cnn is not None:
        model.cnn.eval()

    total_loss = 0.0
    num_batches = 0
    epoch_onset_logits = []
    epoch_onset_targets = []

    for mel_b, onset_tgt, frame_tgt, vel_tgt in tqdm(loader, desc="  Train", leave=False):
        mel_b = mel_b.to(device)
        onset_tgt = onset_tgt.to(device)
        frame_tgt = frame_tgt.to(device)
        vel_tgt = vel_tgt.to(device)

        onset_logits, frame_logits, vel_pred = model(mel_b)

        loss_onset = onset_loss_fn(onset_logits, onset_tgt)
        loss_frame = frame_loss_fn(frame_logits, frame_tgt)

        onset_mask = onset_tgt > 0.5
        if onset_mask.any():
            loss_vel = vel_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask])
        else:
            loss_vel = torch.tensor(0.0, device=device)

        loss = loss_onset + loss_frame + 0.5 * loss_vel

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 3 == 0:
            epoch_onset_logits.append(onset_logits.detach().cpu())
            epoch_onset_targets.append(onset_tgt.detach().cpu())

    avg_loss = total_loss / max(num_batches, 1)
    if epoch_onset_logits:
        f1, prec, rec = compute_onset_f1(
            torch.cat(epoch_onset_logits), torch.cat(epoch_onset_targets))
    else:
        f1, prec, rec = 0, 0, 0

    return avg_loss, f1, prec, rec


def validate(model, loader, onset_loss_fn, frame_loss_fn, vel_loss_fn, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_onset_logits = []
    all_onset_targets = []

    with torch.no_grad():
        for mel_b, onset_tgt, frame_tgt, vel_tgt in loader:
            mel_b = mel_b.to(device)
            onset_tgt = onset_tgt.to(device)
            frame_tgt = frame_tgt.to(device)
            vel_tgt = vel_tgt.to(device)

            onset_logits, frame_logits, vel_pred = model(mel_b)

            loss_onset = onset_loss_fn(onset_logits, onset_tgt)
            loss_frame = frame_loss_fn(frame_logits, frame_tgt)

            onset_mask = onset_tgt > 0.5
            if onset_mask.any():
                loss_vel = vel_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask])
            else:
                loss_vel = torch.tensor(0.0, device=device)

            total_loss += (loss_onset + loss_frame + 0.5 * loss_vel).item()
            num_batches += 1

            all_onset_logits.append(onset_logits.cpu())
            all_onset_targets.append(onset_tgt.cpu())

    avg_loss = total_loss / max(num_batches, 1)
    f1, prec, rec = compute_onset_f1(
        torch.cat(all_onset_logits), torch.cat(all_onset_targets))

    return avg_loss, f1, prec, rec


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Guitar Transcription v3 — Transfer Learning Training")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print()

    # Load frozen piano CNN
    frozen_cnn = load_piano_cnn(PIANO_CKPT, device)

    # Build v3 model
    model = GuitarTranscriptionModel_v3(num_keys=NUM_KEYS).to(device)
    model.cnn = frozen_cnn.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total, {trainable_params:,} trainable")

    # Verify architecture
    dummy = torch.randn(1, N_MELS, CHUNK_FRAMES).to(device)
    o, f, v = model(dummy)
    assert o.shape == (1, NUM_KEYS, CHUNK_FRAMES), f"Onset shape: {o.shape}"
    assert f.shape == (1, NUM_KEYS, CHUNK_FRAMES), f"Frame shape: {f.shape}"
    print(f"Architecture OK: ({N_MELS}, {CHUNK_FRAMES}) -> ({NUM_KEYS}, {CHUNK_FRAMES})")
    del dummy, o, f, v
    torch.cuda.empty_cache()

    # Optimizer with parameter groups (different LR per component)
    param_groups = [
        {"params": model.adapter.parameters(), "lr": 3e-4, "name": "adapter"},
        {"params": model.adapter_residual.parameters(), "lr": 3e-4, "name": "adapter_res"},
        {"params": model.attention.parameters(), "lr": 3e-4, "name": "attention"},
        {"params": model.attn_norm.parameters(), "lr": 3e-4, "name": "attn_norm"},
        {"params": model.onset_lstm.parameters(), "lr": 1e-4, "name": "onset_lstm"},
        {"params": model.frame_lstm.parameters(), "lr": 1e-4, "name": "frame_lstm"},
        {"params": model.onset_head.parameters(), "lr": 3e-4, "name": "onset_head"},
        {"params": model.frame_head.parameters(), "lr": 3e-4, "name": "frame_head"},
        {"params": model.velocity_head.parameters(), "lr": 3e-4, "name": "vel_head"},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Loss functions
    onset_loss_fn = v3OnsetLoss(pos_weight=ONSET_POS_WEIGHT, gamma=FOCAL_GAMMA, dice_weight=DICE_WEIGHT)
    frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([FRAME_POS_WEIGHT]).to(device))
    vel_loss_fn = nn.MSELoss()

    print(f"\nOnset loss: Focal(pw={ONSET_POS_WEIGHT}, gamma={FOCAL_GAMMA}) + Dice(w={DICE_WEIGHT})")
    print(f"Frame loss: BCE(pw={FRAME_POS_WEIGHT})")
    print(f"Phases: warmup(1-{PHASE1_END}), full({PHASE1_END+1}-{PHASE2_END}), "
          f"CNN fine-tune({PHASE2_END+1}-{PHASE3_END})")
    print(f"Curriculum: mono(1-{CURRICULUM_A_END}), low-poly({CURRICULUM_A_END+1}-{CURRICULUM_B_END}), "
          f"full({CURRICULUM_B_END+1}+)")

    # Scheduler
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, total_epochs=NUM_EPOCHS)

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    ckpt_path = SAVE_DIR / "latest_checkpoint.pt"

    if ckpt_path.exists():
        print(f"\nResuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, best F1={best_val_f1:.4f}")

    # Pre-build val loader (always full polyphony for consistent evaluation)
    print("\nBuilding validation set...")
    val_dataset = GuitarV3Dataset(GUITARSET_DIR, split="val", augment=False)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # Collapse detection state
    low_recall_count = 0
    high_recall_low_prec_count = 0
    COLLAPSE_THRESHOLD = 5

    # Training history
    history = []

    print(f"\nStarting training: epoch {start_epoch+1} -> {NUM_EPOCHS}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        # ---- Curriculum: select training data ----
        if epoch < CURRICULUM_A_END:
            max_poly = 2
            stage = "A (mono)"
        elif epoch < CURRICULUM_B_END:
            max_poly = 4
            stage = "B (low-poly)"
        else:
            max_poly = None
            stage = "C (full)"

        # Rebuild train loader for curriculum stage changes
        if epoch == start_epoch or epoch == CURRICULUM_A_END or epoch == CURRICULUM_B_END:
            print(f"  Building train set: stage {stage}...")
            train_dataset = GuitarV3Dataset(
                GUITARSET_DIR, split="train", augment=True,
                max_polyphony_filter=max_poly,
            )
            train_loader = DataLoader(
                train_dataset, batch_size=BATCH_SIZE,
                shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
            )

        # ---- Phase 3: Unfreeze CNN top blocks ----
        if epoch == PHASE2_END and best_val_f1 > 0.3:
            print(f"\n  Phase 3: Unfreezing CNN blocks 3-4 (best F1={best_val_f1:.4f} > 0.3)")
            unfreeze_cnn_top_blocks(model)
            # Add CNN params to optimizer with low LR
            cnn_params = [p for p in model.cnn.parameters() if p.requires_grad]
            if cnn_params:
                optimizer.add_param_group({"params": cnn_params, "lr": 1e-5, "name": "cnn_finetune"})
            print(f"  Total trainable now: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        elif epoch == PHASE2_END:
            print(f"\n  Skipping Phase 3: best F1={best_val_f1:.4f} < 0.3 threshold")

        # ---- Update learning rate ----
        scheduler.step(epoch)

        # ---- Train ----
        train_loss, train_f1, train_prec, train_rec = train_epoch(
            model, train_loader, optimizer, onset_loss_fn, frame_loss_fn, vel_loss_fn, device
        )

        # ---- Validate ----
        val_loss, val_f1, val_prec, val_rec = validate(
            model, val_loader, onset_loss_fn, frame_loss_fn, vel_loss_fn, device
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Report
        phase = "Phase1" if epoch < PHASE1_END else ("Phase2" if epoch < PHASE2_END else "Phase3")
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.0f}s) {phase}/{stage} LR={lr:.2e}")
        print(f"  Train — Loss: {train_loss:.4f} | F1: {train_f1:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | F1: {val_f1:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f}")

        # ---- Collapse detection ----
        if val_rec < 0.02 and val_prec < 0.02:
            low_recall_count += 1
            print(f"  WARNING: Possible collapse ({low_recall_count}/{COLLAPSE_THRESHOLD})")
            if low_recall_count >= COLLAPSE_THRESHOLD:
                print(f"\n  COLLAPSED. Stopping early.")
                break
        else:
            low_recall_count = 0

        if val_rec > 0.5 and val_prec < 0.01:
            high_recall_low_prec_count += 1
            print(f"  WARNING: Over-predicting ({high_recall_low_prec_count}/{COLLAPSE_THRESHOLD})")
            if high_recall_low_prec_count >= COLLAPSE_THRESHOLD:
                print(f"\n  OVER-PREDICTING. Stopping early.")
                break
        else:
            high_recall_low_prec_count = 0

        # ---- Save best by F1 ----
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_f1": val_f1,
                "config": {
                    "sample_rate": SAMPLE_RATE,
                    "hop_length": HOP_LENGTH,
                    "n_mels": N_MELS,
                    "n_fft": N_FFT,
                    "fmin": FMIN,
                    "fmax": FMAX,
                    "num_keys": NUM_KEYS,
                    "guitar_min_midi": GUITAR_MIN_MIDI,
                    "guitar_max_midi": GUITAR_MAX_MIDI,
                    "hidden_size": 192,
                    "num_layers": 2,
                },
            }, str(SAVE_DIR / "best_guitar_v3_model.pt"))
            print(f"  * Best model saved (F1={val_f1:.4f})")

        # ---- Save latest checkpoint ----
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_f1": val_f1,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
        }, str(ckpt_path))

        history.append({
            "epoch": epoch + 1,
            "phase": phase,
            "curriculum": stage,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_f1": round(val_f1, 4),
            "val_prec": round(val_prec, 4),
            "val_rec": round(val_rec, 4),
            "lr": lr,
        })

    # Save training history
    history_path = SAVE_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best val onset F1: {best_val_f1:.4f}")
    print(f"Model: {SAVE_DIR / 'best_guitar_v3_model.pt'}")
    print(f"History: {history_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
