#!/usr/bin/env python3
"""
Bass Transcription v3 Training — Transfer Learning from Piano CNN
==================================================================
Simplified variant of guitar v3 for bass transcription.
40-key output (E1=28 to G4=67), smaller LSTM, no attention layer.

Key differences from guitar v3:
  - 40 keys (E1=28 to G4=67) instead of 49
  - hidden_size=128 (bass is mostly monophonic)
  - No multi-head attention (simpler harmonic structure)
  - 40 epochs (not 60)
  - Simpler curriculum: mono -> double stops (no full polyphony stage)
  - Slakh2100 dataset (more data than GuitarSet)

Architecture:
  Mel(229) -> Piano CNN (frozen) -> Adapter(1792->1024) + Residual
  -> onset_lstm BiLSTM(1024, 128) -> onset_head Linear(256, 40)
  -> frame_lstm BiLSTM(1024+40, 128) -> frame_head Linear(256, 40)
  -> velocity_head Linear(256, 40) + Sigmoid

Spectrogram params MUST match piano: sr=16000, hop=256, n_mels=229, n_fft=2048, fmin=30, fmax=8000

Run on RunPod:
    pip install torch librosa pretty_midi tqdm soundfile pyyaml
    python train_bass_v3.py

Output: /workspace/bass_v3_results/best_bass_v3_model.pt
"""

import os
import sys
import json
import time
import random
import math
import subprocess
import shutil
from pathlib import Path

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

def preflight():
    print("=" * 60)
    print("PRE-FLIGHT CHECKS — Bass v3 Training")
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
    except Exception:
        pass

    # Deps
    deps = ["librosa", "pretty_midi", "tqdm", "soundfile", "pyyaml"]
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import pretty_midi
import yaml
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

# Bass pitch range: E1 (28) to G4 (67) = 40 keys
BASS_MIN_MIDI = 28
BASS_MAX_MIDI = 67
NUM_KEYS = 40

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 40
ONSET_POS_WEIGHT = 20.0
FRAME_POS_WEIGHT = 5.0
FOCAL_GAMMA = 2.0
DICE_WEIGHT = 0.3

# Phase schedule
PHASE1_END = 8    # Adapter warmup
PHASE2_END = 30   # Full training
PHASE3_END = 40   # CNN fine-tuning

# Curriculum
CURRICULUM_A_END = 10  # Monophonic only
# After 10: include double stops

# Slakh bass programs (General MIDI 32-39)
BASS_PROGRAMS = set(range(32, 40))

# Paths
SLAKH_ROOT = Path("/workspace/slakh2100_flac_redux")
BASS_DATA_DIR = Path("/workspace/slakh_bass")
SAVE_DIR = Path("/workspace/bass_v3_results")
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
# V3 BASS MODEL (simplified — no attention)
# ============================================================================

class BassTranscriptionModel_v3(nn.Module):
    """
    Transfer-learned bass transcriber. Simpler than guitar v3:
    no multi-head attention, smaller hidden size.

    Piano CNN (frozen) -> Adapter(1792->1024) -> BiLSTM(128) -> 40-key output

    Input:  Mel spectrogram (B, 229, T)
    Output: onset_logits    (B, 40, T)
            frame_logits    (B, 40, T)
            velocity_pred   (B, 40, T)
    """

    def __init__(self, num_keys=40, hidden_size=128, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_keys = num_keys

        # Frozen piano CNN (set externally)
        self.cnn = None

        # Domain adapter (same as guitar v3)
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

        # No attention for bass (simpler harmonic structure)

        # Onset LSTM
        self.onset_lstm = nn.LSTM(
            input_size=1024, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )

        # Frame LSTM (onset-conditioned)
        self.frame_lstm = nn.LSTM(
            input_size=1024 + num_keys, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
        )

        lstm_dim = hidden_size * 2  # 256

        # Output heads
        self.onset_head = nn.Linear(lstm_dim, num_keys)
        self.frame_head = nn.Linear(lstm_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_dim, num_keys),
            nn.Sigmoid(),
        )

        # Sparse-target bias init
        nn.init.constant_(self.onset_head.bias, -3.0)
        nn.init.constant_(self.frame_head.bias, -2.0)

    def forward(self, mel):
        batch, n_mels, time = mel.shape

        # Frozen CNN
        with torch.no_grad():
            cnn_out = self.cnn(mel.unsqueeze(1))

        cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)

        # Adapter with residual
        adapted = self.adapter(cnn_features) + self.adapter_residual(cnn_features)

        # Onset
        onset_out, _ = self.onset_lstm(adapted)
        onset_logits = self.onset_head(onset_out)
        onset_pred = torch.sigmoid(onset_logits)

        # Frame (onset-conditioned)
        frame_input = torch.cat([adapted, onset_pred.detach()], dim=-1)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)

        # Velocity
        velocity_pred = self.velocity_head(onset_out)

        return (
            onset_logits.permute(0, 2, 1),
            frame_logits.permute(0, 2, 1),
            velocity_pred.permute(0, 2, 1),
        )


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalBCEWithLogitsLoss(nn.Module):
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
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class v3OnsetLoss(nn.Module):
    def __init__(self, pos_weight=20.0, gamma=2.0, dice_weight=0.3):
        super().__init__()
        self.focal = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=gamma)
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal = self.focal(logits, targets)
        dice = self.dice(logits, targets)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice


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
# SLAKH BASS DATA
# ============================================================================

def download_and_extract_slakh():
    """Download Slakh2100 and extract bass stems (same as train_bass_runpod.py)."""
    BASS_DATA_DIR.mkdir(exist_ok=True)

    existing = list(BASS_DATA_DIR.glob("*_Track*"))
    if len(existing) > 100:
        print(f"Found {len(existing)} existing bass tracks, skipping download")
        return

    if SLAKH_ROOT.exists() and len(list(SLAKH_ROOT.glob("**/metadata.yaml"))) > 100:
        print("Slakh already extracted, extracting bass stems...")
    else:
        print("Downloading Slakh2100-redux from Zenodo (~97 GB)...")
        tar_path = "/workspace/slakh2100.tar.gz"

        subprocess.run([
            "wget", "-q", "--show-progress", "-O", tar_path,
            "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
        ], check=True)

        print("Extracting...")
        subprocess.run(["tar", "xzf", tar_path, "-C", "/workspace/"], check=True)
        if os.path.exists(tar_path):
            os.unlink(tar_path)

    print("Extracting bass stems...")
    bass_count = 0
    for split in ["train", "validation", "test"]:
        split_dir = SLAKH_ROOT / split
        if not split_dir.exists():
            continue

        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            metadata_file = track_dir / "metadata.yaml"
            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                meta = yaml.safe_load(f)

            for stem_id, stem_info in meta.get("stems", {}).items():
                program = stem_info.get("program_num", -1)
                is_drum = stem_info.get("is_drum", False)

                if is_drum or program not in BASS_PROGRAMS:
                    continue

                audio_file = track_dir / "stems" / f"{stem_id}.flac"
                midi_file = track_dir / "MIDI" / f"{stem_id}.mid"

                if audio_file.exists() and midi_file.exists():
                    track_name = f"{split}_{track_dir.name}_{stem_id}"
                    out_dir = BASS_DATA_DIR / track_name
                    out_dir.mkdir(exist_ok=True)
                    shutil.copy2(audio_file, out_dir / "bass.flac")
                    shutil.copy2(midi_file, out_dir / "bass.mid")
                    bass_count += 1

    print(f"Extracted {bass_count} bass stems")

    if SLAKH_ROOT.exists():
        print("Cleaning up full Slakh...")
        shutil.rmtree(SLAKH_ROOT, ignore_errors=True)


def load_bass_tracks():
    tracks = []
    for track_dir in sorted(BASS_DATA_DIR.iterdir()):
        if not track_dir.is_dir():
            continue
        audio = track_dir / "bass.flac"
        midi = track_dir / "bass.mid"
        if audio.exists() and midi.exists():
            name = track_dir.name
            if name.startswith("train_"):
                split = "train"
            elif name.startswith("validation_"):
                split = "validation"
            elif name.startswith("test_"):
                split = "test"
            else:
                split = "train"
            tracks.append({
                "track": name, "split": split,
                "audio": str(audio), "midi": str(midi),
            })
    return tracks


# ============================================================================
# DATASET
# ============================================================================

def midi_to_flat_targets(midi_path, total_frames, sr=SAMPLE_RATE, hop=HOP_LENGTH):
    """Convert MIDI to flat pitch targets (40 keys: MIDI 28-67)."""
    onset = np.zeros((NUM_KEYS, total_frames), dtype=np.float32)
    frame = np.zeros((NUM_KEYS, total_frames), dtype=np.float32)
    velocity = np.zeros((NUM_KEYS, total_frames), dtype=np.float32)
    frames_per_sec = sr / hop

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return onset, frame, velocity

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            key_idx = note.pitch - BASS_MIN_MIDI
            if key_idx < 0 or key_idx >= NUM_KEYS:
                continue

            onset_frame = int(note.start * frames_per_sec)
            offset_frame = int(note.end * frames_per_sec)
            if onset_frame >= total_frames:
                continue
            offset_frame = min(offset_frame, total_frames - 1)

            onset[key_idx, onset_frame] = 1.0
            frame[key_idx, onset_frame:offset_frame + 1] = 1.0
            velocity[key_idx, onset_frame] = note.velocity / 127.0

    return onset, frame, velocity


class BassV3Dataset(Dataset):
    """Slakh bass dataset with piano-matching mel spectrograms and flat pitch targets."""

    def __init__(self, track_list, chunks_per_track=8, augment=False):
        self.tracks = track_list
        self.chunks_per_track = chunks_per_track
        self.augment = augment
        self._cache = {}

    def __len__(self):
        return len(self.tracks) * self.chunks_per_track

    def _load_track(self, idx):
        track_idx = idx // self.chunks_per_track
        if track_idx in self._cache:
            return self._cache[track_idx]

        track = self.tracks[track_idx]
        try:
            audio, sr = librosa.load(track["audio"], sr=SAMPLE_RATE, mono=True)
        except Exception:
            audio = np.zeros(CHUNK_SAMPLES * 2, dtype=np.float32)

        # Compute mel spectrogram with piano params
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        )
        mel = np.log(mel + 1e-8).astype(np.float32)
        total_frames = mel.shape[1]

        onset, frame, velocity = midi_to_flat_targets(track["midi"], total_frames)

        result = (mel, onset, frame, velocity, total_frames)
        if len(self._cache) < 200:
            self._cache[track_idx] = result
        return result

    def __getitem__(self, idx):
        mel, onset, frame, velocity, total_frames = self._load_track(idx)

        if total_frames > CHUNK_FRAMES:
            start = np.random.randint(0, total_frames - CHUNK_FRAMES)
            mel_chunk = mel[:, start:start + CHUNK_FRAMES]
            onset_chunk = onset[:, start:start + CHUNK_FRAMES]
            frame_chunk = frame[:, start:start + CHUNK_FRAMES]
            vel_chunk = velocity[:, start:start + CHUNK_FRAMES]
        else:
            pad = CHUNK_FRAMES - total_frames
            mel_chunk = np.pad(mel, ((0, 0), (0, pad)))
            onset_chunk = np.pad(onset, ((0, 0), (0, pad)))
            frame_chunk = np.pad(frame, ((0, 0), (0, pad)))
            vel_chunk = np.pad(velocity, ((0, 0), (0, pad)))

        audio_chunk = mel_chunk  # Already mel, apply augmentation in frequency domain
        if self.augment:
            if random.random() < 0.5:
                # Gain jitter in log-mel domain
                audio_chunk = audio_chunk + random.uniform(-0.3, 0.3)
            if random.random() < 0.2:
                # Frequency masking (SpecAugment style)
                f_start = random.randint(0, N_MELS - 20)
                f_width = random.randint(1, 15)
                audio_chunk[f_start:f_start + f_width, :] = 0.0

        return (
            torch.from_numpy(audio_chunk),
            torch.from_numpy(onset_chunk),
            torch.from_numpy(frame_chunk),
            torch.from_numpy(vel_chunk),
        )


# ============================================================================
# CNN LOADING
# ============================================================================

def load_piano_cnn(checkpoint_path, device):
    print(f"Loading piano CNN from {checkpoint_path}...")
    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    piano_model = PianoTranscriptionModel()
    piano_model.load_state_dict(checkpoint["model_state_dict"])
    cnn = piano_model.cnn

    for param in cnn.parameters():
        param.requires_grad = False
    cnn.eval()

    print(f"  Piano CNN: {sum(p.numel() for p in cnn.parameters()):,} params (frozen)")
    return cnn


def unfreeze_cnn_top_blocks(model):
    """Unfreeze CNN blocks 3+ for Phase 3."""
    freeze_boundary = 8
    unfrozen = 0
    for i, layer in enumerate(model.cnn.children()):
        if i >= freeze_boundary:
            for param in layer.parameters():
                param.requires_grad = True
                unfrozen += 1
    print(f"  Unfroze {unfrozen} CNN parameters")
    return unfrozen


# ============================================================================
# COSINE WARMUP SCHEDULER
# ============================================================================

class CosineWarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * factor)


# ============================================================================
# TRAINING
# ============================================================================

def train_epoch(model, loader, optimizer, onset_loss_fn, frame_loss_fn, vel_loss_fn, device):
    model.train()
    if model.cnn is not None:
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
        loss_vel = vel_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask]) if onset_mask.any() else torch.tensor(0.0, device=device)

        loss = loss_onset + loss_frame + 0.5 * loss_vel

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if num_batches % 5 == 0:
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
            loss_vel = vel_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask]) if onset_mask.any() else torch.tensor(0.0, device=device)

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
    print("Bass Transcription v3 — Transfer Learning Training")
    print("=" * 60)
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    print()

    # Download/extract bass data
    download_and_extract_slakh()
    bass_tracks = load_bass_tracks()

    if len(bass_tracks) < 50:
        print(f"ERROR: Only {len(bass_tracks)} bass tracks. Need more data.")
        sys.exit(1)

    # Splits
    train_tracks = [t for t in bass_tracks if t["split"] == "train"]
    val_tracks = [t for t in bass_tracks if t["split"] in ("validation", "test")]
    if len(val_tracks) < 10:
        random.shuffle(bass_tracks)
        split_idx = int(len(bass_tracks) * 0.9)
        train_tracks = bass_tracks[:split_idx]
        val_tracks = bass_tracks[split_idx:]

    print(f"Train: {len(train_tracks)} tracks, Val: {len(val_tracks)} tracks")

    # Load frozen piano CNN
    frozen_cnn = load_piano_cnn(PIANO_CKPT, device)

    # Build model
    model = BassTranscriptionModel_v3(num_keys=NUM_KEYS).to(device)
    model.cnn = frozen_cnn.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total_params:,} total, {trainable_params:,} trainable")

    # Verify
    dummy = torch.randn(1, N_MELS, CHUNK_FRAMES).to(device)
    o, f, v = model(dummy)
    assert o.shape == (1, NUM_KEYS, CHUNK_FRAMES), f"Onset shape: {o.shape}"
    print(f"Architecture OK: ({N_MELS}, {CHUNK_FRAMES}) -> ({NUM_KEYS}, {CHUNK_FRAMES})")
    del dummy, o, f, v
    torch.cuda.empty_cache()

    # Datasets
    train_dataset = BassV3Dataset(train_tracks, chunks_per_track=8, augment=True)
    val_dataset = BassV3Dataset(val_tracks, chunks_per_track=4, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f"Train: {len(train_loader)} batches/epoch, Val: {len(val_loader)} batches")

    # Optimizer
    param_groups = [
        {"params": model.adapter.parameters(), "lr": 3e-4, "name": "adapter"},
        {"params": model.adapter_residual.parameters(), "lr": 3e-4, "name": "adapter_res"},
        {"params": model.onset_lstm.parameters(), "lr": 1e-4, "name": "onset_lstm"},
        {"params": model.frame_lstm.parameters(), "lr": 1e-4, "name": "frame_lstm"},
        {"params": model.onset_head.parameters(), "lr": 3e-4, "name": "onset_head"},
        {"params": model.frame_head.parameters(), "lr": 3e-4, "name": "frame_head"},
        {"params": model.velocity_head.parameters(), "lr": 3e-4, "name": "vel_head"},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)

    # Loss
    onset_loss_fn = v3OnsetLoss(pos_weight=ONSET_POS_WEIGHT, gamma=FOCAL_GAMMA, dice_weight=DICE_WEIGHT)
    frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([FRAME_POS_WEIGHT]).to(device))
    vel_loss_fn = nn.MSELoss()

    # Scheduler
    scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, total_epochs=NUM_EPOCHS)

    print(f"\nOnset: Focal(pw={ONSET_POS_WEIGHT}, gamma={FOCAL_GAMMA}) + Dice(w={DICE_WEIGHT})")
    print(f"Frame: BCE(pw={FRAME_POS_WEIGHT})")
    print(f"Phases: warmup(1-{PHASE1_END}), full({PHASE1_END+1}-{PHASE2_END}), "
          f"CNN fine-tune({PHASE2_END+1}-{PHASE3_END})")

    # Resume
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float("inf")
    ckpt_path = SAVE_DIR / "latest_checkpoint.pt"

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, best F1={best_val_f1:.4f}")

    # Collapse detection
    low_recall_count = 0
    COLLAPSE_THRESHOLD = 5

    history = []
    print(f"\nStarting: epoch {start_epoch+1} -> {NUM_EPOCHS}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()

        # Phase 3: unfreeze CNN
        if epoch == PHASE2_END and best_val_f1 > 0.3:
            print(f"\n  Phase 3: Unfreezing CNN blocks 3+ (best F1={best_val_f1:.4f})")
            unfreeze_cnn_top_blocks(model)
            cnn_params = [p for p in model.cnn.parameters() if p.requires_grad]
            if cnn_params:
                optimizer.add_param_group({"params": cnn_params, "lr": 1e-5, "name": "cnn_finetune"})

        scheduler.step(epoch)

        train_loss, train_f1, train_prec, train_rec = train_epoch(
            model, train_loader, optimizer, onset_loss_fn, frame_loss_fn, vel_loss_fn, device)

        val_loss, val_f1, val_prec, val_rec = validate(
            model, val_loader, onset_loss_fn, frame_loss_fn, vel_loss_fn, device)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        phase = "Phase1" if epoch < PHASE1_END else ("Phase2" if epoch < PHASE2_END else "Phase3")

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.0f}s) {phase} LR={lr:.2e}")
        print(f"  Train — Loss: {train_loss:.4f} | F1: {train_f1:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | F1: {val_f1:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f}")

        # Collapse detection
        if val_rec < 0.02 and val_prec < 0.02:
            low_recall_count += 1
            print(f"  WARNING: Possible collapse ({low_recall_count}/{COLLAPSE_THRESHOLD})")
            if low_recall_count >= COLLAPSE_THRESHOLD:
                print(f"\n  COLLAPSED. Stopping.")
                break
        else:
            low_recall_count = 0

        # Save best
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
                    "bass_min_midi": BASS_MIN_MIDI,
                    "bass_max_midi": BASS_MAX_MIDI,
                    "hidden_size": 128,
                    "num_layers": 2,
                },
            }, str(SAVE_DIR / "best_bass_v3_model.pt"))
            print(f"  * Best model saved (F1={val_f1:.4f})")

        # Save checkpoint
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
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "val_f1": round(val_f1, 4),
            "val_prec": round(val_prec, 4),
            "val_rec": round(val_rec, 4),
            "lr": lr,
        })

    # Save history
    with open(SAVE_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best val onset F1: {best_val_f1:.4f}")
    print(f"Model: {SAVE_DIR / 'best_bass_v3_model.pt'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
