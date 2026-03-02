#!/usr/bin/env python3
"""
Linear Probe: Validate Piano CNN Transfer to Guitar
====================================================
Frozen piano CNN + Linear(1792, 49), trained for 5 epochs on GuitarSet.
This is a go/no-go test for the v3 transfer learning approach.

Decision criteria:
  - Onset F1 > 0.10 after 5 epochs => TRANSFER WORKS, proceed with v3
  - Onset F1 == 0 after 5 epochs  => Features don't transfer, need different approach

Architecture:
  - Piano CNN (frozen, from best_piano_model.pt): Mel(229) -> Conv2D x4 -> (128, 14, T)
  - Flatten: (B, T, 1792)
  - Linear head: Linear(1792, 49) -> onset logits per frame
  - Loss: BCEWithLogitsLoss(pos_weight=20)

Spectrogram params MUST match piano training EXACTLY:
  sr=16000, hop=256, n_mels=229, n_fft=2048, fmin=30, fmax=8000

Run on RunPod:
    pip install torch librosa jams tqdm
    python train_linear_probe.py

Estimated time: ~30 minutes on any GPU.
Output: /workspace/guitar_v3_data/probe_results.json
"""

import os
import sys
import json
import time
import subprocess
import shutil
from pathlib import Path

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

def preflight():
    print("=" * 60)
    print("PRE-FLIGHT CHECKS — Linear Probe")
    print("=" * 60)
    errors = []

    # GPU
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  [OK] GPU: {name} ({mem:.1f} GB)")
        else:
            print("  [WARN] No GPU — will be slow but functional")
    except ImportError:
        errors.append("PyTorch not installed")

    # Disk
    try:
        total, used, free = shutil.disk_usage("/workspace")
        print(f"  [OK] Disk: {free / 1e9:.1f} GB free")
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
        print(f"  [OK] Installed")
    else:
        print(f"  [OK] Python deps present")

    # Piano checkpoint
    piano_ckpt = Path("/workspace/best_piano_model.pt")
    if not piano_ckpt.exists():
        # Try alternate locations
        for alt in [
            Path("/workspace/piano_model_results/best_piano_model.pt"),
            Path("backend/models/pretrained/best_piano_model.pt"),
        ]:
            if alt.exists():
                piano_ckpt = alt
                break
    if piano_ckpt.exists():
        size_mb = piano_ckpt.stat().st_size / 1e6
        print(f"  [OK] Piano checkpoint: {piano_ckpt} ({size_mb:.0f} MB)")
    else:
        errors.append("Piano checkpoint not found. Expected at /workspace/best_piano_model.pt")

    # GuitarSet data
    guitarset_dir = Path("/workspace/guitarset")
    if guitarset_dir.exists():
        import glob
        wavs = glob.glob(f"{guitarset_dir}/**/*.wav", recursive=True)
        jams_files = glob.glob(f"{guitarset_dir}/**/*.jams", recursive=True)
        print(f"  [OK] GuitarSet: {len(wavs)} WAV, {len(jams_files)} JAMS")
        if len(wavs) < 10:
            errors.append("GuitarSet has too few WAV files. Run prepare_guitar_data.py first.")
    else:
        errors.append("GuitarSet not found at /workspace/guitarset. Run prepare_guitar_data.py first.")

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


preflight()


# ============================================================================
# IMPORTS
# ============================================================================

import glob
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
# CONFIG — MUST match piano training EXACTLY
# ============================================================================

SAMPLE_RATE = 16000
HOP_LENGTH = 256
N_MELS = 229
N_FFT = 2048
FMIN = 30.0
FMAX = 8000.0
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)    # 80000
CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)        # 312

# Guitar pitch range: E2 (MIDI 40) to E6 (MIDI 88) = 49 keys
GUITAR_MIN_MIDI = 40
GUITAR_MAX_MIDI = 88
NUM_KEYS = 49

# Training
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3  # Higher LR for linear probe (only one layer)
POS_WEIGHT = 20.0

# Paths
GUITARSET_DIR = Path("/workspace/guitarset")
OUTPUT_DIR = Path("/workspace/guitar_v3_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find piano checkpoint
PIANO_CKPT = None
for path in [
    Path("/workspace/best_piano_model.pt"),
    Path("/workspace/piano_model_results/best_piano_model.pt"),
    Path("backend/models/pretrained/best_piano_model.pt"),
]:
    if path.exists():
        PIANO_CKPT = path
        break


# ============================================================================
# PIANO MODEL (duplicated for RunPod portability)
# ============================================================================

class PianoTranscriptionModel(nn.Module):
    """
    Exact copy of piano_transcriber.py PianoTranscriptionModel.
    Needed to load the checkpoint and extract the CNN.
    """

    def __init__(self, n_mels=229, num_keys=88,
                 hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_keys = num_keys

        # 229 -> 114 -> 57 -> 28 -> 14
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

        cnn_output_dim = 128 * 14  # 1792

        self.onset_lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.frame_lstm = nn.LSTM(
            input_size=cnn_output_dim + num_keys,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_size * 2

        self.onset_head = nn.Linear(lstm_output_dim, num_keys)
        self.frame_head = nn.Linear(lstm_output_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, num_keys),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, n_mels, time = x.shape
        cnn_out = self.cnn(x.unsqueeze(1))
        cnn_features = cnn_out.permute(0, 3, 1, 2)
        cnn_features = cnn_features.reshape(batch, time, -1)
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
# LINEAR PROBE MODEL
# ============================================================================

class LinearProbe(nn.Module):
    """
    Frozen piano CNN + single Linear layer for guitar pitch detection.

    Input:  Mel spectrogram (B, 229, T)
    Output: onset logits    (B, 49, T)
            frame logits    (B, 49, T)
    """

    def __init__(self, frozen_cnn, num_keys=49):
        super().__init__()
        self.cnn = frozen_cnn
        self.num_keys = num_keys

        # Simple linear heads (the whole point is to test if CNN features work)
        self.onset_head = nn.Linear(1792, num_keys)
        self.frame_head = nn.Linear(1792, num_keys)

        # Initialize with negative bias for sparse targets
        nn.init.constant_(self.onset_head.bias, -3.0)
        nn.init.constant_(self.frame_head.bias, -2.0)

    def forward(self, mel):
        batch, n_mels, time = mel.shape

        # Frozen CNN forward pass — BN stays in eval mode
        with torch.no_grad():
            cnn_out = self.cnn(mel.unsqueeze(1))  # (B, 128, 14, T)

        # Flatten CNN features
        features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)  # (B, T, 1792)

        # Linear heads
        onset_logits = self.onset_head(features).permute(0, 2, 1)  # (B, 49, T)
        frame_logits = self.frame_head(features).permute(0, 2, 1)  # (B, 49, T)

        return onset_logits, frame_logits


# ============================================================================
# DATASET
# ============================================================================

def extract_track_id(wav_name):
    """Extract canonical track ID from GuitarSet wav filename."""
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


class GuitarProbeDataset(Dataset):
    """
    GuitarSet dataset producing mel spectrograms with piano-matching params
    and flat pitch targets (49 keys: MIDI 40-88).
    """

    def __init__(self, data_dir, split="train", chunk_duration=5.0, augment=False):
        self.chunk_duration = chunk_duration
        self.augment = augment

        data_dir = Path(data_dir)
        wav_files = sorted(data_dir.rglob("*.wav"))
        jams_files = sorted(data_dir.rglob("*.jams"))

        # Build JAMS lookup
        jams_lookup = {}
        for jp in jams_files:
            jams_lookup[jp.stem] = jp
            jams_lookup[jp.stem.lower()] = jp

        # Match audio to annotations
        all_samples = []
        for wav_path in wav_files:
            track_id = extract_track_id(wav_path.stem)
            jams_path = jams_lookup.get(track_id) or jams_lookup.get(track_id.lower())
            if jams_path is not None:
                all_samples.append({"audio": str(wav_path), "jams": str(jams_path)})

        # Split by index (deterministic)
        np.random.seed(42)
        np.random.shuffle(all_samples)
        split_idx = int(len(all_samples) * 0.8)

        if split == "train":
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]

        print(f"  {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load audio at piano sample rate
        audio, sr = librosa.load(sample["audio"], sr=SAMPLE_RATE, mono=True)

        # Random chunk
        chunk_samples = int(self.chunk_duration * SAMPLE_RATE)
        if len(audio) > chunk_samples:
            start = np.random.randint(0, len(audio) - chunk_samples)
            audio = audio[start:start + chunk_samples]
            start_time = start / SAMPLE_RATE
        else:
            audio = np.pad(audio, (0, chunk_samples - len(audio)))
            start_time = 0.0

        end_time = start_time + self.chunk_duration

        # Simple augmentation
        if self.augment:
            if np.random.random() < 0.5:
                audio = audio * np.random.uniform(0.7, 1.3)
            if np.random.random() < 0.3:
                noise = np.random.randn(len(audio)).astype(np.float32) * np.random.uniform(0.001, 0.005)
                audio = audio + noise

        # Compute mel spectrogram (MUST match piano params)
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH,
            n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
        )
        mel = np.log(mel + 1e-8).astype(np.float32)
        num_frames = mel.shape[1]

        # Parse JAMS for flat pitch targets
        onset_target = np.zeros((NUM_KEYS, num_frames), dtype=np.float32)
        frame_target = np.zeros((NUM_KEYS, num_frames), dtype=np.float32)

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

                    # Filter to guitar range
                    if midi_note < GUITAR_MIN_MIDI or midi_note > GUITAR_MAX_MIDI:
                        continue

                    # Check if note overlaps with chunk
                    if offset <= start_time or onset >= end_time:
                        continue

                    key_idx = midi_note - GUITAR_MIN_MIDI  # 0-48

                    onset_frame = int((onset - start_time) * frames_per_sec)
                    offset_frame = int((offset - start_time) * frames_per_sec)

                    if 0 <= onset_frame < num_frames:
                        onset_target[key_idx, onset_frame] = 1.0

                    for f in range(max(0, onset_frame), min(num_frames, offset_frame)):
                        frame_target[key_idx, f] = 1.0

        except Exception as e:
            pass  # Return zero targets on error

        # Pad/truncate to fixed size
        if num_frames < CHUNK_FRAMES:
            pad = CHUNK_FRAMES - num_frames
            mel = np.pad(mel, ((0, 0), (0, pad)))
            onset_target = np.pad(onset_target, ((0, 0), (0, pad)))
            frame_target = np.pad(frame_target, ((0, 0), (0, pad)))
        elif num_frames > CHUNK_FRAMES:
            mel = mel[:, :CHUNK_FRAMES]
            onset_target = onset_target[:, :CHUNK_FRAMES]
            frame_target = frame_target[:, :CHUNK_FRAMES]

        return (
            torch.from_numpy(mel),
            torch.from_numpy(onset_target),
            torch.from_numpy(frame_target),
        )


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(logits, targets, threshold=0.5):
    """Compute F1, precision, recall from logits and binary targets."""
    with torch.no_grad():
        preds = (torch.sigmoid(logits) > threshold).float()
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Also compute positive rate
        pos_rate = preds.sum() / preds.numel()
        target_rate = targets.sum() / targets.numel()

        return {
            "f1": f1.item(),
            "precision": precision.item(),
            "recall": recall.item(),
            "pred_pos_rate": pos_rate.item(),
            "target_pos_rate": target_rate.item(),
        }


# ============================================================================
# LOAD PIANO CNN
# ============================================================================

def load_frozen_piano_cnn(checkpoint_path, device):
    """Load piano model, extract CNN, freeze all params, set to eval mode."""
    print(f"Loading piano CNN from {checkpoint_path}...")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    piano_model = PianoTranscriptionModel()
    piano_model.load_state_dict(checkpoint["model_state_dict"])

    cnn = piano_model.cnn

    # Freeze all parameters
    for param in cnn.parameters():
        param.requires_grad = False

    # Keep BatchNorm in eval mode (critical for frozen CNN)
    cnn.eval()

    param_count = sum(p.numel() for p in cnn.parameters())
    print(f"  Piano CNN: {param_count:,} parameters (all frozen)")
    print(f"  Loaded from epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?')}")

    return cnn


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("LINEAR PROBE: Piano CNN -> Guitar Pitch Detection")
    print("=" * 60)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load frozen piano CNN
    frozen_cnn = load_frozen_piano_cnn(PIANO_CKPT, device)

    # Build probe model
    model = LinearProbe(frozen_cnn, num_keys=NUM_KEYS).to(device)

    # Ensure CNN stays in eval mode even when model.train() is called
    def custom_train(mode=True):
        nn.Module.train(model, mode)
        model.cnn.eval()  # Always keep CNN in eval mode
        return model
    model.train = custom_train

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} total, {trainable_params:,} trainable")

    # Verify architecture with dummy input
    dummy = torch.randn(1, N_MELS, CHUNK_FRAMES).to(device)
    onset_out, frame_out = model(dummy)
    assert onset_out.shape == (1, NUM_KEYS, CHUNK_FRAMES), f"Shape mismatch: {onset_out.shape}"
    print(f"Architecture OK: ({N_MELS}, {CHUNK_FRAMES}) -> ({NUM_KEYS}, {CHUNK_FRAMES})")
    del dummy, onset_out, frame_out
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print()

    # Datasets
    print("Loading GuitarSet...")
    train_dataset = GuitarProbeDataset(GUITARSET_DIR, split="train", augment=True)
    val_dataset = GuitarProbeDataset(GUITARSET_DIR, split="val", augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    # Sanity check: positive rate
    mel_b, onset_b, frame_b = next(iter(train_loader))
    onset_pos_rate = onset_b.sum() / onset_b.numel()
    frame_pos_rate = frame_b.sum() / frame_b.numel()
    print(f"Onset positive rate: {onset_pos_rate:.5f} ({onset_pos_rate*100:.3f}%)")
    print(f"Frame positive rate: {frame_pos_rate:.5f} ({frame_pos_rate*100:.3f}%)")
    print()

    # Optimizer (only linear head params)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    # Loss
    pw_tensor = torch.tensor([POS_WEIGHT]).to(device)
    onset_criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)
    frame_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]).to(device))

    print(f"Onset loss: BCEWithLogitsLoss(pos_weight={POS_WEIGHT})")
    print(f"Frame loss: BCEWithLogitsLoss(pos_weight=5.0)")
    print(f"Optimizer: AdamW(lr={LEARNING_RATE})")
    print(f"Epochs: {NUM_EPOCHS}")
    print()

    # Training loop
    results_per_epoch = []
    best_f1 = 0.0

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        # ----- Train -----
        model.train()  # Custom train keeps CNN in eval
        train_loss = 0.0
        num_batches = 0
        epoch_onset_logits = []
        epoch_onset_targets = []

        for mel_b, onset_tgt, frame_tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} Train"):
            mel_b = mel_b.to(device)
            onset_tgt = onset_tgt.to(device)
            frame_tgt = frame_tgt.to(device)

            onset_logits, frame_logits = model(mel_b)

            loss_onset = onset_criterion(onset_logits, onset_tgt)
            loss_frame = frame_criterion(frame_logits, frame_tgt)
            loss = loss_onset + loss_frame

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

            # Collect for metrics (every 3rd batch to save memory)
            if num_batches % 3 == 0:
                epoch_onset_logits.append(onset_logits.detach().cpu())
                epoch_onset_targets.append(onset_tgt.detach().cpu())

        train_loss /= max(num_batches, 1)

        if epoch_onset_logits:
            train_metrics = compute_metrics(
                torch.cat(epoch_onset_logits), torch.cat(epoch_onset_targets)
            )
        else:
            train_metrics = {"f1": 0, "precision": 0, "recall": 0, "pred_pos_rate": 0, "target_pos_rate": 0}

        # ----- Validate -----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        val_onset_logits = []
        val_onset_targets = []
        val_frame_logits = []
        val_frame_targets = []

        with torch.no_grad():
            for mel_b, onset_tgt, frame_tgt in val_loader:
                mel_b = mel_b.to(device)
                onset_tgt = onset_tgt.to(device)
                frame_tgt = frame_tgt.to(device)

                onset_logits, frame_logits = model(mel_b)

                loss_onset = onset_criterion(onset_logits, onset_tgt)
                loss_frame = frame_criterion(frame_logits, frame_tgt)
                loss = loss_onset + loss_frame

                val_loss += loss.item()
                val_batches += 1

                val_onset_logits.append(onset_logits.cpu())
                val_onset_targets.append(onset_tgt.cpu())
                val_frame_logits.append(frame_logits.cpu())
                val_frame_targets.append(frame_tgt.cpu())

        val_loss /= max(val_batches, 1)

        val_onset_metrics = compute_metrics(
            torch.cat(val_onset_logits), torch.cat(val_onset_targets)
        )
        val_frame_metrics = compute_metrics(
            torch.cat(val_frame_logits), torch.cat(val_frame_targets)
        )

        elapsed = time.time() - t0

        # Report
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} ({elapsed:.0f}s)")
        print(f"  Train — Loss: {train_loss:.4f} | "
              f"Onset F1: {train_metrics['f1']:.4f} | "
              f"P: {train_metrics['precision']:.4f} | "
              f"R: {train_metrics['recall']:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | "
              f"Onset F1: {val_onset_metrics['f1']:.4f} | "
              f"P: {val_onset_metrics['precision']:.4f} | "
              f"R: {val_onset_metrics['recall']:.4f}")
        print(f"  Val Frame — "
              f"F1: {val_frame_metrics['f1']:.4f} | "
              f"P: {val_frame_metrics['precision']:.4f} | "
              f"R: {val_frame_metrics['recall']:.4f}")
        print(f"  Pred pos rate: {val_onset_metrics['pred_pos_rate']:.5f} | "
              f"Target pos rate: {val_onset_metrics['target_pos_rate']:.5f}")

        if val_onset_metrics["f1"] > best_f1:
            best_f1 = val_onset_metrics["f1"]
            print(f"  * New best onset F1: {best_f1:.4f}")

        results_per_epoch.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "train_onset_f1": round(train_metrics["f1"], 4),
            "val_onset_f1": round(val_onset_metrics["f1"], 4),
            "val_onset_precision": round(val_onset_metrics["precision"], 4),
            "val_onset_recall": round(val_onset_metrics["recall"], 4),
            "val_frame_f1": round(val_frame_metrics["f1"], 4),
            "val_pred_pos_rate": round(val_onset_metrics["pred_pos_rate"], 6),
            "val_target_pos_rate": round(val_onset_metrics["target_pos_rate"], 6),
            "elapsed_sec": round(elapsed, 1),
        })

    # ============================================================================
    # VERDICT
    # ============================================================================
    print("\n" + "=" * 60)
    print("LINEAR PROBE RESULTS")
    print("=" * 60)

    final_f1 = results_per_epoch[-1]["val_onset_f1"]
    final_precision = results_per_epoch[-1]["val_onset_precision"]
    final_recall = results_per_epoch[-1]["val_onset_recall"]

    print(f"Best onset F1:  {best_f1:.4f}")
    print(f"Final onset F1: {final_f1:.4f}")
    print(f"Final onset P:  {final_precision:.4f}")
    print(f"Final onset R:  {final_recall:.4f}")

    if best_f1 > 0.10:
        verdict = "GO"
        message = (
            f"Piano CNN features ARE useful for guitar pitch detection. "
            f"Onset F1={best_f1:.4f} > 0.10 threshold. "
            f"Proceed with full v3 training."
        )
    elif best_f1 > 0.03:
        verdict = "MARGINAL"
        message = (
            f"Weak signal detected (F1={best_f1:.4f}). "
            f"Transfer may work with the full adapter+LSTM architecture. "
            f"Proceed with caution."
        )
    else:
        verdict = "NO-GO"
        message = (
            f"Piano CNN features do NOT transfer to guitar (F1={best_f1:.4f}). "
            f"Need different approach: larger multi-instrument pretraining, "
            f"or train CNN from scratch with more data."
        )

    print(f"\nVERDICT: {verdict}")
    print(f"  {message}")

    # Save results
    probe_results = {
        "verdict": verdict,
        "message": message,
        "best_onset_f1": round(best_f1, 4),
        "config": {
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "fmin": FMIN,
            "fmax": FMAX,
            "num_keys": NUM_KEYS,
            "pos_weight": POS_WEIGHT,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
        },
        "epochs": results_per_epoch,
    }

    results_path = OUTPUT_DIR / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"\nResults saved: {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
