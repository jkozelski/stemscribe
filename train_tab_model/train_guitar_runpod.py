#!/usr/bin/env python3
"""
Guitar Tablature Transcription Training — v2 (RunPod)

Fixed loss function to prevent collapse on sparse targets.

Run on RunPod:
    pip install torch torchaudio librosa jams scikit-learn tqdm
    python train_guitar_runpod.py

Output: /workspace/guitar_results/best_tab_model.pt
"""

import os
import sys
import glob
import subprocess
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import jams
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ============================================================================
# CONFIG — must match inference in guitar_tab_transcriber.py
# ============================================================================
CONFIG = {
    'sample_rate': 22050,
    'hop_length': 256,
    'n_bins': 84,
    'bins_per_octave': 12,
    'num_strings': 6,
    'num_frets': 20,
    'tuning': [40, 45, 50, 55, 59, 64],  # Standard tuning MIDI

    'batch_size': 16,
    'learning_rate': 3e-5,
    'warmup_epochs': 5,
    'num_epochs': 150,
    'chunk_length_sec': 3.0,

    # v3: Much more aggressive anti-collapse
    'onset_pos_weight': 500.0,
    'frame_pos_weight': 50.0,
    'focal_gamma': 2.0,
    'dice_weight': 0.5,  # Blend: 0.5*Focal + 0.5*Dice
}

DATA_DIR = "/workspace/guitarset"
SAVE_DIR = Path("/workspace/guitar_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DOWNLOAD DATASET
# ============================================================================
def download_guitarset():
    if os.path.exists(DATA_DIR) and len(glob.glob(f'{DATA_DIR}/**/*.wav', recursive=True)) > 0:
        print("GuitarSet already downloaded")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    ANNOT_URL = "https://zenodo.org/records/3371780/files/annotation.zip?download=1"
    AUDIO_URL = "https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1"

    print("Downloading GuitarSet annotations (~39 MB)...")
    subprocess.run(["wget", "-q", "--show-progress", ANNOT_URL, "-O", "/workspace/annotation.zip"], check=True)
    subprocess.run(["unzip", "-q", "/workspace/annotation.zip", "-d", DATA_DIR], check=True)
    os.remove("/workspace/annotation.zip")

    print("Downloading GuitarSet audio (~657 MB)...")
    subprocess.run(["wget", "-q", "--show-progress", AUDIO_URL, "-O", "/workspace/audio_mono-mic.zip"], check=True)
    subprocess.run(["unzip", "-q", "/workspace/audio_mono-mic.zip", "-d", DATA_DIR], check=True)
    os.remove("/workspace/audio_mono-mic.zip")

    wav_files = glob.glob(f'{DATA_DIR}/**/*.wav', recursive=True)
    jams_files = glob.glob(f'{DATA_DIR}/**/*.jams', recursive=True)
    print(f"Downloaded: {len(wav_files)} WAV, {len(jams_files)} JAMS files")


# ============================================================================
# LOSS FUNCTIONS & METRICS
# ============================================================================
class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss + pos_weight for extremely imbalanced binary classification."""

    def __init__(self, pos_weight=500.0, gamma=2.0):
        super().__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma

    def forward(self, logits, targets):
        pw = torch.tensor([self.pos_weight], device=logits.device, dtype=logits.dtype)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction='none'
        )
        probs = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class DiceLoss(nn.Module):
    """Dice Loss — class-imbalance invariant, great for sparse targets."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1.0 - dice


class CombinedOnsetLoss(nn.Module):
    """Blend of Focal BCE + Dice Loss for onset detection."""

    def __init__(self, pos_weight=500.0, gamma=2.0, dice_weight=0.5):
        super().__init__()
        self.focal = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=gamma)
        self.dice = DiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss


def compute_onset_f1(logits, targets, threshold=0.5):
    """Compute onset F1/precision/recall from logits."""
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
# HELPERS
# ============================================================================
def midi_to_tab(midi_note, tuning=CONFIG['tuning']):
    positions = []
    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret < CONFIG['num_frets']:
            positions.append((string_idx, fret))
    return positions


# ============================================================================
# DATASET
# ============================================================================
class GuitarSetDataset(Dataset):
    def __init__(self, data_dir, split='train', chunk_length_sec=3.0,
                 sample_rate=22050, augment=False):
        self.data_dir = Path(data_dir)
        self.chunk_length = chunk_length_sec
        self.sr = sample_rate
        self.hop_length = CONFIG['hop_length']
        self.augment = augment

        self.wav_files = sorted(self.data_dir.rglob('*.wav'))
        self.jams_files = sorted(self.data_dir.rglob('*.jams'))

        print(f"Found {len(self.wav_files)} WAV, {len(self.jams_files)} JAMS")

        self.jams_lookup = {}
        for jp in self.jams_files:
            self.jams_lookup[jp.stem] = jp
            self.jams_lookup[jp.stem.lower()] = jp

        self.samples = self._load_samples(split)
        print(f"Loaded {len(self.samples)} samples for {split}" +
              (" (augmented)" if augment else ""))

    def _extract_track_id(self, wav_name):
        stem = wav_name
        for suffix in ['_mic', '_mix', '_hex_cln', '_hex']:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        for prefix in ['audio_mono-mic_', 'audio_mono-mic-',
                        'audio_mono-pickup_mix_', 'audio_mono-pickup_mix-',
                        'audio_hex-pickup_original_', 'audio_hex-pickup_original-',
                        'audio_hex-pickup_debleeded_', 'audio_hex-pickup_debleeded-']:
            if stem.startswith(prefix):
                stem = stem[len(prefix):]
                break
        return stem

    def _load_samples(self, split):
        samples = []
        for audio_path in self.wav_files:
            track_id = self._extract_track_id(audio_path.stem)
            jams_path = self.jams_lookup.get(track_id) or self.jams_lookup.get(track_id.lower())
            if jams_path is not None:
                samples.append({'audio': str(audio_path), 'jams': str(jams_path)})

        np.random.seed(42)
        np.random.shuffle(samples)
        split_idx = int(len(samples) * 0.8)

        if split == 'train':
            return samples[:split_idx]
        else:
            return samples[split_idx:]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        audio, sr = torchaudio.load(sample['audio'])
        if sr != self.sr:
            audio = torchaudio.functional.resample(audio, sr, self.sr)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.squeeze(0)

        chunk_samples = int(self.chunk_length * self.sr)
        if len(audio) > chunk_samples:
            start = np.random.randint(0, len(audio) - chunk_samples)
            audio = audio[start:start + chunk_samples]
            start_time = start / self.sr
        else:
            audio = F.pad(audio, (0, chunk_samples - len(audio)))
            start_time = 0

        audio_np = audio.numpy()

        # Data augmentation
        if self.augment:
            if random.random() < 0.5:
                audio_np = audio_np * random.uniform(0.7, 1.3)
            if random.random() < 0.3:
                noise = np.random.randn(len(audio_np)).astype(np.float32) * random.uniform(0.001, 0.005)
                audio_np = audio_np + noise

        cqt = librosa.cqt(audio_np, sr=self.sr, hop_length=self.hop_length,
                           n_bins=84, bins_per_octave=12)
        cqt = np.abs(cqt)
        cqt = torch.from_numpy(np.log(cqt + 1e-8)).float()

        jam = jams.load(sample['jams'])

        num_frames = cqt.shape[-1]
        onset_target = torch.zeros(CONFIG['num_strings'], CONFIG['num_frets'], num_frames)
        frame_target = torch.zeros(CONFIG['num_strings'], CONFIG['num_frets'], num_frames)

        end_time = start_time + self.chunk_length

        for annot in jam.annotations:
            if annot.namespace == 'note_midi':
                for obs in annot.data:
                    if start_time <= obs.time < end_time:
                        midi_note = int(obs.value)
                        positions = midi_to_tab(midi_note)
                        if positions:
                            string_idx, fret = positions[0]
                            onset_frame = int((obs.time - start_time) * self.sr / self.hop_length)
                            end_frame = int((obs.time + obs.duration - start_time) * self.sr / self.hop_length)
                            if 0 <= onset_frame < num_frames:
                                onset_target[string_idx, fret, onset_frame] = 1
                            for f in range(max(0, onset_frame), min(num_frames, end_frame)):
                                frame_target[string_idx, fret, f] = 1

        return {'cqt': cqt, 'onsets': onset_target, 'frames': frame_target}


# ============================================================================
# MODEL (returns logits, no sigmoid)
# ============================================================================
class GuitarTabModel(nn.Module):
    def __init__(self, n_bins=84, num_strings=6, num_frets=20):
        super().__init__()
        self.num_strings = num_strings
        self.num_frets = num_frets

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout(0.25),
        )

        self.flat_size = 128 * (n_bins // 8)
        self.lstm = nn.LSTM(self.flat_size, 256, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.onset_head = nn.Linear(512, num_strings * num_frets)
        self.frame_head = nn.Linear(512, num_strings * num_frets)

        # v3: Initialize onset head bias to -2.0
        # sigmoid(-2) ≈ 0.12, so model starts predicting ~12% onset probability
        # instead of 50%, preventing immediate collapse to all-zeros
        nn.init.constant_(self.onset_head.bias, -2.0)
        nn.init.constant_(self.frame_head.bias, -2.0)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_stack(x)
        batch, channels, freq, time = x.shape
        x = x.permute(0, 3, 1, 2).reshape(batch, time, -1)
        x, _ = self.lstm(x)

        # v2: Return RAW LOGITS (no sigmoid)
        onset_logits = self.onset_head(x)
        frame_logits = self.frame_head(x)

        onset_logits = onset_logits.view(batch, time, self.num_strings, self.num_frets)
        onset_logits = onset_logits.permute(0, 2, 3, 1)
        frame_logits = frame_logits.view(batch, time, self.num_strings, self.num_frets)
        frame_logits = frame_logits.permute(0, 2, 3, 1)

        return onset_logits, frame_logits


# ============================================================================
# TRAINING
# ============================================================================
def train_epoch(model, loader, optimizer, onset_criterion, frame_criterion, device):
    model.train()
    total_loss = 0
    all_onset_logits = []
    all_onset_targets = []

    for batch in tqdm(loader, desc='  Train'):
        cqt = batch['cqt'].to(device)
        onsets_target = batch['onsets'].to(device)
        frames_target = batch['frames'].to(device)

        optimizer.zero_grad()
        onset_logits, frame_logits = model(cqt)

        min_len = min(onset_logits.shape[-1], onsets_target.shape[-1])
        onset_logits = onset_logits[..., :min_len]
        frame_logits = frame_logits[..., :min_len]
        onsets_target = onsets_target[..., :min_len]
        frames_target = frames_target[..., :min_len]

        onset_loss = onset_criterion(onset_logits, onsets_target)
        frame_loss = frame_criterion(frame_logits, frames_target)

        loss = onset_loss + frame_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        all_onset_logits.append(onset_logits.detach().cpu())
        all_onset_targets.append(onsets_target.detach().cpu())

    all_logits = torch.cat(all_onset_logits, dim=0)
    all_targets = torch.cat(all_onset_targets, dim=0)
    f1, prec, rec = compute_onset_f1(all_logits, all_targets)

    return total_loss / len(loader), f1, prec, rec


def validate(model, loader, onset_criterion, frame_criterion, device):
    model.eval()
    total_loss = 0
    all_onset_logits = []
    all_onset_targets = []

    with torch.no_grad():
        for batch in loader:
            cqt = batch['cqt'].to(device)
            onsets_target = batch['onsets'].to(device)
            frames_target = batch['frames'].to(device)

            onset_logits, frame_logits = model(cqt)

            min_len = min(onset_logits.shape[-1], onsets_target.shape[-1])
            onset_loss = onset_criterion(onset_logits[..., :min_len], onsets_target[..., :min_len])
            frame_loss = frame_criterion(frame_logits[..., :min_len], frames_target[..., :min_len])

            total_loss += (onset_loss + frame_loss).item()
            all_onset_logits.append(onset_logits[..., :min_len].cpu())
            all_onset_targets.append(onsets_target[..., :min_len].cpu())

    all_logits = torch.cat(all_onset_logits, dim=0)
    all_targets = torch.cat(all_onset_targets, dim=0)
    f1, prec, rec = compute_onset_f1(all_logits, all_targets)

    return total_loss / len(loader), f1, prec, rec


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("Guitar Tab Training v2 — RunPod")
    print("=" * 60)

    # GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU!")

    # Download dataset
    download_guitarset()

    # Create datasets
    train_dataset = GuitarSetDataset(DATA_DIR, split='train', augment=True)
    val_dataset = GuitarSetDataset(DATA_DIR, split='val', augment=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # Model
    model = GuitarTabModel(n_bins=84, num_strings=CONFIG['num_strings'],
                           num_frets=CONFIG['num_frets']).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'],
                                   weight_decay=1e-4)

    # v3: Combined onset loss (Focal + Dice) — much more robust to class imbalance
    onset_criterion = CombinedOnsetLoss(
        pos_weight=CONFIG['onset_pos_weight'],
        gamma=CONFIG['focal_gamma'],
        dice_weight=CONFIG['dice_weight']
    )
    frame_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG['frame_pos_weight']]).to(device)
    )

    # v3: LR warmup then ReduceLROnPlateau
    warmup_epochs = CONFIG['warmup_epochs']
    base_lr = CONFIG['learning_rate']

    print(f"Onset loss: Focal(pw={CONFIG['onset_pos_weight']}, γ={CONFIG['focal_gamma']}) + Dice(w={CONFIG['dice_weight']})")
    print(f"Frame loss: BCE(pw={CONFIG['frame_pos_weight']})")
    print(f"LR: {base_lr} with {warmup_epochs}-epoch warmup")

    # Resume from checkpoint
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    ckpt_path = SAVE_DIR / 'latest_checkpoint.pt'
    if ckpt_path.exists():
        print(f"Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best F1={best_val_f1:.4f}")

    # Training loop
    zero_recall_count = 0
    COLLAPSE_THRESHOLD = 20  # v3: more patience
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=15, factor=0.5, min_lr=1e-6
    )

    print(f"\nStarting training — {CONFIG['num_epochs']} epochs")
    print(f"Collapse detection: stops if recall=0 for {COLLAPSE_THRESHOLD} epochs\n")

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        t0 = time.time()

        # v3: LR warmup
        if epoch < warmup_epochs:
            warmup_lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        train_loss, train_f1, train_prec, train_rec = train_epoch(
            model, train_loader, optimizer, onset_criterion, frame_criterion, device
        )
        val_loss, val_f1, val_prec, val_rec = validate(
            model, val_loader, onset_criterion, frame_criterion, device
        )

        if epoch >= warmup_epochs:
            scheduler.step(val_loss)
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']} ({elapsed:.0f}s) LR={lr:.2e}")
        print(f"  Train — Loss: {train_loss:.4f} | F1: {train_f1:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f}")
        print(f"  Val   — Loss: {val_loss:.4f} | F1: {val_f1:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f}")

        # Collapse detection
        if val_rec < 0.01:
            zero_recall_count += 1
            print(f"  ⚠️  Recall near zero ({zero_recall_count}/{COLLAPSE_THRESHOLD})")
            if zero_recall_count >= COLLAPSE_THRESHOLD:
                print(f"\n🛑 COLLAPSED — stopping. Try higher pos_weight or lower LR.")
                break
        else:
            zero_recall_count = 0

        # Save best by F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'config': CONFIG,
            }, SAVE_DIR / 'best_tab_model.pt')
            print(f"  ✅ Best model saved (F1={val_f1:.4f})")

        # Always save latest checkpoint (for resume)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_f1': val_f1,
            'best_val_f1': best_val_f1,
            'best_val_loss': best_val_loss,
            'config': CONFIG,
        }, ckpt_path)

    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Model saved: {SAVE_DIR / 'best_tab_model.pt'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
