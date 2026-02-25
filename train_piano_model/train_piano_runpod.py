#!/usr/bin/env python3
"""
Piano Transcription CRNN Training — RunPod Edition (v2)
=======================================================
Bulletproof script with pre-flight checks. Trains on whatever MAESTRO
data is available (partial extraction OK — 200+ tracks is enough).

Architecture MUST match piano_transcriber.py exactly:
  Mel (229 bins) → 4×Conv2D+MaxPool(2,1) [48,48,64,96,128 channels]
  → BiLSTM (256 hidden) → onset/frame/velocity heads (88 keys)

Usage:
    nohup python3 /workspace/train_piano_runpod.py > /workspace/training_output.log 2>&1 &
"""

import os
import sys
import csv
import json
import time
import hashlib
import subprocess
import shutil
from pathlib import Path

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

def preflight():
    """Run all checks before doing anything. Exit on fatal errors."""
    print('=' * 60)
    print('PRE-FLIGHT CHECKS')
    print('=' * 60)
    errors = []

    # 1. GPU check
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f'  [OK] GPU: {name} ({mem:.1f} GB)')
            if mem < 8:
                errors.append(f'GPU VRAM too low: {mem:.1f} GB (need 8+)')
        else:
            errors.append('No GPU detected')
    except ImportError:
        errors.append('PyTorch not installed')

    # 2. Disk space (use df command — shutil lies on network volumes)
    try:
        result = subprocess.run(['df', '-BG', '/workspace'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            free_gb = int(parts[3].rstrip('G'))
            print(f'  [OK] Disk: {free_gb} GB free on /workspace')
        else:
            total, used, free = shutil.disk_usage('/workspace')
            free_gb = free // (1024**3)
            print(f'  [OK] Disk: ~{free_gb} GB free (shutil estimate)')
    except Exception as e:
        print(f'  [WARN] Could not check disk: {e}')

    # 3. Delete any leftover zip (wastes 101GB!)
    zip_path = Path('/workspace/maestro-v3.0.0.zip')
    if zip_path.exists():
        size_gb = zip_path.stat().st_size / 1e9
        print(f'  [FIX] Deleting leftover zip ({size_gb:.1f} GB)...')
        zip_path.unlink()
        print(f'  [OK] Freed {size_gb:.1f} GB')

    # 4. Check dataset
    maestro_dir = Path('/workspace/maestro-v3.0.0')
    if maestro_dir.exists():
        wavs = list(maestro_dir.glob('**/*.wav'))
        midis = list(maestro_dir.glob('**/*.midi'))
        print(f'  [OK] MAESTRO data: {len(wavs)} WAVs, {len(midis)} MIDIs')
        if len(wavs) < 50:
            errors.append(f'Only {len(wavs)} WAV files — need at least 50')
    else:
        errors.append('No MAESTRO directory found at /workspace/maestro-v3.0.0')

    # 5. Install system tools if needed
    if shutil.which('unzip') is None:
        print('  [FIX] Installing unzip...')
        subprocess.run(['apt-get', 'update', '-qq'], capture_output=True)
        subprocess.run(['apt-get', 'install', '-y', '-qq', 'unzip'], capture_output=True)

    # 6. Install Python deps
    deps = ['librosa', 'pretty_midi', 'soundfile']
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing.append(dep)
    if missing:
        print(f'  [FIX] Installing: {", ".join(missing)}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)
        print(f'  [OK] Installed {", ".join(missing)}')
    else:
        print(f'  [OK] Python deps: all present')

    # Report
    print('=' * 60)
    if errors:
        for e in errors:
            print(f'  FATAL: {e}')
        print('PRE-FLIGHT: FAILED')
        sys.exit(1)
    else:
        print('PRE-FLIGHT: All checks passed')
    print('=' * 60)
    print()

preflight()

# ============================================================================
# IMPORTS (after deps installed)
# ============================================================================

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import pretty_midi
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG — MUST match piano_transcriber.py EXACTLY
# ============================================================================

CONFIG = {
    'sample_rate': 16000,
    'hop_length': 256,
    'n_mels': 229,
    'n_fft': 2048,
    'fmin': 30.0,
    'fmax': 8000.0,
    'chunk_duration': 5.0,
    'num_keys': 88,
    'min_midi': 21,
    'max_midi': 108,
    'batch_size': 12,
    'learning_rate': 1e-4,
    'num_epochs': 50,
}

SAMPLE_RATE = CONFIG['sample_rate']
HOP_LENGTH = CONFIG['hop_length']
N_MELS = CONFIG['n_mels']
N_FFT = CONFIG['n_fft']
NUM_KEYS = CONFIG['num_keys']
MIN_MIDI = CONFIG['min_midi']
MAX_MIDI = CONFIG['max_midi']
CHUNK_DURATION = CONFIG['chunk_duration']
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)   # 80000
CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)       # 312

OUTPUT_DIR = Path('/workspace/piano_model_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAESTRO_DIR = Path('/workspace/maestro-v3.0.0')
PROGRESS_FILE = Path('/workspace/piano_training_progress.json')

device = torch.device('cuda')

print(f'Config: {SAMPLE_RATE}Hz, {N_MELS} mels, {NUM_KEYS} keys, {CHUNK_FRAMES} frames/chunk')
print(f'Output: {OUTPUT_DIR}')
print()

# ============================================================================
# BUILD TRACK LIST (from whatever data exists)
# ============================================================================

metadata_csv = MAESTRO_DIR / 'maestro-v3.0.0.csv'
tracks = {'train': [], 'validation': [], 'test': []}

if metadata_csv.exists():
    print('Building track list from CSV...')
    with open(metadata_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row['split']
            audio_path = MAESTRO_DIR / row['audio_filename']
            midi_path = MAESTRO_DIR / row['midi_filename']
            if audio_path.exists() and midi_path.exists():
                tracks[split].append({
                    'audio': str(audio_path),
                    'midi': str(midi_path),
                    'duration': float(row['duration']),
                })
else:
    print('No CSV — downloading metadata...')
    try:
        subprocess.run([
            'wget', '-q', '-O', str(metadata_csv),
            'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.csv'
        ], check=True, timeout=30)
        print('Downloaded metadata CSV')
        with open(metadata_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                split = row['split']
                audio_path = MAESTRO_DIR / row['audio_filename']
                midi_path = MAESTRO_DIR / row['midi_filename']
                if audio_path.exists() and midi_path.exists():
                    tracks[split].append({
                        'audio': str(audio_path),
                        'midi': str(midi_path),
                        'duration': float(row['duration']),
                    })
    except Exception:
        print('CSV download failed — using hash-based split')
        for wav_path in sorted(MAESTRO_DIR.glob('**/*.wav')):
            midi_path = wav_path.with_suffix('.midi')
            if not midi_path.exists():
                continue
            h = int(hashlib.md5(wav_path.name.encode()).hexdigest(), 16) % 100
            if h < 80:
                split = 'train'
            elif h < 90:
                split = 'validation'
            else:
                split = 'test'
            tracks[split].append({
                'audio': str(wav_path),
                'midi': str(midi_path),
                'duration': 0.0,
            })

for split, track_list in tracks.items():
    total_hours = sum(t.get('duration', 0) for t in track_list) / 3600
    if total_hours > 0:
        print(f'  {split:12s}: {len(track_list):4d} tracks ({total_hours:.1f} hours)')
    else:
        print(f'  {split:12s}: {len(track_list):4d} tracks')

total_tracks = sum(len(v) for v in tracks.values())
print(f'  Total: {total_tracks} tracks')

if total_tracks < 50:
    print('FATAL: Not enough tracks. Need at least 50.')
    sys.exit(1)

print()

# ============================================================================
# MIDI → TARGETS
# ============================================================================

def midi_to_piano_targets(midi_path, total_frames, sr=SAMPLE_RATE, hop=HOP_LENGTH):
    """Convert MIDI to onset/frame/velocity targets for 88 piano keys."""
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
            key_idx = note.pitch - MIN_MIDI
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

# ============================================================================
# DATASET
# ============================================================================

class PianoDataset(Dataset):
    def __init__(self, track_list, chunks_per_track=8):
        self.tracks = track_list
        self.chunks_per_track = chunks_per_track
        self._cache = {}

    def __len__(self):
        return len(self.tracks) * self.chunks_per_track

    def _load_track(self, idx):
        track_idx = idx // self.chunks_per_track
        if track_idx in self._cache:
            return self._cache[track_idx]

        track = self.tracks[track_idx]
        try:
            audio, _ = librosa.load(track['audio'], sr=SAMPLE_RATE, mono=True)
        except Exception:
            audio = np.zeros(CHUNK_SAMPLES * 2, dtype=np.float32)

        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=N_FFT,
            hop_length=HOP_LENGTH, n_mels=N_MELS,
            fmin=CONFIG['fmin'], fmax=CONFIG['fmax']
        )
        mel = np.log(mel + 1e-8).astype(np.float32)
        total_frames = mel.shape[1]

        onset, frame, velocity = midi_to_piano_targets(track['midi'], total_frames)
        result = (mel, onset, frame, velocity, total_frames)

        if len(self._cache) < 200:
            self._cache[track_idx] = result
        return result

    def __getitem__(self, idx):
        mel, onset, frame, velocity, total_frames = self._load_track(idx)

        if total_frames <= CHUNK_FRAMES:
            pad = CHUNK_FRAMES - total_frames
            mel_chunk = np.pad(mel, ((0, 0), (0, pad)))
            onset_chunk = np.pad(onset, ((0, 0), (0, pad)))
            frame_chunk = np.pad(frame, ((0, 0), (0, pad)))
            vel_chunk = np.pad(velocity, ((0, 0), (0, pad)))
        else:
            start = np.random.randint(0, total_frames - CHUNK_FRAMES)
            mel_chunk = mel[:, start:start + CHUNK_FRAMES]
            onset_chunk = onset[:, start:start + CHUNK_FRAMES]
            frame_chunk = frame[:, start:start + CHUNK_FRAMES]
            vel_chunk = velocity[:, start:start + CHUNK_FRAMES]

        return (
            torch.from_numpy(mel_chunk),
            torch.from_numpy(onset_chunk),
            torch.from_numpy(frame_chunk),
            torch.from_numpy(vel_chunk),
        )

# ============================================================================
# BUILD DATALOADERS
# ============================================================================

train_dataset = PianoDataset(tracks['train'], chunks_per_track=8)
val_dataset = PianoDataset(tracks['validation'] + tracks['test'], chunks_per_track=4)

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG['batch_size'],
    shuffle=True, num_workers=2, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=CONFIG['batch_size'],
    shuffle=False, num_workers=2, pin_memory=True
)

print(f'Train: {len(tracks["train"])} tracks, {len(train_loader)} batches/epoch')
print(f'Val:   {len(tracks["validation"]) + len(tracks["test"])} tracks, {len(val_loader)} batches/epoch')

# Sanity check
mel_b, onset_b, frame_b, vel_b = next(iter(train_loader))
print(f'Batch shapes: mel={mel_b.shape}, onset={onset_b.shape}')
print(f'Onset active: {onset_b.sum():.0f} / {onset_b.numel()}')
print()

# ============================================================================
# MODEL — MUST match piano_transcriber.py PianoTranscriptionModel EXACTLY
# ============================================================================

class PianoTranscriptionModel(nn.Module):
    """
    Exact copy of piano_transcriber.py model architecture.
    CNN channels: 48, 48, 64, 96, 128 with 4 MaxPool(2,1) layers.
    Mel dim: 229 → 114 → 57 → 28 → 14
    CNN output: 128 * 14 = 1792
    """

    def __init__(self, n_mels=229, num_keys=88, hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_keys = num_keys

        self.cnn = nn.Sequential(
            # Block 1: no pool
            nn.Conv2d(1, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),

            # Block 2: 229→114
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            # Block 3: 114→57
            nn.Conv2d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            # Block 4: 57→28
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            # Block 5: 28→14
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
            input_size=cnn_output_dim + num_keys,  # 1792 + 88 = 1880
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_size * 2  # 512

        self.onset_head = nn.Linear(lstm_output_dim, num_keys)
        self.frame_head = nn.Linear(lstm_output_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, num_keys),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, n_mels, time = x.shape

        cnn_out = self.cnn(x.unsqueeze(1))  # (B, 128, 14, T)

        cnn_features = cnn_out.permute(0, 3, 1, 2)  # (B, T, 128, 14)
        cnn_features = cnn_features.reshape(batch, time, -1)  # (B, T, 1792)

        onset_out, _ = self.onset_lstm(cnn_features)
        onset_logits = self.onset_head(onset_out)
        onset_pred = torch.sigmoid(onset_logits)  # (B, T, 88)

        frame_input = torch.cat([cnn_features, onset_pred.detach()], dim=-1)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)
        frame_pred = torch.sigmoid(frame_logits)

        velocity_pred = self.velocity_head(onset_out)

        # (B, T, 88) → (B, 88, T)
        return (
            onset_pred.permute(0, 2, 1),
            frame_pred.permute(0, 2, 1),
            velocity_pred.permute(0, 2, 1),
        )

# ============================================================================
# TRAINING SETUP
# ============================================================================

model = PianoTranscriptionModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'Model: {total_params:,} parameters')

# Verify architecture
dummy = torch.randn(1, N_MELS, CHUNK_FRAMES).to(device)
o, f, v = model(dummy)
assert o.shape == (1, NUM_KEYS, CHUNK_FRAMES), f'Onset shape mismatch: {o.shape}'
assert f.shape == (1, NUM_KEYS, CHUNK_FRAMES), f'Frame shape mismatch: {f.shape}'
assert v.shape == (1, NUM_KEYS, CHUNK_FRAMES), f'Velocity shape mismatch: {v.shape}'
print(f'Architecture verified: ({N_MELS}, {CHUNK_FRAMES}) -> ({NUM_KEYS}, {CHUNK_FRAMES})')
del dummy, o, f, v
torch.cuda.empty_cache()

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

onset_loss_fn = nn.BCELoss()
frame_loss_fn = nn.BCELoss()
velocity_loss_fn = nn.MSELoss()

# Resume from checkpoint
start_epoch = 0
best_val_loss = float('inf')
best_model_path = OUTPUT_DIR / 'best_piano_model.pt'

checkpoint_files = sorted(OUTPUT_DIR.glob('checkpoint_epoch_*.pt'))
if checkpoint_files:
    latest = checkpoint_files[-1]
    print(f'Resuming from {latest.name}...')
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt.get('val_loss', float('inf'))
    print(f'Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.5f}')
elif best_model_path.exists():
    print(f'Found best model, resuming...')
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt.get('val_loss', float('inf'))
    print(f'Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.5f}')

print(f'\nStarting training: epoch {start_epoch} -> {CONFIG["num_epochs"]}')
print(f'Batch size: {CONFIG["batch_size"]}, LR: {CONFIG["learning_rate"]}')
print()

# ============================================================================
# TRAINING LOOP
# ============================================================================

NUM_EPOCHS = CONFIG['num_epochs']

def save_progress(epoch, train_loss, val_loss, lr, elapsed, status='training'):
    """Write JSON progress file for remote monitoring."""
    progress = {
        'status': status,
        'epoch': epoch,
        'total_epochs': NUM_EPOCHS,
        'train_loss': round(train_loss, 5),
        'val_loss': round(val_loss, 5),
        'best_val_loss': round(best_val_loss, 5),
        'lr': lr,
        'epoch_time_sec': round(elapsed, 1),
        'total_tracks': total_tracks,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()

    # ---- Train ----
    model.train()
    running_loss = 0.0
    num_batches = 0

    for mel_batch, onset_tgt, frame_tgt, vel_tgt in train_loader:
        mel_batch = mel_batch.to(device)
        onset_tgt = onset_tgt.to(device)
        frame_tgt = frame_tgt.to(device)
        vel_tgt = vel_tgt.to(device)

        onset_pred, frame_pred, vel_pred = model(mel_batch)

        loss_onset = onset_loss_fn(onset_pred, onset_tgt)
        loss_frame = frame_loss_fn(frame_pred, frame_tgt)

        onset_mask = onset_tgt > 0.5
        if onset_mask.any():
            loss_vel = velocity_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask])
        else:
            loss_vel = torch.tensor(0.0, device=device)

        loss = loss_onset + loss_frame + 0.5 * loss_vel

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    train_loss = running_loss / max(num_batches, 1)

    # ---- Validation ----
    model.eval()
    val_running = 0.0
    val_batches = 0

    with torch.no_grad():
        for mel_batch, onset_tgt, frame_tgt, vel_tgt in val_loader:
            mel_batch = mel_batch.to(device)
            onset_tgt = onset_tgt.to(device)
            frame_tgt = frame_tgt.to(device)
            vel_tgt = vel_tgt.to(device)

            onset_pred, frame_pred, vel_pred = model(mel_batch)

            loss_onset = onset_loss_fn(onset_pred, onset_tgt)
            loss_frame = frame_loss_fn(frame_pred, frame_tgt)

            onset_mask = onset_tgt > 0.5
            if onset_mask.any():
                loss_vel = velocity_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask])
            else:
                loss_vel = torch.tensor(0.0, device=device)

            loss = loss_onset + loss_frame + 0.5 * loss_vel
            val_running += loss.item()
            val_batches += 1

    val_loss = val_running / max(val_batches, 1)
    scheduler.step(val_loss)

    elapsed = time.time() - epoch_start
    lr = optimizer.param_groups[0]['lr']

    print(f'Epoch {epoch:3d}/{NUM_EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
          f'LR: {lr:.2e} | Time: {elapsed:.0f}s')

    save_progress(epoch, train_loss, val_loss, lr, elapsed)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': CONFIG,
        }, str(best_model_path))
        print(f'  >>> Best model saved (val_loss={val_loss:.5f})')

    if (epoch + 1) % 5 == 0:
        ckpt_path = OUTPUT_DIR / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': CONFIG,
        }, str(ckpt_path))
        print(f'  Checkpoint: {ckpt_path.name}')

save_progress(NUM_EPOCHS - 1, train_loss, val_loss, lr, elapsed, status='complete')

print(f'\nTraining complete! Best val_loss: {best_val_loss:.5f}')
print(f'Model saved to: {best_model_path}')
