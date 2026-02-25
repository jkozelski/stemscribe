#!/usr/bin/env python3
"""
Bass Transcription CRNN Training Script — RunPod Edition
=========================================================
Trains on Slakh2100-redux (synthesized bass stems + aligned MIDI).
Output: best_bass_model.pt → backend/models/pretrained/

Usage on RunPod:
    python3 train_bass_runpod.py

Architecture: CQT (84 bins) → Conv2D (freq-only MaxPool) → BiLSTM → onset/frame/velocity heads (4 strings × 24 frets)
"""

import os
import sys
import time
import subprocess
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG — must match inference in bass_transcriber.py EXACTLY
# ============================================================================

CONFIG = {
    'sample_rate': 22050,
    'hop_length': 256,
    'n_bins': 84,           # CQT bins (7 octaves × 12 bins/octave)
    'bins_per_octave': 12,
    'chunk_duration': 5.0,  # seconds per training chunk
    'num_strings': 4,
    'num_frets': 24,
    'tuning': [28, 33, 38, 43],  # E1, A1, D2, G2 (4-string bass, standard)
    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 50,
}

SAMPLE_RATE = CONFIG['sample_rate']
HOP_LENGTH = CONFIG['hop_length']
N_BINS = CONFIG['n_bins']
BINS_PER_OCTAVE = CONFIG['bins_per_octave']
CHUNK_DURATION = CONFIG['chunk_duration']
NUM_STRINGS = CONFIG['num_strings']
NUM_FRETS = CONFIG['num_frets']
TUNING = CONFIG['tuning']
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)

# Output directory (RunPod persistent storage)
OUTPUT_DIR = Path('/workspace/bass_model_results')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path('/workspace/slakh_bass')
SLAKH_ROOT = Path('/workspace/slakh2100_flac_redux')

print(f'Chunk: {CHUNK_DURATION}s = {CHUNK_SAMPLES} samples = {CHUNK_FRAMES} frames')
print(f'Bass tuning (MIDI): {TUNING} = E1, A1, D2, G2')
print(f'Output shape per chunk: ({NUM_STRINGS}, {NUM_FRETS}, {CHUNK_FRAMES})')
print(f'Total output neurons: {NUM_STRINGS * NUM_FRETS} = {NUM_STRINGS}x{NUM_FRETS}')
print(f'Output: {OUTPUT_DIR}')
print()

# ============================================================================
# GPU CHECK
# ============================================================================

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {gpu_name} ({gpu_mem:.1f} GB)')
else:
    print('WARNING: No GPU detected! Training will be very slow.')
    sys.exit(1)

device = torch.device('cuda')

# ============================================================================
# INSTALL DEPENDENCIES
# ============================================================================

def install_deps():
    """Install required packages and system tools."""
    if shutil.which('unzip') is None:
        print('Installing unzip...')
        subprocess.check_call(['apt-get', 'update', '-qq'])
        subprocess.check_call(['apt-get', 'install', '-y', '-qq', 'unzip'])

    deps = ['librosa', 'pretty_midi', 'soundfile', 'pyyaml', 'tqdm']
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing.append(dep)
    if missing:
        print(f'Installing: {", ".join(missing)}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)

install_deps()

import librosa
import pretty_midi
import yaml
from tqdm import tqdm

# ============================================================================
# DOWNLOAD SLAKH2100-REDUX
# ============================================================================

def download_slakh():
    """Download Slakh2100-redux and extract bass stems."""
    DATA_DIR.mkdir(exist_ok=True)

    # Check if bass stems already extracted
    existing_tracks = list(DATA_DIR.glob('*_Track*'))
    if len(existing_tracks) > 100:
        print(f'Found {len(existing_tracks)} existing bass tracks, skipping download')
        return

    # Check if Slakh already extracted but bass not yet separated
    if SLAKH_ROOT.exists() and len(list(SLAKH_ROOT.glob('**/metadata.yaml'))) > 100:
        print(f'Slakh already extracted, will extract bass stems...')
    else:
        print('Downloading Slakh2100-redux from Zenodo...')
        print('This is ~97 GB — will take a while')
        print()

        tar_path = '/workspace/slakh2100.tar.gz'
        total, used, free = shutil.disk_usage('/workspace')
        free_gb = free / 1e9
        print(f'Disk space: {free_gb:.1f} GB free')

        if free_gb < 200:
            print('WARNING: Low disk space. Need ~200GB for download + extraction.')
            print('Clean up old datasets first (e.g., MAESTRO) if needed.')

        # Download
        subprocess.run([
            'wget', '-q', '--show-progress', '-O', tar_path,
            'https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1'
        ], check=True)

        print('Extracting (this takes a while)...')
        subprocess.run(['tar', 'xzf', tar_path, '-C', '/workspace/'], check=True)

        # Delete tar to free space
        if os.path.exists(tar_path):
            os.unlink(tar_path)
            print('Deleted tar.gz to free disk space')

    # Extract bass stems from Slakh
    print('\nExtracting bass stems from Slakh2100...')
    BASS_PROGRAMS = set(range(32, 40))
    bass_tracks = []
    skipped = 0

    for split in ['train', 'validation', 'test']:
        split_dir = SLAKH_ROOT / split
        if not split_dir.exists():
            print(f'Split {split} not found, skipping')
            continue

        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir():
                continue

            metadata_file = track_dir / 'metadata.yaml'
            if not metadata_file.exists():
                skipped += 1
                continue

            with open(metadata_file) as f:
                meta = yaml.safe_load(f)

            stems_meta = meta.get('stems', {})
            for stem_id, stem_info in stems_meta.items():
                program = stem_info.get('program_num', -1)
                is_drum = stem_info.get('is_drum', False)

                if is_drum or program not in BASS_PROGRAMS:
                    continue

                audio_file = track_dir / 'stems' / f'{stem_id}.flac'
                midi_file = track_dir / 'MIDI' / f'{stem_id}.mid'

                if audio_file.exists() and midi_file.exists():
                    track_name = f"{split}_{track_dir.name}_{stem_id}"
                    out_dir = DATA_DIR / track_name
                    out_dir.mkdir(exist_ok=True)

                    shutil.copy2(audio_file, out_dir / 'bass.flac')
                    shutil.copy2(midi_file, out_dir / 'bass.mid')

                    bass_tracks.append({
                        'track': track_name,
                        'split': split,
                        'program': program,
                        'audio': str(out_dir / 'bass.flac'),
                        'midi': str(out_dir / 'bass.mid'),
                    })

    print(f'\nExtracted {len(bass_tracks)} bass stems')
    print(f'  Skipped {skipped} tracks (no metadata)')
    print(f'  Train: {sum(1 for t in bass_tracks if t["split"] == "train")}')
    print(f'  Val: {sum(1 for t in bass_tracks if t["split"] == "validation")}')
    print(f'  Test: {sum(1 for t in bass_tracks if t["split"] == "test")}')

    # Clean up full Slakh to save disk
    if SLAKH_ROOT.exists():
        print('\nCleaning up full Slakh archive to save disk space...')
        shutil.rmtree(SLAKH_ROOT, ignore_errors=True)
        print('Freed disk space')

    if len(bass_tracks) < 100:
        print(f'\nWARNING: Only found {len(bass_tracks)} bass tracks.')
        print('Expected ~1000+. Check disk space and re-run.')

download_slakh()

# ============================================================================
# BUILD TRACK LIST FROM EXTRACTED DATA
# ============================================================================

bass_tracks = []
for track_dir in sorted(DATA_DIR.iterdir()):
    if not track_dir.is_dir():
        continue
    audio = track_dir / 'bass.flac'
    midi = track_dir / 'bass.mid'
    if audio.exists() and midi.exists():
        name = track_dir.name
        split = name.split('_')[0] if name.startswith(('train', 'validation', 'test')) else 'train'
        bass_tracks.append({
            'track': name,
            'split': split,
            'audio': str(audio),
            'midi': str(midi),
        })

print(f'\nTotal bass tracks: {len(bass_tracks)}')

# ============================================================================
# DATASET & PREPROCESSING
# ============================================================================

def midi_note_to_string_fret(midi_note, tuning=TUNING, num_frets=NUM_FRETS):
    """Map a MIDI note to the best (string, fret) position on bass."""
    candidates = []
    for s, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret < num_frets:
            candidates.append((s, fret, fret + s * 0.1))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2])
    return candidates[0][0], candidates[0][1]


def midi_to_targets(midi_path, total_frames, sr=SAMPLE_RATE, hop=HOP_LENGTH):
    """Convert MIDI file to onset/frame/velocity target tensors."""
    onset = np.zeros((NUM_STRINGS, NUM_FRETS, total_frames), dtype=np.float32)
    frame = np.zeros((NUM_STRINGS, NUM_FRETS, total_frames), dtype=np.float32)
    velocity = np.zeros((NUM_STRINGS, NUM_FRETS, total_frames), dtype=np.float32)

    frames_per_sec = sr / hop

    try:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return onset, frame, velocity

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            result = midi_note_to_string_fret(note.pitch)
            if result is None:
                continue
            s, f = result
            onset_frame = int(note.start * frames_per_sec)
            offset_frame = int(note.end * frames_per_sec)
            if onset_frame >= total_frames:
                continue
            offset_frame = min(offset_frame, total_frames - 1)
            vel = note.velocity / 127.0
            onset[s, f, onset_frame] = 1.0
            frame[s, f, onset_frame:offset_frame + 1] = 1.0
            velocity[s, f, onset_frame] = vel

    return onset, frame, velocity


class BassTranscriptionDataset(Dataset):
    """Dataset yielding (cqt_chunk, onset, frame, velocity) from bass audio + MIDI."""

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
            audio, sr = librosa.load(track['audio'], sr=SAMPLE_RATE, mono=True)
        except Exception:
            audio = np.zeros(CHUNK_SAMPLES * 2, dtype=np.float32)

        cqt = np.abs(librosa.cqt(
            audio, sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE
        ))
        cqt = np.log(cqt + 1e-8).astype(np.float32)
        total_frames = cqt.shape[1]
        onset, frame, velocity = midi_to_targets(track['midi'], total_frames)
        result = (cqt, onset, frame, velocity, total_frames)

        if len(self._cache) < 200:
            self._cache[track_idx] = result
        return result

    def __getitem__(self, idx):
        cqt, onset, frame, velocity, total_frames = self._load_track(idx)

        if total_frames <= CHUNK_FRAMES:
            pad_frames = CHUNK_FRAMES - total_frames
            cqt_chunk = np.pad(cqt, ((0, 0), (0, pad_frames)))
            onset_chunk = np.pad(onset, ((0, 0), (0, 0), (0, pad_frames)))
            frame_chunk = np.pad(frame, ((0, 0), (0, 0), (0, pad_frames)))
            vel_chunk = np.pad(velocity, ((0, 0), (0, 0), (0, pad_frames)))
        else:
            start = np.random.randint(0, total_frames - CHUNK_FRAMES)
            cqt_chunk = cqt[:, start:start + CHUNK_FRAMES]
            onset_chunk = onset[:, :, start:start + CHUNK_FRAMES]
            frame_chunk = frame[:, :, start:start + CHUNK_FRAMES]
            vel_chunk = velocity[:, :, start:start + CHUNK_FRAMES]

        return (
            torch.from_numpy(cqt_chunk),
            torch.from_numpy(onset_chunk),
            torch.from_numpy(frame_chunk),
            torch.from_numpy(vel_chunk),
        )

# ============================================================================
# BUILD TRAIN/VAL SPLITS
# ============================================================================

import random

train_tracks = [t for t in bass_tracks if t['split'] == 'train']
val_tracks = [t for t in bass_tracks if t['split'] in ('validation', 'test')]

if len(val_tracks) < 10:
    random.shuffle(bass_tracks)
    split_idx = int(len(bass_tracks) * 0.9)
    train_tracks = bass_tracks[:split_idx]
    val_tracks = bass_tracks[split_idx:]

print(f'Train: {len(train_tracks)} tracks')
print(f'Val:   {len(val_tracks)} tracks')

train_dataset = BassTranscriptionDataset(train_tracks, chunks_per_track=8)
val_dataset = BassTranscriptionDataset(val_tracks, chunks_per_track=4)

train_loader = DataLoader(
    train_dataset, batch_size=CONFIG['batch_size'],
    shuffle=True, num_workers=2, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=CONFIG['batch_size'],
    shuffle=False, num_workers=2, pin_memory=True
)

print(f'Train batches/epoch: {len(train_loader)}')
print(f'Val batches/epoch:   {len(val_loader)}')

# Sanity check
cqt_b, onset_b, frame_b, vel_b = next(iter(train_loader))
print(f'\nBatch shapes:')
print(f'  CQT:      {cqt_b.shape}')
print(f'  Onset:    {onset_b.shape}')
print(f'  Frame:    {frame_b.shape}')
print(f'  Velocity: {vel_b.shape}')
print(f'  Onset active: {onset_b.sum():.0f} / {onset_b.numel()}')
print(f'  Frame active: {frame_b.sum():.0f} / {frame_b.numel()}')

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class BassTranscriptionModel(nn.Module):
    """
    CRNN for bass transcription: CQT -> Conv2D -> BiLSTM -> onset/frame/velocity
    Input:  (batch, n_bins=84, time)
    Output: onset/frame/velocity each (batch, 4, 24, time)
    """

    def __init__(self, n_bins=84, num_strings=4, num_frets=24,
                 hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_strings = num_strings
        self.num_frets = num_frets
        output_size = num_strings * num_frets  # 96

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(dropout),
        )

        cnn_output_dim = 128 * 10  # 1280

        self.onset_lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.frame_lstm = nn.LSTM(
            input_size=cnn_output_dim + output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        lstm_output_dim = hidden_size * 2

        self.onset_head = nn.Linear(lstm_output_dim, output_size)
        self.frame_head = nn.Linear(lstm_output_dim, output_size)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_output_dim, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        batch, n_bins, time = x.shape
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

        def reshape_output(t):
            return t.permute(0, 2, 1).reshape(
                batch, self.num_strings, self.num_frets, time
            )

        return reshape_output(onset_pred), reshape_output(frame_pred), reshape_output(velocity_pred)

# ============================================================================
# TRAINING SETUP
# ============================================================================

model = BassTranscriptionModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f'\nModel parameters: {total_params:,}')

optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=5, factor=0.5
)

onset_loss_fn = nn.BCELoss()
frame_loss_fn = nn.BCELoss()
velocity_loss_fn = nn.MSELoss()

# Resume from checkpoint if available
start_epoch = 0
best_val_loss = float('inf')
best_model_path = OUTPUT_DIR / 'best_bass_model.pt'

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

print(f'\nStarting training from epoch {start_epoch}...')
print(f'Epochs: {CONFIG["num_epochs"]}, Batch size: {CONFIG["batch_size"]}')
print(f'LR: {CONFIG["learning_rate"]}')

# ============================================================================
# TRAINING LOOP
# ============================================================================

NUM_EPOCHS = CONFIG['num_epochs']
train_losses = []
val_losses = []

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()

    # ---- Train ----
    model.train()
    running_loss = 0.0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}')
    for cqt_batch, onset_tgt, frame_tgt, vel_tgt in pbar:
        cqt_batch = cqt_batch.to(device)
        onset_tgt = onset_tgt.to(device)
        frame_tgt = frame_tgt.to(device)
        vel_tgt = vel_tgt.to(device)

        onset_pred, frame_pred, vel_pred = model(cqt_batch)

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
        pbar.set_postfix(loss=f'{loss.item():.4f}')

    train_loss = running_loss / max(num_batches, 1)
    train_losses.append(train_loss)

    # ---- Validation ----
    model.eval()
    val_running = 0.0
    val_batches = 0

    with torch.no_grad():
        for cqt_batch, onset_tgt, frame_tgt, vel_tgt in val_loader:
            cqt_batch = cqt_batch.to(device)
            onset_tgt = onset_tgt.to(device)
            frame_tgt = frame_tgt.to(device)
            vel_tgt = vel_tgt.to(device)

            onset_pred, frame_pred, vel_pred = model(cqt_batch)

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
    val_losses.append(val_loss)

    scheduler.step(val_loss)

    elapsed = time.time() - epoch_start
    lr = optimizer.param_groups[0]['lr']

    print(f'Epoch {epoch}/{NUM_EPOCHS} | '
          f'Train: {train_loss:.4f} | Val: {val_loss:.4f} | '
          f'LR: {lr:.2e} | Time: {elapsed:.0f}s')

    # Save best model
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

    # Periodic checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_path = OUTPUT_DIR / f'checkpoint_epoch_{epoch:03d}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': CONFIG,
        }, str(ckpt_path))
        print(f'  Checkpoint saved: {ckpt_path.name}')

print(f'\nTraining complete! Best val_loss: {best_val_loss:.5f}')
print(f'Best model saved to: {best_model_path}')

# ============================================================================
# QUICK EVAL
# ============================================================================

ckpt = torch.load(str(best_model_path), map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

print(f'\nLoaded best model from epoch {ckpt["epoch"]}, val_loss={ckpt["val_loss"]:.5f}')

with torch.no_grad():
    test_input = torch.randn(1, N_BINS, CHUNK_FRAMES).to(device)
    onset, frame, vel = model(test_input)
    print(f'Inference test:')
    print(f'  Input:    {test_input.shape}')
    print(f'  Onset:    {onset.shape}, range [{onset.min():.3f}, {onset.max():.3f}]')
    print(f'  Frame:    {frame.shape}, range [{frame.min():.3f}, {frame.max():.3f}]')
    print(f'  Velocity: {vel.shape}, range [{vel.min():.3f}, {vel.max():.3f}]')

print(f'\nModel ready! Copy best_bass_model.pt to backend/models/pretrained/')
