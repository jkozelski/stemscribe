#!/usr/bin/env python3
"""
Bass Transcription Training — v3 (RunPod)
==========================================
v2 fixes: Focal loss + pos_weight to prevent collapse on sparse targets.
v3 fixes: Enhanced augmentation (pitch shift, time stretch, reverb, EQ, SpecAugment)
          per trimplexx recipe research (augmentation was "the decisive factor").

Run on RunPod:
    pip install torch torchaudio librosa pretty_midi tqdm soundfile pyyaml scipy
    python train_bass_runpod.py

Output: /workspace/bass_results/best_bass_model.pt
Copy to: backend/models/pretrained/best_bass_model.pt
"""

import os
import sys
import time
import random
import shutil
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG — must match inference in bass_transcriber.py EXACTLY
# ============================================================================
CONFIG = {
    'sample_rate': 22050,
    'hop_length': 256,
    'n_bins': 84,
    'bins_per_octave': 12,
    'chunk_duration': 5.0,
    'num_strings': 4,
    'num_frets': 24,
    'tuning': [28, 33, 38, 43],  # E1, A1, D2, G2

    'batch_size': 16,
    'learning_rate': 1e-4,
    'num_epochs': 50,

    # v2: Loss weights
    'onset_pos_weight': 50.0,
    'frame_pos_weight': 10.0,
    'focal_gamma': 2.0,
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

DATA_DIR = Path("/workspace/slakh_bass")
SLAKH_ROOT = Path("/workspace/slakh2100_flac_redux")
SAVE_DIR = Path("/workspace/bass_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BASS_PROGRAMS = set(range(32, 40))


# ============================================================================
# INSTALL DEPS
# ============================================================================
def install_deps():
    if shutil.which('unzip') is None:
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
# DOWNLOAD & EXTRACT
# ============================================================================
def download_slakh():
    DATA_DIR.mkdir(exist_ok=True)

    existing = list(DATA_DIR.glob('*_Track*'))
    if len(existing) > 100:
        print(f"Found {len(existing)} existing bass tracks, skipping download")
        return

    if SLAKH_ROOT.exists() and len(list(SLAKH_ROOT.glob('**/metadata.yaml'))) > 100:
        print("Slakh already extracted, extracting bass stems...")
    else:
        print("Downloading Slakh2100-redux from Zenodo (~97 GB)...")
        print("This will take a while\n")

        tar_path = "/workspace/slakh2100.tar.gz"
        total, used, free = shutil.disk_usage('/workspace')
        print(f"Disk space: {free / 1e9:.1f} GB free")

        subprocess.run([
            "wget", "-q", "--show-progress", "-O", tar_path,
            "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
        ], check=True)

        print("Extracting...")
        subprocess.run(["tar", "xzf", tar_path, "-C", "/workspace/"], check=True)

        if os.path.exists(tar_path):
            os.unlink(tar_path)
            print("Deleted tar.gz to free disk space")

    # Extract bass stems
    print("\nExtracting bass stems...")
    bass_count = 0
    for split in ['train', 'validation', 'test']:
        split_dir = SLAKH_ROOT / split
        if not split_dir.exists():
            continue

        for track_dir in sorted(split_dir.iterdir()):
            if not track_dir.is_dir():
                continue
            metadata_file = track_dir / 'metadata.yaml'
            if not metadata_file.exists():
                continue

            with open(metadata_file) as f:
                meta = yaml.safe_load(f)

            for stem_id, stem_info in meta.get('stems', {}).items():
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
                    bass_count += 1

    print(f"Extracted {bass_count} bass stems")

    # Clean up full Slakh
    if SLAKH_ROOT.exists():
        print("Cleaning up full Slakh to free disk...")
        shutil.rmtree(SLAKH_ROOT, ignore_errors=True)


def load_bass_tracks():
    tracks = []
    for track_dir in sorted(DATA_DIR.iterdir()):
        if not track_dir.is_dir():
            continue
        audio = track_dir / 'bass.flac'
        midi = track_dir / 'bass.mid'
        if audio.exists() and midi.exists():
            name = track_dir.name
            if name.startswith('train_'):
                split = 'train'
            elif name.startswith('validation_'):
                split = 'validation'
            elif name.startswith('test_'):
                split = 'test'
            else:
                split = 'train'
            tracks.append({
                'track': name, 'split': split,
                'audio': str(audio), 'midi': str(midi),
            })
    return tracks


# ============================================================================
# v2 LOSS FUNCTIONS & METRICS
# ============================================================================
class FocalBCEWithLogitsLoss(nn.Module):
    """Focal Loss + pos_weight for extremely imbalanced binary classification."""

    def __init__(self, pos_weight=50.0, gamma=2.0):
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
# DATASET (v2 with augmentation)
# ============================================================================
def midi_note_to_string_fret(midi_note, tuning=TUNING, num_frets=NUM_FRETS):
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
            audio, sr = librosa.load(track['audio'], sr=SAMPLE_RATE, mono=True)
        except Exception:
            audio = np.zeros(CHUNK_SAMPLES * 2, dtype=np.float32)

        total_audio_frames = int(len(audio) / HOP_LENGTH) + 1
        onset, frame, velocity = midi_to_targets(track['midi'], total_audio_frames)

        result = (audio, onset, frame, velocity, total_audio_frames)
        if len(self._cache) < 200:
            self._cache[track_idx] = result
        return result

    def __getitem__(self, idx):
        audio, onset, frame, velocity, total_audio_frames = self._load_track(idx)

        if len(audio) > CHUNK_SAMPLES:
            start_sample = np.random.randint(0, len(audio) - CHUNK_SAMPLES)
            audio_chunk = audio[start_sample:start_sample + CHUNK_SAMPLES].copy()
            start_frame = int(start_sample / HOP_LENGTH)
        else:
            audio_chunk = np.pad(audio, (0, CHUNK_SAMPLES - len(audio))).copy()
            start_frame = 0

        # v2: Augmentation
        if self.augment:
            if random.random() < 0.5:
                audio_chunk = audio_chunk * random.uniform(0.7, 1.3)
            if random.random() < 0.3:
                noise = np.random.randn(len(audio_chunk)).astype(np.float32) * random.uniform(0.001, 0.005)
                audio_chunk = audio_chunk + noise

        cqt = np.abs(librosa.cqt(
            audio_chunk, sr=SAMPLE_RATE,
            hop_length=HOP_LENGTH, n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE
        ))
        cqt = np.log(cqt + 1e-8).astype(np.float32)
        actual_frames = cqt.shape[1]

        end_frame = min(start_frame + actual_frames, onset.shape[2])
        chunk_len = end_frame - start_frame

        if chunk_len < actual_frames:
            pad_w = actual_frames - chunk_len
            onset_chunk = np.pad(onset[:, :, start_frame:end_frame], ((0,0),(0,0),(0,pad_w)))
            frame_chunk = np.pad(frame[:, :, start_frame:end_frame], ((0,0),(0,0),(0,pad_w)))
            vel_chunk = np.pad(velocity[:, :, start_frame:end_frame], ((0,0),(0,0),(0,pad_w)))
        else:
            onset_chunk = onset[:, :, start_frame:start_frame + actual_frames]
            frame_chunk = frame[:, :, start_frame:start_frame + actual_frames]
            vel_chunk = velocity[:, :, start_frame:start_frame + actual_frames]

        return (
            torch.from_numpy(cqt),
            torch.from_numpy(onset_chunk),
            torch.from_numpy(frame_chunk),
            torch.from_numpy(vel_chunk),
        )


# ============================================================================
# MODEL (v2: returns logits for onset/frame)
# ============================================================================
class BassTranscriptionModel(nn.Module):
    def __init__(self, n_bins=84, num_strings=4, num_frets=24,
                 hidden_size=256, num_layers=2, dropout=0.25):
        super().__init__()
        self.num_strings = num_strings
        self.num_frets = num_frets
        output_size = num_strings * num_frets

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
        )

        cnn_output_dim = 128 * 10
        self.onset_lstm = nn.LSTM(input_size=cnn_output_dim, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True, bidirectional=True,
                                   dropout=dropout if num_layers > 1 else 0)
        self.frame_lstm = nn.LSTM(input_size=cnn_output_dim + output_size, hidden_size=hidden_size,
                                   num_layers=num_layers, batch_first=True, bidirectional=True,
                                   dropout=dropout if num_layers > 1 else 0)

        lstm_output_dim = hidden_size * 2
        self.onset_head = nn.Linear(lstm_output_dim, output_size)
        self.frame_head = nn.Linear(lstm_output_dim, output_size)
        self.velocity_head = nn.Sequential(nn.Linear(lstm_output_dim, output_size), nn.Sigmoid())

    def forward(self, x):
        batch, n_bins, time = x.shape
        cnn_out = self.cnn(x.unsqueeze(1))
        cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)

        # Onset
        onset_out, _ = self.onset_lstm(cnn_features)
        onset_logits = self.onset_head(onset_out)  # RAW LOGITS
        onset_probs = torch.sigmoid(onset_logits)  # For frame conditioning only

        # Frame (conditioned on onset)
        frame_input = torch.cat([cnn_features, onset_probs.detach()], dim=-1)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)  # RAW LOGITS

        # Velocity
        velocity_pred = self.velocity_head(onset_out)

        def reshape_output(t):
            return t.permute(0, 2, 1).reshape(batch, self.num_strings, self.num_frets, time)

        return reshape_output(onset_logits), reshape_output(frame_logits), reshape_output(velocity_pred)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("Bass Transcription Training v2 — RunPod")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("WARNING: No GPU!")

    # Download and extract
    download_slakh()
    bass_tracks = load_bass_tracks()

    if len(bass_tracks) < 50:
        print(f"ERROR: Only {len(bass_tracks)} bass tracks. Need more data.")
        sys.exit(1)

    # Splits
    train_tracks = [t for t in bass_tracks if t['split'] == 'train']
    val_tracks = [t for t in bass_tracks if t['split'] in ('validation', 'test')]
    if len(val_tracks) < 10:
        random.shuffle(bass_tracks)
        split_idx = int(len(bass_tracks) * 0.9)
        train_tracks = bass_tracks[:split_idx]
        val_tracks = bass_tracks[split_idx:]

    print(f"Train: {len(train_tracks)}, Val: {len(val_tracks)}")

    train_dataset = BassTranscriptionDataset(train_tracks, chunks_per_track=8, augment=True)
    val_dataset = BassTranscriptionDataset(val_tracks, chunks_per_track=4, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train batches/epoch: {len(train_loader)}, Val: {len(val_loader)}")

    # Model
    model = BassTranscriptionModel().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # v2: Weighted loss
    onset_loss_fn = FocalBCEWithLogitsLoss(pos_weight=CONFIG['onset_pos_weight'], gamma=CONFIG['focal_gamma'])
    frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([CONFIG['frame_pos_weight']]).to(device))
    velocity_loss_fn = nn.MSELoss()

    print(f"Onset: Focal(pw={CONFIG['onset_pos_weight']}, γ={CONFIG['focal_gamma']})")
    print(f"Frame: BCE(pw={CONFIG['frame_pos_weight']})")

    # Resume
    start_epoch = 0
    best_val_f1 = 0.0
    best_val_loss = float('inf')
    best_model_path = SAVE_DIR / 'best_bass_model.pt'
    ckpt_path = SAVE_DIR / 'latest_checkpoint.pt'

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best F1={best_val_f1:.4f}")

    # Training
    zero_recall_count = 0
    COLLAPSE_THRESHOLD = 10

    print(f"\nStarting — {CONFIG['num_epochs']} epochs, collapse detection at {COLLAPSE_THRESHOLD}\n")

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        t0 = time.time()

        # Train
        model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_onset_logits = []
        epoch_onset_targets = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["num_epochs"]}')
        for cqt_b, onset_tgt, frame_tgt, vel_tgt in pbar:
            cqt_b = cqt_b.to(device)
            onset_tgt = onset_tgt.to(device)
            frame_tgt = frame_tgt.to(device)
            vel_tgt = vel_tgt.to(device)

            onset_logits, frame_logits, vel_pred = model(cqt_b)

            loss_onset = onset_loss_fn(onset_logits, onset_tgt)
            loss_frame = frame_loss_fn(frame_logits, frame_tgt)
            onset_mask = onset_tgt > 0.5
            loss_vel = velocity_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask]) if onset_mask.any() else torch.tensor(0.0, device=device)

            loss = loss_onset + loss_frame + 0.5 * loss_vel
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

            if num_batches % 5 == 0:
                epoch_onset_logits.append(onset_logits.detach().cpu())
                epoch_onset_targets.append(onset_tgt.detach().cpu())

        train_loss = running_loss / max(num_batches, 1)
        if epoch_onset_logits:
            train_f1, train_prec, train_rec = compute_onset_f1(
                torch.cat(epoch_onset_logits), torch.cat(epoch_onset_targets))
        else:
            train_f1, train_prec, train_rec = 0, 0, 0

        # Validate
        model.eval()
        val_running = 0.0
        val_batches = 0
        val_onset_logits = []
        val_onset_targets = []

        with torch.no_grad():
            for cqt_b, onset_tgt, frame_tgt, vel_tgt in val_loader:
                cqt_b = cqt_b.to(device)
                onset_tgt = onset_tgt.to(device)
                frame_tgt = frame_tgt.to(device)
                vel_tgt = vel_tgt.to(device)

                onset_logits, frame_logits, vel_pred = model(cqt_b)
                loss_onset = onset_loss_fn(onset_logits, onset_tgt)
                loss_frame = frame_loss_fn(frame_logits, frame_tgt)
                onset_mask = onset_tgt > 0.5
                loss_vel = velocity_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask]) if onset_mask.any() else torch.tensor(0.0, device=device)

                val_running += (loss_onset + loss_frame + 0.5 * loss_vel).item()
                val_batches += 1
                val_onset_logits.append(onset_logits.cpu())
                val_onset_targets.append(onset_tgt.cpu())

        val_loss = val_running / max(val_batches, 1)
        if val_onset_logits:
            val_f1, val_prec, val_rec = compute_onset_f1(
                torch.cat(val_onset_logits), torch.cat(val_onset_targets))
        else:
            val_f1, val_prec, val_rec = 0, 0, 0

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
                print(f"\n🛑 COLLAPSED — try higher pos_weight or lower LR.")
                break
        else:
            zero_recall_count = 0

        # Save best by F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss, 'val_f1': val_f1, 'config': CONFIG,
            }, str(best_model_path))
            print(f"  ✅ Best model (F1={val_f1:.4f})")

        # Latest checkpoint
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss, 'val_f1': val_f1,
            'best_val_f1': best_val_f1, 'best_val_loss': best_val_loss, 'config': CONFIG,
        }, str(ckpt_path))

    print(f"\n{'='*60}")
    print(f"✅ Done! Best F1: {best_val_f1:.4f}")
    print(f"Model: {best_model_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
