#!/usr/bin/env python3
"""
Guitar Transcription Training — Domain Adaptation from Kong Piano Checkpoint
=============================================================================
Fine-tunes the Kong et al. (2021) piano transcription CRNN on GuitarSet
using the Riley et al. (ICASSP 2024) domain adaptation recipe.

Two-phase training:
  Phase 1: Freeze CNN, train RNN + output heads (20 epochs, ~2h on A100)
  Phase 2: Full fine-tuning with low LR (80 epochs, ~6h on A100)

Run on RunPod:
    pip install torch torchaudio librosa jams mir_eval numpy scipy tqdm audiomentations
    python train_guitar_runpod.py

Output: /workspace/guitar_results/best_guitar_model.pt
Copy to: backend/models/pretrained/best_guitar_model.pt
"""

import os
import sys
import time
import random
import shutil
import subprocess
import warnings
import json
import math

warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ============================================================================
# CONFIG — must match inference in guitar_nn_transcriber.py EXACTLY
# ============================================================================
CONFIG = {
    'sample_rate': 16000,
    'hop_length': 160,
    'n_fft': 2048,
    'n_mels': 229,
    'guitar_midi_min': 40,   # E2
    'guitar_midi_max': 88,   # E6
    'n_pitches': 48,         # 88 - 40
    'chunk_seconds': 10.0,   # 10s chunks for training

    # Phase 1: Frozen CNN
    'phase1_epochs': 20,
    'phase1_lr': 5e-5,       # Higher LR for heads only

    # Phase 2: Full fine-tuning
    'phase2_epochs': 80,
    'phase2_lr': 1e-5,       # Low LR for fine-tuning (Riley et al.)

    'batch_size': 4,         # A100 can handle 4; reduce to 2 for T4

    # Loss weights
    'onset_pos_weight': 50.0,
    'frame_pos_weight': 10.0,
    'focal_gamma': 2.0,

    # Augmentation
    'augment': True,
    'gain_range': (-6, 6),          # dB
    'noise_snr_range': (20, 40),    # dB
    'pitch_shift_range': (-2, 2),   # semitones
    'spec_augment_freq_masks': 2,
    'spec_augment_time_masks': 2,
    'spec_augment_freq_width': 15,
    'spec_augment_time_width': 40,
}

SAMPLE_RATE = CONFIG['sample_rate']
HOP_LENGTH = CONFIG['hop_length']
N_MELS = CONFIG['n_mels']
N_PITCHES = CONFIG['n_pitches']
GUITAR_MIDI_MIN = CONFIG['guitar_midi_min']
GUITAR_MIDI_MAX = CONFIG['guitar_midi_max']
CHUNK_SECONDS = CONFIG['chunk_seconds']
CHUNK_SAMPLES = int(CHUNK_SECONDS * SAMPLE_RATE)
CHUNK_FRAMES = CHUNK_SAMPLES // HOP_LENGTH

DATA_DIR = Path("/workspace/guitarset")
SAVE_DIR = Path("/workspace/guitar_results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = SAVE_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# INSTALL DEPS
# ============================================================================
def install_deps():
    deps = [
        'librosa', 'jams', 'mir_eval', 'pretty_midi', 'tqdm',
        'soundfile', 'audiomentations',
        'piano_transcription_inference',
    ]
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace('-', '_').replace('piano_transcription_inference', 'piano_transcription_inference'))
        except ImportError:
            missing.append(dep)
    if missing:
        print(f'Installing: {", ".join(missing)}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + missing)


install_deps()

import librosa
import jams
import mir_eval
from tqdm import tqdm


# ============================================================================
# DOWNLOAD KONG CHECKPOINT & GUITARSET
# ============================================================================
def download_kong_checkpoint():
    """Download Kong piano transcription checkpoint from Zenodo."""
    ckpt_path = Path("/workspace/kong_piano_checkpoint.pth")
    if ckpt_path.exists():
        print(f"Kong checkpoint already exists ({ckpt_path.stat().st_size / 1e6:.1f} MB)")
        return ckpt_path

    print("Downloading Kong piano checkpoint from Zenodo (172 MB)...")
    url = "https://zenodo.org/record/4034264/files/CRNN_note_F1%3D0.9677_pedal_F1%3D0.9186.pth"
    subprocess.run(["wget", "-q", "--show-progress", "-O", str(ckpt_path), url], check=True)
    print(f"Downloaded: {ckpt_path.stat().st_size / 1e6:.1f} MB")
    return ckpt_path


def download_guitarset():
    """Download GuitarSet from Zenodo."""
    audio_dir = DATA_DIR / "audio_mono-mic"
    annot_dir = DATA_DIR / "annotation"

    if audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 300:
        print(f"GuitarSet already downloaded ({len(list(audio_dir.glob('*.wav')))} audio files)")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download audio (mono mic — clean guitar recordings)
    audio_zip = DATA_DIR / "audio_mono-mic.zip"
    if not audio_zip.exists():
        print("Downloading GuitarSet audio (657 MB)...")
        subprocess.run([
            "wget", "-q", "--show-progress", "-O", str(audio_zip),
            "https://zenodo.org/record/3371780/files/audio_mono-mic.zip"
        ], check=True)

    # Download annotations
    annot_zip = DATA_DIR / "annotation.zip"
    if not annot_zip.exists():
        print("Downloading GuitarSet annotations (39 MB)...")
        subprocess.run([
            "wget", "-q", "--show-progress", "-O", str(annot_zip),
            "https://zenodo.org/record/3371780/files/annotation.zip"
        ], check=True)

    # Extract
    print("Extracting...")
    subprocess.run(["unzip", "-qo", str(audio_zip), "-d", str(DATA_DIR)], check=True)
    subprocess.run(["unzip", "-qo", str(annot_zip), "-d", str(DATA_DIR)], check=True)

    # Clean up zips
    audio_zip.unlink(missing_ok=True)
    annot_zip.unlink(missing_ok=True)

    n_audio = len(list(audio_dir.glob("*.wav"))) if audio_dir.exists() else 0
    print(f"GuitarSet extracted: {n_audio} audio files")


# ============================================================================
# KONG MODEL ARCHITECTURE
# Replicated from bytedance/piano_transcription to load pretrained weights
# ============================================================================
def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias') and layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_gru(rnn):
    for name, param in rnn.named_parameters():
        if 'weight_ih' in name:
            nn.init.xavier_uniform_(param.data)
        elif 'weight_hh' in name:
            nn.init.orthogonal_(param.data)
        elif 'bias' in name:
            param.data.fill_(0.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

    def forward(self, x, pool_size=(2, 2), pool_type='avg'):
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        return x


class AcousticModelCRnn8Dropout(nn.Module):
    """Kong's CRNN acoustic model: 4 ConvBlocks -> FC -> BiGRU -> Linear"""
    def __init__(self, classes_num, midfeat, momentum):
        super().__init__()
        self.conv_block1 = ConvBlock(1, 48)
        self.conv_block2 = ConvBlock(48, 64)
        self.conv_block3 = ConvBlock(64, 96)
        self.conv_block4 = ConvBlock(96, 128)

        self.fc5 = nn.Linear(midfeat, 768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        init_layer(self.fc5)
        init_bn(self.bn5)

        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2,
                          bias=True, batch_first=True, dropout=0.0, bidirectional=True)
        init_gru(self.gru)

        self.fc = nn.Linear(512, classes_num, bias=True)
        init_layer(self.fc)

    def forward(self, input):
        # input: (batch, 1, time, mel_bins)
        x = self.conv_block1(input, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(1, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        # (batch, 128, time, mel_bins // 16)
        x = x.transpose(1, 2).flatten(2)  # (batch, time, 128 * mel_bins // 16)
        x = F.relu(self.bn5(self.fc5(x).transpose(1, 2)).transpose(1, 2))
        x = F.dropout(x, p=0.5, training=self.training)

        x, _ = self.gru(x)
        x = self.fc(x)
        return x


class GuitarTranscriptionModel(nn.Module):
    """
    Guitar transcription model based on Kong's piano architecture.

    Uses same CRNN backbone (4 ConvBlocks -> FC -> BiGRU) but with
    output heads sized for guitar range (48 pitches: E2-E6) instead
    of piano (88 keys).

    Follows Kong's conditioning pattern:
    - Onset output conditions frame prediction
    - Separate GRU for onset regression and frame classification
    """
    def __init__(self, n_pitches=48):
        super().__init__()
        self.n_pitches = n_pitches

        # midfeat = 128 * (229 // 16) = 128 * 14 = 1792
        midfeat = 128 * (N_MELS // 16)

        # Four parallel acoustic models (matching Kong architecture)
        self.frame_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)
        self.onset_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)
        self.offset_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)
        self.velocity_model = AcousticModelCRnn8Dropout(n_pitches, midfeat, momentum=0.01)

        # Conditioning GRUs (matching Kong)
        self.onset_gru = nn.GRU(input_size=n_pitches * 2, hidden_size=n_pitches,
                                num_layers=1, bias=True, batch_first=True, bidirectional=True)
        init_gru(self.onset_gru)
        self.onset_fc = nn.Linear(n_pitches * 2, n_pitches, bias=True)
        init_layer(self.onset_fc)

        self.frame_gru = nn.GRU(input_size=n_pitches * 3, hidden_size=n_pitches,
                                num_layers=1, bias=True, batch_first=True, bidirectional=True)
        init_gru(self.frame_gru)
        self.frame_fc = nn.Linear(n_pitches * 2, n_pitches, bias=True)
        init_layer(self.frame_fc)

    def forward(self, mel):
        """
        Args:
            mel: (batch, time, n_mels) log-mel spectrogram

        Returns:
            onset_output: (batch, time, n_pitches) — onset logits
            frame_output: (batch, time, n_pitches) — frame logits
            velocity_output: (batch, time, n_pitches) — velocity [0, 1]
        """
        # Reshape for CNN: (batch, 1, time, n_mels)
        x = mel.unsqueeze(1)

        # Run parallel acoustic models (returns logits)
        frame_out = self.frame_model(x)       # (batch, time, n_pitches)
        onset_out = self.onset_model(x)       # (batch, time, n_pitches)
        offset_out = self.offset_model(x)     # (batch, time, n_pitches)
        velocity_out = self.velocity_model(x) # (batch, time, n_pitches)

        # Condition onset on velocity (Kong's recipe)
        velocity_sigmoid = torch.sigmoid(velocity_out)
        onset_concat = torch.cat([onset_out, onset_out * velocity_sigmoid], dim=2)
        onset_gru_out, _ = self.onset_gru(onset_concat)
        onset_output = self.onset_fc(onset_gru_out)

        # Condition frame on onset + offset (Kong's recipe)
        onset_sigmoid = torch.sigmoid(onset_output)
        offset_sigmoid = torch.sigmoid(offset_out)
        frame_concat = torch.cat([frame_out, onset_sigmoid, offset_sigmoid], dim=2)
        frame_gru_out, _ = self.frame_gru(frame_concat)
        frame_output = self.frame_fc(frame_gru_out)

        # Velocity uses sigmoid for [0, 1] output
        velocity_output = velocity_sigmoid

        return onset_output, frame_output, velocity_output


# ============================================================================
# WEIGHT LOADING FROM KONG CHECKPOINT
# ============================================================================
def load_kong_weights(model, checkpoint_path):
    """
    Load Kong piano checkpoint weights into guitar model.

    Strategy:
    - CNN layers (conv_block1-4): load directly (frequency features transfer)
    - FC + BN layers: load directly (dimension matches)
    - GRU layers in acoustic models: load directly (temporal features transfer)
    - Output FC layers in acoustic models: SKIP (88 -> 48 dimension mismatch)
    - Conditioning GRUs + FCs: SKIP (dimension mismatch)
    """
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # The Kong checkpoint may have different key prefixes depending on
    # whether it was saved from the full model or the note model
    # Try to detect the format
    kong_keys = list(state_dict.keys())

    # Map Kong sub-models to our sub-models
    # Kong has: frame_model, reg_onset_model, reg_offset_model, velocity_model
    # We have: frame_model, onset_model, offset_model, velocity_model
    key_mapping = {
        'reg_onset_model': 'onset_model',
        'reg_offset_model': 'offset_model',
        'frame_model': 'frame_model',
        'velocity_model': 'velocity_model',
    }

    loaded = 0
    skipped = 0
    our_state = model.state_dict()

    for kong_key, kong_value in state_dict.items():
        # Skip spectrogram / logmel layers (we compute mel externally)
        if 'spectrogram' in kong_key or 'logmel' in kong_key or 'bn0' in kong_key:
            skipped += 1
            continue

        # Skip conditioning GRUs from Kong (dimension mismatch: 88 vs 48 pitches)
        if 'reg_onset_gru' in kong_key or 'reg_onset_fc' in kong_key:
            skipped += 1
            continue
        if 'frame_gru' in kong_key or 'frame_fc' in kong_key:
            skipped += 1
            continue

        # Map key names
        our_key = kong_key
        for kong_prefix, our_prefix in key_mapping.items():
            if kong_key.startswith(kong_prefix):
                our_key = kong_key.replace(kong_prefix, our_prefix, 1)
                break

        # Skip output FC in acoustic models (88 vs 48)
        if our_key.endswith('.fc.weight') or our_key.endswith('.fc.bias'):
            # Check if this is the final output FC (not fc5)
            parts = our_key.split('.')
            if parts[-2] == 'fc' and 'fc5' not in our_key:
                skipped += 1
                continue

        # Try to load
        if our_key in our_state:
            if kong_value.shape == our_state[our_key].shape:
                our_state[our_key] = kong_value
                loaded += 1
            else:
                skipped += 1
        else:
            skipped += 1

    model.load_state_dict(our_state)
    print(f"Kong checkpoint: loaded {loaded} tensors, skipped {skipped}")
    return model


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class FocalBCEWithLogitsLoss(nn.Module):
    """Focal loss for extreme class imbalance in onset detection."""
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


# ============================================================================
# GUITARSET DATASET
# ============================================================================
class GuitarSetDataset(Dataset):
    """
    Loads GuitarSet audio + JAMS annotations, returns fixed-length chunks
    of mel spectrogram + onset/frame/velocity targets.
    """
    def __init__(self, track_list, chunk_frames=CHUNK_FRAMES, augment=False):
        self.tracks = track_list
        self.chunk_frames = chunk_frames
        self.augment = augment
        self._cache = {}

    def __len__(self):
        # Multiple chunks per track for more training data
        return len(self.tracks) * 8

    def _load_track(self, track_idx):
        if track_idx in self._cache:
            return self._cache[track_idx]

        track = self.tracks[track_idx]
        audio_path = track['audio']
        jams_path = track['jams']

        # Load audio
        audio, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio, sr=SAMPLE_RATE, n_fft=CONFIG['n_fft'],
            hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
        mel_db = mel_db.T  # (T, n_mels)

        # Parse JAMS
        jam = jams.load(jams_path)
        n_frames = mel_db.shape[0]

        onsets = np.zeros((n_frames, N_PITCHES), dtype=np.float32)
        frames = np.zeros((n_frames, N_PITCHES), dtype=np.float32)
        velocities = np.zeros((n_frames, N_PITCHES), dtype=np.float32)

        for ann in jam.annotations:
            if ann.namespace == 'note_midi':
                for obs in ann.data:
                    midi_pitch = int(round(obs.value))
                    if GUITAR_MIDI_MIN <= midi_pitch < GUITAR_MIDI_MAX:
                        pitch_idx = midi_pitch - GUITAR_MIDI_MIN
                        onset_frame = int(obs.time * SAMPLE_RATE / HOP_LENGTH)
                        offset_frame = int((obs.time + obs.duration) * SAMPLE_RATE / HOP_LENGTH)

                        onset_frame = min(onset_frame, n_frames - 1)
                        offset_frame = min(offset_frame, n_frames - 1)

                        onsets[onset_frame, pitch_idx] = 1.0
                        frames[onset_frame:offset_frame + 1, pitch_idx] = 1.0
                        vel = obs.confidence if obs.confidence else 0.8
                        velocities[onset_frame:offset_frame + 1, pitch_idx] = vel

        result = (mel_db, onsets, frames, velocities)
        self._cache[track_idx] = result
        return result

    def __getitem__(self, idx):
        track_idx = idx // 8
        mel_db, onsets, frames, velocities = self._load_track(track_idx)

        n_frames = mel_db.shape[0]
        if n_frames > self.chunk_frames:
            start = np.random.randint(0, n_frames - self.chunk_frames)
        else:
            start = 0

        end = start + self.chunk_frames

        # Extract chunk, pad if needed
        mel_chunk = mel_db[start:end]
        onset_chunk = onsets[start:end]
        frame_chunk = frames[start:end]
        vel_chunk = velocities[start:end]

        if mel_chunk.shape[0] < self.chunk_frames:
            pad = self.chunk_frames - mel_chunk.shape[0]
            mel_chunk = np.pad(mel_chunk, ((0, pad), (0, 0)))
            onset_chunk = np.pad(onset_chunk, ((0, pad), (0, 0)))
            frame_chunk = np.pad(frame_chunk, ((0, pad), (0, 0)))
            vel_chunk = np.pad(vel_chunk, ((0, pad), (0, 0)))

        # Augmentation
        if self.augment:
            mel_chunk = self._augment_mel(mel_chunk)

        return (
            torch.from_numpy(mel_chunk.copy()),
            torch.from_numpy(onset_chunk.copy()),
            torch.from_numpy(frame_chunk.copy()),
            torch.from_numpy(vel_chunk.copy()),
        )

    def _augment_mel(self, mel):
        """Apply augmentation to mel spectrogram."""
        mel = mel.copy()

        # Gain jitter (in dB space, mel is already in dB)
        if random.random() < 0.7:
            gain = random.uniform(*CONFIG['gain_range'])
            mel = mel + gain

        # Gaussian noise (in dB space)
        if random.random() < 0.5:
            snr_db = random.uniform(*CONFIG['noise_snr_range'])
            noise_level = 10 ** (-snr_db / 20)
            noise = np.random.randn(*mel.shape).astype(np.float32) * noise_level * 10
            mel = mel + noise

        # SpecAugment: frequency masking
        for _ in range(CONFIG['spec_augment_freq_masks']):
            if random.random() < 0.5:
                f_width = random.randint(1, CONFIG['spec_augment_freq_width'])
                f_start = random.randint(0, max(0, mel.shape[1] - f_width))
                mel[:, f_start:f_start + f_width] = mel.min()

        # SpecAugment: time masking
        for _ in range(CONFIG['spec_augment_time_masks']):
            if random.random() < 0.5:
                t_width = random.randint(1, CONFIG['spec_augment_time_width'])
                t_start = random.randint(0, max(0, mel.shape[0] - t_width))
                mel[t_start:t_start + t_width, :] = mel.min()

        return mel


# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_onset_f1(model, val_loader, device, threshold=0.5):
    """Compute onset F1 using mir_eval at 50ms tolerance."""
    model.eval()
    all_f1 = []
    all_prec = []
    all_rec = []

    with torch.no_grad():
        for mel, onset_targets, frame_targets, vel_targets in val_loader:
            mel = mel.to(device)
            onset_targets = onset_targets.to(device)

            onset_logits, _, _ = model(mel)
            onset_probs = torch.sigmoid(onset_logits)

            for i in range(onset_probs.shape[0]):
                pred = (onset_probs[i] > threshold).cpu().numpy()
                ref = onset_targets[i].cpu().numpy()

                # Convert to event times for mir_eval
                pred_frames = np.where(pred.any(axis=1))[0]
                ref_frames = np.where(ref.any(axis=1))[0]

                pred_times = pred_frames.astype(float) * HOP_LENGTH / SAMPLE_RATE
                ref_times = ref_frames.astype(float) * HOP_LENGTH / SAMPLE_RATE

                if len(ref_times) == 0:
                    continue

                if len(pred_times) == 0:
                    all_f1.append(0.0)
                    all_prec.append(0.0)
                    all_rec.append(0.0)
                    continue

                precision, recall, f1 = mir_eval.onset.f_measure(
                    ref_times, pred_times, window=0.05
                )
                all_f1.append(f1)
                all_prec.append(precision)
                all_rec.append(recall)

    if not all_f1:
        return 0.0, 0.0, 0.0

    return np.mean(all_f1), np.mean(all_prec), np.mean(all_rec)


def compute_batch_f1(onset_logits, onset_targets, threshold=0.5):
    """Quick batch-level F1 for logging (not mir_eval, just element-wise)."""
    with torch.no_grad():
        preds = (torch.sigmoid(onset_logits) > threshold).float()
        tp = (preds * onset_targets).sum()
        fp = (preds * (1 - onset_targets)).sum()
        fn = ((1 - preds) * onset_targets).sum()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1.item(), precision.item(), recall.item()


# ============================================================================
# GUITARSET TRACK LOADING
# ============================================================================
def load_guitarset_tracks():
    """Load all GuitarSet tracks with audio + JAMS pairs."""
    audio_dir = DATA_DIR / "audio_mono-mic"
    annot_dir = DATA_DIR / "annotation"

    if not audio_dir.exists():
        # Try alternate directory names from unzip
        for candidate in DATA_DIR.iterdir():
            if candidate.is_dir() and 'audio' in candidate.name.lower() and 'mic' in candidate.name.lower():
                audio_dir = candidate
                break

    if not annot_dir.exists():
        for candidate in DATA_DIR.iterdir():
            if candidate.is_dir() and 'annot' in candidate.name.lower():
                annot_dir = candidate
                break

    tracks = []
    audio_files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(audio_files)} audio files in {audio_dir}")

    for audio_path in audio_files:
        # GuitarSet naming: PlayerID_Style_Tempo_Key_mic.wav
        # JAMS: PlayerID_Style_Tempo_Key.jams
        stem = audio_path.stem
        # Remove trailing _mic if present
        jams_stem = stem.replace('_mic', '').replace('_mix', '')
        jams_path = annot_dir / f"{jams_stem}.jams"

        if not jams_path.exists():
            # Try with various suffixes removed
            for suffix in ['_mic', '_mix', '_hex_cln', '_hex']:
                test_stem = stem.split(suffix)[0] if suffix in stem else stem
                test_path = annot_dir / f"{test_stem}.jams"
                if test_path.exists():
                    jams_path = test_path
                    break

        if jams_path.exists():
            # Extract player ID (first 2 chars)
            player_id = audio_path.stem[:2]
            tracks.append({
                'audio': str(audio_path),
                'jams': str(jams_path),
                'player': player_id,
                'name': audio_path.stem,
            })

    print(f"Matched {len(tracks)} audio-annotation pairs")
    players = set(t['player'] for t in tracks)
    print(f"Players: {sorted(players)}")
    for p in sorted(players):
        count = sum(1 for t in tracks if t['player'] == p)
        print(f"  Player {p}: {count} tracks")

    return tracks


# ============================================================================
# TRAINING
# ============================================================================
def train_phase(model, train_loader, val_loader, device, phase, epochs, lr,
                freeze_cnn=False, best_val_f1=0.0, start_epoch=0):
    """Run one training phase."""
    print(f"\n{'='*60}")
    print(f"PHASE {phase}: {'Frozen CNN' if freeze_cnn else 'Full Fine-tuning'}")
    print(f"Epochs: {epochs}, LR: {lr:.2e}")
    print(f"{'='*60}\n")

    if freeze_cnn:
        for name, param in model.named_parameters():
            if any(x in name for x in ['conv_block', 'bn1', 'bn2', 'conv1', 'conv2']):
                if 'frame_model' in name or 'onset_model' in name or \
                   'offset_model' in name or 'velocity_model' in name:
                    param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    onset_loss_fn = FocalBCEWithLogitsLoss(
        pos_weight=CONFIG['onset_pos_weight'], gamma=CONFIG['focal_gamma']
    )
    frame_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG['frame_pos_weight']]).to(device)
    )
    velocity_loss_fn = nn.MSELoss()

    collapse_counter = 0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # --- Train ---
        model.train()
        running_loss = 0.0
        num_batches = 0
        epoch_onset_logits = []
        epoch_onset_targets = []

        pbar = tqdm(train_loader, desc=f'P{phase} Epoch {epoch+1}/{epochs}')
        for mel, onset_tgt, frame_tgt, vel_tgt in pbar:
            mel = mel.to(device)
            onset_tgt = onset_tgt.to(device)
            frame_tgt = frame_tgt.to(device)
            vel_tgt = vel_tgt.to(device)

            onset_logits, frame_logits, vel_pred = model(mel)

            loss_onset = onset_loss_fn(onset_logits, onset_tgt)
            loss_frame = frame_loss_fn(frame_logits, frame_tgt)

            # Velocity loss only on active onset frames
            onset_mask = onset_tgt > 0.5
            if onset_mask.any():
                loss_vel = velocity_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask])
            else:
                loss_vel = torch.tensor(0.0, device=device)

            loss = loss_onset + loss_frame + 0.5 * loss_vel

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

            # Sample for batch-level F1
            if num_batches % 5 == 0:
                epoch_onset_logits.append(onset_logits.detach().cpu())
                epoch_onset_targets.append(onset_tgt.detach().cpu())

        scheduler.step()

        train_loss = running_loss / max(num_batches, 1)
        if epoch_onset_logits:
            train_f1, train_prec, train_rec = compute_batch_f1(
                torch.cat(epoch_onset_logits), torch.cat(epoch_onset_targets))
        else:
            train_f1, train_prec, train_rec = 0, 0, 0

        # --- Validate (mir_eval F1 every 5 epochs, batch F1 otherwise) ---
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_f1, val_prec, val_rec = evaluate_onset_f1(model, val_loader, device)
            eval_type = "mir_eval"
        else:
            model.eval()
            val_logits_all = []
            val_targets_all = []
            with torch.no_grad():
                for mel, onset_tgt, frame_tgt, vel_tgt in val_loader:
                    mel = mel.to(device)
                    onset_tgt = onset_tgt.to(device)
                    onset_logits, _, _ = model(mel)
                    val_logits_all.append(onset_logits.cpu())
                    val_targets_all.append(onset_tgt.cpu())
            val_f1, val_prec, val_rec = compute_batch_f1(
                torch.cat(val_logits_all), torch.cat(val_targets_all))
            eval_type = "batch"

        elapsed = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        print(f"\nP{phase} Epoch {epoch+1}/{epochs} ({elapsed:.0f}s) LR={cur_lr:.2e}")
        print(f"  Train — Loss: {train_loss:.4f} | F1: {train_f1:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f}")
        print(f"  Val   — F1: {val_f1:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f} [{eval_type}]")

        # Collapse detection
        if val_rec < 0.01 and epoch > 3:
            collapse_counter += 1
            print(f"  WARNING: Recall near zero ({collapse_counter}/10)")
            if collapse_counter >= 10:
                print("\nCOLLAPSED — stopping early. Try higher pos_weight or lower LR.")
                break
        else:
            collapse_counter = 0

        # Save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'phase': phase,
                'model_state_dict': model.state_dict(),
                'val_f1': val_f1,
                'val_prec': val_prec,
                'val_rec': val_rec,
                'config': CONFIG,
            }, str(SAVE_DIR / 'best_guitar_model.pt'))
            print(f"  -> New best! F1={val_f1:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'phase': phase,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_f1': best_val_f1,
                'config': CONFIG,
            }, str(CHECKPOINT_DIR / f'phase{phase}_epoch{epoch+1}.pt'))

    # Unfreeze all params for next phase
    for param in model.parameters():
        param.requires_grad = True

    return best_val_f1


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("Guitar Transcription Training — Domain Adaptation")
    print("Kong Piano -> GuitarSet Fine-tuning")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu} ({vram:.1f} GB)")
    else:
        print("WARNING: No GPU detected! Training will be very slow.")

    # --- Download data ---
    kong_path = download_kong_checkpoint()
    download_guitarset()

    # --- Load tracks ---
    tracks = load_guitarset_tracks()
    if len(tracks) < 50:
        print(f"ERROR: Only {len(tracks)} tracks found. Expected ~360.")
        sys.exit(1)

    # --- Split: leave player '05' for validation (standard single-fold) ---
    val_player = '05'
    train_tracks = [t for t in tracks if t['player'] != val_player]
    val_tracks = [t for t in tracks if t['player'] == val_player]
    print(f"\nTrain: {len(train_tracks)} tracks, Val: {len(val_tracks)} tracks (player {val_player})")

    # --- Datasets ---
    train_dataset = GuitarSetDataset(train_tracks, augment=CONFIG['augment'])
    val_dataset = GuitarSetDataset(val_tracks, augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )

    print(f"Train batches/epoch: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- Build model ---
    model = GuitarTranscriptionModel(n_pitches=N_PITCHES).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # --- Load Kong weights ---
    print(f"\nLoading Kong piano checkpoint: {kong_path}")
    model = load_kong_weights(model, kong_path)

    # --- Check for resume ---
    resume_path = CHECKPOINT_DIR / 'latest_resume.pt'
    start_phase = 1
    start_epoch = 0
    best_val_f1 = 0.0

    if resume_path.exists():
        print(f"\nResuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        start_phase = ckpt.get('phase', 1)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_f1 = ckpt.get('best_val_f1', 0.0)
        print(f"  Phase {start_phase}, epoch {start_epoch}, best F1={best_val_f1:.4f}")

    # --- Phase 1: Frozen CNN ---
    if start_phase <= 1:
        best_val_f1 = train_phase(
            model, train_loader, val_loader, device,
            phase=1, epochs=CONFIG['phase1_epochs'], lr=CONFIG['phase1_lr'],
            freeze_cnn=True, best_val_f1=best_val_f1,
            start_epoch=start_epoch if start_phase == 1 else 0
        )
        # Save resume checkpoint between phases
        torch.save({
            'phase': 2, 'epoch': 0,
            'model_state_dict': model.state_dict(),
            'best_val_f1': best_val_f1, 'config': CONFIG,
        }, str(resume_path))
        start_epoch = 0

    # --- Phase 2: Full Fine-tuning ---
    best_val_f1 = train_phase(
        model, train_loader, val_loader, device,
        phase=2, epochs=CONFIG['phase2_epochs'], lr=CONFIG['phase2_lr'],
        freeze_cnn=False, best_val_f1=best_val_f1,
        start_epoch=start_epoch if start_phase == 2 else 0
    )

    # --- Final evaluation ---
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")

    # Load best model for final eval
    best_ckpt = torch.load(str(SAVE_DIR / 'best_guitar_model.pt'), map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    final_f1, final_prec, final_rec = evaluate_onset_f1(model, val_loader, device)
    print(f"\nFinal Onset F1: {final_f1:.4f}")
    print(f"Final Precision: {final_prec:.4f}")
    print(f"Final Recall: {final_rec:.4f}")
    print(f"Target: 0.87+")
    print(f"Improvement over Basic Pitch (0.79): +{(final_f1 - 0.79)*100:.1f} percentage points")

    print(f"\nBest model saved to: {SAVE_DIR / 'best_guitar_model.pt'}")
    print(f"Copy to: backend/models/pretrained/best_guitar_model.pt")
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"{'='*60}")

    # Save training summary
    summary = {
        'final_f1': final_f1,
        'final_precision': final_prec,
        'final_recall': final_rec,
        'best_epoch': best_ckpt.get('epoch', -1),
        'best_phase': best_ckpt.get('phase', -1),
        'config': CONFIG,
        'val_player': val_player,
        'num_train_tracks': len(train_tracks),
        'num_val_tracks': len(val_tracks),
    }
    with open(SAVE_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == '__main__':
    main()
