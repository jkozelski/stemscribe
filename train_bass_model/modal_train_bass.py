#!/usr/bin/env python3
"""
Modal A10G Bass Transcription Training — v3 Transfer Learning
=============================================================
Runs the full bass v3 training pipeline on Modal A10G GPU.

Strategy:
  1. Upload piano checkpoint to volume (one-time, from local)
  2. Download Slakh2100-redux (~97GB) and extract bass stems (~2GB) on GPU
  3. Train 40 epochs with transfer learning from piano CNN
  4. Save best model to volume + download locally

Usage:
  # First time: upload piano checkpoint
  cd ~/stemscribe/train_bass_model && ../venv311/bin/python -m modal run modal_train_bass.py::upload_piano_checkpoint

  # Run training
  cd ~/stemscribe/train_bass_model && ../venv311/bin/python -m modal run modal_train_bass.py

  # Download trained model
  cd ~/stemscribe/train_bass_model && ../venv311/bin/python -m modal run modal_train_bass.py::download_model
"""

import modal
import time as _time

# ---------------------------------------------------------------------------
# Modal Image
# ---------------------------------------------------------------------------

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "ffmpeg", "aria2")
    .pip_install(
        "torch>=2.3,<2.5",
        "torchaudio>=2.3,<2.5",
        "librosa>=0.10",
        "pretty_midi",
        "tqdm",
        "soundfile",
        "pyyaml",
        "numpy>=1.24,<2",
        "scipy",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# Separate lightweight image for data download (no GPU needed)
download_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget", "aria2")
    .pip_install("pyyaml", "tqdm")
)

# Volumes for data persistence
data_volume = modal.Volume.from_name("bass-training-data", create_if_missing=True)
results_volume = modal.Volume.from_name("bass-training-results", create_if_missing=True)

app = modal.App("bass-transcription-training", image=training_image)

# ---------------------------------------------------------------------------
# Upload piano checkpoint (run once)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Main entrypoint: upload piano checkpoint if needed, then train."""
    import os

    piano_local = os.path.expanduser(
        "~/stemscribe/backend/models/pretrained/best_piano_model.pt"
    )

    # Check if piano checkpoint is already on volume
    print("Checking if piano checkpoint exists on volume...")
    needs_upload = check_piano_checkpoint.remote()

    if needs_upload:
        if not os.path.exists(piano_local):
            print(f"ERROR: Piano checkpoint not found at {piano_local}")
            print("Cannot train without piano CNN for transfer learning.")
            return
        print(f"Uploading piano checkpoint ({os.path.getsize(piano_local) / 1e6:.0f} MB)...")
        with open(piano_local, "rb") as f:
            piano_bytes = f.read()
        upload_piano_checkpoint_remote.remote(piano_bytes)
        print("Piano checkpoint uploaded.")
    else:
        print("Piano checkpoint already on volume.")

    # Download data (no GPU, separate function)
    print("\nStep 1: Ensuring Slakh2100 bass data is on volume...")
    num_tracks = download_slakh_data.remote()
    print(f"Bass tracks on volume: {num_tracks}")

    if num_tracks and num_tracks < 50:
        print(f"ERROR: Only {num_tracks} tracks. Need at least 50.")
        return

    # Run training on GPU
    print("\nStep 2: Starting bass transcription training on A10G GPU...")
    print("This will take several hours (40 epochs).")
    print("=" * 60)

    result = train_bass.remote()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    if result is None:
        print("Training returned no result. Check Modal logs.")
        return

    best_f1 = result.get("best_val_f1", 0)
    best_epoch = result.get("best_epoch", -1)
    model_bytes = result.get("model_bytes")

    print(f"Best val F1: {best_f1:.4f} (epoch {best_epoch + 1})")

    if model_bytes:
        out_path = os.path.expanduser(
            "~/stemscribe/backend/models/pretrained/best_bass_model.pt"
        )
        with open(out_path, "wb") as f:
            f.write(model_bytes)
        print(f"Model saved to: {out_path} ({len(model_bytes) / 1e6:.1f} MB)")
    else:
        print("WARNING: No model bytes returned. Use download_model entrypoint.")

    # Write results doc
    history = result.get("history", [])
    _write_results_doc(result, history)


def _write_results_doc(result, history):
    """Write training results markdown."""
    import os
    from datetime import datetime

    docs_dir = os.path.expanduser("~/stemscribe/docs")
    os.makedirs(docs_dir, exist_ok=True)
    path = os.path.join(docs_dir, "bass-training-results.md")

    best_f1 = result.get("best_val_f1", 0)
    best_epoch = result.get("best_epoch", -1)
    total_time = result.get("total_time_s", 0)
    num_train = result.get("num_train_tracks", 0)
    num_val = result.get("num_val_tracks", 0)

    lines = [
        "# Bass Transcription Training Results",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Script:** modal_train_bass.py (v3 transfer learning)",
        "",
        "## Configuration",
        "- **GPU:** Modal A10G (24GB VRAM)",
        "- **Model:** BassTranscriptionModel_v3 (piano CNN transfer learning)",
        "- **Dataset:** Slakh2100-redux (bass stems, GM programs 32-39)",
        f"- **Train tracks:** {num_train}",
        f"- **Val tracks:** {num_val}",
        "- **Epochs:** 40 (3-phase: adapter warmup -> full -> CNN fine-tune)",
        "- **Batch size:** 16",
        "- **Pitch range:** E1 (MIDI 28) to G4 (MIDI 67) = 40 keys",
        "- **Spectrogram:** Mel (sr=16000, hop=256, n_mels=229, match piano)",
        "",
        "## Results",
        f"- **Best val onset F1:** {best_f1:.4f}",
        f"- **Best epoch:** {best_epoch + 1}",
        f"- **Total training time:** {total_time / 3600:.1f} hours",
        f"- **Output:** backend/models/pretrained/best_bass_model.pt",
        "",
    ]

    if history:
        lines.append("## Epoch History")
        lines.append("| Epoch | Phase | Train Loss | Val Loss | Val F1 | Val P | Val R | LR |")
        lines.append("|-------|-------|-----------|---------|--------|-------|-------|------|")
        for h in history:
            lines.append(
                f"| {h.get('epoch', '?')} | {h.get('phase', '?')} | "
                f"{h.get('train_loss', 0):.4f} | {h.get('val_loss', 0):.4f} | "
                f"{h.get('val_f1', 0):.4f} | {h.get('val_prec', 0):.4f} | "
                f"{h.get('val_rec', 0):.4f} | {h.get('lr', 0):.2e} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("- Uses frozen piano CNN for feature extraction (transfer learning)")
    lines.append("- Phase 1 (ep 1-8): adapter warmup only")
    lines.append("- Phase 2 (ep 9-30): full LSTM + heads training")
    lines.append("- Phase 3 (ep 31-40): CNN top blocks unfrozen for fine-tuning")
    lines.append("- Focal loss + soft dice for onset detection (sparse targets)")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Results written to: {path}")


# ---------------------------------------------------------------------------
# Helper functions (run on Modal)
# ---------------------------------------------------------------------------

@app.function(volumes={"/data": data_volume})
def check_piano_checkpoint() -> bool:
    """Returns True if piano checkpoint needs uploading."""
    import os
    path = "/data/best_piano_model.pt"
    exists = os.path.exists(path)
    if exists:
        size = os.path.getsize(path)
        print(f"Piano checkpoint exists: {size / 1e6:.0f} MB")
        return False
    print("Piano checkpoint not found on volume.")
    return True


@app.function(volumes={"/data": data_volume})
def upload_piano_checkpoint_remote(piano_bytes: bytes):
    """Store piano checkpoint on volume."""
    with open("/data/best_piano_model.pt", "wb") as f:
        f.write(piano_bytes)
    data_volume.commit()
    print(f"Piano checkpoint saved ({len(piano_bytes) / 1e6:.0f} MB)")


# ---------------------------------------------------------------------------
# Data download (separate function, no GPU needed, long timeout)
# ---------------------------------------------------------------------------

@app.function(
    image=download_image,
    volumes={"/data": data_volume},
    timeout=21600,  # 6 hours for download
    memory=8192,
    cpu=4,
)
def download_slakh_data():
    """
    Download Slakh2100-redux and extract bass stems.
    Uses aria2c for parallel download (16 connections) which is much faster
    than wget from Zenodo.
    """
    import os
    import shutil
    import subprocess
    from pathlib import Path
    import yaml
    from tqdm import tqdm

    SLAKH_ROOT = Path("/data/slakh2100_flac_redux")
    BASS_DATA_DIR = Path("/data/slakh_bass")
    BASS_PROGRAMS = set(range(32, 40))

    BASS_DATA_DIR.mkdir(exist_ok=True)

    existing = list(BASS_DATA_DIR.glob("*_Track*"))
    if len(existing) > 100:
        print(f"Found {len(existing)} existing bass tracks on volume. Skipping download.")
        return len(existing)

    if SLAKH_ROOT.exists() and len(list(SLAKH_ROOT.glob("**/metadata.yaml"))) > 100:
        print("Slakh already extracted on volume, extracting bass stems...")
    else:
        total, used, free = shutil.disk_usage("/data")
        free_gb = free / 1e9
        print(f"Volume disk: {free_gb:.1f} GB free")

        print("Downloading Slakh2100-redux from Zenodo (~97 GB)...")
        print("Using aria2c with 16 parallel connections for speed.")
        tar_path = "/data/slakh2100.tar.gz"

        dl_start = _time.time()
        # aria2c with 16 connections is dramatically faster from Zenodo
        result = subprocess.run([
            "aria2c",
            "--max-connection-per-server=16",
            "--split=16",
            "--min-split-size=10M",
            "--max-concurrent-downloads=1",
            "--file-allocation=none",
            "--continue=true",
            "--auto-file-renaming=false",
            "-d", "/data",
            "-o", "slakh2100.tar.gz",
            "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
        ], capture_output=True, text=True)
        print(result.stdout[-2000:] if result.stdout else "")
        if result.returncode != 0:
            print(f"aria2c stderr: {result.stderr[-2000:] if result.stderr else ''}")
            # Fallback to wget
            print("aria2c failed, falling back to wget...")
            subprocess.run([
                "wget", "--progress=dot:giga", "-O", tar_path,
                "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
            ], check=True)

        dl_elapsed = (_time.time() - dl_start) / 60
        print(f"Download complete in {dl_elapsed:.1f} min")

        if os.path.exists(tar_path):
            size_gb = os.path.getsize(tar_path) / 1e9
            print(f"Archive size: {size_gb:.1f} GB")

        print("Extracting (this takes 15-30 min)...")
        extract_start = _time.time()
        subprocess.run(["tar", "xzf", tar_path, "-C", "/data/"], check=True)
        print(f"Extraction complete in {(_time.time() - extract_start) / 60:.1f} min")

        if os.path.exists(tar_path):
            os.unlink(tar_path)
            print("Deleted tar.gz to free space")

    # Extract bass stems
    print("Extracting bass stems from Slakh...")
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

    # Clean up full Slakh
    if SLAKH_ROOT.exists():
        print("Cleaning up full Slakh from volume...")
        shutil.rmtree(SLAKH_ROOT, ignore_errors=True)

    data_volume.commit()
    print(f"Volume committed with {bass_count} bass tracks")
    return bass_count


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=43200,  # 12 hours
    volumes={
        "/data": data_volume,
        "/results": results_volume,
    },
    memory=32768,
)
def train_bass() -> dict:
    """
    Full bass training pipeline (data already on volume):
    1. Load bass stems from volume
    2. Load piano CNN checkpoint
    3. Train 40 epochs
    4. Return best model bytes + history
    """
    import os
    import sys
    import json
    import random
    import math
    import subprocess
    import shutil
    from pathlib import Path

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

    t_start = _time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {name} ({mem:.1f} GB)")

    # ======================================================================
    # CONFIG
    # ======================================================================
    SAMPLE_RATE = 16000
    HOP_LENGTH = 256
    N_MELS = 229
    N_FFT = 2048
    FMIN = 30.0
    FMAX = 8000.0
    CHUNK_DURATION = 5.0
    CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)
    CHUNK_FRAMES = int(CHUNK_SAMPLES / HOP_LENGTH)

    BASS_MIN_MIDI = 28
    BASS_MAX_MIDI = 67
    NUM_KEYS = 40

    BATCH_SIZE = 16
    NUM_EPOCHS = 40
    ONSET_POS_WEIGHT = 20.0
    FRAME_POS_WEIGHT = 5.0
    FOCAL_GAMMA = 2.0
    DICE_WEIGHT = 0.3

    PHASE1_END = 8
    PHASE2_END = 30
    PHASE3_END = 40

    BASS_PROGRAMS = set(range(32, 40))

    BASS_DATA_DIR = Path("/data/slakh_bass")
    SAVE_DIR = Path("/results/bass_v3")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    PIANO_CKPT = Path("/data/best_piano_model.pt")

    # ======================================================================
    # MODELS (defined inline for Modal portability)
    # ======================================================================
    class PianoTranscriptionModel(nn.Module):
        def __init__(self, n_mels=229, num_keys=88,
                     hidden_size=256, num_layers=2, dropout=0.25):
            super().__init__()
            self.num_keys = num_keys
            self.cnn = nn.Sequential(
                nn.Conv2d(1, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.Conv2d(48, 48, kernel_size=3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
                nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
                nn.Conv2d(48, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
                nn.Conv2d(64, 96, kernel_size=3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
                nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
                nn.Conv2d(96, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d((2, 1)), nn.Dropout2d(dropout),
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
            velocity_pred = self.velocity_head(onset_out)
            return (
                onset_pred.permute(0, 2, 1),
                frame_pred.permute(0, 2, 1) if False else frame_logits,  # unused
                velocity_pred.permute(0, 2, 1),
            )

    class BassTranscriptionModel_v3(nn.Module):
        def __init__(self, num_keys=40, hidden_size=128, num_layers=2, dropout=0.25):
            super().__init__()
            self.num_keys = num_keys
            self.cnn = None  # Set externally (frozen piano CNN)

            self.adapter = nn.Sequential(
                nn.Linear(1792, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            )
            self.adapter_residual = nn.Linear(1792, 1024)

            self.onset_lstm = nn.LSTM(
                input_size=1024, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            )
            self.frame_lstm = nn.LSTM(
                input_size=1024 + num_keys, hidden_size=hidden_size,
                num_layers=num_layers, batch_first=True,
                bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            )

            lstm_dim = hidden_size * 2
            self.onset_head = nn.Linear(lstm_dim, num_keys)
            self.frame_head = nn.Linear(lstm_dim, num_keys)
            self.velocity_head = nn.Sequential(
                nn.Linear(lstm_dim, num_keys), nn.Sigmoid(),
            )

            nn.init.constant_(self.onset_head.bias, -3.0)
            nn.init.constant_(self.frame_head.bias, -2.0)

        def forward(self, mel):
            batch, n_mels, time = mel.shape
            with torch.no_grad():
                cnn_out = self.cnn(mel.unsqueeze(1))
            cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)
            adapted = self.adapter(cnn_features) + self.adapter_residual(cnn_features)
            onset_out, _ = self.onset_lstm(adapted)
            onset_logits = self.onset_head(onset_out)
            onset_pred = torch.sigmoid(onset_logits)
            frame_input = torch.cat([adapted, onset_pred.detach()], dim=-1)
            frame_out, _ = self.frame_lstm(frame_input)
            frame_logits = self.frame_head(frame_out)
            velocity_pred = self.velocity_head(onset_out)
            return (
                onset_logits.permute(0, 2, 1),
                frame_logits.permute(0, 2, 1),
                velocity_pred.permute(0, 2, 1),
            )

    # ======================================================================
    # LOSS FUNCTIONS
    # ======================================================================
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

    # ======================================================================
    # DATA LOADING (download handled by separate function)
    # ======================================================================
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

    # ======================================================================
    # DATASET
    # ======================================================================
    def midi_to_flat_targets(midi_path, total_frames, sr=SAMPLE_RATE, hop=HOP_LENGTH):
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

            audio_chunk = mel_chunk.copy()
            if self.augment:
                if random.random() < 0.5:
                    audio_chunk = audio_chunk + random.uniform(-0.3, 0.3)
                if random.random() < 0.2:
                    f_start = random.randint(0, N_MELS - 20)
                    f_width = random.randint(1, 15)
                    audio_chunk[f_start:f_start + f_width, :] = 0.0

            return (
                torch.from_numpy(audio_chunk),
                torch.from_numpy(onset_chunk),
                torch.from_numpy(frame_chunk),
                torch.from_numpy(vel_chunk),
            )

    # ======================================================================
    # CNN LOADING
    # ======================================================================
    def load_piano_cnn(checkpoint_path, dev):
        print(f"Loading piano CNN from {checkpoint_path}...")
        checkpoint = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)
        piano_model = PianoTranscriptionModel()
        piano_model.load_state_dict(checkpoint["model_state_dict"])
        cnn = piano_model.cnn
        for param in cnn.parameters():
            param.requires_grad = False
        cnn.eval()
        print(f"  Piano CNN: {sum(p.numel() for p in cnn.parameters()):,} params (frozen)")
        return cnn

    def unfreeze_cnn_top_blocks(model):
        freeze_boundary = 8
        unfrozen = 0
        for i, layer in enumerate(model.cnn.children()):
            if i >= freeze_boundary:
                for param in layer.parameters():
                    param.requires_grad = True
                    unfrozen += 1
        print(f"  Unfroze {unfrozen} CNN parameters")
        return unfrozen

    # ======================================================================
    # SCHEDULER
    # ======================================================================
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

    # ======================================================================
    # TRAINING LOOP
    # ======================================================================
    def train_epoch(model, loader, optimizer, onset_loss_fn, frame_loss_fn, vel_loss_fn, dev):
        model.train()
        if model.cnn is not None:
            model.cnn.eval()

        total_loss = 0.0
        num_batches = 0
        epoch_onset_logits = []
        epoch_onset_targets = []

        for mel_b, onset_tgt, frame_tgt, vel_tgt in tqdm(loader, desc="  Train", leave=False):
            mel_b = mel_b.to(dev)
            onset_tgt = onset_tgt.to(dev)
            frame_tgt = frame_tgt.to(dev)
            vel_tgt = vel_tgt.to(dev)

            onset_logits, frame_logits, vel_pred = model(mel_b)

            loss_onset = onset_loss_fn(onset_logits, onset_tgt)
            loss_frame = frame_loss_fn(frame_logits, frame_tgt)
            onset_mask = onset_tgt > 0.5
            loss_vel = vel_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask]) if onset_mask.any() else torch.tensor(0.0, device=dev)

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

    def validate_epoch(model, loader, onset_loss_fn, frame_loss_fn, vel_loss_fn, dev):
        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_onset_logits = []
        all_onset_targets = []

        with torch.no_grad():
            for mel_b, onset_tgt, frame_tgt, vel_tgt in loader:
                mel_b = mel_b.to(dev)
                onset_tgt = onset_tgt.to(dev)
                frame_tgt = frame_tgt.to(dev)
                vel_tgt = vel_tgt.to(dev)

                onset_logits, frame_logits, vel_pred = model(mel_b)
                loss_onset = onset_loss_fn(onset_logits, onset_tgt)
                loss_frame = frame_loss_fn(frame_logits, frame_tgt)
                onset_mask = onset_tgt > 0.5
                loss_vel = vel_loss_fn(vel_pred[onset_mask], vel_tgt[onset_mask]) if onset_mask.any() else torch.tensor(0.0, device=dev)

                total_loss += (loss_onset + loss_frame + 0.5 * loss_vel).item()
                num_batches += 1
                all_onset_logits.append(onset_logits.cpu())
                all_onset_targets.append(onset_tgt.cpu())

        avg_loss = total_loss / max(num_batches, 1)
        f1, prec, rec = compute_onset_f1(
            torch.cat(all_onset_logits), torch.cat(all_onset_targets))
        return avg_loss, f1, prec, rec

    # ======================================================================
    # EXECUTE
    # ======================================================================

    # Check piano checkpoint
    if not PIANO_CKPT.exists():
        print("ERROR: Piano checkpoint not found on volume!")
        print("Run: modal run modal_train_bass.py::upload_piano_checkpoint first")
        return {"error": "Piano checkpoint missing"}

    # Load bass data from volume (already downloaded by download_slakh_data)
    bass_tracks = load_bass_tracks()

    if len(bass_tracks) < 50:
        print(f"ERROR: Only {len(bass_tracks)} bass tracks found. Need at least 50.")
        return {"error": f"Insufficient data: {len(bass_tracks)} tracks"}

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

    # Verify architecture
    dummy = torch.randn(1, N_MELS, CHUNK_FRAMES).to(device)
    o, f, v = model(dummy)
    assert o.shape == (1, NUM_KEYS, CHUNK_FRAMES), f"Shape mismatch: {o.shape}"
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

    # Resume from checkpoint
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
    print(f"\nStarting: epoch {start_epoch + 1} -> {NUM_EPOCHS}\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = _time.time()

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

        val_loss, val_f1, val_prec, val_rec = validate_epoch(
            model, val_loader, onset_loss_fn, frame_loss_fn, vel_loss_fn, device)

        elapsed = _time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        phase = "Phase1" if epoch < PHASE1_END else ("Phase2" if epoch < PHASE2_END else "Phase3")

        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} ({elapsed:.0f}s) {phase} LR={lr:.2e}")
        print(f"  Train -- Loss: {train_loss:.4f} | F1: {train_f1:.4f} | P: {train_prec:.4f} | R: {train_rec:.4f}")
        print(f"  Val   -- Loss: {val_loss:.4f} | F1: {val_f1:.4f} | P: {val_prec:.4f} | R: {val_rec:.4f}")

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
        best_model_path = SAVE_DIR / "best_bass_model.pt"
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
            }, str(best_model_path))
            print(f"  * Best model saved (F1={val_f1:.4f})")

        # Save checkpoint for resume
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_f1": val_f1,
            "best_val_f1": best_val_f1,
            "best_val_loss": best_val_loss,
        }, str(ckpt_path))

        # Commit volumes periodically (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            results_volume.commit()
            print(f"  Volume checkpoint committed")

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

    # Final commit
    results_volume.commit()

    # Save history
    with open(SAVE_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    total_time = _time.time() - t_start
    print(f"\nTraining complete in {total_time / 3600:.1f} hours")
    print(f"Best val onset F1: {best_val_f1:.4f}")

    # Read best model bytes to return
    best_model_path = SAVE_DIR / "best_bass_model.pt"
    model_bytes = None
    if best_model_path.exists():
        with open(best_model_path, "rb") as f:
            model_bytes = f.read()
        print(f"Model size: {len(model_bytes) / 1e6:.1f} MB")

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": max((h["epoch"] - 1 for h in history if h.get("val_f1") == best_val_f1), default=-1),
        "total_time_s": total_time,
        "num_train_tracks": len(train_tracks),
        "num_val_tracks": len(val_tracks),
        "history": history,
        "model_bytes": model_bytes,
    }


# ---------------------------------------------------------------------------
# Download model entrypoint (if training already completed)
# ---------------------------------------------------------------------------

@app.function(volumes={"/results": results_volume})
def get_trained_model() -> bytes:
    """Retrieve the best model from the results volume."""
    path = "/results/bass_v3/best_bass_model.pt"
    import os
    if not os.path.exists(path):
        print("No trained model found on volume.")
        return b""
    with open(path, "rb") as f:
        data = f.read()
    print(f"Model: {len(data) / 1e6:.1f} MB")
    return data
