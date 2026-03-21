"""
BTC Fine-Tuning Pipeline — Phase 1
===================================
Downloads audio for Billboard + JAAH datasets, computes CQT features,
and prepares everything for BTC fine-tuning.

Steps:
1. Parse Billboard metadata + JAAH annotations
2. Download matching audio from YouTube via yt-dlp
3. Compute CQT spectrograms (matching BTC's config)
4. Save paired (spectrogram, label) data for training

Usage:
    cd ~/stemscribe/backend
    ../venv311/bin/python btc_finetune_pipeline.py --step download
    ../venv311/bin/python btc_finetune_pipeline.py --step features
    ../venv311/bin/python btc_finetune_pipeline.py --step train
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "training_data" / "btc_finetune"
AUDIO_DIR = TRAINING_DIR / "audio"
FEATURES_DIR = TRAINING_DIR / "features"
LABELS_DIR = TRAINING_DIR / "labels"

BILLBOARD_DIR = BASE_DIR / "training_data" / "billboard"
BILLBOARD_CSV = BILLBOARD_DIR / "billboard-2.0-index.csv"
BILLBOARD_LABS = BILLBOARD_DIR / "McGill-Billboard"

JAAH_LABS = BASE_DIR / "training_data" / "jaah" / "labs"
JAAH_ANNOTATIONS = BASE_DIR / "training_data" / "jaah" / "annotations"

# BTC config (must match the pre-trained model)
BTC_SR = 22050
BTC_N_BINS = 144
BTC_BINS_PER_OCTAVE = 24
BTC_HOP_LENGTH = 2048
BTC_INST_LEN = 10.0


def load_billboard_songs():
    """Load Billboard song metadata with matching .lab files."""
    songs = {}
    with open(BILLBOARD_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            song_id = str(row['id']).zfill(4)
            title = row.get('title', '').strip()
            artist = row.get('artist', '').strip()
            if title and artist:
                lab_path = BILLBOARD_LABS / song_id / "full.lab"
                if lab_path.exists():
                    songs[f"billboard_{song_id}"] = {
                        'artist': artist,
                        'title': title,
                        'lab_path': str(lab_path),
                        'source': 'billboard',
                    }
    return songs


def load_jaah_songs():
    """Load JAAH jazz song metadata with .lab files."""
    songs = {}
    # JAAH annotations are JSON files with metadata
    jaah_meta = {}
    if JAAH_ANNOTATIONS.exists():
        for jf in JAAH_ANNOTATIONS.glob("*.json"):
            try:
                with open(jf) as f:
                    data = json.load(f)
                name = jf.stem
                artist = data.get('artist', 'Unknown')
                title = data.get('title', name.replace('_', ' ').title())
                jaah_meta[name] = (artist, title)
            except Exception:
                pass

    for lab_file in sorted(JAAH_LABS.glob("*.lab")):
        name = lab_file.stem
        # Try to find metadata
        if name in jaah_meta:
            artist, title = jaah_meta[name]
        else:
            # Guess from filename
            title = name.replace('_', ' ').replace('(', ' (').title()
            artist = "Unknown Jazz Artist"

        songs[f"jaah_{name}"] = {
            'artist': artist,
            'title': title,
            'lab_path': str(lab_file),
            'source': 'jaah',
        }
    return songs


def download_audio(songs, max_songs=None, skip_existing=True):
    """Download audio from YouTube for each song."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    items = list(songs.items())
    if max_songs:
        items = items[:max_songs]

    downloaded = 0
    skipped = 0
    failed = 0

    for i, (song_id, info) in enumerate(items):
        audio_path = AUDIO_DIR / f"{song_id}.wav"
        if skip_existing and audio_path.exists():
            skipped += 1
            continue

        query = f"{info['artist']} {info['title']} audio"
        logger.info(f"[{i+1}/{len(items)}] Downloading: {info['artist']} - {info['title']}")

        try:
            cmd = [
                "yt-dlp",
                f"ytsearch1:{query}",
                "--extract-audio",
                "--audio-format", "wav",
                "--audio-quality", "0",
                "--output", str(audio_path).replace('.wav', '.%(ext)s'),
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "--max-downloads", "1",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if audio_path.exists():
                downloaded += 1
            else:
                # yt-dlp sometimes adds different extension
                for ext in ['.wav', '.webm', '.m4a', '.mp3']:
                    alt = audio_path.with_suffix(ext)
                    if alt.exists() and ext != '.wav':
                        # Convert to wav
                        subprocess.run([
                            "ffmpeg", "-i", str(alt), "-ar", str(BTC_SR),
                            "-ac", "1", str(audio_path), "-y", "-loglevel", "error"
                        ], timeout=60)
                        alt.unlink()
                        break
                if audio_path.exists():
                    downloaded += 1
                else:
                    logger.warning(f"  Failed: {result.stderr[:200] if result.stderr else 'no output'}")
                    failed += 1

            # Rate limit
            time.sleep(1)

        except subprocess.TimeoutExpired:
            logger.warning(f"  Timeout downloading {song_id}")
            failed += 1
        except Exception as e:
            logger.warning(f"  Error: {e}")
            failed += 1

    logger.info(f"Download complete: {downloaded} new, {skipped} existing, {failed} failed")
    return downloaded


def parse_lab_file(lab_path):
    """Parse a .lab file into list of (start, end, chord) tuples."""
    events = []
    with open(lab_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    start = float(parts[0])
                    end = float(parts[1])
                    chord = parts[2]
                    events.append((start, end, chord))
                except ValueError:
                    continue
    return events


def normalize_chord_label(chord):
    """Normalize chord label to BTC's 170-class vocabulary (Harte notation).

    BTC expects: C:maj, A:min, G:7, N, X, etc.
    Billboard uses same format.
    Some JAAH files have extended notation like F:(b3,5,b7,b9)/3
    """
    if chord in ('N', 'X', 'silence', 'Silence'):
        return 'N'

    # Already in Harte notation
    if ':' in chord:
        # Simplify extended JAAH notation
        # F:(b3,5,b7,b9)/3 -> F:min7
        if '(' in chord:
            root = chord.split(':')[0]
            # Just map to basic quality based on intervals
            intervals = chord.split('(')[1].split(')')[0]
            if 'b3' in intervals and 'b7' in intervals:
                return f"{root}:min7"
            elif 'b3' in intervals:
                return f"{root}:min"
            elif 'b7' in intervals:
                return f"{root}:7"
            else:
                return f"{root}:maj"

        # Strip bass note: C:maj/5 -> C:maj
        if '/' in chord:
            chord = chord.split('/')[0]

        return chord

    # Simple chord names (shouldn't happen in these datasets but just in case)
    return chord


def compute_features(songs, skip_existing=True):
    """Compute CQT spectrograms for all downloaded audio."""
    import librosa

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0

    for song_id, info in songs.items():
        audio_path = AUDIO_DIR / f"{song_id}.wav"
        feat_path = FEATURES_DIR / f"{song_id}.npy"
        label_path = LABELS_DIR / f"{song_id}.lab"

        if not audio_path.exists():
            continue

        if skip_existing and feat_path.exists() and label_path.exists():
            skipped += 1
            continue

        try:
            # Load audio
            wav, sr = librosa.load(str(audio_path), sr=BTC_SR, mono=True)
            duration = len(wav) / sr

            if duration < 10:
                logger.warning(f"  {song_id}: too short ({duration:.1f}s), skipping")
                continue

            # Compute CQT (matching BTC's feature extraction)
            feature = None
            current = 0
            while len(wav) > current + int(BTC_SR * BTC_INST_LEN):
                start_idx = int(current)
                end_idx = int(current + BTC_SR * BTC_INST_LEN)
                tmp = librosa.cqt(
                    wav[start_idx:end_idx], sr=sr,
                    n_bins=BTC_N_BINS,
                    bins_per_octave=BTC_BINS_PER_OCTAVE,
                    hop_length=BTC_HOP_LENGTH)
                feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
                current = end_idx

            # Last chunk
            if current < len(wav):
                tmp = librosa.cqt(
                    wav[int(current):], sr=sr,
                    n_bins=BTC_N_BINS,
                    bins_per_octave=BTC_BINS_PER_OCTAVE,
                    hop_length=BTC_HOP_LENGTH)
                feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)

            feature = np.log(np.abs(feature) + 1e-6)

            # Save spectrogram
            np.save(str(feat_path), feature)

            # Parse and normalize labels, then save as .lab
            events = parse_lab_file(info['lab_path'])
            with open(label_path, 'w') as f:
                for start, end, chord in events:
                    norm_chord = normalize_chord_label(chord)
                    f.write(f"{start:.6f}\t{end:.6f}\t{norm_chord}\n")

            processed += 1
            if processed % 50 == 0:
                logger.info(f"  Processed {processed} songs...")

        except Exception as e:
            logger.warning(f"  {song_id}: feature extraction failed: {e}")

    logger.info(f"Features complete: {processed} processed, {skipped} existing")
    return processed


def prepare_btc_training_data():
    """Prepare the final training data in BTC's expected format."""
    feat_files = sorted(FEATURES_DIR.glob("*.npy"))
    lab_files = sorted(LABELS_DIR.glob("*.lab"))

    # Match features to labels
    feat_ids = {f.stem for f in feat_files}
    lab_ids = {f.stem for f in lab_files}
    paired = feat_ids & lab_ids

    logger.info(f"Paired songs: {len(paired)} (features: {len(feat_ids)}, labels: {len(lab_ids)})")

    # Create symlinks or copy to BTC's expected directory structure
    btc_audio_dir = TRAINING_DIR / "btc_audio"
    btc_label_dir = TRAINING_DIR / "btc_labels"
    btc_audio_dir.mkdir(parents=True, exist_ok=True)
    btc_label_dir.mkdir(parents=True, exist_ok=True)

    for song_id in sorted(paired):
        # BTC training expects audio + lab in the same directory
        audio_src = AUDIO_DIR / f"{song_id}.wav"
        lab_src = LABELS_DIR / f"{song_id}.lab"
        audio_dst = btc_audio_dir / f"{song_id}.wav"
        lab_dst = btc_label_dir / f"{song_id}.lab"

        if not audio_dst.exists() and audio_src.exists():
            os.symlink(audio_src, audio_dst)
        if not lab_dst.exists() and lab_src.exists():
            os.symlink(lab_src, lab_dst)

    # Save manifest
    manifest = {
        'total_paired': len(paired),
        'songs': sorted(list(paired)),
        'billboard_count': len([s for s in paired if s.startswith('billboard_')]),
        'jaah_count': len([s for s in paired if s.startswith('jaah_')]),
    }
    with open(TRAINING_DIR / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Training data ready: {manifest['billboard_count']} Billboard + {manifest['jaah_count']} JAAH")
    return manifest


def run_finetune():
    """Run BTC fine-tuning using the prepared data."""
    btc_dir = BASE_DIR.parent / "btc_chord"
    checkpoint = btc_dir / "test" / "btc_model_large_voca.pt"

    if not checkpoint.exists():
        logger.error(f"BTC checkpoint not found at {checkpoint}")
        return

    manifest_path = TRAINING_DIR / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest.json — run --step features first")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    logger.info(f"Starting BTC fine-tuning on {manifest['total_paired']} songs")
    logger.info(f"  Billboard: {manifest['billboard_count']}")
    logger.info(f"  JAAH: {manifest['jaah_count']}")

    # Import BTC training dependencies
    import torch
    import librosa

    sys.path.insert(0, str(btc_dir))
    from btc_model import BTC_model
    from utils.hparams import HParams
    from utils.mir_eval_modules import idx2voca_chord

    # Load config and model
    config = HParams.load(str(btc_dir / "run_config.yaml"))
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Training device: {device}")

    model = BTC_model(config=config.model).to(device)
    import numpy as np
    torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.ndarray, np.dtype, np.dtypes.Float64DType])
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    mean = ckpt['mean']
    std = ckpt['std']

    idx_to_chord = idx2voca_chord()
    chord_to_idx = {v: k for k, v in idx_to_chord.items()}

    # Build training pairs: (feature_segment, label_segment)
    n_timestep = config.model['timestep']  # 108
    time_unit = BTC_INST_LEN / n_timestep
    all_features = []
    all_labels = []

    for song_id in manifest['songs']:
        feat_path = FEATURES_DIR / f"{song_id}.npy"
        lab_path = LABELS_DIR / f"{song_id}.lab"

        if not feat_path.exists() or not lab_path.exists():
            continue

        try:
            # Load feature
            feature = np.load(str(feat_path))  # shape: [n_bins, n_frames]
            feature = feature.T  # [n_frames, n_bins]
            feature = (feature - mean) / std

            # Parse labels
            events = parse_lab_file(str(lab_path))
            if not events:
                continue

            # Create frame-level labels
            n_frames = feature.shape[0]
            frame_labels = np.full(n_frames, chord_to_idx.get('N', 169), dtype=np.int64)

            for start, end, chord in events:
                chord_idx = chord_to_idx.get(chord)
                if chord_idx is None:
                    # Try common mappings
                    simple_map = {
                        'maj': '', 'min': 'min', '7': '7', 'maj7': 'maj7',
                        'min7': 'min7', 'dim': 'dim', 'aug': 'aug',
                        'dim7': 'dim7', 'hdim7': 'hdim7', 'sus4': 'sus4',
                        'sus2': 'sus2', 'min6': 'min6', 'maj6': 'maj6',
                        'minmaj7': 'minmaj7', '9': '9',
                    }
                    chord_idx = chord_to_idx.get(chord, None)

                if chord_idx is None:
                    continue

                start_frame = int(start / time_unit)
                end_frame = int(end / time_unit)
                start_frame = max(0, min(start_frame, n_frames - 1))
                end_frame = max(0, min(end_frame, n_frames))
                frame_labels[start_frame:end_frame] = chord_idx

            # Pad to multiple of n_timestep
            num_pad = n_timestep - (n_frames % n_timestep)
            if num_pad < n_timestep:
                feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant")
                frame_labels = np.pad(frame_labels, (0, num_pad), mode="constant",
                                     constant_values=chord_to_idx.get('N', 169))

            # Split into segments
            n_segments = feature.shape[0] // n_timestep
            for seg in range(n_segments):
                s = seg * n_timestep
                e = s + n_timestep
                all_features.append(feature[s:e])
                all_labels.append(frame_labels[s:e])

        except Exception as ex:
            logger.warning(f"  {song_id}: skipped ({ex})")

    if not all_features:
        logger.error("No training data prepared!")
        return

    X = np.array(all_features, dtype=np.float32)
    Y = np.array(all_labels, dtype=np.int64)
    logger.info(f"Training data: {X.shape[0]} segments, {X.shape[1]} timesteps, {X.shape[2]} features")

    # Split train/val (90/10)
    n_total = X.shape[0]
    indices = np.random.permutation(n_total)
    n_val = max(1, int(n_total * 0.1))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]
    logger.info(f"Train: {len(train_idx)} segments, Val: {len(val_idx)} segments")

    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    batch_size = 32
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    save_dir = TRAINING_DIR / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(100):
        # Train
        model.train()
        train_losses = []
        perm = np.random.permutation(len(X_train))

        for batch_start in range(0, len(X_train), batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]
            x_batch = torch.tensor(X_train[batch_idx]).to(device)
            y_batch = torch.tensor(Y_train[batch_idx]).to(device)

            # Forward through BTC — use built-in loss (handles logits internally)
            encoder_output, _ = model.self_attn_layers(x_batch)
            loss = model.output_layer.loss(encoder_output, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # Validate
        model.eval()
        val_losses = []
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_start in range(0, len(X_val), batch_size):
                x_batch = torch.tensor(X_val[batch_start:batch_start + batch_size]).to(device)
                y_batch = torch.tensor(Y_val[batch_start:batch_start + batch_size]).to(device)

                encoder_output, _ = model.self_attn_layers(x_batch)
                loss = model.output_layer.loss(encoder_output, y_batch)
                val_losses.append(loss.item())

                # Get predictions for accuracy
                prediction, _ = model.output_layer(encoder_output)
                correct += (prediction == y_batch).sum().item()
                total += y_batch.numel()

        avg_val_loss = np.mean(val_losses)
        val_acc = correct / total if total > 0 else 0
        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']

        logger.info(f"Epoch {epoch+1:3d} | train_loss={avg_train_loss:.4f} | "
                    f"val_loss={avg_val_loss:.4f} | val_acc={val_acc:.4f} | lr={lr:.2e}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            save_path = save_dir / "btc_finetuned_best.pt"
            torch.save({
                'model': model.state_dict(),
                'mean': mean,
                'std': std,
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'train_songs': len(manifest['songs']),
            }, str(save_path))
            logger.info(f"  Saved best model (val_loss={avg_val_loss:.4f}, val_acc={val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")
    logger.info(f"Model saved to {save_dir / 'btc_finetuned_best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="BTC Fine-Tuning Pipeline")
    parser.add_argument('--step', choices=['download', 'features', 'train', 'all'],
                        default='all', help='Which step to run')
    parser.add_argument('--max-songs', type=int, default=None,
                        help='Max songs to download (for testing)')
    args = parser.parse_args()

    # Load all songs
    billboard = load_billboard_songs()
    jaah = load_jaah_songs()
    all_songs = {**billboard, **jaah}
    logger.info(f"Total songs: {len(all_songs)} ({len(billboard)} Billboard + {len(jaah)} JAAH)")

    if args.step in ('download', 'all'):
        logger.info("=== STEP 1: Download Audio ===")
        download_audio(all_songs, max_songs=args.max_songs)

    if args.step in ('features', 'all'):
        logger.info("=== STEP 2: Compute CQT Features ===")
        compute_features(all_songs)
        prepare_btc_training_data()

    if args.step in ('train', 'all'):
        logger.info("=== STEP 3: Fine-Tune BTC ===")
        run_finetune()


if __name__ == "__main__":
    main()
