"""
BTC Retrain with Pitch Augmentation + Class Oversampling
=========================================================
Addresses the class imbalance bias that causes F#m -> Am, C#m -> Cm, etc.

Strategy:
1. Load existing 988 songs' CQT features + labels
2. On-the-fly pitch augmentation: each epoch randomly transposes each song segment
3. Class-weighted loss to further boost underrepresented chords
4. Fine-tune from original BTC checkpoint
5. Save new checkpoint, keeping old as backup

Usage:
    cd ~/stemscribe/backend
    ../venv311/bin/python btc_retrain_augmented.py
    ../venv311/bin/python btc_retrain_augmented.py --test-only
"""

import json
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import Counter
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
BTC_DIR = BASE_DIR.parent / "btc_chord"
TRAINING_DIR = BASE_DIR / "training_data" / "btc_finetune"
FEATURES_DIR = TRAINING_DIR / "features"
LABELS_DIR = TRAINING_DIR / "labels"
CHECKPOINT_DIR = TRAINING_DIR / "checkpoints"

# BTC config
BTC_SR = 22050
BTC_N_BINS = 144
BTC_BINS_PER_OCTAVE = 24
BTC_HOP_LENGTH = 2048
BTC_INST_LEN = 10.0
BINS_PER_SEMITONE = BTC_BINS_PER_OCTAVE // 12  # = 2
N_TIMESTEP = 108

ROOT_LIST = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
QUALITY_LIST = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7',
                'maj7', '7', 'dim7', 'hdim7', 'sus2', 'sus4']


def build_chord_mappings():
    """Build idx <-> chord mappings for the 170-class BTC vocabulary."""
    idx_to_chord = {}
    idx_to_chord[169] = 'N'
    idx_to_chord[168] = 'X'
    for i in range(168):
        root_idx = i // 14
        quality_idx = i % 14
        root = ROOT_LIST[root_idx]
        quality = QUALITY_LIST[quality_idx]
        if quality_idx != 1:
            chord = f"{root}:{quality}"
        else:
            chord = root
        idx_to_chord[i] = chord
    chord_to_idx = {v: k for k, v in idx_to_chord.items()}
    return idx_to_chord, chord_to_idx


def transpose_label(label_val, semitones):
    """Transpose a single chord label index by N semitones."""
    if label_val >= 168:  # N or X
        return label_val
    root_idx = label_val // 14
    quality_idx = label_val % 14
    new_root_idx = (root_idx + semitones) % 12
    return new_root_idx * 14 + quality_idx


def transpose_cqt(feature, semitones):
    """Transpose CQT features by rolling frequency bins.
    feature shape: [n_frames, 144]
    """
    shift = semitones * BINS_PER_SEMITONE
    return np.roll(feature, shift, axis=1)


def parse_lab_file(lab_path):
    events = []
    with open(lab_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                parts = line.split()
            if len(parts) >= 3:
                try:
                    events.append((float(parts[0]), float(parts[1]), parts[2]))
                except ValueError:
                    continue
    return events


class PitchAugmentedBTCDataset(Dataset):
    """Dataset that applies on-the-fly pitch augmentation.

    Stores raw segments (un-normalized features + labels).
    Each __getitem__ randomly transposes by 0-11 semitones.
    """

    def __init__(self, segments, mean, std, augment=True):
        """
        segments: list of (feature_segment, label_segment) tuples
                  feature_segment: [108, 144] float32
                  label_segment: [108] int64
        """
        self.segments = segments
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        feat, lbl = self.segments[idx]

        if self.augment:
            # Random transposition 0-11 semitones
            shift = np.random.randint(0, 12)
            if shift > 0:
                feat = transpose_cqt(feat, shift)
                lbl = np.array([transpose_label(int(l), shift) for l in lbl], dtype=np.int64)

        # Normalize
        feat = (feat - self.mean) / self.std

        return torch.tensor(feat, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)


def load_segments(chord_to_idx):
    """Load all feature/label pairs as segments."""
    manifest_path = TRAINING_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    orig_ckpt_path = BTC_DIR / "test" / "btc_model_large_voca.pt"
    torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.ndarray, np.dtype, np.dtypes.Float64DType])
    ckpt = torch.load(str(orig_ckpt_path), map_location='cpu', weights_only=False)
    mean = ckpt['mean']
    std = ckpt['std']

    time_unit = BTC_INST_LEN / N_TIMESTEP
    segments = []
    loaded = 0

    for song_id in manifest['songs']:
        feat_path = FEATURES_DIR / f"{song_id}.npy"
        lab_path = LABELS_DIR / f"{song_id}.lab"
        if not feat_path.exists() or not lab_path.exists():
            continue

        try:
            feature = np.load(str(feat_path)).T  # [n_frames, n_bins]
            events = parse_lab_file(str(lab_path))
            if not events:
                continue

            n_frames = feature.shape[0]
            frame_labels = np.full(n_frames, chord_to_idx.get('N', 169), dtype=np.int64)
            for start, end, chord in events:
                chord_idx = chord_to_idx.get(chord)
                if chord_idx is None:
                    continue
                sf = max(0, min(int(start / time_unit), n_frames - 1))
                ef = max(0, min(int(end / time_unit), n_frames))
                frame_labels[sf:ef] = chord_idx

            # Pad and segment
            num_pad = N_TIMESTEP - (n_frames % N_TIMESTEP)
            if num_pad < N_TIMESTEP:
                feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant")
                frame_labels = np.pad(frame_labels, (0, num_pad), constant_values=169)

            n_segs = feature.shape[0] // N_TIMESTEP
            for s in range(n_segs):
                start = s * N_TIMESTEP
                end = start + N_TIMESTEP
                segments.append((
                    feature[start:end].astype(np.float32),
                    frame_labels[start:end].copy()
                ))

            loaded += 1
        except Exception as e:
            logger.warning(f"  {song_id}: skipped ({e})")

    logger.info(f"Loaded {loaded} songs -> {len(segments)} segments")
    return segments, mean, std


def compute_class_weights_from_segments(segments, n_classes=170):
    """Compute inverse-frequency class weights."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for _, lbl in segments:
        for v in lbl:
            counts[int(v)] += 1
    counts = counts + 1.0  # smoothing
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    weights = np.clip(weights, 0.1, 10.0)
    return weights


def run_training():
    idx_to_chord, chord_to_idx = build_chord_mappings()

    # Step 1: Load segments
    logger.info("=" * 60)
    logger.info("STEP 1: Loading training data")
    logger.info("=" * 60)
    segments, mean, std = load_segments(chord_to_idx)
    if not segments:
        logger.error("No training data!")
        return

    # Step 2: Split train/val
    logger.info("=" * 60)
    logger.info("STEP 2: Train/val split")
    logger.info("=" * 60)
    np.random.seed(42)
    n_total = len(segments)
    indices = np.random.permutation(n_total)
    n_val = max(1, int(n_total * 0.1))
    val_segments = [segments[i] for i in indices[:n_val]]
    train_segments = [segments[i] for i in indices[n_val:]]
    logger.info(f"Train: {len(train_segments)} segments, Val: {len(val_segments)} segments")

    # Effective training size with augmentation:
    # Each epoch, every segment gets a random transposition -> 12x effective diversity
    logger.info(f"With on-the-fly 12-semitone augmentation, effective diversity = 12x")

    # Step 3: Compute class weights
    logger.info("=" * 60)
    logger.info("STEP 3: Class weights")
    logger.info("=" * 60)
    class_weights = compute_class_weights_from_segments(segments)
    for root_idx, root in enumerate(ROOT_LIST):
        root_weights = class_weights[root_idx * 14:(root_idx + 1) * 14]
        logger.info(f"  {root:3s}: avg_weight={np.mean(root_weights):.3f}")

    # Step 4: Create datasets
    train_ds = PitchAugmentedBTCDataset(train_segments, mean, std, augment=True)
    val_ds = PitchAugmentedBTCDataset(val_segments, mean, std, augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0,
                             pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Step 5: Load model
    logger.info("=" * 60)
    logger.info("STEP 5: Loading BTC model")
    logger.info("=" * 60)

    sys.path.insert(0, str(BTC_DIR))
    from btc_model import BTC_model
    from utils.hparams import HParams

    config = HParams.load(str(BTC_DIR / "run_config.yaml"))
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = BTC_model(config=config.model).to(device)

    # Load ORIGINAL BTC checkpoint (the previous fine-tune degraded accuracy)
    orig_ckpt = BTC_DIR / "test" / "btc_model_large_voca.pt"
    finetuned_ckpt = CHECKPOINT_DIR / "btc_finetuned_best.pt"

    logger.info(f"Loading ORIGINAL BTC checkpoint: {orig_ckpt}")
    ckpt = torch.load(str(orig_ckpt), map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])

    # Backup existing fine-tuned checkpoint
    if finetuned_ckpt.exists():
        backup_path = CHECKPOINT_DIR / "btc_finetuned_best_backup.pt"
        if not backup_path.exists():
            import shutil
            shutil.copy2(str(finetuned_ckpt), str(backup_path))
            logger.info(f"Backed up old checkpoint to {backup_path}")
        else:
            backup_path2 = CHECKPOINT_DIR / f"btc_finetuned_best_pre_augment.pt"
            if not backup_path2.exists():
                import shutil
                shutil.copy2(str(finetuned_ckpt), str(backup_path2))
                logger.info(f"Backed up old checkpoint to {backup_path2}")

    # Step 6: Training
    logger.info("=" * 60)
    logger.info("STEP 6: Training with pitch augmentation")
    logger.info("=" * 60)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    n_epochs = 60

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Train
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            encoder_output, _ = model.self_attn_layers(x_batch)
            logits = model.output_layer.output_projection(encoder_output)
            log_probs = F.log_softmax(logits, -1)
            loss = F.nll_loss(log_probs.view(-1, 170), y_batch.view(-1),
                            weight=class_weights_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.view(-1, 170).argmax(dim=-1)
            train_correct += (preds == y_batch.view(-1)).sum().item()
            train_total += y_batch.numel()

        avg_train_loss = np.mean(train_losses)
        train_acc = train_correct / max(train_total, 1)

        # Validate (no augmentation)
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        root_correct = Counter()
        root_total = Counter()

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                encoder_output, _ = model.self_attn_layers(x_batch)
                logits = model.output_layer.output_projection(encoder_output)
                log_probs = F.log_softmax(logits, -1)
                loss = F.nll_loss(log_probs.view(-1, 170), y_batch.view(-1),
                                weight=class_weights_tensor)
                val_losses.append(loss.item())

                preds = logits.view(-1, 170).argmax(dim=-1)
                flat_y = y_batch.view(-1)
                val_correct += (preds == flat_y).sum().item()
                val_total += flat_y.numel()

                for pred, true in zip(preds.cpu().numpy(), flat_y.cpu().numpy()):
                    if true < 168:
                        root_idx = true // 14
                        root_total[root_idx] += 1
                        if pred // 14 == root_idx:  # Root correct (quality may differ)
                            root_correct[root_idx] += 1

        avg_val_loss = np.mean(val_losses)
        val_acc = val_correct / max(val_total, 1)
        scheduler.step(avg_val_loss)
        lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch+1:3d}/{n_epochs} | "
            f"train_loss={avg_train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={avg_val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={lr:.2e} | {epoch_time:.0f}s"
        )

        # Per-root accuracy every 5 epochs
        if (epoch + 1) % 5 == 0:
            logger.info("  Per-root val accuracy (root match):")
            for ri in range(12):
                tot = root_total.get(ri, 0)
                cor = root_correct.get(ri, 0)
                acc = cor / max(tot, 1)
                logger.info(f"    {ROOT_LIST[ri]:3s}: {acc:.4f} ({cor}/{tot})")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = avg_val_loss
            patience_counter = 0

            save_path = CHECKPOINT_DIR / "btc_finetuned_best.pt"
            torch.save({
                'model': model.state_dict(),
                'mean': mean,
                'std': std,
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_acc': val_acc,
                'train_songs': len(train_segments),
                'augmentation': 'on_the_fly_12_semitone_pitch_shift + class_weighted_loss',
            }, str(save_path))
            logger.info(f"  -> New best! val_acc={val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info("=" * 60)
    logger.info(f"Training complete. Best val_acc={best_val_acc:.4f}, loss={best_val_loss:.4f}")
    logger.info(f"Checkpoint: {CHECKPOINT_DIR / 'btc_finetuned_best.pt'}")
    logger.info("=" * 60)


def test_on_songs():
    """Test the retrained model on The Time Comes and Little Wing."""
    logger.info("=" * 60)
    logger.info("TESTING retrained model")
    logger.info("=" * 60)

    import librosa

    sys.path.insert(0, str(BTC_DIR))
    from btc_model import BTC_model
    from utils.hparams import HParams

    config = HParams.load(str(BTC_DIR / "run_config.yaml"))
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    idx_to_chord, chord_to_idx = build_chord_mappings()
    time_unit = BTC_INST_LEN / N_TIMESTEP

    test_songs = {
        "The Time Comes": {
            "stems_dir": Path("/Users/jeffkozelski/stemscribe/outputs/4afcef17-c1ed-4bd1-8eb8-e43afe027ba5/stems/htdemucs_6s/Kozelski_-_The_Time_Comes__-__Lyric_Video"),
            "ground_truth": ["F#:min", "A", "B:min", "E", "B", "D", "C#:min", "G"],
            "key": "F#m / A major",
        },
        "Little Wing": {
            "stems_dir": Path("/Users/jeffkozelski/stemscribe/outputs/a0db058f/stems/htdemucs_6s/Jimi_Hendrix_-_Little_wing_-_HQ"),
            # Hendrix Eb standard -> concert pitch
            "ground_truth": ["D#:min", "F#", "G#", "A#:min", "G#:min", "B", "C#"],
            "ground_truth_guitar": ["E:min", "G", "A", "B:min", "A:min", "C", "D"],
            "key": "D#m (Eb standard, concert pitch)",
        },
    }

    checkpoints_to_test = {}
    backup_path = CHECKPOINT_DIR / "btc_finetuned_best_backup.pt"
    orig_path = BTC_DIR / "test" / "btc_model_large_voca.pt"
    new_path = CHECKPOINT_DIR / "btc_finetuned_best.pt"

    checkpoints_to_test["ORIGINAL BTC"] = orig_path
    if backup_path.exists():
        checkpoints_to_test["PREVIOUS FINETUNE"] = backup_path
    if new_path.exists():
        checkpoints_to_test["NEW (augmented retrain)"] = new_path

    for ckpt_name, ckpt_path in checkpoints_to_test.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {ckpt_name}")
        logger.info(f"  Path: {ckpt_path}")
        logger.info(f"{'='*60}")

        model = BTC_model(config=config.model).to(device)
        torch.serialization.add_safe_globals([np._core.multiarray.scalar, np.ndarray, np.dtype, np.dtypes.Float64DType])
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        ckpt_mean = ckpt['mean']
        ckpt_std = ckpt['std']

        if 'epoch' in ckpt:
            logger.info(f"  Epoch: {ckpt['epoch']}, val_acc: {ckpt.get('val_acc', '?')}")

        for song_name, song_info in test_songs.items():
            stems_dir = song_info["stems_dir"]
            if not stems_dir.exists():
                logger.warning(f"  {song_name}: stems not found")
                continue

            harmonic_stems = ['guitar', 'piano', 'bass']
            mixed = None
            sr = BTC_SR

            for stem in harmonic_stems:
                stem_path = stems_dir / f"{stem}.mp3"
                if stem_path.exists():
                    wav, _ = librosa.load(str(stem_path), sr=sr, mono=True)
                    if mixed is None:
                        mixed = wav
                    else:
                        min_len = min(len(mixed), len(wav))
                        mixed = mixed[:min_len] + wav[:min_len]

            if mixed is None:
                continue

            # Tuning compensation
            tuning = librosa.estimate_tuning(y=mixed, sr=sr)
            logger.info(f"  {song_name}: tuning={tuning:.3f} semitones")
            if abs(tuning) > 0.05:
                mixed = librosa.effects.pitch_shift(mixed, sr=sr, n_steps=-tuning)

            # CQT features
            feature = None
            current = 0
            while len(mixed) > current + int(sr * BTC_INST_LEN):
                si = int(current)
                ei = int(current + sr * BTC_INST_LEN)
                tmp = librosa.cqt(mixed[si:ei], sr=sr, n_bins=BTC_N_BINS,
                                 bins_per_octave=BTC_BINS_PER_OCTAVE, hop_length=BTC_HOP_LENGTH)
                feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
                current = ei
            if current < len(mixed):
                tmp = librosa.cqt(mixed[int(current):], sr=sr, n_bins=BTC_N_BINS,
                                 bins_per_octave=BTC_BINS_PER_OCTAVE, hop_length=BTC_HOP_LENGTH)
                feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)

            feature = np.log(np.abs(feature) + 1e-6)
            feature = feature.T
            feature = (feature - ckpt_mean) / ckpt_std

            n_frames = feature.shape[0]
            num_pad = N_TIMESTEP - (n_frames % N_TIMESTEP)
            if num_pad < N_TIMESTEP:
                feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant")
            num_instance = feature.shape[0] // N_TIMESTEP

            predictions = []
            with torch.no_grad():
                feat_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
                for t in range(num_instance):
                    seg = feat_tensor[:, N_TIMESTEP * t:N_TIMESTEP * (t + 1), :]
                    encoder_output, _ = model.self_attn_layers(seg)
                    prediction, _ = model.output_layer(encoder_output)
                    predictions.extend(prediction.squeeze().cpu().numpy().tolist())

            # Consolidate into chord regions
            chord_durations = Counter()
            if predictions:
                prev = predictions[0]
                start_t = 0.0
                for i, cidx in enumerate(predictions[1:], 1):
                    if cidx != prev:
                        name = idx_to_chord.get(prev, '?')
                        dur = time_unit * i - start_t
                        if dur > 0.2 and name not in ('N', 'X'):
                            chord_durations[name] += dur
                        start_t = time_unit * i
                        prev = cidx
                name = idx_to_chord.get(prev, '?')
                dur = time_unit * len(predictions) - start_t
                if dur > 0.2 and name not in ('N', 'X'):
                    chord_durations[name] += dur

            detected = sorted(chord_durations.items(), key=lambda x: -x[1])
            gt_set = set(song_info['ground_truth'])
            detected_set = {c for c, d in detected if d > 1.0}

            logger.info(f"\n  {song_name} (key: {song_info['key']}):")
            logger.info(f"  Ground truth: {sorted(gt_set)}")
            logger.info(f"  Detected (>1s):")
            for chord, dur in detected[:12]:
                marker = " *" if chord in gt_set else ""
                logger.info(f"    {chord:15s}: {dur:6.1f}s{marker}")

            overlap = gt_set & detected_set
            logger.info(f"  Match: {len(overlap)}/{len(gt_set)} | "
                       f"Extra: {sorted(detected_set - gt_set)} | "
                       f"Missing: {sorted(gt_set - detected_set)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-only', action='store_true')
    args = parser.parse_args()

    if args.test_only:
        test_on_songs()
    else:
        run_training()
        test_on_songs()
