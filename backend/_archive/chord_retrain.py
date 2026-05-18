"""
Retrain chord detector on real-world audio data.

Uses training data built by chord_training_pipeline.py.
Trains a new Transformer model with the same architecture as v8
but on actual song audio matched to Songsterr/UG ground truth.

Usage:
    python chord_retrain.py
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TRAINING_DIR = Path(__file__).parent / "training_data" / "chords"
MODEL_DIR = Path(__file__).parent / "models" / "pretrained"


class ChordDataset(Dataset):
    """Dataset of chroma frames with chord labels."""

    def __init__(self, chroma: np.ndarray, labels: np.ndarray, context: int = 21):
        self.chroma = chroma  # (12, n_frames)
        self.labels = labels  # (n_frames,)
        self.context = context
        self.pad = context // 2
        self.n_frames = chroma.shape[1]

        # Pad chroma for context window
        self.chroma_padded = np.pad(chroma, ((0, 0), (self.pad, self.pad)), mode="edge")

        # Only include frames with labels (non-N)
        # But also include some N frames for balance
        labeled_mask = labels > 0
        n_labeled = labeled_mask.sum()
        n_silent = min(n_labeled // 2, (~labeled_mask).sum())  # Up to 50% N frames

        labeled_indices = np.where(labeled_mask)[0]
        silent_indices = np.where(~labeled_mask)[0]
        if len(silent_indices) > n_silent:
            silent_indices = np.random.choice(silent_indices, n_silent, replace=False)

        self.indices = np.concatenate([labeled_indices, silent_indices])
        np.random.shuffle(self.indices)

        logger.info(f"Dataset: {len(self.indices)} samples ({n_labeled} labeled + {len(silent_indices)} silent)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        # Extract context window (context x 12)
        context_window = self.chroma_padded[:, i:i + self.context].T  # (context, 12)
        label = self.labels[i]
        return torch.tensor(context_window, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class ChordTransformer(nn.Module):
    """Transformer chord classifier — same architecture as v8 but retrainable."""

    def __init__(self, n_classes: int, d_model: int = 128, nhead: int = 8,
                 num_layers: int = 3, seq_len: int = 21, dropout: float = 0.2):
        super().__init__()
        self.proj = nn.Linear(12, d_model)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True,
            dropout=dropout,
        )
        self.tf = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self.seq_len = seq_len
        self.c = seq_len // 2

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x = self.proj(x) + self.pos
        x = self.tf(x)
        return self.fc(x[:, self.c, :])


def train():
    # Load training data
    data_path = TRAINING_DIR / "chord_training_data.npz"
    meta_path = TRAINING_DIR / "metadata.json"

    if not data_path.exists():
        logger.error(f"No training data at {data_path}. Run chord_training_pipeline.py first.")
        return

    logger.info("Loading training data...")
    data = np.load(data_path)
    chroma = data["chroma"]
    labels = data["labels"]
    vocab = list(data["vocab"])

    with open(meta_path) as f:
        meta = json.load(f)

    n_classes = len(vocab)
    logger.info(f"Chroma: {chroma.shape}, Labels: {labels.shape}")
    logger.info(f"Vocab: {n_classes} classes")
    logger.info(f"Songs: {len(meta['songs'])}")

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Top 10 chord classes:")
    sorted_idx = np.argsort(-counts)
    for i in sorted_idx[:10]:
        logger.info(f"  {vocab[unique[i]]}: {counts[i]} frames")

    # Split: 85% train, 15% val (by frame, not by song — simple for now)
    n_total = chroma.shape[1]
    n_train = int(0.85 * n_total)

    # Shuffle indices
    perm = np.random.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    # Build datasets
    context = 21
    train_ds = ChordDataset(chroma[:, train_idx], labels[train_idx], context=context)
    val_ds = ChordDataset(chroma[:, val_idx], labels[val_idx], context=context)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    # Model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model = ChordTransformer(n_classes=n_classes, seq_len=context).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # Class weights for imbalanced data
    class_counts = np.bincount(labels, minlength=n_classes).astype(float)
    class_counts[class_counts == 0] = 1
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * n_classes
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    best_val_acc = 0
    patience = 10
    no_improve = 0

    for epoch in range(100):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += X.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = criterion(logits, y)
                val_loss += loss.item() * X.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += X.size(0)

        train_acc = 100 * train_correct / max(train_total, 1)
        val_acc = 100 * val_correct / max(val_total, 1)
        avg_train_loss = train_loss / max(train_total, 1)
        avg_val_loss = val_loss / max(val_total, 1)

        logger.info(
            f"Epoch {epoch+1:3d} | "
            f"Train: loss={avg_train_loss:.4f} acc={train_acc:.1f}% | "
            f"Val: loss={avg_val_loss:.4f} acc={val_acc:.1f}% | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0

            # Save best model
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            save_path = MODEL_DIR / "v9_chord_model.pt"
            torch.save(model.state_dict(), save_path)

            # Save vocab
            vocab_path = MODEL_DIR / "v9_classes.json"
            with open(vocab_path, "w") as f:
                json.dump(vocab, f)

            logger.info(f"  -> New best! Saved to {save_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"\nTraining complete. Best val accuracy: {best_val_acc:.1f}%")
    logger.info(f"Model saved to {MODEL_DIR / 'v9_chord_model.pt'}")


if __name__ == "__main__":
    train()
