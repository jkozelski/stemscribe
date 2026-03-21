#!/bin/bash
# ============================================================
# pack_for_colab.sh — Bundle everything needed for Colab training
# ============================================================
# Creates stemscribe_btc_colab.tar.gz containing:
#   - btc_finetuned_best.pt (current checkpoint, ~12MB)
#   - run_config.yaml (BTC model config)
#   - features/*.npy (pre-computed CQT spectrograms)
#   - labels/*.lab (chord labels in Harte notation)
#   - manifest.json (dataset manifest)
#
# Usage:
#   cd ~/stemscribe/backend
#   bash pack_for_colab.sh
#
# Then upload stemscribe_btc_colab.tar.gz to Google Drive at:
#   My Drive/stemscribe_btc/stemscribe_btc_colab.tar.gz
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR"
BTC_DIR="$SCRIPT_DIR/../btc_chord"
TRAINING_DIR="$BACKEND_DIR/training_data/btc_finetune"
OUTPUT_DIR="$BACKEND_DIR"
TAR_NAME="stemscribe_btc_colab.tar.gz"
STAGING_DIR="/tmp/stemscribe_btc_colab_staging"

echo "=== StemScribe BTC Colab Packer ==="
echo ""

# Clean staging
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

# 1. Checkpoint
CKPT="$TRAINING_DIR/checkpoints/btc_finetuned_best.pt"
ORIG_CKPT="$BTC_DIR/test/btc_model_large_voca.pt"

if [ -f "$CKPT" ]; then
    echo "[1/5] Copying fine-tuned checkpoint ($(du -h "$CKPT" | cut -f1))..."
    cp "$CKPT" "$STAGING_DIR/btc_finetuned_best.pt"
elif [ -f "$ORIG_CKPT" ]; then
    echo "[1/5] Fine-tuned checkpoint not found, using original ($(du -h "$ORIG_CKPT" | cut -f1))..."
    cp "$ORIG_CKPT" "$STAGING_DIR/btc_finetuned_best.pt"
else
    echo "[1/5] ERROR: No checkpoint found!"
    echo "  Checked: $CKPT"
    echo "  Checked: $ORIG_CKPT"
    exit 1
fi

# 2. Config
CONFIG="$BTC_DIR/run_config.yaml"
if [ -f "$CONFIG" ]; then
    echo "[2/5] Copying run_config.yaml..."
    cp "$CONFIG" "$STAGING_DIR/run_config.yaml"
else
    echo "[2/5] ERROR: run_config.yaml not found at $CONFIG"
    exit 1
fi

# 3. Features
FEAT_DIR="$TRAINING_DIR/features"
if [ -d "$FEAT_DIR" ]; then
    N_FEAT=$(ls "$FEAT_DIR"/*.npy 2>/dev/null | wc -l | tr -d ' ')
    echo "[3/5] Copying $N_FEAT feature files..."
    mkdir -p "$STAGING_DIR/features"
    cp "$FEAT_DIR"/*.npy "$STAGING_DIR/features/"
else
    echo "[3/5] WARNING: No features directory at $FEAT_DIR"
    echo "  The notebook will need to create synthetic data."
fi

# 4. Labels
LAB_DIR="$TRAINING_DIR/labels"
if [ -d "$LAB_DIR" ]; then
    N_LAB=$(ls "$LAB_DIR"/*.lab 2>/dev/null | wc -l | tr -d ' ')
    echo "[4/5] Copying $N_LAB label files..."
    mkdir -p "$STAGING_DIR/labels"
    cp "$LAB_DIR"/*.lab "$STAGING_DIR/labels/"
else
    echo "[4/5] WARNING: No labels directory at $LAB_DIR"
fi

# 5. Manifest
MANIFEST="$TRAINING_DIR/manifest.json"
if [ -f "$MANIFEST" ]; then
    echo "[5/5] Copying manifest.json..."
    cp "$MANIFEST" "$STAGING_DIR/manifest.json"
else
    echo "[5/5] WARNING: No manifest.json found"
fi

# Create tar.gz
echo ""
echo "Creating $TAR_NAME..."
cd /tmp
tar -czf "$OUTPUT_DIR/$TAR_NAME" -C "$STAGING_DIR" .

# Report
SIZE=$(du -h "$OUTPUT_DIR/$TAR_NAME" | cut -f1)
echo ""
echo "=== Done ==="
echo "Output: $OUTPUT_DIR/$TAR_NAME ($SIZE)"
echo ""
echo "Contents:"
tar -tzf "$OUTPUT_DIR/$TAR_NAME" | head -20
N_FILES=$(tar -tzf "$OUTPUT_DIR/$TAR_NAME" | wc -l | tr -d ' ')
echo "  ... ($N_FILES files total)"
echo ""
echo "Next steps:"
echo "  1. Upload to Google Drive: My Drive/stemscribe_btc/stemscribe_btc_colab.tar.gz"
echo "  2. Open colab_chord_training.ipynb in Google Colab"
echo "  3. Set runtime to GPU (Runtime > Change runtime type > T4 GPU)"
echo "  4. Run all cells"

# Cleanup
rm -rf "$STAGING_DIR"
