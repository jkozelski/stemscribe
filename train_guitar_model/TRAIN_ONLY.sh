#!/bin/bash
# =============================================================================
# 🎸 Guitar Lead/Rhythm Training ONLY (data already prepared)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🎸 =============================================="
echo "   Guitar Lead/Rhythm Model Training"
echo "   =============================================="
echo ""

# Activate venv
source "$SCRIPT_DIR/venv/bin/activate"

# Install training framework dependencies
echo "📦 Installing training dependencies..."
pip install ml_collections einops rotary_embedding_torch beartype omegaconf tqdm wandb 2>/dev/null
pip install audiomentations pedalboard auraloss torch_log_wmse torch_l1_snr 2>/dev/null || true
pip install segmentation_models_pytorch timm torchmetrics loralib 2>/dev/null || true

# Check dataset
SONG_COUNT=$(ls -d dataset/training/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "✓ Dataset: $SONG_COUNT songs ready"

if [ "$SONG_COUNT" -lt 5 ]; then
    echo "❌ Not enough training data. Run RUN_ME.sh first."
    exit 1
fi

# Run training
echo ""
echo "🏋️ Starting training..."
echo "   Model: MelBand-Roformer (guitar separation)"
echo "   Config: configs/config_guitar_lead_rhythm.yaml"
echo "   Data: dataset/training/"
echo "   Output: models/"
echo ""
echo "   Estimated time: 2-4 hours on M3 Max"
echo ""

python3 train.py \
    --model_type mel_band_roformer \
    --config_path configs/config_guitar_lead_rhythm.yaml \
    --data_path dataset/training \
    --results_path models \
    --dataset_type 1 \
    --device_ids 0 \
    --num_workers 4 \
    --pin_memory

echo ""
echo "🎉 =============================================="
echo "   Training Complete!"
echo "   =============================================="
echo ""
echo "   Model saved to: models/"
echo ""
echo "   To use in StemScribe:"
echo "   1. Copy the .ckpt file to stemscribe/models/"
echo "   2. Update enhanced_separator.py to use the new model"
echo ""
