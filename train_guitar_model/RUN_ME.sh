#!/bin/bash
# =============================================================================
# 🎸 Guitar Lead/Rhythm Separation Model Training
# =============================================================================
#
# This script will:
# 1. Download 20 classic rock songs with distinct lead/rhythm guitar panning
# 2. Separate the guitar stems using Demucs
# 3. Split stereo guitars into lead/rhythm using AI analysis
# 4. Prepare the dataset for training
# 5. Train a custom MelBand-Roformer model
#
# REQUIREMENTS:
# - Python 3.10+
# - yt-dlp (will install if missing)
# - ffmpeg (brew install ffmpeg)
# - ~10GB disk space
# - ~4-8 hours for full training (GPU recommended)
#
# USAGE:
#   cd ~/stemscribe/train_guitar_model
#   chmod +x RUN_ME.sh
#   ./RUN_ME.sh
#
# =============================================================================

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🎸 =============================================="
echo "   Guitar Lead/Rhythm Separation Training"
echo "   =============================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.10+"
    exit 1
fi

echo "✓ Python: $(python3 --version)"

# Check ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found. Install with: brew install ffmpeg"
    exit 1
fi
echo "✓ ffmpeg available"

# Create and activate virtual environment (with system packages for demucs)
VENV_DIR="$SCRIPT_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

echo "✓ Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install Python dependencies in venv
echo ""
echo "📦 Installing Python dependencies..."
pip install --upgrade pip setuptools wheel

# Install torch first (required before demucs)
echo "📦 Installing PyTorch..."
pip install torch torchaudio

# Install demucs - try multiple methods
echo "📦 Installing Demucs..."
if pip install demucs --prefer-binary 2>/dev/null; then
    echo "✓ Demucs installed from wheel"
elif pip install git+https://github.com/facebookresearch/demucs 2>/dev/null; then
    echo "✓ Demucs installed from GitHub"
else
    echo "⚠️  Could not install demucs in venv, will try system demucs"
fi

# Install remaining deps
echo "📦 Installing other dependencies..."
pip install yt-dlp librosa soundfile numpy

echo "✓ yt-dlp: $(yt-dlp --version)"

# Create directories
mkdir -p dataset/downloads
mkdir -p dataset/stems
mkdir -p dataset/splits
mkdir -p dataset/training
mkdir -p models

# Run data preparation
echo ""
echo "🎵 Starting data preparation..."
echo "   This will download 20 songs and process them."
echo "   Estimated time: 30-60 minutes"
echo ""

python3 prepare_dataset.py --output ./dataset --limit 20

# Check if we have enough data
SONG_COUNT=$(ls -d dataset/training/*/ 2>/dev/null | wc -l | tr -d ' ')

if [ "$SONG_COUNT" -lt 5 ]; then
    echo ""
    echo "⚠️  Only $SONG_COUNT songs processed successfully."
    echo "   Minimum 5 songs needed for training."
    echo "   Check the logs above for download/processing errors."
    exit 1
fi

echo ""
echo "✓ Dataset ready: $SONG_COUNT songs"
echo ""

# Auto-continue to training (unattended mode)
echo ""
echo "🚀 Continuing to training automatically..."
echo "   (Running in unattended mode)"

# Run training
echo ""
echo "🏋️ Starting training..."
echo "   Model: MelBand-Roformer"
echo "   Output: ./models/"
echo ""

python3 -c "
import sys
sys.path.insert(0, '.')
from train import main
main()
" --config_path config.yaml \
  --dataset_type custom \
  --train_data ./dataset/training \
  --results_path ./models \
  --model_type mel_band_roformer

echo ""
echo "🎉 =============================================="
echo "   Training Complete!"
echo "   =============================================="
echo ""
echo "   Model saved to: ./models/"
echo ""
echo "   To use in StemScribe, copy the model checkpoint"
echo "   and update enhanced_separator.py"
echo ""
