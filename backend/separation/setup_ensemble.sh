#!/bin/bash
# StemScribe Ensemble Separation System Setup
# Installs dependencies for Moises.ai-quality audio separation

set -e

echo "🎵 StemScribe Ensemble Separator Setup"
echo "========================================"
echo ""

# Check if we're on macOS with Apple Silicon
if [[ "$(uname)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
    echo "✅ Detected Apple Silicon Mac"
    DEVICE="mps"
else
    echo "ℹ️  Detected other platform (will use CUDA or CPU)"
    DEVICE="auto"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Python version: $PYTHON_VERSION"

# Install PyTorch with MPS support
echo ""
echo "📦 Installing PyTorch (with MPS support for Apple Silicon)..."
pip3 install --upgrade torch torchaudio --break-system-packages 2>/dev/null || \
pip3 install --upgrade torch torchaudio

# Install Demucs
echo ""
echo "📦 Installing Demucs v4..."
pip3 install --upgrade demucs --break-system-packages 2>/dev/null || \
pip3 install --upgrade demucs

# Install audio processing libraries
echo ""
echo "📦 Installing audio processing libraries..."
pip3 install --upgrade soundfile librosa scipy --break-system-packages 2>/dev/null || \
pip3 install --upgrade soundfile librosa scipy

# Install post-processing libraries
echo ""
echo "📦 Installing post-processing libraries..."
pip3 install --upgrade noisereduce pyloudnorm psutil --break-system-packages 2>/dev/null || \
pip3 install --upgrade noisereduce pyloudnorm psutil

# Test the installation
echo ""
echo "🧪 Testing installation..."
python3 -c "
import torch
import demucs
import soundfile
import librosa

print('✅ PyTorch:', torch.__version__)
print('   MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
print('   CUDA available:', torch.cuda.is_available())
print('✅ Demucs: installed')
print('✅ SoundFile: installed')
print('✅ Librosa:', librosa.__version__)
"

echo ""
echo "✅ Installation complete!"
echo ""
echo "To use the ensemble separator, start StemScribe and select 'Ensemble' mode:"
echo "  - Via API: POST /api/url with {\"url\": \"...\", \"ensemble\": true}"
echo "  - Or in the UI: Check the 'Ensemble (HQ)' option"
echo ""
echo "The ensemble separator will use:"
if [[ "$DEVICE" == "mps" ]]; then
    echo "  - Device: Apple Metal Performance Shaders (MPS)"
else
    echo "  - Device: Auto-detect (CUDA > MPS > CPU)"
fi
echo "  - Models: htdemucs_ft + htdemucs (voting for best quality)"
echo "  - Post-processing: Bleed reduction, noise removal, phase alignment"
