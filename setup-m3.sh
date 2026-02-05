#!/bin/bash
# =============================================================================
# StemScribe Setup Script for Apple Silicon (M1/M2/M3)
# =============================================================================
# This script installs all dependencies for StemScribe on your Mac.
# Run with: chmod +x setup-m3.sh && ./setup-m3.sh
# =============================================================================

set -e  # Exit on any error

echo "🎵 StemScribe Setup for Apple Silicon"
echo "======================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo -e "${RED}⚠️  Warning: This script is optimized for Apple Silicon (M1/M2/M3).${NC}"
    echo "   Your Mac appears to be Intel-based. It will still work, just slower."
    read -p "   Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${YELLOW}Step 1/5: Checking Homebrew...${NC}"
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Add to path for Apple Silicon
    echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
    eval "$(/opt/homebrew/bin/brew shellenv)"
else
    echo -e "${GREEN}✓ Homebrew already installed${NC}"
fi

echo ""
echo -e "${YELLOW}Step 2/5: Installing system dependencies...${NC}"
brew install python@3.11 ffmpeg yt-dlp 2>/dev/null || true
echo -e "${GREEN}✓ System dependencies installed${NC}"

echo ""
echo -e "${YELLOW}Step 3/5: Creating Python virtual environment...${NC}"
VENV_DIR="$HOME/.stemscribe-venv"
if [ ! -d "$VENV_DIR" ]; then
    python3.11 -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created at $VENV_DIR${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

echo ""
echo -e "${YELLOW}Step 4/5: Installing Python packages (this takes a few minutes)...${NC}"
pip install --upgrade pip wheel setuptools

# Install PyTorch with MPS (Metal Performance Shaders) support for Apple Silicon
echo "Installing PyTorch with Apple Silicon GPU support..."
pip install torch torchvision torchaudio

# Install audio processing libraries
echo "Installing audio processing libraries..."
pip install demucs basic-pitch music21 librosa soundfile

# Install web server
echo "Installing web server..."
pip install flask flask-cors

# Install optional but useful libraries
pip install numpy scipy

echo -e "${GREEN}✓ Python packages installed${NC}"

echo ""
echo -e "${YELLOW}Step 5/5: Verifying installation...${NC}"

# Test imports
python3 << 'EOF'
import sys
print(f"Python: {sys.version}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"MPS (Metal GPU) available: {torch.backends.mps.is_available()}")

import demucs
print(f"Demucs: ✓")

import basic_pitch
print(f"Basic Pitch: ✓")

import music21
print(f"music21: ✓")

import librosa
print(f"librosa: ✓")

import flask
print(f"Flask: ✓")

print("\n✅ All dependencies verified!")
EOF

echo ""
echo -e "${GREEN}======================================"
echo "🎉 StemScribe Setup Complete!"
echo "======================================${NC}"
echo ""
echo "To start StemScribe:"
echo ""
echo "  1. Activate the environment:"
echo "     source ~/.stemscribe-venv/bin/activate"
echo ""
echo "  2. Navigate to StemScribe folder and run:"
echo "     python backend/app.py"
echo ""
echo "  3. Open frontend/index.html in your browser"
echo ""
echo "Or use the run script:"
echo "     ./run.sh"
echo ""
echo -e "${YELLOW}Note: First song processing downloads the AI model (~1.5GB).${NC}"
echo "      Subsequent songs will be much faster."
echo ""
