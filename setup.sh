#!/bin/bash
# StemScriberr Setup Script
# Installs all required dependencies

set -e

echo "🎵 StemScriberr Setup"
echo "==================="
echo ""

# Check Python version
python3 --version || { echo "Error: Python 3 is required"; exit 1; }

echo "📦 Installing Python dependencies..."
echo ""

# Core dependencies
pip3 install --user flask flask-cors

# Audio processing (this may take a while)
echo "Installing PyTorch (CPU version)..."
pip3 install --user torch torchaudio

echo "Installing Demucs (stem separation)..."
pip3 install --user demucs

echo "Installing Basic Pitch (MIDI transcription)..."
pip3 install --user basic-pitch librosa soundfile

echo "Installing music21 (score assembly)..."
pip3 install --user music21

echo ""
echo "✅ Setup complete!"
echo ""
echo "Run ./run.sh to start StemScriberr"
