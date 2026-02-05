#!/bin/bash
# StemScribe Launcher
# Starts both the backend API and serves the frontend
# Includes caffeinate mode to prevent Mac from sleeping during long jobs

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo ""
echo -e "${PURPLE}  ╔═══════════════════════════════════════╗${NC}"
echo -e "${PURPLE}  ║         ${CYAN}🎵 StemScribe 🎵${PURPLE}              ║${NC}"
echo -e "${PURPLE}  ║   ${NC}Stem Separation & Transcription${PURPLE}     ║${NC}"
echo -e "${PURPLE}  ╚═══════════════════════════════════════╝${NC}"
echo ""

# Add local Python packages to path
export PATH="$PATH:$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    echo -e "${GREEN}✓${NC} Using Python venv"
    source "$SCRIPT_DIR/venv/bin/activate"
else
    export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH"
fi

# Check for yt-dlp
if command -v yt-dlp &> /dev/null; then
    echo -e "${GREEN}✓${NC} yt-dlp found: $(which yt-dlp)"
else
    echo -e "${RED}✗${NC} yt-dlp not found"
    echo -e "  Install with: ${CYAN}brew install yt-dlp${NC}"
fi

# Check for Python dependencies
echo -e "${BLUE}→${NC} Checking Python dependencies..."
python -c "import librosa; import flask; print('  Dependencies OK')" 2>/dev/null || {
    echo -e "${RED}✗${NC} Missing Python dependencies"
    echo -e "  Run: ${CYAN}./setup.sh${NC}"
    exit 1
}

# Kill any existing processes on our ports
echo -e "${BLUE}→${NC} Cleaning up old processes..."
pkill -f "python3.*app.py" 2>/dev/null || true
pkill -f "python3 -m http.server 3000" 2>/dev/null || true
sleep 1

# Start caffeinate to prevent sleep
echo -e "${GREEN}☕${NC} Caffeinate mode: ${CYAN}ON${NC} (Mac won't sleep during processing)"
caffeinate -dims &
CAFFEINATE_PID=$!

# Start backend API
echo -e "${BLUE}→${NC} Starting backend API on ${CYAN}http://localhost:5000${NC}..."
cd backend
python app.py 2>&1 | while read line; do
    echo -e "  ${PURPLE}[API]${NC} $line"
done &
BACKEND_PID=$!
cd ..

sleep 2

# Start frontend server
echo -e "${BLUE}→${NC} Starting frontend on ${CYAN}http://localhost:3000${NC}..."
cd frontend
python -m http.server 3000 2>&1 | while read line; do
    echo -e "  ${CYAN}[WEB]${NC} $line"
done &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ StemScribe is running!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}🌐 Open in browser:${NC}  http://localhost:3000"
echo -e "  ${CYAN}📡 API endpoint:${NC}     http://localhost:5000"
echo ""
echo -e "  ${PURPLE}Features:${NC}"
echo -e "    • Paste YouTube/SoundCloud/Bandcamp URLs"
echo -e "    • Upload local audio files"
echo -e "    • Get stems + MIDI for Logic Pro 12"
echo ""
echo -e "  ${GREEN}☕ Caffeinate mode is ON${NC} - your Mac won't sleep"
echo ""
echo -e "  Press ${RED}Ctrl+C${NC} to stop all services."
echo ""

# Handle shutdown gracefully
cleanup() {
    echo ""
    echo -e "${BLUE}→${NC} Shutting down..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    kill $CAFFEINATE_PID 2>/dev/null || true
    echo -e "${GREEN}✓${NC} All services stopped"
    echo -e "${NC}☕ Caffeinate disabled - Mac can sleep again"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for either process to exit
wait
