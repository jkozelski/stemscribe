FROM python:3.11-slim

WORKDIR /app

# System dependencies for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy frontend for static serving
COPY frontend/ ./frontend/

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT:-5555}/api/health')" || exit 1

# Railway sets PORT env var; default to 5555 for local
EXPOSE ${PORT:-5555}

# Gunicorn: 1 worker (jobs dict is in-memory, multiple workers can't share it)
# --preload shares model memory across workers
CMD gunicorn \
    --bind 0.0.0.0:${PORT:-5555} \
    --workers 1 \
    --timeout 300 \
    --preload \
    --access-logfile - \
    --error-logfile - \
    backend.app:app
