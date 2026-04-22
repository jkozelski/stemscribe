#!/bin/bash
# Deploy StemScribe's Modal stem separation function to cloud GPU
# Usage: ./deploy_modal.sh

set -e

echo "Deploying StemScribe Modal separator..."
cd ~/stemscribe/backend

# Stop existing deployment if present, then redeploy
../venv311/bin/python -m modal app stop stemscribe-separator 2>/dev/null || true
../venv311/bin/python -m modal deploy modal_separator.py

echo ""
echo "Deployment complete!"
echo "To enable Modal in production, set MODAL_ENABLED=true in .env"
echo "To test: cd ~/stemscribe/backend && ../venv311/bin/python -m modal run modal_separator.py"
