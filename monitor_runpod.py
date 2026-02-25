#!/usr/bin/env python3
"""
RunPod Training Monitor
=======================
Checks piano training progress on RunPod from your Mac.
When training completes, downloads the model to backend.

Usage:
    python3 monitor_runpod.py          # Check once
    python3 monitor_runpod.py --watch  # Auto-refresh every 30s
    python3 monitor_runpod.py --download  # Download completed model
"""

import json
import sys
import time
import base64
import urllib.request
import ssl
from pathlib import Path

# RunPod Jupyter config
JUPYTER_URL = "https://ebntur79wv4kil-8888.proxy.runpod.net"
TOKEN = "71qntigc6atakcnbfnt4"
PROGRESS_FILE = "workspace/piano_training_progress.json"
MODEL_FILE = "workspace/piano_model_results/best_piano_model.pt"
LOCAL_MODEL_PATH = Path(__file__).parent / "backend" / "models" / "pretrained" / "best_piano_model.pt"

HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}


def api_get(path):
    """GET from Jupyter API."""
    url = f"{JUPYTER_URL}/api/contents/{path}?token={TOKEN}"
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        resp = urllib.request.urlopen(req, context=ctx, timeout=15)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


def check_progress():
    """Read training progress JSON."""
    data = api_get(PROGRESS_FILE)
    if "error" in data:
        # Try reading the training log instead
        log = api_get("workspace/training_output.log")
        if "error" in log:
            print("Cannot reach RunPod. Pod might be stopped.")
            return None

        # Parse last lines of log
        content = log.get("content", "")
        lines = content.strip().split("\n") if content else []
        if lines:
            print("No progress JSON yet. Last log lines:")
            for line in lines[-10:]:
                print(f"  {line}")
        return None

    content = data.get("content", "")
    if not content:
        print("Progress file empty.")
        return None

    try:
        progress = json.loads(content)
    except json.JSONDecodeError:
        print("Progress file corrupt.")
        return None

    return progress


def display_progress(p):
    """Pretty-print training progress."""
    status = p.get("status", "unknown")
    epoch = p.get("epoch", 0)
    total = p.get("total_epochs", 50)
    train_loss = p.get("train_loss", 0)
    val_loss = p.get("val_loss", 0)
    best = p.get("best_val_loss", 0)
    lr = p.get("lr", 0)
    epoch_time = p.get("epoch_time_sec", 0)
    tracks = p.get("total_tracks", 0)
    ts = p.get("timestamp", "?")

    pct = (epoch + 1) / total * 100 if total > 0 else 0
    bar_len = 30
    filled = int(bar_len * pct / 100)
    bar = "█" * filled + "░" * (bar_len - filled)

    remaining_epochs = total - epoch - 1
    eta_min = remaining_epochs * epoch_time / 60 if epoch_time > 0 else 0

    print()
    print(f"  Piano Training on RunPod")
    print(f"  ========================")
    print(f"  Status:     {status.upper()}")
    print(f"  Progress:   [{bar}] {pct:.0f}%")
    print(f"  Epoch:      {epoch + 1}/{total}")
    print(f"  Train Loss: {train_loss:.5f}")
    print(f"  Val Loss:   {val_loss:.5f}")
    print(f"  Best Loss:  {best:.5f}")
    print(f"  LR:         {lr:.2e}")
    print(f"  Epoch Time: {epoch_time:.0f}s")
    print(f"  Tracks:     {tracks}")
    print(f"  ETA:        ~{eta_min:.0f} min")
    print(f"  Updated:    {ts}")
    print()

    return status


def download_model():
    """Download the trained model from RunPod."""
    print(f"Downloading model from RunPod...")

    data = api_get(MODEL_FILE)
    if "error" in data:
        print(f"Error: {data['error']}")
        return False

    content = data.get("content", "")
    if not content:
        print("Model file is empty!")
        return False

    model_bytes = base64.b64decode(content)
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_MODEL_PATH.write_bytes(model_bytes)

    size_mb = len(model_bytes) / 1024 / 1024
    print(f"Downloaded: {LOCAL_MODEL_PATH} ({size_mb:.1f} MB)")
    return True


def check_files():
    """List workspace files and their sizes."""
    data = api_get("workspace")
    if "error" in data:
        print(f"Cannot reach RunPod: {data['error']}")
        return

    print("\n  RunPod /workspace files:")
    for item in sorted(data.get("content", []), key=lambda x: x.get("name", "")):
        name = item.get("name", "")
        size = item.get("size", 0)
        if size and size > 1_000_000:
            print(f"    {name}: {size / 1024 / 1024 / 1024:.2f} GB")
        elif size:
            print(f"    {name}: {size / 1024:.1f} KB")
        else:
            print(f"    {name}: (directory)")

    # Check model results
    results = api_get("workspace/piano_model_results")
    if "error" not in results:
        items = results.get("content", [])
        if items:
            print(f"\n  Model results ({len(items)} files):")
            for item in items:
                name = item.get("name", "")
                size = item.get("size", 0)
                if size > 1_000_000:
                    print(f"    {name}: {size / 1024 / 1024:.1f} MB")
                else:
                    print(f"    {name}: {size / 1024:.1f} KB")
    print()


if __name__ == "__main__":
    args = sys.argv[1:]

    if "--download" in args:
        download_model()
        sys.exit(0)

    if "--files" in args:
        check_files()
        sys.exit(0)

    if "--watch" in args:
        print("Watching training progress (Ctrl+C to stop)...")
        try:
            while True:
                progress = check_progress()
                if progress:
                    status = display_progress(progress)
                    if status == "complete":
                        print("Training complete! Run with --download to get the model.")
                        break
                else:
                    print("Waiting for training to start...")
                time.sleep(30)
        except KeyboardInterrupt:
            print("\nStopped watching.")
    else:
        check_files()
        progress = check_progress()
        if progress:
            display_progress(progress)
