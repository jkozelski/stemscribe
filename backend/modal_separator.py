"""
Modal cloud GPU stem separation for StemScribe.

Deploys a serverless GPU function that runs htdemucs_6s stem separation
on Modal's infrastructure instead of the local M3. Model weights are
cached in a Modal Volume to avoid re-downloading on cold starts.

Deploy:  cd ~/stemscribe/backend && ../venv311/bin/python -m modal deploy modal_separator.py
Test:    cd ~/stemscribe/backend && ../venv311/bin/python -m modal run modal_separator.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal Image — pre-bake all heavy dependencies so cold starts are fast
# ---------------------------------------------------------------------------

stemscribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "sox")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "demucs==4.0.1",
        "numpy<2",
        "scipy",
        "soundfile",
        "pydub",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# Volume for caching model weights (~1 GB for htdemucs_6s)
model_volume = modal.Volume.from_name("stemscribe-model-cache", create_if_missing=True)

app = modal.App("stemscribe-separator", image=stemscribe_image)

# ---------------------------------------------------------------------------
# GPU Function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    timeout=600,
    volumes={"/model-cache": model_volume},
    memory=8192,
)
def separate_stems_gpu(audio_bytes: bytes, filename: str = "input.mp3") -> dict:
    """
    Run htdemucs_6s stem separation on a cloud GPU.

    Args:
        audio_bytes: Raw audio file bytes (MP3, WAV, or FLAC)
        filename: Original filename (used to determine format)

    Returns:
        Dict mapping stem name to MP3 bytes:
        {"vocals": b"...", "guitar": b"...", "bass": b"...",
         "drums": b"...", "piano": b"...", "other": b"..."}
    """
    import os
    import subprocess
    import tempfile
    from pathlib import Path

    # Point torch hub / demucs cache at the persistent volume
    os.environ["TORCH_HOME"] = "/model-cache/torch"
    os.environ["XDG_CACHE_HOME"] = "/model-cache"

    # Write input audio to a temp file
    work_dir = Path(tempfile.mkdtemp())
    input_path = work_dir / filename
    input_path.write_bytes(audio_bytes)

    output_dir = work_dir / "separated"
    output_dir.mkdir()

    # Run demucs via CLI (most reliable way)
    cmd = [
        "python", "-m", "demucs",
        "--name", "htdemucs_6s",
        "--out", str(output_dir),
        "--mp3",                  # Output as MP3 directly
        "--mp3-bitrate", "320",
        "--device", "cuda",
        str(input_path),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=540)

    if result.returncode != 0:
        # Include both stdout and stderr for debugging
        error_detail = result.stderr or result.stdout or "Unknown error"
        raise RuntimeError(f"Demucs separation failed (exit {result.returncode}): {error_detail}")

    # Collect output stems
    # Demucs outputs to: output_dir/htdemucs_6s/<input_stem>/*.mp3
    input_stem = input_path.stem
    stems_dir = output_dir / "htdemucs_6s" / input_stem

    if not stems_dir.exists():
        # Sometimes the directory name varies — search for it
        for candidate in output_dir.rglob("*.mp3"):
            stems_dir = candidate.parent
            break
        else:
            # Try WAV as fallback
            for candidate in output_dir.rglob("*.wav"):
                stems_dir = candidate.parent
                break
            else:
                raise RuntimeError(
                    f"No output stems found. Output dir contents: "
                    f"{list(output_dir.rglob('*'))}"
                )

    stems_result = {}
    for stem_file in sorted(stems_dir.iterdir()):
        if stem_file.suffix in (".mp3", ".wav"):
            stem_name = stem_file.stem  # e.g. "vocals", "drums", etc.

            # If WAV output, convert to MP3 for transfer
            if stem_file.suffix == ".wav":
                mp3_path = stem_file.with_suffix(".mp3")
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", str(stem_file),
                        "-codec:a", "libmp3lame", "-b:a", "320k",
                        str(mp3_path),
                    ],
                    capture_output=True,
                    timeout=120,
                )
                if mp3_path.exists():
                    stem_file = mp3_path

            stems_result[stem_name] = stem_file.read_bytes()
            size_mb = len(stems_result[stem_name]) / (1024 * 1024)
            print(f"  Stem '{stem_name}': {size_mb:.1f} MB")

    # Commit volume so weights persist for next cold start
    model_volume.commit()

    if not stems_result:
        raise RuntimeError("Demucs produced no output stems")

    expected = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    found = set(stems_result.keys())
    print(f"Stems found: {found}")
    if not found.intersection(expected):
        print(f"WARNING: None of the expected stems {expected} were found. Got: {found}")

    return stems_result


# ---------------------------------------------------------------------------
# Local entrypoint for testing: modal run modal_separator.py
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Quick test: separate a small audio file via Modal GPU."""
    import sys
    from pathlib import Path

    # Use a test file if provided, otherwise look for a default
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    else:
        # Look for any audio file in uploads/
        uploads = Path(__file__).parent.parent / "uploads"
        test_files = list(uploads.glob("*.mp3")) + list(uploads.glob("*.wav"))
        if not test_files:
            print("No test audio files found. Provide a path as argument or put an MP3 in uploads/")
            print("Usage: modal run modal_separator.py -- /path/to/song.mp3")
            return
        test_file = test_files[0]

    print(f"Testing Modal separation with: {test_file}")
    audio_bytes = test_file.read_bytes()
    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"Input size: {size_mb:.1f} MB")

    stems = separate_stems_gpu.remote(audio_bytes, filename=test_file.name)

    print(f"\nReceived {len(stems)} stems:")
    for name, data in stems.items():
        print(f"  {name}: {len(data) / (1024*1024):.1f} MB")

    # Save stems locally for verification
    out_dir = Path(__file__).parent.parent / "outputs" / "modal_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in stems.items():
        out_path = out_dir / f"{name}.mp3"
        out_path.write_bytes(data)
        print(f"  Saved: {out_path}")

    print(f"\nAll stems saved to {out_dir}")
