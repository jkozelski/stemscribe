"""
Modal cloud GPU stem separation for StemScriber.

Deploys a serverless GPU function that runs BS-RoFormer-SW 6-stem separation
on Modal's infrastructure, with an optional second pass using UVR-MDX-NET
Karaoke 2 to split vocals into lead and backing.

Model: BS-Rofo-SW-Fixed by jarredou (HuggingFace)
  - 6 stems: vocals, guitar, bass, drums, piano, other
  - SDR ~12.9 (vs ~10 for htdemucs_6s)
  - Requires ~8-12 GB VRAM (fits on A10G 24GB)

Vocal Split Model: UVR_MDXNET_KARA_2.onnx (auto-downloaded on first use)
  - Takes pre-extracted vocals -> lead vocals + backing vocals
  - Two-pass approach: BS-RoFormer vocals -> KARA_2 split
  - Adds ~30-60s processing time

Deploy:  cd ~/stemscribe/backend && ../venv311/bin/python -m modal deploy modal_separator.py
Test:    cd ~/stemscribe/backend && ../venv311/bin/python -m modal run modal_separator.py
"""

import modal

# ---------------------------------------------------------------------------
# Modal Image — pre-bake all heavy dependencies so cold starts are fast
# ---------------------------------------------------------------------------

stemscribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "audio-separator[gpu]>=0.30",
        "torch>=2.3,<2.5",
        "torchaudio>=2.3,<2.5",
        "numpy>=2",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# Volume for caching model weights (~2 GB for BS-RoFormer-SW + ~150 MB for KARA_2)
model_volume = modal.Volume.from_name("stemscribe-model-cache", create_if_missing=True)

# Model filenames (centralized constants)
ROFORMER_MODEL = "BS-Roformer-SW.ckpt"
KARA_MODEL = "UVR_MDXNET_KARA_2.onnx"

app = modal.App("stemscribe-separator", image=stemscribe_image)

# ---------------------------------------------------------------------------
# GPU Function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/model-cache": model_volume},
    memory=16384,
)
@modal.concurrent(max_inputs=5)
def separate_stems_gpu(audio_bytes: bytes, filename: str = "input.mp3", split_vocals: bool = True) -> dict:
    """
    Run BS-RoFormer-SW 6-stem separation on a cloud GPU, with optional
    second-pass vocal splitting into lead and backing vocals.

    Args:
        audio_bytes: Raw audio file bytes (MP3, WAV, or FLAC)
        filename: Original filename (used to determine format)
        split_vocals: If True, run a second pass with UVR_MDXNET_KARA_2
                      to split vocals into vocals_lead and vocals_backing.
                      The original "vocals" key is kept for backward compat.

    Returns:
        Dict mapping stem name to MP3 bytes. With split_vocals=True:
        {"vocals": b"...", "vocals_lead": b"...", "vocals_backing": b"...",
         "guitar": b"...", "bass": b"...", "drums": b"...",
         "piano": b"...", "other": b"..."}

        With split_vocals=False (original 6-stem output):
        {"vocals": b"...", "guitar": b"...", "bass": b"...",
         "drums": b"...", "piano": b"...", "other": b"..."}
    """
    import os
    import re
    import subprocess
    import tempfile
    from pathlib import Path

    from audio_separator.separator import Separator

    # Write input audio to a temp file
    work_dir = Path(tempfile.mkdtemp())
    input_path = work_dir / filename
    input_path.write_bytes(audio_bytes)

    output_dir = work_dir / "stems"
    output_dir.mkdir()

    print(f"Separating {filename} ({len(audio_bytes) / (1024*1024):.1f} MB) with BS-RoFormer-SW...")

    # Initialize separator with BS-RoFormer-SW 6-stem model
    separator = Separator(
        output_dir=str(output_dir),
        model_file_dir="/model-cache/models",
        output_format="WAV",
        sample_rate=44100,
        use_autocast=True,
    )
    separator.load_model(model_filename=ROFORMER_MODEL)

    # Run separation
    output_files = separator.separate(str(input_path))

    print(f"Pass 1 complete. {len(output_files)} stem files produced.")

    # Convert stems to MP3 and collect results
    stems_result = {}
    vocals_wav_path = None  # Track the vocals WAV for second pass

    for fpath in output_files:
        p = Path(fpath)
        # Resolve relative paths against output_dir
        if not p.is_absolute():
            p = output_dir / p
        if not p.exists():
            # Try the work_dir as fallback
            p2 = work_dir / Path(fpath).name
            if p2.exists():
                p = p2
            else:
                print(f"  WARNING: stem file not found: {fpath}")
                continue

        # Extract stem name from patterns like:
        #   "Deacon_Blues_(bass)_BS-Roformer-SW.wav" -> "bass"
        #   "song_(Vocals).wav" -> "vocals"
        fname = p.stem
        m = re.search(r'\((\w+)\)', fname)
        if m:
            stem_name = m.group(1).lower()
        else:
            stem_name = fname.lower()

        # Keep a reference to the vocals WAV before converting to MP3
        if stem_name == "vocals":
            vocals_wav_path = p

        # Convert WAV to MP3 320kbps for transfer efficiency
        mp3_path = p.with_suffix(".mp3")
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(p),
                "-codec:a", "libmp3lame", "-b:a", "320k",
                str(mp3_path),
            ],
            capture_output=True,
            timeout=120,
        )

        if mp3_path.exists():
            stems_result[stem_name] = mp3_path.read_bytes()
        else:
            # Fallback: send WAV if MP3 conversion fails
            stems_result[stem_name] = p.read_bytes()

        size_mb = len(stems_result[stem_name]) / (1024 * 1024)
        print(f"  Stem '{stem_name}': {size_mb:.1f} MB")

    if not stems_result:
        raise RuntimeError("BS-RoFormer produced no output stems")

    expected = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    found = set(stems_result.keys())
    print(f"Stems found: {found}")
    if not found.intersection(expected):
        print(f"WARNING: None of the expected stems {expected} were found. Got: {found}")

    # ------------------------------------------------------------------
    # Pass 2: Vocal split (lead / backing) using UVR-MDX-NET Karaoke 2
    # ------------------------------------------------------------------
    if split_vocals and vocals_wav_path and vocals_wav_path.exists():
        print("\nPass 2: Splitting vocals into lead/backing with UVR_MDXNET_KARA_2...")

        vocal_split_dir = work_dir / "vocal_split"
        vocal_split_dir.mkdir()

        try:
            kara_separator = Separator(
                output_dir=str(vocal_split_dir),
                model_file_dir="/model-cache/models",
                output_format="WAV",
                sample_rate=44100,
                use_autocast=True,
            )
            kara_separator.load_model(model_filename=KARA_MODEL)

            vocal_outputs = kara_separator.separate(str(vocals_wav_path))
            print(f"  Vocal split produced {len(vocal_outputs)} files.")

            # KARA_2 is an MDX model trained for "Vocals vs Instrumental".
            # When fed pre-extracted vocals:
            #   - Primary output (Vocals): lead vocals (the "vocal" part)
            #   - Secondary output (Instrumental): backing vocals (the "instrumental" of vocals)
            # We detect by filename pattern: "(Vocals)" vs "(Instrumental)"
            lead_path = None
            backing_path = None

            for vf in vocal_outputs:
                vp = Path(vf)
                if not vp.is_absolute():
                    vp = vocal_split_dir / vp
                if not vp.exists():
                    vp2 = work_dir / Path(vf).name
                    if vp2.exists():
                        vp = vp2
                    else:
                        print(f"  WARNING: vocal split file not found: {vf}")
                        continue

                vname = vp.stem.lower()
                if "instrumental" in vname:
                    backing_path = vp
                    print(f"  -> Backing vocals: {vp.name}")
                elif "vocals" in vname:
                    lead_path = vp
                    print(f"  -> Lead vocals: {vp.name}")
                else:
                    # Fallback: first file = primary = lead, second = backing
                    if lead_path is None:
                        lead_path = vp
                        print(f"  -> Lead vocals (by position): {vp.name}")
                    else:
                        backing_path = vp
                        print(f"  -> Backing vocals (by position): {vp.name}")

            # Convert and add to results
            for label, wav_path in [("vocals_lead", lead_path), ("vocals_backing", backing_path)]:
                if wav_path and wav_path.exists():
                    mp3_out = wav_path.with_suffix(".mp3")
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(wav_path),
                            "-codec:a", "libmp3lame", "-b:a", "320k",
                            str(mp3_out),
                        ],
                        capture_output=True,
                        timeout=120,
                    )
                    if mp3_out.exists():
                        stems_result[label] = mp3_out.read_bytes()
                    else:
                        stems_result[label] = wav_path.read_bytes()
                    size_mb = len(stems_result[label]) / (1024 * 1024)
                    print(f"  Stem '{label}': {size_mb:.1f} MB")

            if "vocals_lead" in stems_result and "vocals_backing" in stems_result:
                print("Vocal split successful: vocals_lead + vocals_backing added.")
            else:
                print("WARNING: Vocal split incomplete. Keeping original vocals only.")

        except Exception as e:
            print(f"WARNING: Vocal split failed ({e}). Keeping original vocals only.")
            import traceback
            traceback.print_exc()

    elif split_vocals:
        print("WARNING: No vocals stem found for second pass. Skipping vocal split.")

    # Commit volume so model weights persist for next cold start
    model_volume.commit()

    stem_summary = {k: f"{len(v) / (1024*1024):.1f} MB" for k, v in stems_result.items()}
    print(f"\nFinal stems ({len(stems_result)}): {stem_summary}")

    return stems_result


# ---------------------------------------------------------------------------
# Local entrypoint for testing: modal run modal_separator.py
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(file: str = "", no_vocal_split: bool = False):
    """Quick test: separate a small audio file via Modal GPU.

    Args:
        file: Path to audio file. If empty, looks in uploads/ dir.
        no_vocal_split: If set, skip the lead/backing vocal split (6 stems only).
    """
    from pathlib import Path

    if file:
        test_file = Path(file)
    else:
        # Look for any audio file in uploads/
        uploads = Path(__file__).parent.parent / "uploads"
        test_files = list(uploads.glob("*.mp3")) + list(uploads.glob("*.wav"))
        if not test_files:
            print("No test audio files found.")
            print("Usage: modal run modal_separator.py --file /path/to/song.mp3")
            return
        test_file = test_files[0]

    split_vocals = not no_vocal_split
    print(f"Testing Modal separation with: {test_file}")
    print(f"Vocal split: {'ON' if split_vocals else 'OFF'}")
    audio_bytes = test_file.read_bytes()
    size_mb = len(audio_bytes) / (1024 * 1024)
    print(f"Input size: {size_mb:.1f} MB")

    stems = separate_stems_gpu.remote(audio_bytes, filename=test_file.name, split_vocals=split_vocals)

    print(f"\nReceived {len(stems)} stems:")
    for name, data in stems.items():
        print(f"  {name}: {len(data) / (1024*1024):.1f} MB")

    # Validate expected stems
    expected_base = {"vocals", "drums", "bass", "guitar", "piano", "other"}
    expected_vocal_split = {"vocals_lead", "vocals_backing"}
    found = set(stems.keys())

    missing_base = expected_base - found
    if missing_base:
        print(f"\nWARNING: Missing base stems: {missing_base}")

    if split_vocals:
        missing_vocal = expected_vocal_split - found
        if missing_vocal:
            print(f"WARNING: Missing vocal split stems: {missing_vocal}")
        else:
            print("\nVocal split verified: vocals_lead + vocals_backing present.")

        expected_total = 8  # 6 base + vocals_lead + vocals_backing
    else:
        expected_total = 6

    if len(stems) >= expected_total:
        print(f"PASS: Got {len(stems)} stems (expected >= {expected_total})")
    else:
        print(f"WARN: Got {len(stems)} stems (expected >= {expected_total})")

    # Save stems locally for verification
    out_dir = Path(__file__).parent.parent / "outputs" / "modal_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in stems.items():
        out_path = out_dir / f"{name}.mp3"
        out_path.write_bytes(data)
        print(f"  Saved: {out_path}")

    print(f"\nAll stems saved to {out_dir}")
