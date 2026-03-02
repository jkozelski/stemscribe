#!/usr/bin/env python3
"""
Guitar Dataset Preparation for StemScribe v3
=============================================
Downloads GuitarSet (Zenodo) + IDMT-SMT-Guitar, parses JAMS/XML annotations
into a common JSON manifest format, and creates stratified train/val/test splits.

Output: /workspace/guitar_v3_data/manifest.json

Manifest format (per entry):
{
    "audio": "/path/to/audio.wav",
    "source": "guitarset" | "idmt",
    "player": "00" | "01" | ...,
    "style": "comp" | "solo" | "fingerpicking" | "isolated" | "lick",
    "split": "train" | "val" | "test",
    "notes": [
        {"pitch": 64, "onset": 1.234, "offset": 1.890, "velocity": 0.75, "string": 0, "fret": 0},
        ...
    ],
    "max_polyphony": 3,
    "duration": 29.5
}

Run on RunPod:
    pip install torch librosa jams tqdm lxml
    python prepare_guitar_data.py
"""

import os
import sys
import glob
import json
import subprocess
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

def preflight():
    print("=" * 60)
    print("PRE-FLIGHT CHECKS — Guitar Data Preparation")
    print("=" * 60)
    errors = []

    # Disk space
    try:
        total, used, free = shutil.disk_usage("/workspace")
        free_gb = free / 1e9
        print(f"  [OK] Disk: {free_gb:.1f} GB free")
        if free_gb < 5:
            errors.append(f"Need at least 5 GB free, have {free_gb:.1f} GB")
    except Exception as e:
        print(f"  [WARN] Could not check disk: {e}")

    # Install deps
    deps = ["librosa", "jams", "tqdm", "soundfile"]
    missing = []
    for dep in deps:
        try:
            __import__(dep.replace("-", "_"))
        except ImportError:
            missing.append(dep)
    if missing:
        print(f"  [FIX] Installing: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
        print(f"  [OK] Installed {', '.join(missing)}")
    else:
        print(f"  [OK] Python deps present")

    # Check for lxml (needed for IDMT XML parsing)
    try:
        import xml.etree.ElementTree
        print("  [OK] XML parser available")
    except ImportError:
        errors.append("xml.etree not available")

    print("=" * 60)
    if errors:
        for e in errors:
            print(f"  FATAL: {e}")
        print("PRE-FLIGHT: FAILED")
        sys.exit(1)
    else:
        print("PRE-FLIGHT: All checks passed")
    print("=" * 60)
    print()


preflight()

import numpy as np
import librosa
import jams
from tqdm import tqdm

# ============================================================================
# CONFIG
# ============================================================================

WORKSPACE = Path("/workspace")
GUITARSET_DIR = WORKSPACE / "guitarset"
IDMT_DIR = WORKSPACE / "idmt_guitar"
OUTPUT_DIR = WORKSPACE / "guitar_v3_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Piano-matching mel params (for duration computation reference)
SAMPLE_RATE = 16000

# Guitar pitch range: E2 (40) to E6 (88) = 49 keys
GUITAR_MIN_MIDI = 40
GUITAR_MAX_MIDI = 88
NUM_KEYS = 49

# Standard guitar tuning
TUNING = [40, 45, 50, 55, 59, 64]

# GuitarSet player IDs for stratified splitting
GUITARSET_PLAYERS = ["00", "01", "02", "03", "04", "05"]


# ============================================================================
# DOWNLOAD GUITARSET
# ============================================================================

def download_guitarset():
    """Download GuitarSet from Zenodo (annotations + mono mic audio)."""
    wav_files = glob.glob(f"{GUITARSET_DIR}/**/*.wav", recursive=True)
    jams_files = glob.glob(f"{GUITARSET_DIR}/**/*.jams", recursive=True)

    if wav_files and jams_files:
        print(f"GuitarSet already present: {len(wav_files)} WAV, {len(jams_files)} JAMS")
        return

    GUITARSET_DIR.mkdir(parents=True, exist_ok=True)

    ANNOT_URL = "https://zenodo.org/records/3371780/files/annotation.zip?download=1"
    AUDIO_URL = "https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1"

    print("Downloading GuitarSet annotations (~39 MB)...")
    subprocess.run(
        ["wget", "-q", "--show-progress", ANNOT_URL, "-O", "/workspace/annotation.zip"],
        check=True,
    )
    subprocess.run(
        ["unzip", "-q", "-o", "/workspace/annotation.zip", "-d", str(GUITARSET_DIR)],
        check=True,
    )
    os.remove("/workspace/annotation.zip")

    print("Downloading GuitarSet audio (~657 MB)...")
    subprocess.run(
        ["wget", "-q", "--show-progress", AUDIO_URL, "-O", "/workspace/audio_mono-mic.zip"],
        check=True,
    )
    subprocess.run(
        ["unzip", "-q", "-o", "/workspace/audio_mono-mic.zip", "-d", str(GUITARSET_DIR)],
        check=True,
    )
    os.remove("/workspace/audio_mono-mic.zip")

    wav_files = glob.glob(f"{GUITARSET_DIR}/**/*.wav", recursive=True)
    jams_files = glob.glob(f"{GUITARSET_DIR}/**/*.jams", recursive=True)
    print(f"GuitarSet downloaded: {len(wav_files)} WAV, {len(jams_files)} JAMS\n")


# ============================================================================
# DOWNLOAD IDMT-SMT-GUITAR
# ============================================================================

def download_idmt():
    """Download IDMT-SMT-Guitar dataset."""
    existing = glob.glob(f"{IDMT_DIR}/**/*.wav", recursive=True)
    if len(existing) > 100:
        print(f"IDMT-SMT-Guitar already present: {len(existing)} WAV files")
        return

    IDMT_DIR.mkdir(parents=True, exist_ok=True)

    # IDMT-SMT-Guitar is available from multiple sources
    # Primary: Zenodo mirror or IDMT website
    IDMT_URL = "https://zenodo.org/records/7544110/files/IDMT-SMT-GUITAR_V2.zip?download=1"

    print("Downloading IDMT-SMT-Guitar (~1.5 GB)...")
    zip_path = "/workspace/idmt_guitar.zip"
    result = subprocess.run(
        ["wget", "-q", "--show-progress", IDMT_URL, "-O", zip_path],
        capture_output=False,
    )

    if result.returncode != 0:
        print("  WARN: IDMT download failed, trying alternate URL...")
        # Alternate URL
        ALT_URL = "https://zenodo.org/records/7544110/files/IDMT-SMT-GUITAR_V2.zip?download=1"
        result = subprocess.run(
            ["wget", "-q", "--show-progress", ALT_URL, "-O", zip_path],
            capture_output=False,
        )
        if result.returncode != 0:
            print("  WARN: IDMT download failed. Continuing with GuitarSet only.")
            return

    subprocess.run(["unzip", "-q", "-o", zip_path, "-d", str(IDMT_DIR)], check=True)
    if os.path.exists(zip_path):
        os.remove(zip_path)

    existing = glob.glob(f"{IDMT_DIR}/**/*.wav", recursive=True)
    print(f"IDMT-SMT-Guitar downloaded: {len(existing)} WAV files\n")


# ============================================================================
# PARSE GUITARSET JAMS
# ============================================================================

def extract_track_id(wav_name):
    """Extract the canonical track ID from a GuitarSet wav filename."""
    stem = wav_name
    for suffix in ["_mic", "_mix", "_hex_cln", "_hex"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for prefix in [
        "audio_mono-mic_",
        "audio_mono-mic-",
        "audio_mono-pickup_mix_",
        "audio_mono-pickup_mix-",
        "audio_hex-pickup_original_",
        "audio_hex-pickup_original-",
        "audio_hex-pickup_debleeded_",
        "audio_hex-pickup_debleeded-",
    ]:
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return stem


def extract_player_id(track_id):
    """Extract player ID from GuitarSet track ID (format: XX_STYLE_...)."""
    parts = track_id.split("_")
    if parts and parts[0].isdigit() and len(parts[0]) == 2:
        return parts[0]
    return "unknown"


def extract_style(track_id):
    """Extract playing style from GuitarSet track ID."""
    track_lower = track_id.lower()
    if "comp" in track_lower:
        return "comp"
    elif "solo" in track_lower:
        return "solo"
    elif "finger" in track_lower:
        return "fingerpicking"
    return "mixed"


def parse_guitarset_jams(jams_path, start_time=0.0, end_time=None):
    """
    Parse a GuitarSet JAMS file into a list of note dicts.

    GuitarSet JAMS contain per-string annotations in 'note_midi' namespace.
    Each annotation corresponds to a string (6 annotations total).

    Returns:
        List of {"pitch": int, "onset": float, "offset": float,
                 "velocity": float, "string": int, "fret": int}
    """
    jam = jams.load(str(jams_path))
    notes = []

    for string_idx, annot in enumerate(jam.annotations):
        if annot.namespace != "note_midi":
            continue
        # GuitarSet has 6 note_midi annotations, one per string
        actual_string = min(string_idx, 5)

        for obs in annot.data:
            onset = obs.time
            offset = obs.time + obs.duration
            midi_note = int(round(obs.value))

            if end_time is not None and onset >= end_time:
                continue
            if offset <= start_time:
                continue

            # Clamp to guitar range
            if midi_note < GUITAR_MIN_MIDI or midi_note > GUITAR_MAX_MIDI:
                continue

            # Compute fret from string tuning
            if actual_string < len(TUNING):
                fret = midi_note - TUNING[actual_string]
                if fret < 0 or fret > 24:
                    # Note doesn't fit on this string, find best string
                    fret = -1
                    for s, open_note in enumerate(TUNING):
                        f = midi_note - open_note
                        if 0 <= f <= 24:
                            actual_string = s
                            fret = f
                            break
                    if fret < 0:
                        fret = 0
            else:
                fret = 0

            # GuitarSet doesn't have velocity, use default
            velocity = obs.confidence if hasattr(obs, "confidence") and obs.confidence else 0.8

            notes.append({
                "pitch": midi_note,
                "onset": round(onset, 4),
                "offset": round(offset, 4),
                "velocity": round(float(velocity), 3),
                "string": actual_string,
                "fret": fret,
            })

    notes.sort(key=lambda n: n["onset"])
    return notes


def compute_max_polyphony(notes):
    """Compute maximum number of simultaneously active notes."""
    if not notes:
        return 0
    events = []
    for n in notes:
        events.append((n["onset"], 1))
        events.append((n["offset"], -1))
    events.sort()
    max_poly = 0
    current = 0
    for _, delta in events:
        current += delta
        max_poly = max(max_poly, current)
    return max_poly


def process_guitarset():
    """Process all GuitarSet files into manifest entries."""
    print("Processing GuitarSet...")

    wav_files = sorted(Path(GUITARSET_DIR).rglob("*.wav"))
    jams_files = sorted(Path(GUITARSET_DIR).rglob("*.jams"))

    # Build JAMS lookup
    jams_lookup = {}
    for jp in jams_files:
        jams_lookup[jp.stem] = jp
        jams_lookup[jp.stem.lower()] = jp

    entries = []
    matched = 0
    unmatched = 0

    for wav_path in tqdm(wav_files, desc="  GuitarSet"):
        track_id = extract_track_id(wav_path.stem)
        jams_path = jams_lookup.get(track_id) or jams_lookup.get(track_id.lower())

        if jams_path is None:
            unmatched += 1
            continue

        matched += 1

        # Get audio duration
        try:
            duration = librosa.get_duration(path=str(wav_path))
        except Exception:
            duration = 30.0  # GuitarSet tracks are ~30s

        # Parse annotations
        notes = parse_guitarset_jams(jams_path)
        max_poly = compute_max_polyphony(notes)

        player = extract_player_id(track_id)
        style = extract_style(track_id)

        entries.append({
            "audio": str(wav_path),
            "source": "guitarset",
            "player": player,
            "style": style,
            "split": "",  # assigned later
            "notes": notes,
            "max_polyphony": max_poly,
            "duration": round(duration, 2),
        })

    print(f"  Matched: {matched}, Unmatched: {unmatched}")
    print(f"  Total notes: {sum(len(e['notes']) for e in entries)}")
    return entries


# ============================================================================
# PARSE IDMT-SMT-GUITAR XML
# ============================================================================

def parse_idmt_xml(xml_path):
    """
    Parse an IDMT-SMT-Guitar XML annotation file.

    IDMT uses custom XML format with onset/pitch information.
    Returns list of note dicts.
    """
    import xml.etree.ElementTree as ET

    try:
        tree = ET.parse(str(xml_path))
        root = tree.getroot()
    except Exception as e:
        return []

    notes = []

    # IDMT XML structure varies; try common formats
    # Format 1: <transcription><event>...</event></transcription>
    for event in root.iter("event"):
        onset_el = event.find("onsetSec")
        offset_el = event.find("offsetSec")
        pitch_el = event.find("pitch") or event.find("midiPitch")

        if onset_el is None or pitch_el is None:
            continue

        onset = float(onset_el.text)
        offset = float(offset_el.text) if offset_el is not None else onset + 0.5
        midi_note = int(round(float(pitch_el.text)))

        if midi_note < GUITAR_MIN_MIDI or midi_note > GUITAR_MAX_MIDI:
            continue

        # Find best string/fret
        fret = -1
        string = 0
        for s, open_note in enumerate(TUNING):
            f = midi_note - open_note
            if 0 <= f <= 24:
                string = s
                fret = f
                break
        if fret < 0:
            fret = 0

        notes.append({
            "pitch": midi_note,
            "onset": round(onset, 4),
            "offset": round(offset, 4),
            "velocity": 0.8,
            "string": string,
            "fret": fret,
        })

    # Format 2: look for <note> elements
    if not notes:
        for note_el in root.iter("note"):
            onset_el = note_el.find("onset") or note_el.find("onsetSec")
            pitch_el = note_el.find("pitch") or note_el.find("midiPitch")

            if onset_el is None or pitch_el is None:
                continue

            onset = float(onset_el.text)
            dur_el = note_el.find("duration") or note_el.find("durationSec")
            duration = float(dur_el.text) if dur_el is not None else 0.5
            midi_note = int(round(float(pitch_el.text)))

            if midi_note < GUITAR_MIN_MIDI or midi_note > GUITAR_MAX_MIDI:
                continue

            fret = -1
            string = 0
            for s, open_note in enumerate(TUNING):
                f = midi_note - open_note
                if 0 <= f <= 24:
                    string = s
                    fret = f
                    break
            if fret < 0:
                fret = 0

            notes.append({
                "pitch": midi_note,
                "onset": round(onset, 4),
                "offset": round(onset + duration, 4),
                "velocity": 0.8,
                "string": string,
                "fret": fret,
            })

    notes.sort(key=lambda n: n["onset"])
    return notes


def process_idmt():
    """Process IDMT-SMT-Guitar files into manifest entries."""
    if not IDMT_DIR.exists():
        print("IDMT-SMT-Guitar not found, skipping")
        return []

    print("Processing IDMT-SMT-Guitar...")

    wav_files = sorted(IDMT_DIR.rglob("*.wav"))
    xml_files = sorted(IDMT_DIR.rglob("*.xml"))

    if not wav_files:
        print("  No WAV files found")
        return []

    # Build XML lookup by stem
    xml_lookup = {}
    for xp in xml_files:
        xml_lookup[xp.stem] = xp
        xml_lookup[xp.stem.lower()] = xp

    entries = []
    matched = 0

    for wav_path in tqdm(wav_files, desc="  IDMT"):
        # Try to find matching XML annotation
        xml_path = xml_lookup.get(wav_path.stem) or xml_lookup.get(wav_path.stem.lower())

        if xml_path is None:
            # Try parent directory for grouped annotations
            parent_xml = wav_path.parent / (wav_path.parent.name + ".xml")
            if parent_xml.exists():
                xml_path = parent_xml

        if xml_path is None:
            continue

        matched += 1

        try:
            duration = librosa.get_duration(path=str(wav_path))
        except Exception:
            duration = 2.0

        notes = parse_idmt_xml(xml_path)
        max_poly = compute_max_polyphony(notes)

        # Classify IDMT samples by directory structure
        path_str = str(wav_path).lower()
        if "lick" in path_str or "phrase" in path_str:
            style = "lick"
        else:
            style = "isolated"

        entries.append({
            "audio": str(wav_path),
            "source": "idmt",
            "player": "idmt",
            "style": style,
            "split": "",
            "notes": notes,
            "max_polyphony": max_poly,
            "duration": round(duration, 2),
        })

    print(f"  Matched: {matched} files with annotations")
    print(f"  Total notes: {sum(len(e['notes']) for e in entries)}")
    return entries


# ============================================================================
# STRATIFIED SPLIT (60/20/20)
# ============================================================================

def assign_splits(entries):
    """
    Assign train/val/test splits stratified by player and style.

    GuitarSet: Split by player ID to avoid data leakage.
      - Players 00, 01, 02, 03 -> train (67%)
      - Player 04 -> val (17%)
      - Player 05 -> test (17%)

    IDMT: Random 60/20/20 split (no player info).
    """
    print("\nAssigning splits...")

    # GuitarSet: player-stratified
    train_players = {"00", "01", "02", "03"}
    val_players = {"04"}
    test_players = {"05"}

    # IDMT: shuffle and split
    np.random.seed(42)
    idmt_entries = [e for e in entries if e["source"] == "idmt"]
    np.random.shuffle(idmt_entries)
    n = len(idmt_entries)
    idmt_split_1 = int(n * 0.6)
    idmt_split_2 = int(n * 0.8)

    idmt_train = set(id(e) for e in idmt_entries[:idmt_split_1])
    idmt_val = set(id(e) for e in idmt_entries[idmt_split_1:idmt_split_2])

    counts = defaultdict(int)

    for entry in entries:
        if entry["source"] == "guitarset":
            player = entry["player"]
            if player in train_players:
                entry["split"] = "train"
            elif player in val_players:
                entry["split"] = "val"
            elif player in test_players:
                entry["split"] = "test"
            else:
                entry["split"] = "train"
        elif entry["source"] == "idmt":
            if id(entry) in idmt_train:
                entry["split"] = "train"
            elif id(entry) in idmt_val:
                entry["split"] = "val"
            else:
                entry["split"] = "test"
        else:
            entry["split"] = "train"

        counts[entry["split"]] += 1

    for split, count in sorted(counts.items()):
        print(f"  {split}: {count} entries")

    return entries


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("Guitar Data Preparation for StemScribe v3")
    print("=" * 60)
    print()

    # Download datasets
    download_guitarset()
    download_idmt()

    # Parse annotations
    guitarset_entries = process_guitarset()
    idmt_entries = process_idmt()

    all_entries = guitarset_entries + idmt_entries

    if not all_entries:
        print("ERROR: No data processed!")
        sys.exit(1)

    # Assign splits
    all_entries = assign_splits(all_entries)

    # Summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    total_notes = sum(len(e["notes"]) for e in all_entries)
    total_duration = sum(e["duration"] for e in all_entries)
    avg_polyphony = np.mean([e["max_polyphony"] for e in all_entries])

    print(f"Total entries:     {len(all_entries)}")
    print(f"  GuitarSet:       {len(guitarset_entries)}")
    print(f"  IDMT:            {len(idmt_entries)}")
    print(f"Total notes:       {total_notes}")
    print(f"Total duration:    {total_duration / 60:.1f} minutes")
    print(f"Avg max polyphony: {avg_polyphony:.1f}")

    # Pitch distribution
    all_pitches = [n["pitch"] for e in all_entries for n in e["notes"]]
    if all_pitches:
        print(f"Pitch range:       {min(all_pitches)} - {max(all_pitches)} "
              f"(target: {GUITAR_MIN_MIDI}-{GUITAR_MAX_MIDI})")

    # Style distribution
    style_counts = defaultdict(int)
    for e in all_entries:
        style_counts[e["style"]] += 1
    print(f"Styles:            {dict(style_counts)}")

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(all_entries, f, indent=2)

    print(f"\nManifest saved: {manifest_path}")
    print(f"File size: {manifest_path.stat().st_size / 1e6:.1f} MB")

    # Also save a lightweight version without notes (for quick loading)
    light_entries = []
    for e in all_entries:
        light_entries.append({
            "audio": e["audio"],
            "source": e["source"],
            "player": e["player"],
            "style": e["style"],
            "split": e["split"],
            "num_notes": len(e["notes"]),
            "max_polyphony": e["max_polyphony"],
            "duration": e["duration"],
        })

    light_path = OUTPUT_DIR / "manifest_light.json"
    with open(light_path, "w") as f:
        json.dump(light_entries, f, indent=2)

    print(f"Light manifest:    {light_path}")
    print(f"\n{'=' * 60}")
    print("Data preparation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
