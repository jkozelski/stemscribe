"""
Chord training data pipeline — builds labeled dataset from real songs.

For each song:
1. Scrape UG chords (ground truth chord names)
2. Fetch Songsterr timing data (beat-mapped chord positions)
3. Extract chroma features from audio
4. Align chords to audio frames → training pairs

Usage:
    python chord_training_pipeline.py
"""

import json
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Song catalog: filename_stem → (artist, title, ug_query, songsterr_query)
SONG_CATALOG = {
    "Aint_Wastin_Time_No_More": ("Allman Brothers Band", "Ain't Wastin Time No More", "allman brothers aint wastin time no more"),
    "Blue_Sky": ("Allman Brothers Band", "Blue Sky", "allman brothers blue sky"),
    "Crazy_Fingers_2025_Remaster": ("Grateful Dead", "Crazy Fingers", "grateful dead crazy fingers"),
    "Deacon_Blues": ("Steely Dan", "Deacon Blues", "steely dan deacon blues"),
    "Dirty_Work": ("Steely Dan", "Dirty Work", "steely dan dirty work"),
    "Doin_That_Rag_2013_Remaster": ("Grateful Dead", "Doin' That Rag", "grateful dead doin that rag"),
    "Farmhouse": ("Phish", "Farmhouse", "phish farmhouse"),
    "Foxology": ("Foxology", "Foxology", "foxology"),
    "Franklins_Tower_2013_Remaster": ("Grateful Dead", "Franklin's Tower", "grateful dead franklins tower"),
    "Grateful_Dead_-_The_Music_Never_Stopped_2025_Remas": ("Grateful Dead", "The Music Never Stopped", "grateful dead music never stopped"),
    "Guelah_Papyrus": ("Phish", "Guelah Papyrus", "phish guelah papyrus"),
    "Help_on_the_Way__Slipknot__Franklins_Tower_Live_Oc": ("Grateful Dead", "Help on the Way", "grateful dead help on the way"),
    "Help_on_the_Way__Slipknot_2013_Remaster": ("Grateful Dead", "Help on the Way", "grateful dead help on the way"),
    "Help_on_the_Way_2025_Remaster": ("Grateful Dead", "Help on the Way", "grateful dead help on the way"),
    "It_Must_Have_Been_the_Roses_Live": ("Grateful Dead", "It Must Have Been the Roses", "grateful dead it must have been the roses"),
    "Jimi_Hendrix_-_Little_wing_-_HQ": ("Jimi Hendrix", "Little Wing", "jimi hendrix little wing"),
    "Kid_Charlemagne": ("Steely Dan", "Kid Charlemagne", "steely dan kid charlemagne"),
    "King_Solomons_Marbles_2025_Remaster": ("Grateful Dead", "King Solomon's Marbles", "grateful dead king solomons marbles"),
    "Ramble_On_Remaster": ("Led Zeppelin", "Ramble On", "led zeppelin ramble on"),
    "Rick_Astley_-_Never_Gonna_Give_You_Up_Official_Vid": ("Rick Astley", "Never Gonna Give You Up", "rick astley never gonna give you up"),
    "Scarlet_Begonias_2013_Remaster": ("Grateful Dead", "Scarlet Begonias", "grateful dead scarlet begonias"),
    "Sign_In_Stranger": ("Steely Dan", "Sign In Stranger", "steely dan sign in stranger"),
    "Stash": ("Phish", "Stash", "phish stash"),
    "Tame_Impala_-_Elephant": ("Tame Impala", "Elephant", "tame impala elephant"),
    "Tennessee_Jed_Live_at_LOlympia_Paris_5372_2001_Rem": ("Grateful Dead", "Tennessee Jed", "grateful dead tennessee jed"),
    "The_Fez": ("Steely Dan", "The Fez", "steely dan the fez"),
    "The_Royal_Scam": ("Steely Dan", "The Royal Scam", "steely dan the royal scam"),
    "You_Enjoy_Myself": ("Phish", "You Enjoy Myself", "phish you enjoy myself"),
}

UPLOADS_DIR = Path(__file__).parent.parent / "uploads"
TRAINING_DIR = Path(__file__).parent / "training_data" / "chords"

# Standard chord vocabulary (simplified for training)
CHORD_VOCAB = [
    "N",  # No chord / silence
    # Major
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
    # Minor
    "Cm", "C#m", "Dm", "D#m", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "A#m", "Bm",
    # Dominant 7
    "C7", "C#7", "D7", "D#7", "E7", "F7", "F#7", "G7", "G#7", "A7", "A#7", "B7",
    # Major 7
    "Cmaj7", "C#maj7", "Dmaj7", "D#maj7", "Emaj7", "Fmaj7", "F#maj7", "Gmaj7", "G#maj7", "Amaj7", "A#maj7", "Bmaj7",
    # Minor 7
    "Cm7", "C#m7", "Dm7", "D#m7", "Em7", "Fm7", "F#m7", "Gm7", "G#m7", "Am7", "A#m7", "Bm7",
    # Sus4
    "Csus4", "C#sus4", "Dsus4", "D#sus4", "Esus4", "Fsus4", "F#sus4", "Gsus4", "G#sus4", "Asus4", "A#sus4", "Bsus4",
    # Dim
    "Cdim", "C#dim", "Ddim", "D#dim", "Edim", "Fdim", "F#dim", "Gdim", "G#dim", "Adim", "A#dim", "Bdim",
    # Aug
    "Caug", "C#aug", "Daug", "D#aug", "Eaug", "Faug", "F#aug", "Gaug", "G#aug", "Aaug", "A#aug", "Baug",
]

# Enharmonic map
ENHARMONIC = {"Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B"}


def normalize_to_vocab(chord_name: str) -> str:
    """Map any chord name to our training vocabulary."""
    if not chord_name or chord_name == "N":
        return "N"

    # Strip bass note (inversions)
    if "/" in chord_name:
        chord_name = chord_name.split("/")[0]

    # Extract root
    root = chord_name[0]
    rest = chord_name[1:]
    if rest and rest[0] in ("#", "b"):
        root += rest[0]
        rest = rest[1:]

    # Normalize enharmonics
    root = ENHARMONIC.get(root, root)

    # Map quality to vocab
    rest_lower = rest.lower()
    if rest_lower in ("", "5", "add9", "6", "sus2"):
        quality = ""  # Major
    elif rest_lower in ("m", "min", "min6", "m6"):
        quality = "m"
    elif rest_lower in ("7", "9", "11", "13"):
        quality = "7"
    elif rest_lower in ("maj7", "maj9", "7m", "c7m"):
        quality = "maj7"
    elif rest_lower in ("m7", "min7", "m9", "m11"):
        quality = "m7"
    elif rest_lower in ("mmaj7",):
        quality = "m7"  # Close enough
    elif rest_lower in ("sus4", "sus"):
        quality = "sus4"
    elif rest_lower in ("dim", "dim7", "hdim7", "ø"):
        quality = "dim"
    elif rest_lower in ("aug", "+"):
        quality = "aug"
    else:
        quality = ""  # Default to major

    candidate = root + quality
    if candidate in CHORD_VOCAB:
        return candidate
    return "N"


def find_audio_path(song_stem: str) -> str | None:
    """Find the audio file for a song."""
    for d in UPLOADS_DIR.iterdir():
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.stem == song_stem and f.suffix == ".wav":
                return str(f)
    return None


def scrape_ug_chords(query: str) -> list[dict] | None:
    """Get chord data from Ultimate Guitar."""
    try:
        from routes.ug import _scrape_ug
        result = _scrape_ug(query)
        return result
    except Exception as e:
        logger.warning(f"UG scrape failed for '{query}': {e}")
        return None


def fetch_songsterr_chords(artist: str, title: str) -> tuple[list[dict], int] | None:
    """Search Songsterr and get timed chord events."""
    import requests
    try:
        query = f"{artist} {title}"
        resp = requests.get(
            f"https://www.songsterr.com/api/songs?size=5&pattern={query}",
            timeout=10
        )
        if not resp.ok:
            return None

        songs = resp.json()
        if not songs:
            return None

        # Find best match with hasChords=True
        best = None
        for s in songs:
            if s.get("hasChords"):
                best = s
                break
        if not best:
            best = songs[0]

        song_id = best["songId"]

        # Fetch chord events using our existing eval code
        from chord_eval import fetch_songsterr_chords as fetch_ss
        events, _, _, tempo = fetch_ss(song_id)
        return events, tempo

    except Exception as e:
        logger.warning(f"Songsterr fetch failed for '{artist} - {title}': {e}")
        return None


def extract_chroma(audio_path: str, hop_length: int = 4096, sr: int = 22050) -> tuple[np.ndarray, np.ndarray]:
    """Extract CQT chroma features from audio."""
    import librosa
    y, sr_actual = librosa.load(audio_path, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr_actual, hop_length=hop_length)
    times = librosa.times_like(chroma, sr=sr_actual, hop_length=hop_length)
    return chroma, times


def build_labels_from_ug(ug_data: dict, audio_duration: float, n_frames: int, times: np.ndarray) -> np.ndarray | None:
    """Build frame-level chord labels from UG chord sheet.

    UG doesn't have timing, so we can't directly align.
    Returns None — UG is used for chord vocabulary validation only.
    """
    return None


def build_labels_from_songsterr(events: list[dict], n_frames: int, times: np.ndarray) -> np.ndarray:
    """Build frame-level chord labels from Songsterr timed events."""
    labels = np.zeros(n_frames, dtype=np.int32)  # Default to N (index 0)

    if not events:
        return labels

    # Sort by time
    events = sorted(events, key=lambda e: e["time"])

    for fi in range(n_frames):
        t = times[fi]
        # Find active chord at this time
        active = "N"
        for e in events:
            if e["time"] <= t + 0.3:
                active = e["chord"]
            else:
                break

        normalized = normalize_to_vocab(active)
        if normalized in CHORD_VOCAB:
            labels[fi] = CHORD_VOCAB.index(normalized)

    return labels


def process_song(song_stem: str, info: tuple, hop_length: int = 4096) -> dict | None:
    """Process a single song into training data."""
    artist, title, ug_query = info

    # Find audio
    audio_path = find_audio_path(song_stem)
    if not audio_path:
        logger.warning(f"No audio found for {song_stem}")
        return None

    logger.info(f"Processing: {artist} - {title}")

    # Extract chroma
    logger.info(f"  Extracting chroma from {audio_path}...")
    chroma, times = extract_chroma(audio_path, hop_length=hop_length)
    n_frames = chroma.shape[1]
    logger.info(f"  {n_frames} frames, {times[-1]:.1f}s duration")

    # Get Songsterr timing
    songsterr_result = fetch_songsterr_chords(artist, title)
    songsterr_events = []
    if songsterr_result:
        songsterr_events, tempo = songsterr_result
        logger.info(f"  Songsterr: {len(songsterr_events)} chord events")
    else:
        logger.warning(f"  No Songsterr data")

    # Get UG chords for validation
    ug_data = None
    try:
        ug_data = scrape_ug_chords(ug_query)
        if ug_data:
            logger.info(f"  UG: {len(ug_data.get('chords_used', []))} unique chords")
    except Exception as e:
        logger.warning(f"  UG scrape failed: {e}")

    # Build labels from Songsterr (has timing)
    if not songsterr_events:
        logger.warning(f"  Skipping {song_stem} — no timed chord data")
        return None

    labels = build_labels_from_songsterr(songsterr_events, n_frames, times)

    # Count non-N labels
    non_silent = np.sum(labels > 0)
    if non_silent < 10:
        logger.warning(f"  Only {non_silent} labeled frames — skipping")
        return None

    # Validate against UG chords
    ug_chords = set()
    if ug_data:
        for ch in ug_data.get("chords_used", []):
            norm = normalize_to_vocab(ch)
            if norm != "N":
                ug_chords.add(norm)

    ss_chords = set()
    for e in songsterr_events:
        norm = normalize_to_vocab(e["chord"])
        if norm != "N":
            ss_chords.add(norm)

    overlap = ug_chords & ss_chords if ug_chords else set()
    logger.info(f"  Labels: {non_silent}/{n_frames} frames labeled ({100*non_silent/n_frames:.0f}%)")
    logger.info(f"  Songsterr chords: {sorted(ss_chords)}")
    if ug_chords:
        logger.info(f"  UG chords: {sorted(ug_chords)}")
        logger.info(f"  Overlap: {sorted(overlap)} ({len(overlap)}/{len(ss_chords | ug_chords)})")

    return {
        "song": song_stem,
        "artist": artist,
        "title": title,
        "chroma": chroma,  # (12, n_frames)
        "times": times,
        "labels": labels,  # (n_frames,) int indices into CHORD_VOCAB
        "n_frames": n_frames,
        "songsterr_chords": sorted(ss_chords),
        "ug_chords": sorted(ug_chords) if ug_chords else [],
    }


def main():
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    hop_length = 4096  # ~0.19s per frame at 22050 Hz — finer than v8's 8192

    all_data = []
    skipped = []

    for song_stem, info in SONG_CATALOG.items():
        try:
            result = process_song(song_stem, info, hop_length=hop_length)
            if result:
                all_data.append(result)
            else:
                skipped.append(song_stem)
            time.sleep(1)  # Rate limit UG/Songsterr
        except Exception as e:
            logger.error(f"Failed on {song_stem}: {e}", exc_info=True)
            skipped.append(song_stem)

    # Save training data
    if all_data:
        total_frames = sum(d["n_frames"] for d in all_data)
        labeled_frames = sum(np.sum(d["labels"] > 0) for d in all_data)

        # Concatenate all data
        all_chroma = np.concatenate([d["chroma"] for d in all_data], axis=1)
        all_labels = np.concatenate([d["labels"] for d in all_data])

        output_path = TRAINING_DIR / "chord_training_data.npz"
        np.savez_compressed(
            output_path,
            chroma=all_chroma,
            labels=all_labels,
            vocab=np.array(CHORD_VOCAB),
        )

        # Save metadata
        meta = {
            "songs": [{
                "song": d["song"],
                "artist": d["artist"],
                "title": d["title"],
                "n_frames": d["n_frames"],
                "songsterr_chords": d["songsterr_chords"],
                "ug_chords": d["ug_chords"],
            } for d in all_data],
            "total_frames": total_frames,
            "labeled_frames": int(labeled_frames),
            "vocab_size": len(CHORD_VOCAB),
            "hop_length": hop_length,
            "sr": 22050,
            "skipped": skipped,
        }
        with open(TRAINING_DIR / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING DATA BUILT")
        logger.info(f"{'='*60}")
        logger.info(f"Songs processed: {len(all_data)}")
        logger.info(f"Songs skipped: {len(skipped)} ({skipped})")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"Labeled frames: {labeled_frames} ({100*labeled_frames/total_frames:.0f}%)")
        logger.info(f"Vocab size: {len(CHORD_VOCAB)}")
        logger.info(f"Saved to: {output_path}")
    else:
        logger.error("No training data generated!")


if __name__ == "__main__":
    main()
