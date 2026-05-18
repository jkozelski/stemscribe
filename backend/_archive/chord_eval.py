"""
Chord detection evaluation — compare our detector against Songsterr ground truth.

Usage:
    python chord_eval.py <audio_path> <songsterr_song_id>

Example:
    python chord_eval.py ../uploads/a0db058f/Jimi_Hendrix_-_Little_wing_-_HQ.wav 5586
"""

import sys
import json
import logging
import requests
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SONGSTERR_API = "https://www.songsterr.com/api"
SONGSTERR_CDN = "https://dqsljvtekg760.cloudfront.net"

# Enharmonic equivalences for comparison
ENHARMONIC = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#", "Ab": "G#",
    "Bb": "A#", "Cb": "B",
    "C#": "Db", "D#": "Eb", "E#": "F", "F#": "Gb", "G#": "Ab",
    "A#": "Bb", "B#": "C",
}


def normalize_chord(name: str) -> str:
    """Normalize chord name for comparison (strip quality variants)."""
    if not name or name == "N":
        return "N"
    # Extract root
    root = name[0]
    rest = name[1:]
    if rest and rest[0] in ("#", "b"):
        root += rest[0]
        rest = rest[1:]
    # Strip bass note
    if "/" in rest:
        rest = rest.split("/")[0]
    # Normalize minor variants
    rest = rest.replace("min", "m").replace("mi", "m")
    # Strip extensions for basic comparison (7, maj7, etc.)
    basic_rest = rest
    for ext in ["add9", "sus4", "sus2", "maj7", "mMaj7", "dim7", "hdim7",
                "min7", "m7", "7", "6", "9", "aug", "dim"]:
        if basic_rest == ext:
            break
    return root + rest


def root_only(name: str) -> str:
    """Extract just the root note from a chord name."""
    if not name or name == "N":
        return "N"
    root = name[0]
    rest = name[1:]
    if rest and rest[0] in ("#", "b"):
        root += rest[0]
    return root


def roots_match(a: str, b: str) -> bool:
    """Check if two roots match (including enharmonic equivalents)."""
    if a == b:
        return True
    return ENHARMONIC.get(a) == b or ENHARMONIC.get(b) == a


def quality_category(name: str) -> str:
    """Get basic quality category: major, minor, dom7, etc."""
    if not name or name == "N":
        return "N"
    root = name[0]
    rest = name[1:]
    if rest and rest[0] in ("#", "b"):
        rest = rest[1:]
    if "/" in rest:
        rest = rest.split("/")[0]
    if rest in ("m", "min", "m7", "min7", "m6", "min6", "mMaj7"):
        return "minor"
    if rest in ("dim", "dim7", "hdim7"):
        return "dim"
    if rest in ("aug",):
        return "aug"
    if rest in ("sus2", "sus4"):
        return "sus"
    return "major"


def fetch_songsterr_chords(song_id: int) -> list[dict]:
    """Fetch chord events from Songsterr for a given song ID."""
    meta = requests.get(f"{SONGSTERR_API}/meta/{song_id}", timeout=10).json()
    revision_id = meta.get("revisionId")
    image_hash = meta.get("image")
    tracks = meta.get("tracks", [])

    if not revision_id or not image_hash:
        raise ValueError(f"No revision data for song {song_id}")

    # Fetch all tracks to find chord annotations
    all_tracks = []
    for idx in range(len(tracks)):
        url = f"{SONGSTERR_CDN}/{song_id}/{revision_id}/{image_hash}/{idx}.json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        all_tracks.append(resp.json())

    # Get tempo
    tempo = 120
    if all_tracks:
        automations = all_tracks[0].get("automations", {})
        tempo_list = automations.get("tempo", [])
        if tempo_list:
            tempo = tempo_list[0].get("bpm", 120)

    # Extract chords with timing (from beat.chord.text annotations)
    chord_events = []
    max_measures = max(len(t.get("measures", [])) for t in all_tracks)

    current_time = 0.0
    current_bpm = tempo

    # Get tempo changes
    tempo_changes = []
    if all_tracks:
        for tc in all_tracks[0].get("automations", {}).get("tempo", []):
            tempo_changes.append(tc)

    for mi in range(max_measures):
        sig = [4, 4]
        measure_chords = []

        for tdata in all_tracks:
            measures = tdata.get("measures", [])
            if mi >= len(measures):
                continue
            m = measures[mi]
            s = m.get("signature")
            if s and len(s) == 2:
                sig = s
            for v in m.get("voices", []):
                for b in v.get("beats", []):
                    chord_data = b.get("chord")
                    if chord_data and chord_data.get("text"):
                        chord_name = chord_data["text"].strip()
                        if chord_name and (not measure_chords or measure_chords[-1] != chord_name):
                            measure_chords.append(chord_name)

        # Update tempo
        for tc in tempo_changes:
            if tc.get("measure") == mi:
                current_bpm = tc.get("bpm", current_bpm)

        # Calculate measure duration
        beats_in_measure = sig[0] * (4.0 / sig[1]) if sig[1] > 0 else 4
        measure_duration = (beats_in_measure * 60.0) / current_bpm

        # Space chords evenly in measure
        for ci, ch in enumerate(measure_chords):
            offset = (ci / max(len(measure_chords), 1)) * measure_duration
            chord_events.append({
                "time": round(current_time + offset, 2),
                "chord": ch,
                "measure": mi + 1,
            })

        current_time += measure_duration

    return chord_events, meta.get("title", ""), meta.get("artist", ""), tempo


def evaluate(audio_path: str, song_id: int, detector_version: str = "v8"):
    """Run chord detection and compare against Songsterr ground truth."""

    # 1. Fetch Songsterr ground truth
    print(f"\nFetching Songsterr chords for song {song_id}...")
    truth_events, title, artist, tempo = fetch_songsterr_chords(song_id)
    print(f"Song: {artist} - {title} (tempo: {tempo} BPM)")
    print(f"Songsterr: {len(truth_events)} chord events")

    truth_unique = []
    seen = set()
    for e in truth_events:
        if e["chord"] not in seen:
            seen.add(e["chord"])
            truth_unique.append(e["chord"])
    print(f"Unique chords (Songsterr): {truth_unique}")

    # 2. Run our detector
    print(f"\nRunning {detector_version} chord detector on {audio_path}...")
    if detector_version == "v8":
        from chord_detector_v8 import ChordDetector
    else:
        from chord_detector import ChordDetector

    detector = ChordDetector()
    result = detector.detect(audio_path)
    print(f"Detected: {len(result.chords)} chord events, key: {result.key}")

    det_unique = []
    seen2 = set()
    for c in result.chords:
        if c.chord not in seen2:
            seen2.add(c.chord)
            det_unique.append(c.chord)
    print(f"Unique chords (detector): {det_unique}")

    # 3. Build time-aligned comparison
    # For each Songsterr chord event, find what our detector says at that time
    print(f"\n{'='*80}")
    print(f"{'TIME':>7s}  {'SONGSTERR':<15s}  {'DETECTED':<15s}  {'ROOT':>5s}  {'QUAL':>5s}  MATCH")
    print(f"{'='*80}")

    total = 0
    root_correct = 0
    quality_correct = 0
    exact_correct = 0

    for te in truth_events:
        t = te["time"]
        truth_chord = te["chord"]

        # Find detector chord active at this time
        det_chord = "N"
        for c in result.chords:
            if c.time <= t + 0.5 and c.time + c.duration >= t - 0.5:
                det_chord = c.chord
                break
            elif c.time <= t:
                det_chord = c.chord

        total += 1
        truth_root = root_only(truth_chord)
        det_root = root_only(det_chord)
        r_match = roots_match(truth_root, det_root)
        q_match = quality_category(truth_chord) == quality_category(det_chord)

        if r_match:
            root_correct += 1
        if r_match and q_match:
            quality_correct += 1
        if normalize_chord(truth_chord) == normalize_chord(det_chord):
            exact_correct += 1

        r_sym = "Y" if r_match else "X"
        q_sym = "Y" if q_match else "X"
        match_str = "EXACT" if normalize_chord(truth_chord) == normalize_chord(det_chord) else \
                    "root+qual" if r_match and q_match else \
                    "root" if r_match else "MISS"

        print(f"{t:7.1f}  {truth_chord:<15s}  {det_chord:<15s}  {r_sym:>5s}  {q_sym:>5s}  {match_str}")

    # 4. Summary
    print(f"\n{'='*80}")
    print(f"RESULTS: {artist} - {title}")
    print(f"{'='*80}")
    print(f"Total chord events:    {total}")
    print(f"Root correct:          {root_correct}/{total} ({100*root_correct/max(total,1):.1f}%)")
    print(f"Root+Quality correct:  {quality_correct}/{total} ({100*quality_correct/max(total,1):.1f}%)")
    print(f"Exact match:           {exact_correct}/{total} ({100*exact_correct/max(total,1):.1f}%)")
    print()
    print(f"Detector key: {result.key}")
    print(f"Songsterr chords: {truth_unique}")
    print(f"Detector chords:  {det_unique}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    audio_path = sys.argv[1]
    song_id = int(sys.argv[2])
    version = sys.argv[3] if len(sys.argv) > 3 else "v8"

    evaluate(audio_path, song_id, version)
