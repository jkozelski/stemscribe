#!/usr/bin/env python3
"""
BTC Chord Model Stress Test — 10 Jazz/Steely Dan Songs
Runs BTC chord detection (v10) standalone and outputs results as JSON.
"""

import sys
import os
import json
import subprocess
import logging
import time
from pathlib import Path
from collections import Counter

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BACKEND_DIR = Path("/opt/stemscribe/backend")
AUDIO_DIR = Path("/tmp/btc_test_audio")
RESULTS_FILE = Path("/tmp/btc_test_results.json")

sys.path.insert(0, str(BACKEND_DIR))
os.chdir(str(BACKEND_DIR))

# Songs to test — YouTube search queries
SONGS = [
    {"name": "Blue Bossa", "artist": "Kenny Dorham", "query": "Blue Bossa Kenny Dorham jazz standard"},
    {"name": "Dirty Work", "artist": "Steely Dan", "query": "Steely Dan Dirty Work"},
    {"name": "Deacon Blues", "artist": "Steely Dan", "query": "Steely Dan Deacon Blues"},
    {"name": "Peg", "artist": "Steely Dan", "query": "Steely Dan Peg"},
    {"name": "Autumn Leaves", "artist": "Jazz Standard", "query": "Autumn Leaves jazz standard Cannonball Adderley"},
    {"name": "All The Things You Are", "artist": "Jazz Standard", "query": "All The Things You Are Ella Fitzgerald jazz"},
    {"name": "Take Five", "artist": "Dave Brubeck", "query": "Dave Brubeck Take Five"},
    {"name": "So What", "artist": "Miles Davis", "query": "Miles Davis So What"},
    {"name": "Girl From Ipanema", "artist": "Stan Getz", "query": "Girl From Ipanema Stan Getz Joao Gilberto"},
    {"name": "Black Cow", "artist": "Steely Dan", "query": "Steely Dan Black Cow"},
]

# Known correct chord progressions (main/key changes)
EXPECTED = {
    "Blue Bossa": {
        "key": "Cm",
        "main_chords": ["Cm7", "Fm7", "Dm7b5", "G7", "Ebmaj7", "Ab7", "Db7"],
        "expected_roots": ["C", "F", "D", "G", "Eb", "Ab", "Db"],
        "expected_qualities": ["m7", "m7", "m7b5", "7", "maj7", "7", "7"],
        "notes": "16-bar jazz standard in Cm, brief modulation to Db major"
    },
    "Dirty Work": {
        "key": "E",
        "main_chords": ["E", "Emaj7", "A", "Am", "B7", "C#m", "G#m", "F#m7", "Amaj7"],
        "expected_roots": ["E", "A", "B", "C#", "G#", "F#"],
        "expected_qualities": ["maj", "maj7", "maj", "m", "7", "m", "m", "m7", "maj7"],
        "notes": "Lots of maj7s and 7th chords, classic Fagen/Becker voicings"
    },
    "Deacon Blues": {
        "key": "Eb/C",
        "main_chords": ["Cmaj7", "Dm7", "Em7", "Fmaj7", "G7", "Am7", "Bbmaj7", "Eb", "Ab"],
        "expected_roots": ["C", "D", "E", "F", "G", "A", "Bb", "Eb", "Ab"],
        "expected_qualities": ["maj7", "m7", "m7", "maj7", "7", "m7", "maj7", "maj", "maj"],
        "notes": "Complex jazz-rock, multiple key areas, chromatic movement"
    },
    "Peg": {
        "key": "G",
        "main_chords": ["Gmaj7", "F#7", "Fmaj7", "E7", "Am7", "Cmaj7", "Bm7", "Bb7"],
        "expected_roots": ["G", "F#", "F", "E", "A", "C", "B", "Bb"],
        "expected_qualities": ["maj7", "7", "maj7", "7", "m7", "maj7", "m7", "7"],
        "notes": "Chromatic descending line G-F#-F-E, altered dominants"
    },
    "Autumn Leaves": {
        "key": "Gm/Bb",
        "main_chords": ["Cm7", "F7", "Bbmaj7", "Ebmaj7", "Am7b5", "D7", "Gm"],
        "expected_roots": ["C", "F", "Bb", "Eb", "A", "D", "G"],
        "expected_qualities": ["m7", "7", "maj7", "maj7", "m7b5", "7", "m"],
        "notes": "Classic ii-V-I in Bb major then ii-V-i in Gm"
    },
    "All The Things You Are": {
        "key": "Ab/multi",
        "main_chords": ["Fm7", "Bbm7", "Eb7", "Abmaj7", "Dbmaj7", "Dm7", "G7", "Cmaj7", "Cm7", "F7", "Bbmaj7"],
        "expected_roots": ["F", "Bb", "Eb", "Ab", "Db", "D", "G", "C"],
        "expected_qualities": ["m7", "m7", "7", "maj7", "maj7", "m7", "7", "maj7"],
        "notes": "Moves through Ab, C, Eb, G key centers"
    },
    "Take Five": {
        "key": "Ebm",
        "main_chords": ["Ebm", "Bbm", "Ebm7", "Bbm7", "Abm", "Bbm", "Cbmaj7"],
        "expected_roots": ["Eb", "Bb", "Ab", "Cb"],
        "expected_qualities": ["m", "m", "m7", "m7", "m", "m", "maj7"],
        "notes": "5/4 time, mostly Ebm and Bbm alternation, bridge to Cbmaj7"
    },
    "So What": {
        "key": "Dm",
        "main_chords": ["Dm7", "Ebm7"],
        "expected_roots": ["D", "Eb"],
        "expected_qualities": ["m7", "m7"],
        "notes": "Modal jazz — 16 bars Dm7, 8 bars Ebm7, 8 bars Dm7"
    },
    "Girl From Ipanema": {
        "key": "F",
        "main_chords": ["Fmaj7", "G7", "Gm7", "Gb7", "Gbmaj7", "B9", "F#m9", "D9", "Eb9", "Am7", "D7"],
        "expected_roots": ["F", "G", "Gb", "B", "F#", "D", "Eb", "A"],
        "expected_qualities": ["maj7", "7", "m7", "7", "maj7", "9", "m9", "9", "9", "m7", "7"],
        "notes": "Bossa nova, Fmaj7-G7 verse, chromatic bridge modulation"
    },
    "Black Cow": {
        "key": "Db/Eb",
        "main_chords": ["Dbmaj7", "Eb9", "Fm7", "Cm7", "Bbm7", "Ab7", "Gb7"],
        "expected_roots": ["Db", "Eb", "F", "C", "Bb", "Ab", "Gb"],
        "expected_qualities": ["maj7", "9", "m7", "m7", "m7", "7", "7"],
        "notes": "Funky jazz, 9ths, 13ths, complex extensions"
    },
}


def download_song(song: dict) -> str | None:
    """Download a song via yt-dlp. Returns path to WAV file."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = song["name"].replace(" ", "_").replace("'", "")
    output_path = AUDIO_DIR / f"{safe_name}.wav"
    
    if output_path.exists():
        logger.info(f"Already downloaded: {output_path}")
        return str(output_path)
    
    logger.info(f"Downloading: {song['name']} ({song['artist']})")
    cookies_file = Path("/opt/stemscribe/youtube-cookies.txt")
    cmd = [
        "yt-dlp",
        f"ytsearch1:{song['query']}",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", str(AUDIO_DIR / f"{safe_name}.%(ext)s"),
        "--no-playlist",
        "--max-downloads", "1",
        "--cookies", str(cookies_file),
        "--extractor-args", "youtube:player_client=web",
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.error(f"yt-dlp failed for {song['name']}: {result.stderr[:200]}")
            return None
        
        # yt-dlp may create the file with slightly different naming
        wavs = list(AUDIO_DIR.glob(f"{safe_name}*wav*")) + list(AUDIO_DIR.glob(f"{safe_name}*.wav"))
        if wavs:
            # Rename to expected path if needed
            actual = wavs[0]
            if actual != output_path:
                actual.rename(output_path)
            logger.info(f"Downloaded: {output_path}")
            return str(output_path)
        else:
            logger.error(f"No WAV file found after download for {song['name']}")
            return None
    except Exception as e:
        logger.error(f"Download error for {song['name']}: {e}")
        return None


def run_btc_detection(audio_path: str, artist: str = None, title: str = None) -> dict:
    """Run BTC v10 chord detection on an audio file."""
    from chord_detector_v10 import ChordDetector
    
    detector = ChordDetector(min_duration=0.3)
    start = time.time()
    result = detector.detect(audio_path, artist=artist, title=title)
    elapsed = time.time() - start
    
    # Extract unique chords in order of appearance
    unique_chords = []
    seen = set()
    for c in result.chords:
        if c.chord not in seen and c.chord != 'N':
            seen.add(c.chord)
            unique_chords.append(c.chord)
    
    # Get chord durations for weighting
    chord_durations = Counter()
    for c in result.chords:
        if c.chord != 'N':
            chord_durations[c.chord] += c.duration
    
    # Top chords by duration
    top_chords = [ch for ch, _ in chord_durations.most_common(15)]
    
    # All chord events
    events = [
        {"time": round(c.time, 2), "duration": round(c.duration, 2),
         "chord": c.chord, "root": c.root, "quality": c.quality,
         "confidence": round(c.confidence, 3)}
        for c in result.chords if c.chord != 'N'
    ]
    
    return {
        "key": result.key,
        "unique_chords": unique_chords,
        "top_chords_by_duration": top_chords,
        "total_chord_events": len(events),
        "detection_time_seconds": round(elapsed, 1),
        "tuning_info": result.tuning_info,
        "events": events,
    }


def compare_results(detected: dict, expected: dict) -> dict:
    """Compare detected chords against expected for a song."""
    det_chords = set(detected["top_chords_by_duration"][:15])
    det_roots = set()
    det_qualities = {}
    
    for ch in det_chords:
        root, quality = parse_chord_simple(ch)
        det_roots.add(root)
        det_qualities[root] = quality
    
    # Also check enharmonic equivalents
    ENHARMONIC = {
        'Bb': 'A#', 'A#': 'Bb', 'Db': 'C#', 'C#': 'Db',
        'Eb': 'D#', 'D#': 'Eb', 'Gb': 'F#', 'F#': 'Gb',
        'Ab': 'G#', 'G#': 'Ab', 'Cb': 'B', 'B': 'Cb',
    }
    
    exp_roots = set(expected["expected_roots"])
    exp_chords = set(expected["main_chords"])
    
    # Root accuracy
    root_hits = 0
    root_details = []
    for er in expected["expected_roots"]:
        found = er in det_roots or ENHARMONIC.get(er, '') in det_roots
        if found:
            root_hits += 1
        root_details.append({"expected": er, "found": found})
    
    root_accuracy = root_hits / max(len(expected["expected_roots"]), 1)
    
    # Chord quality check — for each expected chord, did we detect something close?
    quality_hits = 0
    quality_details = []
    for ec in expected["main_chords"]:
        ec_root, ec_qual = parse_chord_simple(ec)
        # Check if we found this root with the right quality
        found_exact = ec in det_chords
        found_enharmonic = False
        found_root_only = False
        
        if not found_exact:
            # Check enharmonic
            enh_root = ENHARMONIC.get(ec_root, '')
            enh_chord = enh_root + ec_qual if enh_root else ''
            found_enharmonic = enh_chord in det_chords
        
        if not found_exact and not found_enharmonic:
            # Check if at least the root was right
            found_root_only = ec_root in det_roots or ENHARMONIC.get(ec_root, '') in det_roots
        
        matched = found_exact or found_enharmonic
        if matched:
            quality_hits += 1
        
        quality_details.append({
            "expected": ec,
            "exact_match": found_exact,
            "enharmonic_match": found_enharmonic,
            "root_only": found_root_only,
            "matched": matched,
        })
    
    quality_accuracy = quality_hits / max(len(expected["main_chords"]), 1)
    
    # Complex chord detection — did BTC detect any 7ths, maj7s, dim, sus, etc.?
    complex_detected = []
    for ch in detected["unique_chords"]:
        _, qual = parse_chord_simple(ch)
        if qual and qual not in ('', 'maj', 'm'):
            complex_detected.append(ch)
    
    return {
        "root_accuracy": round(root_accuracy * 100, 1),
        "root_details": root_details,
        "quality_accuracy": round(quality_accuracy * 100, 1),
        "quality_details": quality_details,
        "complex_chords_detected": complex_detected,
        "has_complex_chords": len(complex_detected) > 0,
        "detected_top_chords": detected["top_chords_by_duration"][:10],
        "expected_main_chords": expected["main_chords"],
        "detected_key": detected["key"],
        "expected_key": expected["key"],
    }


def parse_chord_simple(name: str) -> tuple:
    """Parse chord name into (root, quality)."""
    if not name or name == 'N':
        return ('N', '')
    root = name[0]
    rest = name[1:]
    if rest and rest[0] in ('#', 'b'):
        root += rest[0]
        rest = rest[1:]
    return (root, rest)


def main():
    results = {}
    
    # Phase 1: Download all songs
    logger.info("=" * 60)
    logger.info("PHASE 1: Downloading test songs")
    logger.info("=" * 60)
    
    audio_paths = {}
    for song in SONGS:
        path = download_song(song)
        if path:
            audio_paths[song["name"]] = path
        else:
            logger.warning(f"SKIPPING {song['name']} — download failed")
    
    logger.info(f"\nDownloaded {len(audio_paths)}/{len(SONGS)} songs")
    
    # Phase 2: Run BTC detection on each
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Running BTC chord detection")
    logger.info("=" * 60)
    
    for song in SONGS:
        name = song["name"]
        if name not in audio_paths:
            results[name] = {"error": "download_failed"}
            continue
        
        logger.info(f"\n--- Detecting chords: {name} ({song['artist']}) ---")
        try:
            detected = run_btc_detection(audio_paths[name])
            results[name] = {
                "detected": detected,
                "expected": EXPECTED.get(name, {}),
            }
            
            # Run comparison
            if name in EXPECTED:
                comparison = compare_results(detected, EXPECTED[name])
                results[name]["comparison"] = comparison
                logger.info(f"  Key: {detected['key']} (expected: {EXPECTED[name]['key']})")
                logger.info(f"  Root accuracy: {comparison['root_accuracy']}%")
                logger.info(f"  Quality accuracy: {comparison['quality_accuracy']}%")
                logger.info(f"  Complex chords: {comparison['complex_chords_detected'][:8]}")
                logger.info(f"  Top detected: {detected['top_chords_by_duration'][:8]}")
            
        except Exception as e:
            logger.error(f"Detection failed for {name}: {e}", exc_info=True)
            results[name] = {"error": str(e)}
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"\nResults saved to {RESULTS_FILE}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for song in SONGS:
        name = song["name"]
        r = results.get(name, {})
        if "error" in r:
            logger.info(f"  {name}: ERROR — {r['error']}")
        elif "comparison" in r:
            c = r["comparison"]
            logger.info(f"  {name}: Root={c['root_accuracy']}% Quality={c['quality_accuracy']}% Complex={len(c['complex_chords_detected'])}")
        else:
            logger.info(f"  {name}: No comparison data")


if __name__ == "__main__":
    main()
