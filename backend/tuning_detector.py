"""
Tuning Detection & Chord Name Correction for StemScribe.

Songs recorded in non-standard guitar tunings (half-step down = Eb, full step
down = D, etc.) produce chord names that are shifted from what guitarists expect.
For example, Little Wing by Jimi Hendrix is in Eb tuning: the BTC model correctly
detects the sounding pitches (D#m, F#, G#m) but guitarists expect standard chord
names (Em, G, Am).

This module detects the tuning offset and transposes chord names accordingly.

Detection strategies (in priority order):
1. **Vocabulary comparison** — If a UG/Songsterr chord vocabulary is available,
   test all 12 transpositions to find the best match. This is by far the most
   reliable method.
2. **Pitch analysis** — Use librosa.pyin on the guitar stem to measure the
   median pitch deviation from 440Hz-based equal temperament. If the guitar is
   tuned a half-step down, all pitches will be ~1 semitone flat.
3. **Chroma offset** — Compare the chroma centroid against expected positions
   for standard tuning.

The correction is applied as a post-processing step: chord names are transposed
UP by the detected offset (e.g., if tuning is Eb = half-step down, transpose +1).
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)

# Canonical note names using sharps (matches BTC vocabulary)
NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Flat equivalents for display
NOTE_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

# Map any note name to its pitch class (0-11, C=0)
NOTE_TO_PC = {}
for i, name in enumerate(NOTE_NAMES_SHARP):
    NOTE_TO_PC[name] = i
for i, name in enumerate(NOTE_NAMES_FLAT):
    NOTE_TO_PC[name] = i

# Common tuning names
TUNING_NAMES = {
    0: "Standard (E)",
    1: "Half-step down (Eb)",
    2: "Full step down (D)",
    -1: "Half-step up",
    -2: "Full step up",
}


def _parse_root(chord_name: str) -> Tuple[str, str]:
    """Parse chord name into (root, quality_suffix).

    Examples:
        'Em7' -> ('E', 'm7')
        'F#m' -> ('F#', 'm')
        'Bbmaj7' -> ('Bb', 'maj7')
        'N' -> ('N', '')
    """
    if not chord_name or chord_name == 'N':
        return ('N', '')
    root = chord_name[0]
    rest = chord_name[1:]
    if rest and rest[0] in ('#', 'b'):
        root += rest[0]
        rest = rest[1:]
    return (root, rest)


def transpose_chord(chord_name: str, semitones: int, prefer_flats: bool = False) -> str:
    """Transpose a chord name by the given number of semitones.

    Args:
        chord_name: e.g., 'D#m', 'F#maj7', 'Bbm7'
        semitones: positive = up, negative = down
        prefer_flats: if True, use flat names (Bb instead of A#)

    Returns:
        Transposed chord name with quality preserved.
    """
    if not chord_name or chord_name == 'N':
        return chord_name

    root, quality = _parse_root(chord_name)
    pc = NOTE_TO_PC.get(root)
    if pc is None:
        return chord_name  # Unknown root, return unchanged

    new_pc = (pc + semitones) % 12
    names = NOTE_NAMES_FLAT if prefer_flats else NOTE_NAMES_SHARP
    new_root = names[new_pc]
    return new_root + quality


def detect_tuning_from_vocab(chord_events: list, reference_vocab: list) -> int:
    """Detect tuning offset by comparing detected chords against a reference vocabulary.

    Tests all 12 transpositions of the detected chords and finds which offset
    produces the best match against the reference vocabulary (e.g., from Ultimate Guitar).

    Args:
        chord_events: List of ChordEvent objects (or dicts with 'chord', 'duration' keys)
        reference_vocab: List of chord name strings from UG/Songsterr

    Returns:
        Optimal transposition in semitones (0 = no change, 1 = half-step up, etc.)
    """
    if not chord_events or not reference_vocab:
        return 0

    # Build set of reference roots (normalized to pitch classes)
    ref_pcs = set()
    ref_chord_set = set()
    for ch in reference_vocab:
        root, quality = _parse_root(ch)
        pc = NOTE_TO_PC.get(root)
        if pc is not None:
            ref_pcs.add(pc)
            ref_chord_set.add((pc, quality.lower()))

    # Extract detected chord roots with duration weights
    detected = []
    for ev in chord_events:
        if hasattr(ev, 'chord'):
            chord_name = ev.chord
            duration = ev.duration
        elif isinstance(ev, dict):
            chord_name = ev.get('chord', 'N')
            duration = ev.get('duration', 1.0)
        else:
            continue
        root, quality = _parse_root(chord_name)
        pc = NOTE_TO_PC.get(root)
        if pc is not None:
            detected.append((pc, quality.lower(), duration))

    if not detected:
        return 0

    # Test all 12 transpositions
    best_offset = 0
    best_score = -1

    for offset in range(12):
        # Score 1: Root match (how many detected chord roots appear in reference)
        root_match_dur = 0.0
        # Score 2: Full chord match (root + quality)
        full_match_dur = 0.0
        total_dur = 0.0

        for pc, quality, dur in detected:
            transposed_pc = (pc + offset) % 12
            total_dur += dur
            if transposed_pc in ref_pcs:
                root_match_dur += dur
            # Simplified quality matching (m = m, maj = '', etc.)
            q_norm = quality.replace('maj', '') if quality == 'maj' else quality
            for ref_pc, ref_q in ref_chord_set:
                if transposed_pc == ref_pc:
                    ref_q_norm = ref_q.replace('maj', '') if ref_q == 'maj' else ref_q
                    # Check if qualities are compatible
                    if q_norm == ref_q_norm:
                        full_match_dur += dur
                        break
                    # Minor variants match each other
                    if ('m' in q_norm and 'm' in ref_q_norm) or (q_norm == '' and ref_q_norm == ''):
                        full_match_dur += dur * 0.5
                        break

        if total_dur == 0:
            continue

        # Combined score: root match matters most, full match is a bonus
        score = (root_match_dur / total_dur) * 0.7 + (full_match_dur / total_dur) * 0.3

        if score > best_score:
            best_score = score
            best_offset = offset

    # Only apply if there's a clear winner and it's a musically sensible offset (1-3 semitones)
    # If offset is 0, that's fine too. If offset is > 3, it's likely wrong.
    if best_offset > 6:
        # Treat as negative offset (e.g., 11 semitones up = 1 semitone down)
        best_offset = best_offset - 12

    logger.info(f"Vocab-based tuning detection: offset={best_offset} semitones, "
                f"score={best_score:.2f}")
    return best_offset


def detect_tuning_from_audio(audio_path: str, sr: int = 22050) -> Tuple[int, float]:
    """Detect tuning offset from audio using pitch analysis.

    Uses librosa.pyin to track pitches and measures systematic deviation from
    440Hz-based equal temperament.

    Args:
        audio_path: Path to audio file (preferably guitar stem)
        sr: Sample rate to use

    Returns:
        Tuple of (offset_semitones, effective_a4_hz)
        offset_semitones: 0 = standard, 1 = half-step down (transpose up 1), etc.
        effective_a4_hz: Estimated A4 reference frequency
    """
    try:
        import librosa

        y, sr = librosa.load(audio_path, sr=sr, mono=True)

        # Use pyin for robust pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=60, fmax=1000, sr=sr)
        f0_valid = f0[~np.isnan(f0)]

        if len(f0_valid) < 50:
            logger.info(f"Too few voiced frames ({len(f0_valid)}) for pitch-based tuning detection")
            return 0, 440.0

        # Calculate deviation of each pitch from nearest 440Hz-based note
        semitones_from_a4 = 12 * np.log2(f0_valid / 440.0)
        nearest_semitones = np.round(semitones_from_a4)
        deviations = semitones_from_a4 - nearest_semitones

        median_dev = np.median(deviations)
        effective_a4 = 440.0 * (2 ** (median_dev / 12))

        logger.info(f"Pitch-based tuning: median deviation={median_dev:.3f} semitones "
                    f"({median_dev*100:.1f} cents), effective A4={effective_a4:.1f}Hz")

        # Determine offset:
        # Half-step down: deviation around -0.4 to -0.6 (ambiguous zone) or clearly < -0.35
        # But note: after stem separation, the pitch often stays close to original
        # This method is less reliable than vocab comparison
        if median_dev < -0.35:
            return 1, effective_a4  # Half-step down -> transpose up 1
        elif median_dev < -0.85:
            return 2, effective_a4  # Full step down -> transpose up 2
        elif median_dev > 0.35:
            return -1, effective_a4  # Half-step up -> transpose down 1
        else:
            return 0, effective_a4  # Standard tuning

    except Exception as e:
        logger.warning(f"Pitch-based tuning detection failed: {e}")
        return 0, 440.0


def detect_and_correct_tuning(chord_events: list,
                               reference_vocab: list = None,
                               audio_path: str = None,
                               force_offset: int = None) -> dict:
    """Main entry point: detect tuning and transpose chord names.

    Priority:
    1. force_offset (if provided, skip detection)
    2. Vocabulary comparison (most reliable)
    3. Pitch analysis (fallback)

    Args:
        chord_events: List of ChordEvent objects from chord detection
        reference_vocab: Optional list of expected chord names (from UG, etc.)
        audio_path: Optional path to audio for pitch-based detection
        force_offset: Optional manual override (semitones to transpose up)

    Returns:
        Dict with:
            'chords': List of corrected ChordEvent objects
            'tuning_offset_semitones': Detected/applied offset
            'tuning_name': Human-readable tuning name
            'detection_method': How the offset was determined
            'effective_a4': Estimated A4 frequency (if audio analysis was done)
    """
    offset = 0
    method = "none"
    effective_a4 = 440.0

    if force_offset is not None:
        offset = force_offset
        method = "manual"
        logger.info(f"Using forced tuning offset: {offset}")
    elif reference_vocab:
        offset = detect_tuning_from_vocab(chord_events, reference_vocab)
        method = "vocabulary"
        if audio_path:
            # Also run audio analysis for informational purposes
            _, effective_a4 = detect_tuning_from_audio(audio_path)
    elif audio_path:
        offset, effective_a4 = detect_tuning_from_audio(audio_path)
        method = "pitch_analysis"

    # Apply transposition
    if offset == 0:
        corrected = chord_events
        logger.info("No tuning correction needed (offset=0)")
    else:
        # Determine if we should prefer flats based on the key context
        # For common guitar keys after transposition, sharps are usually fine
        corrected = _transpose_chord_events(chord_events, offset)
        logger.info(f"Applied tuning correction: transposed {len(chord_events)} chords "
                    f"by {offset:+d} semitones ({TUNING_NAMES.get(offset, f'{offset} semitones')})")

    tuning_name = TUNING_NAMES.get(offset, f"{offset:+d} semitones")

    return {
        'chords': corrected,
        'tuning_offset_semitones': offset,
        'tuning_name': tuning_name,
        'detection_method': method,
        'effective_a4': effective_a4,
    }


def _transpose_chord_events(chord_events: list, semitones: int) -> list:
    """Transpose all chord events by the given number of semitones.

    Handles both ChordEvent objects and dicts.
    """
    from chord_detector_v10 import ChordEvent, _parse_chord

    transposed = []
    for ev in chord_events:
        if hasattr(ev, 'chord'):
            # ChordEvent object
            new_chord = transpose_chord(ev.chord, semitones)
            new_root, new_quality = _parse_chord(new_chord)
            transposed.append(ChordEvent(
                time=ev.time,
                duration=ev.duration,
                chord=new_chord,
                root=new_root,
                quality=new_quality,
                confidence=ev.confidence,
            ))
        elif isinstance(ev, dict):
            # Dict format
            new_chord = transpose_chord(ev.get('chord', 'N'), semitones)
            new_root, new_quality = _parse_chord(new_chord)
            new_ev = dict(ev)
            new_ev['chord'] = new_chord
            new_ev['root'] = new_root
            new_ev['quality'] = new_quality
            transposed.append(new_ev)
        else:
            transposed.append(ev)

    return transposed
