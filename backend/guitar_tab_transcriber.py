"""
Guitar Tab Transcriber for StemScriber
=====================================
Uses Basic Pitch (Google/Spotify) for polyphonic pitch detection, then applies
guitar-specific post-processing to produce high-quality guitar tablature MIDI.

Post-processing pipeline:
  1. Basic Pitch raw transcription (polyphonic AMT)
  2. Key detection for spurious note filtering
  3. Guitar range filtering (E2=40 to E6=88)
  4. Polyphony limiting (max 6 simultaneous notes)
  5. Note grouping — simultaneous notes assigned as chords
  6. String/fret assignment using chord-shape-aware FretMapper with hand position model
  7. Very short note cleanup (< 30ms glitches)
  8. Velocity dynamics from Basic Pitch confidence

The output MIDI uses guitar MIDI note numbers (TUNING[string] + fret)
which flow directly into the existing midi_to_gp.py Guitar Pro pipeline.
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# CHECK BASIC PITCH AVAILABILITY
# ============================================================================

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH  # noqa: F401
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    logger.warning("basic_pitch not available — guitar tab transcriber disabled")

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

MODEL_AVAILABLE = BASIC_PITCH_AVAILABLE and PRETTY_MIDI_AVAILABLE

if MODEL_AVAILABLE:
    logger.info("Guitar tab transcriber ready (Basic Pitch + post-processing)")


# ============================================================================
# CONSTANTS
# ============================================================================

SAMPLE_RATE = 22050
NUM_STRINGS = 6
NUM_FRETS = 20

# Standard guitar tuning: E2 A2 D3 G3 B3 E4 (MIDI note numbers)
# Matches midi_to_gp.py TUNINGS['guitar'] exactly
TUNING = [40, 45, 50, 55, 59, 64]

# Guitar range limits
GUITAR_MIN_MIDI = 40   # E2 (low E open)
GUITAR_MAX_MIDI = 88   # E6 (fret 24 on high E, generous upper bound)
MAX_POLYPHONY = 6       # Can't play more than 6 notes at once

# Chord grouping: notes starting within this window are treated as simultaneous
CHORD_ONSET_TOLERANCE = 0.03  # 30ms — typical strum spread

# Key detection: pitch class profiles for major and minor keys (Krumhansl-Kessler)
# Index 0 = C, 1 = C#, ..., 11 = B
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Common open chord shapes — (fret_min, fret_max, frozenset of intervals from root)
# Used to bias fret assignment toward recognizable chord voicings
COMMON_CHORD_INTERVALS = {
    'major':     frozenset([0, 4, 7]),
    'minor':     frozenset([0, 3, 7]),
    'dom7':      frozenset([0, 4, 7, 10]),
    'min7':      frozenset([0, 3, 7, 10]),
    'maj7':      frozenset([0, 4, 7, 11]),
    'sus2':      frozenset([0, 2, 7]),
    'sus4':      frozenset([0, 5, 7]),
    'power':     frozenset([0, 7]),
    'add9':      frozenset([0, 2, 4, 7]),
}


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class TabTranscriptionResult:
    """Result of guitar tab transcription."""
    midi_path: Optional[str]
    num_notes: int
    quality_score: float            # 0.0 - 1.0
    method: str                     # 'basic_pitch_guitar'
    num_strings_used: int           # How many strings had notes (0-6)
    fret_range: Tuple[int, int]     # (min_fret, max_fret)


# ============================================================================
# KEY DETECTION
# ============================================================================

def _detect_key(notes: List) -> Optional[Tuple[str, str]]:
    """
    Detect the musical key from a list of MIDI notes using Krumhansl-Kessler
    pitch-class profile correlation.

    Returns:
        (root_name, mode) e.g. ('G', 'major') or ('E', 'minor'), or None
    """
    if not notes or len(notes) < 8:
        return None

    # Build weighted pitch-class histogram (weight by duration * velocity)
    pc_histogram = np.zeros(12)
    for n in notes:
        pc = n.pitch % 12
        duration = max(0.01, n.end - n.start)
        weight = duration * (n.velocity / 127.0)
        pc_histogram[pc] += weight

    if pc_histogram.sum() == 0:
        return None

    # Normalize
    pc_histogram = pc_histogram / pc_histogram.sum()

    best_key = None
    best_corr = -2.0

    major = np.array(MAJOR_PROFILE)
    minor = np.array(MINOR_PROFILE)

    for root in range(12):
        # Rotate histogram so root aligns with index 0
        rotated = np.roll(pc_histogram, -root)

        # Correlate with major profile
        corr_maj = np.corrcoef(rotated, major)[0, 1]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = (NOTE_NAMES[root], 'major')

        # Correlate with minor profile
        corr_min = np.corrcoef(rotated, minor)[0, 1]
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = (NOTE_NAMES[root], 'minor')

    if best_corr < 0.3:
        return None  # Low confidence — don't trust it

    return best_key


def _get_scale_pitches(key: Tuple[str, str]) -> Set[int]:
    """
    Get the set of pitch classes (0-11) that belong to a key's scale.

    Args:
        key: (root_name, mode) from _detect_key

    Returns:
        Set of pitch class integers (0=C, 1=C#, ...)
    """
    root_pc = NOTE_NAMES.index(key[0])

    if key[1] == 'major':
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else:  # minor (natural)
        intervals = [0, 2, 3, 5, 7, 8, 10]

    return {(root_pc + i) % 12 for i in intervals}


def _filter_by_key(notes: List, key: Tuple[str, str],
                   tolerance: float = 0.15) -> List:
    """
    Filter out notes that are very unlikely given the detected key.

    Only removes low-confidence notes (low velocity) that are off-scale.
    High-velocity off-scale notes are kept (accidentals, chromatic passing tones).

    Args:
        notes: List of MIDI notes
        key: Detected key (root, mode)
        tolerance: Fraction of notes allowed to be off-scale before disabling filter

    Returns:
        Filtered note list
    """
    scale_pcs = _get_scale_pitches(key)

    # Count how many notes are off-scale
    off_scale = sum(1 for n in notes if (n.pitch % 12) not in scale_pcs)
    off_ratio = off_scale / max(len(notes), 1)

    # If too many notes are off-scale, the key detection is probably wrong
    if off_ratio > tolerance:
        return notes

    # Only remove off-scale notes that also have low velocity (likely spurious)
    # Median velocity threshold: notes below median AND off-scale get removed
    velocities = [n.velocity for n in notes]
    vel_threshold = np.percentile(velocities, 30) if velocities else 50

    filtered = []
    removed = 0
    for n in notes:
        pc = n.pitch % 12
        if pc not in scale_pcs and n.velocity < vel_threshold:
            removed += 1
            continue
        filtered.append(n)

    if removed > 0:
        logger.info(f"Key filter ({key[0]} {key[1]}): removed {removed} low-confidence off-scale notes")

    return filtered


# ============================================================================
# CHORD-INFORMED FILTERING
# ============================================================================

def _filter_by_chords(notes: List, chord_progression: list,
                      velocity_percentile: float = 40) -> List:
    """
    Filter notes using detected chord progression. Notes whose pitch class
    does not belong to the chord active at their timestamp have their velocity
    penalized. Low-velocity non-chord-tone notes are removed entirely.

    This compensates for the lower Basic Pitch thresholds (onset=0.30, frame=0.15)
    which catch more real notes but also more hallucinations.

    Only removes notes that are BOTH:
      - Not a chord tone at their timestamp
      - Below the velocity threshold (low confidence from Basic Pitch)

    High-velocity non-chord-tones are kept as passing tones / embellishments.

    Args:
        notes: List of MIDI notes from Basic Pitch
        chord_progression: List of chord dicts with 'time', 'duration', 'chord'
        velocity_percentile: Notes below this percentile AND off-chord get removed

    Returns:
        Filtered note list
    """
    if not chord_progression or not notes:
        return notes

    try:
        from essentia_chord_detector import chord_to_pitch_classes, get_chord_at_time
    except ImportError:
        logger.debug("essentia_chord_detector not available for chord filtering")
        return notes

    # Precompute velocity threshold
    velocities = [n.velocity for n in notes]
    if not velocities:
        return notes
    vel_threshold = np.percentile(velocities, velocity_percentile)

    filtered = []
    removed = 0
    for n in notes:
        # Find what chord is playing at this note's onset
        chord_at_time = get_chord_at_time(chord_progression, n.start)
        if chord_at_time is None:
            # No chord info at this time — keep the note
            filtered.append(n)
            continue

        chord_name = chord_at_time.get('chord', '') if isinstance(chord_at_time, dict) else getattr(chord_at_time, 'chord', '')
        pitch_classes = chord_to_pitch_classes(chord_name)

        if not pitch_classes:
            # Unknown chord — keep the note
            filtered.append(n)
            continue

        note_pc = n.pitch % 12
        if note_pc in pitch_classes:
            # Note is a chord tone — definitely keep
            filtered.append(n)
        elif n.velocity >= vel_threshold:
            # Off-chord but high velocity — likely intentional (passing tone, etc.)
            filtered.append(n)
        else:
            # Off-chord AND low velocity — likely a hallucination
            removed += 1

    if removed > 0:
        logger.info(f"Chord filter: removed {removed} low-confidence non-chord-tone notes")

    return filtered


# ============================================================================
# NOTE GROUPING
# ============================================================================

def _group_simultaneous_notes(notes: List, tolerance: float = CHORD_ONSET_TOLERANCE) -> List[List]:
    """
    Group notes that start within `tolerance` seconds of each other.
    These are likely part of the same chord strum or arpeggio.

    Returns:
        List of note groups. Single notes are groups of length 1.
    """
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start)
    groups = []
    current_group = [sorted_notes[0]]

    for note in sorted_notes[1:]:
        if note.start - current_group[0].start <= tolerance:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]

    groups.append(current_group)
    return groups


# ============================================================================
# STRING/FRET ASSIGNMENT (CHORD-AWARE)
# ============================================================================

def _assign_string_fret(midi_note: int, prev_fret: int = None,
                        prev_string: int = None,
                        hand_position: float = None) -> Optional[Tuple[int, int]]:
    """
    Assign a MIDI note to (string, fret) on guitar using playability heuristics.

    Prefers:
      - Positions near current hand position
      - Lower fret positions (more comfortable)
      - Continuity with previous note (minimal hand movement)
      - Middle strings for single notes
      - Open strings when appropriate

    Args:
        midi_note: MIDI note number
        prev_fret: Previous note's fret (for position continuity)
        prev_string: Previous note's string (0-indexed)
        hand_position: Current average hand position (fret number)

    Returns:
        (string_index, fret) -- 0-indexed string, or None if out of range
    """
    candidates = []

    for s_idx, open_note in enumerate(TUNING):
        fret = midi_note - open_note
        if 0 <= fret <= NUM_FRETS:
            score = 0.0

            # Prefer lower frets
            if fret == 0:
                score += 2       # Open strings are easy
            elif fret <= 4:
                score += 0       # First position
            elif fret <= 7:
                score += 3
            elif fret <= 12:
                score += 8
            else:
                score += 15

            # Prefer middle strings for single notes
            mid = NUM_STRINGS // 2
            score += abs(s_idx - mid) * 2

            # Hand position proximity (weighted average of recent frets)
            if hand_position is not None:
                pos_distance = abs(fret - hand_position)
                if pos_distance <= 2:
                    score -= 4  # Within hand span
                elif pos_distance <= 4:
                    score += 2
                else:
                    score += pos_distance * 1.5

            # Hand position continuity
            if prev_fret is not None:
                fret_jump = abs(fret - prev_fret)
                if fret_jump == 0:
                    score -= 3
                elif fret_jump <= 2:
                    score += 0
                elif fret_jump <= 4:
                    score += 5
                else:
                    score += fret_jump * 2

            # String continuity
            if prev_string is not None:
                string_jump = abs(s_idx - prev_string)
                if string_jump == 0:
                    score -= 2
                elif string_jump == 1:
                    score += 0
                else:
                    score += string_jump * 3

            candidates.append((s_idx, fret, score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2])
    return (candidates[0][0], candidates[0][1])


def _assign_chord_frets(chord_notes: List, prev_fret: int = None,
                        prev_string: int = None,
                        hand_position: float = None) -> List[dict]:
    """
    Assign string/fret positions for a group of simultaneous notes (chord).

    Ensures:
      - Each note on a different string
      - Fret span <= 4 (comfortable hand stretch)
      - Preference for known chord shapes
      - Minimal hand movement from previous position

    Args:
        chord_notes: List of MIDI notes sounding together
        prev_fret: Previous position's fret
        prev_string: Previous position's string
        hand_position: Current average hand position

    Returns:
        List of dicts with 'pitch', 'string', 'fret', 'start', 'end', 'velocity'
    """
    if len(chord_notes) == 1:
        # Single note — use the standard single-note assigner
        note = chord_notes[0]
        result = _assign_string_fret(note.pitch, prev_fret, prev_string, hand_position)
        if result is None:
            return []
        s_idx, fret = result
        return [{
            'pitch': note.pitch,
            'string': s_idx,
            'fret': fret,
            'start': note.start,
            'end': note.end,
            'velocity': note.velocity,
        }]

    # Sort notes low to high for consistent string assignment
    sorted_notes = sorted(chord_notes, key=lambda n: n.pitch)

    # Find all valid (string, fret) for each note
    all_candidates = []
    for note in sorted_notes:
        note_options = []
        for s_idx, open_note in enumerate(TUNING):
            fret = note.pitch - open_note
            if 0 <= fret <= NUM_FRETS:
                note_options.append((s_idx, fret))
        all_candidates.append(note_options)

    # If any note has no valid position, skip it
    if not all(all_candidates):
        # Fall back to individual assignment for whatever we can
        results = []
        for note in sorted_notes:
            r = _assign_string_fret(note.pitch, prev_fret, prev_string, hand_position)
            if r:
                results.append({
                    'pitch': note.pitch, 'string': r[0], 'fret': r[1],
                    'start': note.start, 'end': note.end, 'velocity': note.velocity,
                })
        return results

    # Search for best combination with unique strings and playable fret span
    best_combo = None
    best_score = float('inf')

    def _search(note_idx, used_strings, current_combo):
        nonlocal best_combo, best_score

        if note_idx == len(sorted_notes):
            frets = [f for _, f in current_combo]
            non_open = [f for f in frets if f > 0]

            if non_open:
                fret_span = max(non_open) - min(non_open)
                if fret_span > 5:
                    return  # Hand can't stretch this far

                # Score the combination
                score = 0.0

                # Prefer smaller fret span
                score += fret_span * 3

                # Prefer lower positions
                score += min(non_open) * 0.5

                # Prefer positions near current hand position
                if hand_position is not None:
                    avg_fret = sum(non_open) / len(non_open)
                    score += abs(avg_fret - hand_position) * 2

                # Prefer positions near previous fret
                if prev_fret is not None:
                    avg_fret = sum(frets) / len(frets)
                    score += abs(avg_fret - prev_fret) * 1.5

                # Check if it matches a known chord shape (bonus)
                pitch_classes = frozenset(sorted_notes[i].pitch % 12 for i in range(len(sorted_notes)))
                root_pc = min(pitch_classes)
                intervals = frozenset((pc - root_pc) % 12 for pc in pitch_classes)
                for shape_intervals in COMMON_CHORD_INTERVALS.values():
                    if intervals == shape_intervals:
                        score -= 5  # Reward recognized chord shapes
                        break
            else:
                score = 0  # All open strings — great

            if score < best_score:
                best_score = score
                best_combo = list(current_combo)
            return

        for s_idx, fret in all_candidates[note_idx]:
            if s_idx not in used_strings:
                _search(
                    note_idx + 1,
                    used_strings | {s_idx},
                    current_combo + [(s_idx, fret)]
                )

    _search(0, set(), [])

    if best_combo is None:
        # No valid chord voicing found — fall back to individual assignment
        results = []
        for note in sorted_notes:
            r = _assign_string_fret(note.pitch, prev_fret, prev_string, hand_position)
            if r:
                results.append({
                    'pitch': note.pitch, 'string': r[0], 'fret': r[1],
                    'start': note.start, 'end': note.end, 'velocity': note.velocity,
                })
        return results

    results = []
    for i, (s_idx, fret) in enumerate(best_combo):
        note = sorted_notes[i]
        results.append({
            'pitch': note.pitch,
            'string': s_idx,
            'fret': fret,
            'start': note.start,
            'end': note.end,
            'velocity': note.velocity,
        })

    return results


# ============================================================================
# POST-PROCESSING
# ============================================================================

def _filter_guitar_range(notes: List) -> List:
    """Remove notes outside guitar range."""
    return [n for n in notes if GUITAR_MIN_MIDI <= n.pitch <= GUITAR_MAX_MIDI]


def _remove_short_notes(notes: List, min_duration: float = 0.03) -> List:
    """Remove very short notes (likely glitches)."""
    return [n for n in notes if (n.end - n.start) >= min_duration]


def _limit_polyphony(notes: List, max_voices: int = MAX_POLYPHONY) -> List:
    """
    Limit simultaneous notes to max_voices (6 for guitar).

    When more than max_voices notes overlap, keep the ones with
    highest velocity (strongest detection confidence).
    """
    if not notes:
        return notes

    # Sort by start time
    notes_sorted = sorted(notes, key=lambda n: n.start)

    # Build timeline of active notes
    kept = []
    for note in notes_sorted:
        # Count how many kept notes are active at this note's start
        active = [n for n in kept if n.end > note.start + 0.005]
        if len(active) < max_voices:
            kept.append(note)
        else:
            # Replace weakest active note if this one is stronger
            weakest = min(active, key=lambda n: n.velocity)
            if note.velocity > weakest.velocity:
                kept.remove(weakest)
                kept.append(note)

    return kept


def _fix_overlapping_same_pitch(notes: List) -> List:
    """Fix overlapping notes on the same pitch (re-strikes)."""
    notes_by_pitch = {}
    for n in notes:
        notes_by_pitch.setdefault(n.pitch, []).append(n)

    cleaned = []
    for pitch, pitch_notes in notes_by_pitch.items():
        pitch_notes.sort(key=lambda n: n.start)
        for i, note in enumerate(pitch_notes):
            if i + 1 < len(pitch_notes):
                next_note = pitch_notes[i + 1]
                if note.end > next_note.start:
                    # Truncate to avoid overlap
                    note.end = next_note.start - 0.005
                    if note.end <= note.start:
                        continue  # Skip if too short
            cleaned.append(note)

    return cleaned


# ============================================================================
# TRANSCRIBER CLASS
# ============================================================================

class GuitarTabTranscriber:
    """
    Transcribes guitar audio to MIDI using Basic Pitch + guitar post-processing.

    Usage:
        transcriber = GuitarTabTranscriber()
        result = transcriber.transcribe('guitar_stem.wav', '/output/dir')
    """

    def transcribe(self, audio_path: str, output_dir: str,
                   tempo_hint: float = None,
                   chord_progression: list = None) -> TabTranscriptionResult:
        """
        Full guitar tab transcription pipeline.

        1. Run Basic Pitch polyphonic AMT
        2. Filter to guitar range (E2-E6)
        3. Remove glitch notes (< 30ms)
        4. Limit polyphony to 6 voices
        5. Fix overlapping same-pitch notes
        6. Assign string/fret positions
        7. Write MIDI with guitar program

        Args:
            audio_path: Path to guitar audio file (WAV, MP3, etc.)
            output_dir: Directory for output MIDI file
            tempo_hint: Known tempo from earlier analysis (optional)

        Returns:
            TabTranscriptionResult with MIDI path and quality metrics
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        if not MODEL_AVAILABLE:
            logger.warning("Basic Pitch not available for guitar transcription")
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_guitar', num_strings_used=0,
                fret_range=(0, 0),
            )

        logger.info(f"Guitar tab transcription: {audio_path.name}")

        # --- Step 1: Basic Pitch transcription ---
        # Basic Pitch is a piano-centric model. For guitar we raise thresholds
        # to reduce hallucinated notes, and constrain frequency range tightly.
        # When chord_progression is available, the chord filter (step 2c) provides
        # additional cleanup, so we can afford slightly lower thresholds.
        try:
            has_chords = bool(chord_progression)
            # With chord filtering available, we can be more permissive;
            # without it, use stricter thresholds to avoid garbage output.
            onset_thresh = 0.40 if has_chords else 0.50
            frame_thresh = 0.25 if has_chords else 0.35

            model_output, midi_data, note_events = predict(
                str(audio_path),
                onset_threshold=onset_thresh,   # Higher = fewer ghost notes
                frame_threshold=frame_thresh,   # Higher = less hallucinated sustain
                minimum_note_length=80,         # 80ms min -- guitar notes are rarely shorter
                minimum_frequency=80.0,         # ~E2 (82 Hz)
                maximum_frequency=1400.0,       # ~F6 -- guitar fundamental range
            )
        except Exception as e:
            logger.error(f"Basic Pitch failed: {e}")
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_guitar', num_strings_used=0,
                fret_range=(0, 0),
            )

        if midi_data is None or not midi_data.instruments:
            logger.info("Basic Pitch produced no notes")
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_guitar', num_strings_used=0,
                fret_range=(0, 0),
            )

        raw_notes = midi_data.instruments[0].notes
        logger.info(f"Basic Pitch raw: {len(raw_notes)} notes")

        # --- Step 2: Guitar-specific post-processing ---
        notes = _filter_guitar_range(raw_notes)
        logger.info(f"After guitar range filter: {len(notes)} notes")

        notes = _remove_short_notes(notes, min_duration=0.03)
        notes = _fix_overlapping_same_pitch(notes)
        notes = _limit_polyphony(notes, MAX_POLYPHONY)
        logger.info(f"After cleanup: {len(notes)} notes")

        if not notes:
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_guitar', num_strings_used=0,
                fret_range=(0, 0),
            )

        # --- Step 2b: Key detection and spurious note filtering ---
        detected_key = _detect_key(notes)
        if detected_key:
            logger.info(f"Detected key: {detected_key[0]} {detected_key[1]}")
            notes = _filter_by_key(notes, detected_key)
            logger.info(f"After key filter: {len(notes)} notes")

        # --- Step 2c: Chord-informed filtering ---
        # If chord progression is available, penalize notes that don't belong
        # to the detected chord at their timestamp. This dramatically reduces
        # hallucinated notes from the lower Basic Pitch thresholds.
        if chord_progression and notes:
            notes = _filter_by_chords(notes, chord_progression)
            logger.info(f"After chord filter: {len(notes)} notes")

        if not notes:
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_guitar', num_strings_used=0,
                fret_range=(0, 0),
            )

        # --- Step 3: Group simultaneous notes and assign string/fret ---
        note_groups = _group_simultaneous_notes(notes)
        logger.info(f"Note groups: {len(note_groups)} ({sum(1 for g in note_groups if len(g) > 1)} chords)")

        tab_notes = []
        prev_fret = None
        prev_string = None
        strings_used = set()
        frets_used = []

        # Hand position model: weighted moving average of recent fret positions
        position_history = []
        hand_position = None

        for group in note_groups:
            if len(group) == 1:
                # Single note
                note = group[0]
                result = _assign_string_fret(note.pitch, prev_fret, prev_string, hand_position)
                if result is not None:
                    s_idx, fret = result
                    strings_used.add(s_idx)
                    frets_used.append(fret)
                    prev_fret = fret
                    prev_string = s_idx

                    # Update hand position model
                    position_history.append(fret)
                    if len(position_history) > 8:
                        position_history.pop(0)
                    # Weighted average: recent positions count more
                    weights = list(range(1, len(position_history) + 1))
                    hand_position = sum(f * w for f, w in zip(position_history, weights)) / sum(weights)

                    tab_notes.append({
                        'pitch': note.pitch,
                        'string': s_idx,
                        'fret': fret,
                        'start': note.start,
                        'end': note.end,
                        'velocity': note.velocity,
                    })
            else:
                # Chord (multiple simultaneous notes)
                chord_results = _assign_chord_frets(group, prev_fret, prev_string, hand_position)
                for cr in chord_results:
                    strings_used.add(cr['string'])
                    frets_used.append(cr['fret'])
                    tab_notes.append(cr)

                # Update context from chord
                if chord_results:
                    # Use the lowest note as the reference for next position
                    lowest = min(chord_results, key=lambda x: x['string'])
                    prev_fret = lowest['fret']
                    prev_string = lowest['string']

                    chord_frets = [cr['fret'] for cr in chord_results if cr['fret'] > 0]
                    if chord_frets:
                        avg_fret = sum(chord_frets) / len(chord_frets)
                        position_history.append(avg_fret)
                        if len(position_history) > 8:
                            position_history.pop(0)
                        weights = list(range(1, len(position_history) + 1))
                        hand_position = sum(f * w for f, w in zip(position_history, weights)) / sum(weights)

        if not tab_notes:
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_guitar', num_strings_used=0,
                fret_range=(0, 0),
            )

        # --- Step 4: Tempo detection ---
        if tempo_hint and 40 < tempo_hint < 300:
            tempo = tempo_hint
        else:
            try:
                import librosa
                audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
                tempo_result = librosa.beat.beat_track(y=audio, sr=sr)
                if hasattr(tempo_result[0], '__len__'):
                    tempo = float(tempo_result[0][0])
                else:
                    tempo = float(tempo_result[0])
                tempo = max(40.0, min(300.0, tempo))
            except Exception:
                tempo = 120.0

        # --- Step 5: Write MIDI ---
        midi_out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=25, name='Guitar')

        for tn in tab_notes:
            midi_note = pretty_midi.Note(
                velocity=tn['velocity'],
                pitch=tn['pitch'],
                start=tn['start'],
                end=max(tn['end'], tn['start'] + 0.03),
            )
            instrument.notes.append(midi_note)

        midi_out.instruments.append(instrument)

        midi_filename = f"{audio_path.stem}_tab.mid"
        midi_path = output_dir / midi_filename
        midi_out.write(str(midi_path))

        # --- Step 6: Quality metrics ---
        duration = max(n['end'] for n in tab_notes) - min(n['start'] for n in tab_notes)
        note_density = len(tab_notes) / max(duration, 0.1)

        quality = 0.0
        quality += 0.3 * min(1.0, len(tab_notes) / (duration * 2))   # Note density
        quality += 0.2 * min(1.0, len(strings_used) / 4)              # String diversity
        quality += 0.2 if 1.0 < note_density < 15.0 else 0.0          # Sane density
        quality += 0.15 if max(frets_used) - min(frets_used) < 15 else 0.0
        quality += 0.15                                                 # Base confidence
        quality = min(1.0, quality)

        logger.info(f"Tab MIDI: {midi_path.name}, {len(tab_notes)} notes, "
                    f"strings={sorted(strings_used)}, "
                    f"frets={min(frets_used)}-{max(frets_used)}, "
                    f"tempo={tempo:.0f}, quality={quality:.2f}")

        return TabTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(tab_notes),
            quality_score=quality,
            method='basic_pitch_guitar',
            num_strings_used=len(strings_used),
            fret_range=(min(frets_used), max(frets_used)),
        )


# ============================================================================
# CHORD-TO-TAB GENERATION (fallback when Basic Pitch quality is low)
# ============================================================================

# Standard guitar chord voicings: chord_name -> list of 6 fret values (-1 = muted)
# Order: [low E, A, D, G, B, high E]  (string index 0-5 matches TUNING)
CHORD_VOICINGS = {
    # Major chords
    'C':    [-1, 3, 2, 0, 1, 0],
    'D':    [-1, -1, 0, 2, 3, 2],
    'E':    [0, 2, 2, 1, 0, 0],
    'F':    [1, 3, 3, 2, 1, 1],
    'G':    [3, 2, 0, 0, 0, 3],
    'A':    [-1, 0, 2, 2, 2, 0],
    'B':    [-1, 2, 4, 4, 4, 2],
    # Minor chords
    'Am':   [-1, 0, 2, 2, 1, 0],
    'Bm':   [-1, 2, 4, 4, 3, 2],
    'Cm':   [-1, 3, 5, 5, 4, 3],
    'Dm':   [-1, -1, 0, 2, 3, 1],
    'Em':   [0, 2, 2, 0, 0, 0],
    'Fm':   [1, 3, 3, 1, 1, 1],
    'Gm':   [3, 5, 5, 3, 3, 3],
    # Dominant 7th
    'A7':   [-1, 0, 2, 0, 2, 0],
    'B7':   [-1, 2, 1, 2, 0, 2],
    'C7':   [-1, 3, 2, 3, 1, 0],
    'D7':   [-1, -1, 0, 2, 1, 2],
    'E7':   [0, 2, 0, 1, 0, 0],
    'G7':   [3, 2, 0, 0, 0, 1],
    # Minor 7th
    'Am7':  [-1, 0, 2, 0, 1, 0],
    'Bm7':  [-1, 2, 0, 2, 0, 2],
    'Cm7':  [-1, 3, 5, 3, 4, 3],
    'Em7':  [0, 2, 0, 0, 0, 0],
    'Dm7':  [-1, -1, 0, 2, 1, 1],
    'Fm7':  [1, 3, 1, 1, 1, 1],
    'Gm7':  [3, 5, 3, 3, 3, 3],
    # Flat/sharp chords
    'Bb':   [-1, 1, 3, 3, 3, 1],
    'Eb':   [-1, -1, 1, 3, 4, 3],
    'Ab':   [4, 6, 6, 5, 4, 4],
    'Db':   [-1, -1, 3, 1, 2, 1],
    'Gb':   [2, 4, 4, 3, 2, 2],
    'F#':   [2, 4, 4, 3, 2, 2],
    'C#':   [-1, 4, 6, 6, 6, 4],
    'F#m':  [2, 4, 4, 2, 2, 2],
    'C#m':  [-1, 4, 6, 6, 5, 4],
    'G#m':  [4, 6, 6, 4, 4, 4],
    'Bbm':  [-1, 1, 3, 3, 2, 1],
    'Ebm':  [-1, -1, 1, 3, 4, 2],
    # Enharmonic aliases
    'A#':   [-1, 1, 3, 3, 3, 1],
    'D#':   [-1, -1, 1, 3, 4, 3],
    'G#':   [4, 6, 6, 5, 4, 4],
    'A#m':  [-1, 1, 3, 3, 2, 1],
    'D#m':  [-1, -1, 1, 3, 4, 2],
    'Gbm':  [2, 4, 4, 2, 2, 2],
    'Dbm':  [-1, 4, 6, 6, 5, 4],
    'Abm':  [4, 6, 6, 4, 4, 4],
    # 7th variants for sharps/flats
    'Bb7':  [-1, 1, 3, 1, 3, 1],
    'Eb7':  [-1, -1, 1, 3, 2, 3],
    'Ab7':  [4, 6, 4, 5, 4, 4],
    'F#7':  [2, 4, 2, 3, 2, 2],
    'C#7':  [-1, 4, 6, 4, 6, 4],
    'Db7':  [-1, 4, 6, 4, 6, 4],
    'G#7':  [4, 6, 4, 5, 4, 4],
    'A#7':  [-1, 1, 3, 1, 3, 1],
    'D#7':  [-1, -1, 1, 3, 2, 3],
    'Gb7':  [2, 4, 2, 3, 2, 2],
    # Minor 7th variants
    'Bbm7': [-1, 1, 3, 1, 2, 1],
    'A#m7': [-1, 1, 3, 1, 2, 1],
    'Ebm7': [-1, -1, 1, 3, 2, 2],
    'D#m7': [-1, -1, 1, 3, 2, 2],
    'F#m7': [2, 4, 2, 2, 2, 2],
    'Gbm7': [2, 4, 2, 2, 2, 2],
    'C#m7': [-1, 4, 6, 4, 5, 4],
    'Dbm7': [-1, 4, 6, 4, 5, 4],
    'G#m7': [4, 6, 4, 4, 4, 4],
    'Abm7': [4, 6, 4, 4, 4, 4],
    # Major 7th
    'Cmaj7': [-1, 3, 2, 0, 0, 0],
    'Dmaj7': [-1, -1, 0, 2, 2, 2],
    'Emaj7': [0, 2, 1, 1, 0, 0],
    'Fmaj7': [1, 3, 3, 2, 1, 0],
    'Gmaj7': [3, 2, 0, 0, 0, 2],
    'Amaj7': [-1, 0, 2, 1, 2, 0],
    'Bmaj7': [-1, 2, 4, 3, 4, -1],
    'Bbmaj7': [-1, 1, 3, 2, 3, 1],
    'Ebmaj7': [-1, -1, 1, 3, 3, 3],
    'Abmaj7': [4, 6, 5, 5, 4, 4],
    # Sus chords
    'Asus2': [-1, 0, 2, 2, 0, 0],
    'Asus4': [-1, 0, 2, 2, 3, 0],
    'Dsus2': [-1, -1, 0, 2, 3, 0],
    'Dsus4': [-1, -1, 0, 2, 3, 3],
    'Esus4': [0, 2, 2, 2, 0, 0],
}


def _voicing_to_midi_notes(voicing: list) -> List[int]:
    """Convert a chord voicing (fret list) to MIDI note numbers."""
    notes = []
    for string_idx, fret in enumerate(voicing):
        if fret >= 0:
            midi_note = TUNING[string_idx] + fret
            notes.append(midi_note)
    return notes


def _get_chord_voicing(chord_name: str) -> Optional[list]:
    """
    Look up a chord voicing. Handles common naming variations.

    Returns:
        List of 6 fret values, or None if chord is unknown.
    """
    if not chord_name or chord_name == 'N':
        return None

    # Direct lookup
    if chord_name in CHORD_VOICINGS:
        return CHORD_VOICINGS[chord_name]

    # Try normalizing: remove spaces, handle "min" -> "m" etc.
    normalized = chord_name.replace('min', 'm').replace('maj', 'maj').replace('major', '').replace('minor', 'm')
    if normalized in CHORD_VOICINGS:
        return CHORD_VOICINGS[normalized]

    # Try just the root + basic quality
    root = chord_name[0]
    rest = chord_name[1:]
    if rest and rest[0] in ('#', 'b'):
        root += rest[0]
        rest = rest[1:]

    # Map complex chords to simpler versions we know
    quality = rest.lower()
    if 'maj7' in quality or 'M7' in rest:
        simple = root + 'maj7'
    elif 'min7' in quality or 'm7' in quality:
        simple = root + 'm7'
    elif '7' in quality:
        simple = root + '7'
    elif 'sus2' in quality:
        simple = root + 'sus2'
    elif 'sus4' in quality:
        simple = root + 'sus4'
    elif 'dim' in quality or 'o' == quality:
        # Use minor chord as approximation for dim
        simple = root + 'm'
    elif 'aug' in quality or '+' == quality:
        # Use major chord as approximation for aug
        simple = root
    elif 'min' in quality or 'm' == quality:
        simple = root + 'm'
    else:
        simple = root  # Just the major chord

    if simple in CHORD_VOICINGS:
        return CHORD_VOICINGS[simple]

    return None


class ChordToTabGenerator:
    """
    Generates guitar tablature MIDI from a chord progression + audio onset detection.

    This is a fallback for when audio-based transcription (Basic Pitch) fails.
    Instead of trying to detect individual notes from audio, it:
    1. Uses the already-detected chord progression (BTC/Essentia)
    2. Detects onset/strum events from the guitar audio
    3. Maps each onset to the chord active at that time
    4. Places the full chord voicing at each onset

    The result is a rhythm chart showing chord shapes at the right times --
    what guitarists actually use for rhythm playing.
    """

    def generate(self, audio_path: str, output_dir: str,
                 chord_progression: list,
                 tempo_hint: float = None) -> TabTranscriptionResult:
        """
        Generate chord-based guitar tablature.

        Args:
            audio_path: Path to guitar stem audio
            output_dir: Directory for output MIDI
            chord_progression: List of chord dicts with 'time', 'duration', 'chord'
            tempo_hint: Known tempo (optional)

        Returns:
            TabTranscriptionResult
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio not found: {audio_path}")

        if not chord_progression:
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='chord_to_tab', num_strings_used=0, fret_range=(0, 0),
            )

        logger.info(f"Chord-to-tab generation: {audio_path.name} ({len(chord_progression)} chords)")

        # --- Step 1: Detect onsets from the guitar audio ---
        try:
            import librosa
            audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
            onset_frames = librosa.onset.onset_detect(
                y=audio, sr=sr,
                hop_length=512,
                backtrack=True,
                units='frames',
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)
            logger.info(f"Detected {len(onset_times)} onsets from audio")
        except Exception as e:
            logger.warning(f"Onset detection failed ({e}), using beat positions")
            onset_times = None

        # --- Step 2: If onset detection failed, use beat positions ---
        if onset_times is None or len(onset_times) < 4:
            try:
                import librosa
                audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
                tempo_result = librosa.beat.beat_track(y=audio, sr=sr)
                if hasattr(tempo_result[0], '__len__'):
                    tempo = float(tempo_result[0][0])
                else:
                    tempo = float(tempo_result[0])
                beat_frames = tempo_result[1]
                onset_times = librosa.frames_to_time(beat_frames, sr=sr)
                logger.info(f"Using {len(onset_times)} beat positions (tempo={tempo:.0f})")
            except Exception:
                # Last resort: place chords at their start times
                onset_times = np.array([c.get('time', getattr(c, 'time', 0))
                                       for c in chord_progression])
                logger.info(f"Using {len(onset_times)} chord start times as onsets")

        # --- Step 3: Determine tempo ---
        if tempo_hint and 40 < tempo_hint < 300:
            tempo = tempo_hint
        else:
            try:
                import librosa
                if 'audio' not in dir() or audio is None:
                    audio, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE)
                tempo_result = librosa.beat.beat_track(y=audio, sr=sr)
                if hasattr(tempo_result[0], '__len__'):
                    tempo = float(tempo_result[0][0])
                else:
                    tempo = float(tempo_result[0])
                tempo = max(40.0, min(300.0, tempo))
            except Exception:
                tempo = 120.0

        # --- Step 4: Map each onset to a chord and generate notes ---
        tab_notes = []
        strings_used = set()
        frets_used = []
        chords_placed = 0
        prev_chord_name = None

        # Helper to look up chord at a time
        def _chord_at_time(t):
            for c in chord_progression:
                c_time = c.get('time', getattr(c, 'time', 0))
                c_dur = c.get('duration', getattr(c, 'duration', 0))
                if c_time <= t < c_time + c_dur:
                    return c
            return None

        # Compute a default note duration based on tempo
        beat_duration = 60.0 / max(tempo, 60)
        default_note_duration = beat_duration * 0.9  # Slightly less than a beat

        for onset_t in onset_times:
            chord_event = _chord_at_time(onset_t)
            if chord_event is None:
                continue

            chord_name = chord_event.get('chord', getattr(chord_event, 'chord', ''))
            if not chord_name or chord_name == 'N':
                continue

            voicing = _get_chord_voicing(chord_name)
            if voicing is None:
                continue

            # Determine note end time: next onset or default duration, whichever is shorter
            # Find next onset
            onset_idx = np.searchsorted(onset_times, onset_t)
            if onset_idx + 1 < len(onset_times):
                next_onset = onset_times[onset_idx + 1]
                note_duration = min(default_note_duration, (next_onset - onset_t) * 0.95)
            else:
                note_duration = default_note_duration

            note_duration = max(0.05, note_duration)  # At least 50ms
            note_end = onset_t + note_duration

            # Create notes for each string in the voicing
            for string_idx, fret in enumerate(voicing):
                if fret < 0:
                    continue  # Muted string

                midi_note = TUNING[string_idx] + fret
                # Add slight strum spread (low strings first, ~5ms between strings)
                strum_offset = string_idx * 0.005

                tab_notes.append({
                    'pitch': midi_note,
                    'string': string_idx,
                    'fret': fret,
                    'start': onset_t + strum_offset,
                    'end': note_end,
                    'velocity': 80,  # Consistent velocity for chord strums
                })

                strings_used.add(string_idx)
                frets_used.append(fret)

            chords_placed += 1
            prev_chord_name = chord_name

        if not tab_notes:
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='chord_to_tab', num_strings_used=0, fret_range=(0, 0),
            )

        logger.info(f"Chord-to-tab: {chords_placed} chords placed, {len(tab_notes)} notes")

        # --- Step 5: Write MIDI ---
        if not PRETTY_MIDI_AVAILABLE:
            logger.error("pretty_midi not available for MIDI output")
            return TabTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='chord_to_tab', num_strings_used=0, fret_range=(0, 0),
            )

        midi_out = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=25, name='Guitar')

        for tn in tab_notes:
            midi_note = pretty_midi.Note(
                velocity=tn['velocity'],
                pitch=tn['pitch'],
                start=tn['start'],
                end=max(tn['end'], tn['start'] + 0.03),
            )
            instrument.notes.append(midi_note)

        midi_out.instruments.append(instrument)

        midi_filename = f"{audio_path.stem}_chord_tab.mid"
        midi_path = output_dir / midi_filename
        midi_out.write(str(midi_path))

        # --- Step 6: Quality metrics ---
        # Chord-to-tab is inherently lower quality than note-level transcription
        # but is reliable and produces playable output
        duration = max(tn['end'] for tn in tab_notes) - min(tn['start'] for tn in tab_notes)
        chord_density = chords_placed / max(duration, 0.1)

        quality = 0.0
        quality += 0.25 if chords_placed >= 4 else 0.1   # Enough chords
        quality += 0.20 if len(strings_used) >= 3 else 0.05  # String diversity
        quality += 0.15 if 0.5 < chord_density < 8.0 else 0.0  # Sane density
        quality += 0.15 if len(set(c.get('chord', getattr(c, 'chord', ''))
                                   for c in chord_progression if c.get('chord', getattr(c, 'chord', '')) != 'N')) >= 2 else 0.05
        quality += 0.10  # Base confidence for chord-based approach
        quality = min(0.75, quality)  # Cap at 0.75 -- chord-to-tab isn't note-level

        logger.info(f"Chord-to-tab MIDI: {midi_path.name}, {len(tab_notes)} notes, "
                    f"{chords_placed} chords, tempo={tempo:.0f}, quality={quality:.2f}")

        return TabTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(tab_notes),
            quality_score=quality,
            method='chord_to_tab',
            num_strings_used=len(strings_used),
            fret_range=(min(frets_used), max(frets_used)),
        )


# ============================================================================
# CONVENIENCE FUNCTIONS (matches existing API exactly)
# ============================================================================

_transcriber: Optional[GuitarTabTranscriber] = None


def transcribe_guitar_tab(audio_path: str, output_dir: str,
                          tempo_hint: float = None,
                          chord_progression: list = None) -> Optional[str]:
    """
    Convenience function: transcribe guitar audio to MIDI.

    Tries Basic Pitch first; if quality is too low and chord_progression
    is available, falls back to chord-to-tab generation.

    Returns MIDI file path, or None if transcription failed/low quality.
    """
    global _transcriber

    if _transcriber is None:
        _transcriber = GuitarTabTranscriber()

    try:
        result = _transcriber.transcribe(
            audio_path=audio_path,
            output_dir=output_dir,
            tempo_hint=tempo_hint,
            chord_progression=chord_progression,
        )
        if result.midi_path and result.quality_score > 0.3:
            return result.midi_path

        # Basic Pitch failed or quality too low -- try chord-to-tab fallback
        if chord_progression:
            logger.info("Basic Pitch quality too low, trying chord-to-tab fallback")
            chord_gen = ChordToTabGenerator()
            chord_result = chord_gen.generate(
                audio_path=audio_path,
                output_dir=output_dir,
                chord_progression=chord_progression,
                tempo_hint=tempo_hint,
            )
            if chord_result.midi_path and chord_result.quality_score > 0.2:
                return chord_result.midi_path

        return None
    except Exception as e:
        logger.error(f"Guitar tab transcription failed: {e}")
        return None


def is_available() -> bool:
    """Check if guitar tab transcriber is available."""
    return MODEL_AVAILABLE


# Alias for consistent naming in app.py imports
GUITAR_TAB_MODEL_AVAILABLE = MODEL_AVAILABLE


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    print("Guitar Tab Transcriber (Basic Pitch + Post-Processing)")
    print(f"  Available: {MODEL_AVAILABLE}")
    print(f"  Basic Pitch: {BASIC_PITCH_AVAILABLE}")
    print(f"  Tuning: {TUNING}")
    print(f"  Fret range: 0-{NUM_FRETS}")

    if len(sys.argv) >= 3:
        audio_path = sys.argv[1]
        output_dir = sys.argv[2]

        transcriber = GuitarTabTranscriber()
        result = transcriber.transcribe(audio_path, output_dir)
        print(f"\nResult: {result}")
