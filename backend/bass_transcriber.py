"""
Neural Bass Transcriber for StemScriber
======================================
Uses Basic Pitch (Google/Spotify) for pitch detection, then applies
bass-specific post-processing for high-quality bass tablature MIDI.

Post-processing pipeline:
  1. Basic Pitch raw transcription
  2. Harmonic-analysis-based octave correction
  3. Bass range filtering (E1=28 to G4=67)
  4. Key detection and spurious note filtering
  5. Polyphony limiting (max 4 simultaneous -- rare for bass)
  6. String/fret assignment with hand position model
  7. Short note cleanup

Bass Tuning (Standard 4-string):
  String 0: E1 (MIDI 28)
  String 1: A1 (MIDI 33)
  String 2: D2 (MIDI 38)
  String 3: G2 (MIDI 43)
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Set
from dataclasses import dataclass
from collections import Counter

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
    logger.warning("basic_pitch not available — bass transcriber disabled")

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False

MODEL_AVAILABLE = BASIC_PITCH_AVAILABLE and PRETTY_MIDI_AVAILABLE

if MODEL_AVAILABLE:
    logger.info("Bass transcriber ready (Basic Pitch + post-processing)")


# ============================================================================
# CONSTANTS
# ============================================================================

SAMPLE_RATE = 22050
NUM_STRINGS = 4
NUM_FRETS = 24
TUNING = [28, 33, 38, 43]  # E1, A1, D2, G2

# Bass range limits
BASS_MIN_MIDI = 28   # E1 (low E open)
BASS_MAX_MIDI = 67   # G4 (fret 24 on G string)
MAX_POLYPHONY = 4    # 4-string bass, rarely plays chords

# Octave correction thresholds
BASS_OCTAVE_BOUNDARY = 55  # G3 -- most bass playing is below this
BASS_HIGH_OCTAVE_HARD = 60  # C4 -- almost certainly an octave too high

# Key detection (Krumhansl-Kessler profiles)
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


# ============================================================================
# RESULT DATACLASS
# ============================================================================

@dataclass
class BassTranscriptionResult:
    midi_path: Optional[str]
    num_notes: int
    quality_score: float
    method: str
    num_strings_used: int
    fret_range: Tuple[int, int]


# ============================================================================
# KEY DETECTION (bass-specific)
# ============================================================================

def _detect_bass_key(notes: List) -> Optional[Tuple[str, str]]:
    """
    Detect key from bass notes. Bass lines heavily emphasize root and fifth,
    making this quite reliable.
    """
    if not notes or len(notes) < 6:
        return None

    pc_histogram = np.zeros(12)
    for n in notes:
        pc = n.pitch % 12
        duration = max(0.01, n.end - n.start)
        weight = duration * (n.velocity / 127.0)
        pc_histogram[pc] += weight

    if pc_histogram.sum() == 0:
        return None

    pc_histogram = pc_histogram / pc_histogram.sum()

    best_key = None
    best_corr = -2.0

    major = np.array(MAJOR_PROFILE)
    minor = np.array(MINOR_PROFILE)

    for root in range(12):
        rotated = np.roll(pc_histogram, -root)
        corr_maj = np.corrcoef(rotated, major)[0, 1]
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = (NOTE_NAMES[root], 'major')
        corr_min = np.corrcoef(rotated, minor)[0, 1]
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = (NOTE_NAMES[root], 'minor')

    if best_corr < 0.25:
        return None

    return best_key


def _get_scale_pitches(key: Tuple[str, str]) -> Set[int]:
    """Get set of pitch classes belonging to a key's scale."""
    root_pc = NOTE_NAMES.index(key[0])
    if key[1] == 'major':
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else:
        intervals = [0, 2, 3, 5, 7, 8, 10]
    return {(root_pc + i) % 12 for i in intervals}


def _filter_bass_by_key(notes: List, key: Tuple[str, str]) -> List:
    """
    Remove low-confidence off-scale bass notes.
    Bass lines are more diatonic than guitar, so filtering is more aggressive
    but still preserves chromatic passing tones (short + high velocity).
    """
    scale_pcs = _get_scale_pitches(key)

    off_scale = sum(1 for n in notes if (n.pitch % 12) not in scale_pcs)
    off_ratio = off_scale / max(len(notes), 1)
    if off_ratio > 0.2:
        return notes  # Key detection likely wrong

    velocities = [n.velocity for n in notes]
    vel_threshold = np.percentile(velocities, 35) if velocities else 50

    filtered = []
    removed = 0
    for n in notes:
        pc = n.pitch % 12
        duration = n.end - n.start
        if pc not in scale_pcs and n.velocity < vel_threshold and duration < 0.15:
            removed += 1
            continue
        filtered.append(n)

    if removed > 0:
        logger.info(f"Bass key filter ({key[0]} {key[1]}): removed {removed} spurious notes")

    return filtered


# ============================================================================
# STRING/FRET ASSIGNMENT (with hand position model)
# ============================================================================

def _assign_bass_string_fret(midi_note: int, prev_fret: int = None,
                             prev_string: int = None,
                             hand_position: float = None) -> Optional[Tuple[int, int]]:
    """
    Assign a MIDI note to (string, fret) on bass guitar.

    Enhanced with hand position model for better continuity during
    bass runs and position shifts.
    """
    candidates = []

    for s_idx, open_note in enumerate(TUNING):
        fret = midi_note - open_note
        if 0 <= fret <= NUM_FRETS:
            score = 0.0

            # Prefer lower frets (bass players live in positions 0-7)
            if fret == 0:
                score += 1
            elif fret <= 5:
                score += 0
            elif fret <= 9:
                score += 4
            elif fret <= 12:
                score += 10
            else:
                score += 18

            # Bass players tend to use lower strings more
            score += s_idx * 1.5

            # Hand position proximity
            if hand_position is not None:
                pos_distance = abs(fret - hand_position)
                if pos_distance <= 2:
                    score -= 5  # Within hand span -- very natural
                elif pos_distance <= 4:
                    score += 1
                else:
                    score += pos_distance * 2

            # Hand position continuity
            if prev_fret is not None:
                fret_jump = abs(fret - prev_fret)
                if fret_jump == 0:
                    score -= 4  # Same fret -- very natural for bass
                elif fret_jump <= 2:
                    score += 0
                elif fret_jump <= 4:
                    score += 5
                else:
                    score += fret_jump * 2.5

            # String continuity
            if prev_string is not None:
                string_jump = abs(s_idx - prev_string)
                if string_jump == 0:
                    score -= 3  # Same string -- natural for bass runs
                elif string_jump == 1:
                    score += 0
                else:
                    score += string_jump * 4

            candidates.append((s_idx, fret, score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2])
    return (candidates[0][0], candidates[0][1])


# ============================================================================
# POST-PROCESSING
# ============================================================================

def _filter_bass_range(notes: List) -> List:
    """Remove notes outside bass range."""
    return [n for n in notes if BASS_MIN_MIDI <= n.pitch <= BASS_MAX_MIDI]


def _octave_correct(notes: List) -> List:
    """
    Correct notes that Basic Pitch detected an octave too high using
    harmonic analysis rather than a simple threshold.

    Strategy:
      1. Build a pitch-class histogram of the lower notes (likely correct)
      2. For notes above the boundary, check if shifting down an octave
         creates a pitch class more consistent with the overall distribution
      3. Notes well above the hard boundary are always shifted
      4. Notes in the boundary zone use statistical evidence

    This handles cases where a bass part legitimately plays above G3
    (e.g., fills, chords) while still catching octave errors.
    """
    if not notes:
        return []

    # Build pitch-class histogram from notes that are definitely in bass range
    # (below boundary = reliable)
    low_pc_counts = Counter()
    for n in notes:
        if n.pitch <= BASS_OCTAVE_BOUNDARY:
            low_pc_counts[n.pitch % 12] += 1

    total_low = sum(low_pc_counts.values())

    corrected = []
    shifts = 0

    for note in notes:
        if note.pitch > BASS_HIGH_OCTAVE_HARD and (note.pitch - 12) >= BASS_MIN_MIDI:
            # Well above typical bass range -- almost certainly octave error
            note.pitch -= 12
            shifts += 1
        elif note.pitch > BASS_OCTAVE_BOUNDARY and (note.pitch - 12) >= BASS_MIN_MIDI:
            # In the boundary zone -- use harmonic evidence
            pc_original = note.pitch % 12

            if total_low > 0:
                # Check if this pitch class is common among the reliable low notes
                # If so, this pitch class is "expected" and less likely an error
                pc_freq = low_pc_counts.get(pc_original, 0) / total_low

                # If the pitch class rarely appears in low range, it's more
                # likely to be a harmonic ghost -- shift it down
                if pc_freq < 0.05:
                    note.pitch -= 12
                    shifts += 1
                # Otherwise keep it -- it's a legitimate high note
            else:
                # No low notes for reference -- use conservative threshold
                note.pitch -= 12
                shifts += 1

        corrected.append(note)

    if shifts > 0:
        logger.info(f"Octave correction: shifted {shifts} notes down an octave")

    return corrected


def _remove_short_notes(notes: List, min_duration: float = 0.04) -> List:
    """Remove very short notes. Bass notes are typically longer than guitar."""
    return [n for n in notes if (n.end - n.start) >= min_duration]


def _limit_polyphony(notes: List, max_voices: int = MAX_POLYPHONY) -> List:
    """Limit simultaneous notes (bass rarely plays chords)."""
    if not notes:
        return notes

    notes_sorted = sorted(notes, key=lambda n: n.start)
    kept = []

    for note in notes_sorted:
        active = [n for n in kept if n.end > note.start + 0.005]
        if len(active) < max_voices:
            kept.append(note)
        else:
            weakest = min(active, key=lambda n: n.velocity)
            if note.velocity > weakest.velocity:
                kept.remove(weakest)
                kept.append(note)

    return kept


def _fix_overlapping_same_pitch(notes: List) -> List:
    """Fix overlapping notes on the same pitch."""
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
                    note.end = next_note.start - 0.005
                    if note.end <= note.start:
                        continue
            cleaned.append(note)

    return cleaned


# ============================================================================
# TRANSCRIBER CLASS
# ============================================================================

class BassTranscriber:
    """
    Transcribes bass audio to MIDI using Basic Pitch + bass post-processing.
    """

    def transcribe(self, audio_path: str, output_dir: str,
                   tempo_hint: float = None) -> BassTranscriptionResult:
        """
        Full bass transcription pipeline.

        1. Run Basic Pitch with bass-optimized settings
        2. Octave correction (fix harmonic confusion)
        3. Filter to bass range (E1-G4)
        4. Clean up short notes and overlaps
        5. Limit polyphony to 4
        6. Assign string/fret positions
        7. Write MIDI

        Args:
            audio_path: Path to bass audio file
            output_dir: Output directory for MIDI
            tempo_hint: Known tempo (optional)

        Returns:
            BassTranscriptionResult
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not MODEL_AVAILABLE:
            logger.warning("Basic Pitch not available for bass transcription")
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_bass', num_strings_used=0,
                fret_range=(0, 0),
            )

        logger.info(f"Transcribing bass: {audio_path.name}")

        # --- Step 1: Basic Pitch transcription ---
        # Tuned for bass: lower frequency range, slightly higher thresholds
        # since bass is typically more monophonic
        try:
            # Thresholds tuned for separated bass stems: cleaner signal
            # allows lower thresholds. Bass attacks are still clear so onset
            # stays moderate, but frame threshold lowered for full sustain capture.
            model_output, midi_data, note_events = predict(
                str(audio_path),
                onset_threshold=0.45,       # Slightly lower for separated stems
                frame_threshold=0.30,       # Capture full sustain on clean signal
                minimum_note_length=60,     # 60ms min -- bass notes are longer
                minimum_frequency=30.0,     # ~B0 (31 Hz) -- catches low bass
                maximum_frequency=500.0,    # ~B4 -- bass fundamental range
            )
        except Exception as e:
            logger.error(f"Basic Pitch failed: {e}")
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_bass', num_strings_used=0,
                fret_range=(0, 0),
            )

        if midi_data is None or not midi_data.instruments:
            logger.info("Basic Pitch produced no notes for bass")
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_bass', num_strings_used=0,
                fret_range=(0, 0),
            )

        raw_notes = midi_data.instruments[0].notes
        logger.info(f"Basic Pitch raw: {len(raw_notes)} notes")

        # --- Step 2: Bass-specific post-processing ---
        notes = _octave_correct(raw_notes)
        notes = _filter_bass_range(notes)
        logger.info(f"After range filter + octave correction: {len(notes)} notes")

        notes = _remove_short_notes(notes, min_duration=0.04)
        notes = _fix_overlapping_same_pitch(notes)
        notes = _limit_polyphony(notes, MAX_POLYPHONY)
        logger.info(f"After cleanup: {len(notes)} notes")

        if not notes:
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_bass', num_strings_used=0,
                fret_range=(0, 0),
            )

        # --- Step 2b: Key detection and spurious note filtering ---
        detected_key = _detect_bass_key(notes)
        if detected_key:
            logger.info(f"Detected bass key: {detected_key[0]} {detected_key[1]}")
            notes = _filter_bass_by_key(notes, detected_key)
            logger.info(f"After key filter: {len(notes)} notes")

        if not notes:
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_bass', num_strings_used=0,
                fret_range=(0, 0),
            )

        # --- Step 3: Assign string/fret positions with hand position model ---
        notes.sort(key=lambda n: n.start)

        tab_notes = []
        prev_fret = None
        prev_string = None
        strings_used = set()
        frets_used = []

        # Hand position model: weighted moving average of recent frets
        position_history = []
        hand_position = None

        for note in notes:
            result = _assign_bass_string_fret(note.pitch, prev_fret, prev_string, hand_position)
            if result is not None:
                s_idx, fret = result
                strings_used.add(s_idx)
                frets_used.append(fret)
                prev_fret = fret
                prev_string = s_idx

                # Update hand position model
                position_history.append(fret)
                if len(position_history) > 6:
                    position_history.pop(0)
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

        if not tab_notes:
            return BassTranscriptionResult(
                midi_path=None, num_notes=0, quality_score=0.0,
                method='basic_pitch_bass', num_strings_used=0,
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
        bass_track = pretty_midi.Instrument(
            program=33, is_drum=False, name='Electric Bass'
        )

        for tn in tab_notes:
            midi_note = pretty_midi.Note(
                velocity=tn['velocity'],
                pitch=tn['pitch'],
                start=tn['start'],
                end=max(tn['end'], tn['start'] + 0.04),
            )
            bass_track.notes.append(midi_note)

        midi_out.instruments.append(bass_track)

        midi_filename = f"{audio_path.stem}_bass.mid"
        midi_path = output_dir / midi_filename
        midi_out.write(str(midi_path))

        # --- Step 6: Quality metrics ---
        duration = max(n['end'] for n in tab_notes) - min(n['start'] for n in tab_notes)
        note_density = len(tab_notes) / max(duration, 0.1)
        fret_range = (min(frets_used), max(frets_used))

        quality = 0.0
        quality += 0.25 * min(1.0, len(tab_notes) / (duration * 2))
        quality += 0.2 * min(1.0, len(strings_used) / 3)
        quality += 0.2 if 0.5 < note_density < 10.0 else 0.0
        quality += 0.2 if fret_range[1] - fret_range[0] > 2 else 0.1
        quality += 0.15
        quality = min(1.0, quality)

        logger.info(f"Bass MIDI: {midi_path.name}, {len(tab_notes)} notes, "
                    f"strings={sorted(strings_used)}, frets={fret_range}, "
                    f"tempo={tempo:.0f}, quality={quality:.2f}")

        return BassTranscriptionResult(
            midi_path=str(midi_path),
            num_notes=len(tab_notes),
            quality_score=quality,
            method='basic_pitch_bass',
            num_strings_used=len(strings_used),
            fret_range=fret_range,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_transcriber: Optional[BassTranscriber] = None


def transcribe_bass(audio_path: str, output_dir: str,
                    tempo_hint: float = None) -> Optional[str]:
    """Convenience function: returns MIDI path or None."""
    global _transcriber

    if _transcriber is None:
        _transcriber = BassTranscriber()

    try:
        result = _transcriber.transcribe(audio_path, output_dir, tempo_hint)
        if result.midi_path and result.quality_score > 0.2:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Bass transcription failed: {e}")
        return None


def is_available() -> bool:
    return MODEL_AVAILABLE


BASS_MODEL_AVAILABLE = MODEL_AVAILABLE
