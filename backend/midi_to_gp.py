"""
MIDI to Guitar Pro Converter

Converts MIDI files to Guitar Pro 5 format (.gp5) for proper tablature notation.
Uses pyguitarpro for GP file creation and mido for MIDI parsing.

Features:
- Automatic instrument detection (guitar, bass, drums, piano, vocals)
- Proper tuning assignment based on instrument type
- Note-to-fret mapping with intelligent position selection
- Tempo and time signature preservation
- Articulation hints from velocity
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Standard tunings (low to high string, in MIDI note numbers)
TUNINGS = {
    'guitar': [40, 45, 50, 55, 59, 64],      # E2 A2 D3 G3 B3 E4 (standard)
    'guitar_drop_d': [38, 45, 50, 55, 59, 64],  # D2 A2 D3 G3 B3 E4
    'bass': [28, 33, 38, 43],                 # E1 A1 D2 G2 (4-string)
    'bass_5': [23, 28, 33, 38, 43],           # B0 E1 A1 D2 G2 (5-string)
}

# Fret range
MAX_FRET = 24


@dataclass
class GPNote:
    """Represents a note for Guitar Pro"""
    string: int      # 1-indexed string number
    fret: int        # Fret number (0 = open)
    start_beat: float
    duration_beats: float
    velocity: int
    is_ghost: bool = False
    is_hammer: bool = False
    is_slide: bool = False


def midi_note_to_fret(midi_note: int, tuning: List[int],
                      prev_fret: int = None, prev_string: int = None) -> Optional[Tuple[int, int]]:
    """
    Convert MIDI note to (string, fret) for given tuning.
    Uses context-aware positioning for realistic playability.

    Args:
        midi_note: MIDI note number
        tuning: List of open string MIDI notes (low to high)
        prev_fret: Previous note's fret (for position continuity)
        prev_string: Previous note's string (for hand position)

    Returns:
        (string_number, fret) - string is 1-indexed
    """
    candidates = []

    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= MAX_FRET:
            string_num = string_idx + 1

            # Base score components
            score = 0

            # 1. Position zone preference (prefer positions 1-7 for most playing)
            if fret == 0:
                score += 5  # Open strings are easy
            elif 1 <= fret <= 4:
                score += 0  # First position - most comfortable
            elif 5 <= fret <= 7:
                score += 3  # Second position - still easy
            elif 8 <= fret <= 12:
                score += 8  # Third position - need to shift
            else:
                score += 15  # High frets - harder to play

            # 2. String preference (middle strings easier for single notes)
            # For 6-string: prefer strings 2-5 (indices 1-4)
            mid_string = len(tuning) // 2
            string_distance = abs(string_idx - mid_string)
            score += string_distance * 2

            # 3. Hand position continuity (KEY FOR PLAYABILITY)
            if prev_fret is not None:
                fret_jump = abs(fret - prev_fret)
                if fret_jump == 0:
                    score -= 3  # Same fret - very easy
                elif fret_jump <= 2:
                    score += 0  # Within hand span - natural
                elif fret_jump <= 4:
                    score += 5  # Small shift
                elif fret_jump <= 7:
                    score += 12  # Position shift required
                else:
                    score += 20  # Large jump - avoid if possible

            # 4. String continuity (avoid large string jumps)
            if prev_string is not None:
                string_jump = abs(string_num - prev_string)
                if string_jump == 0:
                    score -= 2  # Same string - natural for runs
                elif string_jump == 1:
                    score += 0  # Adjacent string - easy
                elif string_jump == 2:
                    score += 3  # Skip one string
                else:
                    score += string_jump * 3  # Large jumps harder

            # 5. Bendability (can't bend open strings, hard to bend above fret 15)
            # Not penalizing here, but could add articulation awareness

            candidates.append((string_num, fret, score))

    if not candidates:
        return None

    # Return best candidate (lowest score)
    candidates.sort(key=lambda x: x[2])
    return (candidates[0][0], candidates[0][1])


class FretMapper:
    """
    Intelligent fret position mapper that maintains hand position context
    across a sequence of notes.
    """

    def __init__(self, tuning: List[int]):
        self.tuning = tuning
        self.current_position = 0  # Current hand position (fret)
        self.prev_fret = None
        self.prev_string = None

    def map_note(self, midi_note: int) -> Optional[Tuple[int, int]]:
        """Map a single note, updating position context."""
        result = midi_note_to_fret(
            midi_note, self.tuning,
            prev_fret=self.prev_fret,
            prev_string=self.prev_string
        )

        if result:
            self.prev_string, self.prev_fret = result
            # Update approximate hand position
            self.current_position = max(0, self.prev_fret - 2)

        return result

    def map_chord(self, midi_notes: List[int]) -> List[Tuple[int, int]]:
        """
        Map a chord (simultaneous notes) to fret positions.
        Ensures all notes are playable together within a hand span.
        """
        if not midi_notes:
            return []

        # Sort notes low to high
        sorted_notes = sorted(midi_notes)

        # Find all possible positions for each note
        all_candidates = []
        for note in sorted_notes:
            note_candidates = []
            for string_idx, open_note in enumerate(self.tuning):
                fret = note - open_note
                if 0 <= fret <= MAX_FRET:
                    note_candidates.append((string_idx + 1, fret))
            all_candidates.append(note_candidates)

        if not all(all_candidates):
            # At least one note has no valid position
            return []

        # Find combination where:
        # 1. Each note on different string
        # 2. Fret span <= 4 (or 5 with stretch)
        # 3. Prefer lower positions

        best_combo = None
        best_score = float('inf')

        def find_combos(note_idx, used_strings, current_combo):
            nonlocal best_combo, best_score

            if note_idx == len(sorted_notes):
                # Evaluate this combination
                frets = [f for _, f in current_combo]
                non_open = [f for f in frets if f > 0]

                if non_open:
                    fret_span = max(non_open) - min(non_open)
                    if fret_span > 5:  # Too wide for one hand
                        return

                    # Score: prefer lower positions, smaller span
                    score = min(non_open) + fret_span * 2
                else:
                    score = 0  # All open strings

                if score < best_score:
                    best_score = score
                    best_combo = list(current_combo)
                return

            for string, fret in all_candidates[note_idx]:
                if string not in used_strings:
                    find_combos(
                        note_idx + 1,
                        used_strings | {string},
                        current_combo + [(string, fret)]
                    )

        find_combos(0, set(), [])

        if best_combo:
            # Update context with the lowest note's position
            if best_combo:
                self.prev_string = best_combo[0][0]
                self.prev_fret = best_combo[0][1]

        return best_combo or []

    def reset(self):
        """Reset position context (e.g., at section boundaries)."""
        self.prev_fret = None
        self.prev_string = None
        self.current_position = 0


def get_tuning_for_instrument(instrument: str) -> Tuple[List[int], int]:
    """Get appropriate tuning and string count for instrument type"""
    instrument = instrument.lower()

    if 'bass' in instrument:
        return TUNINGS['bass'], 4
    elif 'guitar' in instrument:
        return TUNINGS['guitar'], 6
    else:
        # Default to guitar for melodic instruments
        return TUNINGS['guitar'], 6


def convert_midi_to_gp(midi_path: str, output_path: str,
                       instrument_type: str = 'guitar',
                       title: str = None, artist: str = None) -> bool:
    """
    Convert a MIDI file to Guitar Pro 5 format.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path for output .gp5 file
        instrument_type: Type of instrument (guitar, bass, drums, piano, vocals)
        title: Song title (optional)
        artist: Artist name (optional)

    Returns:
        True if successful, False otherwise
    """
    try:
        import mido
        import guitarpro
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        logger.info("Install with: pip install mido pyguitarpro")
        return False

    try:
        # Load MIDI
        midi = mido.MidiFile(midi_path)
        logger.info(f"Loaded MIDI: {midi_path} ({len(midi.tracks)} tracks, {midi.ticks_per_beat} ticks/beat)")

        # Create new GP song
        song = guitarpro.models.Song()
        song.title = title or Path(midi_path).stem
        song.artist = artist or ''

        # Get tempo from MIDI
        tempo = 120  # Default
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    tempo = int(mido.tempo2bpm(msg.tempo))
                    break

        song.tempo = tempo
        logger.info(f"Tempo: {tempo} BPM")

        # Get tuning for instrument
        tuning_notes, num_strings = get_tuning_for_instrument(instrument_type)

        # Use the default track (Song() creates one) - don't create a new one
        gp_track = song.tracks[0]
        gp_track.name = instrument_type.replace('_', ' ').title()
        gp_track.channel.instrument = get_gp_instrument(instrument_type)
        gp_track.isPercussionTrack = 'drum' in instrument_type.lower()

        # Clear default measureHeaders and measures - we'll create our own
        song.measureHeaders.clear()
        gp_track.measures.clear()

        # Set up strings/tuning (newer pyguitarpro requires number and value args)
        gp_track.strings = []
        for i, note in enumerate(reversed(tuning_notes)):  # GP uses high-to-low
            string = guitarpro.models.GuitarString(number=i+1, value=note)
            gp_track.strings.append(string)

        # Collect all notes from MIDI
        notes = []
        current_time = 0

        for track in midi.tracks:
            current_time = 0
            active_notes = {}  # note_num -> (start_time, velocity)

            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = (current_time, msg.velocity)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start_time, velocity = active_notes.pop(msg.note)
                        duration = current_time - start_time

                        # Convert time to beats
                        start_beat = start_time / midi.ticks_per_beat
                        duration_beats = max(0.25, duration / midi.ticks_per_beat)  # Min 16th note

                        notes.append({
                            'midi_note': msg.note,
                            'start_beat': start_beat,
                            'duration_beats': duration_beats,
                            'velocity': velocity
                        })

        if not notes:
            logger.warning("No notes found in MIDI file")
            return False

        # Sort by start time
        notes.sort(key=lambda x: x['start_beat'])
        logger.info(f"Found {len(notes)} notes")

        # Group notes into measures (4/4 assumed)
        beats_per_measure = 4
        measures_needed = int(notes[-1]['start_beat'] / beats_per_measure) + 2

        # Create measures
        for i in range(measures_needed):
            measure_header = guitarpro.models.MeasureHeader()
            # Set tempo if attribute exists (API varies by version)
            if hasattr(measure_header, 'tempo'):
                measure_header.tempo.value = tempo
            # Set time signature if attributes exist
            if hasattr(measure_header, 'timeSignature'):
                measure_header.timeSignature.numerator = 4
                measure_header.timeSignature.denominator.value = 4
            song.measureHeaders.append(measure_header)

            measure = guitarpro.models.Measure(gp_track, measure_header)
            gp_track.measures.append(measure)

        # Create FretMapper for intelligent position tracking
        fret_mapper = FretMapper(tuning_notes)

        # Add notes to measures
        for note_data in notes:
            measure_idx = int(note_data['start_beat'] / beats_per_measure)
            if measure_idx >= len(gp_track.measures):
                continue

            measure = gp_track.measures[measure_idx]
            beat_in_measure = note_data['start_beat'] % beats_per_measure

            # Convert MIDI note to fret position with hand position awareness
            if 'drum' in instrument_type.lower():
                # Drums: use voice for different drums
                fret_pos = (1, note_data['midi_note'] % 6)  # Simplified drum mapping
            else:
                # Use FretMapper for context-aware fret selection
                fret_pos = fret_mapper.map_note(note_data['midi_note'])

            if fret_pos is None:
                continue  # Note out of range

            string_num, fret = fret_pos

            # Create beat and note
            voice = measure.voices[0]

            # Find or create beat at this position
            beat = guitarpro.models.Beat(voice)
            beat.status = guitarpro.models.BeatStatus.normal  # CRITICAL: must be normal, not empty!
            beat.start = int(beat_in_measure * 960)  # GP uses 960 ticks per beat

            # Set duration
            if note_data['duration_beats'] >= 4:
                beat.duration.value = 1  # Whole
            elif note_data['duration_beats'] >= 2:
                beat.duration.value = 2  # Half
            elif note_data['duration_beats'] >= 1:
                beat.duration.value = 4  # Quarter
            elif note_data['duration_beats'] >= 0.5:
                beat.duration.value = 8  # Eighth
            else:
                beat.duration.value = 16  # Sixteenth

            # Create note
            gp_note = guitarpro.models.Note(beat)
            gp_note.string = string_num
            gp_note.value = fret
            gp_note.velocity = note_data['velocity']

            # Add articulation hints based on velocity
            if note_data['velocity'] < 50:
                gp_note.effect.ghostNote = True
            elif note_data['velocity'] > 110:
                gp_note.effect.accentuatedNote = True

            beat.notes.append(gp_note)
            voice.beats.append(beat)

        # Fill empty measures with rest beats (AlphaTab requires beats in every measure)
        for measure in gp_track.measures:
            voice = measure.voices[0]
            if not voice.beats:
                # Add a whole rest
                rest_beat = guitarpro.models.Beat(voice)
                rest_beat.status = guitarpro.models.BeatStatus.rest  # Mark as rest
                rest_beat.duration.value = 1  # Whole note duration
                rest_beat.notes = []  # Empty = rest
                voice.beats.append(rest_beat)

        # Note: We use the default track, so no need to append

        # Save GP5 file
        output_path = str(output_path)
        if not output_path.endswith('.gp5'):
            output_path += '.gp5'

        guitarpro.write(song, output_path)
        logger.info(f"✓ Created Guitar Pro file: {output_path}")

        return True

    except Exception as e:
        logger.error(f"MIDI to GP conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def get_gp_instrument(instrument_type: str) -> int:
    """Get Guitar Pro instrument number for instrument type"""
    instrument_type = instrument_type.lower()

    if 'bass' in instrument_type:
        return 33  # Electric Bass (finger)
    elif 'guitar' in instrument_type:
        return 25  # Acoustic Guitar (steel) - or 27 for Electric
    elif 'piano' in instrument_type:
        return 0   # Acoustic Grand Piano
    elif 'drum' in instrument_type:
        return 0   # Drums (channel 10)
    elif 'vocal' in instrument_type:
        return 52  # Choir Aahs
    else:
        return 25  # Default to guitar


def convert_job_midis_to_gp(job, output_dir: Path) -> Dict[str, str]:
    """
    Convert all MIDI files in a job to Guitar Pro format.

    Args:
        job: ProcessingJob with midi_files dict
        output_dir: Directory to save GP files

    Returns:
        Dict mapping stem_name to GP file path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gp_files = {}
    title = job.metadata.get('title', 'Untitled')
    artist = job.metadata.get('artist', '')

    for stem_name, midi_path in job.midi_files.items():
        # Determine instrument type from stem name
        instrument_type = stem_name.lower().split('_')[0]  # guitar_left -> guitar

        # Skip non-melodic stems that don't translate well to GP
        if instrument_type in ['other', 'instrumental']:
            logger.info(f"Skipping GP conversion for {stem_name} (mixed content)")
            continue

        gp_path = output_dir / f"{stem_name}.gp5"

        if convert_midi_to_gp(
            midi_path=midi_path,
            output_path=str(gp_path),
            instrument_type=instrument_type,
            title=f"{title} - {stem_name.replace('_', ' ').title()}",
            artist=artist
        ):
            gp_files[stem_name] = str(gp_path)
        else:
            logger.warning(f"GP conversion failed for {stem_name}")

    return gp_files


# Quick test
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python midi_to_gp.py <midi_file> [instrument_type]")
        sys.exit(1)

    midi_file = sys.argv[1]
    instrument = sys.argv[2] if len(sys.argv) > 2 else 'guitar'
    output = Path(midi_file).stem + '.gp5'

    success = convert_midi_to_gp(midi_file, output, instrument)
    print(f"Conversion {'succeeded' if success else 'failed'}: {output}")
