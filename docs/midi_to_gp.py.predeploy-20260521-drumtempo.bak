"""
MIDI to Guitar Pro Converter

Converts MIDI files to Guitar Pro 5 format (.gp5) for proper tablature notation.
Uses pyguitarpro for GP file creation and mido for MIDI parsing.

Features:
- Automatic instrument detection (guitar, bass, drums, piano, vocals)
- Proper tuning assignment based on instrument type
- A*-Guitar search for optimal fret position assignment (minimizes hand movement)
- Lead mode: bend-aware fret selection (prefer frets 5-15 for bends)
- Tempo and time signature preservation
- Articulation effects: bends, slides, hammer-ons, vibrato (from MIDI pitch bend + CC#20)
"""

import heapq
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

# A*-Guitar search parameters
HAND_SPAN = 4           # Comfortable fret span without shifting
POSITION_WEIGHT = 1.0   # Cost multiplier for fret distance
STRING_WEIGHT = 0.5     # Cost multiplier for string jumps
BEND_PENALTY = 20       # Penalty for bend on open/fret-1
LOOKAHEAD = 12          # Notes to look ahead in A* search window


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


def _get_candidates(midi_note: int, tuning: List[int]) -> List[Tuple[int, int]]:
    """Return all valid (string, fret) pairs for a MIDI note on the given tuning.
    String numbers are 1-indexed."""
    candidates = []
    for string_idx, open_note in enumerate(tuning):
        fret = midi_note - open_note
        if 0 <= fret <= MAX_FRET:
            candidates.append((string_idx + 1, fret))
    return candidates


def _transition_cost(prev_string: int, prev_fret: int,
                     cur_string: int, cur_fret: int,
                     has_bend: bool = False) -> float:
    """
    Cost of moving from one fret position to another.
    Models real guitar ergonomics: fret distance, string jumps, bend constraints.
    """
    cost = 0.0

    # Fret distance — primary driver of hand movement
    fret_delta = abs(cur_fret - prev_fret)
    if fret_delta == 0:
        cost += 0.0
    elif fret_delta <= HAND_SPAN:
        cost += fret_delta * POSITION_WEIGHT
    else:
        # Penalize shifts beyond hand span quadratically
        cost += HAND_SPAN * POSITION_WEIGHT + (fret_delta - HAND_SPAN) ** 2 * POSITION_WEIGHT

    # String jump cost
    string_delta = abs(cur_string - prev_string)
    if string_delta == 0:
        cost += 0.0   # Same string — no extra cost
    elif string_delta == 1:
        cost += 0.5 * STRING_WEIGHT
    else:
        cost += string_delta * STRING_WEIGHT

    # Bend constraints
    if has_bend:
        if cur_fret < 2:
            cost += BEND_PENALTY  # Can't bend open or fret 1
        elif cur_fret > 17:
            cost += 10  # Hard to bend above fret 17

    return cost


def _position_cost(string: int, fret: int, num_strings: int,
                   has_bend: bool = False) -> float:
    """Intrinsic cost of a fret position (no context). Slight preference for
    comfortable zones but much lighter than the old heuristic — the A* path
    optimizer handles most of the work."""
    cost = 0.0

    # Slight zone preference
    if fret == 0:
        cost += 1.0   # Open string — easy but can't bend
    elif 1 <= fret <= 7:
        cost += 0.0   # First/second position
    elif 8 <= fret <= 12:
        cost += 1.0   # Mid-neck
    else:
        cost += 2.0   # Upper frets

    # Bend awareness
    if has_bend:
        if fret < 2:
            cost += BEND_PENALTY
        elif fret > 17:
            cost += 10

    return cost


def midi_note_to_fret(midi_note: int, tuning: List[int],
                      prev_fret: int = None, prev_string: int = None) -> Optional[Tuple[int, int]]:
    """
    Convert MIDI note to (string, fret) for given tuning.
    Single-note greedy fallback used when A* path is not available.

    Args:
        midi_note: MIDI note number
        tuning: List of open string MIDI notes (low to high)
        prev_fret: Previous note's fret (for position continuity)
        prev_string: Previous note's string (for hand position)

    Returns:
        (string_number, fret) - string is 1-indexed
    """
    candidates = _get_candidates(midi_note, tuning)
    if not candidates:
        return None

    num_strings = len(tuning)

    def score(c):
        s, f = c
        base = _position_cost(s, f, num_strings)
        if prev_fret is not None and prev_string is not None:
            base += _transition_cost(prev_string, prev_fret, s, f)
        return base

    best = min(candidates, key=score)
    return best


def astar_fret_search(midi_notes: List[int], tuning: List[int],
                      bend_flags: List[bool] = None,
                      window: int = LOOKAHEAD) -> List[Optional[Tuple[int, int]]]:
    """
    A*-Guitar: find the minimum-cost path of fret assignments for a note sequence.

    Treats the sequence as a graph where each layer is the set of candidate
    (string, fret) positions for one note, and edges are weighted by hand-
    movement cost. Uses A* with a lookahead window to keep complexity bounded
    while still producing globally-aware assignments.

    Args:
        midi_notes: Sequence of MIDI note numbers
        tuning: Open string pitches (low to high)
        bend_flags: Per-note flag indicating if note has a bend articulation
        window: How many notes to optimize together in each A* window

    Returns:
        List of (string, fret) tuples (or None for unmappable notes)
    """
    if not midi_notes:
        return []

    n = len(midi_notes)
    num_strings = len(tuning)

    if bend_flags is None:
        bend_flags = [False] * n

    # Pre-compute candidates for every note
    all_candidates = [_get_candidates(note, tuning) for note in midi_notes]

    results: List[Optional[Tuple[int, int]]] = [None] * n

    # Process in overlapping windows
    start = 0
    prev_state = None  # (string, fret) from previous window's last assigned note

    while start < n:
        end = min(start + window, n)

        # Skip notes with no candidates
        segment_indices = []
        segment_candidates = []
        segment_bends = []
        for i in range(start, end):
            if all_candidates[i]:
                segment_indices.append(i)
                segment_candidates.append(all_candidates[i])
                segment_bends.append(bend_flags[i])

        if not segment_indices:
            start = end
            continue

        # A* search through this segment
        # State: index into segment_candidates, chosen (string, fret)
        # Priority queue: (cost, segment_idx, string, fret, path)

        best_path = _astar_segment(
            segment_candidates, segment_bends, num_strings, prev_state
        )

        # Write results
        for idx, (seg_i, assignment) in enumerate(zip(segment_indices, best_path)):
            results[seg_i] = assignment

        # Carry forward the last assigned position for continuity
        if best_path:
            prev_state = best_path[-1]

        # Overlap: step back a bit so the next window has context
        overlap = min(3, len(segment_indices))
        start = end

    return results


def _astar_segment(candidates: List[List[Tuple[int, int]]],
                   bend_flags: List[bool],
                   num_strings: int,
                   prev_state: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Run A* search over a segment of notes to find the minimum-cost path.

    Each node is (layer_index, string, fret). Edges connect adjacent layers.
    Uses a simple admissible heuristic: remaining layers * minimum possible
    transition cost (0).

    Returns list of (string, fret) assignments for the segment.
    """
    seg_len = len(candidates)
    if seg_len == 0:
        return []

    # For single note, just pick best intrinsic cost
    if seg_len == 1:
        best = min(candidates[0],
                   key=lambda c: _position_cost(c[0], c[1], num_strings, bend_flags[0]) +
                   (_transition_cost(prev_state[0], prev_state[1], c[0], c[1], bend_flags[0])
                    if prev_state else 0))
        return [best]

    # Priority queue: (total_est_cost, cost_so_far, layer, string, fret, path_as_tuple)
    # Using a counter to break ties deterministically
    counter = 0
    pq = []

    # Initialize with all candidates for first note
    for s, f in candidates[0]:
        base = _position_cost(s, f, num_strings, bend_flags[0])
        if prev_state:
            base += _transition_cost(prev_state[0], prev_state[1], s, f, bend_flags[0])
        est = base  # Heuristic for remaining = 0 (admissible)
        heapq.heappush(pq, (est, base, 0, s, f, ((s, f),), counter))
        counter += 1

    # Track best cost to reach each (layer, string, fret) to prune
    best_cost = {}

    while pq:
        est_total, cost_so_far, layer, cur_s, cur_f, path, _ = heapq.heappop(pq)

        # Goal: reached last layer
        if layer == seg_len - 1:
            return list(path)

        # Prune: if we've already reached this state cheaper, skip
        state_key = (layer, cur_s, cur_f)
        if state_key in best_cost and best_cost[state_key] <= cost_so_far:
            continue
        best_cost[state_key] = cost_so_far

        # Expand to next layer
        next_layer = layer + 1
        for ns, nf in candidates[next_layer]:
            trans = _transition_cost(cur_s, cur_f, ns, nf, bend_flags[next_layer])
            pos = _position_cost(ns, nf, num_strings, bend_flags[next_layer])
            new_cost = cost_so_far + trans + pos

            nk = (next_layer, ns, nf)
            if nk in best_cost and best_cost[nk] <= new_cost:
                continue

            # Heuristic: 0 is admissible (optimistic)
            est = new_cost
            heapq.heappush(pq, (est, new_cost, next_layer, ns, nf,
                                path + ((ns, nf),), counter))
            counter += 1

    # Fallback: shouldn't happen if candidates are non-empty
    return [(c[0][0], c[0][1]) for c in candidates]


class FretMapper:
    """
    A*-Guitar fret position mapper. Replaces the old greedy heuristic with
    A* search that minimizes total hand movement across note sequences.

    For batch assignment (preferred): call map_sequence() with all notes at once.
    For single-note streaming: map_note() falls back to greedy with transition costs.

    Lead mode: bend-aware cost penalties to keep bends on frets 2-17.
    """

    def __init__(self, tuning: List[int], lead_mode: bool = False):
        self.tuning = tuning
        self.current_position = 0  # Current hand position (fret)
        self.prev_fret = None
        self.prev_string = None
        self.lead_mode = lead_mode

    def map_sequence(self, midi_notes: List[int],
                     bend_flags: List[bool] = None) -> List[Optional[Tuple[int, int]]]:
        """
        Map an entire note sequence using A*-Guitar search.
        This is the preferred entry point — gives globally optimal fret assignments.

        Args:
            midi_notes: List of MIDI note numbers in time order
            bend_flags: Per-note bend flag (optional)

        Returns:
            List of (string, fret) or None for each note
        """
        results = astar_fret_search(midi_notes, self.tuning,
                                    bend_flags=bend_flags,
                                    window=LOOKAHEAD)

        # Update internal state to last assigned position
        for r in reversed(results):
            if r is not None:
                self.prev_string, self.prev_fret = r
                self.current_position = max(0, self.prev_fret - 2)
                break

        return results

    def map_note(self, midi_note: int, has_bend: bool = False) -> Optional[Tuple[int, int]]:
        """Map a single note with greedy fallback, updating position context."""
        candidates = _get_candidates(midi_note, self.tuning)
        if not candidates:
            return None

        num_strings = len(self.tuning)

        def score(c):
            s, f = c
            base = _position_cost(s, f, num_strings, has_bend)
            if self.prev_fret is not None and self.prev_string is not None:
                base += _transition_cost(self.prev_string, self.prev_fret, s, f, has_bend)
            return base

        result = min(candidates, key=score)

        self.prev_string, self.prev_fret = result
        self.current_position = max(0, self.prev_fret - 2)

        return result

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

        # Collect pitch bend events and articulation CC#20 markers
        pitch_bends = []
        articulation_markers = {}  # beat_time -> articulation type

        ART_TYPES = {0: None, 1: 'bend', 2: 'slide_up', 3: 'slide_down',
                     4: 'hammer_on', 5: 'pull_off', 6: 'vibrato'}

        for track in midi.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'pitchwheel':
                    beat_time = current_time / midi.ticks_per_beat
                    pitch_bends.append({'beat': beat_time, 'value': msg.pitch})
                elif msg.type == 'control_change' and msg.control == 20:
                    beat_time = current_time / midi.ticks_per_beat
                    articulation_markers[beat_time] = ART_TYPES.get(msg.value)

        if articulation_markers:
            logger.info(f"Found {len(articulation_markers)} articulation markers in MIDI")

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

        # Create FretMapper with A*-Guitar search
        use_lead_mode = len(articulation_markers) > 0
        fret_mapper = FretMapper(tuning_notes, lead_mode=use_lead_mode)
        logger.info("Using A*-Guitar search for fret position optimization")

        # Helper: find articulation nearest to a beat time
        def _find_articulation(beat_time, tolerance=0.15):
            for art_beat, art_type in articulation_markers.items():
                if abs(art_beat - beat_time) < tolerance:
                    return art_type
            return None

        # Helper: find max pitch bend during a note
        def _max_bend_in_range(start_beat, end_beat):
            max_val = 0
            for pb in pitch_bends:
                if start_beat <= pb['beat'] <= end_beat:
                    max_val = max(max_val, abs(pb['value']))
            return max_val

        # --- A*-Guitar batch assignment (non-drum tracks) ---
        is_drum = 'drum' in instrument_type.lower()
        fret_positions = []

        if not is_drum:
            # Build per-note bend flags for A* search
            bend_flags = [_find_articulation(n['start_beat']) == 'bend' for n in notes]

            midi_sequence = [n['midi_note'] for n in notes]
            fret_positions = fret_mapper.map_sequence(midi_sequence, bend_flags=bend_flags)
            logger.info(f"A*-Guitar assigned {sum(1 for p in fret_positions if p is not None)}/{len(notes)} notes")

        # Add notes to measures
        for note_idx, note_data in enumerate(notes):
            measure_idx = int(note_data['start_beat'] / beats_per_measure)
            if measure_idx >= len(gp_track.measures):
                continue

            measure = gp_track.measures[measure_idx]
            beat_in_measure = note_data['start_beat'] % beats_per_measure

            # Look up articulation for this note
            note_art = _find_articulation(note_data['start_beat'])

            # Convert MIDI note to fret position
            if is_drum:
                # Drums: use voice for different drums
                fret_pos = (1, note_data['midi_note'] % 6)  # Simplified drum mapping
            else:
                # Use A*-Guitar pre-computed assignment
                fret_pos = fret_positions[note_idx] if note_idx < len(fret_positions) else None

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

            # Apply articulation effects from melody transcriber (CC#20 + pitch bends)
            if note_art:
                try:
                    if note_art == 'bend':
                        # Create GP bend effect
                        max_pb = _max_bend_in_range(
                            note_data['start_beat'],
                            note_data['start_beat'] + note_data['duration_beats']
                        )
                        if max_pb > 0:
                            bend_points = []
                            # GP bend points: position 0-12, value in quarter tones (100 = full step)
                            # Standard bend curve: rise to peak at position 6, sustain
                            bend_value = min(400, int(max_pb / 8192 * 200))  # Scale from MIDI to GP units
                            if bend_value >= 25:  # At least quarter step
                                bend_points.append(guitarpro.models.BendPoint(position=0, value=0))
                                bend_points.append(guitarpro.models.BendPoint(position=6, value=bend_value))
                                bend_points.append(guitarpro.models.BendPoint(position=12, value=bend_value))
                                gp_note.effect.bend = guitarpro.models.BendEffect(
                                    type=guitarpro.models.BendType.bend,
                                    value=bend_value,
                                    points=bend_points,
                                )

                    elif note_art in ('slide_up', 'slide_down'):
                        gp_note.effect.slides = [guitarpro.models.SlideType.shiftSlideTo]

                    elif note_art == 'hammer_on':
                        gp_note.effect.hammer = True

                    elif note_art == 'vibrato':
                        gp_note.effect.vibrato = guitarpro.models.Vibrato.slight

                except Exception as e:
                    # pyguitarpro API varies — if effect fails, note still exports fine
                    logger.debug(f"Could not apply {note_art} effect: {e}")

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
