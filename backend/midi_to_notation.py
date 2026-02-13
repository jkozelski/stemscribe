"""
MIDI to MusicXML Converter for StemScribe
==========================================
Converts MIDI files to MusicXML using music21.
MusicXML can then be rendered in the browser using OpenSheetMusicDisplay.

Supports melody mode with articulation markings (bends, slides, hammer-ons,
vibrato) read from MIDI pitch bend events and CC#20 metadata encoded by
the melody transcriber.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Try to import music21
try:
    from music21 import converter, stream, midi, note, chord, meter, key, tempo
    from music21 import instrument as m21_instrument
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    logger.warning("music21 not available - notation conversion disabled")


def midi_to_musicxml(midi_path: str, output_path: Optional[str] = None,
                     quantize: bool = True,
                     stem_type: str = 'guitar',
                     title: Optional[str] = None,
                     artist: Optional[str] = None,
                     melody_mode: bool = False,
                     audio_path: Optional[str] = None) -> Optional[str]:
    """
    Convert MIDI file to MusicXML.

    Args:
        midi_path: Path to input MIDI file
        output_path: Path for output MusicXML (default: same name with .musicxml)
        quantize: Whether to quantize rhythms for cleaner notation
        stem_type: Instrument type for proper clef/transposition
        title: Song title for metadata
        artist: Artist name for metadata
        melody_mode: If True, enforce monophonic output with articulation markings
        audio_path: Path to source audio for librosa tempo detection fallback

    Returns:
        Path to MusicXML file, or None on failure
    """
    if not MUSIC21_AVAILABLE:
        logger.error("music21 not installed - cannot convert MIDI to MusicXML")
        return None

    midi_path = Path(midi_path)
    if not midi_path.exists():
        logger.error(f"MIDI file not found: {midi_path}")
        return None

    if output_path is None:
        output_path = midi_path.with_suffix('.musicxml')
    else:
        output_path = Path(output_path)

    logger.info(f"🎼 Converting MIDI to MusicXML: {midi_path.name}")

    try:
        # Parse the MIDI file
        score = converter.parse(str(midi_path))

        # Set metadata (title, artist) - IMPORTANT: must be set before other operations
        from music21 import metadata as m21_metadata
        score.metadata = m21_metadata.Metadata()

        # Use provided title or derive from filename
        if title:
            display_title = f"{title} - {stem_type.replace('_', ' ').title()}"
        else:
            # Derive from filename
            display_title = midi_path.stem.replace('_', ' ').title()

        score.metadata.title = display_title
        score.metadata.composer = artist or ""
        logger.debug(f"  Set title: {display_title}")

        # Detect and set tempo
        detected_tempo = _detect_tempo(score, audio_path)
        if detected_tempo:
            has_tempo = any(isinstance(el, tempo.MetronomeMark) for el in score.recurse())
            if not has_tempo:
                for part in score.parts:
                    part.insert(0, tempo.MetronomeMark(number=detected_tempo))
                logger.debug(f"  Set tempo: {detected_tempo:.0f} BPM")

        # Apply instrument-specific settings
        score = _apply_instrument_settings(score, stem_type)

        # Quantize if requested (cleans up rhythm for notation)
        if quantize:
            score = _quantize_score(score)

        # Clean up for better notation
        score = _cleanup_for_notation(score, stem_type, melody_mode=melody_mode,
                                      midi_path_str=str(midi_path))

        # Write MusicXML
        score.write('musicxml', fp=str(output_path))

        logger.info(f"✅ MusicXML created: {output_path.name}")
        return str(output_path)

    except Exception as e:
        logger.error(f"MIDI to MusicXML conversion failed: {e}")
        return None


def _apply_instrument_settings(score: 'stream.Score', stem_type: str) -> 'stream.Score':
    """Apply appropriate instrument, clef, and transposition."""

    instrument_map = {
        'guitar': m21_instrument.AcousticGuitar(),
        'bass': m21_instrument.ElectricBass(),
        'piano': m21_instrument.Piano(),
        'vocals': m21_instrument.Vocalist(),
        'drums': m21_instrument.UnpitchedPercussion(),
        'other': m21_instrument.Piano(),  # Default
    }

    # Get the appropriate instrument
    inst = instrument_map.get(stem_type.lower().split('_')[0], m21_instrument.Piano())

    # Apply to all parts
    for part in score.parts:
        part.insert(0, inst)

    return score


def _quantize_score(score: 'stream.Score') -> 'stream.Score':
    """
    Quantize note timings and durations for cleaner notation.
    Uses intelligent quantization that preserves triplets and swing feel.
    """
    try:
        # Detect if the piece has triplet feel
        has_triplets = _detect_triplets(score)

        if has_triplets:
            # Include triplet divisions: 4 (16ths), 3 (triplets), 6 (16th triplets)
            divisors = [4, 3, 6, 2]
            logger.debug("  Detected triplet feel - using triplet-aware quantization")
        else:
            # Standard binary divisions: 4 (16ths), 2 (8ths)
            divisors = [4, 2]
            logger.debug("  Using standard binary quantization")

        # Use music21's quantize with appropriate divisors
        score.quantize(inPlace=True, quarterLengthDivisors=divisors)
        logger.debug(f"  Quantized with divisors: {divisors}")

    except Exception as e:
        logger.debug(f"  Quantization skipped: {e}")

    return score


def _detect_triplets(score: 'stream.Score') -> bool:
    """
    Detect if a score likely contains triplets based on note timing patterns.
    """
    try:
        triplet_count = 0
        binary_count = 0

        for n in score.recurse().notes:
            if hasattr(n, 'quarterLength'):
                ql = n.quarterLength

                # Check for triplet-like durations
                # Triplet 8th = 1/3 beat = 0.333...
                # Triplet 16th = 1/6 beat = 0.166...
                remainder_3 = abs(ql * 3 - round(ql * 3))
                remainder_4 = abs(ql * 4 - round(ql * 4))

                if remainder_3 < 0.05:  # Close to triplet division
                    triplet_count += 1
                elif remainder_4 < 0.05:  # Close to binary division
                    binary_count += 1

        total = triplet_count + binary_count
        if total > 0:
            triplet_ratio = triplet_count / total
            return triplet_ratio > 0.15  # More than 15% triplets

    except Exception:
        pass

    return False


def _detect_tempo(score: 'stream.Score', audio_path: Optional[str] = None) -> Optional[float]:
    """
    Detect tempo with priority: MIDI embedded > librosa beat tracking > default.
    """
    # 1. Check MIDI for existing MetronomeMark
    try:
        for el in score.recurse():
            if isinstance(el, tempo.MetronomeMark):
                if 40 < el.number < 240:
                    logger.debug(f"  Found tempo in MIDI: {el.number:.0f} BPM")
                    return el.number
    except Exception:
        pass

    # 2. Librosa beat tracking from source audio
    if audio_path and Path(audio_path).exists():
        try:
            import librosa
            y, sr = librosa.load(str(audio_path), duration=60)
            tempo_detected, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(tempo_detected) if hasattr(tempo_detected, '__float__') else float(tempo_detected[0])
            if 40 < tempo_val < 240:
                logger.info(f"  Detected tempo from audio: {tempo_val:.0f} BPM")
                return tempo_val
        except Exception as e:
            logger.debug(f"  Librosa tempo detection failed: {e}")

    # 3. Default
    return 120.0


def _cleanup_for_notation(score: 'stream.Score', stem_type: str,
                          melody_mode: bool = False,
                          midi_path_str: Optional[str] = None) -> 'stream.Score':
    """
    Clean up the score for better notation rendering.
    - Remove very short notes (likely artifacts)
    - Consolidate rests
    - Add time signature if missing
    - Add key signature if detected
    - Add dynamics based on velocity
    - Melody mode: enforce monophonic, add articulation markings
    """
    from music21 import dynamics

    # Melody mode: enforce monophonic output (keep highest note per beat)
    if melody_mode:
        _enforce_monophonic(score)

        # Read and apply articulation markings from MIDI CC#20 data
        if midi_path_str:
            try:
                articulations = _read_midi_articulations(midi_path_str)
                if articulations:
                    _apply_articulation_markings(score, articulations)
            except Exception as e:
                logger.debug(f"  Articulation marking skipped: {e}")

    # Ensure time signature exists
    has_time_sig = False
    for el in score.recurse():
        if isinstance(el, meter.TimeSignature):
            has_time_sig = True
            break

    if not has_time_sig:
        for part in score.parts:
            part.insert(0, meter.TimeSignature('4/4'))
        logger.debug("  Added 4/4 time signature")

    # Try to detect and add key signature
    try:
        detected_key = score.analyze('key')
        if detected_key:
            for part in score.parts:
                part.insert(0, detected_key)
            logger.debug(f"  Detected key: {detected_key}")
    except:
        pass

    # Remove very short notes (artifacts)
    min_duration = 0.125  # 32nd note minimum
    for part in score.parts:
        notes_to_remove = []
        for n in part.recurse().notes:
            if hasattr(n, 'quarterLength') and n.quarterLength < min_duration:
                notes_to_remove.append(n)
        for n in notes_to_remove:
            part.remove(n, recurse=True)

    # Add dynamics based on velocity changes
    try:
        _add_dynamics_from_velocity(score)
    except Exception as e:
        logger.debug(f"  Dynamics addition skipped: {e}")

    # Make rests explicit
    try:
        for part in score.parts:
            part.makeRests(fillGaps=True, inPlace=True)
    except:
        pass

    return score


def _enforce_monophonic(score: 'stream.Score'):
    """
    Enforce monophonic output for melody transcriptions.
    If any beat has multiple simultaneous notes, keep only the highest.
    Also attempts to add articulation markings from MIDI CC data.
    """
    for part in score.parts:
        # Group notes by offset to find simultaneities
        offset_notes = {}
        for n in part.recurse().notes:
            if isinstance(n, chord.Chord):
                # Convert chord to highest note
                offset = n.offset
                if offset not in offset_notes:
                    offset_notes[offset] = []
                offset_notes[offset].append(n)
            elif isinstance(n, note.Note):
                offset = n.offset
                if offset not in offset_notes:
                    offset_notes[offset] = []
                offset_notes[offset].append(n)

        # Remove lower notes in simultaneities
        for offset, notes_at in offset_notes.items():
            if len(notes_at) > 1:
                # Sort by pitch (highest first) and remove all but highest
                sorted_notes = sorted(notes_at, key=lambda n: n.pitch.midi if hasattr(n, 'pitch') else 0, reverse=True)
                for n in sorted_notes[1:]:
                    try:
                        part.remove(n, recurse=True)
                    except Exception:
                        pass

    logger.debug("  Enforced monophonic output for melody mode")


def _read_midi_articulations(midi_path: str) -> Dict[float, str]:
    """
    Read articulation markers from MIDI CC#20 events encoded by melody_transcriber.

    CC#20 values:
        0 = none, 1 = bend, 2 = slide_up, 3 = slide_down,
        4 = hammer_on, 5 = pull_off, 6 = vibrato

    Returns:
        Dict mapping time (in seconds) to articulation type string
    """
    try:
        import mido

        ART_TYPES = {
            0: None, 1: 'bend', 2: 'slide_up', 3: 'slide_down',
            4: 'hammer_on', 5: 'pull_off', 6: 'vibrato'
        }

        mid = mido.MidiFile(midi_path)
        articulations = {}
        ticks_per_beat = mid.ticks_per_beat

        for track in mid.tracks:
            abs_time = 0
            current_tempo = 500000  # default 120 BPM

            for msg in track:
                abs_time += msg.time

                if msg.type == 'set_tempo':
                    current_tempo = msg.tempo
                elif msg.type == 'control_change' and msg.control == 20:
                    art_type = ART_TYPES.get(msg.value)
                    if art_type:
                        # Convert ticks to quarter-note offset
                        time_in_quarters = abs_time / ticks_per_beat
                        articulations[time_in_quarters] = art_type

        logger.debug(f"  Read {len(articulations)} articulation markers from MIDI")
        return articulations

    except Exception as e:
        logger.debug(f"  Could not read MIDI articulations: {e}")
        return {}


def _apply_articulation_markings(score: 'stream.Score', articulations: Dict[float, str]):
    """
    Apply music21 articulation markings to notes based on MIDI CC#20 data.

    Mapping:
        bend      -> BendAfter (fall/doit notation)
        slide_up  -> Glissando to next note
        slide_down -> Glissando to next note
        hammer_on -> Slur (legato) to next note
        pull_off  -> Slur (legato) to next note
        vibrato   -> TextExpression "~" or Trill-like marking
    """
    from music21 import articulations as m21_art
    from music21 import expressions, spanner

    if not articulations:
        return

    markings_added = 0

    for part in score.parts:
        all_notes = list(part.recurse().notes)

        for i, n in enumerate(all_notes):
            if not isinstance(n, note.Note):
                continue

            # Find matching articulation (within 0.25 quarter note tolerance)
            note_offset = n.getOffsetInHierarchy(part)
            matched_art = None
            for art_offset, art_type in articulations.items():
                if abs(note_offset - art_offset) < 0.25:
                    matched_art = art_type
                    break

            if not matched_art:
                continue

            try:
                if matched_art == 'bend':
                    # BendAfter — shows a bend line after the note
                    bend = m21_art.BendAfter()
                    bend.bendHeight = 2  # quarter-tone units
                    n.articulations.append(bend)
                    markings_added += 1

                elif matched_art in ('slide_up', 'slide_down'):
                    # Glissando to next note
                    if i + 1 < len(all_notes):
                        next_note = all_notes[i + 1]
                        if isinstance(next_note, note.Note):
                            gliss = spanner.Glissando(n, next_note)
                            gliss.slideType = 'continuous'
                            part.insert(0, gliss)
                            markings_added += 1

                elif matched_art in ('hammer_on', 'pull_off'):
                    # Slur (legato marking) to next note
                    if i + 1 < len(all_notes):
                        next_note = all_notes[i + 1]
                        slur = spanner.Slur(n, next_note)
                        part.insert(0, slur)
                        markings_added += 1

                elif matched_art == 'vibrato':
                    # Text expression for vibrato
                    vib = expressions.TextExpression('vibr.')
                    vib.style.fontStyle = 'italic'
                    vib.style.fontSize = 8
                    n.activeSite.insert(note_offset, vib)
                    markings_added += 1

            except Exception as e:
                logger.debug(f"  Could not add {matched_art} marking: {e}")

    logger.debug(f"  Added {markings_added} articulation markings to notation")


def _add_dynamics_from_velocity(score: 'stream.Score'):
    """
    Add dynamic markings (f, mf, p, etc.) based on velocity changes.
    Only adds dynamics when there's a significant change.
    """
    from music21 import dynamics

    # Velocity to dynamic mapping
    def velocity_to_dynamic(vel):
        if vel >= 112:
            return 'ff'
        elif vel >= 96:
            return 'f'
        elif vel >= 80:
            return 'mf'
        elif vel >= 64:
            return 'mp'
        elif vel >= 48:
            return 'p'
        else:
            return 'pp'

    for part in score.parts:
        last_dynamic = None
        last_dynamic_time = -10  # Don't add dynamics too frequently

        for n in part.recurse().notes:
            if hasattr(n, 'volume') and n.volume.velocity:
                vel = n.volume.velocity
                current_dynamic = velocity_to_dynamic(vel)
                current_time = n.offset

                # Only add dynamic if it changed significantly and enough time passed
                if current_dynamic != last_dynamic and (current_time - last_dynamic_time) >= 4.0:
                    dyn = dynamics.Dynamic(current_dynamic)
                    n.activeSite.insert(n.offset, dyn)
                    last_dynamic = current_dynamic
                    last_dynamic_time = current_time


def get_musicxml_for_stem(job_dir: Path, stem_name: str) -> Optional[str]:
    """
    Get or create MusicXML for a stem.
    First checks if MusicXML already exists, otherwise converts from MIDI.

    Args:
        job_dir: Job directory path
        stem_name: Name of the stem (e.g., 'guitar', 'bass')

    Returns:
        Path to MusicXML file, or None if not available
    """
    # Check for existing MusicXML
    musicxml_patterns = [
        f"{stem_name}.musicxml",
        f"{stem_name}_enhanced.musicxml",
        f"*{stem_name}*.musicxml",
    ]

    for pattern in musicxml_patterns:
        matches = list(job_dir.glob(pattern))
        if matches:
            return str(matches[0])

    # No MusicXML - try to convert from MIDI
    midi_patterns = [
        f"{stem_name}.mid",
        f"{stem_name}_enhanced.mid",
        f"*{stem_name}*.mid",
    ]

    for pattern in midi_patterns:
        matches = list(job_dir.glob(pattern))
        if matches:
            midi_path = matches[0]
            # Determine stem type from name
            stem_type = stem_name.split('_')[0]  # e.g., 'guitar_left' -> 'guitar'

            # Convert to MusicXML
            musicxml_path = midi_to_musicxml(
                str(midi_path),
                quantize=True,
                stem_type=stem_type
            )
            return musicxml_path

    logger.warning(f"No MIDI or MusicXML found for stem: {stem_name}")
    return None


def batch_convert_job(job_dir: Path) -> Dict[str, str]:
    """
    Convert all MIDI files in a job directory to MusicXML.

    Returns:
        Dict mapping stem names to MusicXML paths
    """
    job_dir = Path(job_dir)
    results = {}

    midi_files = list(job_dir.glob("*.mid"))
    logger.info(f"🎼 Batch converting {len(midi_files)} MIDI files to MusicXML")

    for midi_path in midi_files:
        stem_name = midi_path.stem.replace('_enhanced', '')
        stem_type = stem_name.split('_')[0]

        musicxml_path = midi_to_musicxml(
            str(midi_path),
            quantize=True,
            stem_type=stem_type
        )

        if musicxml_path:
            results[stem_name] = musicxml_path

    return results


# Quick test
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print(f"music21 available: {MUSIC21_AVAILABLE}")

    if MUSIC21_AVAILABLE:
        from music21 import environment
        print(f"music21 version: {environment.Environment()['localCorpusPath']}")
