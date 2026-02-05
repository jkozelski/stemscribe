"""
Chord Analysis Module for StemScribe
=====================================
Analyzes MIDI/audio to detect chord progressions and generates chord charts.

Uses music21's chordify() method to extract harmonic content,
then identifies chord symbols (Am, G7, Cmaj7, etc.)

Features:
- Chord detection from MIDI files
- Chord simplification (complex -> common names)
- Timed chord progression export
- MusicXML chord symbol injection
- JSON chord chart export for UI
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json

logger = logging.getLogger(__name__)

try:
    from music21 import converter, stream, chord as m21_chord, harmony, key, meter
    from music21 import note as m21_note
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    logger.warning("music21 not available - chord analysis disabled")


@dataclass
class ChordEvent:
    """Represents a chord at a specific time"""
    symbol: str           # e.g., "Am", "G7", "Cmaj7"
    root: str            # e.g., "A", "G", "C"
    quality: str         # e.g., "minor", "dominant-seventh", "major"
    bass: Optional[str]  # For slash chords, e.g., "E" in "Am/E"
    start_beat: float    # Start time in beats
    duration_beats: float # Duration in beats
    start_seconds: float # Start time in seconds (if tempo known)

    def to_dict(self):
        return asdict(self)


# Common chord quality mappings for cleaner symbols
QUALITY_SYMBOLS = {
    'major': '',
    'minor': 'm',
    'diminished': 'dim',
    'augmented': 'aug',
    'dominant-seventh': '7',
    'major-seventh': 'maj7',
    'minor-seventh': 'm7',
    'diminished-seventh': 'dim7',
    'half-diminished-seventh': 'm7b5',
    'augmented-seventh': 'aug7',
    'suspended-fourth': 'sus4',
    'suspended-second': 'sus2',
    'power': '5',
    'major-sixth': '6',
    'minor-sixth': 'm6',
    'major-ninth': 'maj9',
    'minor-ninth': 'm9',
    'dominant-ninth': '9',
}


def analyze_chords(midi_path: str,
                   simplify: bool = True,
                   min_duration: float = 0.5,
                   tempo: float = 120.0) -> List[ChordEvent]:
    """
    Analyze a MIDI file and extract chord progression.

    Args:
        midi_path: Path to MIDI file
        simplify: Reduce complex chords to common forms
        min_duration: Minimum chord duration in beats (filters noise)
        tempo: BPM for time calculations

    Returns:
        List of ChordEvent objects
    """
    if not MUSIC21_AVAILABLE:
        logger.error("music21 not available for chord analysis")
        return []

    try:
        # Load MIDI
        score = converter.parse(midi_path)
        logger.info(f"🎵 Analyzing chords in: {Path(midi_path).name}")

        # Get tempo from score if available
        for el in score.recurse():
            if hasattr(el, 'number') and el.__class__.__name__ == 'MetronomeMark':
                tempo = el.number
                break

        # Chordify - combine all simultaneous notes into chords
        chordified = score.chordify()

        # Extract chord events
        events = []
        seconds_per_beat = 60.0 / tempo

        for c in chordified.recurse().getElementsByClass(m21_chord.Chord):
            if c.quarterLength < min_duration:
                continue

            # Get chord symbol
            try:
                # Try to identify the chord
                chord_symbol = c.pitchedCommonName
                root = c.root().name if c.root() else 'N'
                bass = c.bass().name if c.bass() and c.bass() != c.root() else None

                # Determine quality
                quality = _get_chord_quality(c)

                # Build clean symbol
                if simplify:
                    symbol = _simplify_chord(root, quality, bass)
                else:
                    symbol = chord_symbol

                event = ChordEvent(
                    symbol=symbol,
                    root=root,
                    quality=quality,
                    bass=bass,
                    start_beat=float(c.offset),
                    duration_beats=float(c.quarterLength),
                    start_seconds=float(c.offset) * seconds_per_beat
                )
                events.append(event)

            except Exception as e:
                logger.debug(f"Could not identify chord at beat {c.offset}: {e}")
                continue

        # Merge consecutive identical chords
        events = _merge_consecutive_chords(events)

        logger.info(f"✅ Found {len(events)} chord changes")
        return events

    except Exception as e:
        logger.error(f"Chord analysis failed: {e}")
        return []


def _get_chord_quality(c: 'm21_chord.Chord') -> str:
    """Determine chord quality from music21 chord object."""
    try:
        # music21's quality property
        if c.isMinorTriad():
            return 'minor'
        elif c.isMajorTriad():
            return 'major'
        elif c.isDiminishedTriad():
            return 'diminished'
        elif c.isAugmentedTriad():
            return 'augmented'
        elif c.isDominantSeventh():
            return 'dominant-seventh'
        elif c.isMajorSeventh():
            return 'major-seventh'
        elif c.isMinorSeventh():
            return 'minor-seventh'
        elif c.isDiminishedSeventh():
            return 'diminished-seventh'
        elif c.isHalfDiminishedSeventh():
            return 'half-diminished-seventh'
        else:
            return c.quality if hasattr(c, 'quality') else 'major'
    except:
        return 'major'


def _simplify_chord(root: str, quality: str, bass: Optional[str]) -> str:
    """Build a clean, readable chord symbol."""
    # Get quality symbol
    quality_sym = QUALITY_SYMBOLS.get(quality, '')

    # Clean root note (remove octave numbers)
    root = root.replace('-', 'b').rstrip('0123456789')

    # Build symbol
    symbol = f"{root}{quality_sym}"

    # Add bass for slash chords
    if bass and bass != root:
        bass = bass.replace('-', 'b').rstrip('0123456789')
        symbol = f"{symbol}/{bass}"

    return symbol


def _merge_consecutive_chords(events: List[ChordEvent]) -> List[ChordEvent]:
    """Merge consecutive identical chords into longer durations."""
    if not events:
        return events

    merged = [events[0]]

    for event in events[1:]:
        last = merged[-1]
        if event.symbol == last.symbol:
            # Extend the previous chord
            merged[-1] = ChordEvent(
                symbol=last.symbol,
                root=last.root,
                quality=last.quality,
                bass=last.bass,
                start_beat=last.start_beat,
                duration_beats=last.duration_beats + event.duration_beats,
                start_seconds=last.start_seconds
            )
        else:
            merged.append(event)

    return merged


def analyze_job_chords(job_dir: Path,
                       prefer_stems: List[str] = ['piano', 'guitar', 'other']) -> List[ChordEvent]:
    """
    Analyze chords for a processing job.
    Prefers harmonic instruments (piano, guitar) over melodic (bass, vocals).

    Args:
        job_dir: Path to job directory
        prefer_stems: Ordered list of stems to try for chord analysis

    Returns:
        List of ChordEvent objects
    """
    job_dir = Path(job_dir)
    midi_dir = job_dir / 'midi'

    if not midi_dir.exists():
        logger.warning(f"No MIDI directory found: {midi_dir}")
        return []

    # Try each preferred stem
    for stem_name in prefer_stems:
        midi_files = list(midi_dir.glob(f'*{stem_name}*.mid'))
        if midi_files:
            logger.info(f"Using {stem_name} stem for chord analysis")
            return analyze_chords(str(midi_files[0]))

    # Fall back to any available MIDI
    midi_files = list(midi_dir.glob('*.mid'))
    if midi_files:
        return analyze_chords(str(midi_files[0]))

    return []


def export_chord_chart(events: List[ChordEvent],
                       output_path: str,
                       title: str = "Chord Chart",
                       tempo: float = 120.0) -> str:
    """
    Export chord progression as JSON for UI display.

    Args:
        events: List of ChordEvent objects
        output_path: Path for output JSON file
        title: Song title
        tempo: BPM

    Returns:
        Path to created file
    """
    chart = {
        'title': title,
        'tempo': tempo,
        'time_signature': '4/4',
        'chords': [e.to_dict() for e in events],
        'unique_chords': list(set(e.symbol for e in events))
    }

    with open(output_path, 'w') as f:
        json.dump(chart, f, indent=2)

    logger.info(f"✅ Chord chart exported: {output_path}")
    return output_path


def add_chords_to_musicxml(musicxml_path: str,
                           events: List[ChordEvent],
                           output_path: Optional[str] = None) -> Optional[str]:
    """
    Add chord symbols to an existing MusicXML file.

    Args:
        musicxml_path: Path to input MusicXML
        events: Chord events to add
        output_path: Path for output (default: overwrite input)

    Returns:
        Path to modified file, or None on failure
    """
    if not MUSIC21_AVAILABLE:
        return None

    try:
        score = converter.parse(musicxml_path)

        # Get the first part (or create one)
        if score.parts:
            part = score.parts[0]
        else:
            return None

        # Add chord symbols
        for event in events:
            try:
                # Create harmony.ChordSymbol
                cs = harmony.ChordSymbol(event.symbol)
                cs.offset = event.start_beat
                part.insert(event.start_beat, cs)
            except Exception as e:
                logger.debug(f"Could not add chord {event.symbol}: {e}")

        # Save
        out_path = output_path or musicxml_path
        score.write('musicxml', fp=out_path)

        logger.info(f"✅ Added {len(events)} chord symbols to {Path(out_path).name}")
        return out_path

    except Exception as e:
        logger.error(f"Failed to add chords to MusicXML: {e}")
        return None


def get_chord_diagram(chord_symbol: str, instrument: str = 'guitar') -> Optional[Dict]:
    """
    Get fingering diagram for a chord.

    Returns dict with:
    - frets: list of fret numbers per string (-1 = muted, 0 = open)
    - fingers: suggested fingering
    - barre: barre fret if applicable
    """
    # Common guitar chord shapes (standard tuning: EADGBE)
    GUITAR_CHORDS = {
        'C': {'frets': [-1, 3, 2, 0, 1, 0], 'fingers': [0, 3, 2, 0, 1, 0]},
        'D': {'frets': [-1, -1, 0, 2, 3, 2], 'fingers': [0, 0, 0, 1, 3, 2]},
        'Dm': {'frets': [-1, -1, 0, 2, 3, 1], 'fingers': [0, 0, 0, 2, 3, 1]},
        'E': {'frets': [0, 2, 2, 1, 0, 0], 'fingers': [0, 2, 3, 1, 0, 0]},
        'Em': {'frets': [0, 2, 2, 0, 0, 0], 'fingers': [0, 2, 3, 0, 0, 0]},
        'F': {'frets': [1, 3, 3, 2, 1, 1], 'fingers': [1, 3, 4, 2, 1, 1], 'barre': 1},
        'G': {'frets': [3, 2, 0, 0, 0, 3], 'fingers': [2, 1, 0, 0, 0, 3]},
        'A': {'frets': [-1, 0, 2, 2, 2, 0], 'fingers': [0, 0, 1, 2, 3, 0]},
        'Am': {'frets': [-1, 0, 2, 2, 1, 0], 'fingers': [0, 0, 2, 3, 1, 0]},
        'B': {'frets': [-1, 2, 4, 4, 4, 2], 'fingers': [0, 1, 2, 3, 4, 1], 'barre': 2},
        'Bm': {'frets': [-1, 2, 4, 4, 3, 2], 'fingers': [0, 1, 3, 4, 2, 1], 'barre': 2},

        # Seventh chords
        'C7': {'frets': [-1, 3, 2, 3, 1, 0], 'fingers': [0, 3, 2, 4, 1, 0]},
        'D7': {'frets': [-1, -1, 0, 2, 1, 2], 'fingers': [0, 0, 0, 2, 1, 3]},
        'E7': {'frets': [0, 2, 0, 1, 0, 0], 'fingers': [0, 2, 0, 1, 0, 0]},
        'G7': {'frets': [3, 2, 0, 0, 0, 1], 'fingers': [3, 2, 0, 0, 0, 1]},
        'A7': {'frets': [-1, 0, 2, 0, 2, 0], 'fingers': [0, 0, 1, 0, 2, 0]},
        'Am7': {'frets': [-1, 0, 2, 0, 1, 0], 'fingers': [0, 0, 2, 0, 1, 0]},
        'Dm7': {'frets': [-1, -1, 0, 2, 1, 1], 'fingers': [0, 0, 0, 2, 1, 1]},
        'Em7': {'frets': [0, 2, 0, 0, 0, 0], 'fingers': [0, 2, 0, 0, 0, 0]},

        # Major 7
        'Cmaj7': {'frets': [-1, 3, 2, 0, 0, 0], 'fingers': [0, 3, 2, 0, 0, 0]},
        'Dmaj7': {'frets': [-1, -1, 0, 2, 2, 2], 'fingers': [0, 0, 0, 1, 1, 1]},
        'Fmaj7': {'frets': [1, -1, 2, 2, 1, 0], 'fingers': [1, 0, 3, 4, 2, 0]},
        'Gmaj7': {'frets': [3, 2, 0, 0, 0, 2], 'fingers': [2, 1, 0, 0, 0, 3]},
        'Amaj7': {'frets': [-1, 0, 2, 1, 2, 0], 'fingers': [0, 0, 2, 1, 3, 0]},
    }

    return GUITAR_CHORDS.get(chord_symbol)


# Quick test
if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    if len(sys.argv) < 2:
        print("Usage: python chord_analysis.py <midi_file>")
        print("\nAnalyzes MIDI file and outputs chord progression")
        sys.exit(1)

    midi_file = sys.argv[1]
    events = analyze_chords(midi_file)

    print(f"\n🎸 Chord Progression ({len(events)} changes):\n")
    for e in events:
        print(f"  Beat {e.start_beat:6.1f}: {e.symbol:8} ({e.duration_beats:.1f} beats)")

    # Show unique chords
    unique = sorted(set(e.symbol for e in events))
    print(f"\n📋 Unique chords: {', '.join(unique)}")
