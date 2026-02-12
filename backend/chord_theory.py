"""
Chord Theory Engine for StemScribe
===================================
Maps chords to scales, modes, and practice suggestions.
Based on standard music theory (triads/7ths over bass notes, polychords).

Inspired by Rick Beato's chord-over-bass framework:
- Every chord/bass combination maps to specific scales and modes
- Interval relationships between bass and upper structure
- Generic chord naming (MajΔ, MinΔ, DimΔ, AugΔ, Sus4Δ, LydΔ, LocΔ)

Usage:
    from chord_theory import ChordTheoryEngine

    engine = ChordTheoryEngine()
    info = engine.analyze("Am7")
    print(info['scales'])        # ['A Aeolian', 'A Dorian', 'A Phrygian']
    print(info['practice_tip'])  # "Try A Dorian for a jazzier sound"

    # With bass note (inversion/polychord)
    info = engine.analyze("Db/C")
    print(info['chord_type'])    # 'Phrygian'
    print(info['scales'])        # ['C Phrygian', 'C Phrygian Major']
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Note and Interval Constants
# ============================================================================

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enharmonic equivalents
ENHARMONIC = {
    'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'Cb': 'B', 'B#': 'C', 'E#': 'F',
    'C#': 'C#', 'D#': 'D#', 'F#': 'F#', 'G#': 'G#', 'A#': 'A#',
    'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'A': 'A', 'B': 'B'
}

# Semitone intervals for chord qualities
CHORD_INTERVALS = {
    'maj':    [0, 4, 7],
    'min':    [0, 3, 7],
    'dim':    [0, 3, 6],
    'aug':    [0, 4, 8],
    'sus2':   [0, 2, 7],
    'sus4':   [0, 5, 7],
    '7':      [0, 4, 7, 10],
    'maj7':   [0, 4, 7, 11],
    'min7':   [0, 3, 7, 10],
    'mMaj7':  [0, 3, 7, 11],
    'dim7':   [0, 3, 6, 9],
    'hdim7':  [0, 3, 6, 10],
    '6':      [0, 4, 7, 9],
    'min6':   [0, 3, 7, 9],
    '9':      [0, 4, 7, 10, 14],
    'add9':   [0, 4, 7, 14],
}

# ============================================================================
# Scale Definitions (intervals from root in semitones)
# ============================================================================

SCALES = {
    # Major modes
    'Ionian (Major)':       [0, 2, 4, 5, 7, 9, 11],
    'Dorian':               [0, 2, 3, 5, 7, 9, 10],
    'Phrygian':             [0, 1, 3, 5, 7, 8, 10],
    'Lydian':               [0, 2, 4, 6, 7, 9, 11],
    'Mixolydian':           [0, 2, 4, 5, 7, 9, 10],
    'Aeolian (Natural Minor)': [0, 2, 3, 5, 7, 8, 10],
    'Locrian':              [0, 1, 3, 5, 6, 8, 10],

    # Melodic minor modes
    'Melodic Minor':        [0, 2, 3, 5, 7, 9, 11],
    'Dorian b2':            [0, 1, 3, 5, 7, 9, 10],
    'Lydian Augmented':     [0, 2, 4, 6, 8, 9, 11],
    'Lydian Dominant':      [0, 2, 4, 6, 7, 9, 10],
    'Mixolydian b6':        [0, 2, 4, 5, 7, 8, 10],
    'Locrian #2':           [0, 2, 3, 5, 6, 8, 10],
    'Altered (Super Locrian)': [0, 1, 3, 4, 6, 8, 10],

    # Harmonic minor modes
    'Harmonic Minor':       [0, 2, 3, 5, 7, 8, 11],
    'Phrygian Major':       [0, 1, 3, 5, 7, 8, 11],

    # Symmetric scales
    'Whole Tone':           [0, 2, 4, 6, 8, 10],
    'Diminished (WH)':      [0, 2, 3, 5, 6, 8, 9, 11],
    'Diminished (HW)':      [0, 1, 3, 4, 6, 7, 9, 10],
    'Chromatic':            list(range(12)),

    # Pentatonics
    'Major Pentatonic':     [0, 2, 4, 7, 9],
    'Minor Pentatonic':     [0, 3, 5, 7, 10],
    'Blues':                 [0, 3, 5, 6, 7, 10],
}

# ============================================================================
# Chord Quality → Scale Mappings
# ============================================================================

# Maps chord quality to recommended scales (ordered by common usage)
QUALITY_SCALE_MAP = {
    'maj': {
        'primary': ['Ionian (Major)', 'Lydian'],
        'secondary': ['Mixolydian', 'Major Pentatonic'],
        'tip': 'Major chord — Ionian is safe, Lydian adds color with #4'
    },
    'min': {
        'primary': ['Dorian', 'Aeolian (Natural Minor)'],
        'secondary': ['Phrygian', 'Minor Pentatonic', 'Blues'],
        'tip': 'Minor chord — Dorian for jazz, Aeolian for rock/pop, Blues for... blues'
    },
    '7': {
        'primary': ['Mixolydian', 'Blues'],
        'secondary': ['Lydian Dominant', 'Altered (Super Locrian)', 'Diminished (HW)'],
        'tip': 'Dominant 7th — Mixolydian is standard, Altered for jazz tension'
    },
    'maj7': {
        'primary': ['Ionian (Major)', 'Lydian'],
        'secondary': ['Major Pentatonic'],
        'tip': 'Major 7th — Lydian gives that dreamy jazz sound'
    },
    'min7': {
        'primary': ['Dorian', 'Aeolian (Natural Minor)'],
        'secondary': ['Phrygian', 'Minor Pentatonic'],
        'tip': 'Minor 7th — Dorian is the go-to for ii-V-I progressions'
    },
    'mMaj7': {
        'primary': ['Melodic Minor', 'Harmonic Minor'],
        'secondary': [],
        'tip': 'Minor-major 7th — mysterious sound, try Melodic Minor ascending'
    },
    'dim': {
        'primary': ['Locrian', 'Diminished (WH)'],
        'secondary': ['Locrian #2'],
        'tip': 'Diminished triad — Whole-Half diminished scale works great'
    },
    'dim7': {
        'primary': ['Diminished (HW)'],
        'secondary': ['Diminished (WH)'],
        'tip': 'Fully diminished — Half-Whole diminished scale, very symmetrical'
    },
    'hdim7': {
        'primary': ['Locrian', 'Locrian #2'],
        'secondary': [],
        'tip': 'Half-diminished (ø7) — Locrian #2 avoids the ugly b2'
    },
    'aug': {
        'primary': ['Whole Tone', 'Lydian Augmented'],
        'secondary': [],
        'tip': 'Augmented — Whole Tone scale is the classic choice'
    },
    'sus2': {
        'primary': ['Ionian (Major)', 'Mixolydian'],
        'secondary': ['Major Pentatonic'],
        'tip': 'Sus2 — ambiguous, works with major or mixolydian'
    },
    'sus4': {
        'primary': ['Mixolydian', 'Dorian'],
        'secondary': ['Minor Pentatonic'],
        'tip': 'Sus4 — wants to resolve, Mixolydian keeps it floating'
    },
    '6': {
        'primary': ['Ionian (Major)', 'Major Pentatonic'],
        'secondary': ['Lydian'],
        'tip': 'Major 6th — classic jazz chord, Major Pentatonic is buttery smooth'
    },
    'min6': {
        'primary': ['Dorian', 'Melodic Minor'],
        'secondary': ['Minor Pentatonic'],
        'tip': 'Minor 6th — Dorian is perfect (it has the natural 6th)'
    },
    '9': {
        'primary': ['Mixolydian', 'Blues'],
        'secondary': ['Lydian Dominant'],
        'tip': 'Dominant 9th — funky Mixolydian or blues scale'
    },
    'add9': {
        'primary': ['Ionian (Major)', 'Lydian'],
        'secondary': ['Major Pentatonic'],
        'tip': 'Add9 — shimmery, Ionian or Lydian both work beautifully'
    },
}

# ============================================================================
# Triad Over Bass Note Mappings (Beato-style)
# Key = (interval from bass to triad root in semitones, triad quality)
# Value = chord type info
# ============================================================================

# When we detect a chord with a bass note (e.g., Db/C),
# we calculate the interval from bass (C) to chord root (Db) = 1 semitone
# Then look up what that means theoretically

TRIAD_OVER_BASS = {
    # Major triad over bass notes
    (1, 'maj'): {
        'type': 'Phrygian',
        'generic': 'MajΔ/Maj7',
        'intervals': 'b9, 11, b6',
        'scales': ['Phrygian', 'Phrygian Major'],
        'tip': 'Major triad a half step above bass — Phrygian sound, very Spanish/flamenco'
    },
    (2, 'maj'): {
        'type': 'Lydian',
        'generic': 'MajΔ/b7',
        'intervals': '9, #11, 13',
        'scales': ['Lydian', 'Lydian Dominant'],
        'tip': 'Major triad a whole step above — bright Lydian sound'
    },
    (3, 'maj'): {
        'type': 'Minor add extensions',
        'generic': 'MajΔ over min3rd',
        'intervals': 'b3, 5, b7',
        'scales': ['Dorian', 'Aeolian (Natural Minor)'],
        'tip': 'Creates a rich minor sound with added color'
    },
    (4, 'maj'): {
        'type': 'Suspended/Quartal',
        'generic': 'MajΔ over maj3rd',
        'intervals': '3, #5, 7',
        'scales': ['Ionian (Major)', 'Lydian'],
        'tip': 'Upper structure triad — lush extended harmony'
    },
    (5, 'maj'): {
        'type': 'Major with extensions',
        'generic': 'MajΔ/4th',
        'intervals': '4, 6, root',
        'scales': ['Ionian (Major)', 'Mixolydian'],
        'tip': 'IV chord over bass note — classic pop/rock sound'
    },
    (6, 'maj'): {
        'type': 'Tritone sub related',
        'generic': 'MajΔ/b5',
        'intervals': 'b5, b7, b2',
        'scales': ['Lydian Dominant', 'Altered (Super Locrian)'],
        'tip': 'Tritone away — tension city, resolves beautifully'
    },
    (7, 'maj'): {
        'type': 'Power chord extensions',
        'generic': 'MajΔ/5th',
        'intervals': '5, 7, 9',
        'scales': ['Ionian (Major)', 'Lydian'],
        'tip': 'V chord over root — rich voicing of the I chord'
    },
    (8, 'maj'): {
        'type': 'Augmented relationship',
        'generic': 'MajΔ/b6',
        'intervals': 'b6, root, b3',
        'scales': ['Phrygian', 'Harmonic Minor'],
        'tip': 'Dark, cinematic quality'
    },
    (9, 'maj'): {
        'type': '6th chord voicing',
        'generic': 'MajΔ/6th',
        'intervals': '6, #8, #3',
        'scales': ['Dorian', 'Melodic Minor'],
        'tip': 'Creates a 6th chord sound from the bass perspective'
    },
    (10, 'maj'): {
        'type': 'Dominant relationship',
        'generic': 'MajΔ/b7',
        'intervals': 'b7, 9, #11',
        'scales': ['Mixolydian', 'Lydian Dominant'],
        'tip': 'Dominant 9#11 sound — very hip jazz voicing'
    },
    (11, 'maj'): {
        'type': 'Leading tone bass',
        'generic': 'MajΔ/7th',
        'intervals': '7, #9, #5',
        'scales': ['Ionian (Major)', 'Lydian'],
        'tip': 'Bass wants to resolve up — beautiful tension'
    },

    # Minor triad over bass notes
    (1, 'min'): {
        'type': 'Altered Dominant',
        'generic': 'MinΔ/Maj7',
        'intervals': 'b9, 3, #5',
        'scales': ['Phrygian Major', 'Altered (Super Locrian)'],
        'tip': 'Minor triad half step up — altered dominant, very tense'
    },
    (2, 'min'): {
        'type': 'Dorian color',
        'generic': 'MinΔ/b7',
        'intervals': '9, 11, 6',
        'scales': ['Dorian', 'Dorian b2'],
        'tip': 'Rich minor sound with Dorian extensions'
    },
    (5, 'min'): {
        'type': 'Minor over 4th',
        'generic': 'MinΔ/4th',
        'intervals': '4, b6, root',
        'scales': ['Aeolian (Natural Minor)', 'Phrygian'],
        'tip': 'iv chord over bass — deep, melancholic'
    },
    (7, 'min'): {
        'type': 'Minor over 5th',
        'generic': 'MinΔ/5th',
        'intervals': '5, b7, 9',
        'scales': ['Dorian', 'Aeolian (Natural Minor)'],
        'tip': 'v chord over root — natural minor voicing'
    },

    # Diminished triad over bass notes
    (1, 'dim'): {
        'type': 'Dominant Diminished',
        'generic': 'DimΔ/Maj7',
        'intervals': 'b9, 3, 5',
        'scales': ['Phrygian Major', 'Diminished (HW)'],
        'tip': 'Diminished triad half step up — dominant function'
    },

    # Augmented triad over bass notes
    (1, 'aug'): {
        'type': 'Augmented Phrygian',
        'generic': 'AugΔ/Maj7',
        'intervals': 'b9, 11, 6',
        'scales': ['Phrygian', 'Lydian Augmented'],
        'tip': 'Augmented half step up — exotic, film score quality'
    },
}


# ============================================================================
# Chord Function in Key Context
# ============================================================================

# Roman numeral functions and their common scale choices
CHORD_FUNCTION_MAP = {
    # Major key functions
    'I':    {'quality': 'maj7', 'scales': ['Ionian (Major)', 'Lydian']},
    'ii':   {'quality': 'min7', 'scales': ['Dorian']},
    'iii':  {'quality': 'min7', 'scales': ['Phrygian']},
    'IV':   {'quality': 'maj7', 'scales': ['Lydian']},
    'V':    {'quality': '7',    'scales': ['Mixolydian']},
    'vi':   {'quality': 'min7', 'scales': ['Aeolian (Natural Minor)']},
    'vii°': {'quality': 'hdim7', 'scales': ['Locrian']},

    # Minor key functions
    'i':    {'quality': 'min7', 'scales': ['Aeolian (Natural Minor)', 'Dorian', 'Melodic Minor']},
    'ii°':  {'quality': 'hdim7', 'scales': ['Locrian', 'Locrian #2']},
    'III':  {'quality': 'maj7', 'scales': ['Ionian (Major)']},
    'iv':   {'quality': 'min7', 'scales': ['Dorian', 'Aeolian (Natural Minor)']},
    'v':    {'quality': 'min7', 'scales': ['Phrygian', 'Aeolian (Natural Minor)']},
    'V7':   {'quality': '7',    'scales': ['Mixolydian', 'Phrygian Major']},
    'VI':   {'quality': 'maj7', 'scales': ['Lydian']},
    'VII':  {'quality': '7',    'scales': ['Mixolydian']},
}


class ChordTheoryEngine:
    """
    Analyzes chords and provides scale suggestions, practice tips,
    and music theory context for StemScribe's practice mode.
    """

    def __init__(self):
        self._note_to_idx = {}
        for i, note in enumerate(NOTE_NAMES):
            self._note_to_idx[note] = i
        # Add enharmonic mappings
        for flat, sharp in ENHARMONIC.items():
            if sharp in self._note_to_idx:
                self._note_to_idx[flat] = self._note_to_idx[sharp]

    def _note_index(self, note: str) -> int:
        """Get chromatic index (0-11) for a note name."""
        return self._note_to_idx.get(note, -1)

    def _parse_chord(self, chord_str: str) -> Tuple[Optional[str], str, Optional[str]]:
        """
        Parse chord string into (root, quality, bass).
        Examples:
            "Am7"    -> ("A", "min7", None)
            "Db/C"   -> ("Db", "maj", "C")
            "Cmaj7"  -> ("C", "maj7", None)
            "F#dim7" -> ("F#", "dim7", None)
        """
        if not chord_str or chord_str in ('N', '—', 'Unknown'):
            return (None, 'unknown', None)

        # Split bass note
        bass = None
        if '/' in chord_str:
            parts = chord_str.split('/')
            chord_str = parts[0]
            bass = parts[1]

        # Extract root
        root = chord_str[0]
        rest = chord_str[1:]

        if rest and (rest[0] == '#' or rest[0] == 'b'):
            root = chord_str[:2]
            rest = chord_str[2:]

        # Determine quality from remainder
        quality = self._parse_quality(rest)

        return (root, quality, bass)

    def _parse_quality(self, suffix: str) -> str:
        """Parse chord quality from the suffix after the root note."""
        if not suffix or suffix == '':
            return 'maj'

        # Direct mappings (order matters — longer strings first)
        quality_map = {
            'mMaj7': 'mMaj7', 'mMaj': 'mMaj7',
            'maj7': 'maj7', 'Maj7': 'maj7', 'M7': 'maj7', 'Δ7': 'maj7', 'Δ': 'maj7',
            'maj9': 'maj7',  # Simplify for scale purposes
            'min7': 'min7', 'm7': 'min7', '-7': 'min7',
            'min9': 'min7',
            'min6': 'min6', 'm6': 'min6',
            'min': 'min', 'm': 'min', '-': 'min',
            'dim7': 'dim7', 'o7': 'dim7', '°7': 'dim7',
            'hdim7': 'hdim7', 'ø7': 'hdim7', 'ø': 'hdim7',
            'dim': 'dim', 'o': 'dim', '°': 'dim',
            'aug': 'aug', '+': 'aug',
            'sus4': 'sus4',
            'sus2': 'sus2',
            'add9': 'add9',
            '9': '9',
            '7': '7',
            '6': '6',
        }

        for pattern, quality in quality_map.items():
            if suffix.startswith(pattern):
                return quality

        return 'maj'  # Default to major if unrecognized

    def analyze(self, chord_str: str, key: Optional[str] = None) -> Dict:
        """
        Analyze a chord and return theory info, scale suggestions, and practice tips.

        Args:
            chord_str: Chord name (e.g., "Am7", "Db/C", "G7#9")
            key: Optional detected key for context (e.g., "Am", "C")

        Returns:
            Dictionary with:
                - root: Root note
                - quality: Chord quality
                - bass: Bass note (if inversion/polychord)
                - scales: List of recommended scale names with root
                - secondary_scales: Additional options
                - tip: Practice tip string
                - chord_type: Theoretical chord type name
                - intervals: Interval description
                - function: Roman numeral function (if key provided)
        """
        root, quality, bass = self._parse_chord(chord_str)

        if root is None:
            return {
                'root': None, 'quality': 'unknown', 'bass': None,
                'scales': [], 'secondary_scales': [], 'tip': '',
                'chord_type': 'Unknown', 'intervals': '', 'function': None
            }

        result = {
            'root': root,
            'quality': quality,
            'bass': bass,
            'chord_type': quality,
            'intervals': '',
            'function': None
        }

        # Check if this is a chord over bass note (polychord/inversion)
        if bass and bass != root:
            bass_info = self._analyze_over_bass(root, quality, bass)
            if bass_info:
                result.update(bass_info)
                return result

        # Standard chord quality lookup
        scale_info = QUALITY_SCALE_MAP.get(quality, QUALITY_SCALE_MAP.get('maj'))

        if scale_info:
            result['scales'] = [f"{root} {s}" for s in scale_info['primary']]
            result['secondary_scales'] = [f"{root} {s}" for s in scale_info['secondary']]
            result['tip'] = scale_info.get('tip', '')
        else:
            result['scales'] = [f"{root} Ionian (Major)"]
            result['secondary_scales'] = []
            result['tip'] = ''

        # Add function analysis if key is provided
        if key:
            result['function'] = self._get_function(root, quality, key)

        return result

    def _analyze_over_bass(self, root: str, quality: str, bass: str) -> Optional[Dict]:
        """Analyze a triad/chord over a specific bass note."""
        root_idx = self._note_index(root)
        bass_idx = self._note_index(bass)

        if root_idx == -1 or bass_idx == -1:
            return None

        # Calculate interval from bass to chord root
        interval = (root_idx - bass_idx) % 12

        # Simplify quality for lookup
        simple_quality = quality
        if quality in ('maj', 'maj7'):
            simple_quality = 'maj'
        elif quality in ('min', 'min7', 'min6'):
            simple_quality = 'min'

        # Look up in polychord table
        lookup = (interval, simple_quality)
        info = TRIAD_OVER_BASS.get(lookup)

        if info:
            return {
                'scales': [f"{bass} {s}" for s in info['scales']],
                'secondary_scales': [],
                'tip': info['tip'],
                'chord_type': info['type'],
                'intervals': info['intervals'],
                'generic_name': info.get('generic', ''),
            }

        # Fallback: use the chord root's scales
        return None

    def _get_function(self, root: str, quality: str, key: str) -> Optional[str]:
        """Determine the Roman numeral function of a chord in a key."""
        # Parse key
        key_root = key[0]
        key_rest = key[1:]
        if key_rest and (key_rest[0] == '#' or key_rest[0] == 'b'):
            key_root = key[:2]
            key_rest = key[2:]

        key_is_minor = 'm' in key_rest

        key_idx = self._note_index(key_root)
        chord_idx = self._note_index(root)

        if key_idx == -1 or chord_idx == -1:
            return None

        interval = (chord_idx - key_idx) % 12

        # Major key scale degrees
        if not key_is_minor:
            degree_map = {
                0: 'I', 2: 'ii', 4: 'iii', 5: 'IV',
                7: 'V', 9: 'vi', 11: 'vii°'
            }
        else:
            # Natural minor scale degrees
            degree_map = {
                0: 'i', 2: 'ii°', 3: 'III', 5: 'iv',
                7: 'v', 8: 'VI', 10: 'VII'
            }

        return degree_map.get(interval)

    def get_scales_for_progression(self, chords: List[str], key: Optional[str] = None) -> List[Dict]:
        """
        Analyze an entire chord progression and return scale suggestions for each chord.

        Args:
            chords: List of chord names
            key: Optional detected key

        Returns:
            List of analysis dicts, one per chord
        """
        return [self.analyze(chord, key) for chord in chords]

    def suggest_practice_approach(self, chords: List[str], key: Optional[str] = None) -> str:
        """
        Generate a practice suggestion based on the overall progression.

        Args:
            chords: List of chord names in the progression
            key: Optional detected key

        Returns:
            Practice suggestion string
        """
        if not chords:
            return "No chords detected yet."

        analyses = self.get_scales_for_progression(chords, key)

        # Find the most common scale across all chords
        scale_counts = {}
        for a in analyses:
            for s in a.get('scales', []):
                scale_counts[s] = scale_counts.get(s, 0) + 1

        if not scale_counts:
            return "Try the major or minor pentatonic scale as a starting point."

        # Most common scale
        top_scale = max(scale_counts, key=scale_counts.get)
        unique_chords = list(dict.fromkeys(chords))

        if len(unique_chords) <= 3:
            return f"Simple progression — {top_scale} works over most of it. Focus on chord tones for each change."
        elif len(unique_chords) <= 6:
            return f"Try {top_scale} as your home base, then adjust for specific chords. Listen for when you need to shift."
        else:
            return f"Complex progression — start with {top_scale}, but you'll need to follow the chord changes closely. Practice each transition."


# ============================================================================
# Convenience functions
# ============================================================================

def get_scale_suggestion(chord: str, key: Optional[str] = None) -> Dict:
    """Quick scale suggestion for a single chord."""
    engine = ChordTheoryEngine()
    return engine.analyze(chord, key)


def get_progression_analysis(chords: List[str], key: Optional[str] = None) -> List[Dict]:
    """Analyze a full chord progression."""
    engine = ChordTheoryEngine()
    return engine.get_scales_for_progression(chords, key)
