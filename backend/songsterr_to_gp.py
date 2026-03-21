"""
Songsterr JSON → Guitar Pro 5 (.gp5) converter.

Converts Songsterr's per-track JSON format into a GP5 file using pyguitarpro.
Handles notes, rests, bends, slides, hammer-ons, ties, let-ring, time signatures,
tempo, section markers, and tuning.
"""

import io
import logging
from typing import List, Dict, Any, Optional

import guitarpro
from guitarpro.models import (
    Song, Track, Measure, MeasureHeader, Beat, Voice, Note, NoteEffect,
    Duration, GuitarString, TimeSignature, Marker, MidiChannel,
    BendEffect, BendPoint, BendType, SlideType, Velocities,
    NoteType, BeatEffect, BeatStatus, Color, Chord,
)

logger = logging.getLogger(__name__)

# Songsterr duration type → guitarpro Duration.value
DURATION_MAP = {1: 1, 2: 2, 4: 4, 8: 8, 16: 16, 32: 32, 64: 64}

# Songsterr velocity strings → guitarpro velocity values
VELOCITY_MAP = {
    'ppp': Velocities.pianoPianissimo,
    'pp': Velocities.pianissimo,
    'p': Velocities.piano,
    'mp': Velocities.mezzoForte - 10,
    'mf': Velocities.mezzoForte,
    'f': Velocities.forte,
    'ff': Velocities.fortissimo,
    'fff': Velocities.forteFortissimo,
}

# Songsterr instrumentId → MIDI program
INSTRUMENT_MIDI = {
    25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31,
    33: 33, 34: 34, 35: 35, 0: 0,
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _detect_chord_name(notes_data: list, tuning: list) -> Optional[str]:
    """Detect chord name from a list of note dicts with string/fret."""
    frets = [(int(n['string']), int(n['fret'])) for n in notes_data
             if n.get('fret') is not None and n.get('string') is not None
             and not n.get('rest') and not n.get('tie') and not n.get('dead')]
    if len(frets) < 2:
        return None

    # Get sorted (midi, note_name) pairs
    midi_notes = []
    for string_idx, fret in frets:
        if string_idx < len(tuning):
            midi = tuning[string_idx] + fret
            midi_notes.append((midi, NOTE_NAMES[midi % 12]))
    midi_notes.sort()

    if len(midi_notes) < 2:
        return None

    note_names = list(dict.fromkeys(n for _, n in midi_notes))
    bass = note_names[0]

    best = None
    best_priority = -1

    for root in note_names:
        root_midi = NOTE_NAMES.index(root)
        intervals = set((NOTE_NAMES.index(n) - root_midi) % 12 for n in note_names)

        name = None
        priority = 0

        if {0, 4, 7} <= intervals:
            name = root + '7' if 10 in intervals else root + 'maj7' if 11 in intervals else root
            priority = 3
        elif {0, 3, 7} <= intervals:
            name = root + 'm7' if 10 in intervals else root + 'm'
            priority = 3
        elif {0, 5, 7} <= intervals:
            name = root + 'sus4'
            priority = 2
        elif {0, 2, 7} <= intervals:
            name = root + 'sus2'
            priority = 2
        elif {0, 3, 6} <= intervals:
            name = root + 'dim'
            priority = 2
        elif {0, 4, 8} <= intervals:
            name = root + 'aug'
            priority = 2
        elif intervals == {0, 7}:
            name = root + '5'
            priority = 1

        if name:
            p = priority + (10 if root == bass else 0)
            if p > best_priority:
                best_priority = p
                best = (name, root)

    if not best:
        return None

    name, root = best
    if root != bass and best_priority < 10:
        name += '/' + bass
    return name


def songsterr_to_gp5(
    track_data: List[Dict[str, Any]],
    title: str = "Unknown",
    artist: str = "Unknown",
    tempo: int = 120,
) -> bytes:
    """Convert Songsterr JSON track data to GP5 binary."""
    song = Song()
    song.title = title
    song.artist = artist
    song.tempo = tempo
    song.tracks = []
    song.measureHeaders = []

    max_measures = max(len(t.get('measures', [])) for t in track_data) if track_data else 0
    if max_measures == 0:
        return _empty_gp5(title, artist, tempo)

    # Build measure headers from first track's time signatures
    first_measures = track_data[0].get('measures', [])
    for i in range(max_measures):
        header = MeasureHeader(number=i + 1, start=960)

        if i < len(first_measures):
            sig = first_measures[i].get('signature')
            if sig and len(sig) == 2:
                header.timeSignature = TimeSignature()
                header.timeSignature.numerator = sig[0]
                header.timeSignature.denominator = Duration()
                header.timeSignature.denominator.value = sig[1]

            marker_data = first_measures[i].get('marker')
            if marker_data and marker_data.get('text'):
                header.marker = Marker(title=marker_data['text'], color=Color(255, 0, 0))

        song.measureHeaders.append(header)

    # Build each track
    colors = [
        Color(255, 0, 0), Color(0, 128, 255), Color(0, 200, 0),
        Color(255, 165, 0), Color(128, 0, 255), Color(255, 255, 0),
    ]

    for track_idx, tdata in enumerate(track_data):
        tuning = tdata.get('tuning', [64, 59, 55, 50, 45, 40])
        num_strings = tdata.get('strings', len(tuning))
        instrument_id = tdata.get('instrumentId', 27)
        midi_program = INSTRUMENT_MIDI.get(instrument_id, instrument_id)
        ch = track_idx if track_idx < 9 else track_idx + 1

        strings = [GuitarString(number=i + 1, value=v) for i, v in enumerate(tuning)]

        track = Track(
            song,
            number=track_idx + 1,
            name=tdata.get('name') or tdata.get('instrument', f'Track {track_idx + 1}'),
            strings=strings,
            fretCount=tdata.get('frets', 24),
            channel=MidiChannel(channel=ch, effectChannel=ch, instrument=midi_program),
            color=colors[track_idx % len(colors)],
        )
        track.measures = []

        measures_data = tdata.get('measures', [])
        for i in range(max_measures):
            header = song.measureHeaders[i]
            measure = Measure(track, header)

            if i < len(measures_data):
                _populate_measure(measure, measures_data[i], num_strings, tuning)
            else:
                _add_rest_measure(measure)

            track.measures.append(measure)

        song.tracks.append(track)

    buf = io.BytesIO()
    guitarpro.write(song, buf, version=(5, 1, 0))
    return buf.getvalue()


def _populate_measure(measure: Measure, mdata: Dict, num_strings: int,
                      tuning: Optional[list] = None):
    """Populate a guitarpro Measure from Songsterr measure data."""
    voices_data = mdata.get('voices', [])

    # GP5 needs exactly 2 voices
    while len(measure.voices) < 2:
        v = Voice(measure)
        measure.voices.append(v)

    for voice_idx, vdata in enumerate(voices_data[:2]):
        voice = measure.voices[voice_idx]
        voice.beats = []
        last_chord_name = None

        for bdata in vdata.get('beats', []):
            beat = _build_beat(bdata, voice, num_strings, tuning)
            # Deduplicate consecutive chord names
            if beat.effect and beat.effect.chord:
                if beat.effect.chord.name == last_chord_name:
                    beat.effect.chord = None
                else:
                    last_chord_name = beat.effect.chord.name
            voice.beats.append(beat)

        if not voice.beats:
            beat = Beat(voice, status=BeatStatus.rest)
            beat.duration = Duration()
            beat.duration.value = 1
            voice.beats.append(beat)

    # Fill empty second voice with rest
    if len(measure.voices) >= 2 and not measure.voices[1].beats:
        beat = Beat(measure.voices[1], status=BeatStatus.rest)
        beat.duration = Duration()
        beat.duration.value = 1
        measure.voices[1].beats.append(beat)


def _build_beat(bdata: Dict, voice: Voice, num_strings: int,
                tuning: Optional[list] = None) -> Beat:
    """Build a guitarpro Beat from Songsterr beat data."""
    dur_type = bdata.get('type', 4)

    beat = Beat(voice)
    beat.duration = Duration()
    beat.duration.value = DURATION_MAP.get(dur_type, 4)

    # Dotted duration
    dur_tuple = bdata.get('duration', [])
    if isinstance(dur_tuple, list) and len(dur_tuple) == 2:
        num, den = dur_tuple
        if num == 3 and den in (2, 4, 8, 16, 32):
            beat.duration.isDotted = True
            beat.duration.value = den // 2 if den > 2 else 1

    # Tuplet
    tuplet = bdata.get('tuplet')
    if tuplet and isinstance(tuplet, dict):
        enters = tuplet.get('enters', 1)
        times = tuplet.get('times', 1)
        if enters > 0 and times > 0:
            beat.duration.tuplet = guitarpro.models.Tuplet(enters, times)

    # Rest
    if bdata.get('rest'):
        beat.status = BeatStatus.rest
        return beat

    # Build notes
    beat.status = BeatStatus.normal
    velocity = VELOCITY_MAP.get(bdata.get('velocity', 'mf'), Velocities.mezzoForte)
    let_ring = bdata.get('letRing', False)

    for ndata in bdata.get('notes', []):
        if ndata.get('rest'):
            continue
        note = _build_note(ndata, beat, velocity, let_ring)
        if note:
            beat.notes.append(note)

    if not beat.notes:
        beat.status = BeatStatus.rest
    elif tuning and len(beat.notes) >= 2:
        # Detect chord name and attach to beat
        chord_name = _detect_chord_name(bdata.get('notes', []), tuning)
        if chord_name:
            chord = Chord(num_strings)
            chord.name = chord_name
            chord.firstFret = 0
            chord.strings = [-1] * num_strings
            for ndata in bdata.get('notes', []):
                s = ndata.get('string')
                f = ndata.get('fret')
                if s is not None and f is not None and int(s) < num_strings:
                    chord.strings[int(s)] = int(f)
            # Calculate firstFret from played frets
            played = [f for f in chord.strings if f > 0]
            if played:
                chord.firstFret = min(played)
            beat.effect = beat.effect or BeatEffect()
            beat.effect.chord = chord

    return beat


def _build_note(ndata: Dict, beat: Beat, velocity: int, let_ring: bool) -> Optional[Note]:
    """Build a guitarpro Note from Songsterr note data."""
    fret = ndata.get('fret')
    string = ndata.get('string')
    if fret is None or string is None:
        return None

    fret = int(fret)
    string = int(string)
    note = Note(beat, value=max(0, min(fret, 24)), string=string + 1, velocity=velocity)
    note.type = NoteType.normal
    note.effect = NoteEffect()

    if ndata.get('tie'):
        note.type = NoteType.tie
    if ndata.get('dead'):
        note.type = NoteType.dead
    if let_ring or ndata.get('letRing'):
        note.effect.letRing = True
    if ndata.get('hammerOn') or ndata.get('pullOff'):
        note.effect.hammer = True
    if ndata.get('vibrato'):
        note.effect.vibrato = True
    if ndata.get('palmMute'):
        note.effect.palmMute = True
    if ndata.get('staccato'):
        note.effect.staccato = True

    # Bend
    bend_data = ndata.get('bend')
    if bend_data:
        note.effect.bend = _build_bend(bend_data)

    # Slide
    slide = ndata.get('slide')
    if slide:
        slide_map = {
            'shiftSlide': SlideType.shiftSlideTo,
            'legSlide': SlideType.legatoSlideTo,
            'inFromAbove': SlideType.intoFromAbove,
            'inFromBelow': SlideType.intoFromBelow,
            'outDown': SlideType.outDownwards,
            'outUp': SlideType.outUpwards,
        }
        st = slide_map.get(slide, SlideType.shiftSlideTo) if isinstance(slide, str) else SlideType.shiftSlideTo
        note.effect.slides = [st]

    # Harmonic
    harmonic = ndata.get('harmonic')
    if harmonic:
        if harmonic == 'artificial':
            note.effect.harmonic = guitarpro.ArtificialHarmonic()
        elif harmonic == 'pinch':
            note.effect.harmonic = guitarpro.PinchHarmonic()
        else:
            note.effect.harmonic = guitarpro.NaturalHarmonic()

    return note


def _build_bend(bend_data: Dict) -> BendEffect:
    """Convert Songsterr bend to guitarpro BendEffect."""
    bend = BendEffect()
    tone = bend_data.get('tone', 100)
    points = bend_data.get('points', [])

    if points:
        bend.points = []
        for pt in points:
            pos = pt.get('position', 0)
            gp_pos = round(pos * 12 / 60) if pos <= 60 else 12
            bend.points.append(BendPoint(gp_pos, pt.get('tone', 0)))

        if len(points) >= 2:
            s, e = points[0].get('tone', 0), points[-1].get('tone', 0)
            if s == 0 and e > 0:
                bend.type = BendType.bend
            elif s > 0 and e == 0:
                bend.type = BendType.releaseUp
            elif s > 0 and e > 0:
                bend.type = BendType.bendRelease
            else:
                bend.type = BendType.bend
    else:
        bend.type = BendType.bend
        bend.points = [BendPoint(0, 0), BendPoint(6, tone), BendPoint(12, tone)]

    bend.value = tone
    return bend


def _add_rest_measure(measure: Measure):
    """Fill a measure with whole rests."""
    while len(measure.voices) < 2:
        measure.voices.append(Voice(measure))
    for voice in measure.voices:
        beat = Beat(voice, status=BeatStatus.rest)
        beat.duration = Duration()
        beat.duration.value = 1
        voice.beats = [beat]


def _empty_gp5(title: str, artist: str, tempo: int) -> bytes:
    """Create a minimal empty GP5 file."""
    song = Song()
    song.title = title
    song.artist = artist
    song.tempo = tempo
    buf = io.BytesIO()
    guitarpro.write(song, buf, version=(5, 1, 0))
    return buf.getvalue()
