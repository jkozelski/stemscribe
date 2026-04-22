"""
Lead Sheet Generator for StemScriber
=====================================
Generates professional lead sheets (Real Book style) as MusicXML:
- Slash notation (rhythm slashes, not pitched notes)
- Chord symbols above the staff (<harmony> elements)
- Bar lines, time signature, key signature
- Section labels as rehearsal marks (Verse, Chorus, Bridge)
- Repeat signs where sections repeat
- Clean layout for rendering via OSMD (OpenSheetMusicDisplay)

Input: chord events from chord_detector_v10 + audio path for tempo/beat detection
Output: .musicxml file
"""

import logging
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# music21 imports
# ---------------------------------------------------------------------------
try:
    from music21 import (
        stream, note, chord as m21chord, meter, key as m21key,
        tempo, metadata as m21_metadata, bar, expressions,
        duration as m21_duration, pitch, harmony, clef,
    )
    from music21 import instrument as m21_instrument
    MUSIC21_AVAILABLE = True
except ImportError:
    MUSIC21_AVAILABLE = False
    logger.warning("music21 not available - lead sheet generation disabled")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MeasureChord:
    """A chord quantized to a specific beat within a measure."""
    chord_symbol: str       # e.g. "Am7", "Cmaj7"
    root: str               # e.g. "A", "C"
    quality: str            # e.g. "m7", "maj7", ""
    beat: float             # beat position within measure (1-based)
    duration_beats: float   # how many beats this chord lasts


@dataclass
class MeasureInfo:
    """All chord data for a single measure."""
    measure_number: int
    chords: List[MeasureChord] = field(default_factory=list)
    section_label: Optional[str] = None   # "Verse", "Chorus", etc.
    is_repeat_start: bool = False
    is_repeat_end: bool = False


# ---------------------------------------------------------------------------
# Tempo / beat detection
# ---------------------------------------------------------------------------

def detect_tempo_and_beats(audio_path: str, duration_limit: float = 120.0
                           ) -> Tuple[float, float, List[float]]:
    """Detect tempo, time signature numerator, and beat times from audio.

    Returns:
        (bpm, beats_per_measure, beat_times)
    """
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True,
                             duration=duration_limit)
        tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        # librosa may return array
        if hasattr(tempo_est, '__len__'):
            bpm = float(tempo_est[0]) if len(tempo_est) > 0 else 120.0
        else:
            bpm = float(tempo_est)
        if bpm <= 0:
            bpm = 120.0

        beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

        # Simple time-signature heuristic: check if beats group in 3 or 4
        beats_per_measure = _estimate_beats_per_measure(y, sr, bpm)

        logger.info(f"Lead sheet: tempo={bpm:.1f} BPM, "
                    f"beats_per_measure={beats_per_measure}, "
                    f"{len(beat_times)} beats detected")
        return bpm, beats_per_measure, beat_times

    except Exception as e:
        logger.warning(f"Beat detection failed, using defaults: {e}")
        return 120.0, 4.0, []


def _estimate_beats_per_measure(y, sr, bpm: float) -> float:
    """Estimate whether the song is in 3/4, 4/4, or 6/8."""
    try:
        import librosa
        # Use onset strength for downbeat estimation
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        # Try to detect periodicity in onset strength
        # Autocorrelation approach
        import numpy as np
        ac = np.correlate(onset_env, onset_env, mode='full')
        ac = ac[len(ac) // 2:]  # take positive lags only
        if len(ac) < 10:
            return 4.0

        # Convert bpm to frames per beat
        hop_length = 512
        frames_per_beat = (60.0 / bpm) * sr / hop_length

        # Check strength at 3-beat and 4-beat periods
        period_3 = int(round(frames_per_beat * 3))
        period_4 = int(round(frames_per_beat * 4))

        if period_3 < len(ac) and period_4 < len(ac):
            strength_3 = ac[period_3]
            strength_4 = ac[period_4]
            if strength_3 > strength_4 * 1.2:
                return 3.0
        return 4.0

    except Exception:
        return 4.0


# ---------------------------------------------------------------------------
# Chord quantization — snap chord events to a beat/measure grid
# ---------------------------------------------------------------------------

def quantize_chords_to_measures(
    chord_events: List[Dict],
    bpm: float,
    beats_per_measure: float,
    total_duration: float,
) -> List[MeasureInfo]:
    """Quantize chord events (with timestamps) onto a measure grid.

    Args:
        chord_events: List of dicts with 'time', 'duration', 'chord', 'root', 'quality'
        bpm: Detected tempo
        beats_per_measure: Beats per measure (3 or 4 typically)
        total_duration: Total audio duration in seconds

    Returns:
        List of MeasureInfo, one per measure
    """
    if not chord_events:
        return []

    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * beats_per_measure

    # Figure out total number of measures
    last_chord = chord_events[-1]
    end_time = last_chord['time'] + last_chord['duration']
    end_time = max(end_time, total_duration) if total_duration > 0 else end_time
    num_measures = max(1, math.ceil(end_time / measure_duration))

    # Build empty measures
    measures: List[MeasureInfo] = []
    for i in range(num_measures):
        measures.append(MeasureInfo(measure_number=i + 1))

    # Place each chord event into its measure(s)
    for ev in chord_events:
        chord_name = ev.get('chord', 'N')
        if chord_name in ('N', 'X', ''):
            continue

        root = ev.get('root', '')
        quality = ev.get('quality', '')
        start_time = ev['time']
        end_time_ev = start_time + ev['duration']

        # Which measure does this chord start in?
        start_measure_idx = int(start_time / measure_duration)
        # What beat within that measure?
        beat_in_measure = ((start_time - start_measure_idx * measure_duration)
                           / beat_duration) + 1  # 1-based

        # Clamp to valid range
        start_measure_idx = min(start_measure_idx, num_measures - 1)
        beat_in_measure = max(1.0, min(beat_in_measure, beats_per_measure + 0.99))

        # Snap beat to nearest half-beat
        beat_in_measure = round(beat_in_measure * 2) / 2
        if beat_in_measure < 1.0:
            beat_in_measure = 1.0

        # How many beats does this chord span?
        dur_beats = ev['duration'] / beat_duration
        dur_beats = max(1.0, round(dur_beats * 2) / 2)  # snap to half-beats, min 1

        mc = MeasureChord(
            chord_symbol=chord_name,
            root=root,
            quality=quality,
            beat=beat_in_measure,
            duration_beats=dur_beats,
        )

        if start_measure_idx < len(measures):
            measures[start_measure_idx].chords.append(mc)

    # De-duplicate: if a measure has overlapping chords at same beat, keep first
    for m in measures:
        if len(m.chords) > 1:
            seen_beats = set()
            deduped = []
            for c in m.chords:
                beat_key = round(c.beat * 2)
                if beat_key not in seen_beats:
                    seen_beats.add(beat_key)
                    deduped.append(c)
            m.chords = deduped

    # Fill empty measures with the last known chord (sustain)
    last_chord_sym = None
    last_root = ''
    last_quality = ''
    for m in measures:
        if m.chords:
            last_chord_sym = m.chords[-1].chord_symbol
            last_root = m.chords[-1].root
            last_quality = m.chords[-1].quality
        elif last_chord_sym:
            m.chords.append(MeasureChord(
                chord_symbol=last_chord_sym,
                root=last_root,
                quality=last_quality,
                beat=1.0,
                duration_beats=beats_per_measure,
            ))

    return measures


# ---------------------------------------------------------------------------
# Section detection — find Verse, Chorus, Bridge patterns
# ---------------------------------------------------------------------------

def detect_sections(measures: List[MeasureInfo], beats_per_measure: float
                    ) -> List[MeasureInfo]:
    """Detect repeated chord patterns and label sections.

    Uses a simple approach: look for repeated 4- or 8-measure patterns.
    Labels them as Verse (A), Chorus (B), Bridge (C), etc.
    Also marks repeat signs on repeated sections.
    """
    if len(measures) < 4:
        if measures:
            measures[0].section_label = "A"
        return measures

    # Build a simplified chord string per measure for pattern matching
    def measure_sig(m: MeasureInfo) -> str:
        if not m.chords:
            return "N"
        return "|".join(c.chord_symbol for c in m.chords)

    sigs = [measure_sig(m) for m in measures]

    # Try to find repeating patterns of length 4, 8, then 2
    best_pattern_len = _find_best_pattern_length(sigs)

    if best_pattern_len == 0:
        # No clear pattern, just label as one big section
        measures[0].section_label = "A"
        return measures

    # Group measures into sections of best_pattern_len
    section_sigs = {}
    section_names = []
    label_counter = 0
    SECTION_LABELS = ["Verse", "Chorus", "Bridge", "Interlude",
                      "Outro", "Solo", "Pre-Chorus", "Hook"]

    for i in range(0, len(measures), best_pattern_len):
        chunk = tuple(sigs[i:i + best_pattern_len])
        if len(chunk) < best_pattern_len:
            # Partial final section
            chunk_key = chunk
        else:
            chunk_key = chunk

        if chunk_key not in section_sigs:
            if label_counter < len(SECTION_LABELS):
                name = SECTION_LABELS[label_counter]
            else:
                name = f"Section {label_counter + 1}"
            section_sigs[chunk_key] = name
            label_counter += 1

        section_name = section_sigs[chunk_key]
        section_names.append((i, section_name))

        # Set section label on first measure of this group
        if i < len(measures):
            measures[i].section_label = section_name

    # Mark repeats: if same section appears consecutively, use repeat signs
    prev_name = None
    repeat_start_idx = None
    for idx, (meas_idx, name) in enumerate(section_names):
        if name == prev_name:
            # Consecutive same section — mark repeat
            if repeat_start_idx is not None:
                # Mark the start of the first occurrence
                measures[repeat_start_idx].is_repeat_start = True
                # Mark end of first occurrence
                end_of_first = min(meas_idx - 1, len(measures) - 1)
                measures[end_of_first].is_repeat_end = True
                # Remove the section label from the repeat
                measures[meas_idx].section_label = None
                repeat_start_idx = None
        else:
            repeat_start_idx = meas_idx
            prev_name = name

    return measures


def _find_best_pattern_length(sigs: List[str]) -> int:
    """Find the most common repeating pattern length in measure signatures."""
    n = len(sigs)
    if n < 4:
        return 0

    best_len = 0
    best_score = 0

    for pattern_len in [8, 4, 2, 16]:
        if pattern_len > n // 2:
            continue

        matches = 0
        total_comparisons = 0

        for i in range(0, n - pattern_len, pattern_len):
            for j in range(i + pattern_len, n, pattern_len):
                chunk_a = sigs[i:i + pattern_len]
                chunk_b = sigs[j:j + pattern_len]
                if len(chunk_b) < pattern_len:
                    continue
                total_comparisons += 1
                if chunk_a == chunk_b:
                    matches += 1

        if total_comparisons > 0:
            score = matches / total_comparisons
            # Prefer longer patterns if they have decent match rate
            weighted_score = score * (pattern_len ** 0.3)
            if weighted_score > best_score:
                best_score = weighted_score
                best_len = pattern_len

    # Only accept if we found at least some repetition
    if best_score < 0.1:
        return 4  # Default to 4-measure grouping

    return best_len


# ---------------------------------------------------------------------------
# MusicXML generation via music21
# ---------------------------------------------------------------------------

def _parse_chord_kind(quality: str, chord_symbol: str) -> str:
    """Map chord quality string to MusicXML chord-kind value."""
    # music21 harmony.ChordSymbol handles most of this, but we need to
    # make sure the symbol is parseable
    q = quality.lower() if quality else ''

    # Common mappings for music21's ChordSymbol parser
    kind_map = {
        '': 'major',
        'maj': 'major',
        'm': 'minor',
        'min': 'minor',
        '7': 'dominant-seventh',
        'maj7': 'major-seventh',
        'm7': 'minor-seventh',
        'min7': 'minor-seventh',
        'dim': 'diminished',
        'dim7': 'diminished-seventh',
        'aug': 'augmented',
        'sus4': 'suspended-fourth',
        'sus2': 'suspended-second',
        '6': 'major-sixth',
        'm6': 'minor-sixth',
        '9': 'dominant-ninth',
        'maj9': 'major-ninth',
        'm9': 'minor-ninth',
        'add9': 'major',  # simplified
        'mmaj7': 'minor-major-seventh',
        'hdim7': 'half-diminished-seventh',
    }

    return kind_map.get(q, 'major')


def _make_slash_note(duration_beats: float) -> note.Note:
    """Create a single slash-notation note (unpitched, on middle line).

    Uses B4 as the pitch (middle of treble staff) with noteheadFill=False
    for slash appearance. OSMD renders these as rhythm slashes.
    """
    n = note.Note('B4')
    n.duration = m21_duration.Duration(duration_beats)
    n.notehead = 'slash'
    n.stemDirection = 'noStem'
    # Hide the stem for cleaner slash notation
    n.style.hideObjectOnPrint = False
    return n


def _create_chord_symbol(chord_name: str, root: str, quality: str
                         ) -> Optional['harmony.ChordSymbol']:
    """Create a music21 ChordSymbol from our chord data.

    Tries the full chord name first, falls back to root + kind.
    """
    if not root or chord_name in ('N', 'X', ''):
        return None

    try:
        # music21's ChordSymbol can parse standard chord names
        cs = harmony.ChordSymbol(chord_name)
        return cs
    except Exception:
        pass

    # Fallback: construct from root + kind
    try:
        cs = harmony.ChordSymbol()
        cs.root(pitch.Pitch(root))
        cs.chordKind = _parse_chord_kind(quality, chord_name)
        return cs
    except Exception as e:
        logger.debug(f"Could not create ChordSymbol for '{chord_name}': {e}")
        return None


def generate_lead_sheet_musicxml(
    chord_events: List[Dict],
    audio_path: str,
    output_path: str,
    title: str = "Lead Sheet",
    artist: str = "",
    detected_key: Optional[str] = None,
) -> Optional[str]:
    """Generate a professional lead sheet as MusicXML.

    Args:
        chord_events: List of chord dicts with 'time', 'duration', 'chord',
                      'root', 'quality', 'confidence'
        audio_path: Path to audio file for tempo/beat detection
        output_path: Where to write the .musicxml file
        title: Song title
        artist: Artist/composer name
        detected_key: Pre-detected key (e.g. 'Am', 'C', 'F#m')

    Returns:
        Path to the generated .musicxml file, or None on failure
    """
    if not MUSIC21_AVAILABLE:
        logger.error("music21 not installed - cannot generate lead sheet")
        return None

    if not chord_events:
        logger.warning("No chord events provided - cannot generate lead sheet")
        return None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating lead sheet: {title} ({len(chord_events)} chord events)")

    try:
        # 1. Detect tempo and time signature from audio
        bpm, beats_per_measure, beat_times = detect_tempo_and_beats(audio_path)

        # Total audio duration
        try:
            import librosa
            total_duration = librosa.get_duration(path=audio_path)
        except Exception:
            last_ev = chord_events[-1]
            total_duration = last_ev['time'] + last_ev['duration']

        # 2. Quantize chords to measures
        measures = quantize_chords_to_measures(
            chord_events, bpm, beats_per_measure, total_duration
        )
        logger.info(f"Quantized to {len(measures)} measures")

        # 3. Detect sections (Verse, Chorus, Bridge)
        measures = detect_sections(measures, beats_per_measure)

        # 4. Build MusicXML score
        score = _build_score(
            measures=measures,
            bpm=bpm,
            beats_per_measure=beats_per_measure,
            title=title,
            artist=artist,
            detected_key=detected_key,
        )

        # 5. Write MusicXML
        score.write('musicxml', fp=str(output_path))
        logger.info(f"Lead sheet saved: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"Lead sheet generation failed: {e}", exc_info=True)
        return None


def _build_score(
    measures: List[MeasureInfo],
    bpm: float,
    beats_per_measure: float,
    title: str,
    artist: str,
    detected_key: Optional[str],
) -> 'stream.Score':
    """Build a music21 Score with slash notation, chord symbols, and sections."""

    score = stream.Score()

    # Metadata
    md = m21_metadata.Metadata()
    md.title = title
    md.composer = artist
    score.metadata = md

    # Create a single part for the lead sheet
    part = stream.Part()
    part.partName = "Lead Sheet"
    part.partAbbreviation = "L.S."

    # Use treble clef (standard for lead sheets)
    part.insert(0, clef.TrebleClef())

    # Time signature
    beats_int = int(beats_per_measure)
    ts = meter.TimeSignature(f'{beats_int}/4')
    part.insert(0, ts)

    # Tempo marking
    mm = tempo.MetronomeMark(number=round(bpm))
    part.insert(0, mm)

    # Key signature
    if detected_key and detected_key != 'Unknown':
        try:
            ks = m21key.Key(detected_key)
            part.insert(0, ks)
        except Exception:
            pass

    # Build measures
    for minfo in measures:
        m = stream.Measure(number=minfo.measure_number)

        # Section label as rehearsal mark
        if minfo.section_label:
            rehearsal = expressions.RehearsalMark(minfo.section_label)
            rehearsal.style.fontStyle = 'bold'
            m.insert(0, rehearsal)

        # Repeat signs
        if minfo.is_repeat_start:
            m.leftBarline = bar.Repeat(direction='start')
        if minfo.is_repeat_end:
            m.rightBarline = bar.Repeat(direction='end')

        # Add chord symbols and slash notes
        if minfo.chords:
            current_beat_offset = 0.0  # in quarter notes from start of measure

            for i, mc in enumerate(minfo.chords):
                # Chord symbol at this beat position
                cs = _create_chord_symbol(mc.chord_symbol, mc.root, mc.quality)

                # Calculate offset (beat is 1-based, offset is 0-based)
                offset = mc.beat - 1.0

                if cs is not None:
                    m.insert(offset, cs)

                # Figure out duration for this chord's slash notes
                if i + 1 < len(minfo.chords):
                    # Duration until next chord in this measure
                    next_beat = minfo.chords[i + 1].beat
                    slash_dur = next_beat - mc.beat
                else:
                    # Duration until end of measure
                    slash_dur = beats_per_measure - (mc.beat - 1.0)

                slash_dur = max(1.0, slash_dur)  # minimum 1 beat

                # Create slash notes to fill this duration
                # Use whole-beat slashes for clean notation
                remaining = slash_dur
                beat_pos = offset
                while remaining > 0.01:
                    if remaining >= 4.0 and beats_per_measure >= 4:
                        note_dur = 4.0
                    elif remaining >= 2.0:
                        note_dur = 2.0
                    elif remaining >= 1.0:
                        note_dur = 1.0
                    else:
                        note_dur = remaining

                    slash = _make_slash_note(note_dur)
                    m.insert(beat_pos, slash)
                    beat_pos += note_dur
                    remaining -= note_dur
        else:
            # Empty measure — fill with whole-measure slash rest
            slash = _make_slash_note(beats_per_measure)
            m.insert(0, slash)

        part.append(m)

    score.insert(0, part)
    return score


# ---------------------------------------------------------------------------
# Convenience: generate from a ProcessingJob
# ---------------------------------------------------------------------------

def generate_lead_sheet_for_job(job, audio_path: str) -> Optional[str]:
    """Generate a lead sheet from a ProcessingJob's chord data.

    Args:
        job: ProcessingJob instance (must have chord_progression populated)
        audio_path: Path to the original audio file

    Returns:
        Path to the .musicxml file, or None on failure
    """
    from models.job import OUTPUT_DIR

    if not job.chord_progression:
        logger.info("No chord progression in job - skipping lead sheet")
        return None

    output_dir = OUTPUT_DIR / job.job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lead_sheet.musicxml"

    title = "Lead Sheet"
    artist = ""
    if job.metadata:
        title = job.metadata.get('title', job.filename or 'Lead Sheet')
        artist = job.metadata.get('artist', '')

    result = generate_lead_sheet_musicxml(
        chord_events=job.chord_progression,
        audio_path=audio_path,
        output_path=str(output_path),
        title=title,
        artist=artist,
        detected_key=job.detected_key,
    )

    if result:
        # Store path on the job for frontend access
        if not hasattr(job, 'lead_sheet_path'):
            job.lead_sheet_path = None
        job.lead_sheet_path = result
        logger.info(f"Lead sheet generated: {result}")

    return result


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s %(name)s: %(message)s')

    print(f"music21 available: {MUSIC21_AVAILABLE}")

    if not MUSIC21_AVAILABLE:
        print("Install music21: pip install music21")
        sys.exit(1)

    # Quick test with synthetic chord data
    test_chords = [
        {'time': 0.0,  'duration': 2.0, 'chord': 'Am7',  'root': 'A', 'quality': 'm7',  'confidence': 0.9},
        {'time': 2.0,  'duration': 2.0, 'chord': 'Dm7',  'root': 'D', 'quality': 'm7',  'confidence': 0.85},
        {'time': 4.0,  'duration': 2.0, 'chord': 'G7',   'root': 'G', 'quality': '7',   'confidence': 0.88},
        {'time': 6.0,  'duration': 2.0, 'chord': 'Cmaj7', 'root': 'C', 'quality': 'maj7', 'confidence': 0.92},
        {'time': 8.0,  'duration': 2.0, 'chord': 'Fmaj7', 'root': 'F', 'quality': 'maj7', 'confidence': 0.87},
        {'time': 10.0, 'duration': 2.0, 'chord': 'Bm7b5', 'root': 'B', 'quality': 'hdim7', 'confidence': 0.80},
        {'time': 12.0, 'duration': 2.0, 'chord': 'E7',    'root': 'E', 'quality': '7',    'confidence': 0.91},
        {'time': 14.0, 'duration': 2.0, 'chord': 'Am7',   'root': 'A', 'quality': 'm7',   'confidence': 0.93},
        # Repeat the progression
        {'time': 16.0, 'duration': 2.0, 'chord': 'Am7',  'root': 'A', 'quality': 'm7',  'confidence': 0.9},
        {'time': 18.0, 'duration': 2.0, 'chord': 'Dm7',  'root': 'D', 'quality': 'm7',  'confidence': 0.85},
        {'time': 20.0, 'duration': 2.0, 'chord': 'G7',   'root': 'G', 'quality': '7',   'confidence': 0.88},
        {'time': 22.0, 'duration': 2.0, 'chord': 'Cmaj7', 'root': 'C', 'quality': 'maj7', 'confidence': 0.92},
        {'time': 24.0, 'duration': 2.0, 'chord': 'Fmaj7', 'root': 'F', 'quality': 'maj7', 'confidence': 0.87},
        {'time': 26.0, 'duration': 2.0, 'chord': 'Bm7b5', 'root': 'B', 'quality': 'hdim7', 'confidence': 0.80},
        {'time': 28.0, 'duration': 2.0, 'chord': 'E7',    'root': 'E', 'quality': '7',    'confidence': 0.91},
        {'time': 30.0, 'duration': 2.0, 'chord': 'Am7',   'root': 'A', 'quality': 'm7',   'confidence': 0.93},
    ]

    out = Path("/tmp/test_lead_sheet.musicxml")

    # Test without audio (uses defaults for tempo)
    # We'll manually set tempo since there's no audio file
    bpm = 120.0
    beats_per_measure = 4.0
    total_duration = 32.0

    measures = quantize_chords_to_measures(test_chords, bpm, beats_per_measure, total_duration)
    measures = detect_sections(measures, beats_per_measure)

    print(f"\nQuantized to {len(measures)} measures:")
    for m in measures:
        chords_str = " | ".join(f"{c.chord_symbol}(beat {c.beat})" for c in m.chords)
        label = f" [{m.section_label}]" if m.section_label else ""
        repeat = ""
        if m.is_repeat_start:
            repeat += " |:"
        if m.is_repeat_end:
            repeat += " :|"
        print(f"  M{m.measure_number}: {chords_str}{label}{repeat}")

    score = _build_score(
        measures=measures,
        bpm=bpm,
        beats_per_measure=beats_per_measure,
        title="Test Lead Sheet",
        artist="StemScriber",
        detected_key="Am",
    )

    score.write('musicxml', fp=str(out))
    print(f"\nWrote: {out}")
    print(f"File size: {out.stat().st_size} bytes")
    print("Open in MuseScore, Finale, or render with OSMD to verify.")
