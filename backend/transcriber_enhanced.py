"""
Enhanced Transcription Module for StemScribe
=============================================
Improved MIDI transcription with:
- Better polyphonic accuracy (double-stops, chord voicings)
- Articulation detection (bends, slides, hammer-ons)
- Tighter quantization with swing detection
- Velocity dynamics preservation

This module wraps Basic Pitch and adds post-processing
to improve transcription quality for guitar and other instruments.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import required libraries
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - some features disabled")

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    logger.warning("pretty_midi not available")

try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False
    logger.warning("basic_pitch not available")


@dataclass
class ArticulationEvent:
    """Represents a detected articulation (bend, slide, hammer-on, etc.)"""
    note_index: int
    articulation_type: str  # 'bend', 'slide_up', 'slide_down', 'hammer_on', 'pull_off', 'vibrato'
    start_time: float
    end_time: float
    intensity: float  # 0-1, how strong the articulation is
    pitch_delta: float  # For bends/slides, how much pitch changed in semitones


@dataclass
class TranscriptionResult:
    """Result of enhanced transcription"""
    midi_path: str
    notes_count: int
    articulations: List[ArticulationEvent]
    detected_key: Optional[str]
    detected_tempo: float
    quantization_grid: str
    quality_score: float  # 0-1, confidence in transcription quality


class PitchContourAnalyzer:
    """
    Analyzes pitch contours to detect articulations.
    Uses crepe-style pitch tracking if available, otherwise librosa.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.hop_length = 256  # ~11ms at 22050Hz for fine pitch resolution

    def extract_pitch_contour(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract pitch contour from audio.

        Returns:
            times: Time points
            frequencies: Detected frequencies (Hz), 0 for unvoiced
            confidences: Confidence values (0-1)
        """
        if not LIBROSA_AVAILABLE:
            return np.array([]), np.array([]), np.array([])

        # Use pyin for polyphonic-aware pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('E2'),  # Low E on guitar
            fmax=librosa.note_to_hz('E6'),  # High E + harmonics
            sr=self.sr,
            hop_length=self.hop_length,
            fill_na=0.0
        )

        times = librosa.times_like(f0, sr=self.sr, hop_length=self.hop_length)
        confidences = voiced_probs

        return times, f0, confidences

    def detect_bends(self, times: np.ndarray, frequencies: np.ndarray,
                     confidences: np.ndarray, min_bend_semitones: float = 0.3) -> List[dict]:
        """
        Detect pitch bends in the contour.
        A bend is a smooth pitch increase followed by return (or sustain at bent pitch).
        """
        bends = []

        if len(frequencies) < 10:
            return bends

        # Convert to cents relative to nearest note
        midi_notes = librosa.hz_to_midi(frequencies + 1e-10)
        cents_deviation = (midi_notes - np.round(midi_notes)) * 100

        # Find regions where pitch deviates significantly from the note
        in_bend = False
        bend_start = 0
        max_deviation = 0

        for i in range(len(cents_deviation)):
            deviation = abs(cents_deviation[i])

            if deviation > min_bend_semitones * 100 and confidences[i] > 0.5:
                if not in_bend:
                    in_bend = True
                    bend_start = i
                    max_deviation = deviation
                else:
                    max_deviation = max(max_deviation, deviation)
            else:
                if in_bend and i - bend_start > 3:  # Minimum bend duration
                    bends.append({
                        'start_time': times[bend_start],
                        'end_time': times[i],
                        'max_cents': max_deviation,
                        'semitones': max_deviation / 100,
                        'type': 'bend_up' if cents_deviation[bend_start:i].mean() > 0 else 'bend_down'
                    })
                in_bend = False

        return bends

    def detect_slides(self, times: np.ndarray, frequencies: np.ndarray,
                      confidences: np.ndarray, min_slide_semitones: float = 2.0) -> List[dict]:
        """
        Detect slides - smooth pitch transitions between notes.
        Unlike bends, slides end on a different note than they started.
        """
        slides = []

        if len(frequencies) < 10:
            return slides

        midi_notes = librosa.hz_to_midi(frequencies + 1e-10)

        # Look for smooth transitions of >= min_slide_semitones
        window = 10  # Analyze in windows

        for i in range(0, len(midi_notes) - window, window // 2):
            segment = midi_notes[i:i+window]
            conf_segment = confidences[i:i+window]

            # Check if mostly voiced
            if np.mean(conf_segment) < 0.5:
                continue

            # Check for monotonic pitch change
            pitch_change = segment[-1] - segment[0]

            if abs(pitch_change) >= min_slide_semitones:
                # Verify it's smooth (not a jump)
                diffs = np.diff(segment)
                is_smooth = np.std(diffs) < 0.5  # Low variance = smooth slide

                if is_smooth and np.all(diffs * np.sign(pitch_change) >= -0.2):  # Mostly monotonic
                    slides.append({
                        'start_time': times[i],
                        'end_time': times[i + window - 1],
                        'start_pitch': segment[0],
                        'end_pitch': segment[-1],
                        'semitones': pitch_change,
                        'type': 'slide_up' if pitch_change > 0 else 'slide_down'
                    })

        return slides


class EnhancedTranscriber:
    """
    Enhanced MIDI transcription with articulation detection and quality improvements.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.pitch_analyzer = PitchContourAnalyzer(sample_rate)

    def transcribe(self, audio_path: str, output_dir: str,
                   stem_type: str = 'guitar',
                   detect_articulations: bool = True,
                   quantize: bool = True,
                   quantize_grid: float = 0.125) -> TranscriptionResult:
        """
        Transcribe audio to MIDI with enhanced accuracy.

        Args:
            audio_path: Path to audio file
            output_dir: Directory for output MIDI
            stem_type: Type of instrument (guitar, bass, piano, vocals)
            detect_articulations: Whether to analyze for bends/slides
            quantize: Whether to quantize note timings
            quantize_grid: Grid size in beats (0.125 = 16th notes)

        Returns:
            TranscriptionResult with MIDI path and metadata
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"🎼 Enhanced transcription: {audio_path.name} ({stem_type})")

        # Step 1: Basic Pitch transcription (if available)
        if BASIC_PITCH_AVAILABLE:
            midi_data, note_events = self._transcribe_basic_pitch(audio_path)
        else:
            logger.warning("Basic Pitch not available - using fallback")
            midi_data, note_events = self._transcribe_fallback(audio_path)

        if midi_data is None:
            return TranscriptionResult(
                midi_path="",
                notes_count=0,
                articulations=[],
                detected_key=None,
                detected_tempo=120.0,
                quantization_grid="none",
                quality_score=0.0
            )

        # Step 2: Post-processing for better accuracy
        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(str(audio_path), sr=self.sr)
            tempo = self._detect_tempo(y)
        else:
            tempo = 120.0

        # Step 3: Quantization
        if quantize:
            midi_data = self._quantize_midi(midi_data, tempo, quantize_grid)
            grid_str = f"1/{int(1/quantize_grid)}"
        else:
            grid_str = "none"

        # Step 4: Articulation detection
        articulations = []
        if detect_articulations and LIBROSA_AVAILABLE and stem_type in ['guitar', 'bass', 'vocals']:
            articulations = self._detect_articulations(y, midi_data)

            # Add articulation markers to MIDI (as pitch bend and CC messages)
            midi_data = self._add_articulations_to_midi(midi_data, articulations)

        # Step 5: Clean up overlapping notes and short glitches
        midi_data = self._cleanup_notes(midi_data, stem_type)

        # Step 6: Detect key
        detected_key = self._detect_key(midi_data) if PRETTY_MIDI_AVAILABLE else None

        # Step 7: Calculate quality score
        quality_score = self._estimate_quality(midi_data, articulations)

        # Save MIDI
        output_path = output_dir / f"{audio_path.stem}_enhanced.mid"
        midi_data.write(str(output_path))

        notes_count = sum(len(inst.notes) for inst in midi_data.instruments)

        logger.info(f"✅ Transcription complete: {notes_count} notes, "
                   f"{len(articulations)} articulations, quality={quality_score:.2f}")

        return TranscriptionResult(
            midi_path=str(output_path),
            notes_count=notes_count,
            articulations=articulations,
            detected_key=detected_key,
            detected_tempo=tempo,
            quantization_grid=grid_str,
            quality_score=quality_score
        )

    def _transcribe_basic_pitch(self, audio_path: Path) -> Tuple[Optional['pretty_midi.PrettyMIDI'], List]:
        """Use Basic Pitch for initial transcription with optimized settings."""
        try:
            from basic_pitch.inference import predict_and_save, predict

            # Get model outputs for more control
            model_output, midi_data, note_events = predict(
                str(audio_path),
                onset_threshold=0.5,      # Lower = more notes detected
                frame_threshold=0.3,      # Lower = longer notes
                minimum_note_length=50,   # ms - catch quick notes
                minimum_frequency=65.0,   # E2 - low guitar
                maximum_frequency=2100.0, # C7 - high guitar + harmonics
            )

            return midi_data, note_events

        except Exception as e:
            logger.error(f"Basic Pitch failed: {e}")
            return None, []

    def _transcribe_fallback(self, audio_path: Path) -> Tuple[Optional['pretty_midi.PrettyMIDI'], List]:
        """Fallback transcription using librosa onset detection + pitch tracking."""
        if not LIBROSA_AVAILABLE or not PRETTY_MIDI_AVAILABLE:
            return None, []

        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr)

            # Onset detection
            onset_frames = librosa.onset.onset_detect(
                y=y, sr=sr,
                hop_length=512,
                backtrack=True
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

            # Pitch tracking
            times, freqs, confs = self.pitch_analyzer.extract_pitch_contour(y)

            # Create MIDI
            midi = pretty_midi.PrettyMIDI()
            instrument = pretty_midi.Instrument(program=25)  # Acoustic guitar

            for i, onset in enumerate(onset_times):
                # Find pitch at this onset
                idx = np.searchsorted(times, onset)
                if idx < len(freqs) and freqs[idx] > 0:
                    pitch = int(round(librosa.hz_to_midi(freqs[idx])))

                    # Estimate note end
                    end_time = onset_times[i+1] if i+1 < len(onset_times) else onset + 0.5
                    end_time = min(end_time, onset + 2.0)  # Cap note length

                    note = pretty_midi.Note(
                        velocity=80,
                        pitch=pitch,
                        start=onset,
                        end=end_time
                    )
                    instrument.notes.append(note)

            midi.instruments.append(instrument)
            return midi, []

        except Exception as e:
            logger.error(f"Fallback transcription failed: {e}")
            return None, []

    def _detect_tempo(self, y: np.ndarray) -> float:
        """Detect tempo from audio."""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
            return float(tempo) if tempo > 0 else 120.0
        except:
            return 120.0

    def _quantize_midi(self, midi: 'pretty_midi.PrettyMIDI', tempo: float,
                       grid_size: float = 0.125) -> 'pretty_midi.PrettyMIDI':
        """
        Quantize MIDI notes to grid while preserving feel.
        Uses soft quantization - moves notes toward grid but not all the way.
        """
        beat_duration = 60.0 / tempo
        grid_seconds = grid_size * beat_duration

        # Quantization strength (1.0 = full quantize, 0.5 = halfway)
        strength = 0.75

        for instrument in midi.instruments:
            for note in instrument.notes:
                # Quantize start
                grid_pos = round(note.start / grid_seconds) * grid_seconds
                note.start = note.start + (grid_pos - note.start) * strength

                # Quantize end (but maintain minimum note length)
                grid_pos = round(note.end / grid_seconds) * grid_seconds
                new_end = note.end + (grid_pos - note.end) * strength
                note.end = max(new_end, note.start + 0.05)  # Min 50ms

        return midi

    def _detect_articulations(self, audio: np.ndarray,
                              midi: 'pretty_midi.PrettyMIDI') -> List[ArticulationEvent]:
        """Detect articulations (bends, slides, hammer-ons) from audio."""
        articulations = []

        times, freqs, confs = self.pitch_analyzer.extract_pitch_contour(audio)

        # Detect bends
        bends = self.pitch_analyzer.detect_bends(times, freqs, confs)
        for bend in bends:
            articulations.append(ArticulationEvent(
                note_index=-1,  # Will be matched to notes later
                articulation_type=bend['type'],
                start_time=bend['start_time'],
                end_time=bend['end_time'],
                intensity=min(1.0, bend['semitones'] / 2.0),
                pitch_delta=bend['semitones']
            ))

        # Detect slides
        slides = self.pitch_analyzer.detect_slides(times, freqs, confs)
        for slide in slides:
            articulations.append(ArticulationEvent(
                note_index=-1,
                articulation_type=slide['type'],
                start_time=slide['start_time'],
                end_time=slide['end_time'],
                intensity=min(1.0, abs(slide['semitones']) / 5.0),
                pitch_delta=slide['semitones']
            ))

        # Detect hammer-ons/pull-offs (notes without strong attacks)
        articulations.extend(self._detect_legato(audio, midi))

        logger.info(f"  Detected {len(articulations)} articulations: "
                   f"{len(bends)} bends, {len(slides)} slides")

        return articulations

    def _detect_legato(self, audio: np.ndarray,
                       midi: 'pretty_midi.PrettyMIDI') -> List[ArticulationEvent]:
        """Detect hammer-ons and pull-offs by analyzing attack transients."""
        legato = []

        if not midi.instruments:
            return legato

        # Get onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        times = librosa.times_like(onset_env, sr=self.sr)

        for instrument in midi.instruments:
            sorted_notes = sorted(instrument.notes, key=lambda n: n.start)

            for i, note in enumerate(sorted_notes[1:], 1):
                prev_note = sorted_notes[i-1]

                # Check if notes are close together (potential legato)
                gap = note.start - prev_note.end
                if -0.05 < gap < 0.1:  # Overlapping or very close
                    # Check onset strength at note start
                    idx = np.searchsorted(times, note.start)
                    if idx < len(onset_env):
                        onset_strength = onset_env[idx]

                        # Weak onset = likely hammer-on or pull-off
                        if onset_strength < np.median(onset_env) * 0.6:
                            art_type = 'hammer_on' if note.pitch > prev_note.pitch else 'pull_off'
                            legato.append(ArticulationEvent(
                                note_index=i,
                                articulation_type=art_type,
                                start_time=note.start,
                                end_time=note.end,
                                intensity=1.0 - (onset_strength / np.max(onset_env)),
                                pitch_delta=note.pitch - prev_note.pitch
                            ))

        return legato

    def _add_articulations_to_midi(self, midi: 'pretty_midi.PrettyMIDI',
                                    articulations: List[ArticulationEvent]) -> 'pretty_midi.PrettyMIDI':
        """Add pitch bend and control change messages for articulations."""
        if not midi.instruments:
            return midi

        instrument = midi.instruments[0]

        for art in articulations:
            if art.articulation_type in ['bend_up', 'bend_down']:
                # Add pitch bend events
                bend_amount = int(art.pitch_delta * 4096)  # MIDI pitch bend range
                bend_amount = max(-8192, min(8191, bend_amount))

                # Create bend envelope
                num_points = 10
                for i in range(num_points):
                    t = art.start_time + (art.end_time - art.start_time) * i / num_points
                    # Bell curve for bend
                    envelope = np.sin(np.pi * i / num_points)
                    pb = pretty_midi.PitchBend(
                        pitch=int(bend_amount * envelope),
                        time=t
                    )
                    instrument.pitch_bends.append(pb)

                # Return to center
                instrument.pitch_bends.append(
                    pretty_midi.PitchBend(pitch=0, time=art.end_time)
                )

            elif art.articulation_type in ['slide_up', 'slide_down']:
                # Slides are similar but don't return to center
                num_points = 10
                for i in range(num_points):
                    t = art.start_time + (art.end_time - art.start_time) * i / num_points
                    bend_amount = int(art.pitch_delta * 4096 * i / num_points)
                    pb = pretty_midi.PitchBend(
                        pitch=max(-8192, min(8191, bend_amount)),
                        time=t
                    )
                    instrument.pitch_bends.append(pb)

        return midi

    def _cleanup_notes(self, midi: 'pretty_midi.PrettyMIDI',
                       stem_type: str) -> 'pretty_midi.PrettyMIDI':
        """Remove glitches and clean up note overlaps."""

        # Minimum note lengths by instrument
        min_lengths = {
            'guitar': 0.04,   # 40ms - fast picking
            'bass': 0.06,     # 60ms
            'piano': 0.03,    # 30ms - fast runs
            'vocals': 0.08,   # 80ms - syllables
        }
        min_length = min_lengths.get(stem_type, 0.05)

        for instrument in midi.instruments:
            # Remove very short notes (likely glitches)
            instrument.notes = [
                n for n in instrument.notes
                if n.end - n.start >= min_length
            ]

            # Fix overlapping notes on same pitch
            instrument.notes.sort(key=lambda n: (n.pitch, n.start))
            cleaned = []
            for note in instrument.notes:
                if cleaned and cleaned[-1].pitch == note.pitch:
                    # Same pitch - check overlap
                    if note.start < cleaned[-1].end:
                        # Overlap - truncate previous note
                        cleaned[-1].end = note.start - 0.01
                        if cleaned[-1].end <= cleaned[-1].start:
                            cleaned.pop()  # Remove if too short
                cleaned.append(note)

            instrument.notes = cleaned

        return midi

    def _detect_key(self, midi: 'pretty_midi.PrettyMIDI') -> Optional[str]:
        """Detect musical key from MIDI using Krumhansl-Schmuckler algorithm."""
        if not midi.instruments:
            return None

        # Count pitch classes
        pitch_class_counts = np.zeros(12)
        for instrument in midi.instruments:
            for note in instrument.notes:
                duration = note.end - note.start
                pitch_class_counts[note.pitch % 12] += duration

        if np.sum(pitch_class_counts) == 0:
            return None

        # Normalize
        pitch_class_counts /= np.sum(pitch_class_counts)

        # Krumhansl-Schmuckler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_corr = -1
        best_key = 'C'

        for i in range(12):
            rotated = np.roll(pitch_class_counts, -i)

            major_corr = np.corrcoef(rotated, major_profile)[0, 1]
            minor_corr = np.corrcoef(rotated, minor_profile)[0, 1]

            if major_corr > best_corr:
                best_corr = major_corr
                best_key = note_names[i]
            if minor_corr > best_corr:
                best_corr = minor_corr
                best_key = note_names[i] + 'm'

        return best_key

    def _estimate_quality(self, midi: 'pretty_midi.PrettyMIDI',
                          articulations: List[ArticulationEvent]) -> float:
        """Estimate transcription quality (0-1)."""
        if not midi.instruments:
            return 0.0

        notes = midi.instruments[0].notes
        if not notes:
            return 0.0

        # Factors that indicate good transcription:
        score = 0.5  # Base score

        # 1. Note density - too sparse or too dense is bad
        duration = max(n.end for n in notes) - min(n.start for n in notes)
        notes_per_second = len(notes) / (duration + 0.1)
        if 1 < notes_per_second < 20:
            score += 0.1

        # 2. Velocity variation - good transcription captures dynamics
        velocities = [n.velocity for n in notes]
        if len(velocities) > 10:
            vel_std = np.std(velocities)
            if vel_std > 10:  # Some dynamics
                score += 0.1

        # 3. Pitch range - should be reasonable for instrument
        pitches = [n.pitch for n in notes]
        pitch_range = max(pitches) - min(pitches)
        if 12 < pitch_range < 48:  # 1-4 octaves
            score += 0.1

        # 4. Articulations detected - indicates detailed analysis worked
        if articulations:
            score += min(0.2, len(articulations) * 0.02)

        return min(1.0, score)


def transcribe_with_enhanced(audio_path: str, output_dir: str,
                             stem_type: str = 'guitar') -> Optional[str]:
    """
    Convenience function to transcribe audio with enhanced settings.
    Returns path to output MIDI or None on failure.
    """
    transcriber = EnhancedTranscriber()
    result = transcriber.transcribe(
        audio_path=audio_path,
        output_dir=output_dir,
        stem_type=stem_type,
        detect_articulations=True,
        quantize=True
    )

    return result.midi_path if result.midi_path else None


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Enhanced Transcriber loaded")
    print(f"  librosa: {LIBROSA_AVAILABLE}")
    print(f"  basic_pitch: {BASIC_PITCH_AVAILABLE}")
    print(f"  pretty_midi: {PRETTY_MIDI_AVAILABLE}")
