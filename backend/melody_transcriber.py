"""
Monophonic Melody/Lead Transcriber for StemScribe
===================================================
Dedicated pipeline for extracting clean single-note lead lines from
isolated stems (guitar, vocals, bass). Produces much cleaner tabs and
notation than polyphonic Basic Pitch for learning solos and melodies.

5-stage pipeline:
  1. Audio preprocessing (harmonic isolation, bandpass, noise gate)
  2. F0 contour extraction (dual-resolution pyin with confidence merge)
  3. Note segmentation (onset detection + pitch-change onsets)
  4. Articulation classification (bends, slides, hammer-ons, vibrato)
  5. Optional ensemble validation with Basic Pitch

Falls back to existing EnhancedTranscriber/Basic Pitch if quality < 0.4.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ============================================================================
# Optional imports — graceful degradation
# ============================================================================

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available - melody transcriber disabled")

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False
    logger.warning("pretty_midi not available - melody transcriber disabled")

try:
    from scipy.signal import butter, sosfilt, welch
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy.signal not available - bandpass filter disabled")

try:
    from basic_pitch.inference import predict
    BASIC_PITCH_AVAILABLE = True
except ImportError:
    BASIC_PITCH_AVAILABLE = False


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class MelodyNote:
    """A single note in a monophonic melody line."""
    pitch_midi: int           # Quantized MIDI note number
    pitch_hz: float           # Original Hz from pyin (continuous)
    start_time: float         # Seconds
    end_time: float           # Seconds
    velocity: int             # MIDI 0-127
    confidence: float         # pyin confidence 0-1
    articulation: Optional[str] = None  # 'bend', 'slide_up', 'slide_down', 'hammer_on', 'pull_off', 'vibrato'
    pitch_bend_cents: float = 0.0       # For bends: max cents above base note
    vibrato_rate: Optional[float] = None   # Hz
    vibrato_depth: Optional[float] = None  # Cents


@dataclass
class MelodyTranscriptionResult:
    """Result from the melody extraction pipeline."""
    midi_path: Optional[str]
    notes: List[MelodyNote]
    tempo: float
    detected_key: Optional[str]
    quality_score: float          # 0-1
    method_used: str              # 'pyin_primary', 'ensemble', 'basic_pitch_fallback'
    articulation_count: int = 0   # Total articulations detected


# ============================================================================
# Instrument-specific frequency ranges
# ============================================================================

INSTRUMENT_RANGES = {
    'guitar': {'fmin': 80.0, 'fmax': 3000.0, 'bp_low': 75, 'bp_high': 3200},
    'bass':   {'fmin': 30.0, 'fmax': 500.0,  'bp_low': 28, 'bp_high': 550},
    'vocals': {'fmin': 80.0, 'fmax': 1100.0, 'bp_low': 75, 'bp_high': 1200},
    'piano':  {'fmin': 55.0, 'fmax': 4200.0, 'bp_low': 50, 'bp_high': 4400},
}

# Minimum note lengths by instrument (seconds)
MIN_NOTE_LENGTHS = {
    'guitar': 0.035,
    'bass': 0.050,
    'vocals': 0.060,
    'piano': 0.030,
}


# ============================================================================
# MelodyExtractor — the main class
# ============================================================================

class MelodyExtractor:
    """
    Extracts monophonic melody lines from isolated audio stems.

    The key insight: lead guitar, vocal melodies, and bass lines are almost
    always one note at a time. By enforcing a monophonic constraint and using
    pyin (probabilistic YIN) instead of Basic Pitch, we get much cleaner
    output with real pitch contour data for articulations.
    """

    def __init__(self, sample_rate: int = 22050, instrument: str = 'guitar'):
        if not LIBROSA_AVAILABLE or not PRETTY_MIDI_AVAILABLE:
            raise ImportError("librosa and pretty_midi are required for MelodyExtractor")

        self.sr = sample_rate
        self.instrument = instrument
        self.ranges = INSTRUMENT_RANGES.get(instrument, INSTRUMENT_RANGES['guitar'])
        self.min_note_length = MIN_NOTE_LENGTHS.get(instrument, 0.04)

    def transcribe(self, audio_path: str, output_dir: str,
                   instrument: str = None, tempo_hint: float = None,
                   ensemble: bool = True) -> MelodyTranscriptionResult:
        """
        Full melody extraction pipeline.

        Args:
            audio_path: Path to the isolated stem audio file
            output_dir: Directory for output MIDI
            instrument: Override instrument type
            tempo_hint: Known tempo from earlier analysis (optional)
            ensemble: Whether to cross-check with Basic Pitch

        Returns:
            MelodyTranscriptionResult with MIDI path, notes, quality score
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if instrument:
            self.instrument = instrument
            self.ranges = INSTRUMENT_RANGES.get(instrument, INSTRUMENT_RANGES['guitar'])
            self.min_note_length = MIN_NOTE_LENGTHS.get(instrument, 0.04)

        logger.info(f"  Melody extraction: {audio_path.name} (instrument={self.instrument})")

        # Load audio
        try:
            y, sr = librosa.load(str(audio_path), sr=self.sr)
        except Exception as e:
            logger.error(f"  Failed to load audio: {e}")
            return self._empty_result("load_failed")

        if len(y) < sr:  # Less than 1 second
            logger.warning(f"  Audio too short for melody extraction ({len(y)/sr:.1f}s)")
            return self._empty_result("too_short")

        # ---- Stage 1: Preprocessing ----
        y_clean = self._preprocess_audio(y)

        # ---- Stage 2: F0 contour extraction ----
        times, f0, confidence = self._extract_f0_contour(y_clean)

        if f0 is None or len(f0) == 0 or np.all(np.isnan(f0)):
            logger.warning("  pyin returned no pitch data")
            return self._empty_result("no_pitch")

        # ---- Tempo detection ----
        if tempo_hint and 40 < tempo_hint < 300:
            tempo = tempo_hint
        else:
            tempo = self._detect_tempo(y)

        # ---- Stage 3: Note segmentation ----
        onsets = self._detect_onsets(y_clean)
        notes = self._segment_notes(times, f0, confidence, onsets, y)

        if not notes:
            logger.warning("  No notes segmented from contour")
            return self._empty_result("no_notes")

        logger.info(f"  Segmented {len(notes)} notes from f0 contour")

        # ---- Stage 4: Articulation classification ----
        notes = self._classify_articulations(y_clean, notes, times, f0, confidence)
        art_count = sum(1 for n in notes if n.articulation)
        logger.info(f"  Detected {art_count} articulations")

        # ---- Stage 5: Optional ensemble with Basic Pitch ----
        method = 'pyin_primary'
        if ensemble and BASIC_PITCH_AVAILABLE:
            notes, bp_agreement = self._merge_with_basic_pitch(
                notes, str(audio_path)
            )
            method = 'ensemble'
        else:
            bp_agreement = None

        # ---- Quality scoring ----
        quality = self._estimate_quality(notes, f0, confidence, onsets, bp_agreement)

        # ---- Key detection ----
        detected_key = self._detect_key(notes)

        # ---- Write MIDI ----
        midi_obj = self._notes_to_midi(notes, tempo)
        midi_filename = f"{audio_path.stem}_melody.mid"
        midi_path = output_dir / midi_filename
        midi_obj.write(str(midi_path))
        logger.info(f"  Melody MIDI written: {midi_path.name} "
                    f"({len(notes)} notes, quality={quality:.2f})")

        return MelodyTranscriptionResult(
            midi_path=str(midi_path),
            notes=notes,
            tempo=tempo,
            detected_key=detected_key,
            quality_score=quality,
            method_used=method,
            articulation_count=art_count,
        )

    # ========================================================================
    # Stage 1: Audio Preprocessing
    # ========================================================================

    def _preprocess_audio(self, y: np.ndarray) -> np.ndarray:
        """
        Clean the stem audio: harmonic isolation, bandpass filter, noise gate.
        This dramatically improves pyin accuracy by removing bleed and noise.
        """
        # 1. Harmonic isolation — strip percussive bleed and pick attacks
        y_harm = librosa.effects.harmonic(y, margin=3.0)

        # 2. Bandpass filter — restrict to instrument frequency range
        if SCIPY_AVAILABLE:
            bp_low = self.ranges['bp_low']
            bp_high = self.ranges['bp_high']
            nyquist = self.sr / 2.0
            # Clamp to valid range
            bp_high = min(bp_high, nyquist - 1)
            if bp_low < bp_high:
                sos = butter(4, [bp_low / nyquist, bp_high / nyquist], btype='band', output='sos')
                y_harm = sosfilt(sos, y_harm).astype(np.float32)

        # 3. Noise gate — zero out frames below threshold
        frame_length = 2048
        rms = librosa.feature.rms(y=y_harm, frame_length=frame_length, hop_length=512)[0]
        noise_floor = 0.05 * np.max(rms) if np.max(rms) > 0 else 0
        for i, r in enumerate(rms):
            if r < noise_floor:
                start = i * 512
                end = min(start + frame_length, len(y_harm))
                y_harm[start:end] *= 0.01  # Soft gate, not hard zero

        # 4. Normalize
        peak = np.max(np.abs(y_harm))
        if peak > 0:
            y_harm = y_harm / peak

        return y_harm

    # ========================================================================
    # Stage 2: F0 Contour Extraction
    # ========================================================================

    def _extract_f0_contour(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dual-resolution pyin with confidence-based merge and harmonic disambiguation.
        """
        fmin = self.ranges['fmin']
        fmax = self.ranges['fmax']

        # Fine resolution: hop=128 (~5.8ms) — catches fast passages
        f0_fine, voiced_fine, conf_fine = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=self.sr,
            hop_length=128, fill_na=0.0
        )

        # Standard resolution: hop=256 (~11ms) — more stable
        f0_std, voiced_std, conf_std = librosa.pyin(
            y, fmin=fmin, fmax=fmax, sr=self.sr,
            hop_length=256, fill_na=0.0
        )

        # Upsample standard to match fine resolution (2x)
        f0_std_up = np.repeat(f0_std, 2)[:len(f0_fine)]
        conf_std_up = np.repeat(conf_std, 2)[:len(f0_fine)]

        # Merge: prefer fine when confident, standard otherwise
        f0_merged = np.where(
            (conf_fine > 0.5) | (conf_std_up < 0.6),
            f0_fine,
            f0_std_up
        )
        conf_merged = np.maximum(conf_fine, conf_std_up)

        # Confidence-weighted median filter on low-confidence regions
        f0_filtered = f0_merged.copy()
        for i in range(1, len(f0_filtered) - 1):
            if conf_merged[i] < 0.7 and f0_filtered[i] > 0:
                neighbors = f0_filtered[max(0, i-1):i+2]
                valid = neighbors[neighbors > 0]
                if len(valid) > 0:
                    f0_filtered[i] = np.median(valid)

        # Harmonic disambiguation — snap octave jumps back to fundamental
        for i in range(2, len(f0_filtered)):
            if f0_filtered[i] > 0 and f0_filtered[i-1] > 0:
                ratio = f0_filtered[i] / f0_filtered[i-1]
                # If jumped to ~2x or ~3x and confidence is weak, snap down
                if conf_merged[i] < 0.8:
                    if 1.9 < ratio < 2.1:
                        f0_filtered[i] /= 2.0
                    elif 2.9 < ratio < 3.1:
                        f0_filtered[i] /= 3.0
                    elif 0.45 < ratio < 0.55:
                        f0_filtered[i] *= 2.0
                    elif 0.3 < ratio < 0.37:
                        f0_filtered[i] *= 3.0

        times = librosa.times_like(f0_fine, sr=self.sr, hop_length=128)
        return times, f0_filtered, conf_merged

    # ========================================================================
    # Stage 3: Onset Detection + Note Segmentation
    # ========================================================================

    def _detect_onsets(self, y: np.ndarray) -> np.ndarray:
        """Detect note onsets using librosa + onset strength."""
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=self.sr, hop_length=256, backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=256)
        return onset_times

    def _segment_notes(self, times: np.ndarray, f0: np.ndarray,
                       confidence: np.ndarray, onsets: np.ndarray,
                       y_raw: np.ndarray) -> List[MelodyNote]:
        """
        Convert continuous f0 contour into discrete notes using onsets.
        Enforces monophonic constraint — one note at a time.
        """
        if len(onsets) == 0:
            return []

        # Add pitch-change onsets: sudden jumps > 1.5 semitones without a nearby onset
        pitch_change_onsets = []
        midi_contour = np.zeros_like(f0)
        voiced_mask = f0 > 0
        midi_contour[voiced_mask] = librosa.hz_to_midi(f0[voiced_mask])

        for i in range(1, len(midi_contour)):
            if midi_contour[i] > 0 and midi_contour[i-1] > 0:
                jump = abs(midi_contour[i] - midi_contour[i-1])
                if jump >= 1.5:
                    t = times[i]
                    # Only add if no existing onset within 30ms
                    if not any(abs(t - o) < 0.03 for o in onsets):
                        pitch_change_onsets.append(t)

        all_onsets = np.sort(np.concatenate([onsets, pitch_change_onsets]))

        # Compute RMS for velocity estimation
        rms = librosa.feature.rms(y=y_raw, frame_length=2048, hop_length=512)[0]
        rms_times = librosa.times_like(rms, sr=self.sr, hop_length=512)
        rms_max = np.max(rms) if np.max(rms) > 0 else 1.0

        notes = []
        for i, onset in enumerate(all_onsets):
            # Segment: from this onset to the next (or end)
            next_onset = all_onsets[i + 1] if i + 1 < len(all_onsets) else times[-1]
            segment_duration = next_onset - onset

            if segment_duration < self.min_note_length:
                continue

            # Find f0 frames in this segment
            seg_mask = (times >= onset) & (times < next_onset)
            seg_f0 = f0[seg_mask]
            seg_conf = confidence[seg_mask]

            if len(seg_f0) == 0:
                continue

            # Find voiced frames
            voiced = (seg_f0 > 0) & (seg_conf > 0.3)
            if np.sum(voiced) < 2:
                continue

            voiced_f0 = seg_f0[voiced]
            voiced_conf = seg_conf[voiced]

            # Compute median pitch of stable portion
            # "Stable" = within 100 cents of running median
            running_median = np.median(voiced_f0)
            cents_from_median = 1200 * np.log2(voiced_f0 / running_median + 1e-10)
            stable = np.abs(cents_from_median) < 100
            if np.sum(stable) > 0:
                stable_f0 = voiced_f0[stable]
                median_hz = np.median(stable_f0)
            else:
                median_hz = running_median

            if median_hz <= 0:
                continue

            midi_pitch = int(round(librosa.hz_to_midi(median_hz)))
            mean_conf = float(np.mean(voiced_conf))

            # Find note end: where confidence drops or next onset
            # Use the voiced region, not the full segment
            end_time = min(next_onset - 0.01, onset + segment_duration)

            # Velocity from RMS at onset
            rms_idx = np.searchsorted(rms_times, onset)
            if rms_idx < len(rms):
                velocity = int(40 + 87 * min(1.0, rms[rms_idx] / rms_max))
            else:
                velocity = 80

            velocity = max(30, min(127, velocity))

            notes.append(MelodyNote(
                pitch_midi=midi_pitch,
                pitch_hz=float(median_hz),
                start_time=float(onset),
                end_time=float(end_time),
                velocity=velocity,
                confidence=mean_conf,
            ))

        # Remove duplicate notes at same time/pitch
        cleaned = []
        for note in notes:
            if cleaned and abs(note.start_time - cleaned[-1].start_time) < 0.02:
                # Keep the one with higher confidence
                if note.confidence > cleaned[-1].confidence:
                    cleaned[-1] = note
            else:
                cleaned.append(note)

        return cleaned

    # ========================================================================
    # Stage 4: Articulation Classification
    # ========================================================================

    def _classify_articulations(self, y: np.ndarray, notes: List[MelodyNote],
                                times: np.ndarray, f0: np.ndarray,
                                confidence: np.ndarray) -> List[MelodyNote]:
        """Analyze f0 contour within each note to detect articulations."""
        if len(notes) < 2:
            return notes

        # Get onset strengths for hammer-on/pull-off detection
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        onset_times_env = librosa.times_like(onset_env, sr=self.sr)
        median_onset = np.median(onset_env) if len(onset_env) > 0 else 1.0

        for i, note in enumerate(notes):
            # Get f0 contour within this note
            seg_mask = (times >= note.start_time) & (times < note.end_time)
            seg_f0 = f0[seg_mask]
            seg_conf = confidence[seg_mask]
            seg_times = times[seg_mask]

            if len(seg_f0) < 3:
                continue

            voiced = seg_f0 > 0
            if np.sum(voiced) < 3:
                continue

            # Convert to cents deviation from quantized note
            note_hz = librosa.midi_to_hz(note.pitch_midi)
            cents_dev = np.zeros_like(seg_f0)
            cents_dev[voiced] = 1200 * np.log2(seg_f0[voiced] / note_hz + 1e-10)

            # ---- Bend detection ----
            max_dev = np.max(np.abs(cents_dev[voiced]))
            if max_dev > 30:  # > 30 cents = perceptible bend
                # Check if it's a true bend (rises and/or falls back)
                mean_dev = np.mean(cents_dev[voiced])
                if mean_dev > 20:
                    note.articulation = 'bend'
                    note.pitch_bend_cents = float(np.max(cents_dev[voiced]))
                    continue

            # ---- Slide detection ----
            if len(seg_f0[voiced]) > 5:
                start_region = seg_f0[voiced][:3]
                end_region = seg_f0[voiced][-3:]
                start_midi = np.median(librosa.hz_to_midi(start_region))
                end_midi = np.median(librosa.hz_to_midi(end_region))
                slide_semitones = end_midi - start_midi

                if abs(slide_semitones) > 1.0:
                    note.articulation = 'slide_up' if slide_semitones > 0 else 'slide_down'
                    note.pitch_bend_cents = float(abs(slide_semitones) * 100)
                    continue

            # ---- Vibrato detection ----
            vibrato = self._detect_vibrato(cents_dev[voiced], seg_times[voiced])
            if vibrato:
                rate, depth = vibrato
                note.articulation = 'vibrato'
                note.vibrato_rate = rate
                note.vibrato_depth = depth
                continue

            # ---- Hammer-on / Pull-off detection ----
            if i > 0:
                prev_note = notes[i - 1]
                gap = note.start_time - prev_note.end_time

                if -0.05 < gap < 0.06:  # Close or overlapping
                    # Check onset strength
                    idx = np.searchsorted(onset_times_env, note.start_time)
                    if idx < len(onset_env):
                        if onset_env[idx] < median_onset * 0.5:
                            pitch_diff = note.pitch_midi - prev_note.pitch_midi
                            if pitch_diff > 0:
                                note.articulation = 'hammer_on'
                            elif pitch_diff < 0:
                                note.articulation = 'pull_off'

        return notes

    def _detect_vibrato(self, cents_deviation: np.ndarray,
                        times: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Detect vibrato by looking for 4-8 Hz oscillation in pitch deviation.
        Uses scipy Welch PSD estimate.
        """
        if not SCIPY_AVAILABLE or len(cents_deviation) < 20:
            return None

        # Ensure even spacing — resample to uniform time grid
        dt = np.mean(np.diff(times)) if len(times) > 1 else 0.006
        if dt <= 0:
            return None
        fs = 1.0 / dt

        # Detrend
        detrended = cents_deviation - np.mean(cents_deviation)

        # Welch PSD
        try:
            freqs, psd = welch(detrended, fs=fs, nperseg=min(len(detrended), 64))
        except Exception:
            return None

        # Look for peak in vibrato range (4-8 Hz)
        vibrato_mask = (freqs >= 4.0) & (freqs <= 8.0)
        if not np.any(vibrato_mask):
            return None

        vibrato_psd = psd[vibrato_mask]
        vibrato_freqs = freqs[vibrato_mask]

        peak_idx = np.argmax(vibrato_psd)
        peak_power = vibrato_psd[peak_idx]
        peak_freq = vibrato_freqs[peak_idx]

        # Noise floor: mean power outside vibrato range
        noise_mask = ~vibrato_mask & (freqs > 1.0)
        if np.any(noise_mask):
            noise_floor = np.mean(psd[noise_mask])
        else:
            noise_floor = np.mean(psd)

        # Vibrato if peak is 2x above noise floor and depth is perceptible
        if noise_floor > 0 and peak_power > 2.0 * noise_floor:
            depth = float(np.std(detrended))  # Approximate depth in cents
            if depth > 8:  # At least 8 cents — perceptible vibrato
                return (float(peak_freq), depth)

        return None

    # ========================================================================
    # Stage 5: Ensemble with Basic Pitch
    # ========================================================================

    def _merge_with_basic_pitch(self, melody_notes: List[MelodyNote],
                                audio_path: str) -> Tuple[List[MelodyNote], float]:
        """
        Cross-check melody notes against Basic Pitch output.
        Add any high-confidence BP notes that pyin missed.
        Returns (merged_notes, agreement_fraction).
        """
        try:
            _, bp_midi, bp_note_events = predict(
                audio_path,
                onset_threshold=0.7,    # Strict: only very confident notes
                frame_threshold=0.5,
                minimum_note_length=80,
                minimum_frequency=self.ranges['fmin'],
                maximum_frequency=self.ranges['fmax'],
            )
        except Exception as e:
            logger.debug(f"  Basic Pitch ensemble skipped: {e}")
            return melody_notes, None

        if bp_midi is None or not bp_midi.instruments:
            return melody_notes, None

        bp_notes = bp_midi.instruments[0].notes
        if not bp_notes:
            return melody_notes, None

        # Check agreement
        matched = 0
        unmatched_bp = []

        for bp_note in bp_notes:
            bp_time = bp_note.start
            bp_pitch = bp_note.pitch
            found = False

            for mel_note in melody_notes:
                time_close = abs(bp_time - mel_note.start_time) < 0.05
                pitch_close = abs(bp_pitch - mel_note.pitch_midi) <= 2
                if time_close and pitch_close:
                    matched += 1
                    found = True
                    break

            if not found and bp_note.velocity > 80:
                unmatched_bp.append(bp_note)

        # Agreement ratio
        total = max(len(melody_notes), 1)
        agreement = matched / total

        # Add high-confidence BP notes that pyin missed
        added = 0
        for bp_note in unmatched_bp:
            # Only add if it doesn't overlap with existing notes
            overlap = False
            for mel_note in melody_notes:
                if not (bp_note.end < mel_note.start_time or bp_note.start > mel_note.end_time):
                    overlap = True
                    break

            if not overlap:
                melody_notes.append(MelodyNote(
                    pitch_midi=bp_note.pitch,
                    pitch_hz=float(librosa.midi_to_hz(bp_note.pitch)),
                    start_time=bp_note.start,
                    end_time=bp_note.end,
                    velocity=bp_note.velocity,
                    confidence=0.6,  # Lower confidence for BP-only notes
                ))
                added += 1

        if added > 0:
            melody_notes.sort(key=lambda n: n.start_time)
            logger.info(f"  Ensemble: {matched} agreed, {added} BP notes added, "
                       f"agreement={agreement:.0%}")

        return melody_notes, agreement

    # ========================================================================
    # MIDI Output with Pitch Bend Encoding
    # ========================================================================

    def _notes_to_midi(self, notes: List[MelodyNote], tempo: float) -> 'pretty_midi.PrettyMIDI':
        """
        Convert melody notes to MIDI with pitch bend envelopes for articulations.
        Encodes articulation type in CC#20 for downstream GP/notation converters.
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)

        # Choose MIDI program by instrument
        program_map = {
            'guitar': 25,   # Acoustic guitar (steel)
            'bass': 33,     # Electric bass (finger)
            'vocals': 52,   # Choir Aahs
            'piano': 0,     # Acoustic Grand
        }
        program = program_map.get(self.instrument, 25)
        instrument = pretty_midi.Instrument(program=program)

        # Articulation type → CC#20 value mapping
        ART_CC = {
            'bend': 1,
            'slide_up': 2,
            'slide_down': 3,
            'hammer_on': 4,
            'pull_off': 5,
            'vibrato': 6,
        }

        # Pitch bend range: ±2 semitones = ±200 cents = ±8192
        CENTS_TO_BEND = 8192.0 / 200.0  # ~41 per cent

        for note in notes:
            midi_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch_midi,
                start=note.start_time,
                end=note.end_time,
            )
            instrument.notes.append(midi_note)

            # Encode articulation as CC#20
            if note.articulation and note.articulation in ART_CC:
                cc = pretty_midi.ControlChange(
                    number=20,
                    value=ART_CC[note.articulation],
                    time=note.start_time,
                )
                instrument.control_changes.append(cc)

            # Add pitch bend envelopes for expressive articulations
            if note.articulation == 'bend' and note.pitch_bend_cents > 0:
                self._add_bend_envelope(instrument, note, CENTS_TO_BEND)
            elif note.articulation in ('slide_up', 'slide_down'):
                self._add_slide_envelope(instrument, note, CENTS_TO_BEND)
            elif note.articulation == 'vibrato' and note.vibrato_rate:
                self._add_vibrato_envelope(instrument, note, CENTS_TO_BEND)

            # Reset pitch bend after every note
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=0, time=note.end_time)
            )

        midi.instruments.append(instrument)
        return midi

    def _add_bend_envelope(self, instrument, note: MelodyNote, cents_to_bend: float):
        """Add a bend-up-and-back envelope."""
        duration = note.end_time - note.start_time
        num_points = 20
        max_bend = int(note.pitch_bend_cents * cents_to_bend)
        max_bend = max(-8192, min(8191, max_bend))

        for i in range(num_points):
            frac = i / num_points
            t = note.start_time + duration * frac
            # Bell curve: rise to peak at ~40% then return
            if frac < 0.4:
                envelope = frac / 0.4
            elif frac < 0.7:
                envelope = 1.0
            else:
                envelope = 1.0 - (frac - 0.7) / 0.3

            pb = pretty_midi.PitchBend(
                pitch=int(max_bend * envelope),
                time=t,
            )
            instrument.pitch_bends.append(pb)

    def _add_slide_envelope(self, instrument, note: MelodyNote, cents_to_bend: float):
        """Add a slide-in envelope (starts offset, ends at target pitch)."""
        duration = note.end_time - note.start_time
        slide_cents = note.pitch_bend_cents
        if note.articulation == 'slide_down':
            slide_cents = -slide_cents

        num_points = 15
        # Start at offset, slide to center over first 30% of note
        for i in range(num_points):
            frac = i / num_points
            t = note.start_time + duration * frac * 0.3
            # Linear slide from start offset to zero
            envelope = 1.0 - (frac / 1.0)
            bend = int(-slide_cents * cents_to_bend * envelope)
            bend = max(-8192, min(8191, bend))
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=bend, time=t)
            )

    def _add_vibrato_envelope(self, instrument, note: MelodyNote, cents_to_bend: float):
        """Add sinusoidal vibrato pitch oscillation."""
        duration = note.end_time - note.start_time
        rate = note.vibrato_rate or 5.5
        depth = note.vibrato_depth or 20  # cents

        # ~50 points per vibrato cycle, but cap total
        total_cycles = duration * rate
        num_points = min(int(total_cycles * 20), 200)
        if num_points < 4:
            return

        for i in range(num_points):
            t = note.start_time + duration * i / num_points
            phase = 2 * np.pi * rate * (t - note.start_time)
            bend_cents = depth * np.sin(phase)
            bend = int(bend_cents * cents_to_bend)
            bend = max(-8192, min(8191, bend))
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=bend, time=t)
            )

    # ========================================================================
    # Quality Scoring
    # ========================================================================

    def _estimate_quality(self, notes: List[MelodyNote], f0: np.ndarray,
                          confidence: np.ndarray, onsets: np.ndarray,
                          bp_agreement: Optional[float]) -> float:
        """
        Estimate transcription quality on a 0-1 scale.
        Low quality triggers fallback to EnhancedTranscriber/Basic Pitch.
        """
        if not notes:
            return 0.0

        score = 0.0

        # 1. F0 confidence (weight: 0.30)
        voiced = f0 > 0
        if np.sum(voiced) > 0:
            mean_conf = float(np.mean(confidence[voiced]))
            score += 0.30 * min(1.0, mean_conf / 0.7)

        # 2. Note density sanity (weight: 0.15)
        total_duration = notes[-1].end_time - notes[0].start_time
        if total_duration > 0:
            density = len(notes) / total_duration
            # Ideal: 1-12 notes/sec for most music
            if 0.5 < density < 15:
                score += 0.15
            elif 0.2 < density < 20:
                score += 0.08
            # else: 0 — suspiciously sparse or dense

        # 3. Pitch range sanity (weight: 0.15)
        pitches = [n.pitch_midi for n in notes]
        pitch_range = max(pitches) - min(pitches)
        if 5 < pitch_range < 36:
            score += 0.15
        elif 3 < pitch_range < 48:
            score += 0.08

        # 4. Onset/F0 agreement (weight: 0.20)
        if len(onsets) > 0 and len(notes) > 0:
            matched_onsets = 0
            for onset in onsets:
                if any(abs(onset - n.start_time) < 0.05 for n in notes):
                    matched_onsets += 1
            onset_agreement = matched_onsets / len(onsets)
            score += 0.20 * onset_agreement

        # 5. Articulation detection (weight: 0.10)
        art_count = sum(1 for n in notes if n.articulation)
        if art_count > 0:
            art_ratio = art_count / len(notes)
            score += 0.10 * min(1.0, art_ratio * 3)

        # 6. Basic Pitch agreement (weight: 0.10)
        if bp_agreement is not None:
            score += 0.10 * bp_agreement
        else:
            score += 0.05  # Neutral if BP wasn't used

        return min(1.0, score)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _detect_tempo(self, y: np.ndarray) -> float:
        """Estimate tempo using librosa beat tracker."""
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=self.sr)
            if hasattr(tempo, '__len__'):
                tempo = float(tempo[0])
            return max(40.0, min(300.0, tempo))
        except Exception:
            return 120.0

    def _detect_key(self, notes: List[MelodyNote]) -> Optional[str]:
        """Detect musical key from note distribution (Krumhansl-Schmuckler)."""
        if len(notes) < 4:
            return None

        # Count pitch classes weighted by duration
        pitch_class_counts = np.zeros(12)
        for n in notes:
            dur = n.end_time - n.start_time
            pitch_class_counts[n.pitch_midi % 12] += dur

        if np.sum(pitch_class_counts) == 0:
            return None

        # Krumhansl-Schmuckler profiles
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        best_key = None
        best_corr = -1

        for shift in range(12):
            rotated = np.roll(pitch_class_counts, -shift)

            # Major key correlation
            corr_major = float(np.corrcoef(rotated, major_profile)[0, 1])
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = note_names[shift]

            # Minor key correlation
            corr_minor = float(np.corrcoef(rotated, minor_profile)[0, 1])
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = f"{note_names[shift]}m"

        return best_key

    def _empty_result(self, reason: str) -> MelodyTranscriptionResult:
        """Return an empty result with quality 0."""
        logger.debug(f"  Empty melody result: {reason}")
        return MelodyTranscriptionResult(
            midi_path=None,
            notes=[],
            tempo=120.0,
            detected_key=None,
            quality_score=0.0,
            method_used=f'failed_{reason}',
        )


# ============================================================================
# Convenience function (matches pattern of other StemScribe modules)
# ============================================================================

def transcribe_melody(audio_path: str, output_dir: str,
                      instrument: str = 'guitar',
                      tempo_hint: float = None,
                      ensemble: bool = True) -> Optional[str]:
    """
    Convenience function: transcribe melody and return MIDI path (or None).
    This is the function app.py imports.
    """
    try:
        extractor = MelodyExtractor(instrument=instrument)
        result = extractor.transcribe(
            audio_path=audio_path,
            output_dir=output_dir,
            instrument=instrument,
            tempo_hint=tempo_hint,
            ensemble=ensemble,
        )
        if result.midi_path and result.quality_score > 0.4:
            return result.midi_path
        return None
    except Exception as e:
        logger.error(f"Melody transcription failed: {e}")
        return None
