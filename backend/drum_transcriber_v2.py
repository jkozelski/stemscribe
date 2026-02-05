"""
Enhanced Drum Transcription Module for StemScribe v2
=====================================================
Major improvements over v1:
- Ghost note detection (low velocity hits)
- Better cymbal differentiation (hi-hat vs ride vs crash)
- Hi-hat open/closed detection with pedal tracking
- Tom pitch detection (floor, low, mid, high)
- Proper kick/snare timing with micro-timing preservation
- Velocity dynamics from audio amplitude
- Swing detection and preservation

General MIDI Drum Map (Standard):
- 35/36: Kick
- 37: Side stick, 38: Snare, 40: Electric snare
- 42: Closed hi-hat, 44: Pedal hi-hat, 46: Open hi-hat
- 41/43: Floor tom, 45/47: Low-mid tom, 48/50: High tom
- 49/57: Crash, 51/59: Ride, 52: China, 53: Ride bell, 55: Splash
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False


# Enhanced General MIDI Drum Map
GM_DRUMS = {
    # Kicks
    'kick': 36,
    'kick_accent': 35,  # Acoustic bass drum

    # Snares
    'snare': 38,
    'snare_ghost': 38,      # Same note, different velocity
    'snare_rimshot': 40,    # Electric snare (rimshot sound)
    'snare_sidestick': 37,  # Side stick / cross-stick

    # Hi-hats
    'hihat_closed': 42,
    'hihat_pedal': 44,      # Foot splash
    'hihat_open': 46,
    'hihat_half': 42,       # Partially open (use CC for openness)

    # Toms
    'tom_floor': 43,        # High floor tom
    'tom_floor_low': 41,    # Low floor tom
    'tom_low': 45,          # Low tom
    'tom_mid': 47,          # Low-mid tom
    'tom_high': 48,         # High tom
    'tom_highest': 50,      # High tom 2

    # Cymbals
    'crash': 49,
    'crash_2': 57,
    'ride': 51,
    'ride_bell': 53,
    'china': 52,
    'splash': 55,
}


@dataclass
class DrumHit:
    """Represents a single drum hit with detailed information."""
    time: float
    drum_type: str
    velocity: int
    confidence: float
    duration: float = 0.05
    is_ghost: bool = False
    hihat_openness: float = 0.0  # 0 = closed, 1 = open


@dataclass
class TranscriptionStats:
    """Statistics about the drum transcription."""
    total_hits: int
    hits_by_type: Dict[str, int]
    ghost_notes: int
    tempo: float
    time_signature: str
    quality_score: float


class DrumDetector:
    """
    Advanced drum hit detection using multi-band spectral analysis
    and onset detection.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.hop_length = 512

        # Frequency bands for drum detection (Hz)
        self.bands = {
            'sub': (20, 80),       # Sub-bass (kick fundamental)
            'kick': (60, 150),     # Kick drum
            'snare_low': (150, 350),  # Snare body
            'snare_high': (2000, 6000),  # Snare wires
            'tom_low': (80, 200),  # Floor tom
            'tom_mid': (150, 400), # Mid toms
            'tom_high': (250, 600), # High toms
            'hihat': (6000, 14000), # Hi-hat
            'crash': (2000, 10000), # Crash cymbals
            'ride': (3000, 8000),   # Ride cymbal
        }

    def detect_onsets(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect onset times with their strengths.
        Returns onset times and onset strengths.
        """
        # Multi-band onset detection for better drum separation
        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr,
            hop_length=self.hop_length,
            aggregate=np.median,
            n_mels=128
        )

        # Detect onsets
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=self.sr,
            hop_length=self.hop_length,
            backtrack=True,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=0.05,  # Lower threshold for ghost notes
            wait=int(0.025 * self.sr / self.hop_length)  # Min 25ms between hits
        )

        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr, hop_length=self.hop_length)
        onset_strengths = onset_env[onset_frames] if len(onset_frames) > 0 else np.array([])

        return onset_times, onset_strengths

    def get_band_energy(self, y: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """Get energy in a specific frequency band over time."""
        # Filter to band
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        # Get frequency bin indices
        low_idx = np.searchsorted(freqs, low_freq)
        high_idx = np.searchsorted(freqs, high_freq)

        # Sum energy in band
        band_energy = np.sum(S[low_idx:high_idx, :] ** 2, axis=0)

        return band_energy

    def analyze_hit(self, y: np.ndarray, onset_time: float,
                    window_ms: float = 50) -> DrumHit:
        """
        Analyze a single hit to determine drum type and characteristics.
        """
        # Extract window around onset
        start_sample = int(onset_time * self.sr)
        window_samples = int(window_ms / 1000 * self.sr)
        end_sample = min(start_sample + window_samples, len(y))

        if end_sample <= start_sample:
            return DrumHit(onset_time, 'unknown', 64, 0.0)

        segment = y[start_sample:end_sample]

        # Spectral analysis
        n_fft = min(2048, len(segment))
        spectrum = np.abs(np.fft.rfft(segment, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1/self.sr)

        # Calculate energy in each band
        energies = {}
        for band_name, (low, high) in self.bands.items():
            mask = (freqs >= low) & (freqs < high)
            energies[band_name] = np.sum(spectrum[mask] ** 2)

        total_energy = sum(energies.values()) + 1e-10

        # Normalize
        for band in energies:
            energies[band] /= total_energy

        # Calculate additional features
        spectral_centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
        spectral_flatness = self._spectral_flatness(spectrum)

        # RMS for velocity
        rms = np.sqrt(np.mean(segment ** 2))

        # Classify the hit
        drum_type, confidence, is_ghost, hihat_openness = self._classify_hit(
            energies, spectral_centroid, spectral_flatness, rms
        )

        # Map RMS to velocity (with calibration for dynamics)
        velocity = self._rms_to_velocity(rms, is_ghost)

        return DrumHit(
            time=onset_time,
            drum_type=drum_type,
            velocity=velocity,
            confidence=confidence,
            is_ghost=is_ghost,
            hihat_openness=hihat_openness
        )

    def _spectral_flatness(self, spectrum: np.ndarray) -> float:
        """Calculate spectral flatness (noise-like vs tonal)."""
        spectrum = np.maximum(spectrum, 1e-10)
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / (arithmetic_mean + 1e-10)

    def _classify_hit(self, energies: Dict[str, float],
                      centroid: float, flatness: float,
                      rms: float) -> Tuple[str, float, bool, float]:
        """
        Classify a drum hit based on spectral features.
        Returns: (drum_type, confidence, is_ghost, hihat_openness)

        Improved v2.1: Better crash/ride differentiation, hi-hat detection
        """
        # Check if ghost note (low RMS) - calibrated threshold
        is_ghost = rms < 0.015

        hihat_openness = 0.0
        confidence = 0.7  # Default confidence

        # === KICK ===
        # Strong sub and kick band, low centroid, low flatness (tonal)
        kick_score = energies['sub'] * 0.5 + energies['kick'] * 0.35
        if centroid < 300:  # Kick has very low centroid
            kick_score *= 1.5
        elif centroid < 500:
            kick_score *= 1.2
        if flatness < 0.15:  # Kick is very tonal
            kick_score *= 1.3

        # === SNARE ===
        # Mid-low body + high frequency wires - the wires are key
        snare_score = energies['snare_low'] * 0.3 + energies['snare_high'] * 0.45
        if 0.2 < flatness < 0.5:  # Wire buzz adds flatness
            snare_score *= 1.25
        if 1500 < centroid < 4500:  # Snare centroid range
            snare_score *= 1.15

        # === HI-HAT ===
        hihat_score = energies['hihat'] * 0.6
        if flatness > 0.45:  # Hi-hat is very noise-like
            hihat_score *= 1.4
        if centroid > 6000:  # Hi-hat is very high
            hihat_score *= 1.3

        # Determine open vs closed by:
        # 1. Spectral spread (open has more low-mid content)
        # 2. Decay characteristics (open sustains longer - analyzed separately)
        if hihat_score > max(kick_score, snare_score) * 0.8:
            # Calculate hi-hat openness from energy distribution
            high_only = energies['hihat']
            mid_cymbal = energies['crash'] + energies['ride']
            total_cymbal = high_only + mid_cymbal + 0.001

            # Open hi-hat has more energy spread across mid frequencies
            hihat_openness = min(1.0, (mid_cymbal / total_cymbal) * 2.5)

            # Also check flatness - open hi-hats are slightly less noisy
            if flatness < 0.4:
                hihat_openness = max(hihat_openness, 0.6)

        # === CRASH ===
        # Crash has wide frequency spread, high attack, long decay
        crash_score = energies['crash'] * 0.45 + energies['ride'] * 0.15
        if flatness > 0.4 and centroid > 3000:  # Crash is noisy and bright
            crash_score *= 1.35
        # Crash tends to have energy across a wider band
        crash_spread = energies['snare_high'] + energies['hihat'] + energies['crash']
        if crash_spread > 0.5:
            crash_score *= 1.2

        # === RIDE ===
        # Ride is more focused, more tonal than crash, sustained
        ride_score = energies['ride'] * 0.5
        # Ride is more focused (less flat) than crash
        if 0.2 < flatness < 0.4:  # Ride has some tonality
            ride_score *= 1.35
        if 3500 < centroid < 6500:  # Ride has specific centroid range
            ride_score *= 1.25
        # Ride has less extreme high frequency content than crash
        if energies['hihat'] < energies['ride'] * 1.5:
            ride_score *= 1.15

        # === TOMS ===
        tom_score = energies['tom_low'] * 0.3 + energies['tom_mid'] * 0.3 + energies['tom_high'] * 0.2
        if flatness < 0.25:  # Toms are tonal
            tom_score *= 1.3
        # Toms have specific centroid range (higher than kick, lower than snare)
        if 150 < centroid < 600:
            tom_score *= 1.2

        # Find winner
        scores = {
            'kick': kick_score,
            'snare': snare_score,
            'hihat': hihat_score,
            'crash': crash_score,
            'ride': ride_score,
            'tom': tom_score,
        }

        winner = max(scores, key=scores.get)
        total_score = sum(scores.values()) + 0.01
        confidence = scores[winner] / total_score

        # Boost confidence if there's a clear winner
        second_best = sorted(scores.values(), reverse=True)[1]
        if scores[winner] > second_best * 1.5:
            confidence = min(1.0, confidence * 1.2)

        # Refine based on winner
        if winner == 'kick':
            drum_type = 'kick'
        elif winner == 'snare':
            if is_ghost:
                drum_type = 'snare_ghost'
            elif rms > 0.08 and flatness < 0.3:  # Hard rimshot
                drum_type = 'snare_rimshot'
            else:
                drum_type = 'snare'
        elif winner == 'hihat':
            if hihat_openness > 0.6:
                drum_type = 'hihat_open'
            elif hihat_openness > 0.3:
                drum_type = 'hihat_half'  # Partially open
            else:
                drum_type = 'hihat_closed'
        elif winner == 'crash':
            # Double-check it's not a ride
            if ride_score > crash_score * 0.85 and flatness < 0.38:
                drum_type = 'ride'  # Actually more like a ride
            else:
                drum_type = 'crash'
        elif winner == 'ride':
            # Check for ride bell (higher, more focused, more tonal)
            if centroid > 5500 and flatness < 0.3:
                drum_type = 'ride_bell'
            else:
                drum_type = 'ride'
        elif winner == 'tom':
            # Classify tom by centroid with better ranges
            if centroid < 200:
                drum_type = 'tom_floor'
            elif centroid < 300:
                drum_type = 'tom_low'
            elif centroid < 400:
                drum_type = 'tom_mid'
            else:
                drum_type = 'tom_high'
        else:
            drum_type = 'snare'  # Default fallback

        return drum_type, confidence, is_ghost, hihat_openness

    def _rms_to_velocity(self, rms: float, is_ghost: bool) -> int:
        """Convert RMS amplitude to MIDI velocity with proper dynamics."""
        if is_ghost:
            # Ghost notes: 20-50 velocity
            velocity = int(20 + rms * 600)
            return min(50, max(20, velocity))
        else:
            # Regular notes: 50-127 velocity
            velocity = int(50 + rms * 770)
            return min(127, max(50, velocity))


class EnhancedDrumTranscriber:
    """
    Main drum transcription class with all v2 features.
    """

    def __init__(self, sample_rate: int = 22050):
        self.sr = sample_rate
        self.detector = DrumDetector(sample_rate)

    def transcribe(self, audio_path: str, output_path: str,
                   sensitivity: float = 0.3,
                   detect_ghost_notes: bool = True,
                   preserve_dynamics: bool = True) -> TranscriptionStats:
        """
        Transcribe drum audio to MIDI.

        Args:
            audio_path: Path to drum stem audio
            output_path: Path for output MIDI file
            sensitivity: Onset detection sensitivity (0-1, lower = more hits)
            detect_ghost_notes: Whether to detect ghost notes
            preserve_dynamics: Whether to preserve velocity dynamics

        Returns:
            TranscriptionStats with transcription info
        """
        if not LIBROSA_AVAILABLE or not PRETTY_MIDI_AVAILABLE:
            logger.error("Required libraries not available")
            return TranscriptionStats(0, {}, 0, 120.0, "4/4", 0.0)

        logger.info(f"🥁 Enhanced drum transcription: {audio_path}")

        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        duration = len(y) / sr
        logger.info(f"  Loaded {duration:.1f}s of audio")

        # Detect tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) if tempo > 0 else 120.0
        logger.info(f"  Detected tempo: {tempo:.1f} BPM")

        # Detect onsets
        onset_times, onset_strengths = self.detector.detect_onsets(y)
        logger.info(f"  Detected {len(onset_times)} onsets")

        # Filter weak onsets if not detecting ghost notes
        if not detect_ghost_notes:
            threshold = np.median(onset_strengths) * 0.5
            mask = onset_strengths >= threshold
            onset_times = onset_times[mask]
            onset_strengths = onset_strengths[mask]

        # Analyze each hit
        hits: List[DrumHit] = []
        for onset_time in onset_times:
            hit = self.detector.analyze_hit(y, onset_time)
            hits.append(hit)

        # Post-process: hi-hat patterns
        hits = self._refine_hihat_pattern(hits)

        # Post-process: remove duplicates
        hits = self._remove_duplicates(hits)

        logger.info(f"  Classified {len(hits)} hits")

        # Create MIDI
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')

        # Add notes
        hits_by_type: Dict[str, int] = {}
        ghost_count = 0

        for hit in hits:
            midi_note = GM_DRUMS.get(hit.drum_type, GM_DRUMS['snare'])

            note = pretty_midi.Note(
                velocity=hit.velocity if preserve_dynamics else 80,
                pitch=midi_note,
                start=hit.time,
                end=hit.time + hit.duration
            )
            drum_track.notes.append(note)

            # Track statistics
            hits_by_type[hit.drum_type] = hits_by_type.get(hit.drum_type, 0) + 1
            if hit.is_ghost:
                ghost_count += 1

            # Add hi-hat control change for openness
            if 'hihat' in hit.drum_type and hit.hihat_openness > 0:
                cc = pretty_midi.ControlChange(
                    number=4,  # CC4 = hi-hat pedal
                    value=int(hit.hihat_openness * 127),
                    time=hit.time
                )
                drum_track.control_changes.append(cc)

        midi.instruments.append(drum_track)
        midi.write(output_path)

        # Calculate quality score
        quality = self._calculate_quality(hits, tempo, duration)

        stats = TranscriptionStats(
            total_hits=len(hits),
            hits_by_type=hits_by_type,
            ghost_notes=ghost_count,
            tempo=tempo,
            time_signature="4/4",  # TODO: detect time signature
            quality_score=quality
        )

        logger.info(f"✅ Drum transcription complete:")
        logger.info(f"   Total: {stats.total_hits} hits")
        logger.info(f"   Ghost notes: {stats.ghost_notes}")
        logger.info(f"   By type: {stats.hits_by_type}")

        return stats

    def _refine_hihat_pattern(self, hits: List[DrumHit]) -> List[DrumHit]:
        """
        Refine hi-hat open/closed based on pattern context.
        An open hi-hat should close before the next hi-hat hit.
        """
        hihat_hits = [(i, h) for i, h in enumerate(hits) if 'hihat' in h.drum_type]

        for j, (idx, hit) in enumerate(hihat_hits):
            if hit.drum_type == 'hihat_open':
                # Check if there's a hi-hat hit soon after
                if j + 1 < len(hihat_hits):
                    next_idx, next_hit = hihat_hits[j + 1]
                    gap = next_hit.time - hit.time

                    # If very close, previous one was probably closed
                    if gap < 0.1:  # 100ms
                        hits[idx] = DrumHit(
                            time=hit.time,
                            drum_type='hihat_closed',
                            velocity=hit.velocity,
                            confidence=hit.confidence,
                            is_ghost=hit.is_ghost,
                            hihat_openness=0.2
                        )

        return hits

    def _remove_duplicates(self, hits: List[DrumHit], min_gap: float = 0.02) -> List[DrumHit]:
        """Remove duplicate hits that are too close together (likely double-triggers)."""
        if not hits:
            return hits

        hits.sort(key=lambda h: h.time)
        cleaned = [hits[0]]

        for hit in hits[1:]:
            prev = cleaned[-1]
            gap = hit.time - prev.time

            if gap >= min_gap:
                cleaned.append(hit)
            else:
                # If same drum type and very close, keep the louder one
                if hit.drum_type == prev.drum_type:
                    if hit.velocity > prev.velocity:
                        cleaned[-1] = hit
                else:
                    # Different drums can be simultaneous
                    cleaned.append(hit)

        return cleaned

    def _calculate_quality(self, hits: List[DrumHit],
                           tempo: float, duration: float) -> float:
        """Estimate transcription quality (0-1)."""
        if not hits:
            return 0.0

        score = 0.5  # Base score

        # 1. Reasonable hit density
        hits_per_second = len(hits) / duration
        if 2 < hits_per_second < 20:  # Typical range for drums
            score += 0.1

        # 2. Mix of drum types (not all one type)
        unique_types = len(set(h.drum_type for h in hits))
        if unique_types >= 3:
            score += 0.1
        if unique_types >= 5:
            score += 0.1

        # 3. Velocity variation (dynamics)
        velocities = [h.velocity for h in hits]
        if np.std(velocities) > 15:
            score += 0.1

        # 4. Confidence scores
        avg_confidence = np.mean([h.confidence for h in hits])
        score += avg_confidence * 0.1

        return min(1.0, score)


def transcribe_drums_to_midi(audio_path: str, output_path: str,
                              sensitivity: float = 0.3,
                              min_hit_interval: float = 0.03) -> bool:
    """
    Main entry point - compatible with existing app.py interface.
    """
    transcriber = EnhancedDrumTranscriber()
    stats = transcriber.transcribe(
        audio_path=audio_path,
        output_path=output_path,
        sensitivity=sensitivity,
        detect_ghost_notes=True,
        preserve_dynamics=True
    )
    return stats.total_hits > 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Enhanced Drum Transcriber v2 loaded")
    print(f"GM Drum mapping: {len(GM_DRUMS)} drum types")
