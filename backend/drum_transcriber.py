"""
Drum Transcription Module for StemScribe - Improved Version

Uses onset detection + multi-band frequency analysis to transcribe drums to MIDI.
Detects: kick, snare, hi-hat (closed/open), toms, crash, ride

Improvements over basic version:
- Better frequency band thresholds
- Spectral flatness for cymbal detection
- Attack transient analysis
- Tom detection (low-mid with pitch)
"""

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# General MIDI drum note numbers (GM Standard)
GM_DRUMS = {
    'kick': 36,       # Bass Drum 1
    'snare': 38,      # Acoustic Snare
    'snare_rim': 37,  # Side Stick
    'hihat_closed': 42,  # Closed Hi-Hat
    'hihat_open': 46,    # Open Hi-Hat
    'crash': 49,      # Crash Cymbal 1
    'ride': 51,       # Ride Cymbal 1
    'tom_high': 50,   # High Tom
    'tom_mid': 47,    # Low-Mid Tom
    'tom_low': 45,    # Low Tom
    'tom_floor': 43,  # High Floor Tom
}


def transcribe_drums_to_midi(audio_path: str, output_path: str,
                              sensitivity: float = 0.3,
                              min_hit_interval: float = 0.03) -> bool:
    """
    Transcribe a drum audio file to MIDI with improved detection.
    """
    try:
        import librosa
        import pretty_midi
    except ImportError as e:
        logger.error(f"Required library not installed: {e}")
        return False

    logger.info(f"🥁 Transcribing drums from {audio_path}")

    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        duration = len(y) / sr
        logger.info(f"Loaded {duration:.1f}s of audio at {sr}Hz")

        # Detect onsets with better settings
        onset_env = librosa.onset.onset_strength(
            y=y, sr=sr,
            hop_length=512,
            aggregate=np.median  # More robust to noise
        )

        # Dynamic threshold based on signal
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=512,
            backtrack=True,  # Find exact onset
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=5,
            delta=sensitivity * np.mean(onset_env),
            wait=int(min_hit_interval * sr / 512)
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

        logger.info(f"Detected {len(onset_times)} drum hits")

        if len(onset_times) == 0:
            logger.warning("No drum hits detected")
            return False

        # Detect tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) if tempo > 0 else 120.0
        logger.info(f"Detected tempo: {tempo:.1f} BPM")

        # Create MIDI
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        drum_track = pretty_midi.Instrument(program=0, is_drum=True, name='Drums')

        # Track hits by type for logging
        hits_by_type = {}

        for onset_time in onset_times:
            # Get analysis window (50ms for attack analysis)
            start_sample = int(onset_time * sr)
            window_size = int(0.05 * sr)
            end_sample = min(start_sample + window_size, len(y))

            if end_sample <= start_sample + 100:
                continue

            segment = y[start_sample:end_sample]

            # Classify the hit
            drum_type, velocity = classify_drum_hit_improved(segment, sr)

            if drum_type:
                hits_by_type[drum_type] = hits_by_type.get(drum_type, 0) + 1

                # Get MIDI note number
                midi_note = GM_DRUMS.get(drum_type, GM_DRUMS['snare'])

                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=midi_note,
                    start=onset_time,
                    end=onset_time + 0.05  # Shorter for tighter feel
                )
                drum_track.notes.append(note)

        midi.instruments.append(drum_track)
        midi.write(output_path)

        logger.info(f"✅ Drum transcription: {hits_by_type}")
        logger.info(f"Total: {len(drum_track.notes)} hits → {output_path}")

        return True

    except Exception as e:
        logger.error(f"Drum transcription failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def classify_drum_hit_improved(segment: np.ndarray, sr: int) -> tuple:
    """
    Improved drum hit classification using multiple features.
    """
    import librosa

    # === Spectral Analysis ===
    # Use shorter FFT for better time resolution on transients
    n_fft = min(1024, len(segment))
    spectrum = np.abs(np.fft.rfft(segment, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    # Frequency bands (Hz)
    # Kick: 40-100 Hz (fundamental)
    # Snare body: 150-300 Hz
    # Snare wires: 2000-5000 Hz
    # Toms: 80-400 Hz (with pitch)
    # Hi-hat: 5000-12000 Hz
    # Cymbals: 3000-10000 Hz (broader, more wash)

    kick_mask = (freqs >= 40) & (freqs < 120)
    snare_low_mask = (freqs >= 150) & (freqs < 350)
    snare_high_mask = (freqs >= 2000) & (freqs < 6000)
    tom_mask = (freqs >= 80) & (freqs < 500)
    hihat_mask = (freqs >= 6000) & (freqs < 14000)
    cymbal_mask = (freqs >= 2500) & (freqs < 12000)
    low_mid_mask = (freqs >= 100) & (freqs < 800)

    def band_energy(mask):
        return np.sum(spectrum[mask] ** 2) if np.any(mask) else 0

    kick_energy = band_energy(kick_mask)
    snare_low_energy = band_energy(snare_low_mask)
    snare_high_energy = band_energy(snare_high_mask)
    tom_energy = band_energy(tom_mask)
    hihat_energy = band_energy(hihat_mask)
    cymbal_energy = band_energy(cymbal_mask)
    low_mid_energy = band_energy(low_mid_mask)

    total_energy = np.sum(spectrum ** 2) + 1e-10

    # Normalize
    kick_ratio = kick_energy / total_energy
    snare_low_ratio = snare_low_energy / total_energy
    snare_high_ratio = snare_high_energy / total_energy
    hihat_ratio = hihat_energy / total_energy
    cymbal_ratio = cymbal_energy / total_energy
    low_mid_ratio = low_mid_energy / total_energy

    # === Spectral Flatness ===
    # Cymbals/hi-hats have high spectral flatness (noise-like)
    # Kicks/toms have low flatness (tonal)
    spectral_flatness = librosa.feature.spectral_flatness(y=segment, n_fft=n_fft)[0]
    avg_flatness = np.mean(spectral_flatness)

    # === Spectral Centroid ===
    # Where the "center of mass" of the spectrum is
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=n_fft)[0]
    avg_centroid = np.mean(centroid)

    # === Velocity ===
    rms = np.sqrt(np.mean(segment ** 2))
    velocity = min(127, max(30, int(rms * 800)))

    # === Classification Logic ===

    # KICK: Strong low end, low flatness, low centroid
    if kick_ratio > 0.25 and avg_centroid < 800 and avg_flatness < 0.3:
        return 'kick', velocity

    # HI-HAT: Very high frequencies, high flatness
    if hihat_ratio > 0.2 and avg_centroid > 5000 and avg_flatness > 0.4:
        # Distinguish closed vs open by duration/decay
        # Open hi-hat has more sustain in the segment
        segment_decay = np.abs(segment[-len(segment)//4:]).mean() / (np.abs(segment[:len(segment)//4]).mean() + 1e-10)
        if segment_decay > 0.3:
            return 'hihat_open', min(velocity, 90)
        return 'hihat_closed', min(velocity, 85)

    # CRASH/RIDE: Cymbal frequencies, high flatness, but broader than hi-hat
    if cymbal_ratio > 0.25 and avg_flatness > 0.35 and avg_centroid > 2500:
        # Crash has more attack, ride is steadier
        if velocity > 80:
            return 'crash', velocity
        return 'ride', min(velocity, 90)

    # SNARE: Mid frequencies + snare wire buzz (high freq component)
    if snare_low_ratio > 0.15 and snare_high_ratio > 0.08:
        return 'snare', velocity

    # Also catch snare by mid-range + high flatness (wire buzz)
    if low_mid_ratio > 0.3 and avg_flatness > 0.25 and avg_centroid > 1000 and avg_centroid < 4000:
        return 'snare', velocity

    # TOMS: Low-mid with some tonality (low flatness)
    if low_mid_ratio > 0.35 and avg_flatness < 0.3 and avg_centroid < 1500:
        # Classify tom by centroid
        if avg_centroid < 400:
            return 'tom_floor', velocity
        elif avg_centroid < 600:
            return 'tom_low', velocity
        elif avg_centroid < 900:
            return 'tom_mid', velocity
        else:
            return 'tom_high', velocity

    # Default fallback based on centroid
    if avg_centroid < 500:
        return 'kick', velocity
    elif avg_centroid > 4000:
        return 'hihat_closed', min(velocity, 80)
    else:
        return 'snare', velocity


def estimate_tempo(audio_path: str) -> float:
    """Estimate tempo from drum track."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except:
        return 120.0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Improved drum transcriber loaded")
    print(f"GM Drum mapping: {GM_DRUMS}")
