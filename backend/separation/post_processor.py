"""
Post-Processing Pipeline for Audio Separation
Phase alignment, bleed reduction, noise reduction, artifact removal
"""

import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PostProcessor:
    """
    Advanced post-processing for separated audio stems.

    Pipeline:
    1. Phase alignment (minimize cross-stem interference)
    2. Bleed reduction (spectral subtraction)
    3. Noise reduction (spectral gating)
    4. Artifact removal (smooth discontinuities)
    5. Loudness normalization
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def process_all_stems(self,
                         stems: Dict[str, np.ndarray],
                         original_mix: np.ndarray = None,
                         config: Dict = None) -> Dict[str, np.ndarray]:
        """
        Apply full post-processing pipeline to all stems.

        Args:
            stems: Dict of stem_name -> audio array
            original_mix: Original mixed audio (optional)
            config: Post-processing configuration

        Returns:
            Processed stems dict
        """
        config = config or {}

        processed = {}

        for stem_name, audio in stems.items():
            logger.info(f"Post-processing {stem_name}...")

            result = audio.copy()

            # 1. Bleed reduction
            if config.get('bleed_reduction', {}).get('enabled', True):
                other_stems = {k: v for k, v in stems.items() if k != stem_name}
                aggressiveness = config.get('bleed_reduction', {}).get('aggressiveness', 0.5)
                result = self.reduce_bleed(result, other_stems, aggressiveness)

            # 2. Noise reduction
            if config.get('noise_reduction', {}).get('enabled', True):
                threshold_db = config.get('noise_reduction', {}).get('threshold_db', -40)
                result = self.reduce_noise(result, threshold_db)

            # 3. Artifact removal
            if config.get('artifact_removal', True):
                result = self.remove_artifacts(result)

            # 4. Loudness normalization
            if config.get('loudness_normalization', {}).get('enabled', True):
                target_lufs = config.get('loudness_normalization', {}).get('target_lufs', -14.0)
                result = self.normalize_loudness(result, target_lufs)

            processed[stem_name] = result

        # 5. Phase alignment across stems
        if config.get('phase_alignment', True) and original_mix is not None:
            processed = self.phase_align(processed, original_mix)

        return processed

    def reduce_bleed(self,
                    target: np.ndarray,
                    other_stems: Dict[str, np.ndarray],
                    aggressiveness: float = 0.5) -> np.ndarray:
        """
        Reduce cross-stem bleed using spectral subtraction.

        Args:
            target: Target stem audio
            other_stems: Dict of other stems to subtract
            aggressiveness: 0-1, higher = more aggressive subtraction
        """
        try:
            import librosa

            # Convert to mono for processing if stereo
            is_stereo = target.ndim > 1 and target.shape[0] == 2

            if is_stereo:
                # Process each channel
                left = self._reduce_bleed_mono(target[0], other_stems, aggressiveness)
                right = self._reduce_bleed_mono(target[1], other_stems, aggressiveness)
                return np.stack([left, right])
            else:
                return self._reduce_bleed_mono(target, other_stems, aggressiveness)

        except ImportError:
            logger.warning("librosa not available for bleed reduction")
            return target

    def _reduce_bleed_mono(self,
                          target: np.ndarray,
                          other_stems: Dict[str, np.ndarray],
                          aggressiveness: float) -> np.ndarray:
        """Bleed reduction for mono signal"""
        import librosa

        n_fft = 2048
        hop_length = 512

        target_stft = librosa.stft(target.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
        target_mag = np.abs(target_stft)
        target_phase = np.angle(target_stft)

        # Compute combined magnitude of other stems
        other_mag = np.zeros_like(target_mag)
        for stem in other_stems.values():
            if stem.ndim > 1:
                stem = np.mean(stem, axis=0)

            # Ensure same length
            min_len = min(len(target), len(stem))
            stem_stft = librosa.stft(stem[:min_len].astype(np.float32),
                                    n_fft=n_fft, hop_length=hop_length)

            # Pad if needed
            if stem_stft.shape[1] < other_mag.shape[1]:
                pad_width = other_mag.shape[1] - stem_stft.shape[1]
                stem_stft = np.pad(stem_stft, ((0, 0), (0, pad_width)))
            elif stem_stft.shape[1] > other_mag.shape[1]:
                stem_stft = stem_stft[:, :other_mag.shape[1]]

            other_mag += np.abs(stem_stft)

        # Soft spectral subtraction with masking
        # Prevent over-subtraction
        mask = 1.0 - (aggressiveness * other_mag / (target_mag + 1e-8))
        mask = np.clip(mask, 0.15, 1.0)  # Don't subtract more than 85%

        target_mag_clean = target_mag * mask

        # Reconstruct
        result_stft = target_mag_clean * np.exp(1j * target_phase)
        result = librosa.istft(result_stft, hop_length=hop_length, length=len(target))

        return result

    def reduce_noise(self,
                    audio: np.ndarray,
                    threshold_db: float = -40) -> np.ndarray:
        """
        Reduce noise using spectral gating.

        Args:
            audio: Input audio
            threshold_db: Noise gate threshold in dB
        """
        try:
            import noisereduce as nr

            if audio.ndim > 1:
                # Process stereo
                left = nr.reduce_noise(y=audio[0], sr=self.sample_rate,
                                      thresh_n_mult_nonstationary=2,
                                      stationary=False)
                right = nr.reduce_noise(y=audio[1], sr=self.sample_rate,
                                       thresh_n_mult_nonstationary=2,
                                       stationary=False)
                return np.stack([left, right])
            else:
                return nr.reduce_noise(y=audio, sr=self.sample_rate,
                                      thresh_n_mult_nonstationary=2,
                                      stationary=False)

        except ImportError:
            logger.warning("noisereduce not available, using simple gate")
            return self._simple_noise_gate(audio, threshold_db)

    def _simple_noise_gate(self,
                          audio: np.ndarray,
                          threshold_db: float) -> np.ndarray:
        """Simple noise gate fallback"""
        threshold_linear = 10 ** (threshold_db / 20)

        # Compute envelope
        window_size = int(0.01 * self.sample_rate)  # 10ms window

        if audio.ndim > 1:
            envelope = np.max([
                np.convolve(np.abs(audio[0]), np.ones(window_size)/window_size, 'same'),
                np.convolve(np.abs(audio[1]), np.ones(window_size)/window_size, 'same')
            ], axis=0)
        else:
            envelope = np.convolve(np.abs(audio), np.ones(window_size)/window_size, 'same')

        # Soft gate
        gate = np.clip(envelope / threshold_linear, 0, 1) ** 0.5

        if audio.ndim > 1:
            return audio * gate[np.newaxis, :]
        else:
            return audio * gate

    def remove_artifacts(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove common separation artifacts.

        - Musical noise (isolated spectral peaks)
        - Click/pop artifacts
        - Ringing
        """
        try:
            import librosa
            from scipy import ndimage

            if audio.ndim > 1:
                left = self._remove_artifacts_mono(audio[0])
                right = self._remove_artifacts_mono(audio[1])
                return np.stack([left, right])
            else:
                return self._remove_artifacts_mono(audio)

        except ImportError:
            return audio

    def _remove_artifacts_mono(self, audio: np.ndarray) -> np.ndarray:
        """Artifact removal for mono signal"""
        import librosa
        from scipy import ndimage

        n_fft = 2048
        hop_length = 512

        stft = librosa.stft(audio.astype(np.float32), n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # 1. Remove isolated peaks (musical noise)
        # Local median filter
        magnitude_smooth = ndimage.median_filter(magnitude, size=(3, 3))

        # Detect anomalies
        anomaly_mask = magnitude > (magnitude_smooth * 3)

        # Soften anomalies instead of removing completely
        magnitude[anomaly_mask] = magnitude_smooth[anomaly_mask] * 1.5

        # 2. Smooth phase transitions
        # Apply slight Gaussian blur to phase
        phase_smooth = ndimage.gaussian_filter(phase, sigma=0.5)

        # 3. Temporal smoothing on magnitude
        magnitude = ndimage.gaussian_filter1d(magnitude, sigma=1.0, axis=1)

        # Reconstruct
        result = librosa.istft(magnitude * np.exp(1j * phase_smooth),
                              hop_length=hop_length, length=len(audio))

        return result

    def normalize_loudness(self,
                          audio: np.ndarray,
                          target_lufs: float = -14.0) -> np.ndarray:
        """
        Normalize loudness to target LUFS.

        Args:
            audio: Input audio
            target_lufs: Target loudness in LUFS
        """
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(self.sample_rate)

            if audio.ndim > 1:
                # Transpose for pyloudnorm (expects samples, channels)
                audio_t = audio.T
                loudness = meter.integrated_loudness(audio_t)

                if np.isinf(loudness) or loudness < -70:
                    return audio  # Too quiet to normalize

                normalized = pyln.normalize.loudness(audio_t, loudness, target_lufs)
                return normalized.T
            else:
                # Mono - make stereo for measurement
                audio_stereo = np.stack([audio, audio]).T
                loudness = meter.integrated_loudness(audio_stereo)

                if np.isinf(loudness) or loudness < -70:
                    return audio

                normalized = pyln.normalize.loudness(audio_stereo, loudness, target_lufs)
                return normalized[:, 0]

        except ImportError:
            logger.warning("pyloudnorm not available, using peak normalization")
            return self._peak_normalize(audio, target_db=-1.0)

    def _peak_normalize(self, audio: np.ndarray, target_db: float = -1.0) -> np.ndarray:
        """Simple peak normalization fallback"""
        peak = np.max(np.abs(audio))
        if peak < 1e-8:
            return audio

        target_linear = 10 ** (target_db / 20)
        return audio * (target_linear / peak)

    def phase_align(self,
                   stems: Dict[str, np.ndarray],
                   original_mix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Phase-align stems to minimize artifacts when summed.

        Uses cross-correlation to find optimal alignment.
        """
        try:
            from scipy import signal

            aligned = {}

            # Use mix as reference for alignment
            if original_mix.ndim > 1:
                ref = np.mean(original_mix, axis=0)
            else:
                ref = original_mix

            for stem_name, stem in stems.items():
                if stem.ndim > 1:
                    stem_mono = np.mean(stem, axis=0)
                else:
                    stem_mono = stem

                # Find optimal lag using cross-correlation
                corr = signal.correlate(ref[:len(stem_mono)], stem_mono, mode='full')
                lag = np.argmax(np.abs(corr)) - len(stem_mono) + 1

                # Apply shift (limit to small corrections)
                lag = np.clip(lag, -1000, 1000)  # Max ~23ms at 44.1kHz

                if lag != 0:
                    if stem.ndim > 1:
                        aligned[stem_name] = np.roll(stem, lag, axis=1)
                    else:
                        aligned[stem_name] = np.roll(stem, lag)
                    logger.debug(f"Aligned {stem_name} by {lag} samples")
                else:
                    aligned[stem_name] = stem

            return aligned

        except ImportError:
            logger.warning("scipy not available for phase alignment")
            return stems
