"""
Quality Metrics for Audio Separation
SNR, cross-correlation, spectral analysis for stem quality assessment
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class QualityMetrics:
    """
    Comprehensive quality assessment for separated audio stems.
    Used for ensemble voting and quality reporting.
    """

    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate

    def compute_all_metrics(self,
                           target_stem: np.ndarray,
                           other_stems: Dict[str, np.ndarray],
                           original_mix: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all quality metrics for a stem.

        Returns:
            Dict with snr_db, cross_correlation, spectral_entropy, harmonic_mean
        """
        snr = self.compute_snr(target_stem, other_stems)
        cross_corr = self.compute_cross_correlation(target_stem, other_stems)
        entropy = self.compute_spectral_entropy(target_stem)

        # Harmonic mean prioritizes worst metric (robust to bad stems)
        # Normalize metrics to 0-1 range first
        snr_norm = min(1.0, max(0, snr / 20.0))  # 20dB = perfect
        corr_norm = 1.0 - cross_corr  # Lower correlation is better
        entropy_norm = min(1.0, entropy / 8.0)  # 8 = very complex

        # Harmonic mean
        metrics_list = [snr_norm, corr_norm, entropy_norm]
        harmonic_mean = len(metrics_list) / sum(1.0 / (m + 0.001) for m in metrics_list)

        return {
            'snr_db': snr,
            'cross_correlation': cross_corr,
            'spectral_entropy': entropy,
            'snr_normalized': snr_norm,
            'correlation_score': corr_norm,
            'entropy_normalized': entropy_norm,
            'harmonic_mean': harmonic_mean,
            'quality_score': harmonic_mean  # Alias
        }

    def compute_snr(self,
                   target_stem: np.ndarray,
                   other_stems: Dict[str, np.ndarray]) -> float:
        """
        Signal-to-Noise Ratio: Energy ratio of target to contamination.

        Higher = better separation (less bleed from other stems)
        """
        # Ensure mono for calculation
        if target_stem.ndim > 1:
            target_mono = np.mean(target_stem, axis=0)
        else:
            target_mono = target_stem

        target_energy = np.sum(target_mono ** 2) + 1e-10

        # Sum energy from other stems (contamination)
        contamination_energy = 0
        for stem_name, stem in other_stems.items():
            if stem.ndim > 1:
                stem_mono = np.mean(stem, axis=0)
            else:
                stem_mono = stem

            # Trim to same length
            min_len = min(len(target_mono), len(stem_mono))
            contamination_energy += np.sum(stem_mono[:min_len] ** 2)

        contamination_energy = max(contamination_energy, 1e-10)

        snr_linear = target_energy / contamination_energy
        snr_db = 10 * np.log10(snr_linear)

        return float(snr_db)

    def compute_cross_correlation(self,
                                  target_stem: np.ndarray,
                                  other_stems: Dict[str, np.ndarray]) -> float:
        """
        Cross-stem correlation: Measure of bleed (0-1).

        0 = no correlation (perfect separation)
        1 = perfect correlation (maximum bleed)
        """
        if target_stem.ndim > 1:
            target_mono = np.mean(target_stem, axis=0)
        else:
            target_mono = target_stem

        correlations = []

        for stem_name, stem in other_stems.items():
            if stem.ndim > 1:
                stem_mono = np.mean(stem, axis=0)
            else:
                stem_mono = stem

            # Ensure same length
            min_len = min(len(target_mono), len(stem_mono))
            if min_len < 1000:
                continue

            t = target_mono[:min_len]
            s = stem_mono[:min_len]

            # Normalize
            t = (t - np.mean(t)) / (np.std(t) + 1e-10)
            s = (s - np.mean(s)) / (np.std(s) + 1e-10)

            # Correlation coefficient
            corr = np.abs(np.corrcoef(t, s)[0, 1])
            if not np.isnan(corr):
                correlations.append(corr)

        if not correlations:
            return 0.0

        return float(np.mean(correlations))

    def compute_spectral_entropy(self, stem: np.ndarray) -> float:
        """
        Spectral entropy: Measure of spectral complexity.

        Higher = more complex/musical content
        Lower = more tonal/simple (might indicate artifacts)
        """
        try:
            import librosa

            if stem.ndim > 1:
                stem_mono = np.mean(stem, axis=0)
            else:
                stem_mono = stem

            # Compute spectrogram
            stft = librosa.stft(stem_mono.astype(np.float32), n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)

            # Normalize to probability distribution per frame
            power = magnitude ** 2
            power_sum = np.sum(power, axis=0, keepdims=True) + 1e-10
            power_norm = power / power_sum

            # Shannon entropy per frame
            entropy_per_frame = -np.sum(power_norm * np.log2(power_norm + 1e-10), axis=0)

            # Average entropy
            mean_entropy = np.mean(entropy_per_frame)

            # Normalize by max possible entropy (log2 of frequency bins)
            max_entropy = np.log2(magnitude.shape[0])
            normalized_entropy = mean_entropy / max_entropy

            return float(normalized_entropy * 10)  # Scale to ~0-10 range

        except ImportError:
            logger.warning("librosa not available for spectral entropy")
            return 5.0  # Neutral value

    def compute_loudness(self, stem: np.ndarray) -> float:
        """Compute RMS loudness in dB"""
        if stem.ndim > 1:
            stem_mono = np.mean(stem, axis=0)
        else:
            stem_mono = stem

        rms = np.sqrt(np.mean(stem_mono ** 2))
        db = 20 * np.log10(rms + 1e-10)

        return float(db)

    def detect_clipping(self, stem: np.ndarray, threshold: float = 0.99) -> float:
        """Detect percentage of clipped samples"""
        if stem.ndim > 1:
            samples = stem.flatten()
        else:
            samples = stem

        # Normalize to -1, 1 range
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val

        clipped = np.sum(np.abs(samples) >= threshold)
        return float(clipped / len(samples))

    def generate_quality_report(self,
                               stems: Dict[str, np.ndarray],
                               model_name: str = 'unknown') -> Dict:
        """
        Generate comprehensive quality report for all stems.
        """
        report = {
            'model': model_name,
            'stems': {}
        }

        stem_names = list(stems.keys())

        for stem_name, stem in stems.items():
            # Get other stems for cross-correlation
            other_stems = {k: v for k, v in stems.items() if k != stem_name}

            metrics = self.compute_all_metrics(stem, other_stems)
            metrics['loudness_db'] = self.compute_loudness(stem)
            metrics['clipping_ratio'] = self.detect_clipping(stem)

            report['stems'][stem_name] = metrics

        # Overall quality score (average of harmonic means)
        overall = np.mean([s['harmonic_mean'] for s in report['stems'].values()])
        report['overall_quality'] = float(overall)

        return report
