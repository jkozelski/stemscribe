"""
Audio Boundary Detector for StemScribe
=======================================
Pure audio-feature-based boundary detection. Does NOT label sections (verse/chorus).
Outputs boundary timestamps with confidence scores based on multiple audio features:
  - RMS energy envelope changes
  - Spectral contrast changes
  - Onset density changes
  - Chroma-based self-similarity novelty (checkerboard kernel)

Designed to be consumed by an ensemble structure detector (Agent 4).

Dependencies: librosa, numpy, scipy (pre-installed in venv311).
"""

import numpy as np
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import librosa
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    DEPS_AVAILABLE = True
except ImportError as e:
    DEPS_AVAILABLE = False
    logger.warning(f"audio_boundary_detector: missing dependency — {e}")


# ==============================================================================
# Public API
# ==============================================================================

def detect_audio_boundaries(
    audio_path: str,
    sr: int = 22050,
    hop_length: int = 512,
    energy_window_sec: float = 2.0,
    min_section_sec: float = 6.0,
    novelty_kernel_sec: float = 8.0,
) -> List[Dict]:
    """
    Detect boundary timestamps from audio features alone.

    Args:
        audio_path: Path to audio file (full mix preferred).
        sr: Sample rate for loading.
        hop_length: Hop length for feature extraction (512 = ~23ms at 22050Hz).
        energy_window_sec: Window size for RMS energy smoothing.
        min_section_sec: Minimum time between detected boundaries.
        novelty_kernel_sec: Checkerboard kernel width in seconds for SSM novelty.

    Returns:
        List of boundary dicts sorted by time:
        [
            {
                "time": float,              # boundary timestamp in seconds
                "energy_change": float,     # normalized RMS energy change at boundary
                "spectral_change": float,   # normalized spectral contrast change
                "onset_density_change": float,  # change in onset density
                "chroma_novelty": float,    # checkerboard novelty score from SSM
                "confidence": float,        # combined confidence 0-1
            },
            ...
        ]
    """
    if not DEPS_AVAILABLE:
        logger.error("Cannot detect boundaries — missing dependencies")
        return []

    try:
        logger.info(f"Audio boundary detection: {audio_path}")

        y, sr = librosa.load(str(audio_path), sr=sr)
        duration = librosa.get_duration(y=y, sr=sr)
        logger.info(f"Audio duration: {duration:.1f}s, sr={sr}")

        # Extract all features
        energy_curve, energy_times = _compute_rms_energy(y, sr, hop_length, energy_window_sec)
        spectral_curve, spectral_times = _compute_spectral_contrast(y, sr, hop_length)
        onset_curve, onset_times = _compute_onset_density(y, sr, hop_length, window_sec=4.0)
        chroma_novelty, chroma_times = _compute_chroma_novelty(y, sr, hop_length, novelty_kernel_sec)

        # Detect peaks in each feature's derivative (change signal)
        energy_changes = _compute_change_signal(energy_curve, energy_times)
        spectral_changes = _compute_change_signal(spectral_curve, spectral_times)
        onset_changes = _compute_change_signal(onset_curve, onset_times)

        # Combine into a unified novelty function
        # Resample all signals to a common time grid
        dt = hop_length / sr
        common_times = np.arange(0, duration, dt)

        energy_interp = _resample_to_grid(energy_changes, energy_times, common_times)
        spectral_interp = _resample_to_grid(spectral_changes, spectral_times, common_times)
        onset_interp = _resample_to_grid(onset_changes, onset_times, common_times)
        chroma_interp = _resample_to_grid(chroma_novelty, chroma_times, common_times)

        # Normalize each to [0, 1]
        energy_norm = _normalize(energy_interp)
        spectral_norm = _normalize(spectral_interp)
        onset_norm = _normalize(onset_interp)
        chroma_norm = _normalize(chroma_interp)

        # Weighted combination
        # Chroma novelty is most reliable for harmonic boundaries
        # Energy change is important for verse/chorus transitions
        combined = (
            0.35 * chroma_norm +
            0.30 * energy_norm +
            0.20 * spectral_norm +
            0.15 * onset_norm
        )

        # Smooth the combined signal slightly to reduce spurious peaks
        smooth_frames = max(1, int(1.0 / dt))  # ~1 second smoothing
        combined = np.convolve(combined, np.ones(smooth_frames) / smooth_frames, mode='same')

        # Find peaks in combined novelty
        min_distance = max(1, int(min_section_sec / dt))
        peaks, properties = find_peaks(
            combined,
            height=0.08,        # minimum novelty score
            distance=min_distance,
            prominence=0.04,    # must stand out from surroundings
        )

        # Build boundary list
        boundaries = []
        for p in peaks:
            t = float(common_times[p])
            if t < 2.0 or t > duration - 2.0:
                continue  # skip very start/end

            boundary = {
                "time": round(t, 2),
                "energy_change": round(float(energy_norm[p]), 4),
                "spectral_change": round(float(spectral_norm[p]), 4),
                "onset_density_change": round(float(onset_norm[p]), 4),
                "chroma_novelty": round(float(chroma_norm[p]), 4),
                "confidence": round(float(combined[p]), 4),
            }
            boundaries.append(boundary)

        # Sort by time
        boundaries.sort(key=lambda b: b["time"])

        logger.info(f"Detected {len(boundaries)} audio boundaries:")
        for b in boundaries:
            logger.info(
                f"  t={b['time']:.1f}s  conf={b['confidence']:.3f}  "
                f"energy={b['energy_change']:.3f}  spectral={b['spectral_change']:.3f}  "
                f"onset={b['onset_density_change']:.3f}  chroma={b['chroma_novelty']:.3f}"
            )

        return boundaries

    except Exception as e:
        logger.error(f"Audio boundary detection failed: {e}", exc_info=True)
        return []


# ==============================================================================
# Feature extraction
# ==============================================================================

def _compute_rms_energy(y, sr, hop_length, window_sec):
    """Compute RMS energy envelope smoothed over window_sec."""
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    # Smooth with a moving average over window_sec
    n_smooth = max(1, int(window_sec * sr / hop_length))
    if n_smooth > 1:
        kernel = np.ones(n_smooth) / n_smooth
        rms = np.convolve(rms, kernel, mode='same')

    return rms, times


def _compute_spectral_contrast(y, sr, hop_length):
    """
    Compute spectral contrast — measures difference between peaks and valleys
    in the spectrum. Chorus typically has broader, louder spectral content.
    Returns the mean across frequency bands at each time frame.
    """
    contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, hop_length=hop_length, n_bands=6
    )
    # Average across bands to get a single curve per time frame
    mean_contrast = np.mean(contrast, axis=0)
    times = librosa.times_like(mean_contrast, sr=sr, hop_length=hop_length)

    # Smooth over ~2s
    n_smooth = max(1, int(2.0 * sr / hop_length))
    if n_smooth > 1:
        kernel = np.ones(n_smooth) / n_smooth
        mean_contrast = np.convolve(mean_contrast, kernel, mode='same')

    return mean_contrast, times


def _compute_onset_density(y, sr, hop_length, window_sec=4.0):
    """
    Compute onset density — number of note onsets per second,
    smoothed over window_sec. Sections with more rhythmic activity
    (e.g., chorus with strumming) have higher onset density.
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)

    # Smooth over window_sec
    n_smooth = max(1, int(window_sec * sr / hop_length))
    if n_smooth > 1:
        kernel = np.ones(n_smooth) / n_smooth
        onset_env = np.convolve(onset_env, kernel, mode='same')

    return onset_env, times


def _compute_chroma_novelty(y, sr, hop_length, kernel_sec):
    """
    Compute checkerboard novelty from chroma self-similarity matrix.

    This is the most musically meaningful feature — it detects when
    the harmonic content changes (e.g., verse chords -> chorus chords).

    Uses a dedicated larger hop (2048) for the SSM to keep the matrix
    manageable while still providing better resolution than the original
    structure_detector.py (which used hop=4096 = ~186ms).
    With hop=2048 at sr=22050: ~93ms per frame, ~2580 frames for 240s.
    """
    # Use a dedicated hop for chroma SSM — 2048 is a good balance
    # between resolution and computation
    ssm_hop = 2048

    # Use CQT chroma for better harmonic representation
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=ssm_hop)

    # Apply median filter to reduce transient noise
    chroma = median_filter(chroma, size=(1, 9))

    times = librosa.times_like(chroma, sr=sr, hop_length=ssm_hop)
    n_frames = chroma.shape[1]

    # Build recurrence/self-similarity matrix
    R = librosa.segment.recurrence_matrix(
        chroma, width=3, mode='affinity', sym=True
    )

    # Checkerboard kernel for novelty detection
    dt = ssm_hop / sr
    kernel_frames = min(int(kernel_sec / dt), n_frames // 4)
    kernel_frames = max(kernel_frames, 4)

    # Make it even
    if kernel_frames % 2 != 0:
        kernel_frames += 1

    half = kernel_frames // 2

    logger.info(f"Chroma SSM: {n_frames} frames, kernel={kernel_frames} frames ({kernel_frames * dt:.1f}s)")

    novelty = np.zeros(n_frames)
    kernel = np.ones((kernel_frames, kernel_frames))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    for i in range(half, n_frames - half):
        block = R[i - half:i + half, i - half:i + half]
        if block.shape == kernel.shape:
            novelty[i] = max(0, np.sum(block * kernel))

    # Normalize
    nmax = np.max(novelty)
    if nmax > 0:
        novelty /= nmax

    return novelty, times


# ==============================================================================
# Signal processing helpers
# ==============================================================================

def _compute_change_signal(curve, times):
    """
    Compute the absolute first derivative of a feature curve.
    This highlights where the feature changes rapidly (i.e., boundaries).
    """
    if len(curve) < 2:
        return curve

    # First derivative (absolute value of change)
    diff = np.abs(np.diff(curve))
    # Pad to same length
    diff = np.append(diff, 0)

    # Smooth slightly to avoid noise spikes
    if len(diff) > 5:
        kernel = np.ones(5) / 5
        diff = np.convolve(diff, kernel, mode='same')

    return diff


def _resample_to_grid(signal, signal_times, target_times):
    """Resample a signal to a common time grid via linear interpolation."""
    if len(signal) == 0 or len(signal_times) == 0:
        return np.zeros(len(target_times))

    # Ensure same length
    min_len = min(len(signal), len(signal_times))
    signal = signal[:min_len]
    signal_times = signal_times[:min_len]

    return np.interp(target_times, signal_times, signal)


def _normalize(signal):
    """Normalize signal to [0, 1] range."""
    smin = np.min(signal)
    smax = np.max(signal)
    if smax - smin < 1e-10:
        return np.zeros_like(signal)
    return (signal - smin) / (smax - smin)


# ==============================================================================
# CLI for testing
# ==============================================================================

if __name__ == "__main__":
    import sys
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python audio_boundary_detector.py <audio_path>")
        sys.exit(1)

    audio = sys.argv[1]
    boundaries = detect_audio_boundaries(audio)
    print(json.dumps(boundaries, indent=2))
