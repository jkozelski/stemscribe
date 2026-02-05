"""
Stereo Splitter for StemScribe

Splits any stereo stem into left-panned and right-panned components.
Works with any instrument that's been panned in the mix:
- Guitars (Jerry & Bob in Grateful Dead)
- Keyboards (Hammond panned left, Rhodes right)
- Backing vocals (spread across stereo field)
- Horns, strings, synths, etc.

Methods:
1. Simple L/R split - separates left and right channels
2. Mid-Side processing - extracts center vs side content
3. Enhanced (default) - frequency-aware panning detection

"""

import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa/soundfile not available for stereo splitting")


def split_stereo(input_path: str, output_dir: str = None,
                 method: str = 'enhanced', stem_type: str = None) -> dict:
    """
    Split any stereo stem into left-panned and right-panned components.

    Args:
        input_path: Path to the stereo audio file
        output_dir: Directory for output files (default: same as input)
        method: 'simple' (L/R), 'mid_side', or 'enhanced' (combination)
        stem_type: Type of stem for labeling (e.g., 'guitar', 'piano', 'other')

    Returns:
        Dictionary with paths: {'{stem}_left': path, '{stem}_right': path, '{stem}_center': path}
    """
    if not LIBROSA_AVAILABLE:
        logger.error("librosa/soundfile required for stereo splitting")
        return {}

    input_path = Path(input_path)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return {}

    if output_dir is None:
        output_dir = input_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect stem type from filename if not provided
    if stem_type is None:
        stem_type = input_path.stem.lower()
        # Clean up common suffixes
        for suffix in ['_enhanced', '_processed', '_stem']:
            stem_type = stem_type.replace(suffix, '')

    logger.info(f"🎚️ Splitting stereo {stem_type}: {input_path}")

    try:
        # Load audio in stereo
        y, sr = librosa.load(input_path, sr=None, mono=False)

        # Check if actually stereo
        if y.ndim == 1:
            logger.warning("Input is mono - cannot split stereo")
            return {stem_type: str(input_path)}

        if y.shape[0] != 2:
            logger.warning(f"Unexpected channel count: {y.shape[0]}")
            return {stem_type: str(input_path)}

        left = y[0]
        right = y[1]

        logger.info(f"Loaded stereo audio: {len(left)/sr:.1f}s at {sr}Hz")

        # Analyze stereo field
        pan_info = analyze_stereo_field(left, right)
        logger.info(f"Stereo analysis: {pan_info}")

        if method == 'simple':
            results = split_simple_lr(left, right, sr, output_dir, input_path.stem, stem_type)
        elif method == 'mid_side':
            results = split_mid_side(left, right, sr, output_dir, input_path.stem, stem_type)
        else:  # enhanced - the default
            results = split_enhanced(left, right, sr, output_dir, input_path.stem, stem_type, pan_info)

        logger.info(f"✅ Stereo split complete: {list(results.keys())}")
        return results

    except Exception as e:
        logger.error(f"Stereo splitting failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def analyze_stereo_field(left: np.ndarray, right: np.ndarray) -> dict:
    """
    Analyze the stereo field to detect how content is panned.

    Returns info about:
    - Overall balance (left-heavy, right-heavy, centered)
    - Stereo width
    - Whether it seems like two distinct sources
    """
    # Calculate energies
    left_energy = np.sum(left ** 2)
    right_energy = np.sum(right ** 2)
    total_energy = left_energy + right_energy + 1e-10

    # Balance: -1 = all left, +1 = all right, 0 = centered
    balance = (right_energy - left_energy) / total_energy

    # Correlation: 1 = mono, 0 = uncorrelated, -1 = out of phase
    correlation = np.corrcoef(left, right)[0, 1]

    # Stereo width: low correlation = wider stereo
    width = 1 - abs(correlation)

    # Calculate mid and side
    mid = (left + right) / 2
    side = (left - right) / 2

    mid_energy = np.sum(mid ** 2)
    side_energy = np.sum(side ** 2)

    # Side ratio: how much content is on the sides vs center
    side_ratio = side_energy / (mid_energy + side_energy + 1e-10)

    # Detect if left and right seem like distinct sources
    # Compare spectral content between left-only and right-only portions
    left_dominant = left - np.minimum(np.abs(left), np.abs(right)) * np.sign(left)
    right_dominant = right - np.minimum(np.abs(left), np.abs(right)) * np.sign(right)

    # Lower threshold - even slight stereo separation can have distinct sources
    distinct_sources = side_ratio > 0.08 and width > 0.15

    return {
        'balance': float(balance),
        'correlation': float(correlation),
        'width': float(width),
        'side_ratio': float(side_ratio),
        'distinct_sources': distinct_sources,
        'recommendation': 'splittable' if distinct_sources else 'mostly_mono'
    }


def split_simple_lr(left: np.ndarray, right: np.ndarray, sr: int,
                    output_dir: Path, file_stem: str, stem_type: str) -> dict:
    """
    Simple left/right channel split.
    Creates mono files from each channel.
    """
    results = {}

    # Left channel
    left_path = output_dir / f"{file_stem}_left.wav"
    sf.write(str(left_path), left, sr)
    results[f'{stem_type}_left'] = str(left_path)
    logger.info(f"  → Left channel: {left_path.name}")

    # Right channel
    right_path = output_dir / f"{file_stem}_right.wav"
    sf.write(str(right_path), right, sr)
    results[f'{stem_type}_right'] = str(right_path)
    logger.info(f"  → Right channel: {right_path.name}")

    return results


def split_mid_side(left: np.ndarray, right: np.ndarray, sr: int,
                   output_dir: Path, file_stem: str, stem_type: str) -> dict:
    """
    Mid-Side split.
    Mid = center content (what's identical in both channels)
    Side = panned content (distinct left/right content)
    """
    results = {}

    # Mid (center) = (L + R) / 2
    mid = (left + right) / 2

    # Side = (L - R) / 2
    side = (left - right) / 2

    # Normalize to prevent clipping
    mid_max = np.max(np.abs(mid)) + 1e-10
    side_max = np.max(np.abs(side)) + 1e-10

    mid = mid / max(mid_max, 1.0) * 0.9
    side = side / max(side_max, 1.0) * 0.9

    # Center content
    mid_path = output_dir / f"{file_stem}_center.wav"
    sf.write(str(mid_path), mid, sr)
    results[f'{stem_type}_center'] = str(mid_path)
    logger.info(f"  → Center (mid): {mid_path.name}")

    # Side content (difference between L and R)
    side_path = output_dir / f"{file_stem}_sides.wav"
    sf.write(str(side_path), side, sr)
    results[f'{stem_type}_sides'] = str(side_path)
    logger.info(f"  → Sides: {side_path.name}")

    # Also extract left-side and right-side separately
    # For distinct extraction, boost the side component
    left_emphasis = mid * 0.3 + (left - right * 0.5)
    right_emphasis = mid * 0.3 + (right - left * 0.5)

    # Normalize
    left_emphasis = left_emphasis / (np.max(np.abs(left_emphasis)) + 1e-10) * 0.9
    right_emphasis = right_emphasis / (np.max(np.abs(right_emphasis)) + 1e-10) * 0.9

    left_path = output_dir / f"{file_stem}_left_emphasis.wav"
    sf.write(str(left_path), left_emphasis, sr)
    results[f'{stem_type}_left'] = str(left_path)

    right_path = output_dir / f"{file_stem}_right_emphasis.wav"
    sf.write(str(right_path), right_emphasis, sr)
    results[f'{stem_type}_right'] = str(right_path)

    return results


def split_enhanced(left: np.ndarray, right: np.ndarray, sr: int,
                   output_dir: Path, file_stem: str, stem_type: str, pan_info: dict) -> dict:
    """
    Enhanced splitting using frequency-aware processing.

    Different frequencies may be panned differently, so we process
    in frequency bands for better separation.
    """
    results = {}

    # If content is mostly mono, just do simple L/R
    if pan_info['side_ratio'] < 0.1:
        logger.info("  Content is mostly mono - using simple L/R split")
        return split_simple_lr(left, right, sr, output_dir, file_stem, stem_type)

    # Calculate spectrograms for frequency-aware processing
    n_fft = 2048
    hop_length = 512

    # Get spectrograms
    left_stft = librosa.stft(left, n_fft=n_fft, hop_length=hop_length)
    right_stft = librosa.stft(right, n_fft=n_fft, hop_length=hop_length)

    # Calculate magnitude and phase
    left_mag = np.abs(left_stft)
    right_mag = np.abs(right_stft)
    left_phase = np.angle(left_stft)
    right_phase = np.angle(right_stft)

    # Create masks based on which channel is dominant at each time-freq bin
    # This helps separate content that's panned left vs right
    total_mag = left_mag + right_mag + 1e-10

    # Soft masks with some spillover for natural sound
    left_dominance = left_mag / total_mag
    right_dominance = right_mag / total_mag

    # Apply soft threshold to emphasize dominant channel
    threshold = 0.55  # Channel must be >55% to be considered dominant

    left_mask = np.where(left_dominance > threshold,
                         left_dominance ** 0.5,  # Soft boost
                         left_dominance * 0.3)   # Soft cut

    right_mask = np.where(right_dominance > threshold,
                          right_dominance ** 0.5,
                          right_dominance * 0.3)

    # Apply masks
    left_extracted = left_stft * left_mask
    right_extracted = right_stft * right_mask

    # Reconstruct time domain
    left_audio = librosa.istft(left_extracted, hop_length=hop_length, length=len(left))
    right_audio = librosa.istft(right_extracted, hop_length=hop_length, length=len(right))

    # Normalize
    left_audio = left_audio / (np.max(np.abs(left_audio)) + 1e-10) * 0.85
    right_audio = right_audio / (np.max(np.abs(right_audio)) + 1e-10) * 0.85

    # Save enhanced splits
    left_path = output_dir / f"{file_stem}_left.wav"
    sf.write(str(left_path), left_audio, sr)
    results[f'{stem_type}_left'] = str(left_path)
    logger.info(f"  → {stem_type.title()} left (left-panned): {left_path.name}")

    right_path = output_dir / f"{file_stem}_right.wav"
    sf.write(str(right_path), right_audio, sr)
    results[f'{stem_type}_right'] = str(right_path)
    logger.info(f"  → {stem_type.title()} right (right-panned): {right_path.name}")

    # Also create center extraction (common to both)
    center_mask = np.minimum(left_mag, right_mag) / (total_mag / 2)
    center_stft = (left_stft + right_stft) / 2 * center_mask
    center_audio = librosa.istft(center_stft, hop_length=hop_length, length=len(left))
    center_audio = center_audio / (np.max(np.abs(center_audio)) + 1e-10) * 0.85

    center_path = output_dir / f"{file_stem}_center.wav"
    sf.write(str(center_path), center_audio, sr)
    results[f'{stem_type}_center'] = str(center_path)
    logger.info(f"  → {stem_type.title()} center: {center_path.name}")

    return results


def check_if_splittable(audio_path: str) -> dict:
    """
    Quick check if an audio file has enough stereo content to be worth splitting.
    """
    if not LIBROSA_AVAILABLE:
        return {'splittable': False, 'reason': 'librosa not available'}

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=False, duration=30)  # Check first 30s

        if y.ndim == 1:
            return {'splittable': False, 'reason': 'mono audio'}

        left, right = y[0], y[1]
        info = analyze_stereo_field(left, right)

        return {
            'splittable': info['distinct_sources'],
            'width': info['width'],
            'side_ratio': info['side_ratio'],
            'recommendation': info['recommendation'],
            'reason': 'sufficient stereo separation' if info['distinct_sources'] else 'content mostly centered'
        }

    except Exception as e:
        return {'splittable': False, 'reason': str(e)}


def check_audio_has_content(audio_path: str, min_rms: float = 0.01) -> bool:
    """Check if an audio file has meaningful content (not silence)."""
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True, duration=60)
        rms = np.sqrt(np.mean(y ** 2))
        has_content = rms > min_rms
        if not has_content:
            logger.info(f"  ⚠️ {Path(audio_path).name} is mostly silent (RMS={rms:.4f})")
        return has_content
    except Exception as e:
        logger.warning(f"Could not check audio content: {e}")
        return True  # Assume it has content if we can't check


def split_all_stems_by_panning(stems_dict: dict, output_dir: str = None,
                                min_side_ratio: float = 0.15, force_stems: list = None) -> dict:
    """
    Check all stems for stereo content and split those with significant panning.
    Only keeps split stems that actually have audio content.

    Args:
        stems_dict: Dictionary of {stem_name: stem_path}
        output_dir: Directory for output files
        min_side_ratio: Minimum side ratio to consider worth splitting (0-1)
        force_stems: List of stem names to always split (e.g., ['guitar', 'drums'] for dual-guitar bands)

    Returns:
        Dictionary with original stems plus any split stems added (empty stems filtered out)
    """
    if not LIBROSA_AVAILABLE:
        logger.warning("librosa not available - skipping stereo splitting")
        return stems_dict

    force_stems = force_stems or []
    results = {}

    for stem_name, stem_path in stems_dict.items():
        # Force split certain stems (like guitar for Grateful Dead)
        force_this = stem_name in force_stems

        # Check if this stem has splittable stereo content
        check = check_if_splittable(stem_path)

        should_split = (
            force_this or
            (check.get('splittable') and check.get('side_ratio', 0) >= min_side_ratio)
        )

        if should_split:
            reason = "forced" if force_this else f"width={check.get('width', 0):.2f}"
            logger.info(f"🎚️ {stem_name} splitting ({reason})")

            split_results = split_stereo(
                input_path=stem_path,
                output_dir=output_dir,
                stem_type=stem_name
            )

            # Only add split stems that have actual content
            for split_name, split_path in split_results.items():
                if check_audio_has_content(split_path, min_rms=0.008):
                    results[split_name] = split_path
                else:
                    logger.info(f"  ❌ Discarding empty split: {split_name}")

            # Keep original only if we didn't get good splits
            if not any(k.startswith(stem_name) for k in results):
                results[stem_name] = stem_path
        else:
            # Keep original stem
            results[stem_name] = stem_path
            logger.info(f"  {stem_name}: mostly centered (side_ratio={check.get('side_ratio', 0):.2f}), keeping as-is")

    return results


# Backward compatibility alias
split_guitar_stereo = split_stereo


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("Stereo Splitter loaded")
    print("Methods: simple, mid_side, enhanced")
    print("Use split_stereo() for any instrument, or split_all_stems_by_panning() for batch processing")
