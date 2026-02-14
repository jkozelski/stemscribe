"""
Guitar Lead/Rhythm Separator for StemScribe

Uses a trained MelBand-RoFormer model to separate a guitar stem into
lead guitar and rhythm guitar. This is a custom model trained on
lead/rhythm guitar pairs using the ZFTurbo/Music-Source-Separation-Training
framework.

Architecture: MelBand-RoFormer (dim=192, depth=8, stereo, 2 stems)
Checkpoint: train_guitar_model/models/last_mel_band_roformer.ckpt
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import soundfile as sf

logger = logging.getLogger(__name__)

# ============================================================================
# PATHS
# ============================================================================

# Model checkpoint and config paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
GUITAR_MODEL_DIR = PROJECT_ROOT / 'train_guitar_model'
GUITAR_CHECKPOINT = GUITAR_MODEL_DIR / 'models' / 'last_mel_band_roformer.ckpt'
# Use the actual training config (not the initial config.yaml which has wrong dims)
GUITAR_CONFIG = GUITAR_MODEL_DIR / 'configs' / 'config_guitar_lead_rhythm.yaml'

# Check availability
GUITAR_SEPARATOR_AVAILABLE = GUITAR_CHECKPOINT.exists() and GUITAR_CONFIG.exists()

if GUITAR_SEPARATOR_AVAILABLE:
    logger.info(f"Guitar lead/rhythm separator available (checkpoint: {GUITAR_CHECKPOINT.stat().st_size / 1e6:.0f}MB)")
else:
    if not GUITAR_CHECKPOINT.exists():
        logger.warning(f"Guitar separator: checkpoint not found at {GUITAR_CHECKPOINT}")
    if not GUITAR_CONFIG.exists():
        logger.warning(f"Guitar separator: config not found at {GUITAR_CONFIG}")


# ============================================================================
# MODEL LOADING (uses ZFTurbo framework classes)
# ============================================================================

_MelBandRoformer = None  # cached class reference


def _import_mel_band_roformer():
    """
    Import the MelBandRoformer class from train_guitar_model/models/bs_roformer/.

    We use importlib to load by file path because there's a namespace conflict:
    backend/models/ (pretrained weights dir) shadows train_guitar_model/models/
    on sys.path. Using importlib avoids this collision entirely.
    """
    global _MelBandRoformer

    if _MelBandRoformer is not None:
        return _MelBandRoformer

    import importlib.util

    # First, load the 'attend' module (dependency of mel_band_roformer)
    attend_path = GUITAR_MODEL_DIR / 'models' / 'bs_roformer' / 'attend.py'
    attend_spec = importlib.util.spec_from_file_location(
        'guitar_models_bs_roformer_attend', str(attend_path)
    )
    attend_module = importlib.util.module_from_spec(attend_spec)
    sys.modules['guitar_models_bs_roformer_attend'] = attend_module
    attend_spec.loader.exec_module(attend_module)

    # Temporarily make 'attend' importable as 'models.bs_roformer.attend'
    # so mel_band_roformer.py's `from models.bs_roformer.attend import Attend` works
    # We save/restore any existing modules to avoid corruption
    saved_modules = {}
    fake_modules = {
        'models': type(sys)('models'),
        'models.bs_roformer': type(sys)('models.bs_roformer'),
        'models.bs_roformer.attend': attend_module,
    }

    for name, mod in fake_modules.items():
        if name in sys.modules:
            saved_modules[name] = sys.modules[name]
        sys.modules[name] = mod

    try:
        # Monkey-patch beartype to be a no-op during import
        # (beartype v0.18+ has stricter type checking that fails with
        # importlib-loaded modules due to different type hint resolution)
        import beartype._decor.decorcache as _bt_cache
        _original_beartype_fn = _bt_cache.beartype

        def _noop_beartype(obj=None, **kwargs):
            """No-op beartype decorator for importlib compatibility."""
            if obj is not None:
                return obj
            return lambda fn: fn

        _bt_cache.beartype = _noop_beartype

        # Also patch the top-level import
        import beartype as _bt_mod
        _bt_mod.beartype = _noop_beartype

        # Now load mel_band_roformer
        mbr_path = GUITAR_MODEL_DIR / 'models' / 'bs_roformer' / 'mel_band_roformer.py'
        mbr_spec = importlib.util.spec_from_file_location(
            'guitar_mel_band_roformer', str(mbr_path)
        )
        mbr_module = importlib.util.module_from_spec(mbr_spec)
        mbr_spec.loader.exec_module(mbr_module)

        # Restore beartype
        _bt_cache.beartype = _original_beartype_fn
        _bt_mod.beartype = _original_beartype_fn

        _MelBandRoformer = mbr_module.MelBandRoformer
        logger.info("Successfully imported MelBandRoformer from train_guitar_model")

    finally:
        # Restore original sys.modules entries
        for name in fake_modules:
            if name in saved_modules:
                sys.modules[name] = saved_modules[name]
            elif name in sys.modules:
                del sys.modules[name]

    return _MelBandRoformer


def _load_model(device: torch.device = None):
    """
    Load the MelBand-RoFormer model from the trained checkpoint.

    We import the model class directly from the train_guitar_model directory
    rather than depending on ZFTurbo's settings.py (which requires ml_collections,
    wandb, etc.). Instead we parse the YAML config ourselves and instantiate
    MelBandRoformer directly.

    Returns:
        Tuple of (model, config_dict)
    """
    import yaml

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else
                              'mps' if torch.backends.mps.is_available() else 'cpu')

    # Load config (FullLoader needed for !!python/tuple tags in ZFTurbo configs)
    with open(GUITAR_CONFIG, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config['model']

    # Convert the multi_stft_resolutions_window_sizes from list to tuple
    # (YAML loads as list, but MelBandRoformer expects tuple)
    if 'multi_stft_resolutions_window_sizes' in model_params:
        model_params['multi_stft_resolutions_window_sizes'] = tuple(
            model_params['multi_stft_resolutions_window_sizes']
        )

    # Import MelBandRoformer using importlib to avoid conflict with backend/models/
    # The train_guitar_model/models/ directory contains the ZFTurbo model code,
    # but backend/models/ (pretrained weights) takes priority on sys.path
    MelBandRoformer = _import_mel_band_roformer()

    # Instantiate model
    logger.info(f"Loading MelBandRoformer (dim={model_params['dim']}, depth={model_params['depth']}, "
                f"stems={model_params['num_stems']}, stereo={model_params['stereo']})")

    model = MelBandRoformer(**model_params)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {GUITAR_CHECKPOINT}")
    checkpoint = torch.load(GUITAR_CHECKPOINT, map_location=device, weights_only=False)

    # ZFTurbo saves as {"model_state_dict": ..., "epoch": ..., ...}
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        epoch = checkpoint.get('epoch', '?')
        best_metric = checkpoint.get('best_metric', '?')
        logger.info(f"Checkpoint: epoch={epoch}, best_metric={best_metric}")
    else:
        # Plain state dict
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"Guitar model loaded on {device} "
                f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)")

    return model, config


# ============================================================================
# INFERENCE
# ============================================================================

class GuitarSeparator:
    """
    Separates a guitar audio stem into lead guitar and rhythm guitar
    using a trained MelBand-RoFormer model.

    Usage:
        separator = GuitarSeparator(output_dir='/path/to/output')
        lead_path, rhythm_path = separator.separate('guitar_stem.wav')
    """

    def __init__(self, output_dir: str = None):
        self._model = None
        self._config = None
        self._device = None
        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_model(self):
        """Lazy-load model on first use."""
        if self._model is None:
            self._device = torch.device(
                'cuda' if torch.cuda.is_available() else
                'mps' if torch.backends.mps.is_available() else 'cpu'
            )
            self._model, self._config = _load_model(self._device)

    def separate(self, audio_path: str) -> Tuple[str, str]:
        """
        Separate a guitar stem into lead and rhythm guitar.

        Args:
            audio_path: Path to guitar audio file (WAV or MP3)

        Returns:
            Tuple of (lead_guitar_path, rhythm_guitar_path) as WAV files
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Guitar audio not found: {audio_path}")

        self._ensure_model()

        logger.info(f"Separating guitar: {audio_path.name}")

        # Load audio
        audio, sr = sf.read(str(audio_path), dtype='float32')

        # Handle mono → stereo conversion if needed
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=-1)

        # Ensure we have (samples, channels) format
        if audio.shape[0] < audio.shape[1]:
            audio = audio.T

        target_sr = self._config['audio']['sample_rate']  # 44100

        # Resample if needed
        if sr != target_sr:
            logger.info(f"Resampling from {sr} to {target_sr}")
            try:
                import librosa
                # librosa expects (channels, samples) for multi-channel
                audio_resampled = np.stack([
                    librosa.resample(audio[:, ch], orig_sr=sr, target_sr=target_sr)
                    for ch in range(audio.shape[1])
                ], axis=-1)
                audio = audio_resampled
                sr = target_sr
            except ImportError:
                logger.warning("librosa not available for resampling, using audio as-is")

        # Convert to tensor: (batch=1, channels=2, time)
        audio_tensor = torch.from_numpy(audio.T).unsqueeze(0).float().to(self._device)

        # Get inference params from config
        inference_config = self._config.get('inference', {})
        chunk_size = self._config['audio'].get('chunk_size', 131584)
        num_overlap = inference_config.get('num_overlap', 4)

        # Run inference with overlap-add for long audio
        with torch.no_grad():
            result = self._separate_with_overlap(
                audio_tensor, chunk_size=chunk_size, num_overlap=num_overlap
            )

        # Model has num_stems=1, target_instrument=lead_guitar
        # Output shape: (1, 2, time) — batch, channels, time (single stem = no stems dim)
        # Lead guitar = model output, Rhythm guitar = original - lead (residual)
        if result.ndim == 4:
            # Multi-stem output: (batch, stems, channels, time)
            lead_audio = result[0, 0].cpu().numpy()
        else:
            # Single-stem output: (batch, channels, time)
            lead_audio = result[0].cpu().numpy()  # (2, time)

        # Compute rhythm as residual: rhythm = original_mix - lead
        rhythm_audio = audio_tensor[0].cpu().numpy() - lead_audio
        rhythm_audio = rhythm_audio[:, :lead_audio.shape[1]]  # ensure same length

        # Save output files
        base_name = audio_path.stem
        lead_path = self.output_dir / f"{base_name}_lead_guitar.wav"
        rhythm_path = self.output_dir / f"{base_name}_rhythm_guitar.wav"

        sf.write(str(lead_path), lead_audio.T, sr)
        sf.write(str(rhythm_path), rhythm_audio.T, sr)

        logger.info(f"  Lead guitar: {lead_path.name}")
        logger.info(f"  Rhythm guitar: {rhythm_path.name}")

        return str(lead_path), str(rhythm_path)

    def _separate_with_overlap(self, audio: torch.Tensor,
                                chunk_size: int = 131584,
                                num_overlap: int = 4) -> torch.Tensor:
        """
        Run model inference with overlap-add for handling long audio files.

        The model processes fixed-size chunks. For audio longer than chunk_size,
        we use overlapping windows and blend them together for seamless output.

        Args:
            audio: Input tensor (batch, channels, time)
            chunk_size: Size of each processing chunk in samples
            num_overlap: Number of overlapping segments (higher = smoother transitions)

        Returns:
            Separated stems tensor (batch, num_stems, channels, time)
        """
        batch, channels, total_length = audio.shape

        # If audio fits in one chunk, process directly
        if total_length <= chunk_size:
            return self._model(audio)

        # Calculate overlap parameters
        hop_size = chunk_size // num_overlap
        num_chunks = (total_length - chunk_size) // hop_size + 1

        # Pad audio to ensure we cover everything
        pad_length = (num_chunks - 1) * hop_size + chunk_size - total_length
        if pad_length > 0:
            audio = torch.nn.functional.pad(audio, (0, pad_length))

        padded_length = audio.shape[2]

        # Create Hann window for crossfade blending
        window = torch.hann_window(chunk_size, device=audio.device)

        # Initialize output accumulator
        # For single-stem model: output is (batch, channels, time)
        # For multi-stem: output is (batch, stems, channels, time)
        # We accumulate as (batch, channels, time) since our model is single-stem
        output = torch.zeros(batch, channels, padded_length, device=audio.device)
        weight_sum = torch.zeros(padded_length, device=audio.device)

        # Process chunks with overlap
        for i in range(num_chunks + 1):
            start = i * hop_size
            end = start + chunk_size

            if end > padded_length:
                # Last chunk — pad if needed
                chunk = audio[:, :, start:]
                remaining = chunk_size - chunk.shape[2]
                if remaining > 0:
                    chunk = torch.nn.functional.pad(chunk, (0, remaining))
            else:
                chunk = audio[:, :, start:end]

            # Run model — returns (batch, channels, time) for single-stem
            chunk_out = self._model(chunk)

            # Handle multi-stem output (squeeze out stems dim if needed)
            if chunk_out.ndim == 4:
                chunk_out = chunk_out[:, 0]  # Take first (only) stem

            # Apply window and accumulate
            actual_len = min(chunk_size, padded_length - start)
            w = window[:actual_len]

            output[:, :, start:start + actual_len] += chunk_out[:, :, :actual_len] * w
            weight_sum[start:start + actual_len] += w

        # Normalize by accumulated weights
        weight_sum = weight_sum.clamp(min=1e-8)
        output = output / weight_sum

        # Trim to original length
        output = output[:, :, :total_length]

        return output


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

# Global singleton for reuse across calls
_guitar_separator: Optional[GuitarSeparator] = None


def separate_guitar(audio_path: str, output_dir: str = None) -> Tuple[str, str]:
    """
    Separate a guitar stem into lead and rhythm guitar.

    Convenience function that uses a cached singleton separator.

    Args:
        audio_path: Path to guitar audio file
        output_dir: Output directory (default: same as input)

    Returns:
        Tuple of (lead_guitar_path, rhythm_guitar_path)
    """
    global _guitar_separator

    if output_dir is None:
        output_dir = str(Path(audio_path).parent)

    if _guitar_separator is None or str(_guitar_separator.output_dir) != output_dir:
        _guitar_separator = GuitarSeparator(output_dir=output_dir)

    return _guitar_separator.separate(audio_path)


def is_available() -> bool:
    """Check if guitar separator model is available."""
    return GUITAR_SEPARATOR_AVAILABLE
