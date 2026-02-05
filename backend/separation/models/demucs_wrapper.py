"""
Demucs v4 Model Wrapper
Unified interface for all Demucs model variants
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)


class DemucsWrapper:
    """
    Wrapper for Meta's Demucs v4 audio separation models.

    Supported models:
    - htdemucs: Standard hybrid transformer (4 stems)
    - htdemucs_ft: Fine-tuned version (better quality, slower)
    - htdemucs_6s: 6-stem version (drums, bass, guitar, piano, other, vocals)
    - mdx_extra: MDX-Net variant
    """

    # Model configurations
    MODEL_CONFIGS = {
        'htdemucs': {
            'stems': ['drums', 'bass', 'other', 'vocals'],
            'sample_rate': 44100,
            'memory_gb': 6
        },
        'htdemucs_ft': {
            'stems': ['drums', 'bass', 'other', 'vocals'],
            'sample_rate': 44100,
            'memory_gb': 8
        },
        'htdemucs_6s': {
            'stems': ['drums', 'bass', 'guitar', 'piano', 'other', 'vocals'],
            'sample_rate': 44100,
            'memory_gb': 8
        },
        'mdx_extra': {
            'stems': ['drums', 'bass', 'other', 'vocals'],
            'sample_rate': 44100,
            'memory_gb': 4
        }
    }

    def __init__(self, model_name: str = 'htdemucs_ft', device: str = 'auto'):
        """
        Initialize Demucs wrapper.

        Args:
            model_name: One of htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra
            device: 'auto', 'cuda', 'mps', or 'cpu'
        """
        self.model_name = model_name
        self.device_str = device
        self.model = None
        self.loaded = False

        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")

        self.config = self.MODEL_CONFIGS[model_name]
        self._device = None

        logger.info(f"DemucsWrapper initialized: {model_name}")

    @property
    def device(self):
        """Lazy device detection"""
        if self._device is None:
            import torch

            if self.device_str == 'auto':
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self._device = torch.device('mps')
                elif torch.cuda.is_available():
                    self._device = torch.device('cuda')
                else:
                    self._device = torch.device('cpu')
            else:
                self._device = torch.device(self.device_str)

        return self._device

    @property
    def stems(self) -> List[str]:
        """Get list of stem names this model produces"""
        return self.config['stems']

    @property
    def memory_required_gb(self) -> float:
        """Get memory required for this model"""
        return self.config['memory_gb']

    def load_model(self):
        """Load model to device"""
        if self.loaded:
            return

        try:
            import torch
            from demucs.pretrained import get_model
            from demucs.apply import apply_model

            logger.info(f"Loading {self.model_name} to {self.device}...")

            self.model = get_model(self.model_name)
            self.model.to(self.device)
            self.model.eval()

            # Store apply_model reference
            self._apply_model = apply_model

            self.loaded = True
            logger.info(f"Model {self.model_name} loaded successfully")

        except ImportError as e:
            logger.error(f"Demucs not installed: {e}")
            raise RuntimeError("Please install demucs: pip install demucs")

    def unload_model(self):
        """Free memory by unloading model"""
        if self.model is not None:
            import torch

            self.model.cpu()
            del self.model
            self.model = None
            self.loaded = False

            # Clear cache
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            elif self.device.type == 'mps' and hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()

            import gc
            gc.collect()

            logger.info(f"Model {self.model_name} unloaded")

    def separate(self,
                audio: np.ndarray,
                sample_rate: int = 44100,
                shifts: int = 1,
                overlap: float = 0.25,
                progress_callback=None) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems.

        Args:
            audio: Audio array, shape (channels, samples) or (samples,)
            sample_rate: Audio sample rate
            shifts: Number of random shifts for augmentation (more = better, slower)
            overlap: Overlap between chunks (0-0.5)
            progress_callback: Optional callback for progress updates

        Returns:
            Dict mapping stem names to audio arrays
        """
        import torch
        import torchaudio

        if not self.loaded:
            self.load_model()

        # Ensure correct shape: (batch, channels, samples)
        if audio.ndim == 1:
            audio = np.stack([audio, audio])  # Mono to stereo

        if audio.ndim == 2:
            audio = audio[np.newaxis, ...]  # Add batch dimension

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Resample if needed
        if sample_rate != self.config['sample_rate']:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor,
                sample_rate,
                self.config['sample_rate']
            )

        logger.info(f"Separating with {self.model_name}: {audio_tensor.shape}")

        # Apply model
        with torch.no_grad():
            if shifts > 1:
                # Use shift trick for better quality
                sources = self._apply_model(
                    self.model,
                    audio_tensor[0],  # Remove batch dim for apply_model
                    shifts=shifts,
                    overlap=overlap,
                    progress=progress_callback is not None,
                    device=self.device
                )
            else:
                sources = self._apply_model(
                    self.model,
                    audio_tensor[0],
                    shifts=0,
                    overlap=overlap,
                    device=self.device
                )

        # Convert to numpy and create dict
        sources_np = sources.cpu().numpy()

        result = {}
        for i, stem_name in enumerate(self.stems):
            result[stem_name] = sources_np[i]

        logger.info(f"Separation complete: {list(result.keys())}")
        return result

    async def separate_async(self,
                            audio: np.ndarray,
                            sample_rate: int = 44100,
                            **kwargs) -> Dict[str, np.ndarray]:
        """Async wrapper for separate()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.separate(audio, sample_rate, **kwargs)
        )

    def __repr__(self):
        status = 'loaded' if self.loaded else 'not loaded'
        return f"DemucsWrapper({self.model_name}, {self.device.type}, {status})"


class RoFormerWrapper:
    """
    Placeholder for Band-Split RoFormer model.
    Can be implemented when roformer models are installed.
    """

    def __init__(self, model_name: str = 'bs_roformer', device: str = 'auto'):
        self.model_name = model_name
        self.device_str = device
        self.loaded = False
        logger.warning("RoFormerWrapper is a placeholder - implement when models available")

    @property
    def stems(self) -> List[str]:
        return ['drums', 'bass', 'other', 'vocals']

    @property
    def memory_required_gb(self) -> float:
        return 8.0

    def load_model(self):
        logger.warning("RoFormer models not yet implemented")

    def unload_model(self):
        pass

    def separate(self, audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, np.ndarray]:
        raise NotImplementedError("RoFormer models not yet implemented")
