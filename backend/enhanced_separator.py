"""
Enhanced Audio Separator for StemScribe
Uses audio-separator with state-of-the-art models (BS-Roformer, MDX-Net, etc.)

Features:
- Better vocal separation (SDR 12.9 vs Demucs ~10)
- Lead vs backing vocal splitting
- Multiple model options for different use cases
- Drop-in replacement for demucs-based separation
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Check if audio-separator is available
try:
    from audio_separator.separator import Separator
    AUDIO_SEPARATOR_AVAILABLE = True
    logger.info("✅ audio-separator available")
except ImportError:
    AUDIO_SEPARATOR_AVAILABLE = False
    logger.warning("❌ audio-separator not available - install with: pip install audio-separator")


# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Best models by task (based on 2025/2026 benchmarks)
MODELS = {
    # Vocal/Instrumental separation - BEST QUALITY
    'vocals_best': {
        'model': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'description': 'BS-Roformer - Best vocal separation (SDR 12.9)',
        'stems': ['Vocals', 'Instrumental']
    },

    # Vocal/Instrumental - Good quality, faster
    'vocals_fast': {
        'model': 'UVR-MDX-NET-Voc_FT.onnx',
        'description': 'MDX-NET Vocals - Fast, good quality',
        'stems': ['Vocals', 'Instrumental']
    },

    # Karaoke model - for lead/backing split
    'karaoke': {
        'model': 'UVR_MDXNET_KARA_2.onnx',
        'description': 'MDX-NET Karaoke 2 - Splits lead from backing vocals',
        'stems': ['Vocals', 'Instrumental']  # Vocals=lead, Instrumental=backing
    },

    # Full stem separation (like Demucs)
    'stems_4': {
        'model': 'htdemucs_ft.yaml',
        'description': 'Demucs Fine-tuned - 4 stems (vocals, drums, bass, other)',
        'stems': ['Vocals', 'Drums', 'Bass', 'Other']
    },

    'stems_6': {
        'model': 'htdemucs_6s.yaml',
        'description': 'Demucs 6-stem (vocals, drums, bass, guitar, piano, other)',
        'stems': ['Vocals', 'Drums', 'Bass', 'Guitar', 'Piano', 'Other']
    },

    # Instrumental focus
    'instrumental': {
        'model': 'UVR-MDX-NET-Inst_HQ_3.onnx',
        'description': 'MDX-NET Instrumental HQ - Clean instrumentals',
        'stems': ['Vocals', 'Instrumental']
    },

    # De-reverb (for cleaning up stems)
    'dereverb': {
        'model': 'UVR-DeEcho-DeReverb.pth',
        'description': 'Remove reverb/echo from audio',
        'stems': ['No Reverb', 'Reverb']
    },

    # Denoise
    'denoise': {
        'model': 'UVR-DeNoise.pth',
        'description': 'Remove noise from audio',
        'stems': ['No Noise', 'Noise']
    }
}


class EnhancedSeparator:
    """
    Enhanced audio separator using state-of-the-art models.

    Usage:
        separator = EnhancedSeparator(output_dir='/path/to/output')

        # Basic separation (best quality)
        stems = separator.separate('song.wav')

        # Full 6-stem separation
        stems = separator.separate_full('song.wav')

        # Split lead/backing vocals
        lead, backing = separator.split_vocals('vocals.wav')
    """

    def __init__(self, output_dir: str = None, model_dir: str = None):
        """
        Initialize the enhanced separator.

        Args:
            output_dir: Directory for output files
            model_dir: Directory for model cache (default: /tmp/audio-separator-models)
        """
        if not AUDIO_SEPARATOR_AVAILABLE:
            raise ImportError("audio-separator not installed. Run: pip install audio-separator")

        self.output_dir = Path(output_dir) if output_dir else Path.cwd() / 'output'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_dir = model_dir or '/tmp/audio-separator-models'

        self._separator = None
        self._current_model = None

    def _get_separator(self, model_key: str = 'vocals_best') -> Separator:
        """Get or create separator with specified model."""
        model_config = MODELS.get(model_key)
        if not model_config:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        model_name = model_config['model']

        # Reuse separator if same model
        if self._separator and self._current_model == model_name:
            return self._separator

        logger.info(f"🎵 Loading model: {model_config['description']}")

        self._separator = Separator(
            output_dir=str(self.output_dir),
            model_file_dir=self.model_dir
        )
        self._separator.load_model(model_name)
        self._current_model = model_name

        return self._separator

    def separate(self, audio_path: str, model: str = 'vocals_best') -> Dict[str, str]:
        """
        Separate audio into stems using specified model.

        Args:
            audio_path: Path to input audio file
            model: Model key from MODELS dict

        Returns:
            Dictionary mapping stem names to file paths
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        separator = self._get_separator(model)

        logger.info(f"🎚️ Separating: {audio_path.name}")
        output_files = separator.separate(str(audio_path))

        # Map output files to stem names
        stems = {}
        model_config = MODELS[model]

        for i, stem_name in enumerate(model_config['stems']):
            if i < len(output_files):
                stems[stem_name.lower()] = output_files[i]
                logger.info(f"  ✅ {stem_name}: {Path(output_files[i]).name}")

        return stems

    def separate_full(self, audio_path: str, six_stems: bool = True) -> Dict[str, str]:
        """
        Full stem separation (like Demucs).

        Args:
            audio_path: Path to input audio file
            six_stems: If True, use 6-stem model (with guitar/piano)

        Returns:
            Dictionary mapping stem names to file paths
        """
        model = 'stems_6' if six_stems else 'stems_4'
        return self.separate(audio_path, model)

    def split_lead_backing_vocals(self, vocals_path: str) -> Tuple[str, str]:
        """
        Split a vocals track into lead vocals and backing vocals.
        Uses the two-pass karaoke method.

        Args:
            vocals_path: Path to vocals audio file (already separated)

        Returns:
            Tuple of (lead_vocals_path, backing_vocals_path)
        """
        vocals_path = Path(vocals_path)
        if not vocals_path.exists():
            raise FileNotFoundError(f"Vocals file not found: {vocals_path}")

        logger.info(f"🎤 Splitting lead/backing vocals: {vocals_path.name}")

        # Use karaoke model - it outputs lead vocals and backing
        separator = self._get_separator('karaoke')
        output_files = separator.separate(str(vocals_path))

        if len(output_files) >= 2:
            lead_vocals = output_files[0]  # "Vocals" output = lead
            backing_vocals = output_files[1]  # "Instrumental" output = backing

            logger.info(f"  ✅ Lead vocals: {Path(lead_vocals).name}")
            logger.info(f"  ✅ Backing vocals: {Path(backing_vocals).name}")

            return lead_vocals, backing_vocals
        else:
            raise RuntimeError("Karaoke model did not produce expected outputs")

    def separate_with_vocal_split(self, audio_path: str) -> Dict[str, str]:
        """
        Full separation plus lead/backing vocal split.

        Returns stems dict with:
        - vocals_lead
        - vocals_backing
        - drums
        - bass
        - guitar (if 6-stem)
        - piano (if 6-stem)
        - other
        """
        # First, do full separation
        stems = self.separate_full(audio_path, six_stems=True)

        # Then split vocals into lead/backing
        if 'vocals' in stems:
            try:
                lead, backing = self.split_lead_backing_vocals(stems['vocals'])
                stems['vocals_lead'] = lead
                stems['vocals_backing'] = backing
                # Keep original vocals too
            except Exception as e:
                logger.warning(f"Could not split vocals: {e}")

        return stems

    def enhance_audio(self, audio_path: str, dereverb: bool = True, denoise: bool = True) -> str:
        """
        Enhance audio by removing reverb and/or noise.

        Args:
            audio_path: Path to audio file
            dereverb: Remove reverb/echo
            denoise: Remove noise

        Returns:
            Path to enhanced audio file
        """
        current_path = audio_path

        if dereverb:
            logger.info("🔇 Removing reverb...")
            stems = self.separate(current_path, 'dereverb')
            current_path = stems.get('no reverb', current_path)

        if denoise:
            logger.info("🔇 Removing noise...")
            stems = self.separate(current_path, 'denoise')
            current_path = stems.get('no noise', current_path)

        return current_path


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def separate_audio(audio_path: str, output_dir: str = None,
                   model: str = 'vocals_best') -> Dict[str, str]:
    """
    Quick separation using enhanced separator.

    Args:
        audio_path: Path to audio file
        output_dir: Output directory (default: same as input)
        model: Model to use (default: vocals_best)

    Returns:
        Dictionary of stem paths
    """
    if output_dir is None:
        output_dir = str(Path(audio_path).parent)

    separator = EnhancedSeparator(output_dir=output_dir)
    return separator.separate(audio_path, model)


def separate_full_song(audio_path: str, output_dir: str = None,
                       split_vocals: bool = False) -> Dict[str, str]:
    """
    Full song separation with optional vocal split.

    Args:
        audio_path: Path to audio file
        output_dir: Output directory
        split_vocals: If True, also split lead/backing vocals

    Returns:
        Dictionary of stem paths
    """
    if output_dir is None:
        output_dir = str(Path(audio_path).parent)

    separator = EnhancedSeparator(output_dir=output_dir)

    if split_vocals:
        return separator.separate_with_vocal_split(audio_path)
    else:
        return separator.separate_full(audio_path)


def split_vocals(vocals_path: str, output_dir: str = None) -> Tuple[str, str]:
    """
    Split vocals into lead and backing.

    Args:
        vocals_path: Path to vocals audio
        output_dir: Output directory

    Returns:
        Tuple of (lead_path, backing_path)
    """
    if output_dir is None:
        output_dir = str(Path(vocals_path).parent)

    separator = EnhancedSeparator(output_dir=output_dir)
    return separator.split_lead_backing_vocals(vocals_path)


# ============================================================================
# CLI TEST
# ============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python enhanced_separator.py <audio_file> [output_dir]")
        print("\nAvailable models:")
        for key, config in MODELS.items():
            print(f"  {key}: {config['description']}")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n🎵 Enhanced Separator Test")
    print(f"   Input: {audio_path}")
    print(f"   Output: {output_dir or 'same directory'}")

    stems = separate_full_song(audio_path, output_dir, split_vocals=True)

    print(f"\n✅ Separation complete!")
    for name, path in stems.items():
        print(f"   {name}: {path}")
