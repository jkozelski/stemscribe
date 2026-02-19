"""
Model Manager for StemScribe
============================
Handles downloading, caching, and loading pretrained models for:
- BS-RoFormer / MelBand-RoFormer (vocals separation)
- OaF Drums (Onsets and Frames drum transcription)
- Guitar tablature models (future)

Models are downloaded automatically on first use and cached locally.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
import urllib.request
import hashlib
import zipfile
import tarfile

logger = logging.getLogger(__name__)

# Default model directory
DEFAULT_MODEL_DIR = Path.home() / '.stemscribe' / 'models'


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODELS = {
    # BS-RoFormer models (handled by audio-separator, but we list for reference)
    'bs_roformer_vocals': {
        'filename': 'model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        'description': 'BS-RoFormer Viperx - Best vocal separation (SDR 12.97)',
        'type': 'audio-separator',  # Managed by audio-separator library
        'task': 'vocals'
    },

    'melband_roformer_vocals': {
        'filename': 'vocals_mel_band_roformer.ckpt',
        'description': 'MelBand RoFormer - Excellent vocals (SDR 12.6)',
        'type': 'audio-separator',
        'task': 'vocals'
    },

    # Demucs models (also via audio-separator)
    'htdemucs_6s': {
        'filename': 'htdemucs_6s.yaml',
        'description': 'Demucs 6-stem (vocals, drums, bass, guitar, piano, other)',
        'type': 'audio-separator',
        'task': 'stems'
    },

    # OaF Drums - Onsets and Frames
    'oaf_drums': {
        'filename': 'drum_checkpoint',
        'description': 'Onsets and Frames Drums - Trained on E-GMD (444 hours)',
        'type': 'magenta',
        'task': 'drums',
        'url': 'https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/drum_checkpoint.zip',
        'size_mb': 150
    },

    # Basic pitch for melody/vocal transcription
    'basic_pitch': {
        'filename': 'basic_pitch_model',
        'description': 'Spotify Basic Pitch - polyphonic pitch detection',
        'type': 'tensorflow',
        'task': 'pitch',
        'pip_package': 'basic-pitch'
    },

    # MT3 for multi-instrument transcription (optional, large)
    'mt3': {
        'filename': 'mt3_checkpoint',
        'description': 'MT3 - Multi-instrument transcription (Google)',
        'type': 'tensorflow',
        'task': 'transcription',
        'url': 'gs://mt3/checkpoints/mt3/',
        'size_mb': 400,
        'optional': True
    },

    # Guitar tab transcription model (CRNN, trained on GuitarSet)
    'guitar_tab': {
        'filename': 'pretrained/best_tab_model.pt',
        'description': 'Guitar Tab CRNN - 6x20 string/fret prediction (GuitarSet, val_loss 0.0498)',
        'type': 'pytorch',
        'task': 'tab',
    },

    # Drum transcription model (CRNN, trained on E-GMD)
    'drum_nn': {
        'filename': 'pretrained/best_drum_model.pt',
        'description': 'Drum CRNN - 8-class onset/frame/velocity (E-GMD, val_loss 0.0335)',
        'type': 'pytorch',
        'task': 'drums',
    },

    # Bass transcription model (CRNN, trained on Slakh2100)
    'bass_nn': {
        'filename': 'pretrained/best_bass_model.pt',
        'description': 'Bass CRNN - 4-string × 24-fret onset/frame/velocity (Slakh2100)',
        'type': 'pytorch',
        'task': 'bass',
    },

    # Piano transcription model (CRNN, trained on MAESTRO v3)
    'piano_nn': {
        'filename': 'pretrained/best_piano_model.pt',
        'description': 'Piano CRNN - 88-key onset/frame/velocity (MAESTRO v3.0.0)',
        'type': 'pytorch',
        'task': 'piano',
    },
}


class ModelManager:
    """
    Manages downloading, caching, and loading of ML models for StemScribe.
    """

    def __init__(self, model_dir: Optional[str] = None):
        """
        Initialize the model manager.

        Args:
            model_dir: Directory for model cache (default: ~/.stemscribe/models)
        """
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self._loaded_models: Dict[str, Any] = {}

        logger.info(f"📦 Model manager initialized: {self.model_dir}")

    def list_models(self) -> Dict[str, dict]:
        """List all available models with their status."""
        status = {}
        for name, info in MODELS.items():
            model_path = self.model_dir / info['filename']
            status[name] = {
                **info,
                'downloaded': self._is_downloaded(name),
                'path': str(model_path) if model_path.exists() else None
            }
        return status

    def _is_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        if model_name not in MODELS:
            return False

        info = MODELS[model_name]
        model_path = self.model_dir / info['filename']

        # For audio-separator models, they're auto-downloaded
        if info['type'] == 'audio-separator':
            # Check in audio-separator's cache
            audio_sep_dir = Path('/tmp/audio-separator-models')
            return (audio_sep_dir / info['filename']).exists()

        return model_path.exists() or (model_path.with_suffix('.zip')).exists()

    def get_model_path(self, model_name: str, download: bool = True) -> Optional[Path]:
        """
        Get path to a model, downloading if necessary.

        Args:
            model_name: Name of the model
            download: Whether to download if not present

        Returns:
            Path to model directory/file, or None if not available
        """
        if model_name not in MODELS:
            logger.error(f"Unknown model: {model_name}")
            return None

        info = MODELS[model_name]
        model_path = self.model_dir / info['filename']

        # Audio-separator handles its own downloads
        if info['type'] == 'audio-separator':
            logger.info(f"Model {model_name} is managed by audio-separator (auto-downloads)")
            return Path('/tmp/audio-separator-models') / info['filename']

        # Check if already downloaded
        if model_path.exists():
            return model_path

        # Download if requested
        if download and 'url' in info:
            if self._download_model(model_name):
                return model_path

        return None

    def _download_model(self, model_name: str) -> bool:
        """Download a model from its URL."""
        if model_name not in MODELS:
            return False

        info = MODELS[model_name]

        if 'url' not in info:
            logger.warning(f"No URL for model: {model_name}")
            return False

        url = info['url']
        filename = info['filename']
        size_mb = info.get('size_mb', 'unknown')

        logger.info(f"📥 Downloading {model_name} ({size_mb}MB)...")
        logger.info(f"   URL: {url}")

        try:
            # Determine file type
            if url.endswith('.zip'):
                archive_path = self.model_dir / f"{filename}.zip"
                extract_path = self.model_dir / filename
            elif url.endswith('.tar.gz') or url.endswith('.tgz'):
                archive_path = self.model_dir / f"{filename}.tar.gz"
                extract_path = self.model_dir / filename
            else:
                # Direct file download
                archive_path = self.model_dir / filename
                extract_path = None

            # Download with progress
            self._download_file(url, archive_path)

            # Extract if needed
            if extract_path:
                logger.info(f"   Extracting to {extract_path}...")
                self._extract_archive(archive_path, extract_path)
                # Optionally remove archive
                # archive_path.unlink()

            logger.info(f"✅ Model {model_name} downloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to download {model_name}: {e}")
            return False

    def _download_file(self, url: str, output_path: Path):
        """Download a file with progress reporting."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        def progress_hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(100, count * block_size * 100 / total_size)
                if count % 100 == 0:
                    logger.debug(f"   Download progress: {percent:.1f}%")

        urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)

    def _extract_archive(self, archive_path: Path, extract_path: Path):
        """Extract a zip or tar archive."""
        extract_path.mkdir(parents=True, exist_ok=True)

        if str(archive_path).endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_path)
        elif str(archive_path).endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(extract_path)
        else:
            raise ValueError(f"Unknown archive format: {archive_path}")

    def ensure_models_for_task(self, task: str) -> bool:
        """
        Ensure all required models for a task are available.

        Args:
            task: Task name ('vocals', 'drums', 'stems', 'transcription')

        Returns:
            True if all required models are available
        """
        required = [name for name, info in MODELS.items()
                   if info['task'] == task and not info.get('optional', False)]

        all_available = True
        for model_name in required:
            path = self.get_model_path(model_name, download=True)
            if path is None:
                logger.warning(f"Model not available: {model_name}")
                all_available = False

        return all_available


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

_manager: Optional[ModelManager] = None


def get_model_manager() -> ModelManager:
    """Get the singleton model manager instance."""
    global _manager
    if _manager is None:
        _manager = ModelManager()
    return _manager


def ensure_model(model_name: str) -> Optional[Path]:
    """Ensure a model is downloaded and return its path."""
    return get_model_manager().get_model_path(model_name, download=True)


def list_available_models() -> Dict[str, dict]:
    """List all available models with their status."""
    return get_model_manager().list_models()


# ============================================================================
# CLI
# ============================================================================

if __name__ == '__main__':
    import sys

    logging.basicConfig(level=logging.INFO)

    manager = ModelManager()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == 'list':
            print("\n📦 Available Models:\n")
            for name, info in manager.list_models().items():
                status = "✅" if info['downloaded'] else "❌"
                print(f"  {status} {name}")
                print(f"     {info['description']}")
                print(f"     Type: {info['type']}, Task: {info['task']}")
                print()

        elif command == 'download':
            if len(sys.argv) > 2:
                model_name = sys.argv[2]
                path = manager.get_model_path(model_name, download=True)
                if path:
                    print(f"✅ Model available at: {path}")
                else:
                    print(f"❌ Failed to get model: {model_name}")
            else:
                print("Usage: python model_manager.py download <model_name>")

        else:
            print(f"Unknown command: {command}")
            print("Commands: list, download <model_name>")
    else:
        print("Usage: python model_manager.py <command>")
        print("Commands: list, download <model_name>")
