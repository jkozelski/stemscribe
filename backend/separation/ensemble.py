"""
Ensemble Audio Separator
Main orchestrator for multi-model separation with voting and post-processing
"""

import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import json

from .gpu_manager import GPUManager
from .models.demucs_wrapper import DemucsWrapper
from .voting_system import VotingSystem
from .post_processor import PostProcessor
from .quality_metrics import QualityMetrics

logger = logging.getLogger(__name__)


@dataclass
class SeparationResult:
    """Result of ensemble separation"""
    stems: Dict[str, np.ndarray]
    quality_report: Dict
    selection_report: Dict
    processing_time_seconds: float
    models_used: List[str]
    sample_rate: int


class EnsembleSeparator:
    """
    High-quality audio source separation using ensemble of models.

    Features:
    - Multiple model inference (Demucs variants)
    - Intelligent stem voting based on quality metrics
    - Post-processing pipeline (bleed reduction, noise reduction, etc.)
    - GPU/MPS memory management
    - Async processing support
    """

    def __init__(self,
                 models: List[str] = None,
                 device: str = 'auto',
                 enable_voting: bool = True,
                 enable_post_processing: bool = True):
        """
        Initialize ensemble separator.

        Args:
            models: List of model names to use. Default: ['htdemucs_ft', 'htdemucs']
            device: 'auto', 'cuda', 'mps', or 'cpu'
            enable_voting: Whether to use ensemble voting (requires 2+ models)
            enable_post_processing: Whether to apply post-processing
        """
        self.model_names = models or ['htdemucs_ft', 'htdemucs']
        self.device = device
        self.enable_voting = enable_voting and len(self.model_names) > 1
        self.enable_post_processing = enable_post_processing

        # Initialize components
        self.gpu_manager = GPUManager()
        self.voting_system = VotingSystem()
        self.post_processor = PostProcessor()
        self.quality_metrics = QualityMetrics()

        # Model instances (lazy loaded)
        self.models: Dict[str, DemucsWrapper] = {}

        # Progress callback
        self._progress_callback: Optional[Callable] = None

        logger.info(f"EnsembleSeparator initialized with models: {self.model_names}")
        logger.info(f"Device: {self.gpu_manager.device_info}")

    def set_progress_callback(self, callback: Callable[[str, float], None]):
        """Set progress callback: callback(stage_name, progress_0_to_1)"""
        self._progress_callback = callback

    def _report_progress(self, stage: str, progress: float):
        """Report progress to callback if set"""
        if self._progress_callback:
            self._progress_callback(stage, progress)

    def _load_models(self):
        """Lazy load model instances"""
        for model_name in self.model_names:
            if model_name not in self.models:
                self.models[model_name] = DemucsWrapper(
                    model_name=model_name,
                    device=self.gpu_manager.device
                )

    def _unload_all_models(self):
        """Unload all models to free memory"""
        for model in self.models.values():
            model.unload_model()
        self.gpu_manager.clear_memory()

    def separate(self,
                audio: np.ndarray,
                sample_rate: int = 44100,
                post_processing_config: Dict = None) -> SeparationResult:
        """
        Separate audio into stems using ensemble approach.

        Args:
            audio: Audio array, shape (channels, samples) or (samples,)
            sample_rate: Audio sample rate
            post_processing_config: Optional post-processing configuration

        Returns:
            SeparationResult with stems and quality report
        """
        import time
        start_time = time.time()

        self._report_progress('initializing', 0.0)

        # Ensure models are loaded
        self._load_models()

        # Run all models
        model_results = {}
        total_models = len(self.model_names)

        for i, model_name in enumerate(self.model_names):
            self._report_progress(f'separating_{model_name}', i / total_models)
            logger.info(f"Running model {i+1}/{total_models}: {model_name}")

            model = self.models[model_name]

            try:
                stems = model.separate(audio, sample_rate)
                model_results[model_name] = stems
                logger.info(f"  {model_name} complete: {list(stems.keys())}")

            except Exception as e:
                logger.error(f"Model {model_name} failed: {e}")
                continue

        if not model_results:
            raise RuntimeError("All models failed")

        self._report_progress('voting', 0.7)

        # Ensemble voting or single model selection
        if self.enable_voting and len(model_results) > 1:
            logger.info("Running ensemble voting...")
            selected_stems, selection_report = self.voting_system.vote_stems(model_results)
        else:
            # Use first successful model
            model_name = list(model_results.keys())[0]
            selected_stems = model_results[model_name]
            selection_report = {
                'single_model': model_name,
                'selections': {stem: {'selected_model': model_name}
                              for stem in selected_stems.keys()}
            }

        self._report_progress('post_processing', 0.85)

        # Post-processing
        if self.enable_post_processing:
            logger.info("Applying post-processing...")
            config = post_processing_config or {}
            selected_stems = self.post_processor.process_all_stems(
                selected_stems,
                original_mix=audio,
                config=config
            )

        self._report_progress('quality_analysis', 0.95)

        # Generate quality report
        quality_report = self.quality_metrics.generate_quality_report(
            selected_stems,
            model_name='ensemble' if self.enable_voting else list(model_results.keys())[0]
        )

        # Clean up memory
        self._unload_all_models()

        processing_time = time.time() - start_time
        self._report_progress('complete', 1.0)

        logger.info(f"Separation complete in {processing_time:.1f}s")

        return SeparationResult(
            stems=selected_stems,
            quality_report=quality_report,
            selection_report=selection_report,
            processing_time_seconds=processing_time,
            models_used=list(model_results.keys()),
            sample_rate=sample_rate
        )

    async def separate_async(self,
                            audio: np.ndarray,
                            sample_rate: int = 44100,
                            post_processing_config: Dict = None) -> SeparationResult:
        """Async version of separate()"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.separate(audio, sample_rate, post_processing_config)
        )

    def separate_file(self,
                     input_path: str,
                     output_dir: str = None,
                     format: str = 'wav') -> Dict[str, str]:
        """
        Separate audio file and save stems.

        Args:
            input_path: Path to input audio file
            output_dir: Output directory (default: same as input)
            format: Output format (wav, mp3, flac)

        Returns:
            Dict mapping stem names to output file paths
        """
        import soundfile as sf

        input_path = Path(input_path)
        output_dir = Path(output_dir) if output_dir else input_path.parent

        # Load audio
        logger.info(f"Loading {input_path}...")
        audio, sample_rate = sf.read(str(input_path))

        # Transpose if needed (soundfile returns samples x channels)
        if audio.ndim > 1:
            audio = audio.T

        # Run separation
        result = self.separate(audio, sample_rate)

        # Save stems
        output_paths = {}
        stem_dir = output_dir / input_path.stem

        stem_dir.mkdir(parents=True, exist_ok=True)

        for stem_name, stem_audio in result.stems.items():
            output_path = stem_dir / f"{stem_name}.{format}"

            # Transpose back for soundfile
            if stem_audio.ndim > 1:
                stem_audio = stem_audio.T

            sf.write(str(output_path), stem_audio, sample_rate)
            output_paths[stem_name] = str(output_path)
            logger.info(f"Saved {stem_name} to {output_path}")

        # Save quality report
        report_path = stem_dir / 'quality_report.json'
        with open(report_path, 'w') as f:
            json.dump({
                'quality_report': result.quality_report,
                'selection_report': result.selection_report,
                'processing_time_seconds': result.processing_time_seconds,
                'models_used': result.models_used
            }, f, indent=2, default=str)

        logger.info(f"Separation complete. Stems saved to {stem_dir}")

        return output_paths


# Convenience function for quick separation
def separate_audio(audio_path: str,
                  output_dir: str = None,
                  models: List[str] = None,
                  device: str = 'auto') -> Dict[str, str]:
    """
    Convenience function for one-shot audio separation.

    Args:
        audio_path: Path to input audio file
        output_dir: Output directory
        models: Models to use (default: htdemucs_ft, htdemucs)
        device: Device to use

    Returns:
        Dict mapping stem names to output paths
    """
    separator = EnsembleSeparator(models=models, device=device)
    return separator.separate_file(audio_path, output_dir)
