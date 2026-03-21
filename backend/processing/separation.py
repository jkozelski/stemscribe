"""
Stem separation functions — all separate_stems_* variants.

Owns the GPU semaphore, active runner tracking, and shutdown handler.
Includes Modal cloud GPU support (separate_stems_modal) as an alternative to local GPU.
"""

import os
import gc
import json
import subprocess
import threading
import logging
import time
from pathlib import Path


from models.job import ProcessingJob, OUTPUT_DIR, save_job_checkpoint
from processing.utils import convert_wavs_to_mp3

# ============ MODAL CLOUD GPU ============

# Read MODAL_ENABLED from environment — checked at runtime, not import time
def is_modal_enabled():
    return os.environ.get('MODAL_ENABLED', 'false').lower() in ('true', '1', 'yes')

MODAL_ENABLED = is_modal_enabled()  # for backward compat at import time

try:
    import modal
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False


def _flush_mps_memory():
    """Force-flush MPS (Apple Silicon GPU) memory pool to prevent silent OOM hangs."""
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            if hasattr(torch.mps, 'empty_cache'):
                torch.mps.empty_cache()
            if hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            logger.info("MPS memory cache flushed")
    except Exception as e:
        logger.debug(f"MPS memory flush skipped: {e}")

logger = logging.getLogger(__name__)

# ============ GPU QUEUING ============

# Processing queue: only one separation job at a time (GPU memory constraint)
_separation_semaphore = threading.Semaphore(1)
_active_runners = []  # Track active DemucsRunner instances for graceful shutdown
_active_runners_lock = threading.Lock()

# ============ CONDITIONAL IMPORTS ============

# DemucsRunner + EnhancedSeparator + EnsembleSeparator are imported lazily
# to match app.py's try/except pattern.

try:
    from demucs_runner import DemucsRunner, DemucsProgress
    DEMUCS_RUNNER_AVAILABLE = True
except ImportError:
    DEMUCS_RUNNER_AVAILABLE = False

try:
    from enhanced_separator import EnhancedSeparator, AUDIO_SEPARATOR_AVAILABLE
    ENHANCED_SEPARATOR_AVAILABLE = AUDIO_SEPARATOR_AVAILABLE
except ImportError:
    ENHANCED_SEPARATOR_AVAILABLE = False

try:
    from separation import EnsembleSeparator
    ENSEMBLE_SEPARATOR_AVAILABLE = True
except ImportError:
    ENSEMBLE_SEPARATOR_AVAILABLE = False

try:
    from stereo_splitter import split_stereo
    STEREO_SPLITTER_AVAILABLE = True
except ImportError:
    STEREO_SPLITTER_AVAILABLE = False


def shutdown_active_runners():
    """Cancel all active Demucs processes on server shutdown"""
    with _active_runners_lock:
        for runner in _active_runners:
            try:
                runner.cancel()
            except Exception:
                pass
    logger.info("Shutdown: cancelled all active separation processes")


# ============ SEPARATION FUNCTIONS ============

def separate_stems(job: ProcessingJob, audio_path: Path):
    """
    Demucs stem separation with queueing, progress streaming, MPS fallback,
    and graceful shutdown support.

    Features:
    1. Semaphore queue -- only one separation runs at a time (GPU memory)
    2. DemucsRunner with real-time progress parsing from stderr
    3. MPS -> CPU automatic fallback on Apple Silicon memory errors
    4. Runner tracked for graceful cancellation on server shutdown
    5. Checkpoint saves after each stage transition
    """
    # Wait for GPU availability (only one separation at a time)
    if not _separation_semaphore.acquire(blocking=False):
        job.stage = 'Queued — waiting for GPU'
        job.progress = 10
        save_job_checkpoint(job)
        logger.info(f"Job {job.job_id} queued, waiting for GPU semaphore")
        _separation_semaphore.acquire()  # Block until available

    try:
        job.stage = 'Separating stems with AI (6-stem model)'
        job.progress = 15
        save_job_checkpoint(job)
        logger.info(f"Starting stem separation for {audio_path}")

        output_path = OUTPUT_DIR / job.job_id / 'stems'
        output_path.mkdir(parents=True, exist_ok=True)

        # Progress callback to update job status
        def progress_callback(progress: DemucsProgress):
            job_progress = 15 + (progress.percent * 0.25)
            job.progress = int(job_progress)
            if progress.eta_seconds > 0:
                eta_min = progress.eta_seconds / 60
                job.stage = f'Separating stems: {progress.percent:.0f}% (ETA: {eta_min:.1f}m)'
            else:
                job.stage = f'Separating stems: {progress.percent:.0f}%'

        # Try MPS first, fall back to CPU on memory error
        for device in ('auto', 'cpu'):
            runner = DemucsRunner(
                model='htdemucs_6s',
                device=device if device != 'auto' else 'auto',
                progress_callback=progress_callback
            )

            # Track runner for graceful shutdown
            with _active_runners_lock:
                _active_runners.append(runner)

            try:
                result = runner.separate(
                    audio_path=audio_path,
                    output_dir=output_path,
                    timeout_seconds=1800
                )
            finally:
                with _active_runners_lock:
                    if runner in _active_runners:
                        _active_runners.remove(runner)

            if result.success:
                break

            # Retry on MPS memory error only
            if device == 'auto' and result.error_message and 'MPS' in result.error_message:
                logger.warning(f"MPS failed ({result.error_message}), retrying on CPU")
                job.stage = 'Retrying separation on CPU (MPS memory issue)'
                job.progress = 15
                save_job_checkpoint(job)
                # Clean partial output before retry
                import shutil as _shutil
                for child in output_path.iterdir():
                    if child.is_dir():
                        _shutil.rmtree(child, ignore_errors=True)
                continue
            else:
                raise Exception(result.error_message)

        if not result.success:
            raise Exception(result.error_message)

        # Flush GPU memory after separation completes
        _flush_mps_memory()
        gc.collect()

        job.progress = 40

        # Convert WAV to MP3 using ffmpeg
        if result.output_dir and result.output_dir.exists():
            job.stage = 'Converting stems to MP3'
            job.stems = convert_wavs_to_mp3(result.output_dir)
            for stem_name in job.stems:
                logger.info(f"Found stem: {stem_name}")

        if not job.stems:
            raise Exception("No stems were generated")

        save_job_checkpoint(job)
        logger.info(f"Stem separation complete: {len(job.stems)} stems in {result.processing_time_seconds:.1f}s")
        return True

    except Exception as e:
        logger.error(f"Stem separation failed: {e}")
        job.error = str(e)
        return False
    finally:
        _separation_semaphore.release()


def separate_stems_roformer(job: ProcessingJob, audio_path: Path):
    """
    STATE-OF-THE-ART stem separation using BS-RoFormer + Demucs ensemble.

    Pipeline:
    1. BS-RoFormer (SDR 12.9) → best-in-class vocals/instrumental split
    2. Demucs htdemucs_6s on instrumental → drums, bass, guitar, piano, other
    3. smart_separate() handles deep extraction of "other" later

    This beats pure Demucs by ~3 dB SDR on vocals and produces cleaner
    instrument stems by keeping vocal bleed out of the Demucs pass.
    """

    # Wait for GPU availability (only one separation at a time)
    if not _separation_semaphore.acquire(blocking=False):
        job.stage = 'Queued — waiting for GPU'
        job.progress = 10
        save_job_checkpoint(job)
        logger.info(f"Job {job.job_id} queued, waiting for GPU semaphore")
        _separation_semaphore.acquire()

    _sem_released = False  # Track if semaphore was released early (for fallback path)
    try:
        job.stage = 'RoFormer: Separating vocals (state-of-the-art)'
        job.progress = 10
        save_job_checkpoint(job)
        logger.info(f"Starting RoFormer+Demucs ensemble separation for {audio_path}")

        output_path = OUTPUT_DIR / job.job_id / 'stems'
        output_path.mkdir(parents=True, exist_ok=True)

        # ---- Pass 1: BS-RoFormer for vocals/instrumental ----
        roformer_dir = output_path / 'roformer'
        roformer_dir.mkdir(parents=True, exist_ok=True)

        job.stage = '🧠 RoFormer: Extracting vocals (SDR 12.9)'
        job.progress = 12
        save_job_checkpoint(job)

        # Flush GPU memory before loading RoFormer
        _flush_mps_memory()

        separator = EnhancedSeparator(output_dir=str(roformer_dir))
        roformer_stems = separator.separate(str(audio_path), model='vocals_best')

        vocals_path = roformer_stems.get('vocals')
        instrumental_path = roformer_stems.get('instrumental')

        if not vocals_path or not instrumental_path:
            raise Exception("BS-RoFormer did not produce vocals + instrumental")

        # Ensure absolute paths
        if not os.path.isabs(vocals_path):
            vocals_path = str(roformer_dir / Path(vocals_path).name)
        if not os.path.isabs(instrumental_path):
            instrumental_path = str(roformer_dir / Path(instrumental_path).name)

        logger.info(f"  ✅ RoFormer vocals: {Path(vocals_path).name}")
        logger.info(f"  ✅ RoFormer instrumental: {Path(instrumental_path).name}")

        # Free RoFormer memory before loading Demucs
        del separator
        gc.collect()
        _flush_mps_memory()

        job.progress = 22
        save_job_checkpoint(job)

        # ---- Pass 2: Demucs on instrumental for instrument stems ----
        job.stage = '🎸 Separating instruments from instrumental...'
        logger.info("🎸 Running Demucs htdemucs_6s on instrumental track...")

        demucs_dir = output_path / 'demucs_inst'
        demucs_dir.mkdir(parents=True, exist_ok=True)

        def progress_callback(progress: DemucsProgress):
            # Map Demucs progress to job progress 22-40%
            job_progress = 22 + (progress.percent * 0.18)
            job.progress = int(job_progress)
            if progress.eta_seconds > 0:
                eta_min = progress.eta_seconds / 60
                job.stage = f'🎸 Separating instruments: {progress.percent:.0f}% (ETA: {eta_min:.1f}m)'
            else:
                job.stage = f'🎸 Separating instruments: {progress.percent:.0f}%'

        runner = DemucsRunner(model='htdemucs_6s', progress_callback=progress_callback)

        # Track runner for graceful shutdown
        with _active_runners_lock:
            _active_runners.append(runner)
        try:
            result = runner.separate(
                audio_path=Path(instrumental_path),
                output_dir=demucs_dir,
                timeout_seconds=1800
            )
        finally:
            with _active_runners_lock:
                if runner in _active_runners:
                    _active_runners.remove(runner)

        if not result.success:
            raise Exception(f"Demucs instrument separation failed: {result.error_message}")

        # Flush GPU memory after Demucs pass completes
        _flush_mps_memory()
        gc.collect()

        job.progress = 40
        job.stage = 'Converting vocals to MP3'
        save_job_checkpoint(job)

        # ---- Collect all stems ----
        # Convert RoFormer vocals WAV to MP3
        vocals_mp3 = output_path / 'vocals.mp3'
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', vocals_path,
                '-codec:a', 'libmp3lame', '-b:a', '320k',
                str(vocals_mp3)
            ], capture_output=True, timeout=180)
        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg vocals conversion timed out after 180s")

        if vocals_mp3.exists():
            job.stems['vocals'] = str(vocals_mp3)
            logger.info("  ✅ vocals (RoFormer SDR 12.9)")
        else:
            # Fallback: use the WAV directly
            job.stems['vocals'] = vocals_path
            logger.warning("  ⚠️ MP3 conversion failed for vocals, using WAV")

        job.progress = 41
        job.stage = 'Converting instrument stems to MP3'
        save_job_checkpoint(job)

        # Convert Demucs instrument stems WAV to MP3
        if result.output_dir and result.output_dir.exists():
            demucs_mp3s = convert_wavs_to_mp3(result.output_dir)

            # Take drums, bass, guitar, piano, other from Demucs
            # Skip Demucs "vocals" — RoFormer's is better
            for stem_name, stem_path in demucs_mp3s.items():
                if stem_name == 'vocals':
                    logger.info("  ⏭️ Skipping Demucs vocals (using RoFormer instead)")
                    continue
                job.stems[stem_name] = stem_path
                logger.info(f"  ✅ {stem_name} (Demucs)")

        job.progress = 42
        save_job_checkpoint(job)

        if len(job.stems) < 2:
            raise Exception("Ensemble separation produced too few stems")

        logger.info(f"RoFormer+Demucs ensemble complete: {len(job.stems)} stems")
        logger.info("   Vocals: BS-RoFormer (SDR 12.9) | Instruments: Demucs htdemucs_6s")
        save_job_checkpoint(job)
        return True

    except Exception as e:
        logger.error(f"RoFormer ensemble separation failed: {e}")
        import traceback
        traceback.print_exc()

        # Fallback to pure Demucs if RoFormer fails
        # Release semaphore first -- separate_stems() acquires its own
        _separation_semaphore.release()
        _sem_released = True
        logger.info("Falling back to standard Demucs separation...")
        job.stems = {}
        job.error = None
        return separate_stems(job, audio_path)
    finally:
        if not _sem_released:
            _separation_semaphore.release()


def separate_stems_mdx(job: ProcessingJob, audio_path: Path, stereo_split_guitar: bool = False):
    """
    HYBRID stem separation - uses the BEST model for each instrument type.

    OPTIMIZED VERSION: Single separator instance, proper memory cleanup.

    Strategy:
    1. MDX23C-InstVoc: Clean vocals/instrumental split (best for vocals)
    2. htdemucs_6s: ALL instruments from instrumental (drums, bass, guitar, piano, other)
    3. Optional: Stereo split guitar for dual-guitar bands (Allman Brothers, etc.)

    Note: We use htdemucs piano as-is since running multiple models was causing
    overheating. The piano quality is acceptable for most use cases.
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        logger.error("audio-separator not installed. Install with: pip install audio-separator")
        return False


    # Wait for GPU availability (only one separation at a time)
    if not _separation_semaphore.acquire(blocking=False):
        job.stage = 'Queued — waiting for GPU'
        job.progress = 5
        save_job_checkpoint(job)
        logger.info(f"Job {job.job_id} queued (MDX), waiting for GPU semaphore")
        _separation_semaphore.acquire()

    try:
        job.stage = 'HYBRID: Starting multi-model separation'
        job.progress = 10
        logger.info(f"Starting HYBRID stem separation for {audio_path}")
        logger.info("   Strategy: MDX23C (vocals) -> htdemucs_6s (all instruments)")

        output_path = OUTPUT_DIR / job.job_id / 'stems' / 'hybrid'
        output_path.mkdir(parents=True, exist_ok=True)
        # Use a single separator instance to reduce memory usage
        separator = Separator(
            output_dir=str(output_path),
            output_format='mp3',
        )

        # ================================================================
        # STEP 1: MDX23C - Clean vocals/instrumental split
        # ================================================================
        job.stage = 'HYBRID Step 1/2: Extracting clean vocals'
        job.progress = 15

        separator.load_model(model_filename='MDX23C-8KFFT-InstVoc_HQ.ckpt')
        output_files = separator.separate(str(audio_path))
        logger.info(f"Step 1 - MDX23C output: {output_files}")

        # Find vocals and instrumental
        # Note: output_files may be just filenames or full paths depending on audio-separator version
        instrumental_path = None
        for f in output_files:
            # Ensure we have full path
            if not Path(f).is_absolute():
                f = str(output_path / f)

            if 'instrumental' in f.lower() or 'instrum' in f.lower():
                instrumental_path = f
            elif 'vocal' in f.lower():
                job.stems['vocals'] = f
                logger.info(f"  🎤 VOCALS: {Path(f).name}")

        if not instrumental_path:
            raise Exception("MDX23C did not produce instrumental track")

        # Clean up memory before loading next model
        _flush_mps_memory()
        gc.collect()

        # ================================================================
        # STEP 2: htdemucs_6s - Get ALL instruments from instrumental
        # ================================================================
        job.stage = 'HYBRID Step 2/2: Separating instruments'
        job.progress = 40

        separator.load_model(model_filename='htdemucs_6s.yaml')
        inst_outputs = separator.separate(instrumental_path)
        logger.info(f"Step 2 - htdemucs_6s output: {inst_outputs}")

        for f in inst_outputs:
            # Ensure we have full path
            if not Path(f).is_absolute():
                f = str(output_path / f)

            fname = Path(f).stem.lower()
            if 'drum' in fname:
                job.stems['drums'] = f
                logger.info(f"  🥁 DRUMS: {Path(f).name}")
            elif 'bass' in fname:
                job.stems['bass'] = f
                logger.info(f"  🎸 BASS: {Path(f).name}")
            elif 'guitar' in fname:
                job.stems['guitar'] = f
                logger.info(f"  🎸 GUITAR: {Path(f).name}")
            elif 'piano' in fname:
                job.stems['piano'] = f
                logger.info(f"  🎹 PIANO: {Path(f).name}")
            elif 'other' in fname:
                job.stems['other'] = f
                logger.info(f"  🎵 OTHER: {Path(f).name}")

        # Clean up separator to free memory
        del separator
        _flush_mps_memory()
        gc.collect()

        # ================================================================
        # OPTIONAL: Stereo split guitar for dual-guitar bands
        # ================================================================
        if stereo_split_guitar and 'guitar' in job.stems and STEREO_SPLITTER_AVAILABLE:
            job.stage = 'Stereo splitting guitar'
            job.progress = 70
            logger.info("Stereo splitting guitar for dual-guitar separation...")

            try:
                guitar_path = Path(job.stems['guitar'])
                split_output = OUTPUT_DIR / job.job_id / 'stems' / 'guitar_split'
                split_output.mkdir(parents=True, exist_ok=True)

                left, right, center = split_stereo(guitar_path, split_output)

                if left and right:
                    job.stems['guitar_left'] = str(left)
                    job.stems['guitar_right'] = str(right)
                    if center:
                        job.stems['guitar_center'] = str(center)
                    logger.info(f"  🎸 GUITAR LEFT: {Path(left).name}")
                    logger.info(f"  🎸 GUITAR RIGHT: {Path(right).name}")
            except Exception as e:
                logger.warning(f"Guitar stereo split failed: {e}")

        job.progress = 80

        if len(job.stems) < 3:
            raise Exception("Not enough stems were generated")

        logger.info("✅ HYBRID separation complete!")
        logger.info(f"   Final stems: {list(job.stems.keys())}")
        return True

    except Exception as e:
        logger.error(f"HYBRID separation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        job.error = str(e)
        return False
    finally:
        _separation_semaphore.release()


def separate_stems_ensemble(job: ProcessingJob, audio_path: Path):
    """
    ENSEMBLE separation - Moises.ai-quality multi-model approach.

    Strategy:
    1. Run multiple Demucs models (htdemucs_ft, htdemucs, htdemucs_6s)
    2. Vote on best stem for each instrument using quality metrics
    3. Apply post-processing (bleed reduction, noise removal, phase alignment)

    This produces the highest quality separation but takes longer (2-3x).
    Optimized for M3 Max with MPS acceleration.
    """
    if not ENSEMBLE_SEPARATOR_AVAILABLE:
        logger.warning("Ensemble separator not available, falling back to standard separation")
        return separate_stems(job, audio_path)

    import soundfile as sf

    # Wait for GPU availability (only one separation at a time)
    if not _separation_semaphore.acquire(blocking=False):
        job.stage = 'Queued — waiting for GPU'
        job.progress = 3
        save_job_checkpoint(job)
        logger.info(f"Job {job.job_id} queued (ensemble), waiting for GPU semaphore")
        _separation_semaphore.acquire()

    try:
        job.stage = 'Loading audio for ensemble separation'
        job.progress = 5
        logger.info("Starting ENSEMBLE separation (Moises-quality)")

        # Load audio
        audio, sample_rate = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.T  # Transpose to (channels, samples)

        # Create output directory
        output_path = OUTPUT_DIR / job.job_id / 'stems'
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize ensemble separator with progress callback
        def progress_callback(stage: str, progress: float):
            stage_map = {
                'initializing': ('Loading models', 10),
                'separating_htdemucs_ft': ('Separating with htdemucs_ft', 25),
                'separating_htdemucs': ('Separating with htdemucs', 45),
                'voting': ('Voting on best stems', 70),
                'post_processing': ('Post-processing stems', 85),
                'quality_analysis': ('Analyzing quality', 95),
                'complete': ('Saving stems', 98)
            }
            if stage in stage_map:
                job.stage = stage_map[stage][0]
                job.progress = stage_map[stage][1]
            logger.info(f"  Ensemble progress: {stage} ({progress*100:.0f}%)")

        # Use htdemucs_ft + htdemucs for best balance of quality vs speed
        # htdemucs_6s is great but produces different stems (6 vs 4)
        separator = EnsembleSeparator(
            models=['htdemucs_ft', 'htdemucs'],
            device='auto',
            enable_voting=True,
            enable_post_processing=True
        )
        separator.set_progress_callback(progress_callback)

        # Configure post-processing
        post_config = {
            'bleed_reduction': {'enabled': True, 'aggressiveness': 0.4},
            'noise_reduction': {'enabled': True, 'threshold_db': -45},
            'artifact_removal': True,
            'loudness_normalization': {'enabled': True, 'target_lufs': -14.0},
            'phase_alignment': True
        }

        # Run ensemble separation
        result = separator.separate(audio, sample_rate, post_config)

        # Flush GPU memory after ensemble separation
        _flush_mps_memory()
        gc.collect()

        job.stage = 'Saving ensemble stems'
        job.progress = 95

        # Save stems to files
        for stem_name, stem_audio in result.stems.items():
            stem_file = output_path / f'{stem_name}.mp3'

            # Convert to format soundfile can write
            if stem_audio.ndim > 1:
                stem_audio_write = stem_audio.T  # Back to (samples, channels)
            else:
                stem_audio_write = stem_audio

            # Write as WAV first, then convert to MP3
            wav_file = output_path / f'{stem_name}.wav'
            sf.write(str(wav_file), stem_audio_write, sample_rate)

            # Convert to MP3 using ffmpeg (with timeout to prevent stalls — 300s per stem)
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', str(wav_file),
                    '-codec:a', 'libmp3lame', '-q:a', '2',
                    str(stem_file)
                ], capture_output=True, timeout=300)
            except subprocess.TimeoutExpired:
                logger.warning(f"ffmpeg conversion timed out for {stem_name} after 300s")

            # Remove temp WAV
            if stem_file.exists():
                wav_file.unlink()
                job.stems[stem_name] = str(stem_file)
                logger.info(f"  Saved ensemble stem: {stem_name}")
            else:
                # Keep WAV if MP3 conversion failed
                job.stems[stem_name] = str(wav_file)
                logger.warning(f"  Kept WAV (MP3 conversion failed): {stem_name}")

        # Save quality report
        quality_report_path = output_path / 'ensemble_quality_report.json'
        with open(quality_report_path, 'w') as f:
            json.dump({
                'quality_report': result.quality_report,
                'selection_report': result.selection_report,
                'processing_time_seconds': result.processing_time_seconds,
                'models_used': result.models_used
            }, f, indent=2, default=str)

        job.progress = 100
        logger.info(f"✅ Ensemble separation complete: {len(job.stems)} stems in {result.processing_time_seconds:.1f}s")
        logger.info(f"   Models used: {result.models_used}")
        logger.info(f"   Overall quality: {result.quality_report.get('overall_quality', 'N/A'):.3f}")

        return True

    except Exception as e:
        logger.error(f"Ensemble separation failed: {e}")
        import traceback
        traceback.print_exc()
        job.error = str(e)
        return False
    finally:
        _separation_semaphore.release()


# ============ MODAL CLOUD GPU SEPARATION ============

def separate_stems_modal(job: ProcessingJob, audio_path: Path):
    """
    Stem separation via Modal cloud GPU (T4).

    Sends audio bytes to a Modal serverless function running htdemucs_6s,
    receives MP3 stems back, and saves them locally. Falls back to local
    GPU separation if Modal is unavailable or fails.

    No semaphore needed — this does not use local GPU resources.
    """
    if not MODAL_AVAILABLE:
        logger.warning("Modal not installed — falling back to local separation")
        return separate_stems_roformer(job, audio_path)

    try:
        job.stage = 'Uploading to cloud GPU (Modal)'
        job.progress = 10
        save_job_checkpoint(job)
        logger.info(f"Starting Modal cloud GPU separation for {audio_path}")

        # Read audio file
        audio_bytes = audio_path.read_bytes()
        size_mb = len(audio_bytes) / (1024 * 1024)
        logger.info(f"Uploading {size_mb:.1f} MB to Modal cloud GPU")

        job.stage = 'Separating stems on cloud GPU (T4)'
        job.progress = 20
        save_job_checkpoint(job)

        # Look up the DEPLOYED Modal function by app/function name.
        # Using modal.Function.from_name() connects to the already-deployed
        # "stemscribe-separator" app on Modal, instead of trying to run the
        # local function definition (which requires the app to be "running").
        separate_fn = modal.Function.from_name("stemscribe-separator", "separate_stems_gpu")
        stems_data = separate_fn.remote(audio_bytes, filename=audio_path.name)

        job.stage = 'Downloading stems from cloud'
        job.progress = 35
        save_job_checkpoint(job)

        # Save received stems locally
        output_path = OUTPUT_DIR / job.job_id / 'stems'
        output_path.mkdir(parents=True, exist_ok=True)

        for stem_name, stem_bytes in stems_data.items():
            stem_file = output_path / f'{stem_name}.mp3'
            stem_file.write_bytes(stem_bytes)
            job.stems[stem_name] = str(stem_file)
            size_mb = len(stem_bytes) / (1024 * 1024)
            logger.info(f"  Saved stem: {stem_name} ({size_mb:.1f} MB)")

        if not job.stems:
            raise Exception("Modal returned no stems")

        job.progress = 40
        save_job_checkpoint(job)
        logger.info(f"Modal cloud separation complete: {len(job.stems)} stems")
        return True

    except Exception as e:
        logger.error(f"Modal cloud separation failed: {e}")
        import traceback
        traceback.print_exc()

        # Fall back to local GPU
        logger.info("Falling back to local GPU separation...")
        job.stems = {}
        job.error = None
        return separate_stems_roformer(job, audio_path)

