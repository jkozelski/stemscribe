"""
Smart stem extraction — recursive deep extraction of the 'other' stem.

Analyzes RMS energy distribution and recursively runs Demucs to find hidden instruments.
"""

import os
import logging
from pathlib import Path

import numpy as np

from models.job import ProcessingJob, OUTPUT_DIR
from processing.utils import convert_wavs_to_mp3
from processing.separation import (
    _active_runners, _active_runners_lock,
    ENSEMBLE_SEPARATOR_AVAILABLE,
)

logger = logging.getLogger(__name__)

try:
    from demucs_runner import DemucsRunner
except ImportError:
    pass

try:
    from skills import apply_skill, analyze_stems_for_skills
    SKILLS_AVAILABLE = True
except ImportError:
    SKILLS_AVAILABLE = False


def _measure_stem_energy(stem_path: str) -> float:
    """Measure RMS energy of an audio file. Returns 0.0 on failure."""
    try:
        import librosa
        y, sr = librosa.load(stem_path, sr=22050, mono=True, duration=120)
        rms = float(np.sqrt(np.mean(y ** 2)))
        return rms
    except Exception as e:
        logger.debug(f"Energy measurement failed for {stem_path}: {e}")
        return 0.0


def _smart_rename_stem(raw_name: str, existing_stems: dict) -> str:
    """
    Rename an extracted sub-stem intelligently based on what already exists.

    Examples:
        other_vocals → backing_vocals (if 'vocals' exists)
        other_guitar → guitar_2 (if 'guitar' exists)
        other_piano  → keys_2 (if 'piano' exists)
        other_drums  → percussion_2 (if 'drums' exists)
        other_bass   → bass_2 (if 'bass' exists)
        other_other  → other_deep (recursive residual)
    """
    # Strip prefix layers: other_other_vocals → vocals
    base = raw_name
    while base.startswith('other_'):
        base = base[6:]

    if not base:
        return raw_name  # safety fallback

    rename_map = {
        'vocals': 'backing_vocals',
        'guitar': 'guitar_2',
        'piano': 'keys_2',
        'drums': 'percussion_2',
        'bass': 'bass_2',
        'other': 'other_deep',
    }

    # If the base instrument already exists in primary stems, use the smart name
    if base in existing_stems:
        candidate = rename_map.get(base, f'{base}_2')
    else:
        candidate = base  # No conflict — use the clean name

    # Avoid collisions: if candidate already taken, increment suffix
    if candidate in existing_stems:
        i = 2
        while f'{candidate}_{i}' in existing_stems:
            i += 1
        candidate = f'{candidate}_{i}'

    return candidate


def _deep_extract_stem(job: ProcessingJob, stem_path: str, depth: int,
                       max_depth: int, progress_base: float, progress_range: float) -> dict:
    """
    Recursively run Demucs on a stem to extract hidden instruments.

    Returns dict of {smart_name: file_path} for all extracted stems.
    """
    extracted = {}

    if depth > max_depth:
        return extracted

    if not os.path.exists(stem_path):
        return extracted

    pass_label = f"Pass {depth + 1}"
    output_path = OUTPUT_DIR / job.job_id / 'stems' / f'deep_pass_{depth}'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        def progress_callback(progress):
            pct = progress_base + (progress.percent / 100.0) * progress_range
            job.progress = int(pct)
            if progress.eta_seconds > 0:
                eta_min = progress.eta_seconds / 60
                job.stage = f'Extracting hidden instruments ({pass_label}): {progress.percent:.0f}% (ETA: {eta_min:.1f}m)'
            else:
                job.stage = f'Extracting hidden instruments ({pass_label}): {progress.percent:.0f}%'

        runner = DemucsRunner(model='htdemucs_6s', progress_callback=progress_callback)
        with _active_runners_lock:
            _active_runners.append(runner)
        try:
            result = runner.separate(
                audio_path=Path(stem_path),
                output_dir=output_path,
                timeout_seconds=1200  # 20 min max per pass
            )
        finally:
            with _active_runners_lock:
                if runner in _active_runners:
                    _active_runners.remove(runner)

        if not result.success:
            logger.warning(f"Deep extraction {pass_label} failed: {result.error_message}")
            return extracted

        # Find output directory
        stem_dir = result.output_dir
        if not stem_dir or not stem_dir.exists():
            # Fallback: look for the expected path
            stem_dir = output_path / 'htdemucs_6s' / Path(stem_path).stem
            if not stem_dir.exists():
                logger.warning(f"Deep extraction output dir not found for {pass_label}")
                return extracted

        # Convert WAV → MP3
        convert_wavs_to_mp3(stem_dir)

        # Collect results, measure energy, keep only meaningful stems
        residual_path = None
        for stem_file in stem_dir.glob('*.mp3'):
            sub_name = stem_file.stem  # e.g. "vocals", "drums", "other"
            energy = _measure_stem_energy(str(stem_file))

            if energy < 0.005:
                logger.info(f"  ⏭️ {pass_label}: Skipping near-silent stem '{sub_name}' (energy={energy:.4f})")
                continue

            if sub_name == 'other':
                # This is the residual — candidate for deeper pass
                residual_path = str(stem_file)
                residual_energy = energy
                logger.info(f"  📦 {pass_label}: Residual 'other' has energy={energy:.4f}")
                continue

            # Smart rename and add
            smart_name = _smart_rename_stem(sub_name, {**job.stems, **extracted})
            extracted[smart_name] = str(stem_file)
            logger.info(f"  🎵 {pass_label}: Extracted '{smart_name}' (energy={energy:.4f})")

        # Recursive pass on residual if it has significant content
        if residual_path and depth < max_depth:
            # Measure total energy to decide if residual is worth processing
            total_energy = sum(_measure_stem_energy(p) for p in job.stems.values() if os.path.exists(p))
            if total_energy > 0 and (residual_energy / total_energy) > 0.10:
                logger.info(f"  🔄 Residual has {residual_energy/total_energy:.0%} of total energy — going deeper")
                deeper = _deep_extract_stem(
                    job, residual_path, depth + 1, max_depth,
                    progress_base + progress_range,
                    progress_range * 0.5  # Each deeper pass gets half the progress range
                )
                extracted.update(deeper)
            else:
                # Keep the residual as a stem if it has content
                if residual_energy > 0.01:
                    residual_name = _smart_rename_stem('other', {**job.stems, **extracted})
                    extracted[residual_name] = residual_path
                    logger.info(f"  📦 Keeping residual as '{residual_name}'")

        return extracted

    except Exception as e:
        logger.error(f"Deep extraction {pass_label} failed: {e}")
        import traceback
        traceback.print_exc()
        return extracted


def smart_separate(job: ProcessingJob):
    """
    Smart Deep Extraction — automatically analyze and extract all instruments.

    Philosophy: One button, best result every time. No user configuration needed.

    Pipeline:
    1. Measure RMS energy of each stem from initial separation
    2. If 'other' has significant content (>10% of total energy), run deep extraction
    3. Recursively process residuals (max depth 2)
    4. Smart-name all extracted stems (backing_vocals, guitar_2, keys_2, etc.)
    5. Auto-run Skills system on remaining content for frequency-based extraction

    Replaces the old cascade_separate_other() with intelligent, automatic behavior.
    """
    other_path = job.stems.get('other')
    if not other_path or not os.path.exists(other_path):
        logger.info("🧠 Smart separation: No 'other' stem — track is cleanly separated")
        return False

    # Step 1: Measure energy across all stems
    job.stage = 'Analyzing audio complexity...'
    job.progress = 42
    logger.info("🧠 Smart separation: Analyzing stem energy distribution...")

    stem_energies = {}
    for name, path in job.stems.items():
        if os.path.exists(path):
            stem_energies[name] = _measure_stem_energy(path)

    total_energy = sum(stem_energies.values())
    other_energy = stem_energies.get('other', 0.0)

    if total_energy == 0:
        logger.warning("Smart separation: Could not measure stem energies")
        return False

    other_ratio = other_energy / total_energy
    logger.info(f"🧠 Energy distribution:")
    for name, energy in sorted(stem_energies.items(), key=lambda x: -x[1]):
        pct = (energy / total_energy * 100) if total_energy > 0 else 0
        logger.info(f"   {name}: {pct:.1f}% (rms={energy:.4f})")

    # Step 2: Decide whether to deep-extract
    if other_ratio < 0.03:
        logger.info(f"🧠 'other' is only {other_ratio:.0%} of total — simple track, skipping deep extraction")
        return False

    if other_ratio < 0.05:
        logger.info(f"🧠 'other' is {other_ratio:.0%} of total — minor content, skipping deep extraction")
        return False

    logger.info(f"🧠 'other' is {other_ratio:.0%} of total energy — running deep extraction!")

    # Step 3: Deep extraction (recursive, max depth 2)
    job.stage = 'Extracting hidden instruments...'
    job.progress = 44
    extracted = _deep_extract_stem(
        job, other_path, depth=0, max_depth=2,
        progress_base=44, progress_range=10  # 44-54% of overall progress
    )

    if extracted:
        # Add all extracted stems to job
        for name, path in extracted.items():
            job.stems[name] = path
        logger.info(f"🧠✅ Smart extraction found {len(extracted)} additional instruments: {list(extracted.keys())}")
    else:
        logger.info("🧠 Deep extraction didn't find significant additional instruments")

    # Step 4: Auto-run Skills system for frequency-based extraction
    if SKILLS_AVAILABLE:
        job.stage = 'Auto-detecting additional instruments...'
        job.progress = 55
        try:
            # Lazy import to avoid circular dependency with pipeline module
            from processing.pipeline import apply_skills_to_job
            apply_skills_to_job(job)
        except Exception as e:
            logger.warning(f"Auto-skills failed (non-fatal): {e}")

    total_stems = len(job.stems)
    total_sub = sum(len(v) for v in job.sub_stems.values()) if job.sub_stems else 0
    logger.info(f"🧠 Smart separation complete: {total_stems} stems + {total_sub} sub-stems")

    return len(extracted) > 0

