#!/usr/bin/env python3
"""
Quick utility to regenerate chord detection for existing jobs.
Run this to add chord progressions to jobs that were processed before chord detection was available.

Usage:
    python regenerate_chords.py <job_id>
    python regenerate_chords.py all  # Regenerate all jobs
"""

import sys
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'

try:
    from chord_detector import ChordDetector
    CHORD_DETECTOR_AVAILABLE = True
except ImportError:
    CHORD_DETECTOR_AVAILABLE = False
    logger.error("chord_detector not available")


def regenerate_job_chords(job_dir: Path) -> bool:
    """Regenerate chord detection for a single job."""
    if not CHORD_DETECTOR_AVAILABLE:
        logger.error("Chord detector not available")
        return False

    stems_dir = job_dir / 'stems'
    job_file = job_dir / 'job.json'

    # Find best audio source for chord detection
    analyze_path = None
    source_name = None

    # Priority: guitar > piano > other stems
    # Check both flat and nested directory structures
    stem_priorities = ['guitar', 'guitar_left', 'guitar_center', 'piano', 'piano_left', 'bass', 'other']

    for stem_name in stem_priorities:
        # Try flat structure first
        stem_path = stems_dir / f'{stem_name}.wav'
        if stem_path.exists():
            analyze_path = str(stem_path)
            source_name = stem_name
            break

        # Try nested structures (htdemucs_6s, stereo_split, etc.)
        for subdir in stems_dir.iterdir():
            if subdir.is_dir():
                nested_path = subdir / f'{stem_name}.wav'
                if nested_path.exists():
                    analyze_path = str(nested_path)
                    source_name = stem_name
                    break
        if analyze_path:
            break

    if not analyze_path:
        # Try any wav file recursively
        wav_files = list(stems_dir.rglob('*.wav'))
        if wav_files:
            # Prefer guitar or piano
            for wf in wav_files:
                if 'guitar' in wf.stem.lower() or 'piano' in wf.stem.lower():
                    analyze_path = str(wf)
                    source_name = wf.stem
                    break
            if not analyze_path:
                analyze_path = str(wav_files[0])
                source_name = wav_files[0].stem

    if not analyze_path:
        logger.warning(f"No audio found for {job_dir.name}")
        return False

    logger.info(f"\n🎸 Analyzing chords for {job_dir.name} using {source_name}...")

    try:
        detector = ChordDetector()
        progression = detector.detect(analyze_path)

        chord_data = [
            {
                'time': c.time,
                'duration': c.duration,
                'chord': c.chord,
                'root': c.root,
                'quality': c.quality,
                'confidence': c.confidence
            }
            for c in progression.chords
        ]

        logger.info(f"  ✅ Found {len(chord_data)} chord changes, key: {progression.key}")

        # Update job.json if it exists
        if job_file.exists():
            with open(job_file) as f:
                job_data = json.load(f)

            job_data['chord_progression'] = chord_data
            job_data['detected_key'] = progression.key

            with open(job_file, 'w') as f:
                json.dump(job_data, f, indent=2)

            logger.info(f"  ✅ Updated {job_file.name}")
        else:
            # Save standalone chord file
            chord_file = job_dir / 'chords.json'
            with open(chord_file, 'w') as f:
                json.dump({
                    'chord_progression': chord_data,
                    'detected_key': progression.key
                }, f, indent=2)
            logger.info(f"  ✅ Created {chord_file.name}")

        return True

    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")
        return False


def main():
    if not CHORD_DETECTOR_AVAILABLE:
        logger.error("Chord detector not available. Install librosa: pip install librosa")
        sys.exit(1)

    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable jobs:")
        for job_dir in sorted(OUTPUTS_DIR.iterdir()):
            if job_dir.is_dir() and (job_dir / 'stems').exists():
                job_file = job_dir / 'job.json'
                has_chords = False
                if job_file.exists():
                    try:
                        with open(job_file) as f:
                            data = json.load(f)
                            has_chords = bool(data.get('chord_progression'))
                    except:
                        pass
                status = "✅ has chords" if has_chords else "❌ no chords"
                print(f"  {job_dir.name} - {status}")
        sys.exit(1)

    target = sys.argv[1]
    success_count = 0
    total_count = 0

    if target.lower() == 'all':
        # Regenerate all jobs
        for job_dir in sorted(OUTPUTS_DIR.iterdir()):
            if job_dir.is_dir() and (job_dir / 'stems').exists():
                total_count += 1
                if regenerate_job_chords(job_dir):
                    success_count += 1
    else:
        # Single job
        job_dir = OUTPUTS_DIR / target
        if not job_dir.exists():
            logger.error(f"Job not found: {target}")
            sys.exit(1)
        total_count = 1
        if regenerate_job_chords(job_dir):
            success_count = 1

    logger.info(f"\n🎉 Done! Processed {success_count}/{total_count} jobs")


if __name__ == '__main__':
    main()
