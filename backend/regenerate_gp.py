#!/usr/bin/env python3
"""
Quick utility to regenerate Guitar Pro files from existing MIDI files.
Run this after installing pyguitarpro to add TAB notation to existing jobs.

Usage:
    python regenerate_gp.py <job_id>
    python regenerate_gp.py all           # Regenerate all jobs
    python regenerate_gp.py all --force   # Force overwrite existing files
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import our GP converter
from midi_to_gp import convert_midi_to_gp

OUTPUTS_DIR = Path(__file__).parent.parent / 'outputs'


def regenerate_job(job_dir: Path, force: bool = False) -> int:
    """Regenerate GP files for a single job. Returns count of files created."""
    midi_dir = job_dir / 'midi'
    gp_dir = job_dir / 'guitarpro'

    if not midi_dir.exists():
        logger.warning(f"No MIDI directory found: {midi_dir}")
        return 0

    gp_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    midi_files = list(midi_dir.glob('*.mid'))
    logger.info(f"\n🎸 Processing {job_dir.name}: {len(midi_files)} MIDI files")

    for midi_path in midi_files:
        stem_name = midi_path.stem.replace('_enhanced', '').replace('_basic_pitch', '').replace('_transcribed', '')
        instrument_type = stem_name.split('_')[0]

        # Skip stems that don't work well with tablature
        if instrument_type in ['vocals', 'other', 'drums']:
            logger.info(f"  ⏭️  Skipping {stem_name} (not suited for TAB)")
            continue

        gp_path = gp_dir / f"{midi_path.stem}.gp5"

        if gp_path.exists() and not force:
            logger.info(f"  ✓ Already exists: {gp_path.name}")
            continue

        logger.info(f"  🎼 Converting: {midi_path.name} -> {gp_path.name}")

        if convert_midi_to_gp(
            midi_path=str(midi_path),
            output_path=str(gp_path),
            instrument_type=instrument_type,
            title=stem_name.replace('_', ' ').title()
        ):
            count += 1
            logger.info(f"  ✅ Created: {gp_path.name}")
        else:
            logger.warning(f"  ❌ Failed: {midi_path.name}")

    return count


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable jobs:")
        for job_dir in sorted(OUTPUTS_DIR.iterdir()):
            if job_dir.is_dir() and (job_dir / 'midi').exists():
                midi_count = len(list((job_dir / 'midi').glob('*.mid')))
                gp_count = len(list((job_dir / 'guitarpro').glob('*.gp5'))) if (job_dir / 'guitarpro').exists() else 0
                print(f"  {job_dir.name} - {midi_count} MIDI files, {gp_count} GP files")
        sys.exit(1)

    target = sys.argv[1]
    force = '--force' in sys.argv
    total = 0

    if target.lower() == 'all':
        # Regenerate all jobs
        for job_dir in sorted(OUTPUTS_DIR.iterdir()):
            if job_dir.is_dir() and (job_dir / 'midi').exists():
                total += regenerate_job(job_dir, force=force)
    else:
        # Single job
        job_dir = OUTPUTS_DIR / target
        if not job_dir.exists():
            logger.error(f"Job not found: {target}")
            sys.exit(1)
        total = regenerate_job(job_dir, force=force)

    logger.info(f"\n🎉 Done! Created {total} Guitar Pro files")


if __name__ == '__main__':
    main()
