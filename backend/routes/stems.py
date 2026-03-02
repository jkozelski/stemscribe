"""
Stem manipulation routes — split-stem, analyze-stereo, split-vocals.
"""

import logging
from pathlib import Path
from flask import Blueprint, request, jsonify

from models.job import get_job, save_job_to_disk, OUTPUT_DIR

logger = logging.getLogger(__name__)

stems_bp = Blueprint("stems", __name__)


@stems_bp.route('/api/split-stem/<job_id>/<stem_name>', methods=['POST'])
def split_stem_endpoint(job_id, stem_name):
    """Split a stem into left/right/center components using stereo panning analysis."""
    from dependencies import STEREO_SPLITTER_AVAILABLE

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not STEREO_SPLITTER_AVAILABLE:
        return jsonify({'error': 'Stereo splitter not available'}), 500

    from dependencies import split_stereo, check_if_splittable

    # Find the stem
    if stem_name not in job.stems:
        stem_name_lower = stem_name.lower()
        matching = [k for k in job.stems.keys() if k.lower() == stem_name_lower]
        if matching:
            stem_name = matching[0]
        else:
            return jsonify({
                'error': f'Stem "{stem_name}" not found',
                'available_stems': list(job.stems.keys())
            }), 404

    stem_path = job.stems[stem_name]

    # Check if splittable first
    check = check_if_splittable(stem_path)

    # Create output directory for split stems
    job_dir = Path(OUTPUT_DIR) / job_id
    split_dir = job_dir / 'stereo_split'
    split_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Splitting {stem_name} stem for job {job_id}")
    logger.info(f"   Stereo analysis: width={check.get('width', 0):.2f}, side_ratio={check.get('side_ratio', 0):.2f}")

    # Do the split using enhanced method
    split_results = split_stereo(
        input_path=stem_path,
        output_dir=str(split_dir),
        stem_type=stem_name,
        method='enhanced'
    )

    if not split_results:
        return jsonify({
            'error': 'Stereo split failed',
            'analysis': check
        }), 500

    # Add split stems to job's stem dictionary
    for split_name, split_path in split_results.items():
        job.stems[split_name] = split_path

    # Save updated job state
    save_job_to_disk(job)

    return jsonify({
        'success': True,
        'job_id': job_id,
        'original_stem': stem_name,
        'analysis': check,
        'split_stems': {k: v for k, v in split_results.items()},
        'message': f'Split {stem_name} into {len(split_results)} components'
    })


@stems_bp.route('/api/analyze-stereo/<job_id>/<stem_name>', methods=['GET'])
def analyze_stereo_endpoint(job_id, stem_name):
    """Analyze a stem's stereo field without splitting it."""
    from dependencies import STEREO_SPLITTER_AVAILABLE

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not STEREO_SPLITTER_AVAILABLE:
        return jsonify({'error': 'Stereo splitter not available'}), 500

    from dependencies import check_if_splittable

    # Find the stem
    if stem_name not in job.stems:
        stem_name_lower = stem_name.lower()
        matching = [k for k in job.stems.keys() if k.lower() == stem_name_lower]
        if matching:
            stem_name = matching[0]
        else:
            return jsonify({
                'error': f'Stem "{stem_name}" not found',
                'available_stems': list(job.stems.keys())
            }), 404

    stem_path = job.stems[stem_name]
    check = check_if_splittable(stem_path)

    return jsonify({
        'job_id': job_id,
        'stem': stem_name,
        'analysis': check,
        'recommendation': 'Split recommended' if check.get('splittable') else 'Mostly mono - splitting may not help'
    })


@stems_bp.route('/api/split-vocals/<job_id>', methods=['POST'])
def split_vocals_endpoint(job_id):
    """Split vocals into lead and backing vocals using AI (two-pass UVR method)."""
    from dependencies import ENHANCED_SEPARATOR_AVAILABLE

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not ENHANCED_SEPARATOR_AVAILABLE:
        return jsonify({'error': 'Enhanced separator not available (audio-separator not installed)'}), 500

    from dependencies import EnhancedSeparator

    # Find vocals stem
    vocals_path = None
    vocals_key = None
    for key in ['vocals', 'Vocals', 'vocal', 'Vocal']:
        if key in job.stems:
            vocals_path = job.stems[key]
            vocals_key = key
            break

    if not vocals_path:
        return jsonify({
            'error': 'No vocals stem found',
            'available_stems': list(job.stems.keys())
        }), 404

    # Create output directory
    job_dir = Path(OUTPUT_DIR) / job_id
    vocal_split_dir = job_dir / 'vocal_split'
    vocal_split_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Splitting vocals for job {job_id}")

    try:
        separator = EnhancedSeparator(output_dir=str(vocal_split_dir))
        lead_path, backing_path = separator.split_lead_backing_vocals(vocals_path)

        # Add to job stems
        job.stems['vocals_lead'] = lead_path
        job.stems['vocals_backing'] = backing_path

        # Save job
        save_job_to_disk(job)

        return jsonify({
            'success': True,
            'job_id': job_id,
            'original_vocals': vocals_path,
            'lead_vocals': lead_path,
            'backing_vocals': backing_path,
            'message': 'Successfully split vocals into lead and backing'
        })

    except Exception as e:
        logger.error(f"Vocal split failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Vocal split failed: {str(e)}'
        }), 500
