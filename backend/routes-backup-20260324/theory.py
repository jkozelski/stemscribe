"""
Chord and theory routes — chord progressions, scale suggestions, theory analysis.
"""

import logging
from flask import Blueprint, request, jsonify

from models.job import get_job

logger = logging.getLogger(__name__)

theory_bp = Blueprint("theory", __name__)


@theory_bp.route('/api/chords/<job_id>', methods=['GET'])
def get_chords(job_id):
    """Get detected chord progression for a job."""
    from dependencies import CHORD_DETECTOR_AVAILABLE, CHORD_DETECTOR_VERSION

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify({
        'job_id': job_id,
        'chords': job.chord_progression,
        'key': job.detected_key,
        'available': CHORD_DETECTOR_AVAILABLE,
        'detector_version': CHORD_DETECTOR_VERSION,
        'chord_count': len(job.chord_progression),
        'has_inversions': CHORD_DETECTOR_VERSION == 'v8'
    })


@theory_bp.route('/api/theory/<job_id>', methods=['GET'])
def get_chord_theory(job_id):
    """Get scale suggestions and theory analysis for a job's chord progression."""
    from dependencies import CHORD_THEORY_AVAILABLE, _chord_theory_engine

    if not CHORD_THEORY_AVAILABLE:
        return jsonify({'error': 'Chord theory engine not available'}), 500

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not job.chord_progression:
        return jsonify({'error': 'No chord progression detected yet'}), 404

    try:
        key = job.detected_key
        chord_names = [c['chord'] for c in job.chord_progression]

        analyses = _chord_theory_engine.get_scales_for_progression(chord_names, key)
        practice_suggestion = _chord_theory_engine.suggest_practice_approach(chord_names, key)

        theory_data = []
        for i, chord_data in enumerate(job.chord_progression):
            analysis = analyses[i] if i < len(analyses) else {}
            theory_data.append({
                'chord': chord_data['chord'],
                'time': chord_data['time'],
                'duration': chord_data['duration'],
                'scales': analysis.get('scales', []),
                'secondary_scales': analysis.get('secondary_scales', []),
                'tip': analysis.get('tip', ''),
                'chord_type': analysis.get('chord_type', ''),
                'function': analysis.get('function'),
                'intervals': analysis.get('intervals', ''),
            })

        return jsonify({
            'job_id': job_id,
            'key': key,
            'theory': theory_data,
            'practice_suggestion': practice_suggestion,
            'available': True,
            'chord_count': len(theory_data)
        })

    except Exception as e:
        logger.error(f"Chord theory analysis failed: {e}")
        return jsonify({'error': f'Theory analysis failed: {str(e)}'}), 500


@theory_bp.route('/api/theory/chord', methods=['POST'])
def get_single_chord_theory():
    """Get scale suggestions for a single chord (no job required)."""
    from dependencies import CHORD_THEORY_AVAILABLE, _chord_theory_engine

    if not CHORD_THEORY_AVAILABLE:
        return jsonify({'error': 'Chord theory engine not available'}), 500

    data = request.get_json()
    if not data or 'chord' not in data:
        return jsonify({'error': 'Missing chord parameter'}), 400

    chord = data['chord']
    key = data.get('key')

    analysis = _chord_theory_engine.analyze(chord, key)
    return jsonify(analysis)
