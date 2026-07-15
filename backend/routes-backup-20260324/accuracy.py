"""
Accuracy and error pattern API routes.

GET /api/accuracy/<job_id>   — chord accuracy report for a job
GET /api/errors/patterns     — error pattern analysis
GET /api/errors/recent       — recent errors
"""

import logging
from flask import Blueprint, request, jsonify

from middleware.validation import validate_job_id

logger = logging.getLogger(__name__)

accuracy_bp = Blueprint("accuracy", __name__)


@accuracy_bp.route('/api/accuracy/<job_id>', methods=['GET'])
def get_accuracy(job_id):
    """Return chord accuracy report for a job.

    If a pre-computed score exists, returns it.
    Otherwise, attempts to compute one by comparing AI chords vs Songsterr chords.
    """
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400

    from chord_accuracy import get_accuracy_score, score_chord_accuracy, save_accuracy_score
    from models.job import get_job

    # Check for cached score
    cached = get_accuracy_score(job_id)
    if cached:
        return jsonify({'job_id': job_id, **cached})

    # Try to compute on the fly
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    ai_chords = job.chord_progression
    if not ai_chords:
        return jsonify({'error': 'No AI chord data for this job'}), 404

    # Check if Songsterr data is available as ground truth
    songsterr_chords = None
    try:
        import json
        from models.job import OUTPUT_DIR
        # Look for songsterr chord data saved alongside the job
        songsterr_file = OUTPUT_DIR / job_id / 'songsterr_chords.json'
        if songsterr_file.exists():
            with open(songsterr_file) as f:
                songsterr_chords = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load Songsterr chords: {e}")

    if not songsterr_chords:
        return jsonify({
            'error': 'No ground truth (Songsterr) chords available for comparison',
            'ai_chord_count': len(ai_chords),
        }), 404

    report = score_chord_accuracy(ai_chords, songsterr_chords)
    save_accuracy_score(job_id, report)

    return jsonify({'job_id': job_id, **report})


@accuracy_bp.route('/api/accuracy', methods=['GET'])
def list_accuracy_scores():
    """Return all accuracy scores (summary view)."""
    from chord_accuracy import get_all_accuracy_scores

    scores = get_all_accuracy_scores()
    summary = []
    for job_id, report in scores.items():
        summary.append({
            'job_id': job_id,
            'accuracy_percent': report.get('accuracy_percent'),
            'correct': report.get('correct'),
            'wrong': report.get('wrong'),
            'missed': report.get('missed'),
            'scored_at': report.get('scored_at'),
        })

    return jsonify({
        'scores': summary,
        'total': len(summary),
        'average_accuracy': round(
            sum(s['accuracy_percent'] for s in summary if s.get('accuracy_percent') is not None)
            / max(len(summary), 1), 1
        ) if summary else None,
    })


@accuracy_bp.route('/api/errors/patterns', methods=['GET'])
def error_patterns():
    """Return error pattern analysis."""
    from error_tracker import get_error_patterns
    return jsonify(get_error_patterns())


@accuracy_bp.route('/api/errors/recent', methods=['GET'])
def recent_errors():
    """Return recent errors.

    Query params:
        limit — max number of errors (default 50)
    """
    from error_tracker import get_recent_errors
    try:
        limit = int(request.args.get('limit', 50))
        limit = max(1, min(limit, 500))
    except (ValueError, TypeError):
        limit = 50
    return jsonify({'errors': get_recent_errors(limit)})
