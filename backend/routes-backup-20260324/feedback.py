"""
Feedback routes — capture user chord/lyrics corrections as training data.

POST /api/feedback/chord-correction  — save a chord correction
POST /api/feedback/lyrics-correction — save a lyrics correction
GET  /api/feedback/corrections        — list all corrections
"""

import json
import time
import logging
import threading
from pathlib import Path
from flask import Blueprint, request, jsonify

from middleware.validation import validate_job_id, sanitize_text

logger = logging.getLogger(__name__)

feedback_bp = Blueprint("feedback", __name__)

FEEDBACK_FILE = Path(__file__).parent.parent / 'feedback_data.json'
_lock = threading.Lock()


def _load_feedback():
    """Load feedback data from disk."""
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Corrupted feedback file, starting fresh")
    return {'chord_corrections': [], 'lyrics_corrections': []}


def _save_feedback(data):
    """Atomically save feedback data to disk."""
    tmp = FEEDBACK_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(FEEDBACK_FILE)


def _append_correction(category, entry):
    """Thread-safe append of a correction entry."""
    with _lock:
        data = _load_feedback()
        data.setdefault(category, []).append(entry)
        _save_feedback(data)


# ---- Chord correction ----

@feedback_bp.route('/api/feedback/chord-correction', methods=['POST'])
def chord_correction():
    """Receive a user chord correction.

    Body JSON:
        job_id          — processing job ID
        original_chord  — what the AI predicted
        corrected_chord — what the user changed it to
        position        — time position in seconds
        context         — optional surrounding context (e.g. nearby chords)
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'JSON body required'}), 400

    required = ('job_id', 'original_chord', 'corrected_chord', 'position')
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # Validate job_id format
    if not validate_job_id(body['job_id']):
        return jsonify({'error': 'Invalid job_id format'}), 400

    # Sanitize text inputs
    original_chord = sanitize_text(str(body['original_chord']), max_length=50)
    corrected_chord = sanitize_text(str(body['corrected_chord']), max_length=50)

    # Resolve song title from the job if available
    song_title = None
    try:
        from models.job import get_job
        job = get_job(body['job_id'])
        if job:
            song_title = job.metadata.get('title') or job.filename
    except Exception:
        pass

    entry = {
        'timestamp': time.time(),
        'job_id': body['job_id'],
        'song_title': song_title,
        'original_chord': original_chord,
        'corrected_chord': corrected_chord,
        'position': body['position'],
        'context': sanitize_text(str(body.get('context', '')), max_length=500) or None,
    }

    _append_correction('chord_corrections', entry)
    logger.info(f"Chord correction: {entry['original_chord']} -> {entry['corrected_chord']} "
                f"at {entry['position']}s (job {body['job_id'][:8]})")

    return jsonify({'status': 'saved', 'correction': entry}), 201


# ---- Lyrics correction ----

@feedback_bp.route('/api/feedback/lyrics-correction', methods=['POST'])
def lyrics_correction():
    """Receive a user lyrics correction.

    Body JSON:
        job_id         — processing job ID
        original_line  — AI-generated lyric line
        corrected_line — user's fix
        line_index     — line number in the lyrics
    """
    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'JSON body required'}), 400

    required = ('job_id', 'original_line', 'corrected_line', 'line_index')
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({'error': f'Missing fields: {missing}'}), 400

    # Validate job_id format
    if not validate_job_id(body['job_id']):
        return jsonify({'error': 'Invalid job_id format'}), 400

    song_title = None
    try:
        from models.job import get_job
        job = get_job(body['job_id'])
        if job:
            song_title = job.metadata.get('title') or job.filename
    except Exception:
        pass

    entry = {
        'timestamp': time.time(),
        'job_id': body['job_id'],
        'song_title': song_title,
        'original_line': sanitize_text(str(body['original_line']), max_length=1000),
        'corrected_line': sanitize_text(str(body['corrected_line']), max_length=1000),
        'line_index': body['line_index'],
    }

    _append_correction('lyrics_corrections', entry)
    logger.info(f"Lyrics correction at line {entry['line_index']} (job {body['job_id'][:8]})")

    return jsonify({'status': 'saved', 'correction': entry}), 201


# ---- List corrections ----

@feedback_bp.route('/api/feedback/corrections', methods=['GET'])
def list_corrections():
    """Return all collected corrections.

    Query params:
        type — 'chord', 'lyrics', or omit for all
        job_id — filter by job
    """
    data = _load_feedback()
    correction_type = request.args.get('type')
    job_id = request.args.get('job_id')

    result = {}
    if correction_type in (None, 'chord'):
        items = data.get('chord_corrections', [])
        if job_id:
            items = [c for c in items if c.get('job_id') == job_id]
        result['chord_corrections'] = items

    if correction_type in (None, 'lyrics'):
        items = data.get('lyrics_corrections', [])
        if job_id:
            items = [c for c in items if c.get('job_id') == job_id]
        result['lyrics_corrections'] = items

    result['total'] = sum(len(v) for v in result.values() if isinstance(v, list))
    return jsonify(result)
