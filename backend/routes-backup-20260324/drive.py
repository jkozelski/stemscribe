"""
Google Drive API routes.
"""

import logging
from flask import Blueprint, request, jsonify

from models.job import get_job
from auth.middleware import auth_required

logger = logging.getLogger(__name__)

# Conditional imports
try:
    from drive_service import upload_job_to_drive, get_drive_stats, get_drive_service
    DRIVE_AVAILABLE = True
except ImportError:
    DRIVE_AVAILABLE = False

drive_bp = Blueprint("drive", __name__)


@drive_bp.route('/api/drive/auth', methods=['GET'])
@auth_required
def drive_auth():
    """Initiate Google Drive OAuth flow"""
    if not DRIVE_AVAILABLE:
        return jsonify({'error': 'Google Drive integration not available. Install: pip install google-auth-oauthlib google-api-python-client --break-system-packages'}), 500

    try:
        service = get_drive_service()
        if service:
            return jsonify({'status': 'authenticated', 'message': 'Google Drive connected!'})
        else:
            return jsonify({'status': 'auth_required', 'message': 'Please complete OAuth flow in browser'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@drive_bp.route('/api/drive/stats', methods=['GET'])
def drive_stats():
    """Get Google Drive storage stats for StemScribe"""
    if not DRIVE_AVAILABLE:
        return jsonify({'error': 'Drive integration not available'}), 500

    stats = get_drive_stats()
    if stats:
        return jsonify(stats)
    else:
        return jsonify({'error': 'Could not get Drive stats - may need to authenticate'}), 500


@drive_bp.route('/api/drive/upload/<job_id>', methods=['POST'])
@auth_required
def drive_upload_job(job_id):
    """Manually upload a job to Google Drive"""
    if not DRIVE_AVAILABLE:
        return jsonify({'error': 'Drive integration not available'}), 500

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.status != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400

    data = request.get_json() or {}
    keep_stems = data.get('keep_stems', False)

    result = upload_job_to_drive(job, keep_stems=keep_stems)
    if result:
        job.metadata['drive_upload'] = result
        return jsonify({'status': 'uploaded', 'result': result})
    else:
        return jsonify({'error': 'Upload failed'}), 500
