"""
Library routes — browse and manage processed songs.

Each user sees only their own library. Anonymous users are tracked by session cookie.
Jeff (admin) can see all songs with ?all=true.
"""

import shutil
import logging
from urllib.parse import quote, unquote
import requests as http_requests
from flask import Blueprint, jsonify, request, g, Response

from models.job import jobs, get_job, save_job_to_disk, OUTPUT_DIR
from auth.middleware import auth_required
from middleware.validation import validate_job_id

logger = logging.getLogger(__name__)

library_bp = Blueprint("library", __name__)

# Admin emails that can see all songs and delete anything
ADMIN_EMAILS = {'jkozelski@gmail.com', 'jeff@tidepoolartist.com'}

# Allowed thumbnail domains (only proxy these)
_ALLOWED_THUMB_HOSTS = ('i.ytimg.com', 'img.youtube.com')


def _is_admin():
    """Check if the current user is an admin."""
    user = getattr(g, 'current_user', None)
    return user and getattr(user, 'email', None) in ADMIN_EMAILS


def _job_belongs_to_user(job, user, session_id):
    """Check if a job belongs to the current user or session."""
    if user:
        uid = str(user.id)
        # Check direct user_id field or legacy metadata field
        if job.user_id == uid or job.metadata.get('user_id') == uid:
            return True
    if session_id and job.session_id == session_id:
        return True
    return False


def _thumb_url(raw):
    """Rewrite a raw thumbnail URL to go through the proxy endpoint.
    Local paths (starting with /) are returned as-is."""
    if not raw:
        return None
    if raw.startswith('/'):
        return raw  # Local image, serve directly
    return '/api/thumbnail?url=' + quote(raw, safe='')


@library_bp.route('/api/thumbnail', methods=['GET'])
def proxy_thumbnail():
    """Proxy YouTube thumbnails to avoid hotlink-protection 403s."""
    raw_url = request.args.get('url', '')
    if not raw_url:
        return '', 404

    # Decode in case it was double-encoded
    url = unquote(raw_url)

    # Safety: only proxy known YouTube thumbnail hosts
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.hostname not in _ALLOWED_THUMB_HOSTS:
        return '', 403

    try:
        resp = http_requests.get(
            url,
            timeout=8,
            headers={
                'User-Agent': 'Mozilla/5.0 (compatible; StemScriber/1.0)',
                'Referer': 'https://www.youtube.com/',
            }
        )
        content_type = resp.headers.get('Content-Type', 'image/jpeg')
        return Response(
            resp.content,
            status=resp.status_code,
            content_type=content_type,
            headers={
                'Cache-Control': 'public, max-age=86400',
                'X-Content-Type-Options': 'nosniff',
            }
        )
    except Exception as e:
        logger.warning(f"Thumbnail proxy failed for {url}: {e}")
        return '', 502


@library_bp.route('/api/library', methods=['GET'])
@auth_required(optional=True)
def get_library():
    """Get list of processed songs in the user's library.

    Query params:
        all=true  (admin only) — return all songs across all users.
    """
    user = getattr(g, 'current_user', None)
    session_id = request.cookies.get('session_id')
    is_admin = _is_admin()

    # Fallback admin check: try to get user from JWT directly
    if not is_admin and not user:
        try:
            from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
            verify_jwt_in_request(optional=True)
            uid = get_jwt_identity()
            if uid:
                from auth.models import get_user_by_id
                fallback_user = get_user_by_id(uid)
                if fallback_user:
                    user = fallback_user
                    is_admin = getattr(fallback_user, 'email', None) in ADMIN_EMAILS
        except Exception:
            pass

    show_all = request.args.get('all', '').lower() == 'true' and is_admin

    library = []

    for job_id, job in jobs.items():
        if job.status != 'completed' or not job.stems:
            continue

        # Demo songs are visible to everyone
        is_demo = job.metadata.get('demo', False) if job.metadata else False

        # Determine visibility
        if is_demo:
            pass  # Demo songs always shown
        elif is_admin:
            pass  # Admin always sees everything
        elif show_all:
            pass  # Explicit all flag
        elif user:
            # Logged-in user: show only their own jobs
            if not _job_belongs_to_user(job, user, session_id):
                continue
        else:
            # Anonymous: show only their own session's jobs
            if not session_id:
                continue
            if job.session_id != session_id:
                continue

        library.append({
            'job_id': job.job_id,
            'title': job.metadata.get('title', job.filename),
            'artist': job.metadata.get('artist', 'Unknown Artist'),
            'duration': job.metadata.get('duration', 0),
            'created_at': job.created_at,
            'stem_count': len(job.stems),
            'has_midi': len(job.midi_files) > 0,
            'has_gp': len(job.gp_files) > 0,
            'thumbnail': _thumb_url(job.metadata.get('thumbnail')),
            'demo': is_demo,
            'source_url': job.source_url
        })

    # Sort by created_at descending (newest first)
    library.sort(key=lambda x: x['created_at'], reverse=True)

    return jsonify({
        'library': library,
        'total': len(library)
    })


@library_bp.route('/api/library/claim', methods=['POST'])
@auth_required
def claim_jobs():
    """Claim unclaimed jobs by job_id. Body: {"job_ids": ["id1", "id2", ...]}

    A signed-in user can claim jobs that have no user_id (legacy/unclaimed).
    """
    user = g.current_user
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json() or {}
    job_ids = data.get('job_ids', [])
    if not isinstance(job_ids, list):
        return jsonify({'error': 'job_ids must be a list'}), 400

    claimed = []
    errors = []
    uid = str(user.id)

    for jid in job_ids:
        if not validate_job_id(jid):
            errors.append({'job_id': jid, 'error': 'Invalid job ID'})
            continue

        job = get_job(jid)
        if not job:
            errors.append({'job_id': jid, 'error': 'Job not found'})
            continue

        if job.user_id is not None and job.user_id != uid:
            errors.append({'job_id': jid, 'error': 'Job already owned by another user'})
            continue

        job.user_id = uid
        save_job_to_disk(job)
        claimed.append(jid)

    logger.info(f"User {uid} claimed {len(claimed)} jobs")
    return jsonify({'claimed': claimed, 'errors': errors})


@library_bp.route('/api/library/<job_id>', methods=['DELETE'])
@auth_required
def delete_from_library(job_id):
    """Delete a song from the library. Only the owner or admin can delete."""
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    user = getattr(g, 'current_user', None)

    # Authorization: admin can delete anything, owner can delete their own,
    # any signed-in user can delete unclaimed jobs (no user_id)
    if not _is_admin():
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        session_id = request.cookies.get('session_id')
        is_owner = _job_belongs_to_user(job, user, session_id)
        is_unclaimed = job.user_id is None
        if not is_owner and not is_unclaimed:
            return jsonify({'error': 'You do not own this song'}), 403

    try:
        # Remove the output directory
        job_dir = OUTPUT_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)

        # Remove from memory
        if job_id in jobs:
            del jobs[job_id]

        logger.info(f"Deleted job {job_id} from library (by user {getattr(user, 'id', 'unknown')})")
        return jsonify({'status': 'deleted', 'job_id': job_id})

    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500
