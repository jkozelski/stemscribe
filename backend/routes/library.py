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
    """Rewrite a raw thumbnail URL.
    - Local paths (starting with /) are returned as-is.
    - YouTube hosts go through our proxy (hotlink-protection 403 dodge).
    - Everything else (iTunes / archive.org / CoverArtArchive / Wikimedia) is
      returned direct — those CDNs serve cross-origin fine and CSP img-src
      already whitelists them.
    """
    if not raw:
        return None
    if raw.startswith('/'):
        # Absolute URL: the mobile app loads the library from the app shell,
        # so a bare local path resolves against the wrong origin (blank art).
        return 'https://stemscriber.com' + raw
    from urllib.parse import urlparse
    host = (urlparse(raw).hostname or '').lower()
    if host in _ALLOWED_THUMB_HOSTS:
        return 'https://stemscriber.com/api/thumbnail?url=' + quote(raw, safe='')
    return raw


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
    hidden_view = request.args.get('hidden', '').lower() == 'true'
    sort_mode = (request.args.get('sort') or 'recent').lower()
    if sort_mode not in ('recent', 'az'):
        sort_mode = 'recent'

    library = []

    for job_id, job in jobs.items():
        # From-Scratch Sessions (#30) are stem-less by design — keep them.
        is_session = bool(job.metadata.get('kind') == 'session') if job.metadata else False
        if job.status != 'completed' or (not job.stems and not is_session):
            continue

        # Demo songs are visible to everyone
        is_demo = job.metadata.get('demo', False) if job.metadata else False

        # Determine visibility. Admin sees only their own library by default;
        # use ?all=true to see everything across users.
        if is_demo:
            pass  # Demo songs always shown
        elif show_all:
            pass  # Explicit all flag (admin only)
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

        # Hidden songs live in the Hidden tab only (and vice versa)
        is_hidden = bool(job.metadata.get('hidden')) if job.metadata else False
        if hidden_view != is_hidden:
            continue

        library.append({
            'hidden': is_hidden,
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
            'kind': job.metadata.get('kind') if job.metadata else None,
            'source_url': job.source_url
        })

    # Sort: 'recent' = newest first with demos pinned to top.
    #       'az' = pure alphabetical; demos sort naturally (no pinning — per Jeff,
    #              he doesn't want his demo "U2 Apple Music" forced to the top).
    if sort_mode == 'az':
        library.sort(key=lambda x: (x['title'] or '').lower())
    else:
        # Pin demos first, then newest-first for the rest
        library.sort(key=lambda x: (0 if x['demo'] else 1, -float(x['created_at'] or 0)))

    return jsonify({
        'library': library,
        'total': len(library),
        'sort': sort_mode
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


@library_bp.route('/api/library/<job_id>', methods=['PATCH', 'POST'])
@auth_required
def update_library_item(job_id):
    """Update a library item: hide/unhide, or rename (title/artist).

    Body: any of {"hidden": bool, "title": str, "artist": str}.
    Renaming clears the cached track_info so the About panel + album art
    refetch against the corrected name.
    """
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    user = getattr(g, 'current_user', None)
    if not _is_admin():
        if not user:
            return jsonify({'error': 'Authentication required'}), 401
        session_id = request.cookies.get('session_id')
        if not _job_belongs_to_user(job, user, session_id) and job.user_id is not None:
            return jsonify({'error': 'You do not own this song'}), 403

    body = request.get_json(silent=True) or {}
    if job.metadata is None:
        job.metadata = {}
    changed = []

    if 'hidden' in body:
        job.metadata['hidden'] = bool(body['hidden'])
        changed.append('hidden')

    renamed = False
    if isinstance(body.get('title'), str) and body['title'].strip():
        job.metadata['title'] = body['title'].strip()[:200]
        changed.append('title')
        renamed = True
    if isinstance(body.get('artist'), str) and body['artist'].strip():
        job.metadata['artist'] = body['artist'].strip()[:120]
        changed.append('artist')
        renamed = True

    if renamed:
        # Stale bio/art belongs to the old (often junk-filename) identity.
        job.metadata.pop('track_info', None)
        # Refresh album art against the corrected name (cheap, best-effort).
        try:
            import json as _json, urllib.parse as _up, urllib.request as _ur
            _q = _up.quote(f"{job.metadata.get('artist','')} {job.metadata.get('title','')}".strip())
            _req = _ur.Request(f"https://itunes.apple.com/search?term={_q}&entity=song&limit=1",
                               headers={'User-Agent': 'StemScriber/1.0'})
            with _ur.urlopen(_req, timeout=6) as _r:
                _d = _json.loads(_r.read().decode())
            if _d.get('results'):
                _art = (_d['results'][0].get('artworkUrl100') or '').replace('100x100', '300x300')
                if _art:
                    job.metadata['thumbnail'] = _art
                    changed.append('thumbnail')
        except Exception as e:
            logger.debug(f"thumbnail refresh on rename failed: {e}")

    if not changed:
        return jsonify({'error': 'Nothing to update'}), 400

    try:
        save_job_to_disk(job)
    except Exception as e:
        logger.warning(f"save after library update failed for {job_id}: {e}")

    logger.info(f"Library update {job_id}: {changed} (user {getattr(user, 'id', 'admin')})")
    return jsonify({'status': 'ok', 'updated': changed,
                    'title': job.metadata.get('title'),
                    'artist': job.metadata.get('artist'),
                    'hidden': bool(job.metadata.get('hidden')),
                    'thumbnail': _thumb_url(job.metadata.get('thumbnail'))})


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
