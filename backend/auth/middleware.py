"""
Auth middleware — adaptive authentication decorator.

In production (DATABASE_URL set + JWT configured):
    Enforces JWT authentication on protected routes.
    Optionally enforces plan-based rate limits.

In development (no DATABASE_URL):
    Allows anonymous access with a warning log.
    Plan limits are not enforced.

Usage:
    from auth.middleware import auth_required

    @app.route('/api/upload')
    @auth_required          # Requires login in production, open in dev
    def upload():
        ...

    @app.route('/api/upload')
    @auth_required(optional=True)  # Identifies user if logged in, doesn't block
    def upload():
        ...
"""

import os
import logging
from functools import wraps

from flask import jsonify, g, request

logger = logging.getLogger(__name__)

# Detect whether auth infrastructure is available
_AUTH_AVAILABLE = False
_DB_AVAILABLE = bool(os.environ.get('DATABASE_URL'))

try:
    if _DB_AVAILABLE:
        from flask_jwt_extended import jwt_required as _jwt_required, get_jwt_identity, verify_jwt_in_request
        from auth.models import get_user_by_id
        _AUTH_AVAILABLE = True
except ImportError:
    pass


def auth_required(fn=None, *, optional=False):
    """Decorator: enforce JWT auth in production, allow anonymous in dev.

    Args:
        optional: If True, identify user if token present but don't block anonymous.
                  If False (default), require valid JWT in production.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            g.current_user = None

            if not _AUTH_AVAILABLE:
                # No DB/auth — dev mode, allow through
                return fn(*args, **kwargs)

            if optional:
                # Distinguish "no token" from "bad/expired token".
                # No token: legitimate anonymous user — fall through.
                # Bad/expired token: a signed-in user whose JWT went stale —
                # MUST 401, not silently downgrade to anonymous, otherwise
                # rate-limiting collapses (anonymous = free-tier IP buckets
                # which are basically unbounded for a typical user).
                # Bug found 2026-05-27 when a Pro user blew past the cap
                # because their access_token expired mid-session.
                auth_hdr = (request.headers.get('Authorization') or '').strip()
                has_bearer = auth_hdr.lower().startswith('bearer ')
                try:
                    verify_jwt_in_request(optional=True)
                    user_id = get_jwt_identity()
                    if user_id:
                        g.current_user = get_user_by_id(user_id)
                except Exception as e:
                    if has_bearer:
                        return jsonify({
                            'error': 'Your sign-in expired. Please refresh the page and sign in again.',
                            'code': 'token_expired',
                            'details': str(e),
                        }), 401
                    # No bearer: legitimate anonymous, swallow + continue
                return fn(*args, **kwargs)

            # Required auth — enforce JWT
            try:
                verify_jwt_in_request()
                user_id = get_jwt_identity()
                user = get_user_by_id(user_id)
                if not user:
                    return jsonify({'error': 'User not found'}), 401
                g.current_user = user
            except Exception as e:
                return jsonify({
                    'error': 'Authentication required',
                    'code': 'missing_token',
                    'details': str(e),
                }), 401

            return fn(*args, **kwargs)
        return wrapper

    if fn is not None:
        # Called as @auth_required without parentheses
        return decorator(fn)
    # Called as @auth_required() or @auth_required(optional=True)
    return decorator


# ===== Per-job authorization helper (2026-05-31, audit finding CRITICAL #1) =====
# Job UUIDs are unguessable but they LEAK — screenshots, support tickets, browser
# history, Plausible referrers, our own logger.info() calls. Without this check,
# anyone who learns a UUID can download the owner's stems, MIDI, chord chart,
# track metadata. authorize_job_access() must be called on every route that
# returns or mutates per-job data.
#
# Allowed paths:
#   1. Admin (jkozelski@gmail.com)
#   2. User owns the job (job.user_id matches authed user OR legacy metadata.user_id)
#   3. Anonymous upload from same browser (session_id cookie matches job.session_id)
#   4. Job is a public demo (job.metadata.demo == True) — for the home-page demo song
#   5. Valid share token for this job (?share=<token>, share-with-student feature)
#
# Caller must run @auth_required(optional=True) first so g.current_user is populated.

_ADMIN_EMAILS = {'jkozelski@gmail.com'}


def authorize_job_access(job, *, allow_demo=True):
    """Return True if the current request is allowed to access this job's data.
    Caller should 403 on False. Assumes @auth_required(optional=True) ran first."""
    if not job:
        return False
    user = getattr(g, 'current_user', None)
    # 1. Admin
    if user and getattr(user, 'email', None) in _ADMIN_EMAILS:
        return True
    # 2. Owner
    if user:
        uid = str(user.id)
        if getattr(job, 'user_id', None) == uid:
            return True
        meta = getattr(job, 'metadata', None) or {}
        if isinstance(meta, dict) and meta.get('user_id') == uid:
            return True
    # 3. Anonymous session match
    try:
        session_id = request.cookies.get('session_id')
    except Exception:
        session_id = None
    if session_id and getattr(job, 'session_id', None) == session_id:
        return True
    # 4. Public demo
    if allow_demo:
        meta = getattr(job, 'metadata', None) or {}
        if isinstance(meta, dict) and meta.get('demo'):
            return True
    # 5. Valid share token (?share=<token>) — share-with-student. Read-only;
    #    callers that mutate should pass allow_demo=False AND this check is
    #    also skipped for mutations (share tokens never authorize PUT/DELETE).
    if allow_demo:  # piggyback: share is read-only just like demo
        try:
            from auth.share_tokens import verify_share_token
            share = request.args.get('share', '').strip()
            if share:
                ok, _reason = verify_share_token(share, getattr(job, 'job_id', None), request)
                if ok:
                    return True
        except Exception as e:
            logger.warning(f"share-token check failed: {e}")
    return False


def forbidden_response():
    """Standard 403 for unauthorized job access."""
    return jsonify({'error': 'Forbidden'}), 403
