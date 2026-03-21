"""
Rate limiting middleware — Flask-Limiter + plan-based song/duration limits.

Two layers of rate limiting:

1. **Request-level** (Flask-Limiter):
   - Global: 60 requests/min per IP
   - Auth endpoints: 5 requests/min per IP (brute-force protection)
   - These prevent API abuse regardless of plan tier.

2. **Plan-level** (song count + duration):
   - Enforced per-request via decorators applied to processing routes.
   - Free: 3 songs/month, 5 min max duration
   - Premium: 50 songs/month, 15 min max
   - Pro: unlimited, 30 min max
   - Anonymous users tracked by IP hash.

Usage:
    # In app factory:
    from middleware.rate_limit import init_limiter
    init_limiter(app)

    # On individual routes:
    from middleware.rate_limit import limiter, enforce_plan_limits
    @app.route('/api/process', methods=['POST'])
    @enforce_plan_limits
    def process():
        ...
"""

import logging
from functools import wraps

from flask import jsonify, request, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask-Limiter instance (request-level rate limiting)
# ---------------------------------------------------------------------------

def _get_key():
    """Extract client IP for Flask-Limiter keying.

    Priority: CF-Connecting-IP (Cloudflare Tunnel) > X-Forwarded-For > remote_addr.
    """
    # Cloudflare Tunnel sets this to the real visitor IP
    cf_ip = request.headers.get('CF-Connecting-IP', '').strip()
    if cf_ip:
        return cf_ip
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return get_remote_address()

limiter = Limiter(
    
    key_func=_get_key,
    default_limits=["60 per minute"],
    storage_uri="memory://",
    strategy="fixed-window",
)

# Pre-built rate limit strings for common endpoint groups
AUTH_LIMIT = "5 per minute"
WEBHOOK_LIMIT = "30 per minute"
PROCESSING_LIMIT = "10 per minute"
UPLOAD_LIMIT = "5 per minute"         # /api/url, /api/upload — expensive GPU work
SONGSTERR_LIMIT = "30 per minute"     # /api/songsterr/*
LIBRARY_LIMIT = "60 per minute"       # /api/library
BETA_LIMIT = "10 per minute"          # /api/beta/*
SMS_LIMIT = "10 per minute"           # /api/sms/*

def init_limiter(app):
    """Attach Flask-Limiter to the Flask app.

    Call this in the app factory after creating the Flask instance.
    Registers a custom 429 error handler.
    """
    limiter.init_app(app)

    @app.errorhandler(429)
    def rate_limit_exceeded(e):
        return jsonify({
            'error': 'Rate limit exceeded. Please try again in a moment.',
            'retry_after': e.description,
        }), 429

    logger.info("Flask-Limiter initialized (60 req/min default)")

# ---------------------------------------------------------------------------
# Plan-based enforcement decorator
# ---------------------------------------------------------------------------

def enforce_plan_limits(fn):
    """Decorator: check monthly song quota and duration limit before processing.

    Must be applied AFTER any @jwt_required() or authentication decorator,
    so that g.current_user is available (or None for anonymous).

    Sets g.plan_limits for downstream use if needed.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from auth.decorators import (
            get_current_user,
            get_client_ip_hash,
            check_rate_limit,
            get_plan_limits,
            RateLimitExceeded,
            PLAN_LIMITS,
        )

        user = get_current_user()
        ip_hash = get_client_ip_hash() if not user else None

        # Store on g for downstream access
        g.current_user = user
        g.ip_hash = ip_hash
        plan = user.plan if user else 'free'
        g.plan_limits = get_plan_limits(plan)

        # --- Song quota check ---
        try:
            usage_count = check_rate_limit(user=user, ip_hash=ip_hash)
            g.usage_count = usage_count
        except RateLimitExceeded as e:
            _limits = PLAN_LIMITS[e.plan]
            return jsonify({
                'error': str(e),
                'usage': e.usage_count,
                'limit': e.limit,
                'plan': e.plan,
                'upgrade_url': '/pricing',
            }), 429

        return fn(*args, **kwargs)
    return wrapper

def enforce_duration_limit(duration_seconds):
    """Check audio duration against the current user's plan.

    Call this after probing the audio file. Returns a JSON error response
    tuple if the duration exceeds the limit, or None if OK.

    Usage:
        error = enforce_duration_limit(duration_seconds)
        if error:
            return error
    """
    from auth.decorators import check_duration_limit, get_current_user

    user = getattr(g, 'current_user', None) or get_current_user()
    allowed, max_sec = check_duration_limit(duration_seconds, user=user)
    if not allowed:
        plan = user.plan if user else 'free'
        return jsonify({
            'error': (
                f'Audio duration ({int(duration_seconds)}s) exceeds your plan limit '
                f'({max_sec}s / {max_sec // 60} min). Upgrade for longer songs.'
            ),
            'duration': duration_seconds,
            'max_duration': max_sec,
            'plan': plan,
            'upgrade_url': '/pricing',
        }), 413

    return None

def record_usage_event(user=None, ip_hash=None, job_id=None, action='separation'):
    """Record a usage event after successful processing.

    Should be called after the job is accepted (not after completion)
    to prevent the user from spamming requests while one is processing.
    """
    from auth.models import record_usage

    user_id = str(user.id) if user else None
    record_usage(
        user_id=user_id,
        anonymous_ip_hash=ip_hash,
        job_id=job_id,
        action=action,
    )
    logger.debug(
        f"Usage recorded: user={user_id or 'anon'}, ip={ip_hash or 'n/a'}, "
        f"job={job_id}, action={action}"
    )
