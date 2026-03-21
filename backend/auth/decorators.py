"""
Auth decorators for route protection and plan enforcement.

Usage:
    @api_bp.route('/some-endpoint')
    @jwt_required()            # built-in from flask_jwt_extended
    @require_plan('premium')   # custom: ensures user is on premium or pro
    def some_endpoint():
        user = get_current_user()
        ...
"""

import hashlib
import logging
from functools import wraps

from flask import jsonify, request
from flask_jwt_extended import get_jwt_identity, verify_jwt_in_request

from auth.models import get_user_by_id, get_monthly_usage, get_anonymous_monthly_usage

logger = logging.getLogger(__name__)

# Plan hierarchy: higher index = more permissions
PLAN_HIERARCHY = {'free': 0, 'beta': 1, 'premium': 1, 'pro': 2}

PLAN_LIMITS = {
    'free': {
        'songs_per_month': 3,
        'max_duration_sec': 300,     # 5 minutes
        'stems': 4,                  # vocal, drum, bass, other
        'chord_analysis': False,
        'midi_export': False,
        'tab_export': False,
        'priority_queue': False,
        'output_quality': '128kbps',
    },
    'premium': {
        'songs_per_month': 50,
        'max_duration_sec': 900,     # 15 minutes
        'stems': 6,                  # + guitar, piano
        'chord_analysis': True,
        'midi_export': True,
        'tab_export': False,
        'priority_queue': False,
        'output_quality': '320kbps',
    },
    'pro': {
        'songs_per_month': -1,       # unlimited
        'max_duration_sec': 1800,    # 30 minutes
        'stems': 6,
        'chord_analysis': True,
        'midi_export': True,
        'tab_export': True,
        'priority_queue': True,
        'output_quality': 'wav',
    },
    'beta': {
        'songs_per_month': -1,       # unlimited (same as pro)
        'max_duration_sec': 1800,    # 30 minutes
        'stems': 6,
        'chord_analysis': True,
        'midi_export': True,
        'tab_export': True,
        'priority_queue': True,
        'output_quality': 'wav',
    },
}


class RateLimitExceeded(Exception):
    """Raised when a user exceeds their plan's rate limit."""
    def __init__(self, message, plan, usage_count, limit):
        super().__init__(message)
        self.plan = plan
        self.usage_count = usage_count
        self.limit = limit


def get_current_user():
    """Get the authenticated User object, or None if not authenticated."""
    try:
        verify_jwt_in_request(optional=True)
        user_id = get_jwt_identity()
        if user_id:
            return get_user_by_id(user_id)
    except Exception:
        pass
    return None


def get_client_ip_hash():
    """Hash the client IP for anonymous usage tracking."""
    ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if ip and ',' in ip:
        ip = ip.split(',')[0].strip()
    return hashlib.sha256((ip or 'unknown').encode()).hexdigest()[:16]


def require_plan(minimum_plan):
    """Decorator: require user to be on at least the given plan.

    Usage:
        @require_plan('premium')
        def premium_endpoint():
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = get_current_user()
            if not user:
                return jsonify({'error': 'Authentication required'}), 401
            user_level = PLAN_HIERARCHY.get(user.plan, 0)
            required_level = PLAN_HIERARCHY.get(minimum_plan, 0)
            if user_level < required_level:
                return jsonify({
                    'error': f'This feature requires a {minimum_plan} plan or higher',
                    'current_plan': user.plan,
                    'required_plan': minimum_plan,
                    'upgrade_url': '/pricing',
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def check_rate_limit(user=None, ip_hash=None):
    """Check if user/anonymous has exceeded their monthly song limit.

    Raises RateLimitExceeded if limit is exceeded.
    Returns the current usage count.
    """
    if user:
        plan = user.plan
        count = get_monthly_usage(str(user.id))
    else:
        plan = 'free'
        count = get_anonymous_monthly_usage(ip_hash or get_client_ip_hash())

    limits = PLAN_LIMITS[plan]
    max_songs = limits['songs_per_month']

    if max_songs != -1 and count >= max_songs:
        raise RateLimitExceeded(
            f"You've used {count}/{max_songs} songs this month. "
            f"Upgrade your plan to process more.",
            plan=plan,
            usage_count=count,
            limit=max_songs,
        )

    return count


def check_duration_limit(duration_seconds, user=None):
    """Check if audio duration exceeds the user's plan limit.

    Returns (allowed: bool, max_seconds: int).
    """
    plan = user.plan if user else 'free'
    limits = PLAN_LIMITS[plan]
    max_sec = limits['max_duration_sec']
    return duration_seconds <= max_sec, max_sec


def get_plan_limits(plan='free'):
    """Return the limits dict for a given plan."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])
