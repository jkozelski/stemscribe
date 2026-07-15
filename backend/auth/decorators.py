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
PLAN_HIERARCHY = {'free': 0, 'beta': 1, 'premium': 2, 'pro': 2, 'lifetime': 3}

PLAN_LIMITS = {
    'free': {
        'songs_per_month': 5,
        'max_duration_sec': 300,     # 5 minutes
        'stems': 6,
        'chord_analysis': True,
        'midi_export': True,
        'tab_export': False,
        'priority_queue': False,
        'output_quality': '128kbps',
    },
    'pro': {
        # 30/mo base + buyable overage packs (10 songs / $5).
        # Margin math defended pre-launch: 30 base = 60% margin at $10
        # after ads/CAC; overage packs run ~76% margin on top.
        'songs_per_month': 30,
        'max_duration_sec': 1800,    # 30 minutes
        'stems': 6,
        'chord_analysis': True,
        'midi_export': True,
        'tab_export': True,
        'priority_queue': True,
        'output_quality': 'wav',
    },
    # 'premium' was a $20/mo tier dropped pre-launch (per stemscriber_full_state.md
    # pricing table — "Founder vanity tier with no product-shaped demand").
    # Stripe still has 'premium' SKUs from earlier sign-ups, and at least one
    # account (Jeff's, pre-bump) was marked premium. Without this entry, lookups
    # like PLAN_LIMITS[user.plan] raise KeyError and the enforce_plan_limits
    # decorator falls through to free's 5-songs/month cap. Mirror 'pro' so
    # existing premium subscribers get at least what they're paying for.
    'premium': {
        'songs_per_month': 50,
        'max_duration_sec': 900,     # 15 minutes
        'stems': 6,
        'chord_analysis': True,
        'midi_export': True,
        'tab_export': True,
        'priority_queue': True,
        'output_quality': 'wav',
    },
    'lifetime': {
        # Lifetime = same 50/mo cap as paid-historical premium, differentiated
        # by no-expiration access + 30min track length. Predictable Modal cost
        # ceiling even if comped friends go hard.
        'songs_per_month': 50,
        'max_duration_sec': 1800,    # 30 minutes
        'stems': 6,
        'chord_analysis': True,
        'midi_export': True,
        'tab_export': True,
        'priority_queue': True,
        'output_quality': 'wav',
    },
    'beta': {
        'songs_per_month': 30,        # capped to Pro level 2026-07-14 (was -1 unlimited; closed TEARITAPART/beta exploit, RISK-6)
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
    """Hash the client IP for anonymous usage tracking.

    Audit fix (2026-05-31): prefer CF-Connecting-IP (set by Cloudflare and
    trusted because the gunicorn box is only reachable via the Cloudflare
    tunnel), then fall back to remote_addr. Do NOT honor client-supplied
    X-Forwarded-For — attackers can set it to a different IP per request
    and land in a fresh anonymous quota bucket every time, defeating the
    free-tier monthly cap and burning Modal cost.
    """
    ip = request.headers.get('CF-Connecting-IP', '').strip() or request.remote_addr
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

    Authenticated users with extras_balance > 0 (purchased overage packs)
    can keep processing past the monthly cap; extras are consumed by
    consume_extras_if_needed() once the song actually completes.

    Raises RateLimitExceeded if limit + extras is exceeded.
    Returns the current usage count.
    """
    if user:
        plan = user.plan
        count = get_monthly_usage(str(user.id))
        extras = int(getattr(user, 'extras_balance', 0) or 0)
    else:
        plan = 'free'
        count = get_anonymous_monthly_usage(ip_hash or get_client_ip_hash())
        extras = 0

    limits = PLAN_LIMITS[plan]
    max_songs = limits['songs_per_month']

    if max_songs == -1:
        return count

    effective_limit = max_songs + extras
    if count >= effective_limit:
        raise RateLimitExceeded(
            f"You've used {count}/{max_songs} songs this month"
            + (f" (+ {extras} extras)" if extras else "")
            + ". Upgrade your plan or buy a 10-song pack to keep going.",
            plan=plan,
            usage_count=count,
            limit=effective_limit,
        )

    return count


def consume_extras_if_needed(user):
    """Called AFTER a song is queued successfully. If the user's monthly
    usage now exceeds their plan's base cap, decrement extras_balance by 1
    so the pack drains as it's actually used.

    Safe-no-op if user has no extras or is still under the base cap.
    Returns the new extras_balance for caller convenience.
    """
    if not user:
        return 0
    plan = user.plan
    base_cap = PLAN_LIMITS.get(plan, PLAN_LIMITS['free'])['songs_per_month']
    if base_cap == -1:
        return int(getattr(user, 'extras_balance', 0) or 0)

    count = get_monthly_usage(str(user.id))
    extras = int(getattr(user, 'extras_balance', 0) or 0)
    if count > base_cap and extras > 0:
        new_balance = extras - 1
        try:
            from auth.models import update_user_extras_balance
            update_user_extras_balance(str(user.id), new_balance)
        except Exception as e:
            logger.error(f"Failed to decrement extras for user {user.id}: {e}")
            return extras
        return new_balance
    return extras


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
