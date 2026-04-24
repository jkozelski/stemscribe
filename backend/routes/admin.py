"""
Admin dashboard — single-user metrics endpoint.

Gated by auth + hardcoded email allowlist (Jeff only). No PII in responses.
"""

import logging
from functools import wraps

from flask import Blueprint, jsonify, g

from auth.middleware import auth_required
from db import query_all, query_one

logger = logging.getLogger(__name__)

admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

# Single-user allowlist. This dashboard is NOT for general users.
ADMIN_EMAILS = {'jkozelski@gmail.com'}

ACTIVE_JOB_STATUSES = ('pending', 'uploading', 'separating', 'transcribing')


def _admin_only(fn):
    """Require an authenticated user whose email is in ADMIN_EMAILS."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = getattr(g, 'current_user', None)
        email = getattr(user, 'email', None) if user else None
        if not user or email not in ADMIN_EMAILS:
            return jsonify({'error': 'Forbidden'}), 403
        return fn(*args, **kwargs)
    return wrapper


def _date_key(row, col='day'):
    v = row.get(col)
    return v.isoformat() if hasattr(v, 'isoformat') else str(v)


def _signups_per_day():
    rows = query_all(
        """
        SELECT DATE(created_at) AS day, COUNT(*) AS cnt
        FROM users
        WHERE created_at >= NOW() - INTERVAL '30 days'
        GROUP BY day
        ORDER BY day
        """
    )
    return [{'day': _date_key(r), 'count': int(r['cnt'])} for r in rows]


def _cumulative_users():
    rows = query_all(
        """
        SELECT DATE(created_at) AS day, COUNT(*) AS cnt
        FROM users
        GROUP BY day
        ORDER BY day
        """
    )
    out = []
    running = 0
    for r in rows:
        running += int(r['cnt'])
        out.append({'day': _date_key(r), 'total': running})
    return out


def _songs_per_day():
    rows = query_all(
        """
        SELECT DATE(created_at) AS day, COUNT(*) AS cnt
        FROM jobs
        WHERE status = 'complete'
          AND created_at >= NOW() - INTERVAL '30 days'
        GROUP BY day
        ORDER BY day
        """
    )
    return [{'day': _date_key(r), 'count': int(r['cnt'])} for r in rows]


def _daily_active_users():
    rows = query_all(
        """
        SELECT DATE(created_at) AS day, COUNT(DISTINCT user_id) AS cnt
        FROM jobs
        WHERE user_id IS NOT NULL
          AND created_at >= NOW() - INTERVAL '30 days'
        GROUP BY day
        ORDER BY day
        """
    )
    return [{'day': _date_key(r), 'count': int(r['cnt'])} for r in rows]


def _peak_hours():
    rows = query_all(
        """
        SELECT EXTRACT(HOUR FROM created_at)::int AS hour, COUNT(*) AS cnt
        FROM jobs
        WHERE created_at >= NOW() - INTERVAL '30 days'
        GROUP BY hour
        ORDER BY hour
        """
    )
    by_hour = {int(r['hour']): int(r['cnt']) for r in rows}
    return [{'hour': h, 'count': by_hour.get(h, 0)} for h in range(24)]


def _plan_breakdown():
    rows = query_all(
        """
        SELECT COALESCE(plan, 'unknown') AS plan, COUNT(*) AS cnt
        FROM users
        GROUP BY plan
        ORDER BY cnt DESC
        """
    )
    buckets = {'free': 0, 'paid': 0}
    by_plan = {}
    for r in rows:
        plan = r['plan']
        cnt = int(r['cnt'])
        by_plan[plan] = cnt
        if plan == 'free':
            buckets['free'] += cnt
        else:
            buckets['paid'] += cnt
    return {'by_plan': by_plan, 'free_vs_paid': buckets}


def _summary_cards():
    queue_row = query_one(
        "SELECT COUNT(*) AS cnt FROM jobs WHERE status = ANY(%s)",
        (list(ACTIVE_JOB_STATUSES),),
    )
    users_row = query_one("SELECT COUNT(*) AS cnt FROM users")
    jobs_row = query_one(
        "SELECT COUNT(*) AS cnt FROM jobs WHERE status = 'complete'"
    )
    return {
        'queue_depth': int(queue_row['cnt']) if queue_row else 0,
        'total_users': int(users_row['cnt']) if users_row else 0,
        'total_jobs_complete': int(jobs_row['cnt']) if jobs_row else 0,
        # Modal spend is not tracked in the DB; surfaced as null so UI can hide.
        'modal_spend_this_month': None,
    }


@admin_bp.route('/metrics', methods=['GET'])
@auth_required
@_admin_only
def metrics():
    try:
        payload = {
            'summary': _summary_cards(),
            'signups_per_day': _signups_per_day(),
            'cumulative_users': _cumulative_users(),
            'songs_per_day': _songs_per_day(),
            'daily_active_users': _daily_active_users(),
            'peak_hours': _peak_hours(),
            'plans': _plan_breakdown(),
        }
        return jsonify(payload), 200
    except Exception as e:
        logger.exception("admin metrics query failed")
        return jsonify({'error': 'metrics query failed', 'details': str(e)}), 500
