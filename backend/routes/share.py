"""
Share-with-student management endpoints.

POST   /api/jobs/<job_id>/share                — create a share token (returns raw token)
GET    /api/jobs/<job_id>/share                — list active share tokens for this job
DELETE /api/jobs/<job_id>/share/<token>        — revoke a share token

The token is then used in any per-job route as ?share=<token>. authorize_job_access()
in auth/middleware.py checks for it and grants read access if valid (and viewer cap
not exceeded).
"""

import logging
from flask import Blueprint, request, jsonify, g

from auth.middleware import auth_required, authorize_job_access, forbidden_response
from auth.share_tokens import (
    create_share_token, revoke_share_token, list_shares_for_job,
    DEFAULT_MAX_ANONYMOUS_VIEWERS,
)
from middleware.validation import validate_job_id
from models.job import get_job

logger = logging.getLogger(__name__)
share_bp = Blueprint("share", __name__)


# Monthly share-creation cap per plan. Prevents one paid user from becoming
# a free-CDN for hundreds of non-paying viewers via repeated share links.
# Revoked shares still count — the abuse vector is "create → send → revoke →
# repeat", so the cap must be on creation count, not active count. Resets on
# the 1st of each month (calendar-month, UTC).
SHARE_MONTHLY_CAPS = {
    'free': 2,
    'pro': 20,
    'lifetime': 30,
}
SHARE_DEFAULT_CAP = 2  # for unknown / null plan values


def _shares_this_month(user_id: str) -> int:
    """Count share tokens this user has created since the 1st of the current
    calendar month (UTC). Includes revoked shares."""
    from db import query_one
    row = query_one(
        """SELECT COUNT(*) AS n FROM job_share_tokens
           WHERE created_by_user_id = %s
             AND created_at >= date_trunc('month', NOW() AT TIME ZONE 'UTC')""",
        (user_id,),
    )
    return int((row or {}).get('n') or 0)


@share_bp.route('/api/jobs/<job_id>/share', methods=['POST'])
@auth_required
def create_share(job_id):
    """Create a share token for this job. Caller must own the job.
    Body (optional): {"max_anonymous_viewers": 3}
    Returns: {"token": "...", "url_param": "?share=...", "max_anonymous_viewers": 3}
    """
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    # Demo bypass is read-only; share-creation requires real ownership.
    if not authorize_job_access(job, allow_demo=False):
        return forbidden_response()

    user = g.current_user

    # ---- Monthly share-cap check (plan-based) -------------------------------
    plan = (getattr(user, 'plan', None) or 'free').lower()
    monthly_cap = SHARE_MONTHLY_CAPS.get(plan, SHARE_DEFAULT_CAP)
    used = _shares_this_month(str(user.id))
    if used >= monthly_cap:
        return jsonify({
            'error': f'You have used all {monthly_cap} of your monthly shares on the {plan.capitalize()} plan. Resets on the 1st.',
            'code': 'share_cap_reached',
            'used': used,
            'limit': monthly_cap,
            'plan': plan,
        }), 429

    data = request.get_json(silent=True) or {}
    cap = int(data.get('max_anonymous_viewers') or DEFAULT_MAX_ANONYMOUS_VIEWERS)
    cap = max(1, min(cap, 25))  # sanity clamp

    token = create_share_token(job_id, str(user.id), max_anonymous_viewers=cap)
    return jsonify({
        'token': token,
        'url_param': f'?share={token}',
        'max_anonymous_viewers': cap,
        'shares_used_this_month': used + 1,
        'shares_limit_this_month': monthly_cap,
    }), 201


@share_bp.route('/api/jobs/<job_id>/share', methods=['GET'])
@auth_required
def list_shares(job_id):
    """List active share tokens for this job (caller must own it)."""
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job, allow_demo=False):
        return forbidden_response()

    user = g.current_user
    shares = list_shares_for_job(job_id, str(user.id))
    return jsonify({'shares': shares})


@share_bp.route('/api/jobs/<job_id>/share/<token>', methods=['DELETE'])
@auth_required
def revoke_share(job_id, token):
    """Revoke a share token. Only the creator can revoke."""
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    user = g.current_user
    if revoke_share_token(token, str(user.id)):
        return jsonify({'ok': True})
    return jsonify({'error': 'Token not found or not owned by you'}), 404
