"""
Beta invite system — simple code-based access for beta testers.

No database required. Codes are stored in a JSON file on disk.
"""

import os
import json
import logging
import secrets
from pathlib import Path
from datetime import datetime

from flask import Blueprint, request, jsonify, g

from auth.middleware import auth_required
from auth.models import update_user_plan
from middleware.validation import validate_beta_code, sanitize_text

logger = logging.getLogger(__name__)

beta_bp = Blueprint("beta", __name__)

# Store beta codes in a JSON file next to the app
BETA_CODES_FILE = Path(__file__).parent.parent / 'beta_codes.json'

# Master admin key for generating codes (set via env var or use default for dev)
BETA_ADMIN_KEY = os.environ.get('BETA_ADMIN_KEY', 'stemscribe-beta-admin-2026')


def _load_codes():
    """Load beta codes from disk."""
    if BETA_CODES_FILE.exists():
        return json.loads(BETA_CODES_FILE.read_text())
    return {}


def _save_codes(codes):
    """Save beta codes to disk."""
    BETA_CODES_FILE.write_text(json.dumps(codes, indent=2))


@beta_bp.route('/api/beta/generate', methods=['POST'])
def generate_beta_codes():
    """
    Generate beta invite codes.

    Body (JSON):
        admin_key: Admin authentication key
        count: Number of codes to generate (default: 5, max: 50)
        label: Optional label for this batch (e.g., "musician-friends")

    Returns list of generated codes.
    """
    data = request.get_json(silent=True) or {}

    admin_key = data.get('admin_key', '')
    if admin_key != BETA_ADMIN_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    count = min(int(data.get('count', 5)), 50)
    label = data.get('label', 'beta')

    codes = _load_codes()
    new_codes = []

    for _ in range(count):
        # Generate readable codes like STEM-XXXX-XXXX
        code = f"STEM-{secrets.token_hex(2).upper()}-{secrets.token_hex(2).upper()}"
        codes[code] = {
            'created': datetime.utcnow().isoformat(),
            'label': label,
            'redeemed': False,
            'redeemed_by': None,
            'redeemed_at': None,
        }
        new_codes.append(code)

    _save_codes(codes)
    logger.info(f"Generated {count} beta codes (label: {label})")

    return jsonify({
        'codes': new_codes,
        'count': len(new_codes),
        'label': label,
        'total_codes': len(codes),
    })


@beta_bp.route('/api/beta/redeem', methods=['POST'])
@auth_required(optional=True)
def redeem_beta_code():
    """
    Redeem a beta invite code.

    Body (JSON):
        code: The invite code
        name: Tester's name (optional)
        email: Tester's email (optional)

    If authenticated, updates the user's plan to 'beta'.
    If anonymous, still validates the code (backwards compatible).

    Returns success + unlocked features info.
    """
    data = request.get_json(silent=True) or {}

    code, code_error = validate_beta_code(data.get('code', ''))
    if code_error:
        return jsonify({'error': code_error}), 400

    codes = _load_codes()

    if code not in codes:
        return jsonify({'error': 'Invalid invite code'}), 404

    code_data = codes[code]

    if code_data['redeemed']:
        # Allow re-use — beta codes are for friends, not strangers
        # Still update user plan if authenticated
        if g.current_user:
            update_user_plan(str(g.current_user.id), 'beta')
            logger.info(f"Beta plan applied to user {g.current_user.id} (re-redeemed code {code})")
        return jsonify({
            'valid': True,
            'already_redeemed': True,
            'message': 'Welcome back! This code was already activated.',
            'plan': 'beta',
            'features': _beta_features(),
            'plan_updated': g.current_user is not None,
        })

    # Mark as redeemed
    code_data['redeemed'] = True
    code_data['redeemed_by'] = sanitize_text(data.get('name', data.get('email', 'anonymous')), max_length=200)
    code_data['redeemed_at'] = datetime.utcnow().isoformat()

    # Link to user account if authenticated
    if g.current_user:
        code_data['user_id'] = str(g.current_user.id)
        update_user_plan(str(g.current_user.id), 'beta')
        logger.info(f"Beta code redeemed: {code} by user {g.current_user.id}")
    else:
        logger.info(f"Beta code redeemed: {code} by {code_data['redeemed_by']} (anonymous)")

    _save_codes(codes)

    return jsonify({
        'valid': True,
        'message': 'Welcome to StemScriber Beta!',
        'plan': 'beta',
        'features': _beta_features(),
        'plan_updated': g.current_user is not None,
    })


@beta_bp.route('/api/beta/validate', methods=['GET'])
def validate_beta_code():
    """
    Quick validation of a beta code (GET for simplicity).

    Query param: code
    """
    code, code_error = validate_beta_code(request.args.get('code', ''))
    if code_error:
        return jsonify({'valid': False}), 400

    codes = _load_codes()
    if code in codes:
        return jsonify({'valid': True, 'plan': 'beta'})

    return jsonify({'valid': False}), 404


@beta_bp.route('/api/beta/stats', methods=['GET'])
def beta_stats():
    """
    Get beta program stats (admin only).

    Query param: admin_key
    """
    admin_key = request.args.get('admin_key', '')
    if admin_key != BETA_ADMIN_KEY:
        return jsonify({'error': 'Unauthorized'}), 401

    codes = _load_codes()
    total = len(codes)
    redeemed = sum(1 for c in codes.values() if c['redeemed'])

    return jsonify({
        'total_codes': total,
        'redeemed': redeemed,
        'unredeemed': total - redeemed,
        'codes': {k: v for k, v in codes.items()},
    })


def _beta_features():
    """Features unlocked for beta testers."""
    return {
        'songs_per_month': 50,
        'max_duration_sec': 600,
        'stems': 6,
        'chord_detection': True,
        'gp_tabs': True,
        'stereo_split': True,
        'ensemble_mode': True,
    }
