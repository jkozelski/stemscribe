"""
Auth Blueprint — JWT authentication endpoints.

Endpoints:
    POST /auth/register        — Create account
    POST /auth/login           — Get tokens
    POST /auth/refresh         — Refresh access token
    POST /auth/logout          — Revoke refresh token
    POST /auth/forgot-password — Send reset email
    POST /auth/reset-password  — Set new password with token
    GET  /auth/me              — Get current user profile
"""

import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    jwt_required,
    get_jwt_identity,
    get_jwt,
    set_refresh_cookies,
    unset_jwt_cookies,
)
from email_validator import validate_email, EmailNotValidError

from auth.models import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    update_user_password,
    get_monthly_usage,
)
from auth.email import send_reset_email, verify_reset_token
from auth.decorators import get_plan_limits
from db import execute, query_one

logger = logging.getLogger(__name__)

auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


# ---- Helper ----

def _issue_tokens(user):
    """Create access + refresh tokens for a user, return response dict."""
    identity = str(user.id)
    access_token = create_access_token(identity=identity)
    refresh_token = create_refresh_token(identity=identity)
    return {
        'access_token': access_token,
        'refresh_token': refresh_token,
        'user': user.to_dict(),
    }


# ---- Endpoints ----

@auth_bp.route('/register', methods=['POST'])
def register():
    """Create a new user account."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    email = data.get('email', '').strip()
    password = data.get('password', '')
    display_name = data.get('display_name', '').strip() or None

    # Validate email format
    try:
        validated = validate_email(email, check_deliverability=False)
        email = validated.normalized
    except EmailNotValidError as e:
        return jsonify({'error': f'Invalid email: {e}'}), 400

    # Create user (raises ValueError if email taken or bad password)
    try:
        user = create_user(email, password, display_name)
    except ValueError as e:
        return jsonify({'error': str(e)}), 409

    tokens = _issue_tokens(user)
    response = jsonify(tokens)
    set_refresh_cookies(response, tokens['refresh_token'])
    return response, 201


@auth_bp.route('/login', methods=['POST'])
def login():
    """Authenticate with email + password."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    email = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    user = get_user_by_email(email)
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401

    tokens = _issue_tokens(user)
    response = jsonify(tokens)
    set_refresh_cookies(response, tokens['refresh_token'])
    return response, 200


@auth_bp.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    """Get a new access token using the refresh token cookie."""
    identity = get_jwt_identity()
    user = get_user_by_id(identity)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    access_token = create_access_token(identity=identity)
    return jsonify({
        'access_token': access_token,
        'user': user.to_dict(),
    }), 200


@auth_bp.route('/logout', methods=['POST'])
@jwt_required(verify_type=False)
def logout():
    """Revoke the current token (access or refresh) by adding its JTI to the blacklist."""
    jwt_data = get_jwt()
    jti = jwt_data['jti']
    exp = datetime.fromtimestamp(jwt_data['exp'], tz=timezone.utc)

    execute(
        "INSERT INTO token_blacklist (jti, expires_at) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        (jti, exp),
    )

    response = jsonify({'message': 'Logged out'})
    unset_jwt_cookies(response)
    return response, 200


@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Send a password reset email."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    email = data.get('email', '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    # Always return success to prevent email enumeration
    user = get_user_by_email(email)
    if user:
        send_reset_email(email, str(user.id))

    return jsonify({'message': 'If that email exists, a reset link has been sent'}), 200


@auth_bp.route('/reset-password', methods=['POST'])
def reset_password():
    """Reset password using a token from the reset email."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    token = data.get('token', '')
    new_password = data.get('password', '')

    if not token:
        return jsonify({'error': 'Reset token is required'}), 400

    user_id = verify_reset_token(token)
    if not user_id:
        return jsonify({'error': 'Invalid or expired reset token'}), 400

    try:
        update_user_password(user_id, new_password)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    return jsonify({'message': 'Password updated successfully'}), 200


@auth_bp.route('/me', methods=['GET'])
@jwt_required()
def me():
    """Get the current user's profile and usage info."""
    identity = get_jwt_identity()
    user = get_user_by_id(identity)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    usage_count = get_monthly_usage(str(user.id))
    limits = get_plan_limits(user.plan)

    return jsonify({
        'user': user.to_dict(),
        'usage': {
            'songs_this_month': usage_count,
            'songs_limit': limits['songs_per_month'],
            'plan_limits': limits,
        },
    }), 200


# ---- JWT Token Blacklist Check ----
# This callback is registered on the JWTManager in jwt_setup.py

def check_if_token_revoked(jwt_header, jwt_payload):
    """Check if a token's JTI is in the blacklist. Used by @jwt.token_in_blocklist_loader."""
    jti = jwt_payload['jti']
    row = query_one("SELECT jti FROM token_blacklist WHERE jti = %s", (jti,))
    return row is not None
