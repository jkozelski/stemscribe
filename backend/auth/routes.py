"""
Auth Blueprint — JWT authentication endpoints (Google-only).

Endpoints:
    POST /auth/google          — Authenticate or register via Google Sign-In
    POST /auth/refresh         — Refresh access token
    POST /auth/logout          — Revoke refresh token
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

from auth.models import (
    create_user,
    get_user_by_email,
    get_user_by_id,
    get_user_by_google_id,
    create_google_user,
    link_google_account,
    update_user_password,
    get_monthly_usage,
)
from auth.email import send_reset_email, verify_reset_token
from auth.decorators import get_plan_limits
from db import execute, query_one

import os

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
# NOTE: /auth/register, /auth/login, /auth/forgot-password, /auth/reset-password
# have been removed. Authentication is now Google-only.
# Existing email+password users can sign in via Google with the same email
# (the google_login endpoint handles account linking automatically — see Case 2).


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


# ---- Google OAuth ----

@auth_bp.route('/google', methods=['POST'])
def google_login():
    """Authenticate or register via Google Sign-In ID token."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400

    credential = data.get('credential', '')
    if not credential:
        return jsonify({'error': 'Google credential token is required'}), 400

    google_client_id = os.environ.get('GOOGLE_CLIENT_ID')
    if not google_client_id:
        logger.error("GOOGLE_CLIENT_ID not configured")
        return jsonify({'error': 'Google Sign-In is not configured'}), 500

    # Verify the Google ID token
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests

        idinfo = google_id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            google_client_id,
        )
    except ValueError as e:
        logger.warning(f"Google token verification failed: {e}")
        return jsonify({'error': 'Invalid Google token'}), 401

    # Extract user info from verified token
    google_id = idinfo['sub']
    email = idinfo.get('email', '').lower().strip()
    name = idinfo.get('name', '')
    picture = idinfo.get('picture')

    if not email:
        return jsonify({'error': 'Google account has no email'}), 400

    # Case 1: User exists with this google_id — log in and refresh avatar
    user = get_user_by_google_id(google_id)
    if user:
        if picture:
            link_google_account(str(user.id), google_id, picture)
            user = get_user_by_id(str(user.id))
        tokens = _issue_tokens(user)
        response = jsonify(tokens)
        set_refresh_cookies(response, tokens['refresh_token'])
        logger.info(f"Google login: existing user {user.id} ({email})")
        return response, 200

    # Case 2: User exists with this email but no google_id — link & log in
    user = get_user_by_email(email)
    if user:
        link_google_account(str(user.id), google_id, picture)
        # Re-fetch to get updated fields
        user = get_user_by_id(str(user.id))
        tokens = _issue_tokens(user)
        response = jsonify(tokens)
        set_refresh_cookies(response, tokens['refresh_token'])
        logger.info(f"Google login: linked Google to existing user {user.id} ({email})")
        return response, 200

    # Case 3: New user — create account with Google info (no password)
    try:
        user = create_google_user(email, name, google_id, picture)
    except ValueError as e:
        return jsonify({'error': str(e)}), 409

    tokens = _issue_tokens(user)
    response = jsonify(tokens)
    set_refresh_cookies(response, tokens['refresh_token'])
    logger.info(f"Google login: created new user {user.id} ({email})")
    return response, 201


# ---- JWT Token Blacklist Check ----
# This callback is registered on the JWTManager in jwt_setup.py

def check_if_token_revoked(jwt_header, jwt_payload):
    """Check if a token's JTI is in the blacklist. Used by @jwt.token_in_blocklist_loader."""
    jti = jwt_payload['jti']
    row = query_one("SELECT jti FROM token_blacklist WHERE jti = %s", (jti,))
    return row is not None
