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
    create_email_user,
    link_google_account,
    update_user_password,
    get_user_by_apple_id,
    link_apple_account,
    create_apple_user,
    get_monthly_usage,
)
from auth.email import send_reset_email, verify_reset_token
from auth.magic_link import send_magic_link, consume_magic_link, consume_magic_code
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

def _verify_google_credential_and_login(credential: str):
    """Shared core: verify a Google ID-token credential and return (user, tokens, is_new)
    using the same find-or-create logic as the JSON POST endpoint.
    Raises ValueError on token/email/account problems."""
    google_client_id = os.environ.get('GOOGLE_CLIENT_ID')
    if not google_client_id:
        raise RuntimeError('GOOGLE_CLIENT_ID not configured')

    # Accept tokens minted by the web client AND any additional clients listed in
    # GOOGLE_CLIENT_ID_EXTRAS (comma-separated). iOS Capacitor app uses its own
    # iOS-typed OAuth client; tokens it mints carry that client ID as `aud`.
    extras = os.environ.get('GOOGLE_CLIENT_ID_EXTRAS', '')
    allowed_audiences = [google_client_id]
    for cid in extras.split(','):
        cid = cid.strip()
        if cid and cid not in allowed_audiences:
            allowed_audiences.append(cid)

    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    # The google-auth library accepts either a string or list for audience.
    idinfo = google_id_token.verify_oauth2_token(
        credential, google_requests.Request(),
        allowed_audiences if len(allowed_audiences) > 1 else google_client_id,
    )

    google_id = idinfo['sub']
    email = (idinfo.get('email') or '').lower().strip()
    name = idinfo.get('name', '')
    picture = idinfo.get('picture')
    if not email:
        raise ValueError('Google account has no email')

    user = get_user_by_google_id(google_id)
    if user:
        if picture:
            link_google_account(str(user.id), google_id, picture)
            user = get_user_by_id(str(user.id))
        return user, _issue_tokens(user), False
    user = get_user_by_email(email)
    if user:
        link_google_account(str(user.id), google_id, picture)
        user = get_user_by_id(str(user.id))
        return user, _issue_tokens(user), False
    user = create_google_user(email, name, google_id, picture)
    try:
        from auth.email import send_welcome_email
        send_welcome_email(user.email, name)
    except Exception as e:
        logger.warning(f"welcome email dispatch failed for new google user {user.id}: {e}")
    return user, _issue_tokens(user), True


@auth_bp.route('/google', methods=['POST'])
def google_login():
    """Authenticate or register via Google Sign-In ID token (legacy JSON POST path
    used by the GIS popup ux_mode). The new GIS redirect ux_mode posts to
    /auth/google/callback below instead."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Request body must be JSON'}), 400
    credential = data.get('credential', '')
    if not credential:
        return jsonify({'error': 'Google credential token is required'}), 400

    try:
        user, tokens, is_new = _verify_google_credential_and_login(credential)
    except ValueError as e:
        # token verification error (invalid sig / no email / etc.)
        logger.warning(f"Google token verification failed: {e}")
        return jsonify({'error': str(e) if 'email' in str(e) else 'Invalid Google token'}), 401
    except RuntimeError as e:
        logger.error(str(e))
        return jsonify({'error': 'Google Sign-In is not configured'}), 500

    response = jsonify(tokens)
    set_refresh_cookies(response, tokens['refresh_token'])
    status = 201 if is_new else 200
    logger.info(f"Google login (JSON POST): {'created' if is_new else 'existing'} user {user.id}")
    return response, status


@auth_bp.route('/google/callback', methods=['POST', 'GET'])
def google_login_redirect_callback():
    """Handle Google Identity Services 'redirect' ux_mode — Google POSTs the
    credential here as form-encoded data, we verify, sign the user in, then
    return a tiny HTML page that stashes the tokens in localStorage and
    redirects to /app.

    Why this exists: the popup ux_mode fails intermittently with
    `Error 400: origin_mismatch` because it relies on postMessage with strict
    JS-origin checks. The redirect ux_mode does a full-page navigation through
    accounts.google.com instead — no postMessage, no popup, no origin check.
    Same Google sign-in for the user; just doesn't break on iOS Safari edge
    cases or fresh-account propagation lag.
    """
    from flask import make_response
    credential = (
        request.form.get('credential')
        or request.form.get('id_token')          # OpenID implicit (response_mode=form_post)
        or request.values.get('credential')
        or request.values.get('id_token')
        or ''
    )
    if not credential:
        logger.warning("google/callback: no credential in request")
        return redirect_to_app_with_error('missing_credential')

    try:
        user, tokens, is_new = _verify_google_credential_and_login(credential)
    except ValueError as e:
        logger.warning(f"google/callback: verification failed: {e}")
        return redirect_to_app_with_error('invalid_token')
    except RuntimeError as e:
        logger.error(f"google/callback: {e}")
        return redirect_to_app_with_error('not_configured')

    logger.info(f"Google login (REDIRECT): {'created' if is_new else 'existing'} user {user.id}")

    # Stash tokens into localStorage via a tiny self-redirecting HTML page.
    # Tokens are also set as cookies (httponly refresh) for resilience.
    import json as _json
    safe_access = _json.dumps(tokens['access_token'])
    safe_refresh = _json.dumps(tokens['refresh_token'])
    safe_user = _json.dumps(tokens['user'])
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>Signing you in…</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#0d0d12;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;font-size:1rem}</style>'
        '</head><body>Signing you in&hellip;'
        '<script>'
        'try{'
        'localStorage.setItem("access_token",' + safe_access + ');'
        'localStorage.setItem("refresh_token",' + safe_refresh + ');'
        'localStorage.setItem("ss_user_cache",' + safe_user + ');'
        # Mark the invite-gate as passed for paid/lifetime/admin BEFORE handing off
        # to /app, so a signed-in Lifetime user is not bounced to /beta.
        'try{var _gp=((' + safe_user + ').plan||"").toLowerCase();if(_gp=="lifetime"||_gp=="pro"||_gp=="beta")localStorage.setItem("stemscribe-paid-plan",_gp);if(((' + safe_user + ').email||"")=="jkozelski@gmail.com")localStorage.setItem("stemscribe-admin","1");}catch(e){}'
        '}catch(e){}'
        'window.location.replace("/app");'
        '</script>'
        '</body></html>'
    )
    response = make_response(html, 200)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    response.headers['Cache-Control'] = 'no-store'
    set_refresh_cookies(response, tokens['refresh_token'])
    return response


def redirect_to_app_with_error(code: str):
    from flask import redirect
    return redirect(f'/?signin_error={code}', code=302)


# ---- Magic-link email sign-in ----
# Fallback for users who can't / won't use Google. Mirrors the find-or-create
# behavior of the Google flow but with email + click-the-link instead of OAuth.
# DB-backed tokens (15 min expiry, single-use) — see auth/magic_link.py.

@auth_bp.route('/magic-link/request', methods=['POST'])
def magic_link_request():
    """Accept an email, send a sign-in link. Always returns 200 to prevent
    enumeration — the only way to know whether the request worked is to check
    the inbox."""
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    if not email or '@' not in email or len(email) > 254:
        return jsonify({'error': 'A valid email address is required'}), 400

    requesting_ip = request.headers.get('CF-Connecting-IP') or request.remote_addr
    ok = send_magic_link(email, requesting_ip=requesting_ip)
    if not ok:
        # Hard infra failure (Resend key missing, etc.) — distinguish so the
        # frontend can show "Email service unavailable" instead of "check inbox".
        return jsonify({'error': 'Could not send email right now. Try Google sign-in or try again later.'}), 503

    return jsonify({'ok': True, 'message': 'Check your inbox for a sign-in link.'}), 200


@auth_bp.route('/magic-link/verify', methods=['GET'])
def magic_link_verify():
    """User clicked the link in their email. Consume the token, sign them in,
    return the same stash-tokens-in-localStorage HTML the Google redirect uses."""
    from flask import make_response
    token = request.args.get('token', '').strip()
    if not token:
        return redirect_to_app_with_error('missing_token')

    email = consume_magic_link(token)
    if not email:
        return redirect_to_app_with_error('invalid_or_expired_link')

    user = get_user_by_email(email)
    is_new = False
    if not user:
        try:
            user = create_email_user(email)
            is_new = True
            try:
                from auth.email import send_welcome_email
                send_welcome_email(user.email, None)
            except Exception as e:
                logger.warning(f"welcome email dispatch failed for new magic-link user {user.id}: {e}")
        except ValueError as e:
            # Race: another magic-link verify created the row between the
            # get_user_by_email above and now. Re-fetch.
            user = get_user_by_email(email)
            if not user:
                logger.error(f"magic-link verify: could not create or find user for {email}: {e}")
                return redirect_to_app_with_error('user_create_failed')

    tokens = _issue_tokens(user)
    logger.info(f"Magic-link login: {'created' if is_new else 'existing'} user {user.id}")

    import json as _json
    safe_access = _json.dumps(tokens['access_token'])
    safe_refresh = _json.dumps(tokens['refresh_token'])
    safe_user = _json.dumps(tokens['user'])
    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>Signing you in…</title>'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;background:#0d0d12;color:#fff;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;font-size:1rem}</style>'
        '</head><body>Signing you in&hellip;'
        '<script>'
        'try{'
        'localStorage.setItem("access_token",' + safe_access + ');'
        'localStorage.setItem("refresh_token",' + safe_refresh + ');'
        'localStorage.setItem("ss_user_cache",' + safe_user + ');'
        # Mark the invite-gate as passed for paid/lifetime/admin BEFORE handing off
        # to /app, so a signed-in Lifetime user is not bounced to /beta.
        'try{var _gp=((' + safe_user + ').plan||"").toLowerCase();if(_gp=="lifetime"||_gp=="pro"||_gp=="beta")localStorage.setItem("stemscribe-paid-plan",_gp);if(((' + safe_user + ').email||"")=="jkozelski@gmail.com")localStorage.setItem("stemscribe-admin","1");}catch(e){}'
        '}catch(e){}'
        'window.location.replace("/app");'
        '</script>'
        '</body></html>'
    )
    response = make_response(html, 200)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    response.headers['Cache-Control'] = 'no-store'
    set_refresh_cookies(response, tokens['refresh_token'])
    return response


@auth_bp.route('/magic-link/verify-code', methods=['POST'])
def magic_link_verify_code():
    """Native-app sign-in: user typed the 6-digit code from their email.

    Same verification + token issue as /magic-link/verify, but accepts
    {email, code} JSON and returns JSON tokens for the iOS/Android app to
    stash in its own storage. Web flow keeps using the URL link verify.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    code = (data.get('code') or '').strip()
    if not email or '@' not in email:
        return jsonify({'error': 'A valid email is required'}), 400
    if not code:
        return jsonify({'error': 'Code is required'}), 400

    verified_email = consume_magic_code(email, code)
    if not verified_email:
        return jsonify({'error': 'Invalid or expired code'}), 400

    user = get_user_by_email(verified_email)
    is_new = False
    if not user:
        try:
            user = create_email_user(verified_email)
            is_new = True
            try:
                from auth.email import send_welcome_email
                send_welcome_email(user.email, None)
            except Exception as e:
                logger.warning(f"welcome email dispatch failed for new code-flow user {user.id}: {e}")
        except ValueError as e:
            user = get_user_by_email(verified_email)
            if not user:
                logger.error(f"magic-code verify: could not create or find user for {verified_email}: {e}")
                return jsonify({'error': 'Could not create account'}), 500

    tokens = _issue_tokens(user)
    logger.info(f"Magic-code login: {'created' if is_new else 'existing'} user {user.id}")

    response = jsonify({
        'access_token': tokens['access_token'],
        'refresh_token': tokens['refresh_token'],
        'user': tokens['user'],
    })
    set_refresh_cookies(response, tokens['refresh_token'])
    return response, 200


# ---- JWT Token Blacklist Check ----
# This callback is registered on the JWTManager in jwt_setup.py

def check_if_token_revoked(jwt_header, jwt_payload):
    """Check if a token's JTI is in the blacklist. Used by @jwt.token_in_blocklist_loader."""
    jti = jwt_payload['jti']
    row = query_one("SELECT jti FROM token_blacklist WHERE jti = %s", (jti,))
    return row is not None


# ---- Email + Password endpoints (added back 2026-06-06 pre-launch sprint) ----
# These were removed in May when we went Google-only. Auth sprint to Refinery
# launch (June 20) restores them so users can sign up with a real password
# instead of waiting for a magic-link email every time.

@auth_bp.route('/register', methods=['POST'])
def register():
    """Create a new account with email + password.

    If the email already belongs to a passwordless (magic-link or Google) user,
    set their password on the existing account instead of erroring. This is the
    account-link path for users who started with magic-link and now want a real
    password.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    display_name = (data.get('display_name') or '').strip() or None

    if not email or '@' not in email:
        return jsonify({'error': 'A valid email is required'}), 400

    pw_error = None
    try:
        from auth.models import validate_password
        pw_error = validate_password(password)
    except Exception:
        pass
    if pw_error:
        return jsonify({'error': pw_error}), 400

    existing = get_user_by_email(email)
    if existing and existing.password_hash:
        return jsonify({'error': 'An account with this email already exists. Try signing in.'}), 409

    try:
        if existing:
            # Existing passwordless (magic-link or Google) account — set a password
            update_user_password(str(existing.id), password)
            user = get_user_by_id(str(existing.id))
            logger.info(f"Register: set password on existing passwordless user {user.id} ({email})")
            status = 200
        else:
            user = create_user(email=email, password=password, display_name=display_name)
            logger.info(f"Register: created new user {user.id} ({email})")
            status = 201
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"Register failed for {email}: {e}")
        return jsonify({'error': 'Could not create account. Try again.'}), 500

    tokens = _issue_tokens(user)
    response = jsonify(tokens)
    set_refresh_cookies(response, tokens['refresh_token'])
    return response, status


@auth_bp.route('/login', methods=['POST'])
def login():
    """Sign in with email + password. Returns generic error on bad creds to
    avoid leaking which emails are registered."""
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    user = get_user_by_email(email)
    GENERIC = 'Invalid email or password'

    if not user or not user.password_hash:
        # Account exists via Google/magic-link only, or no account at all.
        # Same response either way so we don't leak which emails are registered.
        logger.info(f"Login failed: no password set for {email}")
        return jsonify({'error': GENERIC}), 401

    if not user.check_password(password):
        logger.info(f"Login failed: bad password for {email}")
        return jsonify({'error': GENERIC}), 401

    tokens = _issue_tokens(user)
    response = jsonify(tokens)
    set_refresh_cookies(response, tokens['refresh_token'])
    logger.info(f"Login OK: {user.id} ({email})")
    return response, 200


@auth_bp.route('/set-password', methods=['POST'])
@jwt_required()
def set_password():
    """Authenticated route — lets a signed-in user (typically one who signed in
    via magic-link or Google) set or change their password."""
    data = request.get_json(silent=True) or {}
    new_password = data.get('password') or ''

    uid = get_jwt_identity()
    user = get_user_by_id(uid)
    if not user:
        return jsonify({'error': 'User not found'}), 404

    try:
        update_user_password(str(user.id), new_password)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.exception(f"set-password failed for {user.id}: {e}")
        return jsonify({'error': 'Could not update password'}), 500

    logger.info(f"Password set by {user.id}")
    return jsonify({'ok': True}), 200


@auth_bp.route('/forgot-password', methods=['POST'])
def forgot_password():
    """Send a 6-digit reset code via the existing magic-link sender. The user
    verifies the code via /auth/magic-link/verify-code (existing endpoint) which
    signs them in; once signed in they can call /auth/set-password to set a new
    one. Single email-sending path keeps Resend templates + rate-limiting in one
    place."""
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    if not email or '@' not in email:
        return jsonify({'error': 'A valid email is required'}), 400

    # Reuse magic-link sender. send_magic_link is idempotent + rate-limited.
    try:
        send_magic_link(email)
    except Exception as e:
        logger.exception(f"forgot-password send failed for {email}: {e}")
        # Still return OK to avoid leaking which emails are registered
    return jsonify({'ok': True, 'message': 'If an account exists for that email, a reset code is on the way.'}), 200


# ---- Sign in with Apple ----------------------------------------------------
# Verifies an Apple identity token (JWT signed by Apple) and returns our backend
# JWTs. Audience accepts the iOS bundle ID and any Services IDs listed in
# APPLE_AUDIENCES (comma-separated env var). Added 2026-06-06 pre-launch sprint.

import functools
import time as _time


@functools.lru_cache(maxsize=1)
def _apple_keys_cache_holder():
    return {"fetched_at": 0, "keys": None}


def _apple_public_keys():
    """Fetch Apple's JWKS, cached for 1 hour. Returns a list of jwk dicts."""
    import requests as _requests
    cache = _apple_keys_cache_holder()
    now = _time.time()
    if cache["keys"] and (now - cache["fetched_at"]) < 3600:
        return cache["keys"]
    resp = _requests.get('https://appleid.apple.com/auth/keys', timeout=8)
    resp.raise_for_status()
    keys = resp.json().get('keys', [])
    cache["keys"] = keys
    cache["fetched_at"] = now
    return keys


def _verify_apple_identity_token(token: str):
    """Verify Apple identity_token JWT. Returns the decoded payload dict.

    Raises ValueError on any verification failure.
    """
    import jwt as _jwt
    from jwt.algorithms import RSAAlgorithm

    audiences = [a.strip() for a in (os.environ.get('APPLE_AUDIENCES', 'com.kozelski.stemscriber').split(',')) if a.strip()]
    if not audiences:
        raise ValueError('APPLE_AUDIENCES not configured')

    try:
        unverified = _jwt.get_unverified_header(token)
    except Exception as e:
        raise ValueError(f'Malformed Apple token: {e}')
    kid = unverified.get('kid')
    if not kid:
        raise ValueError('Apple token missing kid')

    keys = _apple_public_keys()
    jwk = next((k for k in keys if k.get('kid') == kid), None)
    if not jwk:
        # Refresh once in case Apple rotated
        _apple_keys_cache_holder()["fetched_at"] = 0
        keys = _apple_public_keys()
        jwk = next((k for k in keys if k.get('kid') == kid), None)
    if not jwk:
        raise ValueError(f'No matching Apple key for kid={kid}')

    public_key = RSAAlgorithm.from_jwk(jwk)
    try:
        payload = _jwt.decode(
            token, public_key, algorithms=['RS256'],
            audience=audiences,
            issuer='https://appleid.apple.com',
        )
    except _jwt.PyJWTError as e:
        raise ValueError(f'Invalid Apple token: {e}')
    return payload


@auth_bp.route('/apple', methods=['POST'])
def apple_login():
    """Authenticate or register via Sign in with Apple.

    Body: {
      identity_token: <JWT from AuthorizationAppleIDProvider>,
      user: { name?: { firstName, lastName }, email? }   # only present on first sign-in
    }
    """
    data = request.get_json(silent=True) or {}
    token = (data.get('identity_token') or data.get('credential') or '').strip()
    if not token:
        return jsonify({'error': 'identity_token is required'}), 400

    try:
        payload = _verify_apple_identity_token(token)
    except ValueError as e:
        logger.warning(f"Apple token verify failed: {e}")
        return jsonify({'error': 'Invalid Apple token'}), 401

    apple_sub = payload.get('sub')
    if not apple_sub:
        return jsonify({'error': 'Apple token missing sub'}), 401
    email_from_token = (payload.get('email') or '').strip().lower()
    email_verified = payload.get('email_verified') in (True, 'true', 1, '1')

    # First-sign-in body may include user.name / user.email — fall back to token
    body_user = data.get('user') or {}
    name_obj = body_user.get('name') or {}
    display_name_from_body = (
        (name_obj.get('firstName') or '') + ' ' + (name_obj.get('lastName') or '')
    ).strip() or None
    email_from_body = (body_user.get('email') or '').strip().lower()
    email = email_from_token or email_from_body

    # Find existing user
    user = get_user_by_apple_id(apple_sub) if 'get_user_by_apple_id' in globals() else None
    if not user and email:
        user = get_user_by_email(email)
        if user:
            link_apple_account(str(user.id), apple_sub)
            user = get_user_by_id(str(user.id))

    is_new = False
    if not user:
        if not email:
            # No email and no existing account — Apple sometimes hides the email.
            # We refuse rather than create an unrecoverable account.
            return jsonify({'error': 'Apple did not share an email. Try Sign in with Apple again with email sharing enabled.'}), 400
        user = create_apple_user(email=email, apple_id=apple_sub, display_name=display_name_from_body)
        is_new = True
        try:
            from auth.email import send_welcome_email
            send_welcome_email(user.email, display_name_from_body or '')
        except Exception as e:
            logger.warning(f"welcome email dispatch failed for new apple user {user.id}: {e}")

    tokens = _issue_tokens(user)
    response = jsonify(tokens)
    set_refresh_cookies(response, tokens['refresh_token'])
    status = 201 if is_new else 200
    logger.info(f"Apple login: {'created' if is_new else 'existing'} user {user.id}")
    return response, status
