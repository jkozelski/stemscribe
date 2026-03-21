"""
Flask-JWT-Extended configuration.

Call init_jwt(app) from the app factory to set up JWT handling.
"""

import os
from datetime import timedelta

from flask_jwt_extended import JWTManager


def init_jwt(app):
    """Configure and initialize Flask-JWT-Extended on the Flask app."""

    # Secret key for signing tokens
    jwt_secret = os.environ.get('JWT_SECRET_KEY', '')
    if not jwt_secret or jwt_secret in ('dev-secret-change-in-production', 'CHANGE_ME_generate_a_64_char_random_string'):
        if os.environ.get('FLASK_ENV') == 'production':
            raise RuntimeError(
                "FATAL: JWT_SECRET_KEY is not set or uses a default value. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
        # Dev mode: use a random secret per-run (tokens won't persist across restarts)
        import secrets
        jwt_secret = secrets.token_urlsafe(64)
        app.logger.warning("JWT_SECRET_KEY not set — using random secret (dev mode only)")
    app.config['JWT_SECRET_KEY'] = jwt_secret

    # Token lifetimes
    access_expires = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 900))  # 15 min
    refresh_expires = int(os.environ.get('JWT_REFRESH_TOKEN_EXPIRES', 2592000))  # 30 days
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(seconds=access_expires)
    app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(seconds=refresh_expires)

    # Store refresh tokens in HTTP-only cookies
    app.config['JWT_TOKEN_LOCATION'] = ['headers', 'cookies']
    app.config['JWT_COOKIE_SECURE'] = os.environ.get('FLASK_ENV') == 'production'
    app.config['JWT_COOKIE_CSRF_PROTECT'] = True
    app.config['JWT_COOKIE_SAMESITE'] = 'Lax'

    # Access tokens come via Authorization header
    app.config['JWT_HEADER_NAME'] = 'Authorization'
    app.config['JWT_HEADER_TYPE'] = 'Bearer'

    jwt = JWTManager(app)

    # Register the token blacklist checker
    from auth.routes import check_if_token_revoked
    jwt.token_in_blocklist_loader(check_if_token_revoked)

    # Custom error handlers for JWT errors
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        return {'error': 'Token has expired', 'code': 'token_expired'}, 401

    @jwt.invalid_token_loader
    def invalid_token_callback(error_string):
        return {'error': 'Invalid token', 'code': 'invalid_token'}, 401

    @jwt.unauthorized_loader
    def missing_token_callback(error_string):
        return {'error': 'Authentication required', 'code': 'missing_token'}, 401

    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        return {'error': 'Token has been revoked', 'code': 'token_revoked'}, 401

    return jwt
