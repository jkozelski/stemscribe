"""
Auth middleware — adaptive authentication decorator.

In production (DATABASE_URL set + JWT configured):
    Enforces JWT authentication on protected routes.
    Optionally enforces plan-based rate limits.

In development (no DATABASE_URL):
    Allows anonymous access with a warning log.
    Plan limits are not enforced.

Usage:
    from auth.middleware import auth_required

    @app.route('/api/upload')
    @auth_required          # Requires login in production, open in dev
    def upload():
        ...

    @app.route('/api/upload')
    @auth_required(optional=True)  # Identifies user if logged in, doesn't block
    def upload():
        ...
"""

import os
import logging
from functools import wraps

from flask import jsonify, g

logger = logging.getLogger(__name__)

# Detect whether auth infrastructure is available
_AUTH_AVAILABLE = False
_DB_AVAILABLE = bool(os.environ.get('DATABASE_URL'))

try:
    if _DB_AVAILABLE:
        from flask_jwt_extended import jwt_required as _jwt_required, get_jwt_identity, verify_jwt_in_request
        from auth.models import get_user_by_id
        _AUTH_AVAILABLE = True
except ImportError:
    pass


def auth_required(fn=None, *, optional=False):
    """Decorator: enforce JWT auth in production, allow anonymous in dev.

    Args:
        optional: If True, identify user if token present but don't block anonymous.
                  If False (default), require valid JWT in production.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            g.current_user = None

            if not _AUTH_AVAILABLE:
                # No DB/auth — dev mode, allow through
                return fn(*args, **kwargs)

            if optional:
                # Try to identify user but don't block
                try:
                    verify_jwt_in_request(optional=True)
                    user_id = get_jwt_identity()
                    if user_id:
                        g.current_user = get_user_by_id(user_id)
                except Exception:
                    pass
                return fn(*args, **kwargs)

            # Required auth — enforce JWT
            try:
                verify_jwt_in_request()
                user_id = get_jwt_identity()
                user = get_user_by_id(user_id)
                if not user:
                    return jsonify({'error': 'User not found'}), 401
                g.current_user = user
            except Exception as e:
                return jsonify({
                    'error': 'Authentication required',
                    'code': 'missing_token',
                    'details': str(e),
                }), 401

            return fn(*args, **kwargs)
        return wrapper

    if fn is not None:
        # Called as @auth_required without parentheses
        return decorator(fn)
    # Called as @auth_required() or @auth_required(optional=True)
    return decorator
