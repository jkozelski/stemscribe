"""
Password reset email via Resend API.

Requires RESEND_API_KEY and APP_URL environment variables.
"""

import os
import logging
import secrets
import time

import resend

logger = logging.getLogger(__name__)

# In-memory store for reset tokens. In production with multiple workers,
# move this to the database or Redis.
_reset_tokens = {}  # {token: {'user_id': str, 'expires': float}}

RESET_TOKEN_EXPIRY_SECONDS = 3600  # 1 hour


def _get_app_url():
    return os.environ.get('APP_URL', 'http://localhost:5555')


def generate_reset_token(user_id: str) -> str:
    """Generate a single-use password reset token."""
    # Clean up any existing tokens for this user
    to_remove = [t for t, v in _reset_tokens.items() if v['user_id'] == user_id]
    for t in to_remove:
        del _reset_tokens[t]

    token = secrets.token_urlsafe(32)
    _reset_tokens[token] = {
        'user_id': user_id,
        'expires': time.time() + RESET_TOKEN_EXPIRY_SECONDS,
    }
    return token


def verify_reset_token(token: str) -> str | None:
    """Verify a reset token. Returns user_id if valid, None if expired/invalid.
    Consumes the token (single-use).
    """
    data = _reset_tokens.pop(token, None)
    if not data:
        return None
    if time.time() > data['expires']:
        return None
    return data['user_id']


def send_reset_email(email: str, user_id: str) -> bool:
    """Send a password reset email. Returns True on success."""
    api_key = os.environ.get('RESEND_API_KEY')
    if not api_key:
        logger.error("RESEND_API_KEY not set, cannot send reset email")
        return False

    resend.api_key = api_key
    token = generate_reset_token(user_id)
    reset_url = f"{_get_app_url()}/reset-password?token={token}"

    try:
        resend.Emails.send({
            'from': 'StemScriber <noreply@stemscribe.app>',
            'to': [email],
            'subject': 'Reset your StemScriber password',
            'html': f"""
                <h2>Password Reset</h2>
                <p>You requested a password reset for your StemScriber account.</p>
                <p><a href="{reset_url}" style="
                    display: inline-block;
                    padding: 12px 24px;
                    background: #6366f1;
                    color: white;
                    text-decoration: none;
                    border-radius: 6px;
                    font-weight: bold;
                ">Reset Password</a></p>
                <p>This link expires in 1 hour. If you didn't request this, you can ignore this email.</p>
                <p style="color: #888; font-size: 12px;">StemScriber - Audio Stem Separation &amp; Transcription</p>
            """,
        })
        logger.info(f"Reset email sent to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send reset email to {email}: {e}")
        return False
