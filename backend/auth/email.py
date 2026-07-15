"""
Password reset email + welcome email via Resend API.

Requires RESEND_API_KEY and APP_URL environment variables.
"""

import os
import logging
import secrets
import threading
import time

import resend

logger = logging.getLogger(__name__)


def _from_address() -> str:
    # Use the verified sender. Display name "Jeff" keeps the email feeling
    # human in inbox previews without burning a domain-verification cycle.
    return os.environ.get('WELCOME_FROM', 'Jeff at StemScriber <noreply@stemscriber.com>')


def _send_welcome_sync(email: str, display_name: str | None) -> None:
    """Inner welcome send — called from a background thread so signup never
    blocks on email delivery. Failures log + swallow (Resend hiccup must not
    prevent the user from signing in successfully)."""
    api_key = os.environ.get('RESEND_API_KEY')
    if not api_key:
        logger.warning("welcome email: RESEND_API_KEY not configured — skipping")
        return
    resend.api_key = api_key

    # Best-effort first-name extraction. Don't be cute; just pull the first
    # token of display_name if it looks like a name. Fall back to "there".
    name = 'there'
    if display_name:
        first = (display_name or '').strip().split(' ', 1)[0]
        if first and 2 <= len(first) <= 30 and first.replace('-', '').replace("'", '').isalpha():
            name = first

    app_url = os.environ.get('APP_URL', 'https://stemscriber.com').rstrip('/') + '/app'

    text = (
        f"Hey {name},\n\n"
        "You're in. StemScriber turns any song you've got into stems + chord chart + practice cockpit in a couple minutes.\n\n"
        "The fastest way to see what it does: drop in a song you already know well. You'll hear the bass without the guitar. "
        "You'll see the chords above the words. You'll loop the bridge until you've got it.\n\n"
        f"→ Open StemScriber: {app_url}\n\n"
        "— Jeff\n"
        "StemScriber"
    )

    html = (
        '<!DOCTYPE html><html><body style="font-family:-apple-system,BlinkMacSystemFont,'
        '\'Segoe UI\',sans-serif;background:#0d0d12;color:#fff;padding:32px 24px;max-width:540px;'
        'margin:0 auto;line-height:1.55;font-size:16px">'
        f'<p>Hey {name},</p>'
        '<p>You\'re in. StemScriber turns any song you\'ve got into stems + chord chart + practice cockpit in a couple minutes.</p>'
        '<p>The fastest way to see what it does: drop in a song you already know well. You\'ll hear the bass without the guitar. '
        'You\'ll see the chords above the words. You\'ll loop the bridge until you\'ve got it.</p>'
        f'<p style="margin:28px 0"><a href="{app_url}" '
        'style="background:#ff7b54;color:#fff;padding:13px 26px;border-radius:8px;'
        'text-decoration:none;font-weight:600;display:inline-block">Open StemScriber</a></p>'
        '<p style="margin-top:32px">— Jeff<br>StemScriber</p>'
        '</body></html>'
    )

    try:
        resend.Emails.send({
            'from': _from_address(),
            'to': [email],
            'subject': "Welcome to StemScriber — let's tear something apart",
            'html': html,
            'text': text,
        })
        logger.info(f"welcome email sent to {email}")
    except Exception as e:
        logger.error(f"welcome email send failed for {email}: {e}")


def send_welcome_email(email: str, display_name: str | None = None) -> None:
    """Fire-and-forget welcome email. Spawns a daemon thread so the caller
    (signup endpoint) returns immediately. Call once per NEW user creation
    (google_user / magic-link / future email-claim student signup)."""
    if not email or '@' not in email:
        return
    try:
        t = threading.Thread(
            target=_send_welcome_sync,
            args=(email, display_name),
            daemon=True,
            name=f"welcome-email-{email[:20]}",
        )
        t.start()
    except Exception as e:
        logger.error(f"welcome email: couldn't spawn thread for {email}: {e}")

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
            'from': 'StemScriber <noreply@stemscriber.com>',
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
