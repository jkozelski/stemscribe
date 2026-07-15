"""
Magic-link email sign-in.

Flow:
  1. User enters email → POST /auth/magic-link/request
       → generate random token (32 bytes), hash it, store hash in DB with
         15-min expiry, send raw token to email via Resend
  2. User clicks link in email → GET /auth/magic-link/verify?token=...
       → look up hash, verify unused + unexpired, mark used,
         find-or-create user, mint JWT tokens, return stash-and-redirect HTML
         (same shape as the Google redirect callback)

Security:
  - Token hash stored in DB; raw token only ever in the email. DB leak ≠ replay.
  - 15-min expiry. Single-use (used_at set on verify).
  - Rate-limit: max 5 requests per email per hour. We always return 200 from
    /request even if the email doesn't exist, so attackers can't enumerate.
  - DB-backed (not in-memory) so multi-worker gunicorn works.
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta, timezone

import resend

from db import execute, execute_returning, query_one

logger = logging.getLogger(__name__)

MAGIC_LINK_EXPIRY_MINUTES = 30
MAX_REQUESTS_PER_EMAIL_PER_HOUR = 5
# Audit fix (2026-05-31): per-IP cap closes the email-flooding attack — without
# this, an attacker hits /auth/magic-link/request with thousands of victim
# emails (up to 5 sign-in mails per address per hour) using your domain as
# the sender. Hurts brand + risks Resend account suspension during launch.
MAX_REQUESTS_PER_IP_PER_HOUR = 30
MAX_REQUESTS_PER_IP_PER_MINUTE = 3


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()


def _get_app_url() -> str:
    return os.environ.get('APP_URL', 'https://stemscriber.com').rstrip('/')


def _from_address() -> str:
    return os.environ.get('MAGIC_LINK_FROM', 'StemScriber <noreply@stemscriber.com>')


def _normalize_email(email: str) -> str:
    return (email or '').strip().lower()


def _too_many_recent_requests(email: str) -> bool:
    """True if this email has hit the rate limit in the last hour."""
    row = query_one(
        """
        SELECT COUNT(*) AS n
          FROM magic_link_tokens
         WHERE email = %s AND created_at > NOW() - INTERVAL '1 hour'
        """,
        (email,),
    )
    return bool(row) and int(row['n']) >= MAX_REQUESTS_PER_EMAIL_PER_HOUR


def _too_many_recent_requests_by_ip(ip: str) -> bool:
    """True if this IP has hit the per-IP rate limit (1 min or 1 hr window)."""
    if not ip:
        return False
    row = query_one(
        """
        SELECT
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 hour')   AS hour_count,
            COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '1 minute') AS minute_count
          FROM magic_link_tokens
         WHERE requesting_ip = %s
        """,
        (ip,),
    )
    if not row:
        return False
    if int(row['minute_count']) >= MAX_REQUESTS_PER_IP_PER_MINUTE:
        return True
    if int(row['hour_count']) >= MAX_REQUESTS_PER_IP_PER_HOUR:
        return True
    return False


def send_magic_link(email: str, requesting_ip: str = None) -> bool:
    """Generate + email a magic link. Returns True on send (or rate-limit
    no-op — callers should not differentiate to prevent enumeration). Returns
    False only on hard email-delivery / config failure."""
    email = _normalize_email(email)
    if not email or '@' not in email:
        return False

    if _too_many_recent_requests(email):
        logger.info(f"magic-link: rate-limited request for {email}")
        return True  # silent success — never tell the requester they hit the cap

    if _too_many_recent_requests_by_ip(requesting_ip):
        logger.warning(f"magic-link: per-IP rate-limited from {requesting_ip}")
        return True  # silent success — same reason, also blocks email-flood attack

    raw_token = secrets.token_urlsafe(32)
    token_hash = _hash_token(raw_token)
    # 6-digit numeric code for the iOS/Android app: user types it back into the
    # app instead of tapping the link (links can't reliably bounce back to the
    # native shell without Universal Links). Always present even when the user
    # only wants the link — cheap to generate, no UX cost for link-takers.
    raw_code = ''.join(secrets.choice('0123456789') for _ in range(6))
    code_hash = _hash_token(raw_code)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=MAGIC_LINK_EXPIRY_MINUTES)

    execute(
        """
        INSERT INTO magic_link_tokens (token_hash, email, expires_at, requesting_ip, code_hash)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (token_hash, email, expires_at, requesting_ip, code_hash),
    )

    link = f"{_get_app_url()}/auth/magic-link/verify?token={raw_token}"

    api_key = os.environ.get('RESEND_API_KEY')
    if not api_key:
        logger.error("magic-link: RESEND_API_KEY not configured")
        return False
    resend.api_key = api_key

    html = (
        '<!DOCTYPE html><html><body style="font-family:-apple-system,BlinkMacSystemFont,'
        'Segoe UI,sans-serif;background:#0d0d12;color:#fff;padding:40px;max-width:560px;'
        'margin:0 auto">'
        '<h2 style="color:#ff7b54;margin:0 0 16px">Sign in to StemScriber</h2>'
        '<p style="color:#ddd;font-size:1rem;line-height:1.5">'
        'Using the app? Type this code:</p>'
        f'<p style="margin:24px 0;text-align:center"><span style="font-family:SF Mono,Menlo,monospace;'
        'font-size:2.4rem;letter-spacing:0.4rem;color:#ff7b54;background:#1a1a24;padding:20px 28px;'
        f'border-radius:12px;display:inline-block;font-weight:700">{raw_code}</span></p>'
        '<p style="color:#ddd;font-size:1rem;line-height:1.5;margin-top:32px">'
        'On the web? Click the button below instead:</p>'
        f'<p style="margin:20px 0"><a href="{link}" '
        'style="background:#ff7b54;color:#fff;padding:14px 28px;border-radius:8px;'
        'text-decoration:none;font-weight:600;display:inline-block">Sign in to StemScriber</a></p>'
        '<p style="color:#888;font-size:0.85rem;margin-top:32px">'
        'Either way works. Code and link both expire in 30 minutes and can only be used once.</p>'
        '<p style="color:#666;font-size:0.8rem;margin-top:24px">'
        'If you didn\'t request this, you can safely ignore this email.</p>'
        '</body></html>'
    )
    text = (
        f"Sign in to StemScriber\n\n"
        f"Code (type into the app):  {raw_code}\n\n"
        f"Or click this link (on the web):\n{link}\n\n"
        "Code and link both expire in 30 minutes, single use.\n\n"
        "If you didn't request this, you can safely ignore this email."
    )

    try:
        resend.Emails.send({
            'from': _from_address(),
            'to': [email],
            'subject': 'Your StemScriber sign-in link',
            'html': html,
            'text': text,
        })
        logger.info(f"magic-link: sent to {email}")
        return True
    except Exception as e:
        logger.error(f"magic-link: Resend send failed for {email}: {e}")
        return False


def consume_magic_link(token: str) -> str | None:
    """Verify a magic-link token. On success: mark used, return the email.
    On any failure (unknown / expired / already-used): return None."""
    if not token:
        return None
    token_hash = _hash_token(token)

    row = query_one(
        """
        SELECT id, email, expires_at, used_at
          FROM magic_link_tokens
         WHERE token_hash = %s
        """,
        (token_hash,),
    )
    if not row:
        logger.info("magic-link: unknown token")
        return None
    if row['used_at'] is not None:
        logger.info(f"magic-link: token already used (id={row['id']})")
        return None
    if row['expires_at'] < datetime.now(timezone.utc):
        logger.info(f"magic-link: token expired (id={row['id']})")
        return None

    # Mark used atomically — if two requests race, only one gets the row.
    updated = execute_returning(
        """
        UPDATE magic_link_tokens
           SET used_at = NOW()
         WHERE id = %s AND used_at IS NULL
         RETURNING id
        """,
        (row['id'],),
    )
    if not updated:
        logger.info(f"magic-link: lost race on used_at (id={row['id']})")
        return None

    return row['email']


def consume_magic_code(email: str, code: str) -> str | None:
    """Verify a 6-digit code for the iOS/Android app sign-in flow.

    Same semantics as consume_magic_link but matched by (email, code_hash)
    pair instead of opaque token. Single-use, 30-min expiry. Returns the
    email on success, None on any failure (unknown / expired / used / mismatch).
    """
    if not email or not code:
        return None
    email = _normalize_email(email)
    # Trim whitespace + strip non-digits to be forgiving about copy/paste
    code = ''.join(ch for ch in code if ch.isdigit())
    if len(code) != 6:
        return None
    code_hash = _hash_token(code)

    row = query_one(
        """
        SELECT id, email, expires_at, used_at
          FROM magic_link_tokens
         WHERE email = %s AND code_hash = %s
         ORDER BY created_at DESC
         LIMIT 1
        """,
        (email, code_hash),
    )
    if not row:
        logger.info(f"magic-link code: no match for {email}")
        return None
    if row['used_at'] is not None:
        logger.info(f"magic-link code: already used (id={row['id']})")
        return None
    if row['expires_at'] < datetime.now(timezone.utc):
        logger.info(f"magic-link code: expired (id={row['id']})")
        return None

    updated = execute_returning(
        """
        UPDATE magic_link_tokens
           SET used_at = NOW()
         WHERE id = %s AND used_at IS NULL
         RETURNING id
        """,
        (row['id'],),
    )
    if not updated:
        logger.info(f"magic-link code: lost race on used_at (id={row['id']})")
        return None

    return row['email']
