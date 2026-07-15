"""
Share-with-student tokens.

A teacher creates a token tied to a specific job; the token grants read-only
access to that job to anyone with the URL. To prevent abuse (one Pro account
becoming a free pipe to a whole teaching studio), each token has a cap on
distinct anonymous viewers (default 3). Signed-in users don't count toward
the cap — once a student creates an account, they're "real" not a leak.

Public surface:
    create_share_token(job_id, user_id, max_anonymous_viewers=3) -> raw token
    verify_share_token(token, job_id, request) -> (allowed: bool, reason: str|None)
    revoke_share_token(token, user_id) -> bool
    list_shares_for_job(job_id, user_id) -> list of {token, created_at, viewer_count, max_viewers, expires_at}

verify_share_token() is the function authorize_job_access() calls.
"""

import hashlib
import logging
import secrets
from datetime import datetime, timezone

from db import execute, execute_returning, query_one, query_all

logger = logging.getLogger(__name__)

DEFAULT_MAX_ANONYMOUS_VIEWERS = 3


def _viewer_hash(ip: str, user_agent: str) -> str:
    """Stable per-device fingerprint for distinct-viewer counting. Privacy
    note: this is a 16-char hex (64-bit truncated SHA-256), not the raw IP.
    Same device + same browser = same hash across requests (so reloads don't
    burn slots). Different device or different browser = different hash."""
    raw = f"{ip or 'unknown'}|{user_agent or 'unknown'}"
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()[:16]


def _email_hash(email: str | None) -> str | None:
    if not email:
        return None
    return hashlib.sha256(email.strip().lower().encode('utf-8')).hexdigest()


def _record_attribution(
    *,
    share_token: str,
    sharer_user_id: str,
    job_id: str,
    viewer_hash: str | None,
    student_user_id: str | None,
    student_email: str | None,
) -> None:
    """Write a share-attribution row + bump the sharer's lifetime counter.
    Idempotent on (share_token, viewer_hash) — repeat anonymous viewers no-op.
    For authed viewers we use a synthetic viewer_hash (`auth:<user_id>`) so the
    unique constraint still de-duplicates the same authed student re-visiting."""
    if not sharer_user_id or not job_id:
        return
    effective_vhash = viewer_hash or (f"auth:{student_user_id}" if student_user_id else None)
    if not effective_vhash:
        return
    new_row = execute_returning(
        """
        INSERT INTO share_attribution
            (share_token, sharer_user_id, job_id, viewer_hash,
             student_user_id, student_email, student_email_hash)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (share_token, viewer_hash) DO NOTHING
        RETURNING id
        """,
        (share_token, sharer_user_id, job_id, effective_vhash,
         student_user_id, student_email, _email_hash(student_email)),
    )
    if new_row:
        # First time we're seeing this viewer for this share. Bump counter.
        try:
            execute(
                "UPDATE users SET students_unlocked_lifetime = students_unlocked_lifetime + 1 WHERE id = %s",
                (sharer_user_id,),
            )
        except Exception as e:
            logger.warning(f"share_attribution: counter bump failed for {sharer_user_id}: {e}")


def create_share_token(
    job_id: str,
    user_id: str,
    max_anonymous_viewers: int = DEFAULT_MAX_ANONYMOUS_VIEWERS,
) -> str:
    """Generate + store a new share token for a job. Returns the raw token
    (URL-safe). Caller must verify the user owns the job before calling."""
    token = secrets.token_urlsafe(24)  # ~32 chars, unguessable
    execute(
        """
        INSERT INTO job_share_tokens (token, job_id, created_by_user_id, max_anonymous_viewers)
        VALUES (%s, %s, %s, %s)
        """,
        (token, job_id, user_id, int(max_anonymous_viewers)),
    )
    logger.info(f"share_token: created token for job {job_id} by user {user_id}")
    return token


def verify_share_token(token: str, job_id: str, request) -> tuple[bool, str | None]:
    """Check that a token is valid for this job, and that the viewer cap
    isn't exceeded. Returns (allowed, reason).

    Side effects:
        - Records this viewer in share_token_viewers if new.

    Reason strings (for caller logging — not user-facing):
        'unknown_token', 'wrong_job', 'revoked', 'expired', 'viewer_cap_reached'
    """
    if not token or not job_id:
        return False, 'unknown_token'

    row = query_one(
        """
        SELECT token, job_id, max_anonymous_viewers, expires_at, revoked_at, created_by_user_id
          FROM job_share_tokens
         WHERE token = %s
        """,
        (token,),
    )
    if not row:
        return False, 'unknown_token'
    if row['job_id'] != job_id:
        return False, 'wrong_job'
    if row['revoked_at'] is not None:
        return False, 'revoked'
    if row['expires_at'] is not None and row['expires_at'] < datetime.now(timezone.utc):
        return False, 'expired'

    # Signed-in viewers never count toward the cap (they're real users, not
    # leak vectors). Same for the creator visiting their own link.
    try:
        from flask import g
        user = getattr(g, 'current_user', None)
    except Exception:
        user = None
    if user is not None:
        # Attribution: capture signed-in student viewing too. If they ever buy
        # Pro/Lifetime, billing webhook updates this row's converted_at.
        # Skip if the viewer IS the sharer (don't count self-views).
        try:
            sharer_id = row.get('created_by_user_id')
            student_id = str(user.id)
            if sharer_id and sharer_id != student_id:
                _record_attribution(
                    share_token=token,
                    sharer_user_id=sharer_id,
                    job_id=job_id,
                    viewer_hash=None,
                    student_user_id=student_id,
                    student_email=getattr(user, 'email', None),
                )
        except Exception as e:
            logger.warning(f"share_token: attribution write failed (authed): {e}")
        return True, None

    # Anonymous viewer — fingerprint, insert if new, recount.
    ip = (request.headers.get('CF-Connecting-IP') or request.remote_addr or '').strip()
    ua = request.headers.get('User-Agent', '')[:512]
    vhash = _viewer_hash(ip, ua)

    inserted = execute_returning(
        """
        INSERT INTO share_token_viewers (token, viewer_hash)
        VALUES (%s, %s)
        ON CONFLICT (token, viewer_hash) DO NOTHING
        RETURNING viewer_hash
        """,
        (token, vhash),
    )
    if not inserted:
        # Repeat visitor (same device) — always allowed, doesn't count.
        return True, None

    # New viewer. Count and check cap.
    count_row = query_one(
        "SELECT COUNT(*) AS n FROM share_token_viewers WHERE token = %s",
        (token,),
    )
    count = int(count_row['n']) if count_row else 0
    if count > int(row['max_anonymous_viewers']):
        logger.info(f"share_token: viewer cap reached ({count}/{row['max_anonymous_viewers']}) for token on job {row['job_id']}")
        return False, 'viewer_cap_reached'

    # New anonymous viewer accepted — write attribution row. Sharer's lifetime
    # counter goes up by one. The student identity (email + user_id) stays null
    # until the email-claim mechanic ships post-launch; conversion timestamps
    # get filled by the Stripe webhook if/when this viewer ever buys Pro.
    try:
        sharer_id = row.get('created_by_user_id')
        if sharer_id:
            _record_attribution(
                share_token=token,
                sharer_user_id=sharer_id,
                job_id=row['job_id'],
                viewer_hash=vhash,
                student_user_id=None,
                student_email=None,
            )
    except Exception as e:
        logger.warning(f"share_token: attribution write failed (anon): {e}")
    return True, None


def revoke_share_token(token: str, user_id: str) -> bool:
    """Mark a token revoked. Only the original creator (or admin) can revoke."""
    row = execute_returning(
        """
        UPDATE job_share_tokens
           SET revoked_at = NOW()
         WHERE token = %s AND created_by_user_id = %s AND revoked_at IS NULL
         RETURNING token
        """,
        (token, user_id),
    )
    if row:
        logger.info(f"share_token: revoked {token[:8]}... by user {user_id}")
        return True
    return False


def list_shares_for_job(job_id: str, user_id: str) -> list:
    """Return all active (non-revoked) share tokens this user created for this job,
    with current viewer counts."""
    rows = query_all(
        """
        SELECT t.token, t.created_at, t.expires_at, t.max_anonymous_viewers,
               COALESCE((SELECT COUNT(*) FROM share_token_viewers v WHERE v.token = t.token), 0) AS viewer_count
          FROM job_share_tokens t
         WHERE t.job_id = %s AND t.created_by_user_id = %s AND t.revoked_at IS NULL
         ORDER BY t.created_at DESC
        """,
        (job_id, user_id),
    )
    return [dict(r) for r in (rows or [])]
