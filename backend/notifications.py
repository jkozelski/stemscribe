"""
Job-completion notifications.

Email-first for launch (2026-05-12) because SMS on the 843 local number has no
10DLC campaign (see memory/project_sms_broken.md). SMS can be re-enabled
post-launch once Twilio verification is in place.

Environment variables:
  RESEND_API_KEY      — Resend API key (already used by auth/email.py).
  APP_URL             — Base URL for the practice link (default stemscriber.com).
  ENABLE_JOB_EMAILS   — "true" to send job-complete emails. Default "false" so
                        it can be toggled on at launch without code deploy.
  EMAIL_FROM_JOB      — From header (default "StemScriber <noreply@stemscriber.com>").

Templates live in backend/email_templates/job-complete.{html,txt}. Variables:
  {{song_title}}, {{practice_url}}, {{job_id}}, {{first_name_prefix}}

The public entry point is `send_job_complete_email(job)`. It is best-effort:
any failure is logged and swallowed so the pipeline never fails because of a
notification problem.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / 'email_templates'


def job_emails_enabled() -> bool:
    """Check if the ENABLE_JOB_EMAILS flag is on."""
    return os.environ.get('ENABLE_JOB_EMAILS', 'false').strip().lower() in ('1', 'true', 'yes', 'on')


def _load_template(name: str) -> Optional[str]:
    """Load a template file from email_templates/. Returns None on failure."""
    path = TEMPLATE_DIR / name
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        logger.warning(f"notifications: failed to load template {name}: {e}")
        return None


def _render(template: str, variables: dict) -> str:
    """Minimal {{var}} substitution. We intentionally avoid a templating
    engine dependency — the templates are trusted, shipped with the repo.
    """
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace('{{' + key + '}}', str(value))
    return rendered


def _get_app_url() -> str:
    return os.environ.get('APP_URL', 'https://stemscriber.com').rstrip('/')


def _lookup_user_email(user_id: str) -> Optional[str]:
    """Look up a user's email by user_id. Returns None if not found or DB unavailable."""
    if not user_id:
        return None
    try:
        from auth.models import get_user_by_id
        user = get_user_by_id(user_id)
        if user and user.email:
            return user.email
    except Exception as e:
        logger.warning(f"notifications: user email lookup failed for {user_id}: {e}")
    return None


def _lookup_user_display_name(user_id: str) -> Optional[str]:
    if not user_id:
        return None
    try:
        from auth.models import get_user_by_id
        user = get_user_by_id(user_id)
        if user and user.display_name:
            return user.display_name
    except Exception:
        pass
    return None


def _recipient_for_job(job) -> Optional[str]:
    """Figure out which email address to send a completion notice to.

    Priority:
      1. job.metadata['notify_email'] — allows anonymous uploads to opt in.
      2. Logged-in user's email (user_id → auth.models.get_user_by_id).
    """
    meta = getattr(job, 'metadata', {}) or {}
    explicit = meta.get('notify_email')
    if explicit and isinstance(explicit, str) and '@' in explicit:
        return explicit.strip()

    user_id = getattr(job, 'user_id', None)
    if user_id:
        return _lookup_user_email(user_id)

    return None


def _song_title_for_job(job) -> str:
    meta = getattr(job, 'metadata', {}) or {}
    title = meta.get('title') or getattr(job, 'filename', None) or 'Your song'
    return str(title)


def _first_name_prefix(job) -> str:
    """Return ' Jeff' (with leading space) or '' so the template reads
    'Hey Jeff,' or just 'Hey,'."""
    user_id = getattr(job, 'user_id', None)
    name = _lookup_user_display_name(user_id) if user_id else None
    if not name:
        return ''
    first = name.strip().split()[0] if name.strip() else ''
    return f' {first}' if first else ''


def send_job_complete_email(job) -> bool:
    """Send a 'job done' email for the given ProcessingJob.

    Best-effort — returns True on success, False otherwise. Never raises.
    Skipped silently when ENABLE_JOB_EMAILS is off or no recipient is known.
    """
    try:
        if not job_emails_enabled():
            logger.debug("notifications: ENABLE_JOB_EMAILS=false, skipping")
            return False

        recipient = _recipient_for_job(job)
        if not recipient:
            logger.info(f"notifications: no recipient for job {getattr(job, 'job_id', '?')}, skipping")
            return False

        api_key = os.environ.get('RESEND_API_KEY')
        if not api_key:
            logger.warning("notifications: RESEND_API_KEY not set, cannot send job-complete email")
            return False

        html_template = _load_template('job-complete.html')
        txt_template = _load_template('job-complete.txt')
        if not html_template or not txt_template:
            return False

        job_id = getattr(job, 'job_id', '')
        song_title = _song_title_for_job(job)
        practice_url = f"{_get_app_url()}/practice.html?job={job_id}"

        variables = {
            'job_id': job_id,
            'song_title': song_title,
            'practice_url': practice_url,
            'first_name_prefix': _first_name_prefix(job),
        }

        html_body = _render(html_template, variables)
        txt_body = _render(txt_template, variables)
        subject = f'Your song "{song_title}" is ready on StemScriber'
        from_addr = os.environ.get('EMAIL_FROM_JOB', 'StemScriber <noreply@stemscriber.com>')

        import resend  # local import so pytest's sys.modules mock works
        resend.api_key = api_key
        resend.Emails.send({
            'from': from_addr,
            'to': [recipient],
            'subject': subject,
            'html': html_body,
            'text': txt_body,
        })
        logger.info(f"notifications: job-complete email sent for {job_id} to {recipient}")
        return True
    except Exception as e:
        # Best-effort: never let a notification failure break the pipeline.
        logger.error(f"notifications: failed to send job-complete email: {e}", exc_info=True)
        return False
