"""
Input validation & sanitization — reusable validators for all API endpoints.

Prevents: SSRF, XSS, path traversal, injection attacks.

Usage:
    from middleware.validation import (
        sanitize_text, validate_job_id, validate_email_format,
        validate_phone_number, validate_url, validate_file_upload,
        validate_beta_code,
    )
"""

import re
import ipaddress
import socket
from urllib.parse import urlparse


# ---- HTML / XSS Sanitization ----

_HTML_TAG_RE = re.compile(r'<[^>]+>')


def strip_html_tags(text: str) -> str:
    """Remove all HTML tags from a string."""
    if not isinstance(text, str):
        return ''
    return _HTML_TAG_RE.sub('', text)


def sanitize_text(text, max_length: int = 5000) -> str:
    """Strip HTML tags, trim whitespace, enforce max length.

    Use this for all free-text user inputs (names, messages, etc.).
    """
    if not isinstance(text, str):
        return ''
    text = strip_html_tags(text).strip()
    if len(text) > max_length:
        text = text[:max_length]
    return text


# ---- Job ID Validation ----

_JOB_ID_RE = re.compile(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$')
_JOB_ID_LOOSE_RE = re.compile(r'^[a-f0-9-]+$')


def validate_job_id(job_id: str) -> bool:
    """Validate that job_id is a UUID format (prevents path traversal).

    Accepts full UUIDs (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)
    and short hex prefixes used by some endpoints.
    """
    if not job_id or not isinstance(job_id, str):
        return False
    if len(job_id) > 36:
        return False
    # Must be hex chars and dashes only — blocks ../ and other traversal
    return bool(_JOB_ID_LOOSE_RE.match(job_id))


def validate_job_id_strict(job_id: str) -> bool:
    """Validate that job_id is a full UUID v4 format."""
    if not job_id or not isinstance(job_id, str):
        return False
    return bool(_JOB_ID_RE.match(job_id))


# ---- Email Validation ----

_EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$')


def validate_email_format(email: str) -> bool:
    """Basic email format validation."""
    if not email or not isinstance(email, str):
        return False
    if len(email) > 254:  # RFC 5321 max
        return False
    return bool(_EMAIL_RE.match(email))


# ---- Phone Number Validation ----

_PHONE_RE = re.compile(r'^[\d\s\+\-\(\)]+$')


def validate_phone_number(phone: str) -> bool:
    """Validate phone number format: digits, +, -, spaces, parens only. Max 20 chars."""
    if not phone or not isinstance(phone, str):
        return False
    phone = phone.strip()
    if len(phone) > 20:
        return False
    return bool(_PHONE_RE.match(phone))


# ---- URL Validation (SSRF Prevention) ----

_BLOCKED_HOSTS = {
    'metadata.google.internal',
    '169.254.169.254',
    'localhost',
    'localhost.localdomain',
}


def validate_url(url: str) -> tuple:
    """Validate URL for safety — blocks SSRF, private IPs, dangerous schemes.

    Returns (is_valid: bool, error_message: str or None).
    """
    if not url or not isinstance(url, str):
        return False, 'URL is required'

    url = url.strip()

    # Max length
    if len(url) > 2048:
        return False, 'URL too long (max 2048 characters)'

    # Scheme check — only http/https
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False, 'Invalid URL scheme. Only http:// and https:// are allowed'

    hostname = parsed.hostname or ''
    if not hostname:
        return False, 'Invalid URL: no hostname'

    # Block known dangerous hostnames
    hostname_lower = hostname.lower()
    for blocked in _BLOCKED_HOSTS:
        if hostname_lower == blocked or hostname_lower.endswith('.' + blocked):
            return False, 'URL not allowed (blocked hostname)'

    # Block 0.0.0.0
    if hostname_lower == '0.0.0.0':
        return False, 'URL not allowed (blocked address)'

    # Resolve hostname and check for private/internal IPs
    try:
        resolved_ips = socket.getaddrinfo(
            hostname, parsed.port or 443, proto=socket.IPPROTO_TCP
        )
    except (socket.gaierror, OSError):
        return False, 'Could not resolve URL hostname'

    for family, _, _, _, sockaddr in resolved_ips:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False, 'URL not allowed (local/private network addresses are blocked)'
        except ValueError:
            return False, 'URL resolved to an invalid IP address'

    return True, None


# ---- File Upload Validation ----

ALLOWED_AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac',
    '.wma', '.aiff', '.webm', '.opus',
}

# Common MIME types for audio files
ALLOWED_AUDIO_MIMES = {
    'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/x-wav', 'audio/wave',
    'audio/flac', 'audio/x-flac', 'audio/ogg', 'audio/x-m4a', 'audio/mp4',
    'audio/aac', 'audio/x-aac', 'audio/x-ms-wma', 'audio/aiff',
    'audio/x-aiff', 'audio/webm', 'audio/opus',
    'application/octet-stream',  # browsers sometimes send this
    'video/webm',  # webm can be audio-only
}

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB


def validate_file_upload(file) -> tuple:
    """Validate an uploaded audio file.

    Args:
        file: werkzeug FileStorage object from request.files

    Returns (is_valid: bool, error_message: str or None).
    """
    from pathlib import Path
    from werkzeug.utils import secure_filename

    if not file or not file.filename:
        return False, 'No file provided'

    # Extension check
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        return False, f'Invalid file type "{ext}". Allowed: {sorted(ALLOWED_AUDIO_EXTENSIONS)}'

    # MIME type check (advisory — browsers can lie, but catches obvious mismatches)
    if file.content_type and file.content_type not in ALLOWED_AUDIO_MIMES:
        # Log but don't block — some browsers send wrong MIME types
        import logging
        logging.getLogger(__name__).warning(
            f"Unexpected MIME type {file.content_type} for file {file.filename}"
        )

    # File size check — read content length from headers if available
    # (actual size enforcement is done by Flask MAX_CONTENT_LENGTH)
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to start
    if size > MAX_FILE_SIZE_BYTES:
        return False, f'File too large ({size // (1024*1024)}MB). Maximum is 50MB'

    # Sanitize filename (this is also done in the route, but belt-and-suspenders)
    safe_name = secure_filename(file.filename)
    if not safe_name:
        return False, 'Invalid filename'

    return True, None


# ---- Beta Code Validation ----

_BETA_CODE_RE = re.compile(r'^[A-Z0-9\-]+$')


def validate_beta_code(code: str) -> tuple:
    """Validate beta code format: alphanumeric + dashes, max 50 chars.

    Returns (sanitized_code: str or None, error_message: str or None).
    """
    if not code or not isinstance(code, str):
        return None, 'Invite code is required'

    code = strip_html_tags(code).strip().upper()

    if len(code) > 50:
        return None, 'Invalid invite code (too long)'

    if not _BETA_CODE_RE.match(code):
        return None, 'Invalid invite code format (alphanumeric and dashes only)'

    return code, None


# ---- Ticket ID Validation ----

def validate_ticket_id(ticket_id: str) -> bool:
    """Validate ticket_id is a UUID format."""
    return validate_job_id_strict(ticket_id)
