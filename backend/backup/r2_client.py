"""
R2 backup client — S3-compatible boto3 session pointed at Cloudflare R2.

Reads credentials from env (loaded via app.py's dotenv pattern):
    R2_ACCOUNT_ID
    R2_ACCESS_KEY_ID
    R2_SECRET_ACCESS_KEY
    R2_BUCKET             (default: stemscriber-backups)

R2 vs S3 quirks:
  - Endpoint URL: https://<account>.r2.cloudflarestorage.com
  - Signature version: s3v4 (NOT v2)
  - Region: 'auto' (R2 is single-region today)
  - Zero egress fees — restores cost only the operation count

Safe no-op when credentials are missing: helpers return False / None so
the upload site can wrap in "best effort" without crashing the pipeline.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_BUCKET_DEFAULT = "stemscriber-backups"


def _r2_creds():
    return {
        "account_id":   os.environ.get("R2_ACCOUNT_ID", "").strip(),
        "access_key":   os.environ.get("R2_ACCESS_KEY_ID", "").strip(),
        "secret_key":   os.environ.get("R2_SECRET_ACCESS_KEY", "").strip(),
        "bucket":       os.environ.get("R2_BUCKET", _BUCKET_DEFAULT).strip(),
    }


def r2_enabled() -> bool:
    """True only if all required env vars are present."""
    c = _r2_creds()
    return bool(c["account_id"] and c["access_key"] and c["secret_key"])


_client = None


def get_client():
    """Return a memoized boto3 S3 client targeted at R2. None if not configured."""
    global _client
    if _client is not None:
        return _client
    if not r2_enabled():
        logger.debug("[r2] not configured — skipping client init")
        return None

    try:
        import boto3
        from botocore.config import Config
    except ImportError:
        logger.warning("[r2] boto3 not installed")
        return None

    c = _r2_creds()
    endpoint = f"https://{c['account_id']}.r2.cloudflarestorage.com"
    _client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=c["access_key"],
        aws_secret_access_key=c["secret_key"],
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 3, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        ),
    )
    return _client


def get_bucket() -> str:
    return _r2_creds()["bucket"]


def upload_file(local_path: str, key: str, content_type: Optional[str] = None) -> bool:
    """Upload a single file to R2. Returns True on success, False on any failure.

    Designed to be called from a background thread that already has its own
    error handling — we LOG the failure but never raise, since backup is
    best-effort and a failed upload should NOT crash the user's job.
    """
    if not r2_enabled():
        return False
    client = get_client()
    if not client:
        return False
    bucket = get_bucket()
    extra = {}
    if content_type:
        extra["ContentType"] = content_type
    try:
        client.upload_file(local_path, bucket, key, ExtraArgs=extra or None)
        return True
    except Exception as e:
        logger.warning(f"[r2] upload failed: {key}: {type(e).__name__}: {e}")
        return False


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> bool:
    """Upload an in-memory blob to R2. Same best-effort semantics as upload_file."""
    if not r2_enabled():
        return False
    client = get_client()
    if not client:
        return False
    bucket = get_bucket()
    try:
        client.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)
        return True
    except Exception as e:
        logger.warning(f"[r2] put_object failed: {key}: {type(e).__name__}: {e}")
        return False


def download_file(key: str, local_path: str) -> bool:
    """Download a key from R2 to local_path. Used by the restore endpoint."""
    if not r2_enabled():
        return False
    client = get_client()
    if not client:
        return False
    bucket = get_bucket()
    try:
        client.download_file(bucket, key, local_path)
        return True
    except Exception as e:
        logger.warning(f"[r2] download failed: {key}: {type(e).__name__}: {e}")
        return False


def list_keys(prefix: str, max_keys: int = 1000) -> list:
    """List keys under a prefix. Returns [] on any failure."""
    if not r2_enabled():
        return []
    client = get_client()
    if not client:
        return []
    bucket = get_bucket()
    try:
        resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)
        return [obj["Key"] for obj in resp.get("Contents", [])]
    except Exception as e:
        logger.warning(f"[r2] list failed: prefix={prefix}: {type(e).__name__}: {e}")
        return []
