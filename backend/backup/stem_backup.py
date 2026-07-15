"""
Stem-backup wrapper — called from the processing pipeline once a job
completes. Uploads stems + chord_chart.json + metadata to R2 in the
background so the foreground request returns immediately.

Key scheme:
    jobs/<job_id>/manifest.json       <- compact JSON with title, artist, owner
    jobs/<job_id>/stems/<filename>    <- each stem mp3/wav
    jobs/<job_id>/chord_chart.json    <- the chart (post-correction)
    jobs/<job_id>/musicxml/*          <- if present
    jobs/<job_id>/midi/*              <- if present
    jobs/<job_id>/gp/*                <- guitar pro tabs if present

DB columns (jobs.r2_stems_prefix, jobs.r2_upload_key etc.) get updated
once the upload completes so the restore endpoint knows what's available.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from backup.r2_client import r2_enabled, upload_file, upload_bytes

logger = logging.getLogger(__name__)

OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/opt/stemscribe/outputs"))


def _ct_for(name: str) -> str:
    n = name.lower()
    if n.endswith(".mp3"):  return "audio/mpeg"
    if n.endswith(".wav"):  return "audio/wav"
    if n.endswith(".json"): return "application/json"
    if n.endswith(".xml") or n.endswith(".musicxml"): return "application/vnd.recordare.musicxml+xml"
    if n.endswith(".mid") or n.endswith(".midi"):     return "audio/midi"
    if n.endswith(".gp5") or n.endswith(".gp") or n.endswith(".gpx"): return "application/octet-stream"
    return "application/octet-stream"


def backup_job(job_id: str, owner_email: Optional[str] = None, async_thread: bool = True) -> None:
    """Schedule (or run) the backup of a completed job to R2.

    `async_thread=True` (default) fire-and-forget on a background thread so
    the caller (typically pipeline.py after stems land on disk) doesn't pay
    the upload latency. Set False for sync execution in tests.
    """
    if not r2_enabled():
        return  # silent no-op when R2 isn't configured

    if async_thread:
        t = threading.Thread(target=_do_backup, args=(job_id, owner_email), daemon=True, name=f"r2-backup-{job_id[:8]}")
        t.start()
    else:
        _do_backup(job_id, owner_email)


def _do_backup(job_id: str, owner_email: Optional[str]) -> None:
    job_dir = OUTPUTS_DIR / job_id
    if not job_dir.exists():
        logger.warning(f"[r2-backup] job dir missing: {job_dir}")
        return

    prefix = f"jobs/{job_id}"
    n_ok = 0
    n_fail = 0

    # 1. manifest
    manifest = {
        "job_id": job_id,
        "owner_email": owner_email,
        "backup_version": 1,
    }
    if upload_bytes(json.dumps(manifest, indent=2).encode(), f"{prefix}/manifest.json", "application/json"):
        n_ok += 1
    else:
        n_fail += 1

    # 2. walk job dir + upload each file under its relative path
    for path in job_dir.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(job_dir)
        key = f"{prefix}/{rel.as_posix()}"
        if upload_file(str(path), key, content_type=_ct_for(path.name)):
            n_ok += 1
        else:
            n_fail += 1

    logger.info(f"[r2-backup] job {job_id}: {n_ok} uploaded, {n_fail} failed")

    # 3. Best-effort DB column update so restore knows what's backed up.
    try:
        from db import execute
        execute(
            "UPDATE jobs SET r2_stems_prefix = %s WHERE id = %s",
            (prefix, job_id),
        )
    except Exception as e:
        logger.warning(f"[r2-backup] could not update jobs.r2_stems_prefix for {job_id}: {e}")
