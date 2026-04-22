"""
Retention sweep — automatic deletion of stale uploads and outputs.

Policy (configurable via env):
    UPLOAD_RETENTION_HOURS  uploads/<jobId>/   default 48h after completion
    OUTPUT_RETENTION_DAYS   outputs/<jobId>/   default 7d  after completion
    RETENTION_DRY_RUN       true|false        default true (logs only)
    RETENTION_INTERVAL_HOURS sweep cadence    default 1h
    RETENTION_ENABLED       true|false        default true

Safeguards:
- Never delete a job still `pending`, `processing`, or `queued`.
- Skip demo songs (`metadata.demo = True`) and any job with
  `metadata.retain = True`.
- Age is measured from the job's completion time when known, falling back
  to the directory's mtime. This handles jobs that failed before completion
  and jobs loaded from older versions that don't have a completed_at field.

Runs as a background daemon thread started from app.create_app().
"""

import os
import time
import shutil
import logging
import threading
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---- Config ----

def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (ValueError, TypeError):
        return default


def _config():
    return {
        'upload_retention_hours': _env_float('UPLOAD_RETENTION_HOURS', 48.0),
        'output_retention_days': _env_float('OUTPUT_RETENTION_DAYS', 7.0),
        'dry_run': _env_bool('RETENTION_DRY_RUN', True),
        'interval_hours': _env_float('RETENTION_INTERVAL_HOURS', 1.0),
        'enabled': _env_bool('RETENTION_ENABLED', True),
    }


# ---- Helpers ----

def _dir_size_bytes(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob('*'):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _completion_time(job, dir_path: Path) -> float:
    """Best-effort completion timestamp for age calculations."""
    # Prefer explicit completion timestamp if set
    if job is not None:
        ts = getattr(job, 'completed_at', None) or (
            job.metadata.get('completed_at') if getattr(job, 'metadata', None) else None
        )
        if ts:
            try:
                return float(ts)
            except (TypeError, ValueError):
                pass
        # Fall back to created_at when job object exists but completion not recorded
        if getattr(job, 'created_at', None):
            return float(job.created_at)
    # No job in memory — use directory mtime
    try:
        return dir_path.stat().st_mtime
    except OSError:
        return time.time()


def _is_exempt(job) -> bool:
    """Return True if the job should never be auto-deleted."""
    if job is None:
        return False
    meta = getattr(job, 'metadata', None) or {}
    if meta.get('retain') is True:
        return True
    if meta.get('demo') is True:
        return True
    # Kozelski demo songs: exempt by artist tag
    artist = (meta.get('artist') or '').strip().lower()
    if artist == 'kozelski':
        return True
    return False


def _is_job_in_progress(job) -> bool:
    """Return True if the job is still being worked on (do not delete)."""
    if job is None:
        return False
    status = getattr(job, 'status', None)
    # Only 'completed' and 'failed' are safe to reap.
    return status not in ('completed', 'failed')


def _delete_dir(path: Path, dry_run: bool) -> int:
    """Delete directory; return bytes freed (0 if dry-run or missing)."""
    if not path.exists():
        return 0
    size = _dir_size_bytes(path)
    if dry_run:
        return size
    try:
        shutil.rmtree(path)
    except OSError as e:
        logger.warning(f"[retention] failed to remove {path}: {e}")
        return 0
    return size


# ---- Sweep ----

def run_sweep(
    upload_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    now: Optional[float] = None,
    config: Optional[dict] = None,
) -> dict:
    """Run one retention sweep. Returns a summary dict.

    Exposed as a function (not a method) so tests can drive it with fake
    directories and fake clocks.
    """
    from models.job import UPLOAD_DIR as _UP, OUTPUT_DIR as _OUT, jobs as _jobs

    cfg = config or _config()
    upload_dir = upload_dir or _UP
    output_dir = output_dir or _OUT
    now = now if now is not None else time.time()

    upload_cutoff = now - (cfg['upload_retention_hours'] * 3600.0)
    output_cutoff = now - (cfg['output_retention_days'] * 86400.0)

    dry_run = bool(cfg['dry_run'])
    summary = {
        'dry_run': dry_run,
        'uploads_deleted': 0,
        'outputs_deleted': 0,
        'uploads_bytes_freed': 0,
        'outputs_bytes_freed': 0,
        'skipped_in_progress': 0,
        'skipped_exempt': 0,
    }

    # ---- Uploads (48h) ----
    if upload_dir.exists():
        for job_dir in _safe_iter(upload_dir):
            if not job_dir.is_dir():
                continue
            job_id = job_dir.name
            job = _jobs.get(job_id)
            if _is_job_in_progress(job):
                summary['skipped_in_progress'] += 1
                continue
            if _is_exempt(job):
                summary['skipped_exempt'] += 1
                continue
            completed_at = _completion_time(job, job_dir)
            if completed_at > upload_cutoff:
                continue  # too young
            age_hours = (now - completed_at) / 3600.0
            bytes_freed = _delete_dir(job_dir, dry_run)
            summary['uploads_deleted'] += 1
            summary['uploads_bytes_freed'] += bytes_freed
            logger.info(
                "[retention] %s upload job_id=%s age=%.1fh bytes=%d",
                'DRY-RUN' if dry_run else 'DELETED',
                job_id, age_hours, bytes_freed,
            )

    # ---- Outputs (7d) — includes any stored chord charts/stems ----
    if output_dir.exists():
        for job_dir in _safe_iter(output_dir):
            if not job_dir.is_dir():
                continue
            job_id = job_dir.name
            job = _jobs.get(job_id)
            if _is_job_in_progress(job):
                summary['skipped_in_progress'] += 1
                continue
            if _is_exempt(job):
                summary['skipped_exempt'] += 1
                continue
            completed_at = _completion_time(job, job_dir)
            if completed_at > output_cutoff:
                continue  # too young
            age_days = (now - completed_at) / 86400.0
            bytes_freed = _delete_dir(job_dir, dry_run)
            summary['outputs_deleted'] += 1
            summary['outputs_bytes_freed'] += bytes_freed
            logger.info(
                "[retention] %s output job_id=%s age=%.1fd bytes=%d",
                'DRY-RUN' if dry_run else 'DELETED',
                job_id, age_days, bytes_freed,
            )
            # Also drop from in-memory jobs registry so /api/library updates
            if not dry_run and job_id in _jobs:
                try:
                    del _jobs[job_id]
                except KeyError:
                    pass

    logger.info(
        "[retention] sweep complete dry_run=%s uploads=%d(%.1fMB) outputs=%d(%.1fMB) "
        "skipped_active=%d skipped_exempt=%d",
        dry_run,
        summary['uploads_deleted'],
        summary['uploads_bytes_freed'] / 1024 / 1024,
        summary['outputs_deleted'],
        summary['outputs_bytes_freed'] / 1024 / 1024,
        summary['skipped_in_progress'],
        summary['skipped_exempt'],
    )
    return summary


def _safe_iter(path: Path):
    try:
        yield from path.iterdir()
    except OSError as e:
        logger.warning(f"[retention] cannot iterate {path}: {e}")
        return


# ---- Background thread ----

def _retention_loop():
    """Main retention loop — sweeps on `RETENTION_INTERVAL_HOURS` cadence."""
    # Stagger initial run so startup isn't disrupted — 60s after boot.
    time.sleep(60)
    while True:
        cfg = _config()
        if not cfg['enabled']:
            time.sleep(max(60.0, cfg['interval_hours'] * 3600.0))
            continue
        try:
            run_sweep(config=cfg)
        except Exception as e:
            logger.error(f"[retention] sweep failed: {e}", exc_info=True)
        time.sleep(max(60.0, cfg['interval_hours'] * 3600.0))


def start_retention_sweeper(app=None):
    """Start the retention sweeper background thread.

    Args:
        app: Flask app (unused, kept for parity with start_watchdog).
    """
    cfg = _config()
    if not cfg['enabled']:
        logger.info("[retention] sweeper disabled via RETENTION_ENABLED=false")
        return None
    thread = threading.Thread(
        target=_retention_loop, daemon=True, name='retention-sweeper'
    )
    thread.start()
    logger.info(
        "[retention] sweeper started — uploads=%.1fh outputs=%.1fd interval=%.1fh dry_run=%s",
        cfg['upload_retention_hours'],
        cfg['output_retention_days'],
        cfg['interval_hours'],
        cfg['dry_run'],
    )
    return thread
