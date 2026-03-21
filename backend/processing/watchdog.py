"""
Processing watchdog — detect stalled jobs and auto-retry.

Runs as a background daemon thread, checking every 30 seconds for jobs
that haven't progressed in 2 minutes. Stalled jobs get retried up to
MAX_RETRIES times (with GPU memory flush) before being marked as failed.

Usage (called from app.py):
    from processing.watchdog import start_watchdog
    start_watchdog(app)
"""

import json
import time
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

STALL_THRESHOLD_SECONDS = 120  # 2 minutes with no progress
MAX_RETRIES = 2
CHECK_INTERVAL_SECONDS = 30   # Check every 30 seconds for faster detection
WATCHDOG_LOG_FILE = Path(__file__).parent.parent / 'watchdog_log.json'

_lock = threading.Lock()

# Track per-job state: {job_id: {'last_progress': int, 'last_check': float, 'retries': int}}
_job_snapshots = {}


def _log_event(event_type, job_id, details=None):
    """Append an event to the watchdog log file."""
    entry = {
        'timestamp': time.time(),
        'time': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'),
        'event': event_type,
        'job_id': job_id,
    }
    if details:
        entry['details'] = details

    try:
        with _lock:
            events = []
            if WATCHDOG_LOG_FILE.exists():
                try:
                    with open(WATCHDOG_LOG_FILE, 'r') as f:
                        events = json.load(f)
                except (json.JSONDecodeError, IOError):
                    events = []
            events.append(entry)
            # Keep last 1000 events
            if len(events) > 1000:
                events = events[-1000:]
            tmp = WATCHDOG_LOG_FILE.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(events, f, indent=2, default=str)
            tmp.replace(WATCHDOG_LOG_FILE)
    except Exception as e:
        logger.warning(f"Watchdog log write failed: {e}")

    logger.info(f"Watchdog [{event_type}] job={job_id} {details or ''}")


def _retry_job(job):
    """Attempt to retry a stalled job."""
    from models.job import OUTPUT_DIR, UPLOAD_DIR, save_job_checkpoint

    job_id = job.job_id
    retry_count = _job_snapshots.get(job_id, {}).get('retries', 0) + 1

    if retry_count > MAX_RETRIES:
        job.status = 'failed'
        job.error = f'Job stalled and failed after {MAX_RETRIES} retry attempts'
        job.stage = 'Failed (watchdog)'
        save_job_checkpoint(job)
        _log_event('failed_permanently', job_id, {
            'retries_exhausted': True,
            'last_progress': job.progress,
            'last_stage': job.stage,
        })
        # Clean up snapshot
        _job_snapshots.pop(job_id, None)
        return

    _log_event('retry', job_id, {
        'retry_number': retry_count,
        'stalled_at_progress': job.progress,
        'stalled_at_stage': job.stage,
    })

    # Update retry count
    _job_snapshots[job_id] = {
        'last_progress': 0,
        'last_check': time.time(),
        'retries': retry_count,
    }

    # Mark the job metadata so the pipeline knows this is a retry
    job.metadata['watchdog_retry'] = retry_count
    job.metadata['watchdog_retry_at'] = time.time()

    # Reset job state for retry
    job.status = 'pending'
    job.progress = 0
    job.stage = f'Retrying (attempt {retry_count}/{MAX_RETRIES})'
    job.error = None
    save_job_checkpoint(job)

    # Find the original audio file
    audio_path = None
    # Check uploads directory
    upload_dir = UPLOAD_DIR / job_id
    if upload_dir.exists():
        for ext in ('*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg', '*.webm', '*.opus'):
            files = list(upload_dir.glob(ext))
            if files:
                audio_path = files[0]
                break

    if not audio_path:
        job.status = 'failed'
        job.error = 'Retry failed: original audio file not found'
        save_job_checkpoint(job)
        _log_event('retry_failed', job_id, {'reason': 'audio_not_found'})
        _job_snapshots.pop(job_id, None)
        return

    # Cancel any active runners for this job and release the GPU semaphore
    # so the retry thread can acquire it. This is the key fix: without this,
    # the stalled thread still holds the semaphore and the retry blocks forever.
    try:
        from processing.separation import (
            _active_runners, _active_runners_lock, _separation_semaphore,
            _flush_mps_memory
        )

        # Cancel active runners (kills the stalled subprocess)
        with _active_runners_lock:
            for runner in list(_active_runners):
                try:
                    runner.cancel()
                    logger.info(f"Watchdog: cancelled stalled runner for {job_id}")
                except Exception:
                    pass
            _active_runners.clear()

        # Force-release the semaphore so the retry can acquire it.
        # The stalled thread's finally block may also release it, so we use
        # a try/release pattern that won't raise if already released.
        try:
            _separation_semaphore.release()
            logger.info(f"Watchdog: released GPU semaphore for retry of {job_id}")
        except ValueError:
            pass  # Already released

        # Flush GPU memory
        _flush_mps_memory()
        import gc
        gc.collect()
        logger.info(f"Watchdog: flushed GPU memory before retry for {job_id}")
    except Exception as e:
        logger.warning(f"Watchdog: pre-retry cleanup failed: {e}")

    # Re-run the pipeline in a new thread
    try:
        from processing.pipeline import process_audio
        thread = threading.Thread(
            target=process_audio,
            args=(job, audio_path),
            kwargs={
                'enhance_stems': False,
                'stereo_split': False,
                'gp_tabs': True,
                'chord_detection': True,
            },
            daemon=True,
        )
        thread.start()
        logger.info(f"Watchdog: started retry {retry_count} for job {job_id}")
    except Exception as e:
        job.status = 'failed'
        job.error = f'Retry failed: {e}'
        save_job_checkpoint(job)
        _log_event('retry_failed', job_id, {'reason': str(e)})
        _job_snapshots.pop(job_id, None)


def _check_jobs():
    """Check all active jobs for stalls."""
    from models.job import jobs

    now = time.time()

    for job_id, job in list(jobs.items()):
        # Only watch jobs that are actively processing
        if job.status != 'processing':
            # Clean up snapshots for completed/failed jobs
            if job_id in _job_snapshots and job.status in ('completed', 'failed'):
                _job_snapshots.pop(job_id, None)
            continue

        snapshot = _job_snapshots.get(job_id)

        if snapshot is None:
            # First time seeing this job — record baseline
            _job_snapshots[job_id] = {
                'last_progress': job.progress,
                'last_check': now,
                'retries': job.metadata.get('watchdog_retry', 0),
            }
            continue

        # Check if progress has changed since last check
        if job.progress != snapshot['last_progress']:
            # Job is making progress — update snapshot
            _job_snapshots[job_id] = {
                'last_progress': job.progress,
                'last_check': now,
                'retries': snapshot['retries'],
            }
            continue

        # Progress hasn't changed — check if stall threshold exceeded
        time_since_progress = now - snapshot['last_check']
        if time_since_progress >= STALL_THRESHOLD_SECONDS:
            _log_event('stall_detected', job_id, {
                'stalled_at_progress': job.progress,
                'stalled_at_stage': job.stage,
                'stalled_for_seconds': round(time_since_progress),
            })
            _retry_job(job)


def _watchdog_loop():
    """Main watchdog loop — runs forever in a daemon thread."""
    logger.info("Watchdog thread started (checking every %ds, stall threshold %ds)",
                CHECK_INTERVAL_SECONDS, STALL_THRESHOLD_SECONDS)
    while True:
        try:
            _check_jobs()
        except Exception as e:
            logger.error(f"Watchdog check failed: {e}")
        time.sleep(CHECK_INTERVAL_SECONDS)


def start_watchdog(app=None):
    """Start the watchdog background thread.

    Args:
        app: Flask app (unused, accepted for consistency with app factory pattern)
    """
    thread = threading.Thread(target=_watchdog_loop, daemon=True, name='job-watchdog')
    thread.start()
    logger.info("Job watchdog started")
    return thread
