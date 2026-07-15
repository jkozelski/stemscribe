"""In-flight job tracker — lets the gunicorn worker_exit hook wait for active
background-thread jobs to finish before the worker process terminates.

Without this, every `systemctl restart stemscribe` kills the worker process
while stem-separation / chord-detection / MIDI-transcription threads are still
running. The worker dies before it can write `status=completed` to disk, leaving
the job stuck at the last-known progress percent forever. The user sees the
"Tour Bus is rollin'..." spinner with no termination.

Hit four times in production on 2026-05-26 — recovered jobs manually by
flipping status=completed + chord_chart_ready=True on disk, but the underlying
race condition needed an actual fix. This is it.

Flow:
  1. Each background thread wraps its target with register(job_id) / unregister(job_id)
  2. On gunicorn worker shutdown (worker_exit hook in gunicorn.conf.py), the hook
     calls wait_for_drain() which blocks until all registered jobs finish or
     the timeout expires.
  3. systemd's TimeoutStopSec is bumped to 600s to give the drain room before
     SIGKILL forces termination.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active = set()                # job_ids currently mid-processing
_drain_event = threading.Event()
_drain_event.set()             # starts "drained" (no active jobs)


def register(job_id: str) -> None:
    """Mark a job as actively processing. Called at the start of a worker thread."""
    if not job_id:
        return
    with _lock:
        _active.add(job_id)
        _drain_event.clear()
    logger.info(f"[job_tracker] register {job_id[:8]} (active: {len(_active)})")


def unregister(job_id: str) -> None:
    """Mark a job as finished. Called in the thread's finally block."""
    if not job_id:
        return
    with _lock:
        _active.discard(job_id)
        empty = not _active
        if empty:
            _drain_event.set()
    logger.info(f"[job_tracker] unregister {job_id[:8]} (active: {len(_active)}{', DRAINED' if empty else ''})")


def active_count() -> int:
    with _lock:
        return len(_active)


def active_ids() -> list:
    with _lock:
        return list(_active)


def wait_for_drain(timeout_sec: float = 600.0) -> bool:
    """Block until all active jobs have unregistered, or timeout expires.

    Returns True if drained cleanly, False if the timeout fired with jobs still
    in-flight (those will be orphaned the old way and need manual recovery).
    """
    n = active_count()
    if n == 0:
        return True

    logger.warning(
        f"[job_tracker] waiting up to {timeout_sec:.0f}s for {n} in-flight "
        f"job(s) to drain: {[j[:8] for j in active_ids()]}"
    )
    start = time.time()
    last_n = n
    while True:
        elapsed = time.time() - start
        remaining = timeout_sec - elapsed
        if remaining <= 0:
            still = active_count()
            logger.warning(
                f"[job_tracker] drain TIMED OUT after {elapsed:.1f}s — "
                f"{still} job(s) still active: {[j[:8] for j in active_ids()]}"
            )
            return False
        # Wake periodically so we can log progress
        if _drain_event.wait(timeout=min(5.0, remaining)):
            logger.info(f"[job_tracker] drained cleanly in {elapsed:.1f}s")
            return True
        cur = active_count()
        if cur != last_n:
            logger.info(f"[job_tracker] drain progress: {last_n} → {cur} (elapsed {elapsed:.0f}s)")
            last_n = cur


def tracked(job_id: str, target, *args, **kwargs):
    """Helper to wrap a thread target so register/unregister always fire.
    Usage:
        threading.Thread(target=tracked, args=(job.job_id, process_audio, job, path, ...))
    """
    register(job_id)
    try:
        return target(*args, **kwargs)
    finally:
        unregister(job_id)
