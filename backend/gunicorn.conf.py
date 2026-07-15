"""Gunicorn config — replaces CLI flags from the old systemd ExecStart and adds
a worker_exit hook that drains in-flight job threads before the worker exits.

Why this file exists: systemd restarts were orphaning in-flight stem-separation
jobs because the gunicorn worker process terminated mid-job (the thread that
was about to write status=completed died with it). This hook waits for the
job_tracker module's active set to drain before the worker actually exits.

Combined with TimeoutStopSec=600 in stemscribe.service, this gives in-flight
work up to ~9 minutes to finish before systemd hard-kills the process.
"""

# ---- Network ----
bind = "127.0.0.1:5555"

# ---- Workers ----
workers = 1
threads = 8
worker_class = "sync"
preload_app = True

# ---- Timeouts ----
timeout = 600              # request handler timeout
graceful_timeout = 600     # how long gunicorn waits for HTTP handlers to drain on SIGTERM

# ---- Logging ----
accesslog = "-"
errorlog = "-"
loglevel = "info"


def worker_exit(server, worker):
    """Called when a worker process is about to exit. Drain in-flight
    background-thread jobs (stem separation, chord detection, etc.) before
    the process terminates."""
    try:
        from processing.job_tracker import wait_for_drain, active_count, active_ids
        n = active_count()
        if n == 0:
            return
        worker.log.warning(
            f"[gunicorn.worker_exit] {n} in-flight job(s) — draining: "
            f"{[j[:8] for j in active_ids()]}"
        )
        # Leave ~60s margin under systemd's 600s TimeoutStopSec so we never
        # get SIGKILL'd mid-drain.
        ok = wait_for_drain(timeout_sec=540.0)
        if ok:
            worker.log.info("[gunicorn.worker_exit] all jobs drained cleanly")
        else:
            worker.log.warning(
                f"[gunicorn.worker_exit] drain timed out — "
                f"{active_count()} job(s) still active; they may be orphaned"
            )
    except Exception as e:
        worker.log.exception(f"[gunicorn.worker_exit] drain hook failed: {e}")
