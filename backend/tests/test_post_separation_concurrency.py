"""Tests for the post-separation concurrency cap.

Regression coverage for the launch-blocker bug surfaced by the 2026-04-25
audit: 6 concurrent jobs on CPX41 hit watchdog stalls because nothing
throttled post-separation work. With the cap, slot 5+ waits in 'Queued
for processing' state until a slot opens.
"""

import threading
import time

from backend.processing import pipeline


def test_semaphore_initial_capacity():
    """Cap is set to 4."""
    assert pipeline._POST_SEPARATION_MAX_CONCURRENT == 4


def test_semaphore_allows_up_to_cap():
    """4 jobs can hold the semaphore concurrently."""
    sem = pipeline._post_separation_semaphore
    # Drain to a known state: try to acquire all 4 slots non-blocking
    acquired = []
    for _ in range(pipeline._POST_SEPARATION_MAX_CONCURRENT):
        if sem.acquire(blocking=False):
            acquired.append(True)
    try:
        assert len(acquired) == pipeline._POST_SEPARATION_MAX_CONCURRENT
        # 5th non-blocking acquire should fail
        assert not sem.acquire(blocking=False)
    finally:
        for _ in acquired:
            sem.release()


def test_blocking_acquire_unblocks_on_release():
    """A queued thread unblocks when a slot is released."""
    sem = pipeline._post_separation_semaphore
    # Saturate
    held = []
    for _ in range(pipeline._POST_SEPARATION_MAX_CONCURRENT):
        if sem.acquire(blocking=False):
            held.append(True)

    unblocked = threading.Event()

    def waiter():
        sem.acquire()  # blocks
        try:
            unblocked.set()
        finally:
            sem.release()

    t = threading.Thread(target=waiter, daemon=True)
    t.start()

    # Should not be unblocked yet — all slots taken
    assert not unblocked.wait(timeout=0.05)

    # Release one slot
    sem.release()
    held.pop()

    # Waiter should now proceed
    assert unblocked.wait(timeout=1.0), "waiter did not unblock after release"

    t.join(timeout=1.0)

    # Cleanup any remaining held slots
    for _ in held:
        sem.release()
