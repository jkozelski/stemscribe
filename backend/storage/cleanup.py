"""
R2 storage cleanup — TTL enforcement for old files.

Can be run as a standalone script (cron / Railway scheduled task)
or called from the app.

Policy:
    - Source audio: delete after 24 hours
    - Stems/MIDI/GP for free users: delete after 7 days
    - Stems/MIDI/GP for paid users: delete after 30 days
    - Token blacklist entries: prune expired rows

Usage:
    python -m storage.cleanup              # Run all cleanup tasks
    python -m storage.cleanup --dry-run    # Preview what would be deleted
"""

import os
import sys
import logging
from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def cleanup_expired_tokens(dry_run: bool = False) -> int:
    """Delete expired entries from the token_blacklist table."""
    from db import execute, query_all

    if dry_run:
        rows = query_all(
            "SELECT COUNT(*) as cnt FROM token_blacklist WHERE expires_at < NOW()"
        )
        count = rows[0]['cnt'] if rows else 0
        logger.info(f"[DRY RUN] Would prune {count} expired token blacklist entries")
        return count

    count = execute("DELETE FROM token_blacklist WHERE expires_at < NOW()")
    logger.info(f"Pruned {count} expired token blacklist entries")
    return count


def cleanup_old_jobs(days_free: int = 7, days_paid: int = 30,
                     source_hours: int = 24, dry_run: bool = False) -> dict:
    """Delete R2 files for expired jobs based on user plan.

    Args:
        days_free: Days to keep files for free users.
        days_paid: Days to keep files for paid users.
        source_hours: Hours to keep source audio files.
        dry_run: If True, don't actually delete anything.

    Returns:
        Dict with counts: {'source_deleted', 'jobs_deleted', 'files_deleted'}.
    """
    from db import query_all
    from storage.r2 import delete_file, delete_job_files, source_key, list_job_files

    stats = {'source_deleted': 0, 'jobs_deleted': 0, 'files_deleted': 0}
    now = datetime.now(timezone.utc)

    # 1. Delete old source audio (all users, 24-hour TTL)
    source_cutoff = now - timedelta(hours=source_hours)
    old_source_jobs = query_all(
        """
        SELECT j.id, j.r2_upload_key, j.source_filename
        FROM jobs j
        WHERE j.r2_upload_key IS NOT NULL
          AND j.created_at < %s
        """,
        (source_cutoff,),
    )
    for job in old_source_jobs:
        if dry_run:
            logger.info(f"[DRY RUN] Would delete source: {job['r2_upload_key']}")
        else:
            delete_file(job['r2_upload_key'])
        stats['source_deleted'] += 1

    # 2. Delete full job files for expired free-tier jobs
    free_cutoff = now - timedelta(days=days_free)
    expired_free_jobs = query_all(
        """
        SELECT j.id FROM jobs j
        LEFT JOIN users u ON j.user_id = u.id
        WHERE j.created_at < %s
          AND (u.plan IS NULL OR u.plan = 'free')
          AND j.status IN ('complete', 'failed')
        """,
        (free_cutoff,),
    )
    for job in expired_free_jobs:
        job_id = str(job['id'])
        if dry_run:
            files = list_job_files(job_id)
            logger.info(f"[DRY RUN] Would delete {len(files)} files for free job {job_id}")
            stats['files_deleted'] += len(files)
        else:
            count = delete_job_files(job_id)
            stats['files_deleted'] += count
        stats['jobs_deleted'] += 1

    # 3. Delete full job files for expired paid-tier jobs
    paid_cutoff = now - timedelta(days=days_paid)
    expired_paid_jobs = query_all(
        """
        SELECT j.id FROM jobs j
        LEFT JOIN users u ON j.user_id = u.id
        WHERE j.created_at < %s
          AND u.plan IN ('premium', 'pro')
          AND j.status IN ('complete', 'failed')
        """,
        (paid_cutoff,),
    )
    for job in expired_paid_jobs:
        job_id = str(job['id'])
        if dry_run:
            files = list_job_files(job_id)
            logger.info(f"[DRY RUN] Would delete {len(files)} files for paid job {job_id}")
            stats['files_deleted'] += len(files)
        else:
            count = delete_job_files(job_id)
            stats['files_deleted'] += count
        stats['jobs_deleted'] += 1

    logger.info(
        f"Cleanup: {stats['source_deleted']} sources, "
        f"{stats['jobs_deleted']} jobs, "
        f"{stats['files_deleted']} files {'(dry run)' if dry_run else 'deleted'}"
    )
    return stats


def main():
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        logger.info("=== DRY RUN MODE ===\n")

    logger.info("--- Token blacklist cleanup ---")
    cleanup_expired_tokens(dry_run=dry_run)

    logger.info("\n--- R2 file cleanup ---")
    cleanup_old_jobs(dry_run=dry_run)

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
