#!/usr/bin/env python3
"""
Simple SQL migration runner for StemScriber.

Usage:
    python migrate.py              # Apply all pending migrations
    python migrate.py --status     # Show migration status

Migrations are numbered SQL files in backend/migrations/:
    001_initial_schema.sql
    002_add_feature_x.sql
    ...

Requires DATABASE_URL environment variable.
"""

import os
import sys
import glob
import re
import logging
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / 'migrations'


def get_connection():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        logger.error("DATABASE_URL environment variable is not set")
        sys.exit(1)
    return psycopg2.connect(database_url)


def ensure_migrations_table(conn):
    """Create schema_migrations table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
    conn.commit()


def get_applied_versions(conn):
    """Return set of already-applied migration version numbers."""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("SELECT version, name, applied_at FROM schema_migrations ORDER BY version")
        rows = cur.fetchall()
    return {r['version']: r for r in rows}


def get_pending_migrations(applied):
    """Find SQL files in migrations/ that haven't been applied yet."""
    pattern = str(MIGRATIONS_DIR / '*.sql')
    files = sorted(glob.glob(pattern))
    pending = []
    for filepath in files:
        filename = os.path.basename(filepath)
        match = re.match(r'^(\d+)_(.+)\.sql$', filename)
        if not match:
            logger.warning(f"Skipping non-conforming file: {filename}")
            continue
        version = int(match.group(1))
        if version not in applied:
            pending.append((version, filename, filepath))
    return pending


def apply_migration(conn, version, name, filepath):
    """Apply a single migration file."""
    with open(filepath, 'r') as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
    logger.info(f"  Applied: {name}")


def show_status(conn, applied):
    """Print current migration status."""
    if not applied:
        logger.info("No migrations applied yet.")
    else:
        logger.info("Applied migrations:")
        for version in sorted(applied.keys()):
            row = applied[version]
            logger.info(f"  [{version:03d}] {row['name']} — {row['applied_at']}")

    pending = get_pending_migrations(applied)
    if pending:
        logger.info(f"\nPending migrations ({len(pending)}):")
        for version, name, _ in pending:
            logger.info(f"  [{version:03d}] {name}")
    else:
        logger.info("\nAll migrations are up to date.")


def main():
    conn = get_connection()
    ensure_migrations_table(conn)
    applied = get_applied_versions(conn)

    if '--status' in sys.argv:
        show_status(conn, applied)
        conn.close()
        return

    pending = get_pending_migrations(applied)

    if not pending:
        logger.info("All migrations are up to date.")
        conn.close()
        return

    logger.info(f"Applying {len(pending)} migration(s)...")
    for version, name, filepath in pending:
        try:
            apply_migration(conn, version, name, filepath)
        except Exception as e:
            logger.error(f"  FAILED: {name} — {e}")
            conn.close()
            sys.exit(1)

    logger.info("All migrations applied successfully.")
    conn.close()


if __name__ == '__main__':
    main()
