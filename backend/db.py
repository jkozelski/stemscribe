"""
Database connection pool for Supabase/Postgres.

Usage:
    from db import get_db, query_one, query_all, execute

Environment:
    DATABASE_URL — Postgres connection string from Supabase
"""

import os
import logging
from contextlib import contextmanager

from psycopg2 import pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

_pool = None


def get_pool():
    """Lazily initialize and return the connection pool."""
    global _pool
    if _pool is None:
        database_url = os.environ.get('DATABASE_URL')
        if not database_url:
            raise RuntimeError(
                "DATABASE_URL environment variable is not set. "
                "Set it to your Supabase Postgres connection string."
            )
        _pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=database_url,
        )
        logger.info("Database connection pool initialized")
    return _pool


@contextmanager
def get_db():
    """Get a database connection from the pool. Auto-commits on success, rolls back on error."""
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def query_one(sql, params=None):
    """Execute a query and return a single row as a dict, or None."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def query_all(sql, params=None):
    """Execute a query and return all rows as a list of dicts."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def execute(sql, params=None):
    """Execute a statement (INSERT, UPDATE, DELETE) and return rowcount."""
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.rowcount


def execute_returning(sql, params=None):
    """Execute a statement with RETURNING clause and return the row as a dict."""
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def close_pool():
    """Close all connections in the pool. Call on app shutdown."""
    global _pool
    if _pool is not None:
        _pool.closeall()
        _pool = None
        logger.info("Database connection pool closed")
