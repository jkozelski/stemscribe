"""Email capture / waitlist endpoint — saves to SQLite."""

import os
import re
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

waitlist_bp = Blueprint('waitlist', __name__)

DB_PATH = Path(os.environ.get(
    'WAITLIST_DB',
    Path(__file__).parent.parent / 'data' / 'waitlist.db',
))


def _get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS waitlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            source TEXT DEFAULT 'landing',
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


@waitlist_bp.route('/api/waitlist', methods=['POST'])
def join_waitlist():
    data = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()

    if not email or not re.match(r'^[^@\s]+@[^@\s]+\.[^@\s]+$', email):
        return jsonify({'error': 'Valid email required'}), 400

    source = (data.get('source') or 'landing').strip()[:50]

    try:
        conn = _get_db()
        conn.execute(
            "INSERT OR IGNORE INTO waitlist (email, source, created_at) VALUES (?, ?, ?)",
            (email, source, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM waitlist").fetchone()[0]
        conn.close()
        logger.info(f"Waitlist signup: {email} (total: {count})")
        return jsonify({'ok': True, 'count': count}), 200
    except Exception as e:
        logger.error(f"Waitlist error: {e}")
        return jsonify({'error': 'Server error'}), 500
