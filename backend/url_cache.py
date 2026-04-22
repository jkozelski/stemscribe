"""
URL-based result caching for StemScribe.

When someone processes a YouTube URL that's already been processed before,
serve the cached results instantly instead of reprocessing.

First user processes "Purple Haze" -> costs $0.03 on Modal GPU.
Next 1000 users who paste the same URL -> instant results, $0.00 cost.
"""

import re
import sqlite3
import logging
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / 'url_cache.db'


def _get_db():
    """Get a SQLite connection (creates table if needed)."""
    conn = sqlite3.connect(str(DB_PATH), timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS url_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url_normalized TEXT UNIQUE NOT NULL,
            source_job_id TEXT NOT NULL,
            title TEXT,
            artist TEXT,
            stem_count INTEGER,
            has_chords BOOLEAN,
            has_gp BOOLEAN,
            has_midi BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            hit_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


def normalize_url(url: str) -> str | None:
    """
    Normalize a YouTube URL to canonical form: "youtube:VIDEO_ID".

    Handles:
      - youtube.com/watch?v=X
      - youtu.be/X
      - music.youtube.com/watch?v=X
      - youtube.com/embed/X
      - youtube.com/v/X
      - youtube.com/shorts/X
      - URLs with extra query params (playlists, timestamps, etc.)

    Returns None if the URL is not a recognized YouTube format.
    """
    if not url:
        return None

    url = url.strip()

    # Try to extract video ID
    video_id = None

    parsed = urlparse(url)
    hostname = (parsed.hostname or '').lower()

    # youtu.be/VIDEO_ID
    if hostname == 'youtu.be':
        video_id = parsed.path.lstrip('/')
        # Remove any trailing path segments
        if '/' in video_id:
            video_id = video_id.split('/')[0]

    # youtube.com variants
    elif hostname in ('www.youtube.com', 'youtube.com', 'm.youtube.com',
                      'music.youtube.com', 'www.music.youtube.com'):
        path = parsed.path

        # /watch?v=VIDEO_ID
        if path == '/watch':
            qs = parse_qs(parsed.query)
            video_id = qs.get('v', [None])[0]

        # /embed/VIDEO_ID or /v/VIDEO_ID or /shorts/VIDEO_ID
        elif path.startswith(('/embed/', '/v/', '/shorts/')):
            parts = path.split('/')
            if len(parts) >= 3:
                video_id = parts[2]

    # Validate video ID format (11 chars, alphanumeric + - + _)
    if video_id:
        video_id = video_id.strip()
        if re.match(r'^[A-Za-z0-9_-]{10,12}$', video_id):
            return f"youtube:{video_id}"

    return None


def check_cache(url: str) -> str | None:
    """
    Check if a URL has been processed before.

    Returns the source_job_id if cached, None otherwise.
    Increments hit_count on a cache hit.
    """
    normalized = normalize_url(url)
    if not normalized:
        return None

    try:
        conn = _get_db()
        row = conn.execute(
            "SELECT source_job_id FROM url_cache WHERE url_normalized = ?",
            (normalized,)
        ).fetchone()

        if row:
            conn.execute(
                "UPDATE url_cache SET hit_count = hit_count + 1 WHERE url_normalized = ?",
                (normalized,)
            )
            conn.commit()
            conn.close()
            return row['source_job_id']

        conn.close()
        return None
    except Exception as e:
        logger.warning(f"URL cache check failed: {e}")
        return None


def add_to_cache(url: str, job_id: str, title: str = '', artist: str = '',
                 stem_count: int = 0, has_chords: bool = False,
                 has_gp: bool = False, has_midi: bool = False):
    """Add a successfully processed URL to the cache."""
    normalized = normalize_url(url)
    if not normalized:
        logger.debug(f"Cannot cache non-YouTube URL: {url}")
        return False

    try:
        conn = _get_db()
        conn.execute(
            """INSERT OR IGNORE INTO url_cache
               (url_normalized, source_job_id, title, artist, stem_count,
                has_chords, has_gp, has_midi)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (normalized, job_id, title, artist, stem_count,
             has_chords, has_gp, has_midi)
        )
        conn.commit()
        conn.close()
        logger.info(f"Cached URL result: {normalized} -> job {job_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to add URL to cache: {e}")
        return False


def clone_job(source_job_id: str, new_job_id: str):
    """
    Create a new ProcessingJob that references the same output files as the source job.

    The cloned job points to the SAME output directory (source_job_id's outputs),
    so no file copying is needed.

    Returns the cloned ProcessingJob, or None if the source job can't be loaded.
    """
    from models.job import ProcessingJob, OUTPUT_DIR, load_job_from_disk

    source_dir = OUTPUT_DIR / source_job_id
    if not source_dir.exists():
        logger.warning(f"Source job directory missing: {source_dir}")
        return None

    source_job = load_job_from_disk(source_dir)
    if not source_job:
        logger.warning(f"Could not load source job: {source_job_id}")
        return None

    if source_job.status != 'completed':
        logger.warning(f"Source job not completed: {source_job_id} (status={source_job.status})")
        return None

    # Verify at least some stem files still exist
    valid_stems = {k: v for k, v in source_job.stems.items() if Path(v).exists()}
    if not valid_stems:
        logger.warning(f"Source job has no valid stem files: {source_job_id}")
        return None

    # Create the cloned job
    cloned = ProcessingJob(
        job_id=new_job_id,
        filename=source_job.filename,
        source_url=source_job.source_url,
    )

    # Copy all result data — pointing to the SAME files (no copying)
    cloned.status = 'completed'
    cloned.progress = 100
    cloned.stage = 'Complete (cached)'
    cloned.stems = valid_stems
    cloned.enhanced_stems = {k: v for k, v in source_job.enhanced_stems.items() if Path(v).exists()}
    cloned.midi_files = {k: v for k, v in source_job.midi_files.items() if Path(v).exists()}
    cloned.musicxml_files = {k: v for k, v in source_job.musicxml_files.items() if Path(v).exists()}
    cloned.gp_files = {k: v for k, v in source_job.gp_files.items() if Path(v).exists()}
    cloned.sub_stems = source_job.sub_stems
    cloned.metadata = dict(source_job.metadata)
    cloned.metadata['cached_from'] = source_job_id
    cloned.chord_progression = source_job.chord_progression
    cloned.detected_key = source_job.detected_key
    cloned.transcription_quality = source_job.transcription_quality
    cloned.pro_tabs = source_job.pro_tabs
    cloned.transcription_mode = source_job.transcription_mode
    cloned.created_at = time.time()

    return cloned


def get_cache_stats() -> dict:
    """Return cache statistics for the /api/cache/stats endpoint."""
    try:
        conn = _get_db()

        total_cached = conn.execute("SELECT COUNT(*) as c FROM url_cache").fetchone()['c']
        total_hits = conn.execute("SELECT COALESCE(SUM(hit_count), 0) as h FROM url_cache").fetchone()['h']

        top_songs = conn.execute(
            """SELECT title, artist, hit_count as hits
               FROM url_cache
               ORDER BY hit_count DESC
               LIMIT 10"""
        ).fetchall()

        conn.close()

        estimated_savings = f"${total_hits * 0.03:.2f}"

        return {
            'total_cached': total_cached,
            'total_hits': total_hits,
            'top_songs': [dict(row) for row in top_songs],
            'estimated_savings': estimated_savings,
        }
    except Exception as e:
        logger.warning(f"Failed to get cache stats: {e}")
        return {
            'total_cached': 0,
            'total_hits': 0,
            'top_songs': [],
            'estimated_savings': '$0.00',
        }
