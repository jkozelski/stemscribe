"""
Error pattern detection — log processing failures with context and find patterns.

Provides:
- log_error(job_id, error_type, message, ...) -> append to error log
- get_error_patterns() -> analyze logs and return pattern report
- get_recent_errors(limit) -> return most recent errors
"""

import json
import time
import logging
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

ERROR_LOG_FILE = Path(__file__).parent / 'error_log.json'
_lock = threading.Lock()


def log_error(job_id: str, error_type: str, error_message: str,
              song_duration: Optional[float] = None,
              source: Optional[str] = None,
              processing_stage: Optional[str] = None,
              extra: Optional[Dict] = None):
    """Log a processing error with full context.

    Args:
        job_id: The processing job ID
        error_type: Category (e.g. 'separation_failed', 'download_failed', 'timeout')
        error_message: The actual error message/traceback
        song_duration: Duration in seconds if known
        source: Where the audio came from ('youtube', 'upload', 'archive', 'spotify', etc.)
        processing_stage: Pipeline stage where it failed (e.g. 'separation', 'transcription')
        extra: Any additional context
    """
    entry = {
        'timestamp': time.time(),
        'job_id': job_id,
        'error_type': error_type,
        'error_message': str(error_message)[:2000],  # Cap message length
        'song_duration': song_duration,
        'source': source,
        'processing_stage': processing_stage,
    }
    if extra:
        entry['extra'] = extra

    try:
        with _lock:
            errors = _load_errors()
            errors.append(entry)
            # Keep last 5000 errors
            if len(errors) > 5000:
                errors = errors[-5000:]
            _save_errors(errors)
    except Exception as e:
        logger.warning(f"Error tracker: failed to log error: {e}")

    logger.info(f"Error tracked: [{error_type}] job={job_id[:8]} stage={processing_stage} source={source}")


def _load_errors() -> List[Dict]:
    """Load error log from disk."""
    if ERROR_LOG_FILE.exists():
        try:
            with open(ERROR_LOG_FILE, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            pass
    return []


def _save_errors(errors: List[Dict]):
    """Atomically save error log."""
    tmp = ERROR_LOG_FILE.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(errors, f, indent=2, default=str)
    tmp.replace(ERROR_LOG_FILE)


def get_recent_errors(limit: int = 50) -> List[Dict]:
    """Return the most recent errors."""
    with _lock:
        errors = _load_errors()
    return errors[-limit:]


def get_error_patterns() -> Dict[str, Any]:
    """Analyze error log and return patterns.

    Returns:
        Dict with pattern analysis including:
        - total_errors
        - by_type: error counts per type
        - by_source: error counts per source
        - by_stage: error counts per processing stage
        - duration_analysis: failure rates by song duration bucket
        - source_comparison: relative failure rates per source
        - time_analysis: errors per day (last 7 days)
        - top_error_messages: most common error messages
    """
    with _lock:
        errors = _load_errors()

    if not errors:
        return {'total_errors': 0, 'message': 'No errors logged yet'}

    # --- By type ---
    by_type = Counter(e.get('error_type', 'unknown') for e in errors)

    # --- By source ---
    by_source = Counter(e.get('source', 'unknown') for e in errors)

    # --- By stage ---
    by_stage = Counter(e.get('processing_stage', 'unknown') for e in errors)

    # --- Duration analysis ---
    duration_buckets = {
        '0-3min': 0,
        '3-5min': 0,
        '5-8min': 0,
        '8-12min': 0,
        '12min+': 0,
        'unknown': 0,
    }
    for e in errors:
        dur = e.get('song_duration')
        if dur is None:
            duration_buckets['unknown'] += 1
        elif dur <= 180:
            duration_buckets['0-3min'] += 1
        elif dur <= 300:
            duration_buckets['3-5min'] += 1
        elif dur <= 480:
            duration_buckets['5-8min'] += 1
        elif dur <= 720:
            duration_buckets['8-12min'] += 1
        else:
            duration_buckets['12min+'] += 1

    # --- Duration insights ---
    duration_insights = []
    total_with_duration = sum(v for k, v in duration_buckets.items() if k != 'unknown')
    if total_with_duration > 0:
        long_song_failures = duration_buckets.get('8-12min', 0) + duration_buckets.get('12min+', 0)
        long_pct = round(100.0 * long_song_failures / total_with_duration, 1)
        if long_pct > 30:
            duration_insights.append(
                f"{long_pct}% of failures are on songs > 8 minutes"
            )

    # --- Source comparison insights ---
    source_insights = []
    if len(by_source) >= 2:
        # Compare each source against the average
        avg_errors = len(errors) / max(len(by_source), 1)
        for src, count in by_source.most_common():
            if count > avg_errors * 2 and count >= 3:
                ratio = round(count / max(avg_errors, 1), 1)
                source_insights.append(
                    f"{src} songs fail {ratio}x more than average"
                )

    # --- Time analysis (last 7 days) ---
    now = time.time()
    seven_days_ago = now - (7 * 86400)
    daily_counts = defaultdict(int)
    for e in errors:
        ts = e.get('timestamp', 0)
        if ts >= seven_days_ago:
            day_key = time.strftime('%Y-%m-%d', time.localtime(ts))
            daily_counts[day_key] += 1

    # --- Top error messages ---
    msg_counter = Counter()
    for e in errors:
        msg = e.get('error_message', '')
        # Truncate for grouping
        short_msg = msg[:100] if msg else 'unknown'
        msg_counter[short_msg] += 1
    top_messages = [{'message': msg, 'count': cnt} for msg, cnt in msg_counter.most_common(10)]

    return {
        'total_errors': len(errors),
        'by_type': dict(by_type.most_common()),
        'by_source': dict(by_source.most_common()),
        'by_stage': dict(by_stage.most_common()),
        'duration_buckets': duration_buckets,
        'duration_insights': duration_insights,
        'source_insights': source_insights,
        'errors_last_7_days': dict(sorted(daily_counts.items())),
        'top_error_messages': top_messages,
    }
