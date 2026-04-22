"""
Lyrics helpers.

Karaoke / lyrics-serving routes were removed 2026-04-22 per legal guidance:
serving composition lyrics (from any source, including LRCLIB or Whisper
transcription) requires a licensed lyrics provider that we have not yet
integrated. The blueprint is intentionally empty so no `/api/lyrics/*`
endpoints are exposed.

Only the `_parse_lrc` helper remains — it is imported by
`routes.chord_sheet` to parse already-in-hand LRC strings for chord-sheet
layout. It does not fetch anything.
"""

import re
import logging

from flask import Blueprint

logger = logging.getLogger(__name__)

# Blueprint is kept registered but exposes no routes. This is intentional:
# removing the blueprint registration in app.py would be fine too, but
# leaving an empty blueprint makes the intent obvious in code review.
lyrics_bp = Blueprint("lyrics", __name__)


def _parse_lrc(lrc_content):
    """Parse LRC format into list of {time, text} dicts.

    Retained as a pure string parser (no network calls). Consumed by
    routes.chord_sheet when it already has LRC content in memory.
    """
    if not lrc_content:
        return []

    lines = []
    pattern = r'\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)'

    for line in lrc_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if not match:
            continue

        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frac = match.group(3) or '0'
        if len(frac) == 2:
            ms = int(frac) * 10
        elif len(frac) == 3:
            ms = int(frac)
        else:
            ms = int(frac)

        total_seconds = minutes * 60 + seconds + ms / 1000
        text = match.group(4).strip()
        lines.append({'time': round(total_seconds, 3), 'text': text})

    lines.sort(key=lambda x: x['time'])
    return lines
