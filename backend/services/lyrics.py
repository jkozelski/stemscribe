"""
Lyrics fetcher for StemScribe.
Uses lrclib.net — free, no API key required.
Returns synced (LRC) lyrics when available, plain otherwise.
"""

import re
import logging
import requests
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

LRCLIB_API = "https://lrclib.net/api/get"


def fetch_lyrics(title: str, artist: str) -> Optional[Dict]:
    """
    Fetch lyrics for a song.

    Returns:
        dict with keys:
          - type: 'synced' | 'plain'
          - lines: list of {time, end_time, text}
            (time/end_time are None for plain lyrics)
        or None if not found.
    """
    if not title or not artist:
        return None

    try:
        resp = requests.get(
            LRCLIB_API,
            params={"artist_name": artist, "track_name": title},
            timeout=5,
            headers={"User-Agent": "StemScribe/1.0 (music-education-app)"},
        )

        if resp.status_code == 404:
            # Try without "The " prefix on artist
            clean_artist = artist
            if clean_artist.lower().startswith("the "):
                clean_artist = clean_artist[4:]
                resp = requests.get(
                    LRCLIB_API,
                    params={"artist_name": clean_artist, "track_name": title},
                    timeout=5,
                    headers={"User-Agent": "StemScribe/1.0 (music-education-app)"},
                )

        if resp.status_code == 404:
            logger.info(f"lrclib: no lyrics found for '{title}' by '{artist}'")
            return None

        resp.raise_for_status()
        data = resp.json()

        synced = (data.get("syncedLyrics") or "").strip()
        plain = (data.get("plainLyrics") or "").strip()

        if synced:
            lines = _parse_lrc(synced)
            if lines:
                return {"type": "synced", "lines": lines}

        if plain:
            lines = [
                {"time": None, "end_time": None, "text": line}
                for line in plain.split("\n")
                if line.strip()
            ]
            return {"type": "plain", "lines": lines}

        return None

    except requests.RequestException as e:
        logger.warning(f"lrclib request failed for '{title}' / '{artist}': {e}")
        return None
    except Exception as e:
        logger.warning(f"Lyrics fetch error for '{title}' / '{artist}': {e}")
        return None


def _parse_lrc(lrc: str) -> List[Dict]:
    """
    Parse LRC format into [{time, end_time, text}].
    LRC line format: [mm:ss.xx] lyric text
    """
    lines = []
    pattern = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\](.*)")

    for raw_line in lrc.split("\n"):
        m = pattern.match(raw_line.strip())
        if not m:
            continue
        minutes = int(m.group(1))
        seconds = float(m.group(2))
        text = m.group(3).strip()
        if text:  # skip instrumental / empty lines
            lines.append(
                {
                    "time": minutes * 60 + seconds,
                    "end_time": None,  # filled in below
                    "text": text,
                }
            )

    # Fill end_time from the next line's start time
    for i in range(len(lines) - 1):
        lines[i]["end_time"] = lines[i + 1]["time"]
    if lines:
        lines[-1]["end_time"] = lines[-1]["time"] + 5.0  # estimate for last line

    return lines
