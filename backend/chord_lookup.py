"""
StemScribe Chord Library — fast filesystem-based chord chart lookup.

Checks chord_library/{artist-slug}/{song-slug}.json for pre-scraped chord charts.
"""

import json
import logging
import os
import re
from pathlib import Path

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

chord_library_bp = Blueprint("chord_library", __name__)

LIBRARY_DIR = Path(__file__).parent / "chord_library"


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug: lowercase, hyphens, no special chars."""
    if not text:
        return ""
    s = text.lower().strip()
    # Replace & with 'and'
    s = s.replace("&", "and")
    # Remove apostrophes and quotes (don't replace with hyphen)
    s = re.sub(r"[''\"]+", "", s)
    # Replace non-alphanumeric with hyphens
    s = re.sub(r"[^a-z0-9]+", "-", s)
    # Collapse multiple hyphens
    s = re.sub(r"-+", "-", s)
    # Strip leading/trailing hyphens
    s = s.strip("-")
    return s


def _strip_the(text: str) -> str:
    """Remove leading 'The ' from artist names."""
    return re.sub(r"^the\s+", "", text, flags=re.IGNORECASE)


def _clean_youtube_title(title: str) -> str:
    """Strip YouTube junk from song titles."""
    # Remove common YouTube suffixes
    junk = [
        r'\(official\s*(audio|video|music\s*video|lyric\s*video|visualizer)\)',
        r'\[official\s*(audio|video|music\s*video|lyric\s*video|visualizer)\]',
        r'\(audio\)', r'\[audio\]',
        r'\(lyrics?\)', r'\[lyrics?\]',
        r'\(remaster(ed)?\s*\d*\)', r'\[remaster(ed)?\s*\d*\]',
        r'\(\d{4}\s*remaster(ed)?\)', r'\[\d{4}\s*remaster(ed)?\]',
        r'\(hd\)', r'\[hd\]', r'\(hq\)', r'\[hq\]',
        r'\(live\)', r'\[live\]',
        r'\(full\s*song\)', r'\(full\s*album\)',
        r'\(feat\.?\s*[^)]*\)', r'\[feat\.?\s*[^]]*\]',
        r'-\s*topic$',
    ]
    cleaned = title
    for pattern in junk:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip().rstrip('-').strip()


def _clean_filename_title(title: str) -> str:
    """Strip file upload junk: durations, clip markers, format tags."""
    junk = [
        r'\b\d+\s*min\b', r'\b\d+m\b', r'\bclip\b', r'\bedit\b',
        r'\bscratch\b', r'\bfinal\b', r'\bmix\b', r'\bmaster\b',
        r'\bv\d+\b', r'\bcopy\b', r'\b\(\d+\)\b',
    ]
    cleaned = title
    for pattern in junk:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip().rstrip('-').strip()


def _strip_artist_prefix(title: str, artist: str) -> str:
    """Strip 'Artist - ' prefix from title if present (common in YouTube metadata).

    Handles cases where the title prefix doesn't exactly match the artist:
    e.g. title="The Jimi Hendrix Experience - Purple Haze" with artist="Jimi Hendrix"
    """
    if not title:
        return title
    # Check if title has an "Artist - Title" pattern (dash separator)
    dash_match = re.match(r'^(.+?)\s*[-–—]\s+(.+)$', title)
    if not dash_match:
        return title
    prefix = dash_match.group(1).strip()
    rest = dash_match.group(2).strip()
    if not rest:
        return title
    # If we have an artist, check if it's related to the prefix
    if artist:
        artist_lower = artist.lower()
        prefix_lower = prefix.lower()
        # Exact match or prefix contains artist name or vice versa
        if (artist_lower == prefix_lower or
                artist_lower in prefix_lower or
                prefix_lower in artist_lower or
                _slugify(_strip_the(artist)) in _slugify(_strip_the(prefix)) or
                _slugify(_strip_the(prefix)) in _slugify(_strip_the(artist))):
            return rest
    return title


def lookup_chord_library(title: str, artist: str) -> dict | None:
    """
    Look up a song in the chord library.

    Tries exact match first, then fuzzy matching (strip 'The', partial matches,
    clean YouTube title junk, strip artist prefix from title).
    Returns the parsed JSON dict or None if not found.
    """
    if not title:
        return None

    # Clean YouTube junk and file upload junk from title
    clean_title = _clean_youtube_title(title)
    clean_title = _clean_filename_title(clean_title)
    # Strip "Artist - " prefix from title (common in YouTube metadata)
    clean_title = _strip_artist_prefix(clean_title, artist)
    title_slug = _slugify(clean_title)
    artist_slug = _slugify(artist) if artist else ""

    if not title_slug:
        return None

    # Build a list of title slugs to try (exact first, then partial)
    title_slugs = [title_slug]
    # Also try without trailing junk (e.g. "scarlet-begonias-4min" → "scarlet-begonias")
    # by progressively stripping trailing slug segments
    parts = title_slug.split('-')
    if len(parts) > 1:
        for i in range(len(parts) - 1, 0, -1):
            candidate_slug = '-'.join(parts[:i])
            if len(candidate_slug) >= 3:
                title_slugs.append(candidate_slug)

    def _try_artist_dirs(slug):
        """Try to find slug.json in matching artist directories."""
        if not LIBRARY_DIR.is_dir():
            return None
        stripped_slug = _slugify(_strip_the(artist)) if artist else ""
        for artist_dir in LIBRARY_DIR.iterdir():
            if not artist_dir.is_dir():
                continue
            candidate = artist_dir / f"{slug}.json"
            if not candidate.is_file():
                continue
            # No artist filter — return first match
            if not artist_slug:
                return _load_json(candidate)
            dir_name = artist_dir.name
            # Fuzzy: slug contains dir name or vice versa
            if artist_slug in dir_name or dir_name in artist_slug:
                return _load_json(candidate)
            if stripped_slug and (stripped_slug in dir_name or dir_name in stripped_slug):
                return _load_json(candidate)
            # Check if any significant word from artist matches the folder
            artist_words = [w for w in re.split(r'[^a-z0-9]+', artist_slug) if len(w) > 2]
            dir_words = set(re.split(r'[^a-z0-9]+', dir_name))
            matches = sum(1 for w in artist_words if w in dir_words)
            if matches >= 1 and matches / max(len(artist_words), 1) >= 0.4:
                return _load_json(candidate)
        return None

    for slug in title_slugs:
        # --- Attempt 1: Exact match with artist ---
        if artist_slug:
            exact_path = LIBRARY_DIR / artist_slug / f"{slug}.json"
            if exact_path.is_file():
                return _load_json(exact_path)

        # --- Attempt 2: Strip "The" from artist ---
        if artist:
            stripped_artist = _strip_the(artist)
            stripped_slug = _slugify(stripped_artist)
            if stripped_slug and stripped_slug != artist_slug:
                path = LIBRARY_DIR / stripped_slug / f"{slug}.json"
                if path.is_file():
                    return _load_json(path)

        # --- Attempt 3: Search all artist folders ---
        result = _try_artist_dirs(slug)
        if result:
            return result

    # --- Attempt 4: Partial filename match in all artist dirs ---
    # e.g. library has "scarlet-begonias.json", we search for files starting with the slug
    if LIBRARY_DIR.is_dir():
        for artist_dir in LIBRARY_DIR.iterdir():
            if not artist_dir.is_dir():
                continue
            for f in artist_dir.glob('*.json'):
                # Check if library filename starts with our slug or vice versa
                lib_slug = f.stem
                if lib_slug.startswith(title_slug) or title_slug.startswith(lib_slug):
                    if not artist_slug:
                        return _load_json(f)
                    dir_name = artist_dir.name
                    if artist_slug in dir_name or dir_name in artist_slug:
                        return _load_json(f)

    return None


def _load_json(path: Path) -> dict | None:
    """Load and parse a JSON file, returning None on error."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load chord library file {path}: {e}")
        return None


@chord_library_bp.route("/api/chords/library", methods=["GET"])
def chord_library_lookup():
    """Look up a song in the StemScribe chord library."""
    title = request.args.get("title", "").strip()
    artist = request.args.get("artist", "").strip()

    if not title:
        return jsonify({"error": "Missing 'title' parameter"}), 400

    result = lookup_chord_library(title, artist)

    if result is None:
        return jsonify({"error": "Song not found in chord library"}), 404

    return jsonify(result)
