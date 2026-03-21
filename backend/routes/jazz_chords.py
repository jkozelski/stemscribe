"""
Jazz chord database — serves chord progressions for 1300+ jazz standards
parsed from the JazzStandards dataset (iReal Pro format).

Provides fuzzy search and full chord data in the same format as Songsterr
so the frontend can consume it identically.
"""

import json
import logging
import os
from difflib import SequenceMatcher
from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

jazz_bp = Blueprint("jazz", __name__)

# Load jazz database at import time
_JAZZ_DB = []
_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "jazz_chords.json")

try:
    with open(_DATA_PATH) as f:
        _JAZZ_DB = json.load(f)
    logger.info("Jazz chord database loaded: %d standards", len(_JAZZ_DB))
except Exception as e:
    logger.warning("Jazz chord database not available: %s", e)


def _fuzzy_score(query, title):
    """Score how well a query matches a title (0-1). Higher = better."""
    q = query.lower().strip()
    t = title.lower().strip()

    if q == t:
        return 1.0
    if t.startswith(q):
        return 0.95
    if q in t:
        return 0.85
    if t in q:
        return 0.80

    return SequenceMatcher(None, q, t).ratio()


def _format_as_songsterr_compat(song):
    """Format a jazz song entry to match the Songsterr chordEvents response format."""
    return {
        "songId": None,
        "title": song["title"],
        "artist": song.get("composer", ""),
        "composer": song.get("composer", ""),
        "key": song.get("key", ""),
        "tempo": song.get("tempo", 120),
        "timeSignature": song.get("timeSignature", [4, 4]),
        "style": song.get("style", "Swing"),
        "sections": song.get("sections", []),
        "measures": song.get("measures", []),
        "chordEvents": song.get("chordEvents", []),
        "chordsUsed": song.get("chordsUsed", []),
        "totalMeasures": song.get("totalMeasures", 0),
        "lyrics": "",
        "syncedLyrics": [],
        "source": "jazz_standards",
        "btc_validated": False,
        "btc_changes": [],
        "btc_stats": {},
    }


def search_jazz_db(query, limit=10):
    """Search the jazz database by title. Returns list of matches with scores."""
    if not _JAZZ_DB or not query:
        return []

    q = query.lower().strip()
    scored = []

    for song in _JAZZ_DB:
        title = song.get("title", "")
        score = _fuzzy_score(q, title)

        composer = song.get("composer", "").lower()
        if q in composer:
            score = max(score, 0.7)

        if score > 0.3:
            scored.append((score, song))

    scored.sort(key=lambda x: -x[0])
    return scored[:limit]


def get_jazz_song(title):
    """Get a specific jazz song by exact or best-match title."""
    if not _JAZZ_DB or not title:
        return None

    q = title.lower().strip()

    for song in _JAZZ_DB:
        if song["title"].lower().strip() == q:
            return song

    results = search_jazz_db(title, limit=1)
    if results and results[0][0] >= 0.7:
        return results[0][1]

    return None


# API Endpoints

@jazz_bp.route("/api/jazz/search", methods=["GET"])
def jazz_search():
    """Fuzzy search the jazz standards database by title or composer."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No search query provided"}), 400

    if not _JAZZ_DB:
        return jsonify({"error": "Jazz database not loaded"}), 503

    results = search_jazz_db(query, limit=int(request.args.get("limit", 10)))

    songs = []
    for score, song in results:
        songs.append({
            "title": song["title"],
            "composer": song.get("composer", ""),
            "key": song.get("key", ""),
            "style": song.get("style", ""),
            "timeSignature": song.get("timeSignature", [4, 4]),
            "totalMeasures": song.get("totalMeasures", 0),
            "chordsUsed": song.get("chordsUsed", []),
            "score": round(score, 3),
            "source": "jazz_standards",
        })

    return jsonify({"results": songs, "total": len(songs), "query": query})


@jazz_bp.route("/api/jazz/chords/<path:song_title>", methods=["GET"])
def jazz_chords(song_title):
    """Return full chord progression for a jazz standard.

    Response format matches Songsterr chordEvents response
    so the frontend can consume it identically.
    """
    song = get_jazz_song(song_title)
    if not song:
        return jsonify({"error": "Jazz standard not found: " + song_title}), 404

    return jsonify(_format_as_songsterr_compat(song))


@jazz_bp.route("/api/jazz/stats", methods=["GET"])
def jazz_stats():
    """Return stats about the jazz chord database."""
    if not _JAZZ_DB:
        return jsonify({"error": "Jazz database not loaded"}), 503

    styles = {}
    keys = {}
    composers = {}
    for song in _JAZZ_DB:
        s = song.get("style", "Unknown")
        styles[s] = styles.get(s, 0) + 1
        k = song.get("key", "Unknown") or "Unknown"
        keys[k] = keys.get(k, 0) + 1
        c = song.get("composer", "Unknown") or "Unknown"
        composers[c] = composers.get(c, 0) + 1

    top_composers = sorted(composers.items(), key=lambda x: -x[1])[:20]

    return jsonify({
        "totalSongs": len(_JAZZ_DB),
        "styles": styles,
        "keys": keys,
        "topComposers": dict(top_composers),
        "source": "JazzStandards (iReal Pro)",
    })
