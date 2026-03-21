"""
Ultimate Guitar chord scraper — search and extract chord sheets.

Uses curl_cffi to bypass Cloudflare with browser TLS fingerprint impersonation.
"""

import html
import json
import logging
import re
import urllib.parse

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)

ug_bp = Blueprint("ug", __name__)

UG_SEARCH_URL = "https://www.ultimate-guitar.com/search.php"
UG_TABS_BASE = "https://tabs.ultimate-guitar.com"
IMPERSONATE = "chrome131"


def _fetch(url: str) -> str:
    """Fetch a URL with browser TLS impersonation."""
    from curl_cffi import requests as cffi_requests

    resp = cffi_requests.get(url, impersonate=IMPERSONATE, timeout=15)
    resp.raise_for_status()
    return resp.text


def _extract_js_store(page_html: str) -> dict | None:
    """Extract JSON from <div class="js-store" data-content="...">."""
    m = re.search(r'class="js-store"\s+data-content="(.*?)"', page_html)
    if not m:
        return None
    decoded = html.unescape(m.group(1))
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        logger.error("Failed to parse js-store JSON")
        return None


def _parse_chord_content(raw: str) -> str:
    """Convert UG [ch]G[/ch] markup to readable chord sheet."""
    text = raw
    text = re.sub(r"\[ch\](.*?)\[/ch\]", r"\1", text)
    text = re.sub(r"\[tab\]", "", text)
    text = re.sub(r"\[/tab\]", "", text)
    text = re.sub(r"\[/?[a-z_]+\]", "", text, flags=re.IGNORECASE)
    return text.strip()


def _scrape_ug(search_query: str) -> dict:
    """Search UG and extract chord data from the highest-rated result."""
    encoded = urllib.parse.quote(search_query)
    search_url = (
        f"{UG_SEARCH_URL}?search_type=title&value={encoded}&type%5B%5D=300"
    )

    # 1. Search page
    search_html = _fetch(search_url)
    store = _extract_js_store(search_html)

    if not store:
        raise ValueError("Could not extract search results from UG")

    # Find chord results
    results = (
        store.get("store", {})
        .get("page", {})
        .get("data", {})
        .get("results", [])
    )

    candidates = []
    for item in results:
        if isinstance(item, dict) and item.get("type") == "Chords":
            candidates.append(item)
        elif isinstance(item, list):
            for sub in item:
                if isinstance(sub, dict) and sub.get("type") == "Chords":
                    candidates.append(sub)

    if not candidates:
        raise ValueError("No chord results found on UG")

    # Sort by rating
    candidates.sort(key=lambda x: x.get("rating", 0), reverse=True)
    best = candidates[0]
    tab_url = best.get("tab_url", "")

    if not tab_url:
        raise ValueError("No tab URL in search results")

    # 2. Fetch tab page
    tab_html = _fetch(tab_url)
    tab_store = _extract_js_store(tab_html)

    if not tab_store:
        raise ValueError("Could not extract tab data from UG page")

    tab_view = (
        tab_store.get("store", {})
        .get("page", {})
        .get("data", {})
        .get("tab_view", {})
    )

    # 3. Extract chord content
    wiki_tab = tab_view.get("wiki_tab", {})
    raw_content = wiki_tab.get("content", "")
    applicature = tab_view.get("applicature", {})

    if not raw_content:
        raise ValueError("No chord content found on tab page")

    chord_sheet = _parse_chord_content(raw_content)

    # Unique chords in order of appearance
    seen = set()
    chords_used = []
    for m in re.finditer(r"\[ch\](.*?)\[/ch\]", raw_content):
        ch = m.group(1)
        if ch not in seen:
            seen.add(ch)
            chords_used.append(ch)

    return {
        "title": best.get("song_name", ""),
        "artist": best.get("artist_name", ""),
        "rating": best.get("rating", 0),
        "votes": best.get("votes", 0),
        "tab_url": tab_url,
        "chords_used": chords_used,
        "chord_sheet": chord_sheet,
        "content": raw_content,
        "applicature": applicature,
    }


@ug_bp.route("/api/ug/chords", methods=["GET"])
def ug_chords():
    """Search Ultimate Guitar and return chord sheet + diagram data."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No search query provided. Use ?q=artist+title"}), 400

    try:
        result = _scrape_ug(query)
        return jsonify(result)
    except ValueError as e:
        logger.warning(f"UG scrape returned no results: {e}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"UG scrape failed: {e}", exc_info=True)
        return jsonify({"error": "Failed to scrape Ultimate Guitar", "details": str(e)}), 502
