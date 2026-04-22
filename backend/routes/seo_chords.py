"""
SEO Chord Pages — server-rendered HTML chord charts for search engines.

Routes:
  /chords/                         → Browse all artists (A-Z index)
  /chords/<artist>/                → All songs by artist
  /chords/<artist>/<song>          → Full chord chart page
  /sitemap-chords.xml              → Sitemap of all chord pages
"""

import json
import logging
import re
from pathlib import Path
from flask import Blueprint, Response, abort, request, url_for
from markupsafe import escape

logger = logging.getLogger(__name__)

seo_chords_bp = Blueprint("seo_chords", __name__)

LIBRARY_DIR = Path(__file__).resolve().parent.parent / "chord_library"
SITE_ORIGIN = "https://stemscriber.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deslugify(slug: str) -> str:
    """Convert a slug back to title case: 'led-zeppelin' → 'Led Zeppelin'."""
    return slug.replace("-", " ").title()


def _load_chart(artist_slug: str, song_slug: str) -> dict | None:
    """Load a chord chart JSON from the library.  Returns None on failure."""
    path = LIBRARY_DIR / artist_slug / f"{song_slug}.json"
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("Failed to load %s: %s", path, e)
        return None


def _get_all_artists() -> list[dict]:
    """Return sorted list of {'slug': ..., 'name': ..., 'count': ...}."""
    if not LIBRARY_DIR.is_dir():
        return []
    artists = []
    for d in sorted(LIBRARY_DIR.iterdir()):
        if not d.is_dir():
            continue
        songs = list(d.glob("*.json"))
        if songs:
            artists.append({
                "slug": d.name,
                "name": _deslugify(d.name),
                "count": len(songs),
            })
    return artists


def _get_songs_for_artist(artist_slug: str) -> list[dict]:
    """Return sorted list of {'slug': ..., 'title': ...} for an artist."""
    artist_dir = LIBRARY_DIR / artist_slug
    if not artist_dir.is_dir():
        return []
    songs = []
    for f in sorted(artist_dir.glob("*.json")):
        songs.append({
            "slug": f.stem,
            "title": _deslugify(f.stem),
        })
    return songs


# ---------------------------------------------------------------------------
# Shared HTML skeleton
# ---------------------------------------------------------------------------

_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
{meta}
<link rel="icon" type="image/png" sizes="32x32" href="/images/favicon.png">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Righteous&display=swap" rel="stylesheet">
<style>
:root {{
  --bg-deep: #0d0d12;
  --bg-dark: #13131a;
  --bg-card: #1a1a24;
  --psych-orange: #ff7b54;
  --psych-pink: #ff6b9d;
  --psych-purple: #c678dd;
  --psych-blue: #61afef;
  --text: #e8e4df;
  --text-dim: #7a7a85;
}}
*, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: 'Space Grotesk', sans-serif;
  background: var(--bg-deep);
  color: var(--text);
  min-height: 100vh;
  line-height: 1.6;
}}
a {{ color: var(--psych-orange); text-decoration: none; }}
a:hover {{ text-decoration: underline; }}

/* Nav */
.seo-nav {{
  position: sticky; top: 0; z-index: 100;
  padding: 0.8rem 2rem;
  display: flex; align-items: center; justify-content: space-between;
  background: rgba(13,13,18,0.92);
  backdrop-filter: blur(12px);
  border-bottom: 1px solid rgba(255,255,255,0.05);
}}
.seo-nav .logo {{
  display: flex; align-items: center; gap: 0.5rem; text-decoration: none;
}}
.seo-nav .logo-text {{
  font-family: 'Righteous', cursive; font-size: 1.3rem;
  background: linear-gradient(90deg, var(--psych-orange), var(--psych-pink), var(--psych-purple));
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}}
.seo-nav .nav-links a {{
  color: var(--text-dim); margin-left: 1.5rem; font-size: 0.9rem;
}}
.seo-nav .nav-links a:hover {{ color: var(--text); text-decoration: none; }}

/* Container */
.seo-container {{
  max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem 4rem;
}}

/* Breadcrumbs */
.breadcrumbs {{
  font-size: 0.85rem; color: var(--text-dim); margin-bottom: 1.5rem;
}}
.breadcrumbs a {{ color: var(--text-dim); }}
.breadcrumbs a:hover {{ color: var(--psych-orange); }}

/* Chart header */
.chart-header {{
  margin-bottom: 2rem;
}}
.chart-header h1 {{
  font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem;
}}
.chart-header .artist-name {{
  font-size: 1.1rem; color: var(--psych-orange);
}}
.chart-meta {{
  display: flex; gap: 1.5rem; margin-top: 0.8rem; font-size: 0.85rem; color: var(--text-dim);
}}
.chart-meta span {{ display: flex; align-items: center; gap: 0.3rem; }}

/* Chords used */
.chords-used {{
  display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0 2rem;
}}
.chords-used .chord-badge {{
  background: var(--bg-card); border: 1px solid rgba(255,255,255,0.08);
  padding: 0.25rem 0.7rem; border-radius: 6px; font-size: 0.85rem;
  font-weight: 600; color: var(--psych-blue);
}}

/* Sections */
.chart-section {{
  margin-bottom: 1.5rem;
}}
.section-label {{
  font-size: 0.8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.08em; color: var(--psych-purple);
  margin-bottom: 0.4rem; padding: 0.2rem 0.6rem;
  background: rgba(198,120,221,0.08); border-radius: 4px;
  display: inline-block;
}}
.chart-line {{
  font-family: 'Courier New', Courier, monospace;
  white-space: pre-wrap; font-size: 0.92rem; line-height: 1.5;
  margin-bottom: 0.1rem;
}}
.chart-line.chords {{
  color: var(--psych-orange); font-weight: 700;
}}
.chart-line.lyrics {{
  color: var(--text); margin-bottom: 0.6rem;
}}

/* CTA */
.cta-box {{
  margin-top: 2.5rem; padding: 1.5rem 2rem;
  background: linear-gradient(135deg, rgba(255,123,84,0.08), rgba(198,120,221,0.08));
  border: 1px solid rgba(255,123,84,0.15); border-radius: 12px;
  text-align: center;
}}
.cta-box h3 {{
  font-size: 1.1rem; margin-bottom: 0.5rem;
}}
.cta-box p {{
  font-size: 0.9rem; color: var(--text-dim); margin-bottom: 1rem;
}}
.cta-btn {{
  display: inline-block; padding: 0.7rem 2rem;
  background: linear-gradient(135deg, var(--psych-orange), var(--psych-pink));
  color: #fff; font-weight: 600; border-radius: 8px; font-size: 0.95rem;
  text-decoration: none; transition: opacity 0.2s;
}}
.cta-btn:hover {{ opacity: 0.9; text-decoration: none; }}

/* Artist index */
.letter-group {{ margin-bottom: 2rem; }}
.letter-group h2 {{
  font-size: 1.4rem; color: var(--psych-orange); border-bottom: 1px solid rgba(255,255,255,0.06);
  padding-bottom: 0.3rem; margin-bottom: 0.8rem;
}}
.artist-list {{ list-style: none; }}
.artist-list li {{ margin-bottom: 0.4rem; }}
.artist-list .song-count {{ color: var(--text-dim); font-size: 0.85rem; margin-left: 0.5rem; }}

/* Song list */
.song-list {{ list-style: none; columns: 2; column-gap: 2rem; }}
.song-list li {{ margin-bottom: 0.4rem; break-inside: avoid; }}

/* Footer */
.seo-footer {{
  text-align: center; padding: 2rem 1rem; font-size: 0.8rem; color: var(--text-dim);
  border-top: 1px solid rgba(255,255,255,0.04); margin-top: 3rem;
}}

@media (max-width: 640px) {{
  .seo-container {{ padding: 1rem; }}
  .chart-header h1 {{ font-size: 1.5rem; }}
  .song-list {{ columns: 1; }}
  .chart-meta {{ flex-wrap: wrap; gap: 0.8rem; }}
}}
</style>
</head>
"""

_NAV = """\
<body>
<nav class="seo-nav">
  <a href="/" class="logo">
    <img src="/images/logomark.png" alt="StemScriber" width="32" height="32" style="border-radius:6px;">
    <span class="logo-text">StemScriber</span>
  </a>
  <div class="nav-links">
    <a href="/chords/">Chord Library</a>
    <a href="/app">Open App</a>
  </div>
</nav>
"""

_FOOTER = """\
<footer class="seo-footer">
  <p>&copy; 2025 StemScriber &middot;
    <a href="/">Home</a> &middot;
    <a href="/chords/">All Chords</a> &middot;
    <a href="/app">Open App</a> &middot;
    <a href="/terms.html">Terms</a> &middot;
    <a href="/privacy.html">Privacy</a> &middot;
    <a href="/dmca.html">DMCA</a>
  </p>
</footer>
</body></html>
"""


def _page(meta: str, body: str) -> str:
    return _HEAD.format(meta=meta) + _NAV + body + _FOOTER


# ---------------------------------------------------------------------------
# Route: /chords/  — Artist A-Z index
# ---------------------------------------------------------------------------

@seo_chords_bp.route("/chords/")
def chords_index():
    artists = _get_all_artists()
    total_songs = sum(a["count"] for a in artists)

    meta = (
        f'<title>Guitar Chords &amp; Lyrics — {len(artists)} Artists, {total_songs:,} Songs | StemScriber</title>\n'
        f'<meta name="description" content="Free guitar chord charts and lyrics for {total_songs:,} songs by {len(artists)} artists. '
        f'Browse chords A-Z or practice with stem separation on StemScriber.">\n'
        f'<link rel="canonical" href="{SITE_ORIGIN}/chords/">\n'
        '<meta property="og:type" content="website">\n'
        f'<meta property="og:title" content="Guitar Chords &amp; Lyrics — {total_songs:,} Songs | StemScriber">\n'
        f'<meta property="og:url" content="{SITE_ORIGIN}/chords/">'
    )

    # Group by first letter
    groups: dict[str, list] = {}
    for a in artists:
        letter = a["name"][0].upper() if a["name"] else "#"
        if not letter.isalpha():
            letter = "#"
        groups.setdefault(letter, []).append(a)

    body = '<div class="seo-container">\n'
    body += f'<h1>Guitar Chords &amp; Lyrics</h1>\n'
    body += f'<p style="color:var(--text-dim); margin-bottom:2rem;">{len(artists)} artists &middot; {total_songs:,} songs</p>\n'

    for letter in sorted(groups.keys()):
        body += f'<div class="letter-group"><h2>{escape(letter)}</h2><ul class="artist-list">\n'
        for a in groups[letter]:
            body += (
                f'<li><a href="/chords/{escape(a["slug"])}/">{escape(a["name"])}</a>'
                f'<span class="song-count">({a["count"]})</span></li>\n'
            )
        body += '</ul></div>\n'

    body += '</div>\n'
    return _page(meta, body)


# ---------------------------------------------------------------------------
# Route: /chords/<artist>/  — Song list for one artist
# ---------------------------------------------------------------------------

@seo_chords_bp.route("/chords/<artist_slug>/")
def chords_artist(artist_slug: str):
    songs = _get_songs_for_artist(artist_slug)
    if not songs:
        abort(404)

    artist_name = _deslugify(artist_slug)

    meta = (
        f'<title>{escape(artist_name)} — Guitar Chords &amp; Lyrics | StemScriber</title>\n'
        f'<meta name="description" content="All {len(songs)} {escape(artist_name)} chord charts with lyrics. '
        f'Free guitar chords — practice with stem separation on StemScriber.">\n'
        f'<link rel="canonical" href="{SITE_ORIGIN}/chords/{escape(artist_slug)}/">\n'
        '<meta property="og:type" content="music.musician">\n'
        f'<meta property="og:title" content="{escape(artist_name)} Chords | StemScriber">\n'
        f'<meta property="og:url" content="{SITE_ORIGIN}/chords/{escape(artist_slug)}/">'
    )

    body = '<div class="seo-container">\n'
    body += f'<div class="breadcrumbs"><a href="/chords/">Chords</a> / {escape(artist_name)}</div>\n'
    body += f'<h1>{escape(artist_name)} — Chords &amp; Lyrics</h1>\n'
    body += f'<p style="color:var(--text-dim); margin-bottom:1.5rem;">{len(songs)} songs</p>\n'
    body += '<ul class="song-list">\n'
    for s in songs:
        body += f'<li><a href="/chords/{escape(artist_slug)}/{escape(s["slug"])}">{escape(s["title"])}</a></li>\n'
    body += '</ul>\n'

    body += (
        '<div class="cta-box">'
        f'<h3>Practice {escape(artist_name)} with stem separation</h3>'
        '<p>Upload any song and isolate guitar, bass, drums, or vocals. Practice along at any speed.</p>'
        '<a href="/app" class="cta-btn">Open StemScriber</a>'
        '</div>\n'
    )

    body += '</div>\n'
    return _page(meta, body)


# ---------------------------------------------------------------------------
# Route: /chords/<artist>/<song>  — Full chord chart page
# ---------------------------------------------------------------------------

@seo_chords_bp.route("/chords/<artist_slug>/<song_slug>")
def chords_song(artist_slug: str, song_slug: str):
    chart = _load_chart(artist_slug, song_slug)
    if chart is None:
        abort(404)

    title = chart.get("title") or _deslugify(song_slug)
    artist = chart.get("artist") or _deslugify(artist_slug)
    key = chart.get("key") or ""
    capo = chart.get("capo", 0)
    chords_used = chart.get("chords_used", [])
    sections = chart.get("sections", [])

    page_title = f"{title} Chords — {artist}"
    desc = f"Chords and lyrics for {title} by {artist}."
    if chords_used:
        desc += f" Uses: {', '.join(chords_used[:8])}."
    desc += " Free on StemScriber."

    meta = (
        f'<title>{escape(page_title)} | StemScriber</title>\n'
        f'<meta name="description" content="{escape(desc)}">\n'
        f'<link rel="canonical" href="{SITE_ORIGIN}/chords/{escape(artist_slug)}/{escape(song_slug)}">\n'
        '<meta property="og:type" content="music.song">\n'
        f'<meta property="og:title" content="{escape(page_title)}">\n'
        f'<meta property="og:description" content="{escape(desc)}">\n'
        f'<meta property="og:url" content="{SITE_ORIGIN}/chords/{escape(artist_slug)}/{escape(song_slug)}">'
    )

    # --- Build body ---
    body = '<div class="seo-container">\n'

    # Breadcrumbs
    body += (
        '<div class="breadcrumbs">'
        f'<a href="/chords/">Chords</a> / '
        f'<a href="/chords/{escape(artist_slug)}/">{escape(artist)}</a> / '
        f'{escape(title)}'
        '</div>\n'
    )

    # Header
    body += '<div class="chart-header">\n'
    body += f'<h1>{escape(title)} Chords</h1>\n'
    body += f'<div class="artist-name">{escape(artist)}</div>\n'
    body += '<div class="chart-meta">'
    if key:
        body += f'<span>Key: <strong>{escape(key)}</strong></span>'
    if capo:
        body += f'<span>Capo: <strong>Fret {capo}</strong></span>'
    if chords_used:
        body += f'<span>{len(chords_used)} chords</span>'
    body += '</div>\n'
    body += '</div>\n'

    # Chords used badges
    if chords_used:
        body += '<div class="chords-used">\n'
        for c in chords_used:
            body += f'<span class="chord-badge">{escape(c)}</span>\n'
        body += '</div>\n'

    # Sections
    for section in sections:
        section_name = section.get("name", "")
        lines = section.get("lines", [])
        body += '<div class="chart-section">\n'
        if section_name:
            body += f'<div class="section-label">{escape(section_name)}</div>\n'
        for line in lines:
            chords_text = line.get("chords") or ""
            lyrics_text = line.get("lyrics") or ""
            if chords_text:
                body += f'<div class="chart-line chords">{escape(chords_text)}</div>\n'
            if lyrics_text:
                body += f'<div class="chart-line lyrics">{escape(lyrics_text)}</div>\n'
        body += '</div>\n'

    # CTA
    body += (
        '<div class="cta-box">'
        f'<h3>Practice "{escape(title)}" with stem separation</h3>'
        '<p>Upload this song to StemScriber to isolate guitar, bass, drums, or vocals. '
        'Slow it down, loop sections, and practice along with the stems.</p>'
        f'<a href="/app" class="cta-btn">Practice This Song &rarr;</a>'
        '</div>\n'
    )

    # Internal links — more songs by this artist
    more_songs = _get_songs_for_artist(artist_slug)
    other_songs = [s for s in more_songs if s["slug"] != song_slug][:12]
    if other_songs:
        body += f'<h3 style="margin-top:2.5rem; margin-bottom:1rem;">More {escape(artist)} Chords</h3>\n'
        body += '<ul class="song-list">\n'
        for s in other_songs:
            body += f'<li><a href="/chords/{escape(artist_slug)}/{escape(s["slug"])}">{escape(s["title"])}</a></li>\n'
        body += '</ul>\n'

    body += '</div>\n'
    return _page(meta, body)


# ---------------------------------------------------------------------------
# Route: /sitemap-chords.xml  — Sitemap of all chord pages
# ---------------------------------------------------------------------------

@seo_chords_bp.route("/sitemap-chords.xml")
def sitemap_chords():
    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    # Index page
    lines.append(f"  <url><loc>{SITE_ORIGIN}/chords/</loc><changefreq>weekly</changefreq><priority>0.8</priority></url>")

    if LIBRARY_DIR.is_dir():
        for artist_dir in sorted(LIBRARY_DIR.iterdir()):
            if not artist_dir.is_dir():
                continue
            artist_slug = artist_dir.name
            songs = sorted(artist_dir.glob("*.json"))
            if not songs:
                continue

            # Artist page
            lines.append(
                f"  <url><loc>{SITE_ORIGIN}/chords/{artist_slug}/</loc>"
                f"<changefreq>monthly</changefreq><priority>0.6</priority></url>"
            )

            # Song pages
            for song_file in songs:
                song_slug = song_file.stem
                lines.append(
                    f"  <url><loc>{SITE_ORIGIN}/chords/{artist_slug}/{song_slug}</loc>"
                    f"<changefreq>monthly</changefreq><priority>0.7</priority></url>"
                )

    lines.append("</urlset>")
    xml = "\n".join(lines)
    return Response(xml, mimetype="application/xml")
