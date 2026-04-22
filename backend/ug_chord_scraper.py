"""
UG Chord Scraper — bulk extract chord vocabularies from Ultimate Guitar.

Scrapes chord sheets, extracts the chord names used in each song,
and builds our own chord database (StemScriber format).

Usage:
    python ug_chord_scraper.py --step search   # Build song URL list from popular artists
    python ug_chord_scraper.py --step scrape    # Scrape chords from each URL
    python ug_chord_scraper.py --step cleanup   # Delete raw UG data, keep only our DB
"""

import argparse
import html
import json
import logging
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent / "training_data" / "chord_db"
SONG_LIST = DB_DIR / "song_urls.json"
CHORD_DB = DB_DIR / "chord_database.json"
RAW_DIR = DB_DIR / "raw_ug"

IMPERSONATE = "chrome131"

# Popular artists across genres for broad chord coverage
SEED_ARTISTS = [
    # Classic Rock
    "The Beatles", "Led Zeppelin", "Pink Floyd", "The Rolling Stones",
    "Queen", "Eagles", "Fleetwood Mac", "AC/DC", "Jimi Hendrix",
    "The Who", "Creedence Clearwater Revival", "The Doors",
    # Jazz / Fusion
    "Steely Dan", "Pat Metheny", "George Benson", "Wes Montgomery",
    "Joe Pass", "Django Reinhardt", "Miles Davis",
    # Classic / Soft Rock
    "Tom Petty", "Bob Dylan", "Neil Young", "Crosby Stills Nash",
    "James Taylor", "Cat Stevens", "Simon and Garfunkel",
    # Blues
    "BB King", "Stevie Ray Vaughan", "Eric Clapton", "Buddy Guy",
    "Muddy Waters", "John Mayer",
    # 90s-2000s Rock
    "Nirvana", "Pearl Jam", "Radiohead", "Red Hot Chili Peppers",
    "Foo Fighters", "Green Day", "Oasis", "Weezer", "Blink 182",
    "Soundgarden", "Alice In Chains",
    # Pop / Singer-Songwriter
    "Ed Sheeran", "Adele", "Taylor Swift", "Coldplay", "U2",
    "Maroon 5", "Bruno Mars", "Billie Eilish",
    # Country
    "Johnny Cash", "Willie Nelson", "Hank Williams",
    "Merle Haggard", "Chris Stapleton",
    # R&B / Soul / Funk
    "Stevie Wonder", "Marvin Gaye", "Earth Wind and Fire",
    "Al Green", "Bill Withers", "Curtis Mayfield",
    # Metal
    "Black Sabbath", "Metallica", "Iron Maiden", "Megadeth",
    # Jam Bands
    "Grateful Dead", "Phish", "Allman Brothers Band",
    "Dave Matthews Band",
    # Reggae / World
    "Bob Marley",
    # Modern / Indie
    "Tame Impala", "Arctic Monkeys", "The Black Keys",
    "Jack White", "Kings of Leon",
    # Prog
    "Yes", "Genesis", "Rush", "King Crimson",
]


def _fetch(url: str) -> str:
    from curl_cffi import requests as cffi_requests
    resp = cffi_requests.get(url, impersonate=IMPERSONATE, timeout=20)
    resp.raise_for_status()
    return resp.text


def _extract_js_store(page_html: str) -> dict | None:
    m = re.search(r'class="js-store"\s+data-content="(.*?)"', page_html)
    if not m:
        return None
    decoded = html.unescape(m.group(1))
    try:
        return json.loads(decoded)
    except json.JSONDecodeError:
        return None


def _extract_chords_from_content(raw_content: str) -> list:
    """Extract unique chord names from UG [ch]...[/ch] markup."""
    seen = set()
    chords = []
    for m in re.finditer(r"\[ch\](.*?)\[/ch\]", raw_content):
        ch = m.group(1).strip()
        if ch and ch not in seen:
            seen.add(ch)
            chords.append(ch)
    return chords


def _extract_chord_sequence(raw_content: str) -> list:
    """Extract chord sequence (with repeats) from UG markup."""
    return [m.group(1).strip() for m in re.finditer(r"\[ch\](.*?)\[/ch\]", raw_content) if m.group(1).strip()]


def step_search():
    """Collect top chord tab URLs from UG's explore endpoint (most popular)."""
    DB_DIR.mkdir(parents=True, exist_ok=True)

    existing = {}
    if SONG_LIST.exists():
        existing = json.loads(SONG_LIST.read_text())

    total_before = len(existing)
    logger.info(f"Starting search. {total_before} songs already indexed.")

    # Paginate through explore endpoint — 50 tabs per page, up to 20 pages = 1000 songs
    for page in range(1, 21):
        explore_url = (
            f"https://www.ultimate-guitar.com/explore?"
            f"type%5B%5D=Chords&order=hitstotal_desc&page={page}"
        )
        try:
            page_html = _fetch(explore_url)
            store = _extract_js_store(page_html)
            if not store:
                logger.warning(f"No js-store on page {page}")
                break

            tabs = (store.get("store", {})
                   .get("page", {})
                   .get("data", {})
                   .get("data", {})
                   .get("tabs", []))

            if not tabs:
                logger.info(f"No more tabs on page {page}")
                break

            page_count = 0
            for tab in tabs:
                if not isinstance(tab, dict):
                    continue
                tab_url = tab.get("tab_url", "")
                song_name = tab.get("song_name", "")
                artist_name = tab.get("artist_name", "")
                rating = tab.get("rating", 0)
                votes = tab.get("votes", 0)

                if not tab_url or not song_name:
                    continue

                if tab_url not in existing:
                    existing[tab_url] = {
                        "artist": artist_name,
                        "song": song_name,
                        "rating": rating,
                        "votes": votes,
                        "url": tab_url,
                        "scraped": False,
                    }
                    page_count += 1

            logger.info(f"Page {page}: {page_count} new songs ({len(tabs)} total on page)")
            SONG_LIST.write_text(json.dumps(existing, indent=2))
            time.sleep(2)

        except Exception as e:
            logger.warning(f"Explore page {page} failed: {e}")
            time.sleep(5)

    total_after = len(existing)
    logger.info(f"Search complete. {total_after} total songs ({total_after - total_before} new)")


def step_scrape():
    """Scrape chord data from each collected URL."""
    if not SONG_LIST.exists():
        logger.error("No song list found. Run --step search first.")
        return

    songs = json.loads(SONG_LIST.read_text())
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Load existing chord DB
    chord_db = {}
    if CHORD_DB.exists():
        chord_db = json.loads(CHORD_DB.read_text())

    to_scrape = {k: v for k, v in songs.items() if not v.get("scraped")}
    logger.info(f"{len(to_scrape)} songs to scrape, {len(chord_db)} already in DB")

    count = 0
    errors = 0
    for url, info in to_scrape.items():
        try:
            page_html = _fetch(url)
            store = _extract_js_store(page_html)

            if not store:
                logger.warning(f"No js-store: {info['song']}")
                errors += 1
                info["scraped"] = True
                continue

            tab_view = (store.get("store", {})
                       .get("page", {})
                       .get("data", {})
                       .get("tab_view", {}))
            wiki_tab = tab_view.get("wiki_tab", {})
            raw_content = wiki_tab.get("content", "")

            if not raw_content:
                info["scraped"] = True
                continue

            # Save raw content temporarily
            safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{info['artist']}_{info['song']}")[:100]
            (RAW_DIR / f"{safe_name}.json").write_text(json.dumps({
                "url": url,
                "content": raw_content,
            }))

            # Extract chord data — this is the facts we keep
            chords_unique = _extract_chords_from_content(raw_content)
            chord_sequence = _extract_chord_sequence(raw_content)

            if chords_unique:
                db_key = f"{info['artist']} - {info['song']}"
                chord_db[db_key] = {
                    "artist": info["artist"],
                    "song": info["song"],
                    "chords": chords_unique,
                    "chord_count": len(chord_sequence),
                    "unique_count": len(chords_unique),
                }
                count += 1

            info["scraped"] = True
            time.sleep(1.0)  # Rate limit

        except Exception as e:
            logger.warning(f"Scrape failed for {info.get('song', '?')}: {e}")
            errors += 1
            time.sleep(2)

        # Save progress every 50 songs
        if (count + errors) % 50 == 0:
            SONG_LIST.write_text(json.dumps(songs, indent=2))
            CHORD_DB.write_text(json.dumps(chord_db, indent=2))
            logger.info(f"Progress: {count} scraped, {errors} errors, {len(chord_db)} in DB")

    # Final save
    SONG_LIST.write_text(json.dumps(songs, indent=2))
    CHORD_DB.write_text(json.dumps(chord_db, indent=2))
    logger.info(f"Scrape complete. {count} new songs, {errors} errors, {len(chord_db)} total in chord DB")


def step_cleanup():
    """Delete raw UG content, keep only our chord database."""
    if RAW_DIR.exists():
        import shutil
        file_count = len(list(RAW_DIR.iterdir()))
        shutil.rmtree(RAW_DIR)
        logger.info(f"Deleted {file_count} raw UG files from {RAW_DIR}")
    else:
        logger.info("No raw UG data to clean up")

    # Remove the raw content field from song_urls if present
    if SONG_LIST.exists():
        songs = json.loads(SONG_LIST.read_text())
        for info in songs.values():
            info.pop("content", None)
        SONG_LIST.write_text(json.dumps(songs, indent=2))

    if CHORD_DB.exists():
        db = json.loads(CHORD_DB.read_text())
        logger.info(f"Chord database retained: {len(db)} songs")
    else:
        logger.warning("No chord database found!")


def show_stats():
    """Show current database stats."""
    if CHORD_DB.exists():
        db = json.loads(CHORD_DB.read_text())
        all_chords = set()
        for entry in db.values():
            all_chords.update(entry.get("chords", []))
        artists = set(e.get("artist", "") for e in db.values())
        print(f"Songs: {len(db)}")
        print(f"Artists: {len(artists)}")
        print(f"Unique chords across DB: {len(all_chords)}")
        print(f"Sample chords: {sorted(list(all_chords))[:30]}")
    else:
        print("No chord database yet.")

    if SONG_LIST.exists():
        songs = json.loads(SONG_LIST.read_text())
        scraped = sum(1 for v in songs.values() if v.get("scraped"))
        print(f"URLs indexed: {len(songs)} ({scraped} scraped)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UG Chord Scraper")
    parser.add_argument("--step", choices=["search", "scrape", "cleanup", "stats"],
                       required=True, help="Pipeline step to run")
    args = parser.parse_args()

    if args.step == "search":
        step_search()
    elif args.step == "scrape":
        step_scrape()
    elif args.step == "cleanup":
        step_cleanup()
    elif args.step == "stats":
        show_stats()
