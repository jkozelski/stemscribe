#!/usr/bin/env python3
"""Pre-seed the detector_router cache from existing prod outputs.

Walks /opt/stemscribe/outputs/ on the VPS (or any provided directory), extracts
(title, artist) from each job's metadata, and routes them via Claude in a
single batch. The cache gets warmed BEFORE Week 2's prod flip so the first
real user upload of an already-processed song hits cache (0 latency) instead
of cold-calling Claude (~2s).

Usage (locally, dry-run):
    ./venv311/bin/python backend/scripts/preseed_router_cache.py --outputs-dir /path/to/outputs --dry-run

Usage (on VPS, real):
    ssh root@5.161.203.112 "cd /opt/stemscribe && ./venv311/bin/python \\
        backend/scripts/preseed_router_cache.py --outputs-dir /opt/stemscribe/outputs"

Cost estimate: ~$0.003 per cold lookup. 500 jobs = ~$1.50. 5000 jobs = ~$15.
Set --max-calls to bound spend.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Allow running from repo root or backend/
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from processing.detector_router import route_detector, _load_cache, _cache_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preseed_router_cache")


def _extract_title_artist(metadata_path: Path) -> tuple[str, str] | None:
    """Pull title + artist from a job_metadata.json. Falls back to parsing
    the filename for the common 'Artist - Title.mp3' pattern."""
    try:
        m = json.loads(metadata_path.read_text())
    except Exception as e:
        logger.debug(f"unreadable: {metadata_path}: {e}")
        return None

    meta = m.get("metadata") or {}
    title = meta.get("title") or ""
    artist = meta.get("artist") or ""
    if title and artist:
        return title.strip(), artist.strip()

    # Fallback — parse filename "Artist - Title.mp3"
    fname = m.get("filename") or m.get("original_filename") or ""
    if " - " in fname:
        artist_part, _, title_part = fname.partition(" - ")
        title_part = title_part.rsplit(".", 1)[0]  # drop extension
        return title_part.strip(), artist_part.strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-dir", type=Path, required=True,
                    help="Path to outputs/ dir containing per-job folders")
    ap.add_argument("--max-calls", type=int, default=2000,
                    help="Cap on cold Claude calls to bound spend (default 2000 = ~$6)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be routed without calling Claude")
    ap.add_argument("--sleep-ms", type=int, default=100,
                    help="Sleep between Claude calls to be a polite API citizen")
    args = ap.parse_args()

    outputs = args.outputs_dir
    if not outputs.exists():
        logger.error(f"outputs dir not found: {outputs}")
        sys.exit(2)

    # Pull every (title, artist) pair from existing jobs
    songs = {}  # cache_key → (title, artist)
    for meta_path in outputs.glob("*/job_metadata.json"):
        pair = _extract_title_artist(meta_path)
        if not pair:
            continue
        title, artist = pair
        if not title:
            continue
        key = _cache_key(title, artist)
        songs.setdefault(key, (title, artist))

    logger.info(f"Found {len(songs)} unique (title, artist) pairs across {outputs}")

    # Filter out songs already in cache
    cache = _load_cache()
    cold = {k: v for k, v in songs.items() if k not in cache}
    logger.info(f"  {len(cache)} already cached, {len(cold)} cold")

    if args.dry_run:
        logger.info("DRY RUN — songs that would be routed (first 20):")
        for k, (title, artist) in list(cold.items())[:20]:
            print(f"  '{title}' by '{artist}'")
        return

    if not cold:
        logger.info("Nothing to do — cache fully warm.")
        return

    if len(cold) > args.max_calls:
        logger.warning(f"Capping at --max-calls={args.max_calls} (would have done {len(cold)})")
        cold = dict(list(cold.items())[:args.max_calls])

    # Route each
    jazz_count = general_count = fallback_count = 0
    for i, (key, (title, artist)) in enumerate(cold.items(), 1):
        decision = route_detector(title, artist)
        if decision["source"] == "fallback":
            fallback_count += 1
        elif decision["path"] == "jazz":
            jazz_count += 1
        else:
            general_count += 1
        if i % 25 == 0:
            logger.info(
                f"  [{i}/{len(cold)}] jazz={jazz_count}, general={general_count}, "
                f"fallback={fallback_count}"
            )
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    logger.info(
        f"DONE. Routed {len(cold)} songs: "
        f"jazz={jazz_count}, general={general_count}, fallback={fallback_count}. "
        f"Approx spend: ${len(cold) * 0.003:.2f}"
    )


if __name__ == "__main__":
    main()
