#!/usr/bin/env python3
"""
StemScribe YouTube Pre-Processor

Searches YouTube for songs from a list and processes them through the
StemScribe pipeline, pre-populating the URL cache with stems, chords, and tabs.

Usage:
    python preprocess_songs.py                          # Process all songs
    python preprocess_songs.py --dry-run                # Just show YouTube matches
    python preprocess_songs.py --limit 10 --delay 10    # First 10 songs, 10s delay
    python preprocess_songs.py --file my_songs.txt      # Custom song list
"""

import argparse
import csv
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

# Add backend to path so we can import url_cache
sys.path.insert(0, str(Path(__file__).parent.parent))
from url_cache import normalize_url, check_cache

logger = logging.getLogger("preprocess")

# Defaults
DEFAULT_SONG_FILE = Path(__file__).parent / "top_500_songs.txt"
DEFAULT_API_BASE = "http://localhost:5555"
DEFAULT_DELAY = 5
DEFAULT_POLL_INTERVAL = 10
DEFAULT_POLL_TIMEOUT = 600  # 10 minutes
MODAL_COST_PER_SONG = 0.03
RESULTS_FILE = Path(__file__).parent.parent.parent / "docs" / "preprocess-results.csv"


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_song_file(filepath: Path) -> list[dict]:
    """
    Parse a song list file. Format: "Song Title - Artist Name"
    Lines starting with # are comments. Blank lines are skipped.
    """
    songs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Split on " - " (first occurrence only)
            parts = line.split(" - ", 1)
            if len(parts) != 2:
                logger.warning(f"  Line {line_num}: Can't parse '{line}', skipping")
                continue
            title, artist = parts[0].strip(), parts[1].strip()
            if title and artist:
                songs.append({"title": title, "artist": artist})
    return songs


def search_youtube(title: str, artist: str) -> dict | None:
    """
    Use yt-dlp to search YouTube and return video info.
    Returns dict with 'url', 'video_id', 'yt_title' or None if not found.
    """
    query = f"ytsearch1:{title} {artist} official audio"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--get-url",
                "--get-id",
                "--get-title",
                query,
                "--no-playlist",
                "--no-download",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.debug(f"  yt-dlp stderr: {result.stderr.strip()}")
            return None

        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if len(lines) < 2:
            return None

        # yt-dlp outputs: title, then video_id, then URL(s)
        yt_title = lines[0]
        video_id = lines[1]

        # Validate video_id looks right (11 chars, alphanumeric + dash/underscore)
        if not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
            # Sometimes order varies — try to find the ID
            for line in lines:
                if re.match(r"^[a-zA-Z0-9_-]{11}$", line):
                    video_id = line
                    break
            else:
                logger.warning(f"  Could not extract video ID from yt-dlp output")
                return None

        url = f"https://www.youtube.com/watch?v={video_id}"

        return {
            "url": url,
            "video_id": video_id,
            "yt_title": yt_title,
        }

    except subprocess.TimeoutExpired:
        logger.warning(f"  yt-dlp timed out searching for: {title} - {artist}")
        return None
    except FileNotFoundError:
        logger.error("  yt-dlp not found! Install with: pip install yt-dlp")
        sys.exit(1)
    except Exception as e:
        logger.warning(f"  yt-dlp error: {e}")
        return None


def check_api_health(api_base: str) -> bool:
    """Check if the StemScribe API is running."""
    try:
        resp = requests.get(f"{api_base}/api/health", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False
    except Exception:
        return False


def wait_for_api(api_base: str, max_wait: int = 60):
    """Wait for the API to come online."""
    logger.info(f"Waiting for API at {api_base}...")
    for i in range(max_wait // 5):
        if check_api_health(api_base):
            logger.info("API is online!")
            return True
        time.sleep(5)
    return False


def submit_song(api_base: str, youtube_url: str) -> str | None:
    """
    Submit a YouTube URL to StemScribe for processing.
    Returns job_id or None on failure.
    """
    try:
        resp = requests.post(
            f"{api_base}/api/url",
            json={
                "url": youtube_url,
                "chord_detection": True,
                "gp_tabs": True,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("job_id")
        else:
            logger.warning(f"  API returned {resp.status_code}: {resp.text[:200]}")
            return None
    except requests.ConnectionError:
        logger.error("  API connection refused — is the server running?")
        return None
    except Exception as e:
        logger.warning(f"  Submit error: {e}")
        return None


def poll_job(api_base: str, job_id: str, poll_interval: int = 10, timeout: int = 600) -> dict:
    """
    Poll a job until completion or timeout.
    Returns status dict with 'status', 'stems', 'has_chords', 'has_gp'.
    """
    max_polls = timeout // poll_interval
    for i in range(max_polls):
        time.sleep(poll_interval)
        try:
            resp = requests.get(f"{api_base}/api/status/{job_id}", timeout=30)
            if resp.status_code != 200:
                logger.warning(f"  Status check failed: {resp.status_code}")
                continue
            status = resp.json()
            job_status = status.get("status", "unknown")

            if job_status == "completed":
                return {
                    "status": "completed",
                    "stems": status.get("stem_count", status.get("stems", 0)),
                    "has_chords": status.get("has_chords", False),
                    "has_gp": status.get("has_gp", False),
                }
            elif job_status in ("failed", "error"):
                error_msg = status.get("error", "Unknown error")
                logger.warning(f"  Job failed: {error_msg}")
                return {"status": "failed", "error": error_msg}

            # Still processing
            progress = status.get("progress", "?")
            stage = status.get("stage", "processing")
            logger.info(f"  Progress: {progress}% - {stage}")

        except Exception as e:
            logger.warning(f"  Poll error: {e}")

    return {"status": "timeout"}


def write_csv_header(filepath: Path):
    """Write CSV header if file doesn't exist."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if not filepath.exists():
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "title", "artist", "youtube_url", "video_id",
                "status", "stems", "has_chords", "has_gp", "processing_seconds",
            ])


def append_csv_row(filepath: Path, row: dict):
    """Append a result row to the CSV."""
    with open(filepath, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            row.get("title", ""),
            row.get("artist", ""),
            row.get("youtube_url", ""),
            row.get("video_id", ""),
            row.get("status", ""),
            row.get("stems", ""),
            row.get("has_chords", ""),
            row.get("has_gp", ""),
            row.get("processing_seconds", ""),
        ])


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process songs through StemScribe by searching YouTube"
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_SONG_FILE,
        help=f"Song list file (default: {DEFAULT_SONG_FILE.name})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max songs to process (0=all)",
    )
    parser.add_argument(
        "--skip-cached",
        action="store_true",
        default=True,
        help="Skip songs already in URL cache (default: true)",
    )
    parser.add_argument(
        "--no-skip-cached",
        action="store_true",
        help="Don't skip cached songs (reprocess everything)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DEFAULT_DELAY,
        help=f"Seconds between submissions (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=1,
        help="Max concurrent jobs (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just search YouTube and show URLs, don't process",
    )
    parser.add_argument(
        "--api",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"API base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_FILE,
        help=f"Results CSV path (default: {RESULTS_FILE})",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    if args.no_skip_cached:
        args.skip_cached = False

    setup_logging(args.verbose)

    # Parse song list
    if not args.file.exists():
        logger.error(f"Song file not found: {args.file}")
        sys.exit(1)

    songs = parse_song_file(args.file)
    if not songs:
        logger.error("No songs found in file!")
        sys.exit(1)

    total_songs = len(songs)
    if args.limit > 0:
        songs = songs[: args.limit]

    logger.info(f"Loaded {len(songs)} songs (from {total_songs} total in file)")

    # Check API health (unless dry run)
    if not args.dry_run:
        if not check_api_health(args.api):
            logger.warning(f"API not responding at {args.api}")
            if not wait_for_api(args.api):
                logger.error("API is not available. Start the server first:")
                logger.error("  cd ~/stemscribe/backend && ../venv311/bin/python app.py")
                sys.exit(1)

    # Set up results CSV
    write_csv_header(args.output)

    # Counters
    processed = 0
    cached_skipped = 0
    failed = 0
    not_found = 0
    total_time = 0.0

    try:
        for idx, song in enumerate(songs, 1):
            title = song["title"]
            artist = song["artist"]

            logger.info(f"\n[{idx}/{len(songs)}] Searching: {title} - {artist}")

            # Search YouTube
            yt_info = search_youtube(title, artist)
            if not yt_info:
                logger.warning(f"  NOT FOUND on YouTube: {title} - {artist}")
                not_found += 1
                append_csv_row(args.output, {
                    "title": title,
                    "artist": artist,
                    "status": "not_found",
                })
                continue

            youtube_url = yt_info["url"]
            video_id = yt_info["video_id"]
            logger.info(f"  YouTube: {youtube_url}")
            logger.info(f"  YT Title: {yt_info['yt_title']}")

            # Dry run — just show the match
            if args.dry_run:
                append_csv_row(args.output, {
                    "title": title,
                    "artist": artist,
                    "youtube_url": youtube_url,
                    "video_id": video_id,
                    "status": "dry_run",
                })
                continue

            # Check cache
            if args.skip_cached:
                cached = check_cache(youtube_url)
                if cached:
                    logger.info(f"  SKIP (cached): {title} - {artist}")
                    cached_skipped += 1
                    append_csv_row(args.output, {
                        "title": title,
                        "artist": artist,
                        "youtube_url": youtube_url,
                        "video_id": video_id,
                        "status": "cached",
                    })
                    continue

            # Submit to API
            start_time = time.time()
            job_id = submit_song(args.api, youtube_url)
            if not job_id:
                logger.warning(f"  FAILED to submit: {title} - {artist}")
                failed += 1
                # API might be down — wait and retry once
                if not check_api_health(args.api):
                    logger.info("  API seems down, waiting 60s and retrying...")
                    time.sleep(60)
                    if not check_api_health(args.api):
                        logger.info("  API still down, waiting 120s...")
                        time.sleep(120)
                        if not check_api_health(args.api):
                            logger.error("  API still down after 3 min. Stopping.")
                        break
                    job_id = submit_song(args.api, youtube_url)
                    if not job_id:
                        append_csv_row(args.output, {
                            "title": title,
                            "artist": artist,
                            "youtube_url": youtube_url,
                            "video_id": video_id,
                            "status": "submit_failed",
                        })
                        continue
                    # Retry succeeded, undo the failed counter
                    failed -= 1
                else:
                    append_csv_row(args.output, {
                        "title": title,
                        "artist": artist,
                        "youtube_url": youtube_url,
                        "video_id": video_id,
                        "status": "submit_failed",
                    })
                    continue

            logger.info(f"  Submitted job: {job_id}")

            # Poll for completion
            result = poll_job(args.api, job_id, DEFAULT_POLL_INTERVAL, DEFAULT_POLL_TIMEOUT)
            elapsed = time.time() - start_time

            status = result.get("status", "unknown")
            stems = result.get("stems", 0)
            has_chords = result.get("has_chords", False)
            has_gp = result.get("has_gp", False)

            if status == "completed":
                processed += 1
                total_time += elapsed
                logger.info(
                    f"  Status: completed ({stems} stems, chords: "
                    f"{'yes' if has_chords else 'no'}, "
                    f"GP: {'yes' if has_gp else 'no'}) in {elapsed:.1f}s"
                )
            elif status == "timeout":
                failed += 1
                logger.warning(f"  TIMEOUT after {elapsed:.1f}s")
            else:
                failed += 1
                logger.warning(f"  FAILED: {result.get('error', 'unknown error')}")

            append_csv_row(args.output, {
                "title": title,
                "artist": artist,
                "youtube_url": youtube_url,
                "video_id": video_id,
                "status": status,
                "stems": stems,
                "has_chords": has_chords,
                "has_gp": has_gp,
                "processing_seconds": f"{elapsed:.1f}",
            })

            # Delay between songs
            if idx < len(songs) and args.delay > 0:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user! Saving progress...")

    # Final summary
    estimated_cost = processed * MODAL_COST_PER_SONG
    logger.info("\n" + "=" * 50)
    logger.info("Pre-processing complete!")
    logger.info(f"  Processed:        {processed}")
    logger.info(f"  Cached (skipped): {cached_skipped}")
    logger.info(f"  Failed:           {failed}")
    logger.info(f"  Not found:        {not_found}")
    if processed > 0:
        logger.info(f"  Avg time/song:    {total_time / processed:.1f}s")
    logger.info(f"  Total GPU cost:   ~${estimated_cost:.2f}")
    logger.info(f"  Results saved to: {args.output}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
