"""
Audio downloading from URLs via yt-dlp.
"""

import re
import json
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_from_url(job, url: str, output_dir: Path):
    """Download audio from YouTube or other supported platforms using yt-dlp"""
    job.stage = 'Downloading audio from URL'
    job.progress = 5
    logger.info(f"Downloading from URL: {url}")

    try:
        # YouTube cookies file (needed for VPS/cloud to avoid bot detection)
        cookies_file = Path(__file__).parent.parent.parent / 'youtube-cookies.txt'
        extra_args = []
        if cookies_file.exists():
            extra_args += ['--cookies', str(cookies_file)]
        extra_args += ['--extractor-args', 'youtube:player_client=web']

        # First, get metadata
        metadata_cmd = [
            'yt-dlp',
            '--dump-json',
            '--no-download',
            *extra_args,
            url
        ]

        result = subprocess.run(metadata_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            metadata = json.loads(result.stdout)

            # Determine best artist: prefer YouTube Music "artist" field (from music metadata),
            # then try parsing "Artist - Title" from the video title,
            # fall back to uploader (channel name) as last resort.
            raw_artist = metadata.get('artist') or ''
            raw_title = metadata.get('title', 'Unknown')
            raw_track = metadata.get('track') or ''  # YouTube Music track name

            # Use YouTube Music track/artist if available (these come from official metadata)
            best_title = raw_track if raw_track else raw_title
            best_artist = raw_artist

            # If artist is empty or looks like a YouTube channel name, parse from title
            if not best_artist or best_artist == metadata.get('uploader', ''):
                # Try common title patterns: "Artist - Title", "Artist | Title", "Artist: Title"
                for sep in [' - ', ' — ', ' – ', ' | ', ': ']:
                    if sep in raw_title:
                        parts = raw_title.split(sep, 1)
                        parsed_artist = parts[0].strip()
                        parsed_title = parts[1].strip()
                        if parsed_artist and parsed_title:
                            best_artist = parsed_artist
                            # Only override title if we didn't get a track name from YT Music
                            if not raw_track:
                                best_title = parsed_title
                            break

            # Final fallback to uploader
            if not best_artist:
                best_artist = metadata.get('uploader', 'Unknown')

            job.metadata = {
                'title': best_title,
                'artist': best_artist,
                'duration': metadata.get('duration', 0),
                'thumbnail': metadata.get('thumbnail', ''),
                'webpage_url': metadata.get('webpage_url', url)
            }
            logger.info(f"Metadata: {job.metadata['title']} by {job.metadata['artist']}")

        # Sanitize filename
        safe_title = re.sub(r'[^\w\s-]', '', job.metadata.get('title', 'audio'))[:50]
        safe_title = safe_title.strip().replace(' ', '_')
        output_template = str(output_dir / f"{safe_title}.%(ext)s")

        # Download audio
        download_cmd = [
            'yt-dlp',
            '-x',                          # Extract audio only
            '--audio-format', 'wav',       # Best quality for processing
            '--audio-quality', '0',        # Highest quality
            '--no-playlist',               # Don't download playlists
            '--max-filesize', '500M',      # Limit file size
            *extra_args,
            '-o', output_template,
            url
        ]

        job.stage = f'Downloading: {job.metadata.get("title", "audio")}'

        result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
            logger.error(f"yt-dlp error: {result.stderr}")
            raise Exception(f"Download failed: {result.stderr[:200]}")

        # Find the downloaded file
        for ext in ['wav', 'mp3', 'webm', 'm4a', 'opus']:
            audio_file = output_dir / f"{safe_title}.{ext}"
            if audio_file.exists():
                job.filename = audio_file.name
                job.progress = 10
                logger.info(f"Downloaded: {audio_file}")
                return audio_file

        # Try to find any audio file in the directory
        for f in output_dir.glob('*.*'):
            if f.suffix.lower() in ['.wav', '.mp3', '.webm', '.m4a', '.opus', '.ogg']:
                job.filename = f.name
                job.progress = 10
                logger.info(f"Found downloaded file: {f}")
                return f

        raise Exception("Downloaded file not found")

    except subprocess.TimeoutExpired:
        raise Exception("Download timed out (5 min limit)")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise
