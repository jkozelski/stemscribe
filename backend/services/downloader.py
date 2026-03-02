"""
Audio downloading from URLs via yt-dlp.
"""

import os
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
        # First, get metadata
        # Use browser cookies + Deno JS solver to bypass YouTube bot detection
        metadata_cmd = [
            'yt-dlp',
            '--cookies-from-browser', 'chrome',
            '--remote-components', 'ejs:github',
            '--dump-json',
            '--no-download',
            url
        ]

        result = subprocess.run(metadata_cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            metadata = json.loads(result.stdout)
            job.metadata = {
                'title': metadata.get('title', 'Unknown'),
                'artist': metadata.get('artist') or metadata.get('uploader', 'Unknown'),
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
        # Use browser cookies + Deno JS solver to bypass YouTube bot detection
        download_cmd = [
            'yt-dlp',
            '--cookies-from-browser', 'chrome',
            '--remote-components', 'ejs:github',
            '-x',                          # Extract audio only
            '--audio-format', 'wav',       # Best quality for processing
            '--audio-quality', '0',        # Highest quality
            '--no-playlist',               # Don't download playlists
            '--max-filesize', '500M',      # Limit file size
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
