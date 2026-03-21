"""
URL resolution: detect supported platforms, validate URLs, and resolve streaming service URLs to YouTube.
"""

import re
import json
import subprocess
import logging
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def validate_url_no_ssrf(url: str) -> bool:
    """Block dangerous URLs (local network, file://, DNS rebinding, etc.)."""
    import ipaddress
    import socket

    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https'):
        return False
    hostname = parsed.hostname or ''
    if not hostname:
        return False

    # Resolve DNS to check the actual IP (prevents DNS rebinding)
    try:
        resolved_ips = socket.getaddrinfo(hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError):
        return False  # Can't resolve = don't allow

    for family, _, _, _, sockaddr in resolved_ips:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            return False

    # Block cloud metadata endpoints by hostname
    blocked_hosts = ('metadata.google.internal', 'metadata.aws.', '169.254.169.254')
    for b in blocked_hosts:
        if hostname == b or hostname.startswith(b):
            return False

    return True


# Supported URL patterns (direct download via yt-dlp)
URL_PATTERNS = [
    r'(youtube\.com|youtu\.be)',
    r'soundcloud\.com',
    r'bandcamp\.com',
    r'vimeo\.com',
    r'dailymotion\.com',
    r'mixcloud\.com',
    r'audiomack\.com',
    r'archive\.org/(details|download)',  # Internet Archive Live Music
]

# Streaming service patterns (need special handling)
STREAMING_PATTERNS = {
    'spotify': r'open\.spotify\.com/(track|album|playlist)/([a-zA-Z0-9]+)',
    'apple_music': r'music\.apple\.com/.+/album/.+/(\d+)',
}


def is_supported_url(url):
    """Check if URL is from a supported platform"""
    for pattern in URL_PATTERNS:
        if re.search(pattern, url, re.IGNORECASE):
            return True
    return False


def is_streaming_url(url):
    """Check if URL is from a streaming service (Spotify/Apple Music)"""
    for service, pattern in STREAMING_PATTERNS.items():
        if re.search(pattern, url, re.IGNORECASE):
            return service
    return None


def get_spotify_track_info(url):
    """Extract track info from Spotify URL using oEmbed API (no auth needed)"""
    try:
        import urllib.request
        import urllib.parse

        # Use Spotify's oEmbed endpoint to get track info
        oembed_url = f"https://open.spotify.com/oembed?url={urllib.parse.quote(url)}"

        req = urllib.request.Request(oembed_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

            # Title format is usually "Song Name - Artist Name"
            title = data.get('title', '')

            # Extract thumbnail
            thumbnail = data.get('thumbnail_url', '')

            logger.info(f"Spotify track info: {title}")
            return {
                'title': title,
                'thumbnail': thumbnail,
                'search_query': title  # Use full title for YouTube search
            }
    except Exception as e:
        logger.error(f"Failed to get Spotify info: {e}")
        return None


def get_apple_music_track_info(url):
    """Extract track info from Apple Music URL"""
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode('utf-8')

            # Extract title from meta tags
            title_match = re.search(r'<meta property="og:title" content="([^"]+)"', html)
            title = title_match.group(1) if title_match else None

            # Extract thumbnail
            thumb_match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
            thumbnail = thumb_match.group(1) if thumb_match else ''

            if title:
                # Clean up title (remove " - Apple Music" suffix if present)
                title = re.sub(r'\s*[-\u2013]\s*Apple Music\s*$', '', title)
                logger.info(f"Apple Music track info: {title}")
                return {
                    'title': title,
                    'thumbnail': thumbnail,
                    'search_query': title
                }
    except Exception as e:
        logger.error(f"Failed to get Apple Music info: {e}")
    return None


def search_youtube_for_song(search_query):
    """Search YouTube for a song and return the best match URL"""
    try:
        # Use yt-dlp to search YouTube (cookies + Deno bypass bot detection)
        search_cmd = [
            'yt-dlp',
            '--cookies-from-browser', 'chrome',
            '--remote-components', 'ejs:github',
            '--dump-json',
            '--no-download',
            '--default-search', 'ytsearch1',  # Get first result
            f'ytsearch1:{search_query}'
        ]

        result = subprocess.run(search_cmd, capture_output=True, text=True, timeout=30)

        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            video_url = data.get('webpage_url') or f"https://www.youtube.com/watch?v={data.get('id')}"
            logger.info(f"Found YouTube video: {video_url}")
            return video_url, data
        else:
            logger.error(f"YouTube search failed: {result.stderr}")
            return None, None

    except Exception as e:
        logger.error(f"YouTube search error: {e}")
        return None, None
