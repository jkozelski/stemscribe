"""
Internet Archive Live Music Pipeline for StemScribe
=====================================================
Search, browse, and batch-process live recordings from archive.org.
The Live Music Archive has 250,000+ free, legal concert recordings.

API Reference:
    - Search: https://archive.org/advancedsearch.php?q=...&output=json
    - Metadata: https://archive.org/metadata/{identifier}
    - Files: https://archive.org/metadata/{identifier}/files
    - Download: https://archive.org/download/{identifier}/{filename}

Usage:
    from archive_pipeline import ArchivePipeline

    pipeline = ArchivePipeline()

    # Search for shows
    shows = pipeline.search_shows("grateful dead", year="1977")

    # Get tracks for a specific show
    tracks = pipeline.get_show_tracks("gd1977-05-08.sbd.hicks.4982.sbeok.shnf")

    # Get the direct URL for a track
    url = pipeline.get_track_url("gd1977-05-08.sbd.hicks.4982.sbeok.shnf", "gd77-05-08d1t01.mp3")
"""

import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("requests not available - Archive.org pipeline disabled")


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class ArchiveShow:
    """A live music recording from archive.org."""
    identifier: str
    title: str
    date: str = ""
    creator: str = ""
    venue: str = ""
    description: str = ""
    source: str = ""           # sbd, aud, mtx, etc.
    avg_rating: float = 0.0
    num_reviews: int = 0
    downloads: int = 0
    collection: str = ""
    url: str = ""

    def __post_init__(self):
        if not self.url:
            self.url = f"https://archive.org/details/{self.identifier}"


@dataclass
class ArchiveTrack:
    """A single audio track from an archive.org recording."""
    filename: str
    title: str = ""
    track_number: int = 0
    format: str = ""           # Flac, VBR MP3, Ogg Vorbis, Shorten
    size: int = 0              # bytes
    length: float = 0.0        # seconds
    source: str = "original"   # original or derivative
    download_url: str = ""
    identifier: str = ""

    def __post_init__(self):
        if self.identifier and not self.download_url:
            self.download_url = f"https://archive.org/download/{self.identifier}/{self.filename}"


# ============================================================================
# Known Collections (bands with taper-friendly policies)
# ============================================================================

COLLECTIONS = {
    'grateful_dead':    'GratefulDead',
    'dead':             'GratefulDead',
    'gd':               'GratefulDead',
    'umphreys':         'UmphreysMcGee',
    'umphreys_mcgee':   'UmphreysMcGee',
    'widespread_panic': 'WidespreadPanic',
    'panic':            'WidespreadPanic',
    'string_cheese':    'StringCheeseIncident',
    'sci':              'StringCheeseIncident',
    'little_feat':      'LittleFeat',
    'ween':             'Ween',
    'smashing_pumpkins': 'SmashingPumpkins',
    'tenacious_d':      'TenD',
    # Generic etree collection (all bands)
    'all':              'etree',
    'live_music':       'etree',
}

# Audio file extensions to look for (preference order)
AUDIO_FORMATS = {
    'flac': {'ext': '.flac', 'format': 'Flac', 'priority': 1, 'lossless': True},
    'shn':  {'ext': '.shn',  'format': 'Shorten', 'priority': 2, 'lossless': True},
    'mp3':  {'ext': '.mp3',  'format': 'VBR MP3', 'priority': 3, 'lossless': False},
    'ogg':  {'ext': '.ogg',  'format': 'Ogg Vorbis', 'priority': 4, 'lossless': False},
}


class ArchivePipeline:
    """
    Search, browse, and prepare downloads from the Internet Archive
    Live Music Archive for processing through StemScribe.
    """

    SEARCH_URL = "https://archive.org/advancedsearch.php"
    SCRAPE_URL = "https://archive.org/services/search/v1/scrape"
    METADATA_URL = "https://archive.org/metadata"
    DOWNLOAD_BASE = "https://archive.org/download"

    def __init__(self, rate_limit: float = 1.0):
        """
        Args:
            rate_limit: Minimum seconds between API requests (be nice to archive.org)
        """
        self.rate_limit = rate_limit
        self._last_request_time = 0
        self.session = requests.Session() if REQUESTS_AVAILABLE else None
        if self.session:
            self.session.headers.update({
                'User-Agent': 'StemScribe/1.0 (music transcription tool; https://github.com/jkozelski/stemscribe)'
            })

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict = None, timeout: int = 15) -> Optional[dict]:
        """Make a rate-limited GET request, return JSON or None."""
        if not self.session:
            logger.error("requests library not available")
            return None

        self._rate_limit_wait()

        try:
            resp = self.session.get(url, params=params, timeout=timeout)

            if resp.status_code == 429:
                # Rate limited — back off
                logger.warning("Archive.org rate limit hit, backing off 10s...")
                time.sleep(10)
                resp = self.session.get(url, params=params, timeout=timeout)

            resp.raise_for_status()
            return resp.json()

        except Exception as e:
            logger.error(f"Archive.org request failed: {e}")
            return None

    # ========================================================================
    # Search
    # ========================================================================

    def search_shows(self, query: str, collection: str = None,
                     year: str = None, sort: str = "avg_rating desc",
                     rows: int = 25, page: int = 1) -> List[ArchiveShow]:
        """
        Search for live music recordings.

        Args:
            query: Search terms (band name, song title, venue, etc.)
            collection: Collection to search in (e.g., "GratefulDead", "etree")
                        Can also use friendly names like "grateful_dead", "dead"
            year: Filter by year (e.g., "1977")
            sort: Sort order (default: highest rated first)
            rows: Results per page (max 100)
            page: Page number (1-indexed)

        Returns:
            List of ArchiveShow objects
        """
        # Build Lucene query
        q_parts = []

        if collection:
            # Resolve friendly names
            col = COLLECTIONS.get(collection.lower().replace(' ', '_'), collection)
            q_parts.append(f"collection:({col})")
        else:
            # Default to live music
            q_parts.append("mediatype:(etree)")

        if query:
            q_parts.append(f"({query})")

        if year:
            q_parts.append(f"year:({year})")

        q = " AND ".join(q_parts)

        params = {
            'q': q,
            'output': 'json',
            'rows': min(rows, 100),
            'page': page,
            'sort[]': sort,
            'fl[]': [
                'identifier', 'title', 'date', 'creator', 'venue',
                'description', 'source', 'avg_rating', 'num_reviews',
                'downloads', 'collection'
            ]
        }

        logger.info(f"🎵 Searching Archive.org: {q}")
        data = self._get(self.SEARCH_URL, params=params)

        if not data or 'response' not in data:
            return []

        shows = []
        for doc in data['response'].get('docs', []):
            show = ArchiveShow(
                identifier=doc.get('identifier', ''),
                title=doc.get('title', 'Unknown Show'),
                date=doc.get('date', ''),
                creator=doc.get('creator', ''),
                venue=doc.get('venue', ''),
                description=str(doc.get('description', ''))[:500],  # Truncate long descriptions
                source=doc.get('source', ''),
                avg_rating=float(doc.get('avg_rating', 0) or 0),
                num_reviews=int(doc.get('num_reviews', 0) or 0),
                downloads=int(doc.get('downloads', 0) or 0),
                collection=str(doc.get('collection', '')),
            )
            shows.append(show)

        logger.info(f"🎵 Found {len(shows)} shows (total: {data['response'].get('numFound', 0)})")
        return shows

    def search_by_band(self, band_name: str, year: str = None,
                       sort: str = "avg_rating desc", rows: int = 25) -> List[ArchiveShow]:
        """
        Search for shows by a specific band.

        Args:
            band_name: Band/artist name (e.g., "grateful dead", "umphreys mcgee")
            year: Optional year filter
            sort: Sort order
            rows: Number of results

        Returns:
            List of ArchiveShow objects
        """
        # Check if we have a known collection for this band
        band_key = band_name.lower().replace(' ', '_').replace("'", '')
        collection = COLLECTIONS.get(band_key)

        if collection:
            return self.search_shows("", collection=collection, year=year, sort=sort, rows=rows)
        else:
            return self.search_shows(band_name, collection="etree", year=year, sort=sort, rows=rows)

    # ========================================================================
    # Show Details & Track Listing
    # ========================================================================

    def get_show_details(self, identifier: str) -> Optional[Dict]:
        """
        Get full metadata for a show.

        Args:
            identifier: Archive.org item identifier

        Returns:
            Dict with show metadata or None
        """
        url = f"{self.METADATA_URL}/{identifier}/metadata"
        data = self._get(url)

        if not data or 'result' not in data:
            return data  # Some responses don't wrap in 'result'

        return data.get('result', data)

    def get_show_tracks(self, identifier: str,
                        prefer_format: str = "mp3",
                        include_derivatives: bool = True) -> List[ArchiveTrack]:
        """
        Get audio tracks for a show.

        Args:
            identifier: Archive.org item identifier
            prefer_format: Preferred audio format ("flac", "mp3", "ogg", "shn")
            include_derivatives: Include auto-generated format conversions

        Returns:
            List of ArchiveTrack objects, sorted by track number
        """
        url = f"{self.METADATA_URL}/{identifier}/files"
        data = self._get(url)

        if not data or 'result' not in data:
            return []

        files = data['result']
        tracks = []
        seen_titles = set()

        for f in files:
            filename = f.get('name', '')
            file_format = f.get('format', '')
            source = f.get('source', 'original')

            # Skip non-audio files
            if not self._is_audio_file(filename, file_format):
                continue

            # Skip derivatives if not wanted
            if not include_derivatives and source == 'derivative':
                continue

            title = f.get('title', '') or self._title_from_filename(filename)
            track_num = self._parse_track_number(f.get('track', ''), filename)

            track = ArchiveTrack(
                filename=filename,
                title=title,
                track_number=track_num,
                format=file_format,
                size=int(f.get('size', 0) or 0),
                length=float(f.get('length', 0) or 0),
                source=source,
                identifier=identifier,
            )
            tracks.append(track)

        # If multiple formats exist, prefer the requested format
        tracks = self._deduplicate_tracks(tracks, prefer_format)

        # Sort by track number
        tracks.sort(key=lambda t: (t.track_number, t.filename))

        logger.info(f"🎵 Found {len(tracks)} audio tracks for {identifier}")
        return tracks

    def get_track_url(self, identifier: str, filename: str) -> str:
        """Get the direct download URL for a specific track."""
        return f"{self.DOWNLOAD_BASE}/{identifier}/{filename}"

    # ========================================================================
    # Setlist Extraction
    # ========================================================================

    def extract_setlist(self, identifier: str) -> List[str]:
        """
        Try to extract a setlist from show metadata.
        Setlists are usually embedded in the description field.

        Returns:
            List of song titles (best effort)
        """
        url = f"{self.METADATA_URL}/{identifier}/metadata"
        data = self._get(url)

        if not data:
            return []

        # Get metadata (handle both wrapped and unwrapped responses)
        meta = data.get('result', data)
        description = meta.get('description', '')

        if not description:
            return []

        # Try to parse setlist from description
        songs = []

        # Common patterns in archive.org descriptions:
        # "Set 1:", "Set 2:", "Encore:", followed by song titles
        # Or just a list of songs separated by newlines or ">"

        # Clean HTML tags
        description = re.sub(r'<[^>]+>', '\n', description)

        lines = description.split('\n')
        in_setlist = False

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect setlist section headers
            if re.match(r'^(set\s*[12]|encore|e:)', line, re.IGNORECASE):
                in_setlist = True
                continue

            # Lines starting with track markers
            if re.match(r'^(d\d+t\d+|t\d+|\d+[\.\)]\s)', line):
                # Track reference format: d1t01, t01, 1., 1)
                song = re.sub(r'^(d\d+t\d+|t\d+|\d+[\.\)])\s*', '', line)
                if song:
                    songs.append(song.strip())
                continue

            # Songs separated by >
            if '>' in line and len(line) < 200:
                parts = [s.strip() for s in line.split('>') if s.strip()]
                songs.extend(parts)
                continue

            # If we're in a setlist section, each line might be a song
            if in_setlist and len(line) < 80 and not line.startswith(('http', 'Source', 'Lineage', 'Transfer')):
                songs.append(line)

        return songs

    # ========================================================================
    # Batch Processing Helpers
    # ========================================================================

    def get_show_for_processing(self, identifier: str,
                                 prefer_format: str = "mp3") -> Dict:
        """
        Get all info needed to process a show through StemScribe.

        Returns:
            Dict with:
                - show: ArchiveShow metadata
                - tracks: List of ArchiveTrack with download URLs
                - setlist: Extracted song names
                - archive_url: URL for direct yt-dlp download
        """
        # Get show metadata
        meta_url = f"{self.METADATA_URL}/{identifier}"
        data = self._get(meta_url)

        if not data:
            return {'error': f'Could not fetch metadata for {identifier}'}

        metadata = data.get('metadata', data)

        show = ArchiveShow(
            identifier=identifier,
            title=metadata.get('title', 'Unknown Show'),
            date=metadata.get('date', ''),
            creator=metadata.get('creator', ''),
            venue=metadata.get('venue', '') or metadata.get('coverage', ''),
            description=str(metadata.get('description', ''))[:500],
            source=metadata.get('source', ''),
        )

        # Get tracks
        tracks = self.get_show_tracks(identifier, prefer_format=prefer_format)

        # Extract setlist
        setlist = self.extract_setlist(identifier)

        return {
            'show': {
                'identifier': show.identifier,
                'title': show.title,
                'date': show.date,
                'creator': show.creator,
                'venue': show.venue,
                'description': show.description,
                'source': show.source,
                'url': show.url,
            },
            'tracks': [
                {
                    'filename': t.filename,
                    'title': t.title,
                    'track_number': t.track_number,
                    'format': t.format,
                    'size': t.size,
                    'length': t.length,
                    'download_url': t.download_url,
                }
                for t in tracks
            ],
            'setlist': setlist,
            'archive_url': f"https://archive.org/details/{identifier}",
            'track_count': len(tracks),
        }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _is_audio_file(self, filename: str, file_format: str) -> bool:
        """Check if a file is an audio track (not metadata, images, etc.)."""
        lower = filename.lower()

        # Check by extension
        audio_exts = ('.mp3', '.flac', '.ogg', '.shn', '.wav')
        if any(lower.endswith(ext) for ext in audio_exts):
            return True

        # Check by format field
        audio_formats = ('VBR MP3', 'Flac', 'Ogg Vorbis', 'Shorten', '64Kbps MP3', 'WAVE')
        if file_format in audio_formats:
            return True

        return False

    def _title_from_filename(self, filename: str) -> str:
        """Extract a song title from a filename."""
        # Remove extension
        name = re.sub(r'\.\w{2,4}$', '', filename)
        # Remove common prefixes like "gd77-05-08d1t01"
        name = re.sub(r'^[a-z]{1,4}\d{2,4}[-_]\d{2}[-_]\d{2}[a-z]?\d*t?\d*[-_]?', '', name, flags=re.IGNORECASE)
        # Clean up underscores and dashes
        name = name.replace('_', ' ').replace('-', ' ').strip()
        return name or filename

    def _parse_track_number(self, track_field: str, filename: str) -> int:
        """Extract track number from metadata or filename."""
        # Try the track field first
        if track_field:
            match = re.search(r'(\d+)', str(track_field))
            if match:
                return int(match.group(1))

        # Try to extract from filename patterns like d1t03, t03, track03
        patterns = [
            r'd\d+t(\d+)',      # d1t03
            r't(\d+)',          # t03
            r'track(\d+)',      # track03
            r'[-_](\d{2,3})\.',  # -03.mp3
        ]
        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)
            if match:
                return int(match.group(1))

        return 0

    def _deduplicate_tracks(self, tracks: List[ArchiveTrack],
                            prefer_format: str = "mp3") -> List[ArchiveTrack]:
        """
        When multiple formats of the same track exist, pick the preferred one.
        Groups tracks by track number/title and selects best format.
        """
        # Group by a normalized key (track number + approximate title)
        groups = {}
        for track in tracks:
            # Key = track number if available, otherwise filename stem
            if track.track_number > 0:
                key = track.track_number
            else:
                key = re.sub(r'\.\w{2,4}$', '', track.filename).lower()

            if key not in groups:
                groups[key] = []
            groups[key].append(track)

        # For each group, pick the preferred format
        result = []
        prefer_ext = AUDIO_FORMATS.get(prefer_format, {}).get('ext', '.mp3')

        for key, group in groups.items():
            if len(group) == 1:
                result.append(group[0])
            else:
                # Prefer the requested format, then by priority
                preferred = None
                for track in group:
                    if track.filename.lower().endswith(prefer_ext):
                        preferred = track
                        break

                if not preferred:
                    # Fall back to priority ordering
                    def format_priority(t):
                        for fmt, info in AUDIO_FORMATS.items():
                            if t.filename.lower().endswith(info['ext']):
                                return info['priority']
                        return 99
                    preferred = min(group, key=format_priority)

                result.append(preferred)

        return result

    @staticmethod
    def identifier_from_url(url: str) -> Optional[str]:
        """
        Extract the archive.org identifier from a URL.

        Handles:
            https://archive.org/details/gd1977-05-08.sbd.hicks.4982.sbeok.shnf
            https://archive.org/download/gd1977-05-08.sbd.hicks.4982.sbeok.shnf/file.mp3
            https://archive.org/embed/gd1977-05-08.sbd.hicks.4982.sbeok.shnf
        """
        match = re.search(r'archive\.org/(?:details|download|embed)/([^/?#]+)', url)
        return match.group(1) if match else None

    @staticmethod
    def is_archive_url(url: str) -> bool:
        """Check if a URL is an archive.org link."""
        return bool(re.search(r'archive\.org/(details|download|embed)/', url))


# ============================================================================
# Convenience functions
# ============================================================================

_pipeline = None

def get_pipeline() -> ArchivePipeline:
    """Get or create the singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ArchivePipeline()
    return _pipeline


def search_archive(query: str, collection: str = None,
                   year: str = None, rows: int = 25) -> List[Dict]:
    """Quick search for shows. Returns list of dicts."""
    pipeline = get_pipeline()
    shows = pipeline.search_shows(query, collection=collection, year=year, rows=rows)
    return [
        {
            'identifier': s.identifier,
            'title': s.title,
            'date': s.date,
            'creator': s.creator,
            'venue': s.venue,
            'source': s.source,
            'avg_rating': s.avg_rating,
            'num_reviews': s.num_reviews,
            'downloads': s.downloads,
            'url': s.url,
        }
        for s in shows
    ]


def get_show_info(identifier: str) -> Dict:
    """Get full show info with tracks. Returns dict."""
    pipeline = get_pipeline()
    return pipeline.get_show_for_processing(identifier)
