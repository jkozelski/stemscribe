"""
Songsterr Integration for StemScribe

Fetches real, human-curated guitar tabs from Songsterr.
Works by scraping the embedded JSON state from Songsterr pages.

GP5 files are hosted on CloudFront CDN.
"""

import requests
import logging
import re
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Songsterr URLs
SONGSTERR_SEARCH = "https://www.songsterr.com/a/wa/bestMatchForQueryString"
SONGSTERR_SONG = "https://www.songsterr.com/a/wsa/{slug}-tab-s{song_id}"

# CloudFront CDN for GP5 files
CLOUDFRONT_GP5 = "https://d12drcwhcokzqv.cloudfront.net/{revision_id}.gp5"


@dataclass
class SongsterrTab:
    """Represents a tab from Songsterr"""
    song_id: int
    title: str
    artist: str
    revision_id: int
    gp5_url: str
    tracks: List[Dict[str, Any]]
    
    def __str__(self):
        return f"{self.artist} - {self.title} (ID: {self.song_id})"


class SongsterrAPI:
    """Client for Songsterr - scrapes embedded JSON from pages"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
        self.cache_dir = cache_dir or Path('./songsterr_cache')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def search(self, query: str) -> Optional[SongsterrTab]:
        """
        Search for a tab and return the best match.
        
        Args:
            query: Search string (e.g., "Stairway to Heaven Led Zeppelin")
            
        Returns:
            SongsterrTab if found, None otherwise
        """
        try:
            # Songsterr's bestMatch redirects to the tab page
            params = {'s': query}
            response = self.session.get(SONGSTERR_SEARCH, params=params, timeout=15)
            response.raise_for_status()
            
            # Parse the embedded JSON state
            return self._parse_page(response.text)
            
        except requests.RequestException as e:
            logger.error(f"Songsterr search failed: {e}")
            return None
    
    def get_tab_by_id(self, song_id: int, slug: str = "song") -> Optional[SongsterrTab]:
        """
        Get tab info by song ID.
        
        Args:
            song_id: Songsterr song ID
            slug: URL slug (optional, will work with generic)
            
        Returns:
            SongsterrTab if found
        """
        try:
            url = SONGSTERR_SONG.format(slug=slug, song_id=song_id)
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            
            return self._parse_page(response.text)
            
        except requests.RequestException as e:
            logger.error(f"Songsterr get tab failed: {e}")
            return None
    
    def _parse_page(self, html: str) -> Optional[SongsterrTab]:
        """Parse Songsterr page HTML to extract tab info from embedded JSON"""
        try:
            # Find the embedded JSON state
            match = re.search(r'<script id="state" type="application/json"[^>]*>(.+?)</script>', html, re.DOTALL)
            if not match:
                logger.error("Could not find embedded state JSON")
                return None
            
            state = json.loads(match.group(1))
            
            # Extract metadata
            meta = state.get('meta', {}).get('current', {})
            if not meta:
                logger.error("No metadata in state")
                return None
            
            song_id = meta.get('songId', 0)
            revision_id = meta.get('revisionId', 0)
            title = meta.get('title', 'Unknown')
            artist = meta.get('artist', 'Unknown')
            tracks = meta.get('tracks', [])
            
            if not song_id or not revision_id:
                logger.error(f"Missing song_id or revision_id: {song_id}, {revision_id}")
                return None
            
            # Construct GP5 URL
            gp5_url = CLOUDFRONT_GP5.format(revision_id=revision_id)
            
            tab = SongsterrTab(
                song_id=song_id,
                title=title,
                artist=artist,
                revision_id=revision_id,
                gp5_url=gp5_url,
                tracks=tracks
            )
            
            logger.info(f"Found tab: {tab}")
            return tab
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON state: {e}")
            return None
        except Exception as e:
            logger.error(f"Failed to parse page: {e}")
            return None
    
    def download_gp5(self, tab: SongsterrTab, output_dir: Path) -> Optional[Path]:
        """
        Download GP5 file for a tab.
        
        Args:
            tab: SongsterrTab object with gp5_url
            output_dir: Directory to save the file
            
        Returns:
            Path to downloaded GP5 file, or None if failed
        """
        # Check cache first
        safe_filename = self._safe_filename(f"{tab.artist} - {tab.title}")
        cached_path = self.cache_dir / f"{safe_filename}_{tab.revision_id}.gp5"
        
        if cached_path.exists():
            logger.info(f"Using cached GP5: {cached_path}")
            output_path = output_dir / f"{safe_filename}.gp5"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy(cached_path, output_path)
            return output_path
        
        # Download from CloudFront
        try:
            logger.info(f"Downloading GP5 from: {tab.gp5_url}")
            response = self.session.get(tab.gp5_url, timeout=30)
            response.raise_for_status()
            
            # Verify it's actually a GP5 file (should start with specific bytes)
            if len(response.content) < 100:
                logger.error(f"GP5 file too small: {len(response.content)} bytes")
                return None
            
            # Save to cache
            cached_path.write_bytes(response.content)
            logger.info(f"Cached GP5: {cached_path} ({len(response.content)} bytes)")
            
            # Copy to output dir
            output_path = output_dir / f"{safe_filename}.gp5"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(response.content)
            
            return output_path
            
        except requests.RequestException as e:
            logger.error(f"GP5 download failed: {e}")
            return None
    
    def find_and_download(self, title: str, artist: str, output_dir: Path) -> Optional[Path]:
        """
        Search for a tab and download the GP5 file.
        
        Args:
            title: Song title
            artist: Artist name
            output_dir: Where to save the GP5 file
            
        Returns:
            Path to GP5 file if successful
        """
        # Clean up search query
        clean_title = self._clean_title(title)
        clean_artist = self._clean_artist(artist)
        
        query = f"{clean_title} {clean_artist}"
        logger.info(f"Searching Songsterr for: {query}")
        
        tab = self.search(query)
        if not tab:
            # Try title only
            tab = self.search(clean_title)
        
        if not tab:
            logger.warning(f"No Songsterr tab found for: {artist} - {title}")
            return None
        
        # Verify it's the right song (fuzzy match)
        if not self._is_match(tab, clean_title, clean_artist):
            logger.warning(f"Songsterr result doesn't match: got '{tab.artist} - {tab.title}'")
            # Still return it - user can verify
        
        return self.download_gp5(tab, output_dir)
    
    def _clean_title(self, title: str) -> str:
        """Clean up song title for better matching"""
        patterns = [
            r'\s*\(.*?remaster.*?\)\s*',
            r'\s*\(.*?live.*?\)\s*',
            r'\s*\(.*?version.*?\)\s*',
            r'\s*\(.*?edit.*?\)\s*',
            r'\s*\[.*?\]\s*',
            r'\s*-\s*\d{4}\s*$',
        ]
        
        result = title
        for pattern in patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        return result.strip()
    
    def _clean_artist(self, artist: str) -> str:
        """Clean up artist name"""
        if artist.lower().startswith('the '):
            return artist[4:]
        return artist
    
    def _is_match(self, tab: SongsterrTab, title: str, artist: str) -> bool:
        """Check if tab matches the requested song"""
        title_lower = title.lower()
        artist_lower = artist.lower()
        tab_title = tab.title.lower()
        tab_artist = tab.artist.lower()
        
        # Check title match
        title_match = (
            title_lower in tab_title or 
            tab_title in title_lower or
            title_lower == tab_title
        )
        
        # Check artist match
        artist_match = (
            artist_lower in tab_artist or 
            tab_artist in artist_lower or
            artist_lower == tab_artist
        )
        
        return title_match and artist_match
    
    def _safe_filename(self, name: str) -> str:
        """Convert string to safe filename"""
        safe = re.sub(r'[<>:"/\\|?*]', '', name)
        safe = re.sub(r'\s+', '_', safe)
        return safe[:100]


# Convenience functions
def search_songsterr(query: str) -> Optional[Dict[str, Any]]:
    """Quick search for a tab"""
    api = SongsterrAPI()
    tab = api.search(query)
    
    if tab:
        return {
            'song_id': tab.song_id,
            'title': tab.title,
            'artist': tab.artist,
            'revision_id': tab.revision_id,
            'gp5_url': tab.gp5_url,
            'tracks': [
                {
                    'name': t.get('name', ''),
                    'instrument': t.get('instrument', ''),
                    'is_guitar': t.get('isGuitar', False),
                    'is_bass': t.get('isBassGuitar', False),
                    'is_drums': t.get('isDrums', False),
                }
                for t in tab.tracks
            ]
        }
    return None


def download_tab(title: str, artist: str, output_dir: str) -> Optional[str]:
    """Download a tab GP5 file"""
    api = SongsterrAPI()
    path = api.find_and_download(title, artist, Path(output_dir))
    return str(path) if path else None


# Test
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    api = SongsterrAPI()
    
    # Test search
    print("\n=== Searching for 'Stairway to Heaven Led Zeppelin' ===")
    tab = api.search("Stairway to Heaven Led Zeppelin")
    if tab:
        print(f"  Found: {tab}")
        print(f"  Revision ID: {tab.revision_id}")
        print(f"  GP5 URL: {tab.gp5_url}")
        print(f"  Tracks: {len(tab.tracks)}")
        for t in tab.tracks[:3]:
            print(f"    - {t.get('name', 'Unknown')}: {t.get('instrument', 'Unknown')}")
    
    # Test Franklin's Tower
    print("\n=== Searching for 'Franklin's Tower Grateful Dead' ===")
    tab = api.search("Franklin's Tower Grateful Dead")
    if tab:
        print(f"  Found: {tab}")
        print(f"  GP5 URL: {tab.gp5_url}")
