"""
Track Info Fetcher for StemScribe

Fetches contextual information about tracks:
- Artist biography and history
- Album info (name, year, details)
- Song background and trivia
- Player/member mapping for stem labeling
- Era-specific lineups
- Learning tips

Sources:
- Wikipedia API
- MusicBrainz API
- Last.fm API (if available)
- Local knowledge base
"""

import os
import re
import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Last.fm API key (free tier)
LASTFM_API_KEY = os.environ.get('LASTFM_API_KEY', '')

# Wikipedia API endpoint
WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# MusicBrainz API endpoint
MUSICBRAINZ_API = "https://musicbrainz.org/ws/2/"

# ============================================================================
# PLAYER POSITION MAPPING
# Maps stem types to player names based on typical panning positions
# ============================================================================

PLAYER_POSITIONS = {
    'grateful dead': {
        'guitar_left': 'Bob Weir',
        'guitar_right': 'Jerry Garcia',
        'guitar_center': 'Guitar (both)',
        'guitar': 'Guitar',
        'bass': 'Phil Lesh',
        'bass_left': 'Phil Lesh',
        'bass_center': 'Phil Lesh',
        'drums': 'Billy & Mickey',
        'drums_left': 'Bill Kreutzmann',
        'drums_right': 'Mickey Hart',
        'vocals': 'Vocals',
        'vocals_left': 'Jerry Garcia',
        'vocals_right': 'Bob Weir',
        'vocals_center': 'Lead Vocal',
        'piano': 'Keys',
        'piano_left': 'Keys (left)',
        'piano_right': 'Keys (right)',
        'other': 'Other',
        'other_left': 'Other (left)',
        'other_right': 'Other (right)',
    },
    'phish': {
        'guitar_left': 'Trey Anastasio',
        'guitar_right': 'Trey Anastasio',
        'guitar_center': 'Trey Anastasio',
        'guitar': 'Trey Anastasio',
        'bass': 'Mike Gordon',
        'drums': 'Jon Fishman',
        'piano': 'Page McConnell',
        'piano_left': 'Page (piano)',
        'piano_right': 'Page (organ)',
        'vocals': 'Vocals',
        'other': 'Other',
    },
    'allman brothers': {
        'guitar_left': 'Duane Allman',
        'guitar_right': 'Dickey Betts',
        'guitar_center': 'Guitars',
        'guitar': 'Guitars',
        'bass': 'Berry Oakley',
        'drums': 'Drums',
        'drums_left': 'Butch Trucks',
        'drums_right': 'Jaimoe',
        'piano': 'Gregg Allman',
        'vocals': 'Gregg Allman',
        'other': 'Other',
    },
    'led zeppelin': {
        'guitar': 'Jimmy Page',
        'guitar_left': 'Jimmy Page (rhythm)',
        'guitar_right': 'Jimmy Page (lead)',
        'bass': 'John Paul Jones',
        'drums': 'John Bonham',
        'piano': 'John Paul Jones',
        'vocals': 'Robert Plant',
    },
    'the beatles': {
        'guitar_left': 'John Lennon',
        'guitar_right': 'George Harrison',
        'bass': 'Paul McCartney',
        'drums': 'Ringo Starr',
        'piano': 'Piano',
        'vocals_left': 'John Lennon',
        'vocals_right': 'Paul McCartney',
    },
    'pink floyd': {
        'guitar': 'David Gilmour',
        'guitar_left': 'David Gilmour',
        'guitar_right': 'David Gilmour',
        'bass': 'Roger Waters',
        'drums': 'Nick Mason',
        'piano': 'Richard Wright',
        'vocals': 'Vocals',
    },
}

# ============================================================================
# ERA-SPECIFIC LINEUPS
# Different keyboardists, members by era
# ============================================================================

GRATEFUL_DEAD_ERAS = {
    # Era name: (start_year, end_year, keyboardist, notes)
    'pigpen': (1965, 1972, 'Ron "Pigpen" McKernan', 'Bluesy, raw sound. Hammond organ and harmonica.'),
    'keith': (1971, 1979, 'Keith Godchaux', 'Jazz-influenced piano. Donna Jean on vocals.'),
    'brent': (1979, 1990, 'Brent Mydland', 'Synths, Hammond, powerful vocals. Peak touring era.'),
    'vince_bruce': (1990, 1992, 'Vince Welnick & Bruce Hornsby', 'Transition period. Bruce on piano.'),
    'vince': (1992, 1995, 'Vince Welnick', 'Final touring years.'),
}

# ============================================================================
# ALBUM DATABASE
# Key albums with year, songs, and notes
# ============================================================================

ALBUM_DATABASE = {
    'grateful dead': {
        'american beauty': {
            'year': 1970,
            'label': 'Warner Bros.',
            'songs': ['box of rain', 'friend of the devil', 'sugar magnolia', 'operator', 'candyman',
                      'ripple', 'brokedown palace', 'till the morning comes', 'attics of my life', 'truckin'],
            'description': 'Acoustic-leaning masterpiece. Folk and country influences at their peak.',
            'personnel': {
                'jerry garcia': 'Guitar, vocals, pedal steel',
                'bob weir': 'Guitar, vocals',
                'phil lesh': 'Bass, vocals',
                'bill kreutzmann': 'Drums',
                'mickey hart': 'Drums, percussion',
                'pigpen': 'Keyboards, harmonica, vocals',
            }
        },
        'workingmans dead': {
            'year': 1970,
            'label': 'Warner Bros.',
            'songs': ['uncle johns band', 'high time', 'dire wolf', 'new speedway boogie',
                      'cumberland blues', 'black peter', 'easy wind', 'caseys jones'],
            'description': 'Tight vocal harmonies and country rock. CSNY influence.',
            'personnel': {
                'jerry garcia': 'Guitar, vocals, pedal steel',
                'bob weir': 'Guitar, vocals',
                'phil lesh': 'Bass, vocals',
                'bill kreutzmann': 'Drums',
                'mickey hart': 'Drums',
                'pigpen': 'Keyboards, harmonica, vocals',
            }
        },
        'live/dead': {
            'year': 1969,
            'label': 'Warner Bros.',
            'songs': ['dark star', 'st stephen', 'the eleven', 'turn on your lovelight',
                      'death dont have no mercy', 'feedback', 'and we bid you goodnight'],
            'description': 'First official live release. Extended psychedelic jams. Dark Star is 23 minutes.',
            'personnel': {
                'jerry garcia': 'Lead guitar, vocals',
                'bob weir': 'Rhythm guitar, vocals',
                'phil lesh': 'Bass',
                'bill kreutzmann': 'Drums',
                'mickey hart': 'Drums',
                'tom constanten': 'Keyboards',
            }
        },
        'europe 72': {
            'year': 1972,
            'label': 'Warner Bros.',
            'songs': ['cumberland blues', 'he\'s gone', 'one more saturday night', 'jack straw',
                      'you win again', 'china cat sunflower', 'i know you rider', 'brown eyed women',
                      'hurts me too', 'ramble on rose', 'sugar magnolia', 'mr charlie', 'tennessee jed',
                      'truckin', 'epilogue', 'prelude', 'morning dew'],
            'description': 'Triple live album from legendary European tour. Peak Pigpen era.',
            'personnel': {
                'jerry garcia': 'Lead guitar, vocals',
                'bob weir': 'Rhythm guitar, vocals',
                'phil lesh': 'Bass, vocals',
                'bill kreutzmann': 'Drums',
                'keith godchaux': 'Piano',
                'donna jean godchaux': 'Vocals',
                'pigpen': 'Organ, harmonica, vocals',
            }
        },
        'terrapin station': {
            'year': 1977,
            'label': 'Arista',
            'songs': ['estimated prophet', 'dancing in the street', 'passenger', 'samson and delilah',
                      'sunrise', 'terrapin station'],
            'description': 'Studio polish meets Dead jamming. Orchestral arrangements on title track.',
            'personnel': {
                'jerry garcia': 'Lead guitar, vocals',
                'bob weir': 'Rhythm guitar, vocals',
                'phil lesh': 'Bass',
                'bill kreutzmann': 'Drums',
                'mickey hart': 'Drums, percussion',
                'keith godchaux': 'Piano',
                'donna jean godchaux': 'Vocals',
            }
        },
        'in the dark': {
            'year': 1987,
            'label': 'Arista',
            'songs': ['touch of grey', 'hells buckets', 'when push comes to shove', 'west l.a. fadeaway',
                      'tons of steel', 'throwing stones', 'black muddy river'],
            'description': 'Commercial breakthrough. Touch of Grey hit MTV. Brent era peak.',
            'personnel': {
                'jerry garcia': 'Lead guitar, vocals',
                'bob weir': 'Rhythm guitar, vocals',
                'phil lesh': 'Bass',
                'bill kreutzmann': 'Drums',
                'mickey hart': 'Drums, percussion',
                'brent mydland': 'Keyboards, vocals',
            }
        },
    },
    'phish': {
        'a live one': {
            'year': 1995,
            'label': 'Elektra',
            'songs': ['bouncing around the room', 'stash', 'gumbo', 'montana', 'tweezer',
                      'simple', 'chalk dust torture', 'you enjoy myself', 'harry hood'],
            'description': 'Double live album. Definitive 90s Phish. 30+ minute Tweezer.',
        },
        'billy breathes': {
            'year': 1996,
            'label': 'Elektra',
            'songs': ['free', 'character zero', 'waste', 'taste', 'cars trucks buses',
                      'talk', 'theme from the bottom', 'train song', 'bliss', 'billy breathes', 'swept away', 'steep', 'prince caspian'],
            'description': 'Studio masterpiece. More focused songwriting.',
        },
    },
}

# ============================================================================
# LOCAL KNOWLEDGE BASE
# ============================================================================

LOCAL_KNOWLEDGE = {
    'grateful dead': {
        'bio': "The Grateful Dead were an American rock band formed in 1965 in Palo Alto, California. Known for their unique blend of rock, folk, country, jazz, bluegrass, blues, and psychedelic music, they became icons of the counterculture movement.",
        'wikipedia_url': 'https://en.wikipedia.org/wiki/Grateful_Dead',
        'members': {
            'jerry garcia': 'Lead guitar, vocals. Known for his melodic improvisations and sweet tone. Usually panned LEFT.',
            'bob weir': 'Rhythm guitar, vocals. Master of jazz chords and syncopated rhythms. Usually panned RIGHT.',
            'phil lesh': 'Bass. Classically trained, known for contrapuntal bass lines. Usually CENTER.',
            'bill kreutzmann': 'Drums. Solid groove master. Usually panned LEFT.',
            'mickey hart': 'Drums, percussion. Brought world rhythms to the band. Usually panned RIGHT.',
        },
        'learning_tips': "Focus on the interplay between Jerry and Bob - they rarely play the same thing. Jerry often plays melodic fills while Bob comps with unusual chord voicings. The key to Dead music is LISTENING to the other players.",
        'common_keys': ['G', 'A', 'E', 'D', 'C'],
        'style': 'Improvisational rock with folk, country, and jazz influences'
    },
    'phish': {
        'bio': "Phish is an American rock band formed in Burlington, Vermont in 1983. Known for musical improvisation, extended jams, and blending of genres including funk, progressive rock, and jazz fusion.",
        'wikipedia_url': 'https://en.wikipedia.org/wiki/Phish',
        'members': {
            'trey anastasio': 'Guitar, vocals. Virtuoso player known for composed sections and fiery improv.',
            'mike gordon': 'Bass. Melodic and adventurous bass lines.',
            'page mcconnell': 'Keyboards. Jazz-trained, adds sophisticated harmony.',
            'jon fishman': 'Drums. Inventive and propulsive.',
        },
        'learning_tips': "Phish songs often have composed 'through-composed' sections. Learn the head first, then study how they develop themes in jams. Trey's tone comes from careful pick attack and envelope filter use.",
        'style': 'Progressive rock with funk and jazz fusion elements'
    },
    'allman brothers': {
        'bio': "The Allman Brothers Band pioneered Southern rock and jam band music. Formed in 1969 in Jacksonville, Florida, known for extended improvisations and twin lead guitar harmonies.",
        'wikipedia_url': 'https://en.wikipedia.org/wiki/The_Allman_Brothers_Band',
        'members': {
            'duane allman': 'Lead/slide guitar. Legendary slide player, melodic and soulful. Usually panned LEFT.',
            'dickey betts': 'Lead guitar. Country-influenced, wrote many classics. Usually panned RIGHT.',
            'gregg allman': 'Keyboards, vocals. Soulful voice and Hammond organ.',
            'berry oakley': 'Bass. Melodic, fluid bass lines.',
            'butch trucks': 'Drums. Usually panned LEFT.',
            'jaimoe': 'Drums. Usually panned RIGHT.',
        },
        'learning_tips': "Study the twin guitar harmonies - often in thirds and sixths. Duane's slide work is all about tone and phrasing. The band's groove is deep and patient.",
        'style': 'Southern rock, blues rock, jam band'
    },
}

# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def fetch_track_info(track_name: str, artist: str = None,
                     source_url: str = None) -> Dict[str, Any]:
    """
    Fetch track info:
    - INSTANT for known artists (local knowledge)
    - External API lookup for unknown artists (Wikipedia/MusicBrainz)
    """
    info = {
        'track': track_name,
        'artist': artist,
        'source': source_url,
        'bio': None,
        'song_info': None,
        'album': None,
        'album_year': None,
        'album_description': None,
        'personnel': None,
        'learning_tips': None,
        'members': None,
        'style': None,
        'player_mapping': None,
        'era': None,
        'fetched_from': []
    }

    logger.info(f"🔍 Track info: {track_name} / {artist}")

    # Extract artist from track name if not provided
    if not artist and track_name:
        artist = extract_artist_from_title(track_name)
        info['artist'] = artist

    if not artist:
        info['bio'] = "Process a track to see artist information."
        info['learning_tips'] = "Upload a song to see track information and learning tips."
        return info

    artist_lower = artist.lower().strip()

    # ==========================================
    # PHASE 1: Check local knowledge (INSTANT)
    # ==========================================
    found_local = False
    for known_artist, knowledge in LOCAL_KNOWLEDGE.items():
        # Stricter matching to avoid false positives
        # Require exact match OR significant overlap (not just substring)
        is_match = False
        if artist_lower == known_artist:
            is_match = True  # Exact match
        elif len(known_artist) >= 6 and known_artist in artist_lower:
            # Known artist is substantial substring (e.g., "grateful dead" in "grateful dead live")
            is_match = True
        elif len(artist_lower) >= 6 and artist_lower in known_artist:
            # Input is substantial substring of known artist
            is_match = True

        if not is_match:
            continue

        logger.info(f"  ✓ Found locally: {known_artist}")
        info['bio'] = knowledge.get('bio')
        info['members'] = knowledge.get('members')
        info['learning_tips'] = knowledge.get('learning_tips')
        info['style'] = knowledge.get('style')
        info['common_keys'] = knowledge.get('common_keys')
        info['wikipedia_url'] = knowledge.get('wikipedia_url')
        info['fetched_from'].append('local_knowledge')
        info['player_mapping'] = get_player_mapping(known_artist)
        found_local = True
        break

    # Local album lookup
    if track_name:
        album_info = find_album_for_song(track_name, artist_lower)
        if album_info:
            info['album'] = album_info.get('album_name')
            info['album_year'] = album_info.get('year')
            info['album_description'] = album_info.get('description')
            info['personnel'] = album_info.get('personnel')
            info['fetched_from'].append('album_database')

    # Grateful Dead era
    if 'grateful dead' in artist_lower and info.get('album_year'):
        try:
            era_info = get_grateful_dead_era(int(info['album_year']))
            if era_info:
                info['era'] = era_info
        except:
            pass

    # Try to find song-specific Wikipedia page
    if track_name:
        try:
            # Clean up track name (remove "Live", parentheses, etc.)
            clean_track = track_name.split('(')[0].strip()

            # Try song-specific search: "Song Name (song)" or "Song Name (Artist song)"
            song_wiki = fetch_wikipedia_info(f"{clean_track} (song)")
            if not song_wiki:
                song_wiki = fetch_wikipedia_info(f"{clean_track} {artist}")
            if not song_wiki:
                song_wiki = fetch_wikipedia_info(clean_track)

            if song_wiki and song_wiki.get('url'):
                info['song_wikipedia_url'] = song_wiki.get('url')
                info['song_info'] = song_wiki.get('extract', '')[:500]
                logger.info(f"  ✓ Found song Wikipedia: {info['song_wikipedia_url']}")
        except Exception as e:
            logger.debug(f"  Song Wikipedia lookup failed: {e}")

    # Always add Genius link as fallback for song info (works for almost any song)
    if track_name and artist:
        try:
            genius_url = generate_genius_url(track_name, artist)
            info['genius_url'] = genius_url
            logger.info(f"  ✓ Generated Genius URL: {genius_url}")
        except Exception as e:
            logger.debug(f"  Genius URL generation failed: {e}")

    # If found locally, return instantly (FAST PATH)
    if found_local:
        logger.info(f"  ✓ Returning local data instantly")
        return info

    # ==========================================
    # PHASE 2: Unknown artist - fetch from APIs
    # ==========================================
    logger.info(f"  → Unknown artist '{artist}', trying external APIs...")

    # Try Wikipedia
    try:
        wiki_info = fetch_wikipedia_info(artist)
        if wiki_info:
            info['bio'] = wiki_info.get('extract', '')[:800]
            info['wikipedia_url'] = wiki_info.get('url')
            info['thumbnail'] = wiki_info.get('thumbnail')
            info['fetched_from'].append('wikipedia')
            logger.info(f"  ✓ Got Wikipedia info")
    except Exception as e:
        logger.debug(f"  Wikipedia failed: {e}")

    # Try MusicBrainz for album
    if not info.get('album') and track_name:
        try:
            mb_album = fetch_musicbrainz_album(artist, track_name)
            if mb_album:
                info['album'] = mb_album.get('album_name')
                info['album_year'] = mb_album.get('year')
                info['fetched_from'].append('musicbrainz')
                logger.info(f"  ✓ Got album: {info['album']}")
        except Exception as e:
            logger.debug(f"  MusicBrainz failed: {e}")

    # Ensure we always return something useful
    if not info['bio']:
        info['bio'] = f"{artist} - Use stem separation to isolate and study individual instrument parts."
    if not info['learning_tips']:
        info['learning_tips'] = f"Listen to {artist}'s recordings and use the isolated stems to study each instrument."

    logger.info(f"  ✓ Done: {info['fetched_from']}")
    return info


def get_player_mapping(artist_key: str) -> Dict[str, str]:
    """
    Get the player position mapping for an artist.
    Returns a dict mapping stem names to player names.
    """
    return PLAYER_POSITIONS.get(artist_key, {}).copy()


# Bands that benefit from stereo splitting (dual guitars, dual drums, panned keyboards)
STEREO_SPLIT_RECOMMENDED = {
    'grateful dead',      # Jerry left, Bob right, dual drums
    'allman brothers',    # Duane left, Dickey right, dual drums
    'allman brothers band',
    'the beatles',        # John left, George right
    'beatles',
    'led zeppelin',       # Page multi-tracked
    'pink floyd',         # Gilmour multi-tracked
    'eagles',             # Multiple guitar layers
    'steely dan',         # Multiple guitar layers
    'fleetwood mac',      # Multiple guitar layers
    'television',         # Tom Verlaine/Richard Lloyd
    'the who',            # Pete multi-tracked
    'cream',              # Clapton multi-tracked
    'derek and the dominos',
    'phish',              # Trey multi-tracked
}


def should_stereo_split(artist: str) -> bool:
    """
    Check if stereo splitting is recommended for this artist.
    Returns True for bands with known dual guitar/drum setups.
    """
    if not artist:
        return False
    artist_key = artist.lower().strip()
    # Check direct match
    if artist_key in STEREO_SPLIT_RECOMMENDED:
        return True
    # Check partial match
    for band in STEREO_SPLIT_RECOMMENDED:
        if band in artist_key or artist_key in band:
            return True
    return False


def get_player_mapping_from_personnel(personnel: Dict[str, str], artist_key: str) -> Dict[str, str]:
    """
    Create player mapping from album personnel info.
    """
    mapping = get_player_mapping(artist_key) or {}

    # Update with album-specific personnel
    for name, role in personnel.items():
        role_lower = role.lower()
        name_title = name.title()

        if 'lead guitar' in role_lower or 'guitar' in role_lower:
            if 'vocals' in role_lower:
                mapping['vocals'] = name_title
            # Keep existing L/R mapping if present
        if 'bass' in role_lower:
            mapping['bass'] = name_title
            mapping['bass_left'] = name_title
            mapping['bass_center'] = name_title
        if 'drums' in role_lower:
            if mapping.get('drums') and 'Mickey' in mapping.get('drums', ''):
                continue  # Keep dual drummer setup
            mapping['drums'] = name_title
        if 'keyboard' in role_lower or 'piano' in role_lower or 'organ' in role_lower:
            mapping['piano'] = name_title
            mapping['piano_left'] = name_title
            mapping['piano_right'] = name_title

    return mapping


def find_album_for_song(song_name: str, artist_key: str) -> Optional[Dict[str, Any]]:
    """
    Find which album a song appears on.
    """
    song_lower = song_name.lower().strip()

    # Clean up common patterns in song names
    for pattern in [' - live', ' (live)', ' - remastered', ' (remastered)', ' - ', ' – ']:
        if pattern in song_lower:
            song_lower = song_lower.split(pattern)[0].strip()

    # Check if we have albums for this artist
    artist_albums = None
    for known_artist, albums in ALBUM_DATABASE.items():
        if known_artist in artist_key or artist_key in known_artist:
            artist_albums = albums
            break

    if not artist_albums:
        return None

    # Search for the song in albums
    for album_name, album_data in artist_albums.items():
        songs = album_data.get('songs', [])
        for album_song in songs:
            # Fuzzy match - check if song name is contained or matches
            if song_lower in album_song.lower() or album_song.lower() in song_lower:
                return {
                    'album_name': album_name.title(),
                    'year': album_data.get('year'),
                    'description': album_data.get('description'),
                    'personnel': album_data.get('personnel'),
                    'label': album_data.get('label'),
                }

    return None


def get_grateful_dead_era(year: int) -> Optional[Dict[str, Any]]:
    """
    Determine which Grateful Dead era based on year.
    Returns keyboardist and era notes.
    """
    for era_name, (start, end, keyboardist, notes) in GRATEFUL_DEAD_ERAS.items():
        if start <= year <= end:
            return {
                'era_name': era_name,
                'keyboardist': keyboardist,
                'notes': notes,
                'years': f"{start}-{end}"
            }
    return None


def extract_artist_from_title(title: str) -> Optional[str]:
    """
    Try to extract artist name from a track title.
    """
    if not title:
        return None

    title = title.strip()

    # Pattern: "Artist - Song"
    if ' - ' in title:
        parts = title.split(' - ', 1)
        return parts[0].strip()

    # Pattern: "Artist – Song" (en-dash)
    if ' – ' in title:
        parts = title.split(' – ', 1)
        return parts[0].strip()

    # Pattern: "Song by Artist"
    if ' by ' in title.lower():
        parts = re.split(r'\s+by\s+', title, flags=re.IGNORECASE)
        if len(parts) >= 2:
            return parts[-1].strip()

    return None


def generate_genius_url(track_name: str, artist: str) -> str:
    """
    Generate a Genius.com URL for a song.
    Genius URLs follow the pattern: https://genius.com/Artist-name-song-name-lyrics
    """
    # Clean up track name - remove common suffixes
    clean_track = track_name.split('(')[0].strip()
    for suffix in [' - Live', ' - Remastered', ' - Remaster', ' Live', ' Remastered']:
        if clean_track.lower().endswith(suffix.lower()):
            clean_track = clean_track[:-len(suffix)].strip()

    # Clean up artist name
    clean_artist = artist.strip()
    # Remove "The " prefix for URL
    if clean_artist.lower().startswith('the '):
        clean_artist = clean_artist[4:]

    # Convert to URL-friendly format: spaces to hyphens, remove special chars
    def slugify(text):
        # Replace special chars and spaces with hyphens
        slug = re.sub(r'[^\w\s-]', '', text)  # Remove special chars
        slug = re.sub(r'[\s_]+', '-', slug)   # Replace spaces with hyphens
        slug = re.sub(r'-+', '-', slug)       # Collapse multiple hyphens
        return slug.strip('-').title()

    artist_slug = slugify(clean_artist)
    track_slug = slugify(clean_track)

    return f"https://genius.com/{artist_slug}-{track_slug}-lyrics"


def fetch_wikipedia_info(query: str) -> Optional[Dict[str, Any]]:
    """Fetch summary from Wikipedia API."""
    if not query:
        return None

    try:
        encoded_query = urllib.parse.quote(query.replace(' ', '_'))
        url = f"{WIKIPEDIA_API}{encoded_query}"

        req = urllib.request.Request(url, headers={
            'User-Agent': 'StemScribe/1.0 (music learning tool)'
        })

        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))

            if data.get('type') == 'disambiguation':
                return None

            return {
                'title': data.get('title'),
                'extract': data.get('extract'),
                'url': data.get('content_urls', {}).get('desktop', {}).get('page'),
                'thumbnail': data.get('thumbnail', {}).get('source')
            }

    except Exception as e:
        logger.debug(f"Wikipedia fetch failed for {query}: {e}")
        return None


def fetch_lastfm_info(artist: str, track: str = None) -> Optional[Dict[str, Any]]:
    """Fetch info from Last.fm API."""
    if not LASTFM_API_KEY or not artist:
        return None

    try:
        params = {
            'method': 'artist.getinfo',
            'artist': artist,
            'api_key': LASTFM_API_KEY,
            'format': 'json'
        }

        url = f"http://ws.audioscrobbler.com/2.0/?{urllib.parse.urlencode(params)}"

        req = urllib.request.Request(url, headers={'User-Agent': 'StemScribe/1.0'})

        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))
            artist_data = data.get('artist', {})

            return {
                'bio': artist_data.get('bio', {}).get('summary'),
                'tags': [t['name'] for t in artist_data.get('tags', {}).get('tag', [])],
                'similar': [s['name'] for s in artist_data.get('similar', {}).get('artist', [])[:5]],
                'listeners': artist_data.get('stats', {}).get('listeners')
            }

    except Exception as e:
        logger.debug(f"Last.fm fetch failed: {e}")
        return None


def fetch_musicbrainz_album(artist: str, track: str) -> Optional[Dict[str, Any]]:
    """
    Fetch album info from MusicBrainz API.
    Works for ANY artist - not just those in our local database.
    """
    if not artist or not track:
        return None

    try:
        # Clean up track name
        track_clean = track.lower().strip()
        for pattern in [' - live', ' (live)', ' - remastered', ' (remastered)', ' - ', ' – ']:
            if pattern in track_clean:
                track_clean = track_clean.split(pattern)[0].strip()

        # Search for the recording
        query = f'recording:"{track_clean}" AND artist:"{artist}"'
        encoded_query = urllib.parse.quote(query)
        url = f"{MUSICBRAINZ_API}recording/?query={encoded_query}&fmt=json&limit=5"

        req = urllib.request.Request(url, headers={
            'User-Agent': 'StemScribe/1.0 (jkozelski@gmail.com)'  # MusicBrainz requires contact info
        })

        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))

        recordings = data.get('recordings', [])
        if not recordings:
            logger.debug(f"No MusicBrainz recordings found for {track} by {artist}")
            return None

        # Get the first recording with release info
        for recording in recordings:
            releases = recording.get('releases', [])
            if releases:
                # Prefer studio albums over compilations/singles
                for release in releases:
                    release_group = release.get('release-group', {})
                    primary_type = release_group.get('primary-type', '')

                    # Skip compilations and singles if we have other options
                    if primary_type == 'Album':
                        return {
                            'album_name': release.get('title'),
                            'year': release.get('date', '')[:4] if release.get('date') else None,
                            'release_id': release.get('id'),
                            'artist_credit': recording.get('artist-credit', [{}])[0].get('name'),
                            'description': f"From the album '{release.get('title')}'",
                            'source': 'musicbrainz'
                        }

                # Fallback to first release if no Album type found
                release = releases[0]
                return {
                    'album_name': release.get('title'),
                    'year': release.get('date', '')[:4] if release.get('date') else None,
                    'release_id': release.get('id'),
                    'artist_credit': recording.get('artist-credit', [{}])[0].get('name'),
                    'description': f"Released on '{release.get('title')}'",
                    'source': 'musicbrainz'
                }

        return None

    except Exception as e:
        logger.debug(f"MusicBrainz fetch failed: {e}")
        return None


def fetch_musicbrainz_artist_members(artist: str) -> Optional[Dict[str, str]]:
    """
    Fetch band members from MusicBrainz for any artist.
    """
    if not artist:
        return None

    try:
        # Search for the artist
        query = f'artist:"{artist}"'
        encoded_query = urllib.parse.quote(query)
        url = f"{MUSICBRAINZ_API}artist/?query={encoded_query}&fmt=json&limit=1"

        req = urllib.request.Request(url, headers={
            'User-Agent': 'StemScribe/1.0 (jkozelski@gmail.com)'
        })

        with urllib.request.urlopen(req, timeout=3) as response:
            data = json.loads(response.read().decode('utf-8'))

        artists = data.get('artists', [])
        if not artists:
            return None

        artist_id = artists[0].get('id')
        if not artist_id:
            return None

        # Fetch artist details with relations (band members)
        detail_url = f"{MUSICBRAINZ_API}artist/{artist_id}?inc=artist-rels&fmt=json"

        req = urllib.request.Request(detail_url, headers={
            'User-Agent': 'StemScribe/1.0 (jkozelski@gmail.com)'
        })

        time.sleep(0.2)  # Brief pause for MusicBrainz rate limit

        with urllib.request.urlopen(req, timeout=3) as response:
            artist_data = json.loads(response.read().decode('utf-8'))

        relations = artist_data.get('relations', [])
        members = {}

        for rel in relations:
            if rel.get('type') == 'member of band':
                member = rel.get('artist', {})
                member_name = member.get('name', '')
                attributes = rel.get('attributes', [])

                # Build role description from attributes
                role = ', '.join(attributes) if attributes else 'Member'

                if member_name:
                    members[member_name.lower()] = role

        return members if members else None

    except Exception as e:
        logger.debug(f"MusicBrainz artist fetch failed: {e}")
        return None


def generate_learning_tips(style: str, artist: str) -> str:
    """Generate generic learning tips based on style."""
    tips = []
    style_lower = style.lower()

    if 'jazz' in style_lower:
        tips.append("Focus on chord extensions (7ths, 9ths, 13ths) and voice leading.")
    if 'blues' in style_lower:
        tips.append("Master the blues scale and learn to bend notes expressively.")
    if 'rock' in style_lower:
        tips.append("Work on your rhythm playing - tight timing is essential.")
    if 'folk' in style_lower or 'country' in style_lower:
        tips.append("Learn the common chord progressions (I-IV-V, I-V-vi-IV).")
    if 'funk' in style_lower:
        tips.append("The groove is king - practice with a metronome.")
    if 'improv' in style_lower or 'jam' in style_lower:
        tips.append("Learn to listen actively while playing - react to what others do.")

    if not tips:
        tips.append(f"Study recordings of {artist} to internalize their unique style.")

    return ' '.join(tips)


def get_instrument_tips(instrument: str, style: str = None) -> str:
    """Get tips for learning a specific instrument part."""
    tips = {
        'guitar': "Focus on tone first, speed will come. Practice cleanly at slow tempos.",
        'bass': "Lock in with the drums. The bass is the bridge between rhythm and harmony.",
        'drums': "Use the transcribed MIDI to study the groove at slow speeds. Focus on ghost notes.",
        'piano': "Work on both hands separately before combining. Voice leading matters!",
        'vocals': "The melody is your guide - learn the phrasing and rhythmic feel.",
        'other': "This stem may contain multiple instruments - use stereo split if panned."
    }

    return tips.get(instrument.lower().split('_')[0], tips['other'])


def get_stem_display_name(stem_name: str, player_mapping: Dict[str, str] = None) -> str:
    """
    Get the display name for a stem, using player name if available.
    """
    if player_mapping and stem_name in player_mapping:
        return player_mapping[stem_name]

    # Default formatting
    return stem_name.replace('_', ' ').title()


# Export for use in app.py
__all__ = [
    'fetch_track_info',
    'extract_artist_from_title',
    'get_instrument_tips',
    'get_stem_display_name',
    'get_player_mapping',
    'find_album_for_song',
    'LOCAL_KNOWLEDGE',
    'PLAYER_POSITIONS',
    'ALBUM_DATABASE',
]
