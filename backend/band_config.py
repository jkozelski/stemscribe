"""
Band Configuration for StemScribe
==================================
Easy-to-edit band configurations. Add new bands here!

For most bands, MusicBrainz handles album/personnel lookup automatically.
This config is for:
1. Stereo split recommendations (dual guitars, dual drums)
2. Player position mappings for known panning
3. Special handling like era detection
"""

# ============================================================================
# STEREO SPLIT BANDS
# These bands benefit from stereo splitting due to:
# - Dual lead guitars
# - Dual drummers
# - Heavy multi-tracking
# - Known panning conventions
# ============================================================================

STEREO_SPLIT_BANDS = {
    # Jam Bands
    'grateful dead',
    'phish',
    'allman brothers',
    'allman brothers band',
    'widespread panic',
    'string cheese incident',
    'moe.',
    'umphrey\'s mcgee',
    'goose',
    'pigeons playing ping pong',
    'trey anastasio band',
    'phil lesh and friends',
    'dead and company',
    'billy strings',
    'tedeschi trucks band',
    'derek trucks band',
    'gov\'t mule',
    'little feat',
    'the band',

    # Classic Rock - Dual/Multi-tracked Guitars
    'the rolling stones',
    'rolling stones',
    'the who',
    'led zeppelin',
    'cream',
    'jimi hendrix experience',
    'jimi hendrix',
    'the doors',
    'the eagles',
    'eagles',
    'fleetwood mac',
    'pink floyd',
    'the beatles',
    'beatles',
    'queen',
    'aerosmith',
    'van halen',
    'def leppard',
    'boston',
    'journey',
    'kansas',
    'styx',
    'foreigner',
    'heart',

    # Southern Rock
    'lynyrd skynyrd',
    'marshall tucker band',
    'the marshall tucker band',
    'molly hatchet',
    'blackfoot',
    '38 special',
    'outlaws',
    'atlanta rhythm section',
    'wet willie',

    # Blues Rock
    'stevie ray vaughan',
    'stevie ray vaughan and double trouble',
    'eric clapton',
    'derek and the dominos',
    'bb king',
    'b.b. king',
    'buddy guy',
    'gary moore',
    'joe bonamassa',
    'kenny wayne shepherd',
    'john mayer trio',

    # Prog Rock
    'yes',
    'genesis',
    'king crimson',
    'rush',
    'emerson lake and palmer',
    'elp',
    'jethro tull',
    'gentle giant',
    'camel',
    'porcupine tree',
    'tool',
    'dream theater',

    # Funk/Soul
    'steely dan',
    'earth wind and fire',
    'earth wind & fire',
    'parliament',
    'funkadelic',
    'parliament-funkadelic',
    'tower of power',
    'average white band',
    'the meters',
    'sly and the family stone',

    # Punk/Alt - Multi-tracked
    'television',
    'sonic youth',
    'the pixies',
    'radiohead',
    'my bloody valentine',

    # 80s/90s Rock
    'guns n\' roses',
    'guns n roses',
    'metallica',
    'iron maiden',
    'judas priest',
    'black sabbath',
    'deep purple',
    'scorpions',
    'ac/dc',
    'acdc',
    'thin lizzy',  # Dual guitar harmonies!
}


# ============================================================================
# PLAYER POSITIONS
# For bands with known stereo panning conventions
# ============================================================================

PLAYER_POSITIONS = {
    'grateful dead': {
        'guitar_left': 'Bob Weir',
        'guitar_right': 'Jerry Garcia',
        'bass': 'Phil Lesh',
        'drums_left': 'Bill Kreutzmann',
        'drums_right': 'Mickey Hart',
        'piano': 'Keys',
    },
    'phish': {
        'guitar': 'Trey Anastasio',
        'bass': 'Mike Gordon',
        'drums': 'Jon Fishman',
        'piano': 'Page McConnell',
    },
    'allman brothers': {
        'guitar_left': 'Duane/Warren/Derek',
        'guitar_right': 'Dickey Betts',
        'bass': 'Berry/Allen/Oteil',
        'drums_left': 'Butch Trucks',
        'drums_right': 'Jaimoe',
        'piano': 'Gregg Allman',
    },
    'little feat': {
        'guitar_left': 'Lowell George',
        'guitar_right': 'Paul Barrere',
        'bass': 'Kenny Gradney',
        'drums': 'Richie Hayward',
        'piano': 'Bill Payne',
        'percussion': 'Sam Clayton',
    },
    'rolling stones': {
        'guitar_left': 'Keith Richards',
        'guitar_right': 'Ronnie Wood/Mick Taylor',
        'bass': 'Bill Wyman/Darryl Jones',
        'drums': 'Charlie Watts',
        'piano': 'Various',
    },
    'the who': {
        'guitar': 'Pete Townshend',
        'bass': 'John Entwistle',
        'drums': 'Keith Moon',
        'vocals': 'Roger Daltrey',
    },
    'led zeppelin': {
        'guitar': 'Jimmy Page',
        'guitar_left': 'Jimmy Page (rhythm)',
        'guitar_right': 'Jimmy Page (lead)',
        'bass': 'John Paul Jones',
        'drums': 'John Bonham',
        'piano': 'John Paul Jones',
    },
    'thin lizzy': {
        'guitar_left': 'Scott Gorham',
        'guitar_right': 'Gary Moore/Brian Robertson',
        'bass': 'Phil Lynott',
        'drums': 'Brian Downey',
    },
    'lynyrd skynyrd': {
        'guitar_left': 'Gary Rossington',
        'guitar_center': 'Allen Collins',
        'guitar_right': 'Steve Gaines',
        'bass': 'Leon Wilkeson',
        'drums': 'Artimus Pyle',
        'piano': 'Billy Powell',
    },
    'the band': {
        'guitar': 'Robbie Robertson',
        'bass': 'Rick Danko',
        'drums': 'Levon Helm',
        'piano': 'Richard Manuel/Garth Hudson',
    },
    'tedeschi trucks band': {
        'guitar_left': 'Derek Trucks',
        'guitar_right': 'Mark Rivers/Tim Lefebvre era guitars',
        'bass': 'Tim Lefebvre/Kebbi Williams',
        'drums_left': 'Tyler Greenwell',
        'drums_right': 'JJ Johnson',
        'piano': 'Gabe Dixon',
    },
    'widespread panic': {
        'guitar_left': 'John Bell',
        'guitar_right': 'Jimmy Herring',
        'bass': 'Dave Schools',
        'drums': 'Sunny Ortiz/Duane Trucks',
        'piano': 'John Hermann',
    },
    'cream': {
        'guitar': 'Eric Clapton',
        'bass': 'Jack Bruce',
        'drums': 'Ginger Baker',
    },
    'derek and the dominos': {
        'guitar_left': 'Eric Clapton',
        'guitar_right': 'Duane Allman',
        'bass': 'Carl Radle',
        'drums': 'Jim Gordon',
        'piano': 'Bobby Whitlock',
    },
    'rush': {
        'guitar': 'Alex Lifeson',
        'bass': 'Geddy Lee',
        'drums': 'Neil Peart',
        'piano': 'Geddy Lee',
    },
    'yes': {
        'guitar': 'Steve Howe',
        'bass': 'Chris Squire',
        'drums': 'Various',
        'piano': 'Rick Wakeman/Tony Kaye',
    },
    'pink floyd': {
        'guitar': 'David Gilmour',
        'bass': 'Roger Waters',
        'drums': 'Nick Mason',
        'piano': 'Richard Wright',
    },
    'queen': {
        'guitar': 'Brian May',
        'bass': 'John Deacon',
        'drums': 'Roger Taylor',
        'piano': 'Freddie Mercury',
    },
}


# ============================================================================
# QUICK ADD FUNCTIONS
# ============================================================================

def should_stereo_split(artist: str) -> bool:
    """Check if stereo splitting is recommended."""
    if not artist:
        return False
    artist_lower = artist.lower().strip()
    for band in STEREO_SPLIT_BANDS:
        if band in artist_lower or artist_lower in band:
            return True
    return False


def get_player_positions(artist: str) -> dict:
    """Get player positions for an artist.

    Uses strict matching to avoid false positives - requires exact match
    or significant substring overlap (6+ chars).
    """
    if not artist:
        return {}
    artist_lower = artist.lower().strip()
    for band, positions in PLAYER_POSITIONS.items():
        # Stricter matching - avoid false positives
        if artist_lower == band:
            return positions.copy()  # Exact match
        elif len(band) >= 6 and band in artist_lower:
            return positions.copy()  # Band is substantial substring of artist
        elif len(artist_lower) >= 6 and artist_lower in band:
            return positions.copy()  # Artist is substantial substring of band
    return {}


def add_band_to_stereo_split(band_name: str):
    """Add a band to the stereo split list at runtime."""
    STEREO_SPLIT_BANDS.add(band_name.lower().strip())


def add_player_positions(band_name: str, positions: dict):
    """Add player positions for a band at runtime."""
    PLAYER_POSITIONS[band_name.lower().strip()] = positions


# ============================================================================
# QUICK KNOWLEDGE LOOKUP (for learning tips)
# ============================================================================

STYLE_TIPS = {
    'jam band': "Focus on listening and communication. These bands prioritize interplay over virtuosity. Learn the composed sections first, then study how they develop ideas in jams.",
    'southern rock': "Deep groove and feel. Don't rush. Twin guitar harmonies often in thirds and sixths. Blues-based with country influences.",
    'blues rock': "It's all about the feel and dynamics. Bends, vibrato, and phrasing matter more than speed. Listen to how notes breathe.",
    'prog rock': "Complex compositions with odd time signatures. Learn the parts exactly first, then understand the theory.",
    'classic rock': "Strong melodies and hooks. Guitar tone is key. Study the rhythm playing as much as the leads.",
    'funk': "The one is everything. Lock with the drums. Less is more - it's about the pocket.",
    'punk': "Energy and attitude. Simple chords played with conviction. Tight, aggressive playing.",
}


def get_learning_tips(artist: str, style: str = None) -> str:
    """Get learning tips based on artist or style."""
    # Map artists to styles
    artist_styles = {
        'grateful dead': 'jam band',
        'phish': 'jam band',
        'allman brothers': 'southern rock',
        'lynyrd skynyrd': 'southern rock',
        'little feat': 'southern rock',
        'stevie ray vaughan': 'blues rock',
        'eric clapton': 'blues rock',
        'bb king': 'blues rock',
        'yes': 'prog rock',
        'rush': 'prog rock',
        'king crimson': 'prog rock',
        'rolling stones': 'classic rock',
        'led zeppelin': 'classic rock',
        'the who': 'classic rock',
        'parliament': 'funk',
        'tower of power': 'funk',
        'steely dan': 'jazz fusion / studio perfection',
    }

    if artist:
        artist_lower = artist.lower()
        for band, band_style in artist_styles.items():
            if band in artist_lower:
                style = band_style
                break

    if style and style.lower() in STYLE_TIPS:
        return STYLE_TIPS[style.lower()]

    return "Study the recordings carefully. Focus on tone, timing, and feel as much as the notes."


# ============================================================================
# DUAL DRUMMER BANDS (for special drum handling)
# ============================================================================

DUAL_DRUMMER_BANDS = {
    'grateful dead',
    'allman brothers',
    'allman brothers band',
    'tedeschi trucks band',
    'santana',  # Multiple percussion
    'king crimson',  # Various eras
    'doobie brothers',
    'the doobie brothers',
    'slipknot',
    'adam ant',
}


def has_dual_drummers(artist: str) -> bool:
    """Check if band has dual drummers."""
    if not artist:
        return False
    artist_lower = artist.lower().strip()
    for band in DUAL_DRUMMER_BANDS:
        if band in artist_lower or artist_lower in band:
            return True
    return False
