"""
Extended Band Database for StemScribe
======================================
Deep coverage for jam bands and classic rock artists.
Includes:
- Player position mappings (stereo panning)
- Era-specific lineups
- Album databases with song lists
- Learning tips and style guides

Merge this with track_info.py for full coverage.
"""

# ============================================================================
# PHISH - Complete Coverage
# ============================================================================

PHISH_PLAYER_POSITIONS = {
    'guitar': 'Trey Anastasio',
    'guitar_left': 'Trey Anastasio',
    'guitar_right': 'Trey Anastasio',
    'guitar_center': 'Trey Anastasio',
    'bass': 'Mike Gordon',
    'bass_left': 'Mike Gordon',
    'bass_center': 'Mike Gordon',
    'drums': 'Jon Fishman',
    'drums_left': 'Jon Fishman',
    'drums_right': 'Jon Fishman',
    'piano': 'Page McConnell',
    'piano_left': 'Page (piano)',
    'piano_right': 'Page (organ/clav)',
    'vocals': 'Vocals',
    'vocals_left': 'Trey/Mike',
    'vocals_right': 'Page',
    'other': 'Other',
}

PHISH_ERAS = {
    # Era name: (start_year, end_year, description, notable_gear)
    'early': (1983, 1988, 'College/bar band era. Raw, experimental, comedy songs.', {
        'trey_guitar': 'Various guitars, small amps',
        'page_keys': 'Basic keyboards',
    }),
    'giant_country_horns': (1991, 1991, 'Brief horn section era with Giant Country Horns.', {}),
    'classic_1.0': (1989, 1996, 'Classic era. Marathon jams, composed suites, peak improvisation.', {
        'trey_guitar': 'Custom Languedoc guitars, Mesa Boogie Mark III',
        'page_keys': 'Hammond B3, Fender Rhodes, Clavinet',
    }),
    'late_1.0': (1997, 2000, 'Funk era. Dense, ambient jams. Cow Funk.', {
        'trey_guitar': 'Languedoc, tube screamer, envelope filter',
        'page_keys': 'B3, Rhodes, Mini Moog, Clav',
    }),
    'hiatus': (2000, 2002, 'Band on hiatus.', {}),
    '2.0': (2003, 2004, 'Short reunion. Darker, heavier sound.', {
        'trey_guitar': 'Languedoc, Divided Sky custom',
        'page_keys': 'Expanded synth rig',
    }),
    'breakup': (2004, 2008, 'Band broken up.', {}),
    '3.0_early': (2009, 2012, 'Reunion era. Return to roots, cleaner sound.', {
        'trey_guitar': 'Languedoc G2, Komet amps',
        'page_keys': 'Nord, Hammond clone, Rhodes',
    }),
    '3.0_mature': (2013, 2020, 'Peak 3.0. Extended jams, more risks.', {
        'trey_guitar': 'Languedoc, Komet 60, various pedals',
        'page_keys': 'Full rig with synths',
    }),
    '4.0': (2021, 2026, 'Current era. Tighter playing, deep song rotation.', {
        'trey_guitar': 'Languedoc, Komet Concorde',
        'page_keys': 'Modern hybrid rig',
    }),
}

PHISH_ALBUMS = {
    'junta': {
        'year': 1989,
        'label': 'Elektra (1992 re-release)',
        'songs': ['you enjoy myself', 'fee', 'fluffhead', 'fluff\'s travels', 'esther',
                  'golgi apparatus', 'foam', 'dinner and a movie', 'david bowie',
                  'the divided sky', 'sanity', 'icculus', 'contact', 'union federal'],
        'description': 'Double album debut. Establishes composed suite style. YEM and Fluffhead are epics.',
        'era': 'classic_1.0',
    },
    'lawn boy': {
        'year': 1990,
        'label': 'Absolute A-Go-Go',
        'songs': ['the squirming coil', 'reba', 'my sweet one', 'split open and melt',
                  'the oh kee pa ceremony', 'bathtub gin', 'run like an antelope',
                  'lawn boy', 'bouncing around the room'],
        'description': 'Cleaner production. Reba and Antelope become jam vehicles.',
        'era': 'classic_1.0',
    },
    'a picture of nectar': {
        'year': 1992,
        'label': 'Elektra',
        'songs': ['llama', 'eliza', 'cavern', 'poor heart', 'stash', 'manteca',
                  'guelah papyrus', 'magilla', 'the landlady', 'glide', 'tweezer',
                  'the mango song', 'chalk dust torture', 'faht', 'catapult', 'tweezer reprise'],
        'description': 'Major label debut. Stash and Tweezer become legendary jam vehicles.',
        'era': 'classic_1.0',
    },
    'rift': {
        'year': 1993,
        'label': 'Elektra',
        'songs': ['rift', 'fast enough for you', 'maze', 'sparkle', 'horn', 'the wedge',
                  'my friend my friend', 'weigh', 'all things reconsidered', 'mound',
                  'it\'s ice', 'lengthwise', 'the horse', 'silent in the morning'],
        'description': 'Concept album about dreams. Tighter compositions. Maze is key.',
        'era': 'classic_1.0',
    },
    'hoist': {
        'year': 1994,
        'label': 'Elektra',
        'songs': ['julius', 'down with disease', 'if i could', 'riker\'s mailbox', 'axilla ii',
                  'lifeboy', 'sample in a jar', 'wolfman\'s brother', 'scent of a mule',
                  'dog faced boy', 'demand'],
        'description': 'Polished studio sound. Down with Disease and Wolfman become staples.',
        'era': 'classic_1.0',
    },
    'a live one': {
        'year': 1995,
        'label': 'Elektra',
        'songs': ['bouncing around the room', 'stash', 'gumbo', 'montana', 'tweezer',
                  'simple', 'chalk dust torture', 'you enjoy myself', 'harry hood'],
        'description': 'Double live album. Definitive 90s Phish. 30+ minute Tweezer. Essential.',
        'era': 'classic_1.0',
    },
    'billy breathes': {
        'year': 1996,
        'label': 'Elektra',
        'songs': ['free', 'character zero', 'waste', 'taste', 'cars trucks buses',
                  'talk', 'theme from the bottom', 'train song', 'bliss', 'billy breathes',
                  'swept away', 'steep', 'prince caspian'],
        'description': 'Studio masterpiece. More focused songwriting. Produced by Steve Lillywhite.',
        'era': 'classic_1.0',
    },
    'slip stitch and pass': {
        'year': 1997,
        'label': 'Elektra',
        'songs': ['cities', 'wolfman\'s brother', 'jesus just left chicago', 'weigh',
                  'mike\'s song', 'i am hydrogen', 'weekapaug groove', 'hello my baby'],
        'description': 'Live album from Hamburg. Tight European show. Great Mike\'s Groove.',
        'era': 'late_1.0',
    },
    'the story of the ghost': {
        'year': 1998,
        'label': 'Elektra',
        'songs': ['ghost', 'birds of a feather', 'meat', 'guyute', 'fikus', 'shafty',
                  'limb by limb', 'frankie says', 'brian and robert', 'water in the sky',
                  'roggae', 'wading in the velvet sea', 'the moma dance', 'end of session'],
        'description': 'Funk era studio album. Ghost and Moma Dance are key jams.',
        'era': 'late_1.0',
    },
    'farmhouse': {
        'year': 2000,
        'label': 'Elektra',
        'songs': ['farmhouse', 'twist', 'bug', 'back on the train', 'heavy things',
                  'gotta jibboo', 'dirt', 'piper', 'sleep', 'the inlaw josie wales',
                  'first tube'],
        'description': 'Accessible rock album. Piper and First Tube are jam highlights.',
        'era': 'late_1.0',
    },
    'round room': {
        'year': 2002,
        'label': 'Elektra',
        'songs': ['pebbles and marbles', 'anything but me', 'round room', 'mexican cousin',
                  'friday', 'seven below', 'mock song', 'waves', 'thunder'],
        'description': 'Quick studio session. Seven Below is standout.',
        'era': '2.0',
    },
    'fuego': {
        'year': 2014,
        'label': 'JEMP',
        'songs': ['fuego', 'the line', 'devotion to a dream', 'sing monica', 'wombat',
                  'wingsuit', 'waiting all night', 'snow', 'monica'],
        'description': 'First 3.0 studio album. Fuego becomes major jam vehicle.',
        'era': '3.0_mature',
    },
    'big boat': {
        'year': 2016,
        'label': 'JEMP',
        'songs': ['friends', 'home', 'blaze on', 'breath and burning', 'tide turns',
                  'things people do', 'waking up dead', 'i always wanted it this way',
                  'running out of time', 'petrichor', 'miss you', 'more'],
        'description': 'Dense studio production. Blaze On becomes singalong favorite.',
        'era': '3.0_mature',
    },
    'sigma oasis': {
        'year': 2020,
        'label': 'JEMP',
        'songs': ['everything\'s right', 'steam', 'leaves', 'thread', 'mercury',
                  'sigma oasis'],
        'description': 'Quarantine album. Thread and Mercury are highlights.',
        'era': '3.0_mature',
    },
}

PHISH_KNOWLEDGE = {
    'bio': "Phish is an American rock band formed in Burlington, Vermont in 1983. Known for musical improvisation, extended jams, blending of genres including funk, progressive rock, psychedelic rock, and jazz fusion. Often compared to the Grateful Dead but with their own distinct style focused on composed sections, signal jamming, and musical communication.",
    'members': {
        'trey anastasio': 'Lead guitar, vocals. Virtuoso player known for composed sections and fiery improv. Uses custom Languedoc guitars with a distinctive compressed clean tone that can shift to singing sustain. Master of the "bliss jam" build.',
        'mike gordon': 'Bass, vocals. Melodic and adventurous bass lines. Often leads jams with quirky melodic ideas. Known for unusual time feel and walking bass lines that drive Type II jams.',
        'page mcconnell': 'Keyboards, vocals. Jazz-trained, adds sophisticated harmony. Hammond B3, Rhodes, Clavinet, and synths. Often holds down the groove while Trey and Mike explore.',
        'jon fishman': 'Drums, vacuum. Inventive and propulsive drumming. Known for complex fills and ability to follow Trey\'s signals. The vacuum solo is a fan favorite.',
    },
    'learning_tips': "Phish songs often have composed 'through-composed' sections with specific parts that must be learned exactly. The 'head' comes first, then study how they develop themes in jams. Trey's tone comes from careful pick attack, compression, and use of the envelope filter. Pay attention to 'signaling' - hand and eye cues that indicate jam direction. Type I jams stay in the song's key/groove, Type II jams leave the song structure entirely.",
    'common_keys': ['A', 'E', 'D', 'G', 'C', 'B', 'F#m'],
    'style': 'Progressive rock with funk, jazz fusion, and psychedelic elements',
    'jam_types': {
        'type_1': 'Jamming within the song structure and key',
        'type_2': 'Leaving the song entirely for exploratory improv',
        'bliss': 'Euphoric major-key build to peak',
        'ambient': 'Spacey, quiet exploration',
        'funk': 'Locked groove with envelope filter',
    }
}


# ============================================================================
# ALLMAN BROTHERS BAND - Complete Coverage
# ============================================================================

ALLMAN_PLAYER_POSITIONS = {
    'guitar': 'Guitars',
    'guitar_left': 'Duane Allman/Warren Haynes/Derek Trucks',  # Changes by era
    'guitar_right': 'Dickey Betts/Jack Pearson',
    'guitar_center': 'Guitars (both)',
    'bass': 'Berry Oakley/Allen Woody/Oteil Burbridge',
    'bass_left': 'Bass',
    'bass_center': 'Bass',
    'drums': 'Drums',
    'drums_left': 'Butch Trucks/Jaimoe',  # Different panning by era
    'drums_right': 'Jaimoe/Butch Trucks',
    'piano': 'Gregg Allman',
    'piano_left': 'Gregg (organ)',
    'piano_right': 'Gregg (piano)',
    'vocals': 'Gregg Allman',
    'other': 'Other',
}

ALLMAN_ERAS = {
    'original': (1969, 1971, 'Original lineup with Duane. Peak blues-rock era.', {
        'duane': 'Gibson Les Paul, SG, ES-335. Fender Showman/Super Reverb. Coricidin bottle slide.',
        'dickey': 'Gibson Les Paul, SG. Marshall and Fender amps.',
        'gregg': 'Hammond B3 with Leslie',
    }),
    'post_duane': (1972, 1973, 'After Duane\'s death. Chuck Leavell on keys.', {
        'dickey': 'Gibson Les Paul, PRS later. Lead guitar duties.',
        'chuck': 'Piano added to Hammond',
    }),
    'post_berry': (1973, 1976, 'After Berry Oakley\'s death. Lamar Williams on bass.', {}),
    'reunion_79': (1979, 1982, 'First reunion. Dan Toler and David Goldflies.', {}),
    'warren_era': (1989, 2000, 'Warren Haynes and Allen Woody era. Blues power.', {
        'warren': 'Gibson Les Paul, various. Marshall amps. Slide master.',
        'dickey': 'PRS guitars',
    }),
    'derek_oteil': (2000, 2014, 'Derek Trucks and Oteil Burbridge. Peak modern era.', {
        'derek': 'Gibson SG, Open E tuning, no picks. Fender Vibro-King.',
        'warren': 'Les Paul, SG. Mesa Boogie, Marshall.',
        'oteil': '6-string bass. Melodic style.',
    }),
}

ALLMAN_ALBUMS = {
    'the allman brothers band': {
        'year': 1969,
        'label': 'Atco/Capricorn',
        'songs': ['don\'t want you no more', 'it\'s not my cross to bear', 'black hearted woman',
                  'trouble no more', 'every hungry woman', 'dreams', 'whipping post'],
        'description': 'Debut album. Whipping Post is 5 minutes here but becomes 23+ live.',
        'era': 'original',
        'personnel': {
            'duane allman': 'Lead guitar, slide',
            'dickey betts': 'Lead guitar',
            'gregg allman': 'Organ, piano, vocals',
            'berry oakley': 'Bass',
            'butch trucks': 'Drums',
            'jaimoe': 'Drums, congas',
        },
    },
    'idlewild south': {
        'year': 1970,
        'label': 'Atco/Capricorn',
        'songs': ['revival', 'don\'t keep me wonderin\'', 'midnight rider', 'in memory of elizabeth reed',
                  'hoochie coochie man', 'please call home', 'leave my blues at home'],
        'description': 'Second album. In Memory of Elizabeth Reed becomes signature instrumental.',
        'era': 'original',
    },
    'at fillmore east': {
        'year': 1971,
        'label': 'Capricorn',
        'songs': ['statesboro blues', 'done somebody wrong', 'stormy monday', 'you don\'t love me',
                  'hot \'lanta', 'in memory of elizabeth reed', 'whipping post'],
        'description': 'THE live album. 23-minute Whipping Post. Defines Southern rock live sound.',
        'era': 'original',
        'personnel': {
            'duane allman': 'Lead/slide guitar (left channel)',
            'dickey betts': 'Lead guitar (right channel)',
            'gregg allman': 'Hammond B3, vocals',
            'berry oakley': 'Bass',
            'butch trucks': 'Drums (left)',
            'jaimoe': 'Drums (right)',
        },
    },
    'eat a peach': {
        'year': 1972,
        'label': 'Capricorn',
        'songs': ['ain\'t wastin\' time no more', 'les brers in a minor', 'melissa',
                  'mountain jam', 'one way out', 'trouble no more', 'stand back',
                  'blue sky', 'little martha'],
        'description': 'Tribute to Duane. Mountain Jam is 33 minutes. Blue Sky is Dickey\'s beautiful instrumental.',
        'era': 'post_duane',
    },
    'brothers and sisters': {
        'year': 1973,
        'label': 'Capricorn',
        'songs': ['wasted words', 'ramblin\' man', 'come and go blues', 'jelly jelly',
                  'southbound', 'jessica', 'pony boy'],
        'description': 'Commercial peak. Jessica is iconic instrumental. Ramblin\' Man was hit single.',
        'era': 'post_berry',
        'personnel': {
            'dickey betts': 'Lead guitar, vocals (Ramblin\' Man)',
            'gregg allman': 'Organ, piano, vocals',
            'chuck leavell': 'Piano',
            'lamar williams': 'Bass',
            'butch trucks': 'Drums',
            'jaimoe': 'Drums',
        },
    },
    'shades of two worlds': {
        'year': 1991,
        'label': 'Epic',
        'songs': ['end of the line', 'bad rain', 'nobody knows', 'desert blues',
                  'get on with your life', 'midnight man', 'kind of bird', 'come on in my kitchen'],
        'description': 'Warren Haynes era begins. Return to form.',
        'era': 'warren_era',
    },
    'where it all begins': {
        'year': 1994,
        'label': 'Epic',
        'songs': ['all night train', 'sailin\' \'cross the devil\'s sea', 'back where it all begins',
                  'soulshine', 'no one to run with', 'change my way of living', 'mean woman blues',
                  'everybody\'s got a mountain to climb', 'what\'s done is done', 'temptation is a gun'],
        'description': 'Strong 90s album. Soulshine becomes Warren\'s signature song.',
        'era': 'warren_era',
    },
    'hittin\' the note': {
        'year': 2003,
        'label': 'Peach/Sanctuary',
        'songs': ['firing line', 'high cost of low living', 'desdemona', 'woman across the river',
                  'old friend', 'who\'s been talking', 'maydell', 'rockin\' horse',
                  'heart of stone', 'instrumental illness'],
        'description': 'Derek Trucks era. Instrumental Illness is showcase for dual-guitar with Derek.',
        'era': 'derek_oteil',
        'personnel': {
            'warren haynes': 'Guitar, vocals',
            'derek trucks': 'Slide guitar',
            'gregg allman': 'Organ, piano, vocals',
            'oteil burbridge': 'Bass',
            'butch trucks': 'Drums',
            'jaimoe': 'Drums',
            'marc quinones': 'Percussion',
        },
    },
}

ALLMAN_KNOWLEDGE = {
    'bio': "The Allman Brothers Band pioneered Southern rock and jam band music. Formed in 1969 in Jacksonville, Florida, known for extended improvisations, twin lead guitar harmonies, and dual drummers. Duane Allman's slide guitar and the interplay with Dickey Betts created a template for jam bands. The band survived tragic losses of Duane and Berry Oakley to continue for decades.",
    'members': {
        'duane allman': 'Lead/slide guitar. Legendary slide player, melodic and soulful. Usually panned LEFT. Coricidin bottle slide. Killed in motorcycle accident 1971. One of the greatest guitarists ever.',
        'dickey betts': 'Lead guitar. Country-influenced, wrote many classics including Jessica and Blue Sky. Usually panned RIGHT. Master of melodic twin harmonies.',
        'gregg allman': 'Hammond B3 organ, piano, vocals. Soulful voice and swampy organ. The heart of the band\'s sound.',
        'berry oakley': 'Bass. Melodic, fluid bass lines influenced by Jack Casady. Died 1972.',
        'warren haynes': 'Guitar, vocals. Blues powerhouse. Joined 1989. Carries Duane\'s legacy with his own style.',
        'derek trucks': 'Slide guitar. Prodigy who joined at 20. Open E tuning, finger-style. Married to Susan Tedeschi.',
        'butch trucks': 'Drums. Usually panned LEFT. Powerful, driving.',
        'jaimoe': 'Drums, percussion. Usually panned RIGHT. Jazz-influenced.',
        'oteil burbridge': 'Bass. Joined 1997. 6-string bass, incredibly melodic. Now with Dead & Company.',
    },
    'learning_tips': "Study the twin guitar harmonies - often in thirds and sixths. Duane's slide work is all about tone and phrasing - learn to phrase like a vocalist. The band's groove is deep and patient - don't rush. Dickey's country-influenced bends and vibrato are key to his sound. The dual drummers lock together but have distinct feels - Butch is more driving, Jaimoe more jazzy.",
    'common_keys': ['A', 'E', 'G', 'B', 'D'],
    'style': 'Southern rock, blues rock, jam band',
}


# ============================================================================
# ADDITIONAL BANDS - Quick Coverage
# ============================================================================

ADDITIONAL_KNOWLEDGE = {
    'widespread panic': {
        'bio': "Widespread Panic is an American rock band formed in Athens, Georgia in 1986. Known for high-energy performances and prolific touring. Unique blend of Southern rock, jam, and hard rock.",
        'members': {
            'john bell': 'Guitar, vocals. Rhythm-focused, crunchy tone.',
            'jimmy herring': 'Lead guitar (2006-present). Jazz fusion virtuoso.',
            'dave schools': 'Bass. Driving, aggressive style.',
            'sunny ortiz': 'Drums. Powerful, rock-solid.',
            'john hermann': 'Keyboards. B3 and piano.',
            'duane trucks': 'Drums (2016-present). Butch Trucks\' son.',
        },
        'style': 'Southern rock jam band',
    },
    'string cheese incident': {
        'bio': "The String Cheese Incident is a jam band formed in 1993 in Crested Butte, Colorado. Known for genre-blending including bluegrass, rock, electronica, and world music.",
        'members': {
            'bill nershi': 'Acoustic guitar, vocals',
            'michael kang': 'Electric mandolin, violin',
            'kyle hollingsworth': 'Keyboards',
            'keith moseley': 'Bass',
            'michael travis': 'Drums',
            'jason hann': 'Percussion',
        },
        'style': 'Jam band with bluegrass, electronica, world music',
    },
    'moe.': {
        'bio': "moe. is a jam band formed in 1989 in Buffalo, New York. Known for accessible rock songs and extended improvisations. All lowercase 'moe.' is correct.",
        'members': {
            'al schnier': 'Guitar, vocals',
            'chuck garvey': 'Guitar, vocals',
            'rob derhak': 'Bass, vocals',
            'jim loughlin': 'Percussion, vibes',
            'vinnie amico': 'Drums',
        },
        'style': 'Progressive rock jam band',
    },
    'goose': {
        'bio': "Goose is an American jam band formed in 2014 in Norwalk, Connecticut. Part of the new wave of jam bands with soaring vocals and patient jam development.",
        'members': {
            'rick mitarotonda': 'Guitar, vocals. Main songwriter.',
            'peter anspach': 'Keyboards, guitar, vocals',
            'trevor weekz': 'Bass',
            'ben atkind': 'Drums',
        },
        'style': 'Modern jam band with indie rock influence',
    },
    'billy strings': {
        'bio': "Billy Strings (William Apostol) is a Grammy-winning bluegrass musician known for virtuosic flatpicking and psychedelic jam explorations within bluegrass.",
        'members': {
            'billy strings': 'Acoustic guitar, vocals. Virtuoso flatpicker.',
            'jarrod walker': 'Mandolin',
            'billy failing': 'Banjo',
            'royal masat': 'Bass',
        },
        'style': 'Progressive bluegrass, psychedelic bluegrass',
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_player_positions(artist_key: str) -> dict:
    """Get player position mapping for an artist."""
    mappings = {
        'phish': PHISH_PLAYER_POSITIONS,
        'allman brothers': ALLMAN_PLAYER_POSITIONS,
        'allman brothers band': ALLMAN_PLAYER_POSITIONS,
    }

    artist_lower = artist_key.lower()
    for key, positions in mappings.items():
        if key in artist_lower or artist_lower in key:
            return positions.copy()

    return {}


def get_era_info(artist_key: str, year: int) -> dict:
    """Get era-specific info for an artist and year."""
    era_dbs = {
        'phish': PHISH_ERAS,
        'allman brothers': ALLMAN_ERAS,
        'allman brothers band': ALLMAN_ERAS,
    }

    artist_lower = artist_key.lower()
    for key, eras in era_dbs.items():
        if key in artist_lower or artist_lower in key:
            for era_name, (start, end, desc, gear) in eras.items():
                if start <= year <= end:
                    return {
                        'era_name': era_name,
                        'description': desc,
                        'gear': gear,
                        'years': f"{start}-{end}"
                    }

    return {}


def get_album_info(artist_key: str, song_name: str) -> dict:
    """Find album info for a song."""
    album_dbs = {
        'phish': PHISH_ALBUMS,
        'allman brothers': ALLMAN_ALBUMS,
        'allman brothers band': ALLMAN_ALBUMS,
    }

    artist_lower = artist_key.lower()
    song_lower = song_name.lower().strip()

    # Clean up song name
    for pattern in [' - live', ' (live)', ' - remastered', '(remastered)']:
        song_lower = song_lower.replace(pattern, '')

    for key, albums in album_dbs.items():
        if key in artist_lower or artist_lower in key:
            for album_name, album_data in albums.items():
                songs = album_data.get('songs', [])
                for album_song in songs:
                    if song_lower in album_song.lower() or album_song.lower() in song_lower:
                        return {
                            'album_name': album_name.title(),
                            'year': album_data.get('year'),
                            'description': album_data.get('description'),
                            'personnel': album_data.get('personnel'),
                            'era': album_data.get('era'),
                            'label': album_data.get('label'),
                        }

    return {}


def get_knowledge(artist_key: str) -> dict:
    """Get local knowledge for an artist."""
    knowledge_dbs = {
        'phish': PHISH_KNOWLEDGE,
        'allman brothers': ALLMAN_KNOWLEDGE,
        'allman brothers band': ALLMAN_KNOWLEDGE,
    }

    # Also check additional bands
    knowledge_dbs.update(ADDITIONAL_KNOWLEDGE)

    artist_lower = artist_key.lower()
    for key, knowledge in knowledge_dbs.items():
        if key in artist_lower or artist_lower in key:
            return knowledge.copy()

    return {}


# ============================================================================
# BANDS THAT SHOULD AUTO-ENABLE STEREO SPLIT
# ============================================================================

STEREO_SPLIT_BANDS = {
    # Dual guitar bands
    'phish',  # Trey often multi-tracked
    'allman brothers',
    'allman brothers band',
    'widespread panic',
    'tedeschi trucks band',
    'derek trucks band',
    'gov\'t mule',

    # Already in track_info.py
    'grateful dead',
    'the beatles',
    'beatles',
    'led zeppelin',
    'pink floyd',
    'eagles',
    'steely dan',
    'fleetwood mac',
    'television',
    'the who',
    'cream',
}


def should_stereo_split(artist: str) -> bool:
    """Check if stereo splitting is recommended for this artist."""
    if not artist:
        return False
    artist_lower = artist.lower().strip()
    for band in STEREO_SPLIT_BANDS:
        if band in artist_lower or artist_lower in band:
            return True
    return False
