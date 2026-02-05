# StemScribe - Audio Stem Separation & Music Learning Platform

## Overview
StemScribe is a web-based tool that separates audio tracks into individual stems (vocals, guitar, bass, drums, keys, other) and transcribes them to MIDI/tablature for music learning. It's designed with a focus on jam bands like the Grateful Dead, with special features for dual-guitar and dual-drummer bands.

## Tech Stack
- **Backend**: Python/Flask API (`~/stemscribe/backend/app.py`)
- **Frontend**: Single-page HTML/CSS/JS (`~/stemscribe/frontend/index.html`)
- **Audio Separation**: Demucs (htdemucs_6s model - 6 stems)
- **MIDI Transcription**: Basic Pitch + custom drum transcriber
- **Notation**: MusicXML generation for tablature display

## File Structure
```
~/stemscribe/
├── backend/
│   ├── app.py              # Main Flask API server
│   ├── stereo_splitter.py  # Splits panned instruments (Jerry/Bob guitars)
│   ├── track_info.py       # Artist info, album lookup, player mapping
│   ├── drum_transcriber.py # Enhanced drum-to-MIDI transcription
│   ├── enhancer.py         # Audio enhancement (pedalboard/noisereduce)
│   ├── drive_service.py    # Google Drive integration
│   ├── skills.py           # Skill system for cascaded separation
│   └── venv/               # Python virtual environment
├── frontend/
│   └── index.html          # Full UI (Neve console mixer style)
├── uploads/                # Temporary upload storage
└── outputs/                # Processed stems, MIDI, MusicXML
```

## Key Features

### Core Functionality
- **6-Stem Separation**: Vocals, Guitar, Bass, Drums, Piano/Keys, Other
- **MIDI Transcription**: All stems transcribed to MIDI files
- **Tablature Generation**: MusicXML for notation display in browser
- **URL Support**: YouTube, Spotify, SoundCloud, Bandcamp via yt-dlp
- **Google Drive Upload**: Auto-uploads processed files

### Stereo Splitting (NEW)
- Separates panned instruments into L/R components
- **Grateful Dead**: Jerry Garcia (right) / Bob Weir (left) on separate tracks
- **Dual Drummers**: Bill Kreutzmann (left) / Mickey Hart (right)
- Auto-enables for known dual-guitar bands (GD, Allman Brothers, Beatles, etc.)
- Uses librosa for frequency-aware panning detection

### Track Info System (NEW)
- **Auto-loads player names** on stems (Phil Lesh, Billy & Mickey, etc.)
- **Local knowledge base**: Grateful Dead, Phish, Allman Brothers, Led Zeppelin, Beatles, Pink Floyd
- **MusicBrainz integration**: Album lookup for any artist
- **Wikipedia API**: Artist bios and song info
- **Era detection**: Identifies Grateful Dead keyboardist by album year (Pigpen/Keith/Brent/Vince)

### UI Features
- **Neve Console Mixer**: Realistic mixing board with VU meters and faders
- **Transport Controls**: Play/pause with time scrubbing
- **Mute/Solo**: Per-stem mute and solo buttons
- **Tour Bus Animation**: Progress indicator during processing
- **Settings Panel**: HQ Vocals, Vocal Focus, Enhance Stems, Stereo Split toggles

## Player Position Mapping (Grateful Dead)
```python
'guitar_left': 'Bob Weir'
'guitar_right': 'Jerry Garcia'
'bass': 'Phil Lesh'
'drums_left': 'Bill Kreutzmann'
'drums_right': 'Mickey Hart'
'piano': 'Keys' (era-specific: Pigpen/Keith/Brent/Vince)
```

## Running the Server
```bash
cd ~/stemscribe/backend
source venv/bin/activate
python3 app.py
```
Server runs on http://localhost:5000

## Dependencies (in venv)
- flask, flask-cors
- yt-dlp (YouTube downloads)
- demucs (stem separation)
- basic-pitch (MIDI transcription)
- librosa, soundfile (stereo splitting)
- music21 (MusicXML)
- google-api-python-client (Drive)
- numpy, scipy

## API Endpoints
- `POST /api/upload` - Upload audio file
- `POST /api/url` - Process from URL
- `GET /api/status/<job_id>` - Job status
- `GET /api/info/<job_id>` - Track info (artist, album, player mapping)
- `GET /api/download/<job_id>/stem/<name>` - Download stem
- `GET /api/download/<job_id>/midi/<name>` - Download MIDI

## Recent Additions (Feb 2026)
1. Stereo splitter for dual-guitar/drum bands
2. Track info with player name auto-labeling
3. MusicBrainz album lookup
4. Auto-enable stereo split for known bands
5. Era-specific Grateful Dead keyboardist detection

## Known Bands with Special Handling
- Grateful Dead (dual guitars, dual drums)
- Allman Brothers Band (dual guitars, dual drums)
- The Beatles (dual guitars)
- Led Zeppelin (multi-tracked guitar)
- Pink Floyd (multi-tracked guitar)
- Phish, Eagles, Steely Dan, Fleetwood Mac, Television, The Who, Cream
