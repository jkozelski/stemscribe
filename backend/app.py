"""
StemScribe - Audio Stem Separation & Transcription API
A slick backend for separating audio into stems and transcribing to MIDI

Features:
- 6-stem separation (vocals, drums, bass, guitar, piano, other) via Demucs htdemucs_6s
- MIDI transcription via Basic Pitch
- YouTube/URL download support via yt-dlp
"""

import os
import sys
import json
import uuid
import shutil
import logging
import subprocess
import re
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import threading
import time
import numpy as np

# Add homebrew to path
os.environ['PATH'] = os.environ.get('PATH', '') + ':/opt/homebrew/bin:/usr/local/bin'

app = Flask(__name__)
CORS(app)

# Frontend directory (relative to backend)
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend'

# Disable caching for development - makes Cmd+R actually refresh
@app.after_request
def add_no_cache_headers(response):
    """Add headers to disable caching during development"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Serve frontend files
@app.route('/')
def serve_index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/practice.html')
def serve_practice():
    return send_from_directory(FRONTEND_DIR, 'practice.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve any static files from frontend directory"""
    if (FRONTEND_DIR / filename).exists():
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({'error': 'Not found'}), 404

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


# Google Drive integration (import after logger is configured)
try:
    from drive_service import upload_job_to_drive, cleanup_old_stems, get_drive_stats, get_drive_service
    DRIVE_AVAILABLE = True
    logger.info("Google Drive integration available")
except ImportError as e:
    DRIVE_AVAILABLE = False
    logger.warning(f"Google Drive integration not available: {e}")

# Skills system for enhanced stem processing
try:
    from skills import get_skill, get_all_skills, apply_skill, SKILL_REGISTRY, analyze_stems_for_skills
    SKILLS_AVAILABLE = True
    logger.info(f"Skills system available with {len(SKILL_REGISTRY)} skills")
except ImportError as e:
    SKILLS_AVAILABLE = False
    logger.warning(f"Skills system not available: {e}")

# Stem enhancement for natural sound
try:
    from enhancer import enhance_stem, enhance_all_stems, PEDALBOARD_AVAILABLE, NOISEREDUCE_AVAILABLE
    ENHANCER_AVAILABLE = PEDALBOARD_AVAILABLE
    logger.info(f"Stem enhancer available (pedalboard={PEDALBOARD_AVAILABLE}, noisereduce={NOISEREDUCE_AVAILABLE})")
except ImportError as e:
    ENHANCER_AVAILABLE = False
    logger.warning(f"Stem enhancer not available: {e}")

# Drum transcription (separate from Basic Pitch)
try:
    from drum_transcriber import transcribe_drums_to_midi
    DRUM_TRANSCRIBER_AVAILABLE = True
    logger.info("Drum transcriber available")
except ImportError as e:
    DRUM_TRANSCRIBER_AVAILABLE = False
    logger.warning(f"Drum transcriber not available: {e}")

# Stereo splitting for panned instruments
try:
    from stereo_splitter import split_stereo, split_all_stems_by_panning, check_if_splittable
    STEREO_SPLITTER_AVAILABLE = True
    logger.info("Stereo splitter available")
except ImportError as e:
    STEREO_SPLITTER_AVAILABLE = False
    logger.warning(f"Stereo splitter not available: {e}")

# Enhanced separator with audio-separator (BS-Roformer, UVR models)
try:
    from enhanced_separator import (
        EnhancedSeparator, AUDIO_SEPARATOR_AVAILABLE,
        separate_audio, separate_full_song, split_vocals, MODELS as SEPARATOR_MODELS
    )
    ENHANCED_SEPARATOR_AVAILABLE = AUDIO_SEPARATOR_AVAILABLE
    if ENHANCED_SEPARATOR_AVAILABLE:
        logger.info(f"✅ Enhanced separator available ({len(SEPARATOR_MODELS)} models)")
except ImportError as e:
    ENHANCED_SEPARATOR_AVAILABLE = False
    logger.warning(f"Enhanced separator not available: {e}")

# Track info fetcher for context and learning tips
try:
    from track_info import fetch_track_info, extract_artist_from_title, get_instrument_tips, LOCAL_KNOWLEDGE, should_stereo_split
    TRACK_INFO_AVAILABLE = True
    logger.info(f"Track info available (local knowledge: {len(LOCAL_KNOWLEDGE)} artists)")
except ImportError as e:
    TRACK_INFO_AVAILABLE = False
    logger.warning(f"Track info not available: {e}")

# ============================================================================
# NEW ENHANCED MODULES (Feb 2026)
# ============================================================================

# Enhanced transcriber with articulation detection (bends, slides, hammer-ons)
try:
    from transcriber_enhanced import EnhancedTranscriber, transcribe_with_enhanced
    ENHANCED_TRANSCRIBER_AVAILABLE = True
    logger.info("Enhanced transcriber available (articulations, polyphony)")
except ImportError as e:
    ENHANCED_TRANSCRIBER_AVAILABLE = False
    logger.warning(f"Enhanced transcriber not available: {e}")

# Improved drum transcription v2 (ghost notes, cymbal differentiation)
try:
    from drum_transcriber_v2 import EnhancedDrumTranscriber, transcribe_drums_to_midi as transcribe_drums_v2
    DRUM_TRANSCRIBER_V2_AVAILABLE = True
    logger.info("Drum transcriber v2 available (ghost notes, hi-hat states)")
except ImportError as e:
    DRUM_TRANSCRIBER_V2_AVAILABLE = False
    logger.warning(f"Drum transcriber v2 not available: {e}")

# OaF-based drum transcription (neural network)
try:
    from oaf_drum_transcriber import OaFDrumTranscriber, transcribe_drums, OAF_AVAILABLE
    OAF_DRUM_TRANSCRIBER_AVAILABLE = True
    logger.info(f"OaF Drum transcriber available (neural: {OAF_AVAILABLE})")
except ImportError as e:
    OAF_DRUM_TRANSCRIBER_AVAILABLE = False
    OAF_AVAILABLE = False
    logger.warning(f"OaF Drum transcriber not available: {e}")

# Model manager for pretrained models
try:
    from models.model_manager import ModelManager, ensure_model, list_available_models
    MODEL_MANAGER_AVAILABLE = True
    logger.info("Model manager available")
except ImportError as e:
    MODEL_MANAGER_AVAILABLE = False
    logger.warning(f"Model manager not available: {e}")

# Chord detection (V8 Transformer model preferred, falls back to V7 then basic template matching)
CHORD_DETECTOR_VERSION = None
try:
    from chord_detector_v8 import ChordDetector, detect_chords
    CHORD_DETECTOR_AVAILABLE = True
    CHORD_DETECTOR_VERSION = 'v8'
    logger.info("✅ Chord detector V8 available (337 classes, inversions, mMaj7)")
except ImportError:
    try:
        from chord_detector_v7 import ChordDetector, detect_chords
        CHORD_DETECTOR_AVAILABLE = True
        CHORD_DETECTOR_VERSION = 'v7'
        logger.info("Chord detector V7 available (25 classes)")
    except ImportError:
        try:
            from chord_detector import ChordDetector, detect_chords
            CHORD_DETECTOR_AVAILABLE = True
            CHORD_DETECTOR_VERSION = 'basic'
            logger.info("Basic chord detector available (template matching)")
        except ImportError as e:
            CHORD_DETECTOR_AVAILABLE = False
            CHORD_DETECTOR_VERSION = None
            logger.warning(f"Chord detector not available: {e}")

# Chord theory engine (scale suggestions, Beato-style chord-over-bass analysis)
try:
    from chord_theory import ChordTheoryEngine, get_scale_suggestion, get_progression_analysis
    CHORD_THEORY_AVAILABLE = True
    _chord_theory_engine = ChordTheoryEngine()
    logger.info("✅ Chord theory engine available (scale suggestions, polychord analysis)")
except ImportError as e:
    CHORD_THEORY_AVAILABLE = False
    _chord_theory_engine = None
    logger.warning(f"Chord theory engine not available: {e}")

# Monophonic melody/lead transcriber (clean lead lines, articulations)
try:
    from melody_transcriber import MelodyExtractor, transcribe_melody
    MELODY_TRANSCRIBER_AVAILABLE = True
    logger.info("✅ Melody transcriber available (monophonic lead extraction, vibrato, bends)")
except ImportError as e:
    MELODY_TRANSCRIBER_AVAILABLE = False
    logger.warning(f"Melody transcriber not available: {e}")

# Internet Archive Live Music pipeline (search/browse/batch)
try:
    from archive_pipeline import ArchivePipeline, search_archive, get_show_info, get_pipeline as get_archive_pipeline
    ARCHIVE_PIPELINE_AVAILABLE = True
    logger.info("✅ Archive.org Live Music pipeline available (250k+ free concert recordings)")
except ImportError as e:
    ARCHIVE_PIPELINE_AVAILABLE = False
    logger.warning(f"Archive.org pipeline not available: {e}")

# MIDI to Guitar Pro conversion
try:
    from midi_to_gp import convert_midi_to_gp, convert_job_midis_to_gp
    GP_CONVERTER_AVAILABLE = True
    logger.info("Guitar Pro converter available")
except ImportError as e:
    GP_CONVERTER_AVAILABLE = False
    logger.warning(f"Guitar Pro converter not available: {e}")

# MIDI to MusicXML notation conversion (articulations, melody mode, dynamics)
try:
    from midi_to_notation import midi_to_musicxml as _notation_convert, MUSIC21_AVAILABLE
    NOTATION_CONVERTER_AVAILABLE = True
    logger.info("Notation converter available (articulations, dynamics, triplet quantization)")
except ImportError as e:
    NOTATION_CONVERTER_AVAILABLE = False
    MUSIC21_AVAILABLE = False
    logger.warning(f"Notation converter not available: {e}")

# Ensemble separation system (Moises.ai-quality multi-model approach)
try:
    from separation import EnsembleSeparator, GPUManager
    _gpu_manager = GPUManager()
    # Import the new robust DemucsRunner (fixes progress-as-error bug)
    from demucs_runner import DemucsRunner, DemucsProgress
    ENSEMBLE_SEPARATOR_AVAILABLE = True
    logger.info(f"Ensemble separator available ({_gpu_manager.device_info.device_name}, {_gpu_manager.device_info.total_memory_gb:.1f}GB)")
except Exception as e:
    ENSEMBLE_SEPARATOR_AVAILABLE = False
    _gpu_manager = None
    logger.warning(f"Ensemble separator not available: {e}")

# Extended band database (Phish, Allman Brothers, 70+ bands)
try:
    from band_config import (
        should_stereo_split as extended_should_stereo_split,
        get_player_positions as get_extended_positions,
        has_dual_drummers,
        get_learning_tips as get_extended_tips,
        STEREO_SPLIT_BANDS
    )
    EXTENDED_BAND_CONFIG_AVAILABLE = True
    logger.info(f"Extended band config available ({len(STEREO_SPLIT_BANDS)} bands)")
except ImportError as e:
    EXTENDED_BAND_CONFIG_AVAILABLE = False
    logger.warning(f"Extended band config not available: {e}")

# Directories - relative to script location
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
UPLOAD_DIR = SCRIPT_DIR / 'uploads'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Job tracking
jobs = {}

# ============ JOB PERSISTENCE ============

def save_job_to_disk(job):
    """Save job metadata to disk for persistence across server restarts"""
    try:
        job_file = OUTPUT_DIR / job.job_id / 'job_metadata.json'
        job_data = job.to_dict()
        job_data['saved_at'] = time.time()

        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2, default=str)

        logger.info(f"💾 Saved job metadata: {job.job_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save job metadata: {e}")
        return False


def load_job_from_disk(job_dir: Path):
    """Load a job from disk"""
    try:
        job_file = job_dir / 'job_metadata.json'
        if not job_file.exists():
            return None

        with open(job_file, 'r') as f:
            data = json.load(f)

        # Recreate the job object
        job = ProcessingJob(
            job_id=data['job_id'],
            filename=data.get('filename', 'Unknown'),
            source_url=data.get('source_url')
        )

        # Restore all attributes
        job.status = data.get('status', 'completed')
        job.progress = data.get('progress', 100)
        job.stage = data.get('stage', 'Complete')
        job.stems = data.get('stems', {})
        job.enhanced_stems = data.get('enhanced_stems', {})
        job.midi_files = data.get('midi_files', {})
        job.musicxml_files = data.get('musicxml_files', {})
        job.gp_files = data.get('gp_files', {})
        job.sub_stems = data.get('sub_stems', {})
        job.selected_skills = data.get('selected_skills', [])
        job.error = data.get('error')
        job.created_at = data.get('created_at', time.time())
        job.metadata = data.get('metadata', {})
        job.chord_progression = data.get('chord_progression', [])
        job.detected_key = data.get('detected_key')
        job.transcription_quality = data.get('transcription_quality', {})
        job.pro_tabs = data.get('pro_tabs', {})

        # Verify that stem files still exist
        valid_stems = {}
        for stem_name, stem_path in job.stems.items():
            if Path(stem_path).exists():
                valid_stems[stem_name] = stem_path
        job.stems = valid_stems

        # Auto-discover GP files that may have been generated later
        gp_dir = job_dir / 'guitarpro'
        logger.info(f"🎸 Checking for GP files in: {gp_dir}")
        if gp_dir.exists():
            gp_files_found = list(gp_dir.glob('*.gp5'))
            logger.info(f"🎸 Found {len(gp_files_found)} GP files: {[f.name for f in gp_files_found]}")
            for gp_file in gp_files_found:
                stem_name = gp_file.stem  # e.g., guitar_basic_pitch
                if stem_name not in job.gp_files:
                    job.gp_files[stem_name] = str(gp_file)
                    logger.info(f"🎸 Auto-discovered GP file: {stem_name}")
        else:
            logger.info(f"🎸 No guitarpro directory at {gp_dir}")

        return job
    except Exception as e:
        logger.warning(f"Failed to load job from {job_dir}: {e}")
        return None


def load_all_jobs_from_disk():
    """Load all saved jobs from disk on startup"""
    loaded_count = 0

    if not OUTPUT_DIR.exists():
        return loaded_count

    for job_dir in OUTPUT_DIR.iterdir():
        if job_dir.is_dir():
            job = load_job_from_disk(job_dir)
            if job and job.stems:  # Only load jobs with valid stems
                jobs[job.job_id] = job
                loaded_count += 1

    if loaded_count > 0:
        logger.info(f"📚 Loaded {loaded_count} jobs from library")

    return loaded_count


def get_job(job_id: str):
    """Get a job from memory, or auto-load from disk if not found"""
    job = None

    # First check memory
    if job_id in jobs:
        job = jobs[job_id]
    else:
        # Try to load from disk
        job_dir = OUTPUT_DIR / job_id
        if job_dir.exists():
            job = load_job_from_disk(job_dir)
            if job:
                jobs[job_id] = job  # Cache it in memory
                logger.info(f"📂 Auto-loaded job {job_id} from disk")

    # Always check for new GP files (they may have been generated after job was loaded)
    if job:
        job_dir = OUTPUT_DIR / job_id
        gp_dir = job_dir / 'guitarpro'
        if gp_dir.exists():
            for gp_file in gp_dir.glob('*.gp5'):
                stem_name = gp_file.stem
                if stem_name not in job.gp_files:
                    job.gp_files[stem_name] = str(gp_file)
                    logger.info(f"🎸 Discovered new GP file: {stem_name}")

    return job


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
                title = re.sub(r'\s*[-–]\s*Apple Music\s*$', '', title)
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
        # Use yt-dlp to search YouTube
        search_cmd = [
            'yt-dlp',
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


class ProcessingJob:
    def __init__(self, job_id, filename, source_url=None, skills=None):
        self.job_id = job_id
        self.filename = filename
        self.source_url = source_url
        self.status = 'pending'
        self.progress = 0
        self.stage = 'Initializing'
        self.stems = {}
        self.enhanced_stems = {}  # Enhanced versions of stems
        self.midi_files = {}
        self.musicxml_files = {}
        self.gp_files = {}       # Guitar Pro files for tablature notation
        self.sub_stems = {}  # Skills-generated sub-stems: {skill_id: {sub_stem_name: path}}
        self.selected_skills = skills or []  # Skills to apply
        self.error = None
        self.created_at = time.time()
        self.metadata = {}  # Store title, artist, duration, etc.
        # NEW: Enhanced transcription fields
        self.chord_progression = []  # Detected chords with timestamps
        self.detected_key = None     # Detected musical key
        self.articulations = {}      # Per-stem articulation counts
        self.transcription_quality = {}  # Quality scores per stem
        self.pro_tabs = {}           # Professional tabs from Songsterr/UG
        self.transcription_mode = {}  # Per-stem: 'melody', 'enhanced', 'basic_pitch'

    def to_dict(self):
        """Convert job to dict with numpy types converted for JSON serialization"""
        data = {
            'job_id': self.job_id,
            'filename': self.filename,
            'source_url': self.source_url,
            'status': self.status,
            'progress': self.progress,
            'stage': self.stage,
            'stems': self.stems,
            'enhanced_stems': self.enhanced_stems,
            'midi_files': self.midi_files,
            'musicxml_files': self.musicxml_files,
            'gp_files': self.gp_files,
            'sub_stems': self.sub_stems,
            'selected_skills': self.selected_skills,
            'error': self.error,
            'metadata': self.metadata,
            # NEW: Enhanced transcription data
            'chord_progression': self.chord_progression,
            'detected_key': self.detected_key,
            'transcription_quality': self.transcription_quality,
            'pro_tabs': self.pro_tabs,  # Professional tabs from Songsterr
            'transcription_mode': self.transcription_mode,
        }
        # Convert numpy types to native Python types for JSON serialization
        return convert_numpy_types(data)


def download_from_url(job: ProcessingJob, url: str, output_dir: Path):
    """Download audio from YouTube or other supported platforms using yt-dlp"""
    job.stage = 'Downloading audio from URL'
    job.progress = 5
    logger.info(f"Downloading from URL: {url}")

    try:
        # First, get metadata
        metadata_cmd = [
            'yt-dlp',
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
        download_cmd = [
            'yt-dlp',
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


def check_stem_has_content(stem_path: str, threshold: float = 0.01) -> bool:
    """Check if a stem has significant audio content worth processing further"""
    try:
        from scipy.io import wavfile
        import numpy as np

        # Handle MP3 files by checking file size as proxy
        if stem_path.endswith('.mp3'):
            file_size = os.path.getsize(stem_path)
            # If MP3 is less than 50KB, probably mostly silence
            return file_size > 50000

        sr, audio = wavfile.read(stem_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        # Calculate RMS energy
        if len(audio.shape) == 2:
            audio = np.mean(audio, axis=1)
        rms = np.sqrt(np.mean(audio ** 2))

        return rms > threshold
    except Exception as e:
        logger.warning(f"Could not check stem content: {e}")
        return True  # Assume it has content if we can't check


def convert_wavs_to_mp3(directory: Path) -> dict:
    """
    Convert all WAV files in a directory to MP3 using ffmpeg.
    This avoids the lameenc dependency which doesn't support Python 3.14.
    Returns dict of {stem_name: mp3_path}
    """
    converted = {}
    for wav_file in directory.glob('*.wav'):
        mp3_file = wav_file.with_suffix('.mp3')
        try:
            result = subprocess.run([
                'ffmpeg', '-y', '-i', str(wav_file),
                '-codec:a', 'libmp3lame', '-b:a', '320k',
                str(mp3_file)
            ], capture_output=True, text=True)

            if mp3_file.exists():
                wav_file.unlink()  # Remove WAV to save space
                converted[wav_file.stem] = str(mp3_file)
                logger.info(f"Converted {wav_file.stem} to MP3")
            else:
                # Keep WAV if conversion failed
                converted[wav_file.stem] = str(wav_file)
                logger.warning(f"MP3 conversion failed for {wav_file.stem}, keeping WAV")
        except Exception as e:
            converted[wav_file.stem] = str(wav_file)
            logger.warning(f"MP3 conversion error for {wav_file.stem}: {e}")

    return converted


def separate_stems(job: ProcessingJob, audio_path: Path):
    """
    FIXED: Use Demucs to separate audio into stems with proper progress handling.

    This version uses DemucsRunner which properly parses stderr progress output
    instead of treating progress bars as errors.

    Key fixes:
    1. Uses DemucsRunner which properly parses stderr progress output
    2. Streams progress updates to job.progress in real-time
    3. Only reports actual errors, not progress bar output
    4. Handles long songs (7-19 min) without timeout issues
    """
    job.stage = 'Separating stems with AI (6-stem model)'
    job.progress = 15
    logger.info(f"Starting stem separation for {audio_path}")

    output_path = OUTPUT_DIR / job.job_id / 'stems'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Progress callback to update job status
        def progress_callback(progress: DemucsProgress):
            # Map Demucs progress (0-100%) to job progress (15-40%)
            job_progress = 15 + (progress.percent * 0.25)
            job.progress = int(job_progress)

            if progress.eta_seconds > 0:
                eta_min = progress.eta_seconds / 60
                job.stage = f'Separating stems: {progress.percent:.0f}% (ETA: {eta_min:.1f}m)'
            else:
                job.stage = f'Separating stems: {progress.percent:.0f}%'

        # Use the new DemucsRunner (fixes progress-as-error bug)
        runner = DemucsRunner(
            model='htdemucs_6s',
            progress_callback=progress_callback
        )

        # Run separation with 30-minute timeout (enough for 20+ min songs)
        result = runner.separate(
            audio_path=audio_path,
            output_dir=output_path,
            timeout_seconds=1800
        )

        if not result.success:
            raise Exception(result.error_message)

        job.progress = 40

        # Convert WAV to MP3 using ffmpeg
        if result.output_dir and result.output_dir.exists():
            job.stage = 'Converting stems to MP3'
            job.stems = convert_wavs_to_mp3(result.output_dir)
            for stem_name in job.stems:
                logger.info(f"Found stem: {stem_name}")

        if not job.stems:
            raise Exception("No stems were generated")

        logger.info(f"✅ Stem separation complete: {len(job.stems)} stems in {result.processing_time_seconds:.1f}s")
        return True

    except Exception as e:
        logger.error(f"Stem separation failed: {e}")
        job.error = str(e)
        return False


def separate_stems_mdx(job: ProcessingJob, audio_path: Path, stereo_split_guitar: bool = False):
    """
    HYBRID stem separation - uses the BEST model for each instrument type.

    OPTIMIZED VERSION: Single separator instance, proper memory cleanup.

    Strategy:
    1. MDX23C-InstVoc: Clean vocals/instrumental split (best for vocals)
    2. htdemucs_6s: ALL instruments from instrumental (drums, bass, guitar, piano, other)
    3. Optional: Stereo split guitar for dual-guitar bands (Allman Brothers, etc.)

    Note: We use htdemucs piano as-is since running multiple models was causing
    overheating. The piano quality is acceptable for most use cases.
    """
    try:
        from audio_separator.separator import Separator
    except ImportError:
        logger.error("audio-separator not installed. Install with: pip install audio-separator")
        return False

    import gc

    job.stage = 'HYBRID: Starting multi-model separation'
    job.progress = 10
    logger.info(f"🎹 Starting HYBRID stem separation for {audio_path}")
    logger.info(f"   Strategy: MDX23C (vocals) → htdemucs_6s (all instruments)")

    output_path = OUTPUT_DIR / job.job_id / 'stems' / 'hybrid'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Use a single separator instance to reduce memory usage
        separator = Separator(
            output_dir=str(output_path),
            output_format='mp3',
        )

        # ================================================================
        # STEP 1: MDX23C - Clean vocals/instrumental split
        # ================================================================
        job.stage = 'HYBRID Step 1/2: Extracting clean vocals'
        job.progress = 15

        separator.load_model(model_filename='MDX23C-8KFFT-InstVoc_HQ.ckpt')
        output_files = separator.separate(str(audio_path))
        logger.info(f"Step 1 - MDX23C output: {output_files}")

        # Find vocals and instrumental
        # Note: output_files may be just filenames or full paths depending on audio-separator version
        instrumental_path = None
        for f in output_files:
            # Ensure we have full path
            if not Path(f).is_absolute():
                f = str(output_path / f)

            if 'instrumental' in f.lower() or 'instrum' in f.lower():
                instrumental_path = f
            elif 'vocal' in f.lower():
                job.stems['vocals'] = f
                logger.info(f"  🎤 VOCALS: {Path(f).name}")

        if not instrumental_path:
            raise Exception("MDX23C did not produce instrumental track")

        # Clean up memory before loading next model
        gc.collect()

        # ================================================================
        # STEP 2: htdemucs_6s - Get ALL instruments from instrumental
        # ================================================================
        job.stage = 'HYBRID Step 2/2: Separating instruments'
        job.progress = 40

        separator.load_model(model_filename='htdemucs_6s.yaml')
        inst_outputs = separator.separate(instrumental_path)
        logger.info(f"Step 2 - htdemucs_6s output: {inst_outputs}")

        for f in inst_outputs:
            # Ensure we have full path
            if not Path(f).is_absolute():
                f = str(output_path / f)

            fname = Path(f).stem.lower()
            if 'drum' in fname:
                job.stems['drums'] = f
                logger.info(f"  🥁 DRUMS: {Path(f).name}")
            elif 'bass' in fname:
                job.stems['bass'] = f
                logger.info(f"  🎸 BASS: {Path(f).name}")
            elif 'guitar' in fname:
                job.stems['guitar'] = f
                logger.info(f"  🎸 GUITAR: {Path(f).name}")
            elif 'piano' in fname:
                job.stems['piano'] = f
                logger.info(f"  🎹 PIANO: {Path(f).name}")
            elif 'other' in fname:
                job.stems['other'] = f
                logger.info(f"  🎵 OTHER: {Path(f).name}")

        # Clean up separator to free memory
        del separator
        gc.collect()

        # ================================================================
        # OPTIONAL: Stereo split guitar for dual-guitar bands
        # ================================================================
        if stereo_split_guitar and 'guitar' in job.stems and STEREO_SPLITTER_AVAILABLE:
            job.stage = 'Stereo splitting guitar'
            job.progress = 70
            logger.info("Stereo splitting guitar for dual-guitar separation...")

            try:
                guitar_path = Path(job.stems['guitar'])
                split_output = OUTPUT_DIR / job.job_id / 'stems' / 'guitar_split'
                split_output.mkdir(parents=True, exist_ok=True)

                left, right, center = split_stereo(guitar_path, split_output)

                if left and right:
                    job.stems['guitar_left'] = str(left)
                    job.stems['guitar_right'] = str(right)
                    if center:
                        job.stems['guitar_center'] = str(center)
                    logger.info(f"  🎸 GUITAR LEFT: {Path(left).name}")
                    logger.info(f"  🎸 GUITAR RIGHT: {Path(right).name}")
            except Exception as e:
                logger.warning(f"Guitar stereo split failed: {e}")

        job.progress = 80

        if len(job.stems) < 3:
            raise Exception("Not enough stems were generated")

        logger.info(f"✅ HYBRID separation complete!")
        logger.info(f"   Final stems: {list(job.stems.keys())}")
        return True

    except Exception as e:
        logger.error(f"HYBRID separation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        job.error = str(e)
        return False


def separate_stems_hq_vocals(job: ProcessingJob, audio_path: Path):
    """
    High Quality Vocals Mode - Two-pass separation for cleaner vocals.

    Pass 1: Use htdemucs (2-stem) to get pristine vocals + accompaniment
    Pass 2: Use htdemucs_6s on accompaniment to get drums, bass, guitar, piano, other

    This approach works better on complex, densely-produced tracks where vocals
    get muddied by the 6-stem model (Steely Dan, Toto, etc.)
    """
    job.stage = 'HQ Vocals: Pass 1 - Isolating vocals'
    job.progress = 10
    logger.info(f"🎤 Starting HQ Vocals separation for {audio_path}")

    output_path = OUTPUT_DIR / job.job_id / 'stems'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # PASS 1: 2-stem separation for clean vocals
        # htdemucs default is vocals + no_vocals (accompaniment)
        logger.info("Pass 1: Running 2-stem model for clean vocals...")
        result = subprocess.run([
            'python3', '-m', 'demucs',
            '--out', str(output_path),
            '-n', 'htdemucs',
            '--two-stems', 'vocals',
            str(audio_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Demucs pass 1 failed: {result.stderr[:500]}")
        job.progress = 25

        # Convert WAV to MP3
        pass1_dir = output_path / 'htdemucs' / audio_path.stem
        convert_wavs_to_mp3(pass1_dir)
        job.progress = 30

        # Get the vocals and accompaniment from pass 1
        vocals_path = pass1_dir / 'vocals.mp3'
        accompaniment_path = pass1_dir / 'no_vocals.mp3'

        if vocals_path.exists():
            job.stems['vocals'] = str(vocals_path)
            logger.info("✅ Pass 1 complete - clean vocals extracted")
        else:
            raise Exception("Pass 1 failed - no vocals generated")

        # PASS 2: 6-stem separation on the accompaniment
        job.stage = 'HQ Vocals: Pass 2 - Separating instruments'
        job.progress = 35
        logger.info("Pass 2: Running 6-stem model on accompaniment...")

        pass2_output = output_path / 'pass2'
        pass2_output.mkdir(parents=True, exist_ok=True)

        result = subprocess.run([
            'python3', '-m', 'demucs',
            '--out', str(pass2_output),
            '-n', 'htdemucs_6s',
            str(accompaniment_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Demucs pass 2 failed: {result.stderr[:500]}")
        job.progress = 45

        # Convert WAV to MP3 and collect instrument stems from pass 2
        pass2_dir = pass2_output / 'htdemucs_6s' / 'no_vocals'
        if pass2_dir.exists():
            convert_wavs_to_mp3(pass2_dir)
            job.progress = 50
            for stem_file in pass2_dir.glob('*.mp3'):
                stem_name = stem_file.stem
                # Skip the vocals from pass 2 (we have better ones from pass 1)
                if stem_name != 'vocals':
                    job.stems[stem_name] = str(stem_file)
                    logger.info(f"Found instrument stem: {stem_name}")

        if len(job.stems) < 2:
            raise Exception("Pass 2 failed - insufficient stems generated")

        logger.info(f"✅ HQ Vocals separation complete: {len(job.stems)} stems")
        return True

    except Exception as e:
        logger.error(f"HQ Vocals separation failed: {e}")
        job.error = str(e)
        return False


def separate_stems_vocal_focus(job: ProcessingJob, audio_path: Path):
    """
    Vocal Focus Mode - Maximum vocal isolation using fine-tuned model.

    Uses htdemucs_ft (fine-tuned) with aggressive settings to pull vocals
    out of even the densest mixes. Prioritizes vocal clarity over instrument
    stem quality.

    Best for: Steely Dan, Toto, 80s production, dense chorus sections
    """
    job.stage = 'Vocal Focus: Aggressive vocal extraction'
    job.progress = 10
    logger.info(f"🎯 Starting Vocal Focus separation for {audio_path}")

    output_path = OUTPUT_DIR / job.job_id / 'stems'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Use htdemucs_ft (fine-tuned) - often better on difficult vocals
        # with --two-stems vocals for maximum vocal focus
        logger.info("Running fine-tuned model with vocal focus...")
        result = subprocess.run([
            'python3', '-m', 'demucs',
            '--out', str(output_path),
            '-n', 'htdemucs_ft',
            '--two-stems', 'vocals',
            '--shifts', '2',
            str(audio_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Demucs vocal focus failed: {result.stderr[:500]}")
        job.progress = 45

        # Convert WAV to MP3
        ft_dir = output_path / 'htdemucs_ft' / audio_path.stem
        convert_wavs_to_mp3(ft_dir)
        job.progress = 50

        # Get the vocal and instrumental from fine-tuned pass
        vocals_path = ft_dir / 'vocals.mp3'
        instrumental_path = ft_dir / 'no_vocals.mp3'

        if vocals_path.exists():
            job.stems['vocals'] = str(vocals_path)
            logger.info("✅ Vocal Focus: Clean vocals extracted")
        else:
            raise Exception("Vocal Focus failed - no vocals generated")

        if instrumental_path.exists():
            job.stems['instrumental'] = str(instrumental_path)
            logger.info("✅ Vocal Focus: Instrumental track extracted")

        # Now run 6-stem on instrumental to get other stems
        job.stage = 'Vocal Focus: Separating instruments from backing track'
        job.progress = 55
        logger.info("Running 6-stem separation on instrumental...")

        pass2_output = output_path / 'pass2_instruments'
        pass2_output.mkdir(parents=True, exist_ok=True)

        result = subprocess.run([
            'python3', '-m', 'demucs',
            '--out', str(pass2_output),
            '-n', 'htdemucs_6s',
            str(instrumental_path)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Demucs pass 2 failed: {result.stderr[:500]}")
        job.progress = 70

        # Convert WAV to MP3 and collect instrument stems
        pass2_dir = pass2_output / 'htdemucs_6s' / 'no_vocals'
        if pass2_dir.exists():
            convert_wavs_to_mp3(pass2_dir)
            job.progress = 75
            for stem_file in pass2_dir.glob('*.mp3'):
                stem_name = stem_file.stem
                # Don't overwrite our better vocals
                if stem_name != 'vocals':
                    job.stems[stem_name] = str(stem_file)
                    logger.info(f"Found instrument stem: {stem_name}")

        logger.info(f"✅ Vocal Focus complete: {len(job.stems)} stems")
        return True

    except Exception as e:
        logger.error(f"Vocal Focus separation failed: {e}")
        job.error = str(e)
        return False


def separate_stems_ensemble(job: ProcessingJob, audio_path: Path):
    """
    ENSEMBLE separation - Moises.ai-quality multi-model approach.

    Strategy:
    1. Run multiple Demucs models (htdemucs_ft, htdemucs, htdemucs_6s)
    2. Vote on best stem for each instrument using quality metrics
    3. Apply post-processing (bleed reduction, noise removal, phase alignment)

    This produces the highest quality separation but takes longer (2-3x).
    Optimized for M3 Max with MPS acceleration.
    """
    if not ENSEMBLE_SEPARATOR_AVAILABLE:
        logger.warning("Ensemble separator not available, falling back to standard separation")
        return separate_stems(job, audio_path)

    import soundfile as sf

    job.stage = 'Loading audio for ensemble separation'
    job.progress = 5
    logger.info("🎯 Starting ENSEMBLE separation (Moises-quality)")

    try:
        # Load audio
        audio, sample_rate = sf.read(str(audio_path))
        if audio.ndim > 1:
            audio = audio.T  # Transpose to (channels, samples)

        # Create output directory
        output_path = OUTPUT_DIR / job.job_id / 'stems'
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize ensemble separator with progress callback
        def progress_callback(stage: str, progress: float):
            stage_map = {
                'initializing': ('Loading models', 10),
                'separating_htdemucs_ft': ('Separating with htdemucs_ft', 25),
                'separating_htdemucs': ('Separating with htdemucs', 45),
                'voting': ('Voting on best stems', 70),
                'post_processing': ('Post-processing stems', 85),
                'quality_analysis': ('Analyzing quality', 95),
                'complete': ('Saving stems', 98)
            }
            if stage in stage_map:
                job.stage = stage_map[stage][0]
                job.progress = stage_map[stage][1]
            logger.info(f"  Ensemble progress: {stage} ({progress*100:.0f}%)")

        # Use htdemucs_ft + htdemucs for best balance of quality vs speed
        # htdemucs_6s is great but produces different stems (6 vs 4)
        separator = EnsembleSeparator(
            models=['htdemucs_ft', 'htdemucs'],
            device='auto',
            enable_voting=True,
            enable_post_processing=True
        )
        separator.set_progress_callback(progress_callback)

        # Configure post-processing
        post_config = {
            'bleed_reduction': {'enabled': True, 'aggressiveness': 0.4},
            'noise_reduction': {'enabled': True, 'threshold_db': -45},
            'artifact_removal': True,
            'loudness_normalization': {'enabled': True, 'target_lufs': -14.0},
            'phase_alignment': True
        }

        # Run ensemble separation
        result = separator.separate(audio, sample_rate, post_config)

        job.stage = 'Saving ensemble stems'
        job.progress = 95

        # Save stems to files
        for stem_name, stem_audio in result.stems.items():
            stem_file = output_path / f'{stem_name}.mp3'

            # Convert to format soundfile can write
            if stem_audio.ndim > 1:
                stem_audio_write = stem_audio.T  # Back to (samples, channels)
            else:
                stem_audio_write = stem_audio

            # Write as WAV first, then convert to MP3
            wav_file = output_path / f'{stem_name}.wav'
            sf.write(str(wav_file), stem_audio_write, sample_rate)

            # Convert to MP3 using ffmpeg
            import subprocess
            subprocess.run([
                'ffmpeg', '-y', '-i', str(wav_file),
                '-codec:a', 'libmp3lame', '-q:a', '2',
                str(stem_file)
            ], capture_output=True)

            # Remove temp WAV
            if stem_file.exists():
                wav_file.unlink()
                job.stems[stem_name] = str(stem_file)
                logger.info(f"  Saved ensemble stem: {stem_name}")
            else:
                # Keep WAV if MP3 conversion failed
                job.stems[stem_name] = str(wav_file)
                logger.warning(f"  Kept WAV (MP3 conversion failed): {stem_name}")

        # Save quality report
        quality_report_path = output_path / 'ensemble_quality_report.json'
        with open(quality_report_path, 'w') as f:
            json.dump({
                'quality_report': result.quality_report,
                'selection_report': result.selection_report,
                'processing_time_seconds': result.processing_time_seconds,
                'models_used': result.models_used
            }, f, indent=2, default=str)

        job.progress = 100
        logger.info(f"✅ Ensemble separation complete: {len(job.stems)} stems in {result.processing_time_seconds:.1f}s")
        logger.info(f"   Models used: {result.models_used}")
        logger.info(f"   Overall quality: {result.quality_report.get('overall_quality', 'N/A'):.3f}")

        return True

    except Exception as e:
        logger.error(f"Ensemble separation failed: {e}")
        import traceback
        traceback.print_exc()
        job.error = str(e)
        return False


def _measure_stem_energy(stem_path: str) -> float:
    """Measure RMS energy of an audio file. Returns 0.0 on failure."""
    try:
        import librosa
        y, sr = librosa.load(stem_path, sr=22050, mono=True, duration=120)
        rms = float(np.sqrt(np.mean(y ** 2)))
        return rms
    except Exception as e:
        logger.debug(f"Energy measurement failed for {stem_path}: {e}")
        return 0.0


def _smart_rename_stem(raw_name: str, existing_stems: dict) -> str:
    """
    Rename an extracted sub-stem intelligently based on what already exists.

    Examples:
        other_vocals → backing_vocals (if 'vocals' exists)
        other_guitar → guitar_2 (if 'guitar' exists)
        other_piano  → keys_2 (if 'piano' exists)
        other_drums  → percussion_2 (if 'drums' exists)
        other_bass   → bass_2 (if 'bass' exists)
        other_other  → other_deep (recursive residual)
    """
    # Strip prefix layers: other_other_vocals → vocals
    base = raw_name
    while base.startswith('other_'):
        base = base[6:]

    if not base:
        return raw_name  # safety fallback

    rename_map = {
        'vocals': 'backing_vocals',
        'guitar': 'guitar_2',
        'piano': 'keys_2',
        'drums': 'percussion_2',
        'bass': 'bass_2',
        'other': 'other_deep',
    }

    # If the base instrument already exists in primary stems, use the smart name
    if base in existing_stems:
        candidate = rename_map.get(base, f'{base}_2')
    else:
        candidate = base  # No conflict — use the clean name

    # Avoid collisions: if candidate already taken, increment suffix
    if candidate in existing_stems:
        i = 2
        while f'{candidate}_{i}' in existing_stems:
            i += 1
        candidate = f'{candidate}_{i}'

    return candidate


def _deep_extract_stem(job: ProcessingJob, stem_path: str, depth: int,
                       max_depth: int, progress_base: float, progress_range: float) -> dict:
    """
    Recursively run Demucs on a stem to extract hidden instruments.

    Returns dict of {smart_name: file_path} for all extracted stems.
    """
    extracted = {}

    if depth > max_depth:
        return extracted

    if not os.path.exists(stem_path):
        return extracted

    pass_label = f"Pass {depth + 1}"
    output_path = OUTPUT_DIR / job.job_id / 'stems' / f'deep_pass_{depth}'
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        def progress_callback(progress):
            pct = progress_base + (progress.percent / 100.0) * progress_range
            job.progress = int(pct)
            if progress.eta_seconds > 0:
                eta_min = progress.eta_seconds / 60
                job.stage = f'Extracting hidden instruments ({pass_label}): {progress.percent:.0f}% (ETA: {eta_min:.1f}m)'
            else:
                job.stage = f'Extracting hidden instruments ({pass_label}): {progress.percent:.0f}%'

        runner = DemucsRunner(model='htdemucs_6s', progress_callback=progress_callback)
        result = runner.separate(
            audio_path=Path(stem_path),
            output_dir=output_path,
            timeout_seconds=1200  # 20 min max per pass
        )

        if not result.success:
            logger.warning(f"Deep extraction {pass_label} failed: {result.error_message}")
            return extracted

        # Find output directory
        stem_dir = result.output_dir
        if not stem_dir or not stem_dir.exists():
            # Fallback: look for the expected path
            stem_dir = output_path / 'htdemucs_6s' / Path(stem_path).stem
            if not stem_dir.exists():
                logger.warning(f"Deep extraction output dir not found for {pass_label}")
                return extracted

        # Convert WAV → MP3
        convert_wavs_to_mp3(stem_dir)

        # Collect results, measure energy, keep only meaningful stems
        residual_path = None
        for stem_file in stem_dir.glob('*.mp3'):
            sub_name = stem_file.stem  # e.g. "vocals", "drums", "other"
            energy = _measure_stem_energy(str(stem_file))

            if energy < 0.005:
                logger.info(f"  ⏭️ {pass_label}: Skipping near-silent stem '{sub_name}' (energy={energy:.4f})")
                continue

            if sub_name == 'other':
                # This is the residual — candidate for deeper pass
                residual_path = str(stem_file)
                residual_energy = energy
                logger.info(f"  📦 {pass_label}: Residual 'other' has energy={energy:.4f}")
                continue

            # Smart rename and add
            smart_name = _smart_rename_stem(sub_name, {**job.stems, **extracted})
            extracted[smart_name] = str(stem_file)
            logger.info(f"  🎵 {pass_label}: Extracted '{smart_name}' (energy={energy:.4f})")

        # Recursive pass on residual if it has significant content
        if residual_path and depth < max_depth:
            # Measure total energy to decide if residual is worth processing
            total_energy = sum(_measure_stem_energy(p) for p in job.stems.values() if os.path.exists(p))
            if total_energy > 0 and (residual_energy / total_energy) > 0.10:
                logger.info(f"  🔄 Residual has {residual_energy/total_energy:.0%} of total energy — going deeper")
                deeper = _deep_extract_stem(
                    job, residual_path, depth + 1, max_depth,
                    progress_base + progress_range,
                    progress_range * 0.5  # Each deeper pass gets half the progress range
                )
                extracted.update(deeper)
            else:
                # Keep the residual as a stem if it has content
                if residual_energy > 0.01:
                    residual_name = _smart_rename_stem('other', {**job.stems, **extracted})
                    extracted[residual_name] = residual_path
                    logger.info(f"  📦 Keeping residual as '{residual_name}'")

        return extracted

    except Exception as e:
        logger.error(f"Deep extraction {pass_label} failed: {e}")
        import traceback
        traceback.print_exc()
        return extracted


def smart_separate(job: ProcessingJob):
    """
    Smart Deep Extraction — automatically analyze and extract all instruments.

    Philosophy: One button, best result every time. No user configuration needed.

    Pipeline:
    1. Measure RMS energy of each stem from initial separation
    2. If 'other' has significant content (>10% of total energy), run deep extraction
    3. Recursively process residuals (max depth 2)
    4. Smart-name all extracted stems (backing_vocals, guitar_2, keys_2, etc.)
    5. Auto-run Skills system on remaining content for frequency-based extraction

    Replaces the old cascade_separate_other() with intelligent, automatic behavior.
    """
    other_path = job.stems.get('other')
    if not other_path or not os.path.exists(other_path):
        logger.info("🧠 Smart separation: No 'other' stem — track is cleanly separated")
        return False

    # Step 1: Measure energy across all stems
    job.stage = 'Analyzing audio complexity...'
    job.progress = 42
    logger.info("🧠 Smart separation: Analyzing stem energy distribution...")

    stem_energies = {}
    for name, path in job.stems.items():
        if os.path.exists(path):
            stem_energies[name] = _measure_stem_energy(path)

    total_energy = sum(stem_energies.values())
    other_energy = stem_energies.get('other', 0.0)

    if total_energy == 0:
        logger.warning("Smart separation: Could not measure stem energies")
        return False

    other_ratio = other_energy / total_energy
    logger.info(f"🧠 Energy distribution:")
    for name, energy in sorted(stem_energies.items(), key=lambda x: -x[1]):
        pct = (energy / total_energy * 100) if total_energy > 0 else 0
        logger.info(f"   {name}: {pct:.1f}% (rms={energy:.4f})")

    # Step 2: Decide whether to deep-extract
    if other_ratio < 0.05:
        logger.info(f"🧠 'other' is only {other_ratio:.0%} of total — simple track, skipping deep extraction")
        return False

    if other_ratio < 0.10:
        logger.info(f"🧠 'other' is {other_ratio:.0%} of total — minor content, skipping deep extraction")
        return False

    logger.info(f"🧠 'other' is {other_ratio:.0%} of total energy — running deep extraction!")

    # Step 3: Deep extraction (recursive, max depth 2)
    job.stage = 'Extracting hidden instruments...'
    job.progress = 44
    extracted = _deep_extract_stem(
        job, other_path, depth=0, max_depth=2,
        progress_base=44, progress_range=10  # 44-54% of overall progress
    )

    if extracted:
        # Add all extracted stems to job
        for name, path in extracted.items():
            job.stems[name] = path
        logger.info(f"🧠✅ Smart extraction found {len(extracted)} additional instruments: {list(extracted.keys())}")
    else:
        logger.info("🧠 Deep extraction didn't find significant additional instruments")

    # Step 4: Auto-run Skills system for frequency-based extraction
    if SKILLS_AVAILABLE:
        job.stage = 'Auto-detecting additional instruments...'
        job.progress = 55
        try:
            apply_skills_to_job(job)
        except Exception as e:
            logger.warning(f"Auto-skills failed (non-fatal): {e}")

    total_stems = len(job.stems)
    total_sub = sum(len(v) for v in job.sub_stems.values()) if job.sub_stems else 0
    logger.info(f"🧠 Smart separation complete: {total_stems} stems + {total_sub} sub-stems")

    return len(extracted) > 0


def quantize_midi(midi_path: str, grid_size: float = 0.125, min_note_length: float = 0.05):
    """
    Quantize MIDI notes to a rhythmic grid for tighter transcription.

    Args:
        midi_path: Path to the MIDI file
        grid_size: Quantization grid in beats (0.125 = 16th notes, 0.25 = 8th notes)
        min_note_length: Minimum note length in seconds (removes tiny noise notes)

    Returns:
        True if quantization succeeded, False otherwise
    """
    try:
        import pretty_midi

        # Load the MIDI file
        midi = pretty_midi.PrettyMIDI(midi_path)

        if not midi.instruments:
            logger.warning(f"No instruments in MIDI file: {midi_path}")
            return False

        # Get tempo for beat calculations (default 120 BPM if not found)
        tempo = 120.0
        if midi.get_tempo_changes()[1].size > 0:
            tempo = midi.get_tempo_changes()[1][0]

        # Convert grid_size from beats to seconds
        beat_duration = 60.0 / tempo
        grid_seconds = grid_size * beat_duration

        total_notes_original = 0
        total_notes_quantized = 0
        notes_removed = 0

        for instrument in midi.instruments:
            original_notes = instrument.notes.copy()
            total_notes_original += len(original_notes)
            quantized_notes = []

            for note in original_notes:
                # Skip very short notes (likely noise)
                note_length = note.end - note.start
                if note_length < min_note_length:
                    notes_removed += 1
                    continue

                # Quantize start time to grid
                quantized_start = round(note.start / grid_seconds) * grid_seconds

                # Quantize end time to grid (minimum 1 grid unit)
                quantized_end = round(note.end / grid_seconds) * grid_seconds
                if quantized_end <= quantized_start:
                    quantized_end = quantized_start + grid_seconds

                # Create quantized note
                note.start = quantized_start
                note.end = quantized_end
                quantized_notes.append(note)

            # Replace notes with quantized versions
            instrument.notes = quantized_notes
            total_notes_quantized += len(quantized_notes)

        # Save the quantized MIDI
        midi.write(midi_path)

        logger.info(f"🎹 Quantized MIDI: {total_notes_original} → {total_notes_quantized} notes "
                   f"(removed {notes_removed} tiny notes, grid={grid_size} beats)")
        return True

    except ImportError:
        logger.warning("pretty_midi not installed - skipping quantization")
        return False
    except Exception as e:
        logger.error(f"MIDI quantization failed: {e}")
        return False


def transcribe_to_midi(job: ProcessingJob, quantize: bool = True, grid_size: float = 0.125,
                       detect_articulations: bool = True):
    """
    Enhanced MIDI transcription with articulation detection and improved drum handling.

    Uses enhanced transcriber for melodic instruments (guitar, bass, piano, vocals)
    and drum transcriber v2 for drums. Falls back to Basic Pitch when modules unavailable.

    Args:
        job: The processing job
        quantize: Whether to quantize MIDI to grid (default True)
        grid_size: Quantization grid in beats (0.125 = 16th notes, 0.25 = 8th notes)
        detect_articulations: Whether to detect bends, slides, hammer-ons (default True)
    """
    import traceback

    job.stage = 'Transcribing to MIDI (enhanced)'
    job.progress = 60
    logger.info("Starting enhanced MIDI transcription")

    midi_output_dir = OUTPUT_DIR / job.job_id / 'midi'
    midi_output_dir.mkdir(parents=True, exist_ok=True)

    total_stems = len(job.stems)
    successful = 0

    # These stems don't work well with transcription (mixed content)
    SKIP_STEMS = {'other', 'instrumental'}

    # Initialize enhanced transcriber if available
    enhanced_transcriber = None
    if ENHANCED_TRANSCRIBER_AVAILABLE:
        try:
            enhanced_transcriber = EnhancedTranscriber()
            logger.info("🎸 Using enhanced transcriber with articulation detection")
        except Exception as e:
            logger.warning(f"Could not initialize enhanced transcriber: {e}")

    for idx, (stem_name, stem_path) in enumerate(job.stems.items()):
        try:
            # Skip mixed stems
            if stem_name.lower() in SKIP_STEMS:
                logger.info(f"⏭️ Skipping {stem_name} (mixed content)")
                continue

            # Verify the stem file exists
            if not Path(stem_path).exists():
                logger.error(f"Stem file not found: {stem_path}")
                continue

            # ==== DRUMS: Use OaF neural transcriber, fallback to v2/v1 ====
            if stem_name.lower() == 'drums' or '_drums' in stem_name.lower():
                drum_midi_path = midi_output_dir / f"{stem_name}_transcribed.mid"

                # Try OaF neural network transcriber first (best quality)
                if OAF_DRUM_TRANSCRIBER_AVAILABLE:
                    logger.info(f"🥁 Transcribing {stem_name} with OaF drum transcriber...")
                    job.stage = f'Transcribing {stem_name} (neural network)'

                    try:
                        result = transcribe_drums(
                            audio_path=stem_path,
                            output_path=str(drum_midi_path),
                            sensitivity=0.5,
                            prefer_neural=True  # Use OaF when available
                        )

                        if result.get('success') and result.get('total_hits', 0) > 0:
                            job.midi_files[stem_name] = str(drum_midi_path)
                            # Estimate quality from hit count distribution
                            quality = min(1.0, 0.5 + len(result.get('hits_by_type', {})) * 0.1)
                            job.transcription_quality[stem_name] = quality
                            logger.info(f"✓ Drum MIDI ({result.get('backend', 'unknown')}): "
                                       f"{result['total_hits']} hits, "
                                       f"types: {list(result.get('hits_by_type', {}).keys())}")
                            successful += 1
                            continue
                        else:
                            logger.warning(f"OaF returned no hits, trying v2")
                    except Exception as e:
                        logger.warning(f"OaF drum transcriber failed: {e}")

                # Try v2 spectral transcriber (ghost notes, cymbal detection)
                if DRUM_TRANSCRIBER_V2_AVAILABLE:
                    logger.info(f"🥁 Transcribing {stem_name} with spectral drum transcriber v2...")
                    job.stage = f'Transcribing {stem_name} (ghost notes, cymbals)'

                    try:
                        drum_transcriber = EnhancedDrumTranscriber()
                        stats = drum_transcriber.transcribe(
                            audio_path=stem_path,
                            output_path=str(drum_midi_path),
                            detect_ghost_notes=True,
                            preserve_dynamics=True
                        )

                        if stats.total_hits > 0:
                            job.midi_files[stem_name] = str(drum_midi_path)
                            job.transcription_quality[stem_name] = stats.quality_score
                            logger.info(f"✓ Drum MIDI: {stats.total_hits} hits, "
                                       f"ghost: {stats.ghost_notes}, "
                                       f"quality: {stats.quality_score:.2f}")
                            successful += 1
                        else:
                            logger.warning(f"No drum hits detected in {stem_name}")
                    except Exception as e:
                        logger.warning(f"Drum v2 failed, trying v1: {e}")
                        # Fall back to v1
                        if DRUM_TRANSCRIBER_AVAILABLE:
                            if transcribe_drums_to_midi(stem_path, str(drum_midi_path)):
                                job.midi_files[stem_name] = str(drum_midi_path)
                                successful += 1
                    continue

                elif DRUM_TRANSCRIBER_AVAILABLE:
                    # Fall back to v1 drum transcriber
                    logger.info(f"🥁 Transcribing {stem_name} with drum transcriber v1...")
                    job.stage = f'Transcribing {stem_name}'
                    drum_midi_path = midi_output_dir / f"{stem_name}_transcribed.mid"
                    if transcribe_drums_to_midi(stem_path, str(drum_midi_path)):
                        job.midi_files[stem_name] = str(drum_midi_path)
                        logger.info(f"✓ Created drum MIDI: {drum_midi_path}")
                        successful += 1
                    continue
                else:
                    logger.info(f"⏭️ Skipping {stem_name} (no drum transcriber available)")
                    continue

            # ==== MELODIC INSTRUMENTS ====
            stem_type = stem_name.lower().split('_')[0]  # Handle guitar_left, bass_right, etc.

            # ---- Try melody transcriber first for lead/monophonic stems ----
            is_lead_stem = stem_name.lower() in ['guitar_lead', 'vocals', 'vocals_lead', 'bass']
            # Also treat plain 'guitar' as lead candidate if no guitar_lead sub-stem
            if stem_name.lower() == 'guitar' and 'guitar_lead' not in job.stems:
                is_lead_stem = True

            if is_lead_stem and MELODY_TRANSCRIBER_AVAILABLE:
                logger.info(f"🎵 Transcribing {stem_name} with melody extractor (monophonic)...")
                job.stage = f'Transcribing {stem_name} (melody extraction)'

                try:
                    melody_ext = MelodyExtractor(instrument=stem_type)
                    mel_result = melody_ext.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        instrument=stem_type,
                        tempo_hint=job.metadata.get('tempo'),
                        ensemble=True,
                    )

                    if mel_result.midi_path and Path(mel_result.midi_path).exists() and mel_result.quality_score > 0.4:
                        job.midi_files[stem_name] = mel_result.midi_path
                        job.transcription_quality[stem_name] = mel_result.quality_score
                        job.articulations[stem_name] = mel_result.articulation_count
                        job.transcription_mode[stem_name] = 'melody'

                        if mel_result.detected_key and not job.detected_key:
                            job.detected_key = mel_result.detected_key

                        logger.info(f"✓ Melody MIDI for {stem_name}: "
                                   f"{len(mel_result.notes)} notes, "
                                   f"{mel_result.articulation_count} articulations, "
                                   f"quality: {mel_result.quality_score:.2f}")
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                    else:
                        quality = mel_result.quality_score if mel_result else 0
                        logger.info(f"  Melody quality too low ({quality:.2f}), "
                                   f"falling back to enhanced transcriber")
                except Exception as e:
                    logger.warning(f"Melody transcriber failed for {stem_name}: {e}")
                    # Fall through to enhanced transcriber

            # ---- Enhanced transcriber (polyphonic, articulations) ----
            if enhanced_transcriber and stem_type in ['guitar', 'bass', 'vocals', 'piano']:
                logger.info(f"🎼 Transcribing {stem_name} with enhanced transcriber...")
                job.stage = f'Transcribing {stem_name} (articulations)'

                try:
                    result = enhanced_transcriber.transcribe(
                        audio_path=stem_path,
                        output_dir=str(midi_output_dir),
                        stem_type=stem_type,
                        detect_articulations=detect_articulations,
                        quantize=quantize,
                        quantize_grid=grid_size
                    )

                    if result.midi_path and Path(result.midi_path).exists():
                        job.midi_files[stem_name] = result.midi_path
                        job.transcription_quality[stem_name] = result.quality_score
                        job.articulations[stem_name] = len(result.articulations)
                        job.transcription_mode[stem_name] = 'enhanced'

                        # Store detected key if not already set
                        if result.detected_key and not job.detected_key:
                            job.detected_key = result.detected_key

                        logger.info(f"✓ Enhanced MIDI for {stem_name}: "
                                   f"{result.notes_count} notes, "
                                   f"{len(result.articulations)} articulations, "
                                   f"quality: {result.quality_score:.2f}")
                        successful += 1
                        job.progress = 60 + int((idx + 1) / total_stems * 35)
                        continue
                except Exception as e:
                    logger.warning(f"Enhanced transcriber failed for {stem_name}: {e}")
                    # Fall through to Basic Pitch

            # ==== FALLBACK: Use Basic Pitch ====
            try:
                from basic_pitch.inference import predict_and_save
                from basic_pitch import ICASSP_2022_MODEL_PATH
            except ImportError as e:
                logger.warning(f"Basic Pitch not available for {stem_name}: {e}")
                continue

            logger.info(f"Transcribing {stem_name} with Basic Pitch...")
            job.stage = f'Transcribing {stem_name}'

            predict_and_save(
                audio_path_list=[stem_path],
                output_directory=str(midi_output_dir),
                save_midi=True,
                sonify_midi=False,
                save_model_outputs=False,
                save_notes=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH,
                onset_threshold=0.6,
                frame_threshold=0.4,
                minimum_note_length=80,
            )

            # Find the generated MIDI file
            stem_basename = Path(stem_path).stem
            midi_file = midi_output_dir / f"{stem_basename}_basic_pitch.mid"
            if not midi_file.exists():
                midi_file = midi_output_dir / f"{stem_basename}.mid"

            if midi_file.exists():
                if quantize:
                    job.stage = f'Quantizing {stem_name} MIDI'
                    quantize_midi(str(midi_file), grid_size=grid_size)

                job.midi_files[stem_name] = str(midi_file)
                logger.info(f"✓ Created MIDI for {stem_name}: {midi_file}")
                successful += 1
            else:
                logger.warning(f"MIDI file not found at expected path: {midi_file}")

            job.progress = 60 + int((idx + 1) / total_stems * 35)

        except Exception as e:
            logger.error(f"Transcription failed for {stem_name}: {e}")
            logger.error(traceback.format_exc())

    logger.info(f"MIDI transcription complete: {successful}/{total_stems} stems transcribed")
    return successful > 0


def detect_chords_for_job(job: ProcessingJob, audio_path: Path):
    """
    Detect chord progression from original audio or mix of stems.

    Stores chords in job.chord_progression with timestamps.
    """
    if not CHORD_DETECTOR_AVAILABLE:
        logger.info("Chord detector not available - skipping")
        return

    job.stage = 'Detecting chord progression'
    logger.info(f"🎸 Detecting chords (model: {CHORD_DETECTOR_VERSION})...")

    try:
        detector = ChordDetector()

        # Prefer to analyze the original audio if we have it
        analyze_path = str(audio_path) if audio_path.exists() else None

        # If original not available, try guitar or piano stem
        if not analyze_path or not Path(analyze_path).exists():
            if 'guitar' in job.stems:
                analyze_path = job.stems['guitar']
                logger.info("Using guitar stem for chord detection")
            elif 'guitar_left' in job.stems:
                analyze_path = job.stems['guitar_left']
            elif 'piano' in job.stems:
                analyze_path = job.stems['piano']
                logger.info("Using piano stem for chord detection")
            else:
                logger.info("No suitable source for chord detection")
                return

        progression = detector.detect(analyze_path)

        # Store results - V8 includes bass note for inversions, older versions don't
        job.chord_progression = []
        for c in progression.chords:
            chord_data = {
                'time': c.time,
                'duration': c.duration,
                'chord': c.chord,
                'root': c.root,
                'quality': c.quality,
                'confidence': c.confidence
            }
            # V8 adds bass note for chord inversions (e.g., C/E -> bass='E')
            if hasattr(c, 'bass'):
                chord_data['bass'] = c.bass
            chord_data['detector_version'] = CHORD_DETECTOR_VERSION
            job.chord_progression.append(chord_data)

        job.detected_key = progression.key

        logger.info(f"✅ Detected {len(job.chord_progression)} chord changes, key: {progression.key} (model: {CHORD_DETECTOR_VERSION})")

    except Exception as e:
        logger.warning(f"Chord detection failed: {e}")


def convert_midi_to_musicxml(job: ProcessingJob):
    """Convert MIDI files to MusicXML for notation display.

    Uses the midi_to_notation module for superior quality including:
    - Triplet-aware quantization
    - Melody mode with articulation markings (bends, slides, hammer-ons)
    - Dynamic markings from velocity
    - Artifact removal
    - Librosa beat-tracking fallback for tempo detection
    """
    if not NOTATION_CONVERTER_AVAILABLE:
        logger.warning("Notation converter not available - skipping MusicXML conversion")
        return False

    job.stage = 'Converting to notation'
    job.progress = 95
    logger.info("Converting MIDI to MusicXML via notation module")

    xml_output_dir = OUTPUT_DIR / job.job_id / 'musicxml'
    xml_output_dir.mkdir(parents=True, exist_ok=True)

    job.musicxml_files = {}

    # Extract song title from metadata or filename
    song_title = job.metadata.get('title', '') if job.metadata else ''
    if not song_title:
        song_title = job.filename.rsplit('.', 1)[0] if job.filename else "Untitled"
        for suffix in ['_Live', ' Live', ' (Live)', '_Remastered', ' Remastered']:
            if song_title.endswith(suffix):
                song_title = song_title[:-len(suffix)]
        song_title = song_title.replace('_', ' ')

    artist_name = job.metadata.get('artist', '') if job.metadata else ''

    for stem_name, midi_path in job.midi_files.items():
        try:
            stem_type = stem_name.lower().split('_')[0]
            xml_path = xml_output_dir / f"{stem_name}.musicxml"

            # Determine melody_mode from transcription_mode tracking
            is_melody = (hasattr(job, 'transcription_mode') and
                        job.transcription_mode.get(stem_name) == 'melody')

            # Get audio path for librosa tempo fallback
            audio_path = job.stems.get(stem_name) or job.stems.get(stem_type)

            result = _notation_convert(
                midi_path=midi_path,
                output_path=str(xml_path),
                quantize=True,
                stem_type=stem_type,
                title=song_title,
                artist=artist_name,
                melody_mode=is_melody,
                audio_path=audio_path,
            )

            if result and Path(result).exists():
                job.musicxml_files[stem_name] = result
                mode_str = " (melody mode)" if is_melody else ""
                logger.info(f"  ✓ MusicXML for {stem_name}{mode_str}")

        except Exception as e:
            logger.error(f"MusicXML conversion failed for {stem_name}: {e}")

    return len(job.musicxml_files) > 0


def apply_skills_to_job(job: ProcessingJob):
    """Analyze audio and apply only relevant skills based on content detection"""
    if not SKILLS_AVAILABLE:
        return

    job.stage = 'Analyzing audio content...'
    logger.info("🎧 Listening to stems to detect instruments...")

    # Smart detection - analyze stems and decide which skills to run
    skills_to_run = analyze_stems_for_skills(job.stems)

    if not skills_to_run:
        logger.info("No additional instrument content detected - skipping skill extraction")
        return

    job.selected_skills = skills_to_run  # Store for reference
    job.stage = f'Extracting {len(skills_to_run)} detected instruments'
    logger.info(f"🎯 Detected instruments, running skills: {skills_to_run}")

    output_dir = OUTPUT_DIR / job.job_id / 'stems'

    for skill_id in skills_to_run:
        try:
            skill = get_skill(skill_id)
            if not skill:
                logger.warning(f"Unknown skill: {skill_id}")
                continue

            job.stage = f'Applying {skill.emoji} {skill.name}'
            logger.info(f"Applying skill: {skill.name}")

            # Apply the skill
            sub_stem_paths = apply_skill(skill_id, job.stems, str(output_dir))

            if sub_stem_paths:
                # Convert absolute paths to relative paths for API
                job.sub_stems[skill_id] = {}
                for sub_stem_name, abs_path in sub_stem_paths.items():
                    # Store just the filename for the API
                    rel_path = f"skill_{skill_id}/{os.path.basename(abs_path)}"
                    job.sub_stems[skill_id][sub_stem_name] = rel_path
                    logger.info(f"  Generated sub-stem: {sub_stem_name}")

        except Exception as e:
            logger.error(f"Skill {skill_id} failed: {e}")


def process_audio(job: ProcessingJob, audio_path: Path, hq_vocals: bool = False, vocal_focus: bool = False, enhance_stems: bool = False, stereo_split: bool = False, gp_tabs: bool = True, chord_detection: bool = True, mdx_model: bool = False, ensemble_mode: bool = False):
    """Main processing pipeline - simplified: separate → (optional stereo split) → transcribe → done"""
    try:
        job.status = 'processing'

        # Step 1: Separate stems
        if ensemble_mode:
            logger.info("🎯 Using ENSEMBLE mode (Moises-quality multi-model)")
            if not separate_stems_ensemble(job, audio_path):
                job.status = 'failed'
                return
        elif mdx_model:
            logger.info("🎹 Using HYBRID model (MDX23C + dedicated piano extraction)")
            # Pass stereo_split to enable guitar splitting for dual-guitar bands
            if not separate_stems_mdx(job, audio_path, stereo_split_guitar=stereo_split):
                job.status = 'failed'
                return
        elif vocal_focus:
            logger.info("🎯 Using Vocal Focus mode")
            if not separate_stems_vocal_focus(job, audio_path):
                job.status = 'failed'
                return
        elif hq_vocals:
            logger.info("🎤 Using HQ Vocals mode")
            if not separate_stems_hq_vocals(job, audio_path):
                job.status = 'failed'
                return
        else:
            # Standard 6-stem separation
            if not separate_stems(job, audio_path):
                job.status = 'failed'
                return

        # Optional: Enhance stems (off by default)
        if enhance_stems and ENHANCER_AVAILABLE:
            job.stage = 'Enhancing stems'
            job.progress = 52
            try:
                enhanced_output_dir = OUTPUT_DIR / job.job_id / 'stems' / 'enhanced'
                enhanced_output_dir.mkdir(parents=True, exist_ok=True)
                job.enhanced_stems = enhance_all_stems(job.stems, str(enhanced_output_dir))
                logger.info(f"✅ Enhanced {len(job.enhanced_stems)} stems")
            except Exception as e:
                logger.warning(f"Stem enhancement failed: {e}")

        # Auto-split vocals into lead/backing using BS-Roformer (higher quality than Demucs)
        if ENHANCED_SEPARATOR_AVAILABLE and 'vocals' in job.stems:
            try:
                job.stage = 'Splitting lead/backing vocals (BS-Roformer)'
                job.progress = 45
                logger.info("🎤 Auto-splitting vocals with BS-Roformer for better quality...")

                vocal_split_dir = OUTPUT_DIR / job.job_id / 'stems' / 'vocal_split'
                vocal_split_dir.mkdir(parents=True, exist_ok=True)

                separator = EnhancedSeparator(output_dir=str(vocal_split_dir))
                lead_path, backing_path = separator.split_lead_backing_vocals(job.stems['vocals'])

                if lead_path and backing_path:
                    job.stems['vocals_lead'] = lead_path
                    job.stems['vocals_backing'] = backing_path
                    logger.info(f"✅ Vocals split: lead + backing")
                else:
                    logger.warning("Vocal split returned empty paths")
            except Exception as e:
                logger.warning(f"Auto vocal split failed (non-fatal): {e}")

        # Smart Deep Extraction: Automatically analyze and extract all instruments
        # Replaces manual cascade separation — one button, best result every time
        try:
            smart_result = smart_separate(job)
            if smart_result:
                logger.info(f"🧠 Smart extraction found additional instruments ({len(job.stems)} total stems)")
            else:
                logger.info("🧠 Track is cleanly separated — no deep extraction needed")
        except Exception as e:
            logger.warning(f"Smart separation failed (non-fatal): {e}")

        # Step 2: Detect chord progression (optional - before transcription)
        if chord_detection and CHORD_DETECTOR_AVAILABLE:
            job.stage = 'Detecting chords and key'
            job.progress = 58
            try:
                detect_chords_for_job(job, audio_path)
            except Exception as e:
                logger.warning(f"Chord detection failed (non-fatal): {e}")
        elif not chord_detection:
            logger.info("⏭️ Skipping chord detection (disabled)")

        # Step 3: Transcribe to MIDI
        job.progress = 60
        midi_success = transcribe_to_midi(job)
        if midi_success:
            logger.info(f"✓ MIDI transcription succeeded for {len(job.midi_files)} stems")

            # Step 3: Convert MIDI to MusicXML for notation
            xml_success = convert_midi_to_musicxml(job)
            if xml_success:
                logger.info(f"✓ MusicXML conversion succeeded for {len(job.musicxml_files)} stems")

            # Step 4: Convert MIDI to Guitar Pro for tablature (optional)
            if gp_tabs and GP_CONVERTER_AVAILABLE:
                job.stage = 'Creating Guitar Pro tabs'
                job.progress = 97
                try:
                    gp_output_dir = OUTPUT_DIR / job.job_id / 'guitarpro'
                    gp_files = convert_job_midis_to_gp(job, gp_output_dir)
                    if gp_files:
                        job.gp_files = gp_files
                        logger.info(f"✓ Guitar Pro conversion succeeded for {len(gp_files)} stems")
                except Exception as e:
                    logger.warning(f"Guitar Pro conversion failed (non-fatal): {e}")
            elif not gp_tabs:
                logger.info("⏭️ Skipping Guitar Pro tabs (disabled)")

            # Step 5: Try to fetch professional tabs from Songsterr (background, non-blocking)
            try:
                title = job.metadata.get('title', '') if job.metadata else ''
                artist = job.metadata.get('artist', '') if job.metadata else ''
                if title:
                    job.stage = 'Fetching professional tabs'
                    job.progress = 98
                    from songsterr import SongsterrAPI
                    api = SongsterrAPI()
                    tab = api.search(f"{title} {artist}".strip())
                    if tab:
                        output_dir = Path(job.output_dir)
                        gp5_path = api.download_gp5(tab, output_dir)
                        if gp5_path:
                            if not hasattr(job, 'pro_tabs') or job.pro_tabs is None:
                                job.pro_tabs = {}
                            job.pro_tabs['songsterr'] = {
                                'path': str(gp5_path),
                                'title': tab.title,
                                'artist': tab.artist,
                                'song_id': tab.song_id,
                                'tracks': [t.get('name', '') for t in tab.tracks[:5]]
                            }
                            logger.info(f"✓ Downloaded professional tabs from Songsterr: {tab.title}")
                        else:
                            logger.info("⚠ Songsterr tab found but download failed")
                    else:
                        logger.info("ℹ No Songsterr tab found for this song")
            except Exception as e:
                logger.debug(f"Pro tabs fetch skipped: {e}")
        else:
            logger.warning("⚠ MIDI transcription failed - check logs above for details")

        job.progress = 100
        job.stage = 'Complete'
        job.status = 'completed'
        logger.info(f"Job {job.job_id} completed successfully with {len(job.stems)} stems, {len(job.sub_stems)} skill outputs")

        # Auto-upload to Google Drive (transcriptions only, not stems)
        if DRIVE_AVAILABLE:
            try:
                job.stage = 'Uploading to Google Drive'
                drive_result = upload_job_to_drive(job, keep_stems=False)
                if drive_result:
                    job.metadata['drive_upload'] = drive_result
                    logger.info(f"Uploaded job {job.job_id} to Google Drive")
            except Exception as e:
                logger.warning(f"Drive upload failed (non-fatal): {e}")

        # Save job to disk for library persistence
        save_job_to_disk(job)

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        job.status = 'failed'
        job.error = str(e)


def process_url(job: ProcessingJob, url: str, hq_vocals: bool = False, vocal_focus: bool = False, enhance_stems: bool = False, stereo_split: bool = False, gp_tabs: bool = True, chord_detection: bool = True, mdx_model: bool = False, ensemble_mode: bool = False):
    """Download from URL then process"""
    try:
        job.status = 'processing'

        # Create upload directory for this job
        job_upload_dir = UPLOAD_DIR / job.job_id
        job_upload_dir.mkdir(exist_ok=True)

        # Step 0: Download audio
        audio_path = download_from_url(job, url, job_upload_dir)

        # DISABLED: Stereo splitting creates too many overlapping stems
        # Instead we use cascade separation on the 'other' stem
        # if not stereo_split and TRACK_INFO_AVAILABLE:
        #     artist = job.metadata.get('artist', '').lower()
        #     title = job.metadata.get('title', '').lower()
        #     if should_stereo_split(artist) or should_stereo_split(title):
        #         stereo_split = True
        #         logger.info(f"🎸 Auto-enabled stereo split for dual-guitar band: {artist}")

        # Steps 1-2: Separate and transcribe
        process_audio(job, audio_path, hq_vocals=hq_vocals, vocal_focus=vocal_focus, enhance_stems=enhance_stems, stereo_split=stereo_split, gp_tabs=gp_tabs, chord_detection=chord_detection, mdx_model=mdx_model, ensemble_mode=ensemble_mode)

    except Exception as e:
        logger.error(f"URL processing failed: {e}")
        job.status = 'failed'
        job.error = str(e)


@app.route('/api/health', methods=['GET'])
def health():
    # Check if yt-dlp is available
    ytdlp_available = shutil.which('yt-dlp') is not None

    # Get ensemble separator info if available
    ensemble_info = None
    if ENSEMBLE_SEPARATOR_AVAILABLE and _gpu_manager is not None:
        try:
            ensemble_info = {
                'available': True,
                'device': _gpu_manager.device_info.device_type.value,
                'device_name': _gpu_manager.device_info.device_name,
                'memory_gb': _gpu_manager.device_info.total_memory_gb
            }
        except:
            ensemble_info = {'available': True}
    elif ENSEMBLE_SEPARATOR_AVAILABLE:
        ensemble_info = {'available': True}

    return jsonify({
        'status': 'ok',
        'service': 'StemScribe API',
        'yt_dlp_available': ytdlp_available,
        'ensemble_separator': ensemble_info,
        'separation_modes': ['standard', 'hq_vocals', 'vocal_focus', 'mdx'] + (['ensemble'] if ENSEMBLE_SEPARATOR_AVAILABLE else [])
    })


@app.route('/api/library', methods=['GET'])
def get_library():
    """Get list of all processed songs in the library"""
    library = []

    for job_id, job in jobs.items():
        if job.status == 'completed' and job.stems:
            library.append({
                'job_id': job.job_id,
                'title': job.metadata.get('title', job.filename),
                'artist': job.metadata.get('artist', 'Unknown Artist'),
                'duration': job.metadata.get('duration', 0),
                'created_at': job.created_at,
                'stem_count': len(job.stems),
                'has_midi': len(job.midi_files) > 0,
                'has_gp': len(job.gp_files) > 0,
                'thumbnail': job.metadata.get('thumbnail'),
                'source_url': job.source_url
            })

    # Sort by created_at descending (newest first)
    library.sort(key=lambda x: x['created_at'], reverse=True)

    return jsonify({
        'library': library,
        'total': len(library)
    })


@app.route('/api/library/<job_id>', methods=['DELETE'])
def delete_from_library(job_id):
    """Delete a song from the library"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    try:
        # Remove the output directory
        job_dir = OUTPUT_DIR / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)

        # Remove from memory
        del jobs[job_id]

        logger.info(f"🗑️ Deleted job {job_id} from library")
        return jsonify({'status': 'deleted', 'job_id': job_id})

    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/skills', methods=['GET'])
def list_skills():
    """List available enhancement skills"""
    if not SKILLS_AVAILABLE:
        return jsonify({'skills': [], 'available': False})

    skills = []
    for skill in get_all_skills():
        skills.append({
            'id': skill.id,
            'name': skill.name,
            'emoji': skill.emoji,
            'description': skill.description,
            'generates': skill.generates,
            'genre_tags': skill.genre_tags
        })

    return jsonify({'skills': skills, 'available': True})


@app.route('/api/upload', methods=['POST'])
def upload_audio():
    """Upload an audio file for processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file type
    allowed_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.aiff', '.webm', '.opus'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed: {allowed_extensions}'}), 400

    # Get selected skills from form data
    skills = request.form.getlist('skills')  # e.g., ['horn_hunter', 'guitar_god']
    if not skills and request.form.get('skills'):
        # Handle comma-separated string
        skills = [s.strip() for s in request.form.get('skills').split(',') if s.strip()]

    # Check for processing options
    hq_vocals = request.form.get('hq_vocals', 'false').lower() == 'true'
    vocal_focus = request.form.get('vocal_focus', 'false').lower() == 'true'
    enhance_stems = request.form.get('enhance_stems', 'false').lower() == 'true'  # Off by default
    stereo_split = request.form.get('stereo_split', 'false').lower() == 'true'  # Split panned instruments
    gp_tabs = request.form.get('gp_tabs', 'true').lower() == 'true'  # Generate Guitar Pro tabs
    chord_detection = request.form.get('chord_detection', 'true').lower() == 'true'  # Detect chords
    mdx_model = request.form.get('mdx_model', 'false').lower() == 'true'  # Use MDX23C for better piano
    ensemble_mode = request.form.get('ensemble', 'false').lower() == 'true'  # Use ensemble multi-model (Moises-quality)

    # Create job with skills
    job_id = str(uuid.uuid4())[:8]
    job = ProcessingJob(job_id, file.filename, skills=skills)
    jobs[job_id] = job

    # Save uploaded file
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(exist_ok=True)
    audio_path = job_upload_dir / file.filename
    file.save(str(audio_path))

    mode_str = 'ENSEMBLE' if ensemble_mode else ('MDX' if mdx_model else 'standard')
    logger.info(f"Created job {job_id} for file {file.filename} - mode: {mode_str}, gp_tabs: {gp_tabs}, chord_detection: {chord_detection}")

    # Start processing in background thread
    thread = threading.Thread(target=process_audio, args=(job, audio_path, hq_vocals, vocal_focus, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode))
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Processing started',
        'filename': file.filename,
        'skills': skills
    })


@app.route('/api/url', methods=['POST'])
def process_url_endpoint():
    """Process audio from a URL (YouTube, Spotify, Apple Music, etc.)"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url'].strip()

    # Validate URL
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid URL format'}), 400

    # Check if yt-dlp is available
    if not shutil.which('yt-dlp'):
        return jsonify({
            'error': 'yt-dlp not installed. Run: brew install yt-dlp'
        }), 500

    # Check if this is a streaming service URL (Spotify/Apple Music)
    streaming_service = is_streaming_url(url)
    original_url = url
    track_info = None

    if streaming_service:
        logger.info(f"Detected {streaming_service} URL, extracting track info...")

        # Get track info from streaming service
        if streaming_service == 'spotify':
            track_info = get_spotify_track_info(url)
        elif streaming_service == 'apple_music':
            track_info = get_apple_music_track_info(url)

        if not track_info:
            return jsonify({
                'error': f'Could not extract track info from {streaming_service}. Try pasting a direct track link.'
            }), 400

        # Search YouTube for the song
        youtube_url, yt_data = search_youtube_for_song(track_info['search_query'])

        if not youtube_url:
            return jsonify({
                'error': f'Could not find "{track_info["search_query"]}" on YouTube.'
            }), 404

        logger.info(f"Redirecting {streaming_service} to YouTube: {youtube_url}")
        url = youtube_url

    elif not is_supported_url(url):
        return jsonify({
            'error': 'Unsupported URL. Supported: YouTube, Spotify, Apple Music, SoundCloud, Bandcamp, Vimeo, Archive.org'
        }), 400

    # Get selected skills from request data
    skills = data.get('skills', [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',') if s.strip()]

    # Check for processing options
    hq_vocals = data.get('hq_vocals', False)
    vocal_focus = data.get('vocal_focus', False)
    enhance_stems = data.get('enhance_stems', False)  # Off by default
    stereo_split = data.get('stereo_split', False)  # Split panned instruments
    gp_tabs = data.get('gp_tabs', True)  # Generate Guitar Pro tabs
    chord_detection = data.get('chord_detection', True)  # Detect chords
    mdx_model = data.get('mdx_model', False)  # Use MDX23C for better piano
    ensemble_mode = data.get('ensemble', False)  # Use ensemble multi-model (Moises-quality)

    # Create job with skills
    job_id = str(uuid.uuid4())[:8]
    job = ProcessingJob(job_id, 'Downloading...', source_url=original_url, skills=skills)
    jobs[job_id] = job

    # Store streaming service info if applicable
    if track_info:
        job.metadata['original_service'] = streaming_service
        job.metadata['original_url'] = original_url
        job.metadata['search_query'] = track_info['search_query']
        if track_info.get('thumbnail'):
            job.metadata['thumbnail'] = track_info['thumbnail']

    mode_str = 'ENSEMBLE' if ensemble_mode else ('MDX' if mdx_model else 'standard')
    logger.info(f"Created job {job_id} for URL {url} - mode: {mode_str}, gp_tabs: {gp_tabs}, chord_detection: {chord_detection}")

    # Start processing in background thread
    thread = threading.Thread(target=process_url, args=(job, url, hq_vocals, vocal_focus, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode))
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Download and processing started',
        'url': url,
        'source': streaming_service or 'direct',
        'track_info': track_info
    })


@app.route('/api/find-tabs/<job_id>', methods=['GET'])
def find_tabs(job_id):
    """Find matching tabs on Songsterr and Ultimate Guitar for a job"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    
    title = job.metadata.get('title', '') if job.metadata else ''
    artist = job.metadata.get('artist', '') if job.metadata else ''
    
    if not title:
        return jsonify({'error': 'No song title available'}), 400
    
    # Clean up title for URL
    def slugify(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', '', text)
        text = re.sub(r'[\s_]+', '-', text)
        return text.strip('-')
    
    title_slug = slugify(title)
    artist_slug = slugify(artist) if artist else ''
    search_query = f"{title} {artist}".strip().replace(' ', '+')
    
    # Build URLs for external tab sites
    tabs = {
        'songsterr': {
            'search_url': f"https://www.songsterr.com/?pattern={search_query}",
            'name': 'Songsterr',
            'icon': '🎸'
        },
        'ultimate_guitar': {
            'search_url': f"https://www.ultimate-guitar.com/search.php?search_type=title&value={search_query}",
            'name': 'Ultimate Guitar', 
            'icon': '🎵'
        },
        'songsterr_direct': None,
        'ug_direct': None
    }
    
    # Try to find direct Songsterr link
    try:
        from songsterr import SongsterrAPI
        api = SongsterrAPI()
        tab = api.search(f"{title} {artist}")
        if tab:
            tabs['songsterr_direct'] = {
                'url': f"https://www.songsterr.com/a/wsa/{slugify(tab.artist)}-{slugify(tab.title)}-tab-s{tab.song_id}",
                'title': tab.title,
                'artist': tab.artist,
                'song_id': tab.song_id
            }
    except Exception as e:
        logger.warning(f"Songsterr search failed: {e}")
    
    return jsonify({
        'job_id': job_id,
        'title': title,
        'artist': artist,
        'tabs': tabs
    })


@app.route('/api/download-pro-tabs/<job_id>', methods=['POST'])
def download_pro_tabs(job_id):
    """
    Download professional tabs from Songsterr for this job.
    These are human-curated, accurate tabs that can be used as-is or
    as a quality reference alongside AI-generated transcriptions.

    POST /api/download-pro-tabs/abc123

    Returns paths to downloaded GP5 files for each instrument track.
    """
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    title = job.metadata.get('title', '') if job.metadata else ''
    artist = job.metadata.get('artist', '') if job.metadata else ''

    if not title:
        return jsonify({'error': 'No song title available'}), 400

    try:
        from songsterr import SongsterrAPI
        api = SongsterrAPI()

        # Search for the tab
        query = f"{title} {artist}".strip()
        logger.info(f"Searching Songsterr for: {query}")
        tab = api.search(query)

        if not tab:
            return jsonify({
                'success': False,
                'error': 'No tab found on Songsterr',
                'search_url': f"https://www.songsterr.com/?pattern={query.replace(' ', '+')}"
            }), 404

        # Download the GP5 file
        output_dir = Path(job.output_dir)
        gp5_path = api.download_gp5(tab, output_dir)

        if not gp5_path:
            return jsonify({
                'success': False,
                'error': 'Failed to download GP5 file'
            }), 500

        # Store in job
        if not hasattr(job, 'pro_tabs') or job.pro_tabs is None:
            job.pro_tabs = {}

        job.pro_tabs['songsterr'] = {
            'path': str(gp5_path),
            'title': tab.title,
            'artist': tab.artist,
            'song_id': tab.song_id,
            'revision_id': tab.revision_id,
            'tracks': [
                {
                    'name': t.get('name', ''),
                    'instrument': t.get('instrument', ''),
                    'is_guitar': t.get('isGuitar', False),
                    'is_bass': t.get('isBassGuitar', False),
                    'is_drums': t.get('isDrums', False),
                }
                for t in tab.tracks
            ],
            'url': f"https://www.songsterr.com/a/wsa/{tab.song_id}"
        }

        logger.info(f"✅ Downloaded pro tabs: {tab.title} by {tab.artist}")

        return jsonify({
            'success': True,
            'job_id': job_id,
            'tab': job.pro_tabs['songsterr'],
            'download_path': f"/api/download/{job_id}/pro_tabs/songsterr.gp5"
        })

    except ImportError:
        return jsonify({
            'success': False,
            'error': 'Songsterr module not available'
        }), 500
    except Exception as e:
        logger.error(f"Failed to download pro tabs: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/download/<job_id>/pro_tabs/<filename>', methods=['GET'])
def download_pro_tab_file(job_id, filename):
    """Download a professional tab file (GP5)"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not hasattr(job, 'pro_tabs') or not job.pro_tabs:
        return jsonify({'error': 'No professional tabs available'}), 404

    # Find the file
    if 'songsterr' in job.pro_tabs:
        tab_path = Path(job.pro_tabs['songsterr']['path'])
        if tab_path.exists():
            return send_file(
                tab_path,
                as_attachment=True,
                download_name=filename
            )

    return jsonify({'error': 'Tab file not found'}), 404


@app.route('/api/split-stem/<job_id>/<stem_name>', methods=['POST'])
def split_stem_endpoint(job_id, stem_name):
    """
    Split a stem into left/right/center components using stereo panning analysis.

    POST /api/split-stem/abc123/guitar

    Returns the split stem paths and stereo analysis info.
    """
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not STEREO_SPLITTER_AVAILABLE:
        return jsonify({'error': 'Stereo splitter not available'}), 500

    # Find the stem
    if stem_name not in job.stems:
        # Try case-insensitive match
        stem_name_lower = stem_name.lower()
        matching = [k for k in job.stems.keys() if k.lower() == stem_name_lower]
        if matching:
            stem_name = matching[0]
        else:
            return jsonify({
                'error': f'Stem "{stem_name}" not found',
                'available_stems': list(job.stems.keys())
            }), 404

    stem_path = job.stems[stem_name]

    # Check if splittable first
    check = check_if_splittable(stem_path)

    # Create output directory for split stems
    job_dir = Path(OUTPUT_DIR) / job_id
    split_dir = job_dir / 'stereo_split'
    split_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🎚️ Splitting {stem_name} stem for job {job_id}")
    logger.info(f"   Stereo analysis: width={check.get('width', 0):.2f}, side_ratio={check.get('side_ratio', 0):.2f}")

    # Do the split using enhanced method
    split_results = split_stereo(
        input_path=stem_path,
        output_dir=str(split_dir),
        stem_type=stem_name,
        method='enhanced'
    )

    if not split_results:
        return jsonify({
            'error': 'Stereo split failed',
            'analysis': check
        }), 500

    # Add split stems to job's stem dictionary
    for split_name, split_path in split_results.items():
        job.stems[split_name] = split_path

    # Save updated job state
    save_job(job)

    return jsonify({
        'success': True,
        'job_id': job_id,
        'original_stem': stem_name,
        'analysis': check,
        'split_stems': {k: v for k, v in split_results.items()},
        'message': f'Split {stem_name} into {len(split_results)} components'
    })


@app.route('/api/analyze-stereo/<job_id>/<stem_name>', methods=['GET'])
def analyze_stereo_endpoint(job_id, stem_name):
    """
    Analyze a stem's stereo field without splitting it.
    Useful to check if a stem is worth splitting.
    """
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not STEREO_SPLITTER_AVAILABLE:
        return jsonify({'error': 'Stereo splitter not available'}), 500

    # Find the stem
    if stem_name not in job.stems:
        stem_name_lower = stem_name.lower()
        matching = [k for k in job.stems.keys() if k.lower() == stem_name_lower]
        if matching:
            stem_name = matching[0]
        else:
            return jsonify({
                'error': f'Stem "{stem_name}" not found',
                'available_stems': list(job.stems.keys())
            }), 404

    stem_path = job.stems[stem_name]
    check = check_if_splittable(stem_path)

    return jsonify({
        'job_id': job_id,
        'stem': stem_name,
        'analysis': check,
        'recommendation': 'Split recommended' if check.get('splittable') else 'Mostly mono - splitting may not help'
    })


@app.route('/api/split-vocals/<job_id>', methods=['POST'])
def split_vocals_endpoint(job_id):
    """
    Split vocals into lead and backing vocals using AI (two-pass UVR method).

    POST /api/split-vocals/abc123

    Returns paths to lead_vocals and backing_vocals files.
    """
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not ENHANCED_SEPARATOR_AVAILABLE:
        return jsonify({'error': 'Enhanced separator not available (audio-separator not installed)'}), 500

    # Find vocals stem
    vocals_path = None
    vocals_key = None
    for key in ['vocals', 'Vocals', 'vocal', 'Vocal']:
        if key in job.stems:
            vocals_path = job.stems[key]
            vocals_key = key
            break

    if not vocals_path:
        return jsonify({
            'error': 'No vocals stem found',
            'available_stems': list(job.stems.keys())
        }), 404

    # Create output directory
    job_dir = Path(OUTPUT_DIR) / job_id
    vocal_split_dir = job_dir / 'vocal_split'
    vocal_split_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"🎤 Splitting vocals for job {job_id}")

    try:
        # Use enhanced separator for vocal splitting
        separator = EnhancedSeparator(output_dir=str(vocal_split_dir))
        lead_path, backing_path = separator.split_lead_backing_vocals(vocals_path)

        # Add to job stems
        job.stems['vocals_lead'] = lead_path
        job.stems['vocals_backing'] = backing_path

        # Save job
        save_job(job)

        return jsonify({
            'success': True,
            'job_id': job_id,
            'original_vocals': vocals_path,
            'lead_vocals': lead_path,
            'backing_vocals': backing_path,
            'message': 'Successfully split vocals into lead and backing'
        })

    except Exception as e:
        logger.error(f"Vocal split failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({
            'error': f'Vocal split failed: {str(e)}'
        }), 500


@app.route('/api/available-models', methods=['GET'])
def get_available_models():
    """
    Get list of available separation and transcription models.
    """
    models = {}

    # Separation models
    if ENHANCED_SEPARATOR_AVAILABLE:
        models['enhanced'] = {
            name: {
                'description': config['description'],
                'stems': config['stems']
            }
            for name, config in SEPARATOR_MODELS.items()
        }

    models['demucs'] = {
        'htdemucs_6s': {
            'description': 'Demucs 6-stem (current default)',
            'stems': ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
        }
    }

    # Transcription models
    transcription_models = {}

    if OAF_DRUM_TRANSCRIBER_AVAILABLE:
        transcription_models['drums_oaf'] = {
            'description': 'OaF Drums - Neural network trained on E-GMD (444 hours)',
            'available': OAF_AVAILABLE,
            'task': 'drums'
        }

    if DRUM_TRANSCRIBER_V2_AVAILABLE:
        transcription_models['drums_spectral'] = {
            'description': 'Spectral drum transcriber with ghost notes and cymbal detection',
            'available': True,
            'task': 'drums'
        }

    if ENHANCED_TRANSCRIBER_AVAILABLE:
        transcription_models['melodic_enhanced'] = {
            'description': 'Enhanced pitch transcriber with articulation detection',
            'available': True,
            'task': 'melodic'
        }

    models['transcription'] = transcription_models

    # Get pretrained model status from model manager
    pretrained_status = {}
    if MODEL_MANAGER_AVAILABLE:
        try:
            pretrained_status = list_available_models()
        except Exception as e:
            logger.warning(f"Could not get pretrained model status: {e}")

    return jsonify({
        'enhanced_separator_available': ENHANCED_SEPARATOR_AVAILABLE,
        'stereo_splitter_available': STEREO_SPLITTER_AVAILABLE,
        'oaf_drums_available': OAF_DRUM_TRANSCRIBER_AVAILABLE and OAF_AVAILABLE,
        'drum_transcriber_v2_available': DRUM_TRANSCRIBER_V2_AVAILABLE,
        'enhanced_transcriber_available': ENHANCED_TRANSCRIBER_AVAILABLE,
        'model_manager_available': MODEL_MANAGER_AVAILABLE,
        'models': models,
        'pretrained': pretrained_status
    })


@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the status of a processing job"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Debug: log what files are available
    print(f"\n📊 STATUS REQUEST for {job_id}")
    print(f"   MIDI files: {list(job.midi_files.keys()) if job.midi_files else 'NONE'}")
    print(f"   MusicXML files: {list(job.musicxml_files.keys()) if job.musicxml_files else 'NONE'}")
    print(f"   GP files: {list(job.gp_files.keys()) if job.gp_files else 'NONE'}", flush=True)

    return jsonify(job.to_dict())


@app.route('/api/chords/<job_id>', methods=['GET'])
def get_chords(job_id):
    """Get detected chord progression for a job."""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Include timing for each chord, key signature, and detector info
    return jsonify({
        'job_id': job_id,
        'chords': job.chord_progression,
        'key': job.detected_key,
        'available': CHORD_DETECTOR_AVAILABLE,
        'detector_version': CHORD_DETECTOR_VERSION,
        'chord_count': len(job.chord_progression),
        'has_inversions': CHORD_DETECTOR_VERSION == 'v8'
    })


@app.route('/api/theory/<job_id>', methods=['GET'])
def get_chord_theory(job_id):
    """Get scale suggestions and theory analysis for a job's chord progression."""
    if not CHORD_THEORY_AVAILABLE:
        return jsonify({'error': 'Chord theory engine not available'}), 500

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not job.chord_progression:
        return jsonify({'error': 'No chord progression detected yet'}), 404

    try:
        key = job.detected_key
        chord_names = [c['chord'] for c in job.chord_progression]

        # Get individual chord analyses
        analyses = _chord_theory_engine.get_scales_for_progression(chord_names, key)

        # Get overall practice suggestion
        practice_suggestion = _chord_theory_engine.suggest_practice_approach(chord_names, key)

        # Build response with timing info merged
        theory_data = []
        for i, chord_data in enumerate(job.chord_progression):
            analysis = analyses[i] if i < len(analyses) else {}
            theory_data.append({
                'chord': chord_data['chord'],
                'time': chord_data['time'],
                'duration': chord_data['duration'],
                'scales': analysis.get('scales', []),
                'secondary_scales': analysis.get('secondary_scales', []),
                'tip': analysis.get('tip', ''),
                'chord_type': analysis.get('chord_type', ''),
                'function': analysis.get('function'),
                'intervals': analysis.get('intervals', ''),
            })

        return jsonify({
            'job_id': job_id,
            'key': key,
            'theory': theory_data,
            'practice_suggestion': practice_suggestion,
            'available': True,
            'chord_count': len(theory_data)
        })

    except Exception as e:
        logger.error(f"Chord theory analysis failed: {e}")
        return jsonify({'error': f'Theory analysis failed: {str(e)}'}), 500


@app.route('/api/theory/chord', methods=['POST'])
def get_single_chord_theory():
    """Get scale suggestions for a single chord (no job required)."""
    if not CHORD_THEORY_AVAILABLE:
        return jsonify({'error': 'Chord theory engine not available'}), 500

    data = request.get_json()
    if not data or 'chord' not in data:
        return jsonify({'error': 'Missing chord parameter'}), 400

    chord = data['chord']
    key = data.get('key')

    analysis = _chord_theory_engine.analyze(chord, key)
    return jsonify(analysis)


@app.route('/api/quality/<job_id>', methods=['GET'])
def get_transcription_quality(job_id):
    """Get transcription quality metrics for a job."""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    return jsonify({
        'job_id': job_id,
        'quality_scores': job.transcription_quality,
        'articulations': job.articulations,
        'detected_key': job.detected_key,
        'enhanced_transcriber_used': ENHANCED_TRANSCRIBER_AVAILABLE,
        'drum_transcriber_v2_used': DRUM_TRANSCRIBER_V2_AVAILABLE
    })


@app.route('/api/info/<job_id>', methods=['GET'])
def get_track_info(job_id):
    """Get contextual info about a track (artist bio, learning tips, etc.)"""
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        if not TRACK_INFO_AVAILABLE:
            return jsonify({'error': 'Track info module not available'}), 500

        # Get track name from job
        track_name = job.filename
        if job.metadata.get('search_query'):
            track_name = job.metadata['search_query']
        elif job.metadata.get('title'):
            track_name = job.metadata['title']

        logger.info(f"Fetching track info for job {job_id}: {track_name}")

        # Fetch info with timeout protection
        info = fetch_track_info(
            track_name=track_name,
            artist=job.metadata.get('artist'),
            source_url=job.source_url
        )

        # Add instrument-specific tips for each stem
        info['stem_tips'] = {}
        for stem_name in job.stems.keys():
            info['stem_tips'][stem_name] = get_instrument_tips(stem_name, info.get('style'))

        # Add stereo split recommendation
        artist = job.metadata.get('artist') or info.get('artist')
        info['stereo_split_recommended'] = should_stereo_split(artist)

        return jsonify(info)

    except Exception as e:
        logger.error(f"Track info error for job {job_id}: {e}")
        # Return basic info instead of failing completely
        return jsonify({
            'error': None,
            'track': job.filename if job else 'Unknown',
            'artist': job.metadata.get('artist') if job else None,
            'bio': 'Track information temporarily unavailable.',
            'learning_tips': 'Use the stem separation to isolate and study individual parts.',
            'fetched_from': ['fallback']
        })


@app.route('/api/info/search', methods=['POST'])
def search_track_info():
    """Search for track info by artist/song name"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if not TRACK_INFO_AVAILABLE:
        return jsonify({'error': 'Track info module not available'}), 500

    track_name = data.get('track', '')
    artist = data.get('artist', '')

    if not track_name and not artist:
        return jsonify({'error': 'Provide track or artist name'}), 400

    info = fetch_track_info(track_name=track_name, artist=artist)
    return jsonify(info)


@app.route('/api/download/<job_id>/<file_type>/<filename>', methods=['GET'])
def download_file(job_id, file_type, filename):
    """Download a stem or MIDI file"""
    # Print to stdout immediately so we see it in terminal
    print(f"\n{'='*50}")
    print(f"📥 DOWNLOAD REQUEST: {file_type}/{filename}")
    print(f"   Job ID: {job_id}")
    print(f"{'='*50}\n", flush=True)
    logger.info(f"📥 Download request: {job_id}/{file_type}/{filename}")
    try:
        job = get_job(job_id)
        if not job:
            logger.warning(f"  ❌ Job {job_id} not found")
            return jsonify({'error': 'Job not found'}), 404
        logger.info(f"  ✓ Job loaded: {job.filename}")

        if file_type == 'stem':
            if filename not in job.stems:
                available = list(job.stems.keys())
                return jsonify({'error': f'Stem not found. Available: {available}'}), 404
            file_path = job.stems[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'Stem file missing from disk: {file_path}'}), 404
            return send_file(file_path, as_attachment=True)

        elif file_type == 'enhanced':
            if filename not in job.enhanced_stems:
                available = list(job.enhanced_stems.keys())
                return jsonify({'error': f'Enhanced stem not found. Available: {available}'}), 404
            file_path = job.enhanced_stems[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'Enhanced stem file missing from disk: {file_path}'}), 404
            return send_file(file_path, as_attachment=True)

        elif file_type == 'midi':
            if filename not in job.midi_files:
                available = list(job.midi_files.keys())
                return jsonify({'error': f'MIDI file not found. Available: {available}'}), 404
            file_path = job.midi_files[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'MIDI file missing from disk: {file_path}'}), 404
            return send_file(file_path, as_attachment=True)

        elif file_type == 'musicxml':
            logger.info(f"  📄 MusicXML request for '{filename}'")
            logger.info(f"     Available: {list(job.musicxml_files.keys()) if job.musicxml_files else 'NONE'}")
            if filename not in job.musicxml_files:
                available = list(job.musicxml_files.keys())
                logger.warning(f"  ❌ MusicXML '{filename}' not found. Available: {available}")
                return jsonify({'error': f'MusicXML not found. Available: {available}'}), 404
            file_path = job.musicxml_files[filename]
            logger.info(f"     Path: {file_path}")
            if not Path(file_path).exists():
                logger.error(f"  ❌ File missing: {file_path}")
                return jsonify({'error': f'MusicXML file missing from disk: {file_path}'}), 404
            logger.info(f"  ✓ Sending file: {Path(file_path).name} ({Path(file_path).stat().st_size} bytes)")
            return send_file(file_path, as_attachment=True, mimetype='application/xml')

        elif file_type == 'gp' or file_type == 'guitarpro':
            if filename not in job.gp_files:
                available = list(job.gp_files.keys())
                return jsonify({'error': f'Guitar Pro not found. Available: {available}'}), 404
            file_path = job.gp_files[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'GP file missing from disk: {file_path}'}), 404
            return send_file(file_path, as_attachment=True,
                            mimetype='application/x-gp5',
                            download_name=f"{filename}.gp5")

        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        import traceback
        logger.error(f"Download error for {job_id}/{file_type}/{filename}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/download/<job_id>/substem/<skill_id>/<filename>', methods=['GET'])
def download_substem(job_id, skill_id, filename):
    """Download a skill-generated sub-stem"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if skill_id not in job.sub_stems:
        return jsonify({'error': f'Skill {skill_id} not found in job'}), 404

    # Find the sub-stem by filename
    for sub_stem_name, rel_path in job.sub_stems[skill_id].items():
        if os.path.basename(rel_path) == filename or sub_stem_name == filename.replace('.wav', ''):
            # Construct full path
            full_path = OUTPUT_DIR / job_id / 'stems' / rel_path
            if full_path.exists():
                return send_file(str(full_path), as_attachment=True)

    return jsonify({'error': 'Sub-stem file not found'}), 404


@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': [job.to_dict() for job in jobs.values()]
    })


# ============ Google Drive Endpoints ============

@app.route('/api/drive/auth', methods=['GET'])
def drive_auth():
    """Initiate Google Drive OAuth flow"""
    if not DRIVE_AVAILABLE:
        return jsonify({'error': 'Google Drive integration not available. Install: pip install google-auth-oauthlib google-api-python-client --break-system-packages'}), 500

    try:
        service = get_drive_service()
        if service:
            return jsonify({'status': 'authenticated', 'message': 'Google Drive connected!'})
        else:
            return jsonify({'status': 'auth_required', 'message': 'Please complete OAuth flow in browser'}), 401
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/drive/stats', methods=['GET'])
def drive_stats():
    """Get Google Drive storage stats for StemScribe"""
    if not DRIVE_AVAILABLE:
        return jsonify({'error': 'Drive integration not available'}), 500

    stats = get_drive_stats()
    if stats:
        return jsonify(stats)
    else:
        return jsonify({'error': 'Could not get Drive stats - may need to authenticate'}), 500


@app.route('/api/drive/upload/<job_id>', methods=['POST'])
def drive_upload_job(job_id):
    """Manually upload a job to Google Drive"""
    if not DRIVE_AVAILABLE:
        return jsonify({'error': 'Drive integration not available'}), 500

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.status != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400

    data = request.get_json() or {}
    keep_stems = data.get('keep_stems', False)

    result = upload_job_to_drive(job, keep_stems=keep_stems)
    if result:
        job.metadata['drive_upload'] = result
        return jsonify({'status': 'uploaded', 'result': result})
    else:
        return jsonify({'error': 'Upload failed'}), 500


# =============================================================================
# Archive.org Live Music API
# =============================================================================

@app.route('/api/archive/search', methods=['GET'])
def archive_search():
    """
    Search the Internet Archive Live Music collection.

    Query params:
        q: Search query (band name, date, venue, etc.)
        collection: Archive.org collection ID (e.g., "GratefulDead")
        year: Filter by year (e.g., "1977")
        sort: Sort order — "date", "rating", "downloads" (default: "downloads")
        rows: Number of results (default: 25, max: 100)

    Example: /api/archive/search?q=grateful+dead+1977&sort=rating
    """
    if not ARCHIVE_PIPELINE_AVAILABLE:
        return jsonify({'error': 'Archive.org pipeline not available', 'available': False}), 500

    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'Missing search query (q parameter)'}), 400

    collection = request.args.get('collection')
    year = request.args.get('year')
    sort = request.args.get('sort', 'downloads')
    rows = min(int(request.args.get('rows', 25)), 100)

    try:
        results = search_archive(query, collection=collection, year=year, rows=rows)

        # Sort results
        if sort == 'rating':
            results.sort(key=lambda x: x.get('avg_rating', 0), reverse=True)
        elif sort == 'date':
            results.sort(key=lambda x: x.get('date', ''))

        return jsonify({
            'query': query,
            'collection': collection,
            'year': year,
            'results': results,
            'count': len(results),
            'available': True,
        })
    except Exception as e:
        logger.error(f"Archive search failed: {e}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500


@app.route('/api/archive/collections', methods=['GET'])
def archive_collections():
    """
    List known Archive.org live music collections (bands).
    Returns collection IDs and friendly names for the search UI.
    """
    if not ARCHIVE_PIPELINE_AVAILABLE:
        return jsonify({'error': 'Archive.org pipeline not available', 'available': False}), 500

    try:
        from archive_pipeline import COLLECTIONS
        collections = [
            {'id': cid, 'name': name}
            for name, cid in COLLECTIONS.items()
        ]
        # Sort alphabetically by name
        collections.sort(key=lambda x: x['name'])
        return jsonify({
            'collections': collections,
            'count': len(collections),
            'available': True,
        })
    except Exception as e:
        logger.error(f"Failed to get collections: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/archive/show/<identifier>', methods=['GET'])
def archive_show_details(identifier):
    """
    Get full details for an Archive.org show, including track list.

    Path param:
        identifier: Archive.org item identifier (e.g., "gd1977-05-08.sbd.hicks.4982.sbeok.shnf")

    Query params:
        format: Preferred audio format — "mp3", "flac", "ogg" (default: "mp3")

    Returns show metadata, track list with download URLs, and extracted setlist.
    """
    if not ARCHIVE_PIPELINE_AVAILABLE:
        return jsonify({'error': 'Archive.org pipeline not available', 'available': False}), 500

    prefer_format = request.args.get('format', 'mp3')

    try:
        result = get_show_info(identifier)
        if 'error' in result:
            return jsonify(result), 404
        result['available'] = True
        return jsonify(result)
    except Exception as e:
        logger.error(f"Archive show details failed for {identifier}: {e}")
        return jsonify({'error': f'Failed to get show details: {str(e)}'}), 500


@app.route('/api/archive/process', methods=['POST'])
def archive_process_track():
    """
    Process an Archive.org track through StemScribe's full pipeline.

    Body (JSON):
        url: Direct download URL or archive.org page URL
        identifier: Archive.org identifier (optional, extracted from URL if missing)
        filename: Specific filename to process (optional)
        skills: List of skills to enable (optional)

    This feeds the track into the same pipeline as /api/url — yt-dlp handles
    archive.org natively, so we just forward to the URL processing endpoint.
    """
    if not ARCHIVE_PIPELINE_AVAILABLE:
        return jsonify({'error': 'Archive.org pipeline not available', 'available': False}), 500

    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing request body'}), 400

    url = data.get('url', '').strip()
    identifier = data.get('identifier', '').strip()
    filename = data.get('filename', '').strip()

    # Build the URL if we have identifier + filename but no URL
    if not url and identifier and filename:
        url = f"https://archive.org/download/{identifier}/{filename}"
    elif not url and identifier:
        url = f"https://archive.org/details/{identifier}"

    if not url:
        return jsonify({'error': 'Provide url, or identifier + filename'}), 400

    # Get processing options
    skills = data.get('skills', [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',') if s.strip()]

    hq_vocals = data.get('hq_vocals', False)
    vocal_focus = data.get('vocal_focus', False)
    enhance_stems = data.get('enhance_stems', False)
    stereo_split = data.get('stereo_split', False)
    gp_tabs = data.get('gp_tabs', True)
    chord_detection = data.get('chord_detection', True)
    mdx_model = data.get('mdx_model', False)
    ensemble_mode = data.get('ensemble', False)

    # Create job
    job_id = str(uuid.uuid4())[:8]
    display_name = filename or identifier or 'Archive.org track'
    job = ProcessingJob(job_id, display_name, source_url=url, skills=skills)
    jobs[job_id] = job

    # Store archive metadata
    job.metadata['source'] = 'archive.org'
    job.metadata['archive_identifier'] = identifier
    if filename:
        job.metadata['archive_filename'] = filename

    # Process in background thread (same pipeline as /api/url)
    thread = threading.Thread(
        target=process_url,
        args=(job, url),
        kwargs={
            'hq_vocals': hq_vocals,
            'vocal_focus': vocal_focus,
            'enhance_stems': enhance_stems,
            'stereo_split': stereo_split,
            'gp_tabs': gp_tabs,
            'chord_detection': chord_detection,
            'mdx_model': mdx_model,
            'ensemble_mode': ensemble_mode,
        }
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Processing Archive.org track',
        'filename': display_name,
        'source': 'archive.org',
        'identifier': identifier,
        'skills': skills,
    })


@app.route('/api/archive/batch', methods=['POST'])
def archive_batch_process():
    """
    Batch-process multiple tracks from an Archive.org show.

    Body (JSON):
        identifier: Archive.org show identifier
        tracks: List of filenames to process (optional — processes all if omitted)
        format: Preferred audio format (default: "mp3")
        skills: Skills to enable for all tracks

    Returns list of job IDs, one per track.
    """
    if not ARCHIVE_PIPELINE_AVAILABLE:
        return jsonify({'error': 'Archive.org pipeline not available', 'available': False}), 500

    data = request.get_json()
    if not data or 'identifier' not in data:
        return jsonify({'error': 'Missing identifier'}), 400

    identifier = data['identifier']
    requested_tracks = data.get('tracks', [])
    prefer_format = data.get('format', 'mp3')
    skills = data.get('skills', [])

    try:
        pipeline = get_archive_pipeline()
        all_tracks = pipeline.get_show_tracks(identifier, prefer_format=prefer_format)

        if not all_tracks:
            return jsonify({'error': f'No audio tracks found for {identifier}'}), 404

        # Filter to requested tracks if specified
        if requested_tracks:
            all_tracks = [t for t in all_tracks if t.filename in requested_tracks]
            if not all_tracks:
                return jsonify({'error': 'None of the requested tracks were found'}), 404

        # Create a job for each track
        job_ids = []
        for track in all_tracks:
            url = track.download_url
            job_id = str(uuid.uuid4())[:8]
            display_name = track.title or track.filename
            job = ProcessingJob(job_id, display_name, source_url=url, skills=skills)
            jobs[job_id] = job
            job.metadata['source'] = 'archive.org'
            job.metadata['archive_identifier'] = identifier
            job.metadata['archive_filename'] = track.filename
            job.metadata['archive_track_number'] = track.track_number

            thread = threading.Thread(
                target=process_url,
                args=(job, url),
                kwargs={
                    'gp_tabs': data.get('gp_tabs', True),
                    'chord_detection': data.get('chord_detection', True),
                }
            )
            thread.daemon = True
            thread.start()

            job_ids.append({
                'job_id': job_id,
                'track': track.filename,
                'title': display_name,
                'track_number': track.track_number,
            })

        return jsonify({
            'identifier': identifier,
            'jobs': job_ids,
            'total_tracks': len(job_ids),
            'message': f'Processing {len(job_ids)} tracks from {identifier}',
        })

    except Exception as e:
        logger.error(f"Archive batch process failed: {e}")
        return jsonify({'error': f'Batch processing failed: {str(e)}'}), 500


@app.route('/api/cleanup', methods=['POST'])
def cleanup_old_files():
    """Clean up old stem files to save disk space"""
    data = request.get_json() or {}
    max_age_days = data.get('max_age_days', 7)

    result = cleanup_old_stems(OUTPUT_DIR, max_age_days=max_age_days)
    return jsonify({
        'status': 'cleaned',
        'deleted_files': result['deleted'],
        'freed_mb': result['freed_mb']
    })


if __name__ == '__main__':
    logger.info("Starting StemScribe API server...")
    logger.info(f"yt-dlp available: {shutil.which('yt-dlp') is not None}")

    # Load saved jobs from library
    loaded = load_all_jobs_from_disk()
    logger.info(f"📚 Library: {loaded} songs available")

    # debug=False to avoid Flask reloader crash with samplerate on Apple Silicon
    port = int(os.environ.get('PORT', 5555))
    app.run(host='0.0.0.0', port=port, debug=False)
