"""
Lyrics routes — fetch synced lyrics from LRCLIB for karaoke mode.
Supports manual lyrics overrides saved per job.
"""

import json
import os
import re
import logging
import subprocess
from pathlib import Path

import numpy as np
import requests
from flask import Blueprint, request, jsonify

from models.job import get_job, OUTPUT_DIR
from middleware.validation import validate_job_id

logger = logging.getLogger(__name__)

lyrics_bp = Blueprint("lyrics", __name__)

LRCLIB_API = "https://lrclib.net/api"
LRCLIB_TIMEOUT = 10


def _override_path(job_id):
    """Path to a job's lyrics override file."""
    return OUTPUT_DIR / job_id / 'lyrics_override.json'


def _load_override(job_id):
    """Load saved lyrics override for a job, if any."""
    path = _override_path(job_id)
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load lyrics override for {job_id}: {e}")
    return None


def _save_override(job_id, data):
    """Save lyrics override for a job."""
    path = _override_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _parse_lrc(lrc_content):
    """Parse LRC format into list of {time, text} dicts."""
    if not lrc_content:
        return []

    lines = []
    pattern = r'\[(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\](.*)'

    for line in lrc_content.split('\n'):
        line = line.strip()
        if not line:
            continue
        match = re.match(pattern, line)
        if not match:
            continue

        minutes = int(match.group(1))
        seconds = int(match.group(2))
        frac = match.group(3) or '0'
        # Normalize fractional seconds
        if len(frac) == 2:
            ms = int(frac) * 10
        elif len(frac) == 3:
            ms = int(frac)
        else:
            ms = int(frac)

        total_seconds = minutes * 60 + seconds + ms / 1000
        text = match.group(4).strip()
        lines.append({'time': round(total_seconds, 3), 'text': text})

    lines.sort(key=lambda x: x['time'])
    return lines


def _clean_title(title):
    """Strip YouTube/video junk from title for better lyrics search."""
    if not title:
        return title
    # Remove common suffixes
    title = re.sub(
        r'\s*[-–—|]\s*(Official\s*(Video|Audio|Music\s*Video|Lyric\s*Video)?|'
        r'HQ|HD|4K|Lyrics?|Audio|Full\s*Album|Remaster(ed)?|Live|'
        r'ft\.?\s*.+|feat\.?\s*.+)\s*$',
        '', title, flags=re.IGNORECASE
    )
    # Remove [bracketed] and (parenthesized) tags
    title = re.sub(r'\s*[\[\(](Official|Video|Audio|Lyrics?|HQ|HD|4K|Live|Remastered)[\]\)]\s*', '', title, flags=re.IGNORECASE)
    return title.strip()


def _extract_artist_title(display_name):
    """Try to split 'Artist - Title' from a job display name."""
    if not display_name:
        return None, None

    display_name = _clean_title(display_name)

    # Try "Artist - Title" pattern
    for sep in [' - ', ' – ', ' — ', ' | ']:
        if sep in display_name:
            parts = display_name.split(sep, 1)
            return parts[0].strip(), parts[1].strip()

    return None, display_name


def _whisper_fallback(job):
    """Transcribe vocals with Whisper when no online lyrics are found."""
    vocal_path = None
    if job.stems and 'vocals' in job.stems:
        import os
        vp = job.stems['vocals']
        if os.path.exists(vp):
            vocal_path = vp

    if not vocal_path:
        return None

    try:
        from word_timestamps import get_word_timestamps
        logger.info(f"Whisper fallback: transcribing {vocal_path}")
        word_ts = get_word_timestamps(vocal_path)
        if not word_ts:
            return None

        # Group words into natural lyric lines
        synced_lines = []
        current_words = []
        line_start = None
        MAX_LINE_DURATION = 5.0
        PAUSE_THRESHOLD = 1.5

        for i, w in enumerate(word_ts):
            if line_start is None:
                line_start = w['start']
                current_words.append(w['word'])
                continue

            prev_end = word_ts[i - 1]['end']
            gap = w['start'] - prev_end
            line_duration = w['end'] - line_start

            if gap > PAUSE_THRESHOLD or line_duration > MAX_LINE_DURATION:
                if current_words:
                    synced_lines.append({
                        'time': round(line_start, 3),
                        'text': ' '.join(current_words).strip(),
                    })
                current_words = [w['word']]
                line_start = w['start']
            else:
                current_words.append(w['word'])

        if current_words:
            synced_lines.append({
                'time': round(line_start, 3),
                'text': ' '.join(current_words).strip(),
            })

        logger.info(f"Whisper fallback: got {len(synced_lines)} lyric lines")
        return synced_lines if synced_lines else None

    except Exception as e:
        logger.warning(f"Whisper fallback failed: {e}")
        return None


@lyrics_bp.route('/api/lyrics/<job_id>', methods=['GET'])
def get_lyrics(job_id):
    """
    Fetch synced lyrics for a processed job.

    Priority: saved override > LRCLIB > Whisper fallback.

    Query params:
        artist: Override artist name
        title: Override song title
    """
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Check for saved lyrics override first
    override = _load_override(job_id)
    if override:
        logger.info(f"Using saved lyrics override for {job_id}")
        return jsonify(override)

    # Allow manual override via query params
    artist = request.args.get('artist', '').strip()
    title = request.args.get('title', '').strip()

    # Extract from job metadata if not provided
    if not artist or not title:
        job_artist = getattr(job, 'artist', None) or (job.metadata.get('artist') if hasattr(job, 'metadata') else None)
        job_title = getattr(job, 'title', None) or (job.metadata.get('title') if hasattr(job, 'metadata') else None)

        if job_artist and job_title:
            artist = artist or job_artist
            title = title or _clean_title(job_title)
        else:
            # Parse from display name
            parsed_artist, parsed_title = _extract_artist_title(job.display_name)
            artist = artist or parsed_artist or ''
            title = title or parsed_title or ''

    if not title:
        return jsonify({'error': 'Could not determine song title', 'hint': 'Pass ?artist=X&title=Y'}), 400

    # Try LRCLIB exact match first
    try:
        if artist and title:
            params = {'artist_name': artist, 'track_name': title}
            logger.info(f"LRCLIB search: '{title}' by '{artist}'")
        else:
            params = {'q': title}
            logger.info(f"LRCLIB search: '{title}'")

        resp = requests.get(f"{LRCLIB_API}/search", params=params, timeout=LRCLIB_TIMEOUT)
        resp.raise_for_status()
        results = resp.json()

        if not results:
            # Fallback: search with just title
            if artist:
                resp2 = requests.get(f"{LRCLIB_API}/search", params={'q': f"{artist} {title}"}, timeout=LRCLIB_TIMEOUT)
                resp2.raise_for_status()
                results = resp2.json()

        if not results:
            # Fallback: Whisper on vocal stem
            whisper_lyrics = _whisper_fallback(job)
            if whisper_lyrics:
                return jsonify({
                    'found': True,
                    'has_synced': True,
                    'synced_lyrics': whisper_lyrics,
                    'plain_lyrics': '\n'.join(l['text'] for l in whisper_lyrics),
                    'metadata': {
                        'title': title,
                        'artist': artist,
                    },
                    'source': 'whisper',
                })

            return jsonify({
                'found': False,
                'artist': artist,
                'title': title,
                'message': 'No lyrics found',
            }), 404

        # Pick best match (first result with synced lyrics preferred)
        best = None
        for r in results:
            if r.get('syncedLyrics'):
                best = r
                break
        if not best:
            best = results[0]

        synced = best.get('syncedLyrics')
        plain = best.get('plainLyrics')

        parsed = _parse_lrc(synced) if synced else None

        return jsonify({
            'found': True,
            'has_synced': bool(synced),
            'synced_lyrics': parsed,
            'plain_lyrics': plain,
            'metadata': {
                'title': best.get('trackName', ''),
                'artist': best.get('artistName', ''),
                'album': best.get('albumName', ''),
                'duration': best.get('duration', 0),
            },
            'source': 'lrclib.net',
        })

    except requests.RequestException as e:
        logger.error(f"LRCLIB request failed: {e}")
        return jsonify({'error': f'Lyrics service unavailable: {str(e)}'}), 502


@lyrics_bp.route('/api/lyrics/<job_id>', methods=['PUT'])
def save_lyrics(job_id):
    """Save corrected lyrics for a job (override Whisper/LRCLIB)."""
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    data = request.get_json()
    if not data or 'synced_lyrics' not in data:
        return jsonify({'error': 'Request must include synced_lyrics'}), 400

    override = {
        'found': True,
        'has_synced': True,
        'synced_lyrics': data['synced_lyrics'],
        'plain_lyrics': data.get('plain_lyrics', '\n'.join(
            l.get('text', '') for l in data['synced_lyrics']
        )),
        'metadata': data.get('metadata', {}),
        'source': 'manual',
    }

    _save_override(job_id, override)
    logger.info(f"Saved lyrics override for {job_id}: {len(data['synced_lyrics'])} lines")
    return jsonify({'status': 'saved', 'lines': len(data['synced_lyrics'])})


def _detect_vocal_onsets(audio_path, min_gap=1.0):
    """
    Analyze a vocal stem to find all singing onset times.

    Returns a list of timestamps (in seconds) where singing begins after silence.
    min_gap: minimum silence gap (seconds) between phrases to count as a new onset.
    """
    try:
        result = subprocess.run([
            'ffmpeg', '-i', str(audio_path),
            '-f', 'f32le', '-ac', '1', '-ar', '22050', '-'
        ], capture_output=True, timeout=60)

        if result.returncode != 0 or len(result.stdout) == 0:
            return []

        samples = np.frombuffer(result.stdout, dtype=np.float32)
        sr = 22050

        # Compute RMS energy in 50ms windows
        window_size = int(sr * 0.05)
        num_windows = len(samples) // window_size
        rms = np.array([
            np.sqrt(np.mean(samples[i * window_size:(i + 1) * window_size] ** 2))
            for i in range(num_windows)
        ])

        # Adaptive threshold: 5x the noise floor (first 2 seconds)
        noise_floor = np.median(rms[:int(sr / window_size * 2)])
        threshold = max(noise_floor * 5, 0.005)

        # Find all onset points (transitions from below to above threshold)
        onsets = []
        in_vocal = False
        last_offset_time = -min_gap  # allow first onset

        for i in range(len(rms)):
            t = i * 0.05
            if not in_vocal and rms[i] > threshold:
                # Check 2 consecutive windows to avoid blips
                if i + 1 < len(rms) and rms[i + 1] > threshold:
                    if t - last_offset_time >= min_gap:
                        onsets.append(round(t, 2))
                    in_vocal = True
            elif in_vocal and rms[i] < threshold * 0.5:
                # Require sustained silence (3 windows below half-threshold)
                if i + 2 < len(rms) and rms[i + 1] < threshold * 0.5 and rms[i + 2] < threshold * 0.5:
                    in_vocal = False
                    last_offset_time = t

        return onsets

    except Exception as e:
        logger.error(f"Vocal onset detection failed: {e}")
        return []


@lyrics_bp.route('/api/lyrics/<job_id>/auto-sync', methods=['POST'])
def auto_sync_lyrics(job_id):
    """
    Auto-sync lyrics to the vocal stem using onset detection.

    Analyzes the vocal stem to find when each phrase starts,
    then maps the provided lyrics to those onset times.

    Body: {"lyrics": ["line 1", "line 2", ...]}
    Optional: {"lyrics": [...], "min_gap": 1.5}
    """
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    data = request.get_json()
    if not data or 'lyrics' not in data:
        return jsonify({'error': 'Request must include lyrics array'}), 400

    lyrics = data['lyrics']
    min_gap = data.get('min_gap', 1.0)

    # Find vocal stem
    vocals_path = job.stems.get('vocals_lead') or job.stems.get('vocals')
    if not vocals_path or not Path(vocals_path).exists():
        return jsonify({'error': 'No vocal stem found for this job'}), 404

    logger.info(f"Auto-syncing {len(lyrics)} lyrics lines to vocal stem for {job_id}")

    # Detect vocal onsets
    onsets = _detect_vocal_onsets(vocals_path, min_gap=min_gap)

    if not onsets:
        return jsonify({'error': 'Could not detect vocal onsets in the stem'}), 500

    logger.info(f"Detected {len(onsets)} vocal onsets: {onsets[:10]}...")

    # Filter out empty lyrics lines
    text_lines = [l for l in lyrics if l.strip()]

    # Map lyrics to onsets
    # If more lyrics than onsets, some lines share an onset (multi-line phrase)
    # If more onsets than lyrics, extra onsets are instrumental phrases (ignored)
    synced = []
    onset_idx = 0

    for i, line in enumerate(text_lines):
        if onset_idx < len(onsets):
            time = onsets[onset_idx]
            onset_idx += 1
        else:
            # Out of onsets — space remaining lines 3s apart from last
            time = synced[-1]['time'] + 3.0 if synced else 0.0

        synced.append({'time': time, 'text': line})

    # Save as override
    override = {
        'found': True,
        'has_synced': True,
        'synced_lyrics': synced,
        'plain_lyrics': '\n'.join(text_lines),
        'metadata': {
            'title': job.metadata.get('title', ''),
            'artist': job.metadata.get('artist', ''),
        },
        'source': 'vocal_onset_sync',
    }

    _save_override(job_id, override)

    logger.info(f"Auto-synced {len(synced)} lyrics lines for {job_id}")

    return jsonify({
        'status': 'synced',
        'lines': len(synced),
        'onsets_detected': len(onsets),
        'synced_lyrics': synced,
    })
