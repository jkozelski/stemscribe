"""
Core API routes — upload, url, status, health, download, jobs, cleanup, skills, models, quality.
"""

import os
from urllib.parse import quote as _url_quote
import re
import io
import uuid
import shutil
import zipfile
import subprocess
import threading
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file, g, make_response

from models.job import (
    ProcessingJob, jobs, get_job, OUTPUT_DIR, UPLOAD_DIR,
)
from processing.pipeline import process_audio, process_url
from services.url_resolver import (
    is_supported_url, is_streaming_url,
    get_spotify_track_info, get_apple_music_track_info, search_youtube_for_song,
    validate_url_no_ssrf as _validate_url_no_ssrf,
)

from auth.middleware import auth_required
from middleware.validation import (
    validate_job_id as _validate_job_id_v2,
    validate_file_upload,
    sanitize_text,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


# ============ VALIDATION HELPERS ============

def _validate_job_id(job_id: str) -> bool:
    """Validate job_id is a safe hex string (UUID prefix)."""
    return _validate_job_id_v2(job_id)


def _safe_path(base_dir: Path, untrusted_path: str) -> Path:
    """Resolve a path and ensure it stays within base_dir (prevents path traversal)."""
    resolved = (base_dir / untrusted_path).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ValueError(f"Path traversal detected: {untrusted_path}")
    return resolved


# ============ HEALTH ============

@api_bp.route('/api/config', methods=['GET'])
def get_config():
    """Public config endpoint — exposes non-secret settings to the frontend."""
    return jsonify({
        'google_client_id': os.environ.get('GOOGLE_CLIENT_ID', ''),
    })


@api_bp.route('/api/health', methods=['GET'])
def health():
    from dependencies import ENSEMBLE_SEPARATOR_AVAILABLE, _gpu_manager

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
        except Exception:
            ensemble_info = {'available': True}
    elif ENSEMBLE_SEPARATOR_AVAILABLE:
        ensemble_info = {'available': True}

    return jsonify({
        'status': 'ok',
        'service': 'StemScriber API',
        'yt_dlp_available': ytdlp_available,
        'ensemble_separator': ensemble_info,
        'separation_modes': ['standard', 'mdx'] + (['ensemble'] if ENSEMBLE_SEPARATOR_AVAILABLE else [])
    })


# ============ SKILLS ============

@api_bp.route('/api/skills', methods=['GET'])
def list_skills():
    """List available enhancement skills"""
    from dependencies import SKILLS_AVAILABLE
    if not SKILLS_AVAILABLE:
        return jsonify({'skills': [], 'available': False})

    from dependencies import get_all_skills
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


# ============ UPLOAD ============

@api_bp.route('/api/upload', methods=['POST'])
@auth_required(optional=True)
def upload_audio():
    """Upload an audio file for processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Validate file (type, size, MIME, filename)
    is_valid, file_error = validate_file_upload(file)
    if not is_valid:
        return jsonify({'error': file_error}), 400

    # Get selected skills from form data
    skills = request.form.getlist('skills')
    if not skills and request.form.get('skills'):
        skills = [s.strip() for s in request.form.get('skills').split(',') if s.strip()]

    # Check for processing options
    enhance_stems = request.form.get('enhance_stems', 'false').lower() == 'true'
    stereo_split = request.form.get('stereo_split', 'false').lower() == 'true'
    gp_tabs = request.form.get('gp_tabs', 'true').lower() == 'true'
    chord_detection = request.form.get('chord_detection', 'true').lower() == 'true'
    mdx_model = request.form.get('mdx_model', 'false').lower() == 'true'
    ensemble_mode = request.form.get('ensemble', 'false').lower() == 'true'

    # Determine user plan
    plan = request.form.get('plan', 'free')

    # Create job with skills
    job_id = str(uuid.uuid4())
    job = ProcessingJob(job_id, file.filename, skills=skills)
    job.metadata['plan'] = plan

    # Tag job with owner
    job.user_id = str(g.current_user.id) if getattr(g, 'current_user', None) else None
    session_id = request.cookies.get('session_id') or str(uuid.uuid4())
    job.session_id = session_id

    jobs[job_id] = job

    # Save uploaded file (sanitize filename to prevent path traversal)
    from werkzeug.utils import secure_filename
    safe_name = secure_filename(file.filename) or 'upload.wav'
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(exist_ok=True)
    audio_path = job_upload_dir / safe_name
    file.save(str(audio_path))

    # ── Extract title/artist from ID3 tags or filename ──
    try:
        from tinytag import TinyTag
        tag = TinyTag.get(str(audio_path))
        if tag.title:
            job.metadata['title'] = tag.title.strip()
        if tag.artist:
            job.metadata['artist'] = tag.artist.strip()
        if tag.album:
            job.metadata['album'] = tag.album.strip()
        if tag.duration:
            job.metadata['duration'] = int(tag.duration)
    except Exception:
        pass  # ID3 tags not available

    # Fallback: parse filename if no ID3 title found
    if not job.metadata.get('title'):
        raw = file.filename or safe_name
        # Strip extension
        stem = raw.rsplit('.', 1)[0] if '.' in raw else raw
        import re as _re
        # Strip leading track numbers (01, 02, 1-, 01-, etc.) BEFORE the artist-title split,
        # otherwise "05 - Alright.mp3" gets parsed as artist="05", title="Alright".
        stem = _re.sub(r'^[\d]{1,3}[\s._-]+', '', stem)
        # Try "Artist - Title" split on the track-number-stripped stem
        if ' - ' in stem:
            parts = stem.split(' - ', 1)
            artist_candidate = parts[0].replace('_', ' ').strip()
            # Guard: don't accept a pure-digit artist (leftover track number edge case)
            if artist_candidate and not artist_candidate.isdigit() and not job.metadata.get('artist'):
                job.metadata['artist'] = artist_candidate
            name = parts[1].replace('_', ' ').strip()
        else:
            name = stem.replace('_', ' ').replace('-', ' ').strip()
        # Title-case it
        job.metadata['title'] = name.title() if name == name.lower() or name == name.upper() else name

    mode_str = 'ENSEMBLE' if ensemble_mode else ('MDX' if mdx_model else 'standard')
    logger.info(f"Created job {job_id} for file {file.filename} - title: {job.metadata.get('title')}, artist: {job.metadata.get('artist')}, mode: {mode_str}, plan: {plan}")

    # Start processing in background thread
    thread = threading.Thread(target=process_audio, args=(job, audio_path, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode))
    thread.daemon = True
    thread.start()

    resp = make_response(jsonify({
        'job_id': job_id,
        'message': 'Processing started',
        'filename': file.filename,
        'skills': skills
    }))
    if not request.cookies.get('session_id'):
        resp.set_cookie('session_id', session_id, httponly=True, max_age=86400, samesite='Lax')
    return resp


# ============ URL PROCESSING ============

@api_bp.route('/api/url', methods=['POST'])
@auth_required(optional=True)
def process_url_endpoint():
    """Process audio from a URL (YouTube, Spotify, Apple Music, etc.)"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url'].strip()

    # Validate URL format and block SSRF targets
    if not url.startswith(('http://', 'https://')):
        return jsonify({'error': 'Invalid URL format'}), 400
    if not _validate_url_no_ssrf(url):
        return jsonify({'error': 'URL not allowed (local/private network addresses are blocked)'}), 400
    if len(url) > 2048:
        return jsonify({'error': 'URL too long'}), 400

    # Check if yt-dlp is available
    if not shutil.which('yt-dlp'):
        return jsonify({
            'error': 'yt-dlp not installed. Run: brew install yt-dlp'
        }), 500

    # Spotify/Apple Music/Tidal DRM circumvention path DISABLED —
    # converting streaming URLs to YouTube downloads is DMCA §1201 anti-circumvention
    # and cannot be covered by the upload consent attestation. Permanently off.
    streaming_service = is_streaming_url(url)
    if streaming_service:
        return jsonify({
            'error': f'{streaming_service.replace("_", " ").title()} URLs are not supported. Upload an audio file you own, or paste a Bandcamp, SoundCloud, or Archive.org URL.'
        }), 400

    if not is_supported_url(url):
        return jsonify({
            'error': 'Unsupported URL. Supported: SoundCloud, Bandcamp, Vimeo, Archive.org'
        }), 400

    # Check URL cache before doing any processing
    from url_cache import normalize_url, check_cache, clone_job as cache_clone_job

    job_id = str(uuid.uuid4())
    cached_job_id = check_cache(url)
    if cached_job_id:
        cloned = cache_clone_job(cached_job_id, job_id)
        if cloned:
            # Tag with owner
            cloned.user_id = str(g.current_user.id) if getattr(g, 'current_user', None) else None
            session_id = request.cookies.get('session_id') or str(uuid.uuid4())
            cloned.session_id = session_id
            cloned.source_url = original_url
            jobs[job_id] = cloned
            logger.info(f"Cache hit for {url} -> cloned from {cached_job_id}")
            resp = make_response(jsonify({
                'job_id': job_id,
                'message': 'Instant results (previously processed)',
                'cached': True,
                'filename': cloned.filename,
                'url': url,
                'source': streaming_service or 'direct',
                'track_info': track_info
            }))
            if not request.cookies.get('session_id'):
                resp.set_cookie('session_id', session_id, httponly=True, max_age=86400, samesite='Lax')
            return resp

    # Get selected skills from request data
    skills = data.get('skills', [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',') if s.strip()]

    # Check for processing options
    enhance_stems = data.get('enhance_stems', False)
    stereo_split = data.get('stereo_split', False)
    gp_tabs = data.get('gp_tabs', True)
    chord_detection = data.get('chord_detection', True)
    mdx_model = data.get('mdx_model', False)
    ensemble_mode = data.get('ensemble', False)

    # Determine user plan
    plan = data.get('plan', 'free')

    # Create job with skills
    job = ProcessingJob(job_id, 'Downloading...', source_url=original_url, skills=skills)
    job.metadata['plan'] = plan

    # Tag job with owner
    job.user_id = str(g.current_user.id) if getattr(g, 'current_user', None) else None
    session_id = request.cookies.get('session_id') or str(uuid.uuid4())
    job.session_id = session_id

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
    thread = threading.Thread(target=process_url, args=(job, url, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode))
    thread.daemon = True
    thread.start()

    resp = make_response(jsonify({
        'job_id': job_id,
        'message': 'Download and processing started',
        'url': url,
        'source': streaming_service or 'direct',
        'track_info': track_info
    }))
    if not request.cookies.get('session_id'):
        resp.set_cookie('session_id', session_id, httponly=True, max_age=86400, samesite='Lax')
    return resp


# ============ STATUS ============

@api_bp.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the status of a processing job.

    Query params:
        slim=1  Return only {status, progress, stage, error} (~80 bytes)
                instead of the full job dict (~5-10KB). Use this for polling
                during processing; fetch full status once when status='completed'.
    """
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Slim mode: return only the fields that change during processing
    if request.args.get('slim') == '1':
        slim_data = {
            'status': job.status,
            'progress': job.progress,
            'stage': job.stage,
            'error': job.error,
        }
        etag = f'"{job.status}-{job.progress}-{hash(job.stage or "")}"'
        if request.headers.get('If-None-Match') == etag:
            return '', 304
        resp = jsonify(slim_data)
        resp.headers['ETag'] = etag
        resp.headers['Cache-Control'] = 'no-cache'
        return resp

    # Full status (used when job completes or for initial load)
    logger.debug(f"Full status request for {job_id}: stems={list(job.stems.keys()) if job.stems else 'NONE'}")
    data = job.to_dict()
    # Proxy YouTube thumbnails to avoid hotlink-protection 403s
    meta = data.get("metadata", {})
    if meta.get("thumbnail") and "ytimg.com" in meta["thumbnail"]:
        meta["thumbnail"] = "/api/thumbnail?url=" + _url_quote(meta["thumbnail"], safe="")
    return jsonify(data)


# ============ AVAILABLE MODELS ============

@api_bp.route('/api/available-models', methods=['GET'])
def get_available_models():
    """Get list of available separation and transcription models."""
    from dependencies import (
        ENHANCED_SEPARATOR_AVAILABLE, STEREO_SPLITTER_AVAILABLE,
        GUITAR_SEPARATOR_AVAILABLE, OAF_DRUM_TRANSCRIBER_AVAILABLE,
        OAF_AVAILABLE, DRUM_TRANSCRIBER_V2_AVAILABLE,
        ENHANCED_TRANSCRIBER_AVAILABLE, MODEL_MANAGER_AVAILABLE,
    )

    models = {}

    # Separation models
    if ENHANCED_SEPARATOR_AVAILABLE:
        from dependencies import SEPARATOR_MODELS
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
            from dependencies import list_available_models
            pretrained_status = list_available_models()
        except Exception as e:
            logger.warning(f"Could not get pretrained model status: {e}")

    return jsonify({
        'enhanced_separator_available': ENHANCED_SEPARATOR_AVAILABLE,
        'stereo_splitter_available': STEREO_SPLITTER_AVAILABLE,
        'guitar_separator_available': GUITAR_SEPARATOR_AVAILABLE,
        'oaf_drums_available': OAF_DRUM_TRANSCRIBER_AVAILABLE and OAF_AVAILABLE,
        'drum_transcriber_v2_available': DRUM_TRANSCRIBER_V2_AVAILABLE,
        'enhanced_transcriber_available': ENHANCED_TRANSCRIBER_AVAILABLE,
        'model_manager_available': MODEL_MANAGER_AVAILABLE,
        'models': models,
        'pretrained': pretrained_status
    })


# ============ QUALITY ============

@api_bp.route('/api/quality/<job_id>', methods=['GET'])
def get_transcription_quality(job_id):
    """Get transcription quality metrics for a job."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    from dependencies import ENHANCED_TRANSCRIBER_AVAILABLE, DRUM_TRANSCRIBER_V2_AVAILABLE

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


# ============ DOWNLOAD ============

@api_bp.route('/api/download/<job_id>/thumbnail', methods=['GET'])
def download_thumbnail(job_id):
    """Serve a job's thumbnail image."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    thumb_path = OUTPUT_DIR / job_id / 'thumbnail.jpg'
    if not thumb_path.exists():
        thumb_path = OUTPUT_DIR / job_id / 'thumbnail.png'
    if not thumb_path.exists():
        return jsonify({'error': 'No thumbnail'}), 404
    from flask import send_file
    return send_file(str(thumb_path), mimetype='image/jpeg')


@api_bp.route('/api/download/<job_id>/<file_type>/<filename>', methods=['GET'])
def download_file(job_id, file_type, filename):
    """Download a stem or MIDI file"""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    allowed_file_types = ('stem', 'enhanced', 'midi', 'musicxml', 'gp', 'guitarpro')
    if file_type not in allowed_file_types:
        return jsonify({'error': f'Invalid file type. Allowed: {allowed_file_types}'}), 400
    if '..' in filename or '/' in filename:
        return jsonify({'error': 'Invalid filename'}), 400

    logger.info(f"Download request: {job_id}/{file_type}/{filename}")
    try:
        job = get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        logger.info(f"  Job loaded: {job.filename}")

        if file_type == 'stem':
            if filename not in job.stems:
                available = list(job.stems.keys())
                return jsonify({'error': f'Stem not found. Available: {available}'}), 404
            file_path = job.stems[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'Stem file missing from disk: {file_path}'}), 404
            return send_file(file_path, mimetype='audio/wav', conditional=True)

        elif file_type == 'enhanced':
            if filename not in job.enhanced_stems:
                available = list(job.enhanced_stems.keys())
                return jsonify({'error': f'Enhanced stem not found. Available: {available}'}), 404
            file_path = job.enhanced_stems[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'Enhanced stem file missing from disk: {file_path}'}), 404
            return send_file(file_path, mimetype='audio/wav', conditional=True)

        elif file_type == 'midi':
            if filename not in job.midi_files:
                available = list(job.midi_files.keys())
                return jsonify({'error': f'MIDI file not found. Available: {available}'}), 404
            file_path = job.midi_files[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'MIDI file missing from disk: {file_path}'}), 404
            return send_file(file_path, as_attachment=True)

        elif file_type == 'musicxml':
            logger.info(f"  MusicXML request for '{filename}'")
            logger.info(f"     Available: {list(job.musicxml_files.keys()) if job.musicxml_files else 'NONE'}")
            if filename not in job.musicxml_files:
                available = list(job.musicxml_files.keys())
                logger.warning(f"  MusicXML '{filename}' not found. Available: {available}")
                return jsonify({'error': f'MusicXML not found. Available: {available}'}), 404
            file_path = job.musicxml_files[filename]
            logger.info(f"     Path: {file_path}")
            if not Path(file_path).exists():
                logger.error(f"  File missing: {file_path}")
                return jsonify({'error': f'MusicXML file missing from disk: {file_path}'}), 404
            logger.info(f"  Sending file: {Path(file_path).name} ({Path(file_path).stat().st_size} bytes)")
            return send_file(file_path, as_attachment=True, mimetype='application/xml')

        elif file_type in ('gp', 'guitarpro'):
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


@api_bp.route('/api/download/<job_id>/substem/<skill_id>/<filename>', methods=['GET'])
def download_substem(job_id, skill_id, filename):
    """Download a skill-generated sub-stem"""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    if '..' in filename or '/' in filename or '..' in skill_id or '/' in skill_id:
        return jsonify({'error': 'Invalid filename or skill ID'}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if skill_id not in job.sub_stems:
        return jsonify({'error': f'Skill {skill_id} not found in job'}), 404

    for sub_stem_name, rel_path in job.sub_stems[skill_id].items():
        if os.path.basename(rel_path) == filename or sub_stem_name == filename.replace('.wav', ''):
            try:
                full_path = _safe_path(OUTPUT_DIR, f"{job_id}/stems/{rel_path}")
            except ValueError:
                return jsonify({'error': 'Invalid path'}), 400
            if full_path.exists():
                return send_file(str(full_path), as_attachment=True)

    return jsonify({'error': 'Sub-stem file not found'}), 404


# ============ ZIP DOWNLOAD ============

@api_bp.route('/api/download/<job_id>/zip', methods=['GET'])
@auth_required(optional=True)
def download_zip(job_id):
    """Download all stems, MIDI, GP, and chord chart as a single ZIP file."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.status != 'completed':
        return jsonify({'error': 'Job is not completed yet'}), 400

    # Build a safe download filename from metadata
    artist = job.metadata.get('artist', '').strip()
    title = job.metadata.get('title', '').strip()
    if not title:
        # Try to parse "Artist - Title" from the raw title or filename
        raw = job.metadata.get('title', '') or job.filename or 'StemScriber Export'
        if ' - ' in raw:
            parts = raw.split(' - ', 1)
            artist = parts[0].strip()
            title = parts[1].strip()
        else:
            title = raw
    safe_title = re.sub(r'[^\w\s\-]', '', f"{artist} - {title}" if artist else title).strip()
    if not safe_title:
        safe_title = 'StemScriber Export'
    zip_filename = f"{safe_title} ( StemScriber).zip"

    # Create ZIP in memory
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Stems (WAV)
        for stem_name, stem_path in job.stems.items():
            p = Path(stem_path)
            if p.exists():
                zf.write(str(p), f"stems/{stem_name}.wav")

        # Enhanced stems
        for stem_name, stem_path in (job.enhanced_stems or {}).items():
            p = Path(stem_path)
            if p.exists():
                zf.write(str(p), f"stems_enhanced/{stem_name}.wav")

        # MIDI files
        for midi_name, midi_path in (job.midi_files or {}).items():
            p = Path(midi_path)
            if p.exists():
                ext = p.suffix or '.mid'
                zf.write(str(p), f"midi/{midi_name}{ext}")

        # Guitar Pro files
        for gp_name, gp_path in (job.gp_files or {}).items():
            p = Path(gp_path)
            if p.exists():
                ext = p.suffix or '.gp5'
                zf.write(str(p), f"guitarpro/{gp_name}{ext}")

        # MusicXML files
        for mx_name, mx_path in (job.musicxml_files or {}).items():
            p = Path(mx_path)
            if p.exists():
                ext = p.suffix or '.musicxml'
                zf.write(str(p), f"musicxml/{mx_name}{ext}")

        # Chord chart JSON
        chart_path = OUTPUT_DIR / job_id / 'chord_chart.json'
        if chart_path.exists():
            zf.write(str(chart_path), 'chord_chart.json')

    buffer.seek(0)
    logger.info(f"ZIP download for job {job_id}: {zip_filename} ({buffer.getbuffer().nbytes} bytes)")

    return send_file(
        buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename,
    )


# ============ MP3 STEM DOWNLOAD ============

@api_bp.route('/api/download/<job_id>/stem/<stem_name>/mp3', methods=['GET'])
@auth_required(optional=True)
def download_stem_mp3(job_id, stem_name):
    """Download a stem converted to MP3 (cached on disk next to WAV)."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    if '..' in stem_name or '/' in stem_name:
        return jsonify({'error': 'Invalid stem name'}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Check enhanced stems first, then regular
    wav_path = None
    if stem_name in (job.enhanced_stems or {}):
        wav_path = job.enhanced_stems[stem_name]
    elif stem_name in (job.stems or {}):
        wav_path = job.stems[stem_name]

    if not wav_path or not Path(wav_path).exists():
        return jsonify({'error': 'Stem not found'}), 404

    # Determine bitrate based on user plan
    bitrate = '128k'  # default for free
    user = getattr(g, 'current_user', None)
    if user:
        plan = getattr(user, 'plan', None) or (user.get('plan') if isinstance(user, dict) else None)
        if plan in ('pro', 'premium'):
            bitrate = '320k'

    # Check for cached MP3
    mp3_path = Path(wav_path).with_suffix(f'.{bitrate.replace("k","")}.mp3')
    if not mp3_path.exists():
        # Convert WAV → MP3 using ffmpeg
        if not shutil.which('ffmpeg'):
            return jsonify({'error': 'ffmpeg not available on server'}), 500
        try:
            result = subprocess.run(
                ['ffmpeg', '-y', '-i', str(wav_path), '-codec:a', 'libmp3lame', '-b:a', bitrate, str(mp3_path)],
                capture_output=True, timeout=120,
            )
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr.decode(errors='replace')[:500]}")
                return jsonify({'error': 'MP3 conversion failed'}), 500
        except subprocess.TimeoutExpired:
            return jsonify({'error': 'MP3 conversion timed out'}), 500

    # Build a nice download name
    artist = job.metadata.get('artist', '').strip()
    title = job.metadata.get('title', '').strip() or job.filename or 'stem'
    display = f"{artist} - {title}" if artist else title
    safe_display = re.sub(r'[^\w\s\-]', '', display).strip() or 'stem'
    download_name = f"{safe_display} ({stem_name}).mp3"

    return send_file(
        str(mp3_path),
        mimetype='audio/mpeg',
        as_attachment=True,
        download_name=download_name,
    )


# ============ JOBS LIST ============

@api_bp.route('/api/peaks/<job_id>/<stem_name>', methods=['GET'])
def get_peaks(job_id, stem_name):
    """Return waveform peaks for a stem (used for visual rendering without loading audio)."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Find the stem file
    file_path = None
    if stem_name in (job.enhanced_stems or {}):
        file_path = job.enhanced_stems[stem_name]
    elif stem_name in (job.stems or {}):
        file_path = job.stems[stem_name]

    if not file_path or not Path(file_path).exists():
        return jsonify({'error': 'Stem not found'}), 404

    # Check for cached peaks
    peaks_path = Path(file_path).with_suffix('.peaks.json')
    if peaks_path.exists():
        return send_file(peaks_path, mimetype='application/json')

    # Generate peaks
    try:
        import soundfile as sf
        import numpy as np

        data, sr = sf.read(str(file_path), dtype='float32')
        if data.ndim > 1:
            data = data.mean(axis=1)  # Mix to mono

        num_peaks = 200
        chunk_size = max(1, len(data) // num_peaks)
        peaks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            peaks.append(float(np.max(np.abs(chunk))))

        result = {'peaks': peaks, 'duration': len(data) / sr}

        # Cache peaks
        import json
        peaks_path.write_text(json.dumps(result))

        return jsonify(result)
    except Exception as e:
        logger.error(f"Peaks generation failed: {e}")
        return jsonify({'error': str(e)}), 500


@api_bp.route('/api/jobs', methods=['GET'])
@auth_required(optional=True)
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': [job.to_dict() for job in jobs.values()]
    })


# ============ RAG CHORD RECALL ============

@api_bp.route('/api/chord-recall', methods=['POST'])
def chord_recall():
    """RAG chord recall — DISABLED 2026-04-21 per Jeff.
    Index still references 15,000+ scraped songs from pre-Apr-16 cleanup;
    legal cleanup gap per Alexandra Mayo April 10 call. Endpoint returns 410 Gone
    so any stale clients fail fast instead of getting wrong-song results.
    """
    return jsonify({'match': False, 'disabled': True, 'reason': 'RAG chord recall disabled'}), 410


# ============ MANUAL CHORD CHART ============

@api_bp.route('/api/chord-chart/<job_id>', methods=['GET', 'PUT'])
def get_chord_chart(job_id):
    """Serve or update manual chord chart JSON for a job."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    import json
    chart_path = OUTPUT_DIR / job_id / 'chord_chart.json'
    if request.method == 'PUT':
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No data'}), 400
        chart_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chart_path, 'w') as f:
            json.dump(data, f, indent=2)
        return jsonify({'status': 'saved'})
    if chart_path.exists():
        with open(chart_path) as f:
            return jsonify(json.load(f))
    # Fall back to auto-generated chart (saved when a manual chart already existed)
    auto_path = OUTPUT_DIR / job_id / 'chord_chart_auto.json'
    if auto_path.exists():
        with open(auto_path) as f:
            return jsonify(json.load(f))
    return jsonify({'error': 'No chord chart found'}), 404


# ============ CLEANUP ============

@api_bp.route('/api/cleanup', methods=['POST'])
@auth_required
def cleanup_old_files():
    """Clean up old stem files to save disk space"""
    from dependencies import DRIVE_AVAILABLE
    data = request.get_json() or {}
    try:
        max_age_days = max(1, min(int(data.get('max_age_days', 7)), 365))
    except (ValueError, TypeError):
        max_age_days = 7

    try:
        if DRIVE_AVAILABLE:
            from dependencies import cleanup_old_stems
        else:
            # Basic cleanup without drive integration
            def cleanup_old_stems(output_dir, max_age_days=7):
                import time
                deleted = 0
                freed = 0
                cutoff = time.time() - (max_age_days * 86400)
                for job_dir in output_dir.iterdir():
                    if job_dir.is_dir() and job_dir.stat().st_mtime < cutoff:
                        size = sum(f.stat().st_size for f in job_dir.rglob('*') if f.is_file())
                        shutil.rmtree(job_dir)
                        deleted += 1
                        freed += size
                return {'deleted': deleted, 'freed_mb': round(freed / 1024 / 1024, 1)}

        result = cleanup_old_stems(OUTPUT_DIR, max_age_days=max_age_days)
        return jsonify({
            'status': 'cleaned',
            'deleted_files': result['deleted'],
            'freed_mb': result['freed_mb']
        })
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500


# ============ URL CACHE STATS ============

@api_bp.route('/api/cache/stats', methods=['GET'])
def cache_stats():
    """Return URL cache statistics — cached songs, hit counts, estimated savings."""
    from url_cache import get_cache_stats
    return jsonify(get_cache_stats())
