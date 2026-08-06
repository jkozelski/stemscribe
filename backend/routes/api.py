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
import hashlib
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

from auth.middleware import auth_required, authorize_job_access, forbidden_response, _ADMIN_EMAILS
from middleware.rate_limit import enforce_plan_limits, record_usage_event
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
@enforce_plan_limits
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
    gp_tabs = request.form.get('gp_tabs', 'false').lower() == 'true'  # 2026-07-23: MIDI/tabs OFF by default (opt-in)
    chord_detection = request.form.get('chord_detection', 'true').lower() == 'true'
    mdx_model = request.form.get('mdx_model', 'false').lower() == 'true'
    ensemble_mode = request.form.get('ensemble', 'false').lower() == 'true'

    # Determine user plan server-side (never trust the client field - RISK-10)
    plan = g.current_user.plan if getattr(g, 'current_user', None) else 'free'

    # Create job with skills
    job_id = str(uuid.uuid4())
    job = ProcessingJob(job_id, file.filename, skills=skills)
    job.metadata['plan'] = plan

    # Persist user-rights attestation. Per the v1.1 legal posture (2026-04-30),
    # attestation is recorded on EVERY upload (file or URL), not session-scoped.
    # The frontend modal sends attestation_at + attestation_type + the user's
    # explicit confirmation that they have rights to the audio. Stored on the
    # job for audit trail. See docs/legal-faq.md.
    attestation_at = request.form.get('attestation_at')
    if attestation_at:
        job.metadata['attestation_at'] = attestation_at
        job.metadata['attestation_type'] = request.form.get('attestation_type', 'file_upload_user_rights_confirmation')
        job.metadata['attestation_user_agent'] = request.headers.get('User-Agent', '')
        job.metadata['attestation_ip_hash'] = hashlib.sha256(
            (request.remote_addr or '').encode()
        ).hexdigest()[:16] if request.remote_addr else None
        logger.info(f"Job {job_id}: file-upload attestation recorded ({job.metadata['attestation_type']})")

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

    # ── Embedded cover art wins over the pipeline's iTunes title-match lookup,
    # which returns wrong art for indie tracks. Setting metadata['thumbnail']
    # here makes pipeline.py skip its auto-lookup (guarded by `if not thumbnail`). ──
    try:
        from mutagen import File as _MFile
        import base64 as _b64, io as _io
        _mf = _MFile(str(audio_path))
        _img = None
        if _mf is not None:
            _tags = getattr(_mf, 'tags', None)
            if _tags:
                for _k in list(_tags.keys()):
                    if _k.startswith('APIC'):
                        _img = _tags[_k].data
                        break
                if _img is None and 'covr' in _tags:
                    _img = bytes(_tags['covr'][0])
            if _img is None and getattr(_mf, 'pictures', None):
                _img = _mf.pictures[0].data
        if _img:
            try:
                from PIL import Image as _Img
                _im = _Img.open(_io.BytesIO(_img)).convert('RGB')
                _im.thumbnail((400, 400))
                _b = _io.BytesIO()
                _im.save(_b, 'JPEG', quality=82)
                _img = _b.getvalue()
            except Exception:
                pass  # fall back to raw embedded bytes if PIL fails
            job.metadata['thumbnail'] = 'data:image/jpeg;base64,' + _b64.b64encode(_img).decode('ascii')
            job.metadata['cover_source'] = 'embedded'
            logger.info('✓ using embedded cover art from upload')
    except Exception as _e:
        logger.info(f'embedded cover art extraction skipped: {_e}')

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

    # Start processing in background thread.
    # Wrapped via job_tracker.tracked so the gunicorn worker_exit hook
    # can drain in-flight jobs before a SIGTERM-driven restart, preventing
    # the orphan-at-60% bug we hit 4x on 2026-05-26.
    from processing.job_tracker import tracked as _tracked
    thread = threading.Thread(
        target=_tracked,
        args=(job.job_id, process_audio, job, audio_path, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode),
    )
    thread.daemon = True
    thread.start()

    record_usage_event(
        user=getattr(g, 'current_user', None),
        ip_hash=getattr(g, 'ip_hash', None),
        job_id=None,  # in-memory job not in DB jobs table; FK violates otherwise
        action='separation',
    )

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
@enforce_plan_limits
def process_url_endpoint():
    """Process audio from a URL (YouTube, Spotify, Apple Music, etc.)"""
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({'error': 'No URL provided'}), 400

    url = data['url'].strip()
    # `original_url` is referenced later for cache cloning + job creation. It used
    # to be set inside the streaming-service resolver (Spotify→YouTube), which is
    # now disabled. For all currently-supported URL sources the original IS the
    # final URL, so default it here.
    original_url = url
    track_info = None

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

    # YouTube / Dailymotion / Mixcloud — deliberately excluded per the v1.1
    # legal posture review (2026-04-30). These platforms have a major-label-mixed
    # UGC profile that fits the MP3.com / Napster / ReDigi plaintiff theory
    # too closely. Users who want to use audio from these sources can capture
    # it themselves and upload the resulting file via the regular file-upload
    # flow (where the rights warranty in ToS §4.3 applies).
    from services.url_resolver import is_excluded_url
    excluded = is_excluded_url(url)
    if excluded:
        return jsonify({
            'error': f'{excluded.title()} links aren\'t supported here. If you have the right to use this audio, capture it on your own device and upload the file. See /audio-capture-help.html for instructions.'
        }), 400

    if not is_supported_url(url):
        return jsonify({
            'error': 'Unsupported URL. Supported: Bandcamp, ReverbNation, SoundCloud, Audiomack, Internet Archive, Vimeo. For other sources, capture the audio yourself and upload the file.'
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

    # Determine user plan server-side (never trust the client field - RISK-10)
    plan = g.current_user.plan if getattr(g, 'current_user', None) else 'free'

    # Create job with skills
    job = ProcessingJob(job_id, 'Downloading...', source_url=original_url, skills=skills)
    job.metadata['plan'] = plan

    # Persist user-rights attestation when the user pasted a URL via the
    # YouTube fallback flow. The frontend modal sends attestation_at +
    # attestation_type before processing. Stored on the job for audit trail
    # and to support good-faith DMCA / §230 user-directed-content posture.
    # See docs/legal-faq.md ("YouTube URL acceptance").
    attestation_at = data.get('attestation_at')
    if attestation_at:
        job.metadata['attestation_at'] = attestation_at
        job.metadata['attestation_type'] = data.get('attestation_type', 'user_rights_confirmation')
        job.metadata['attestation_user_agent'] = request.headers.get('User-Agent', '')
        job.metadata['attestation_ip_hash'] = hashlib.sha256(
            (request.remote_addr or '').encode()
        ).hexdigest()[:16] if request.remote_addr else None
        logger.info(f"Job {job_id}: user-rights attestation recorded ({job.metadata['attestation_type']})")

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
        # Use ONLY the real album cover (cover_art_url from MusicBrainz/CoverArtArchive).
        # The Wikipedia thumbnail fallback was dropped 2026-05-23 because it returns
        # garbage on ambiguous artist names (e.g. "Animals" matched a wildlife
        # biology article PNG). Better to show no cover (letter-tile placeholder)
        # than a wildlife photo. Frontends read metadata['thumbnail'].
        _thumb = track_info.get('cover_art_url')
        if _thumb:
            job.metadata['thumbnail'] = _thumb

    if not job.metadata.get('thumbnail'):
        import re as _re
        _arch = _re.search(r'archive\.org/(?:details|download|stream)/([^/?#]+)', url)
        if _arch:
            job.metadata['thumbnail'] = f'https://archive.org/services/img/{_arch.group(1)}'
            job.metadata['cover_source'] = 'archive.org'

    mode_str = 'ENSEMBLE' if ensemble_mode else ('MDX' if mdx_model else 'standard')
    logger.info(f"Created job {job_id} for URL {url} - mode: {mode_str}, gp_tabs: {gp_tabs}, chord_detection: {chord_detection}")

    # Start processing in background thread.
    # Wrapped via job_tracker.tracked for the same reason as the file-upload
    # path — drain in-flight URL imports before SIGTERM kills the worker.
    from processing.job_tracker import tracked as _tracked
    thread = threading.Thread(
        target=_tracked,
        args=(job.job_id, process_url, job, url, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode),
    )
    thread.daemon = True
    thread.start()

    record_usage_event(
        user=getattr(g, 'current_user', None),
        ip_hash=getattr(g, 'ip_hash', None),
        job_id=None,  # in-memory job not in DB jobs table; FK violates otherwise
        action='separation',
    )

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
@auth_required(optional=True)
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
    if not authorize_job_access(job):
        return forbidden_response()

    # chord_chart.json is written ~halfway through the pipeline. After Phase 3
    # (2026-05-11) the post-sep slot releases at that point and MIDI/MusicXML/
    # GP continues in a daemon outside the slot — so the chart is viewable in
    # practice mode well before status flips to 'completed'. Expose this via
    # a `chord_chart_ready` flag the frontend can use to show "View chart now".
    chord_chart_ready = (OUTPUT_DIR / job_id / 'chord_chart.json').exists()

    # Slim mode: return only the fields that change during processing
    if request.args.get('slim') == '1':
        slim_data = {
            'status': job.status,
            'progress': job.progress,
            'stage': job.stage,
            'error': job.error,
            'chord_chart_ready': chord_chart_ready,
        }
        # ETag includes chord_chart_ready so the polling client sees the
        # transition from False→True even when status/progress/stage don't change.
        etag = f'"{job.status}-{job.progress}-{hash(job.stage or "")}-{int(chord_chart_ready)}"'
        if request.headers.get('If-None-Match') == etag:
            return '', 304
        resp = jsonify(slim_data)
        resp.headers['ETag'] = etag
        resp.headers['Cache-Control'] = 'no-cache'
        return resp

    # Full status (used when job completes or for initial load)
    logger.debug(f"Full status request for {job_id}: stems={list(job.stems.keys()) if job.stems else 'NONE'}")
    data = job.to_dict()
    data['chord_chart_ready'] = chord_chart_ready
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
@auth_required(optional=True)
def get_transcription_quality(job_id):
    """Get transcription quality metrics for a job."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    from dependencies import ENHANCED_TRANSCRIBER_AVAILABLE, DRUM_TRANSCRIBER_V2_AVAILABLE

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()

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
@auth_required(optional=True)
def download_thumbnail(job_id):
    """Serve a job's thumbnail image."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()
    thumb_path = OUTPUT_DIR / job_id / 'thumbnail.jpg'
    mimetype = 'image/jpeg'
    if not thumb_path.exists():
        thumb_path = OUTPUT_DIR / job_id / 'thumbnail.png'
        mimetype = 'image/png'
    if not thumb_path.exists():
        return jsonify({'error': 'No thumbnail'}), 404
    from flask import send_file
    return send_file(str(thumb_path), mimetype=mimetype)


@api_bp.route('/api/download/<job_id>/<file_type>/<filename>', methods=['GET'])
@auth_required(optional=True)
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
        if not authorize_job_access(job):
            return forbidden_response()
        logger.info(f"  Job loaded: {job.filename}")

        if file_type == 'stem':
            if filename not in job.stems:
                available = list(job.stems.keys())
                return jsonify({'error': f'Stem not found. Available: {available}'}), 404
            file_path = job.stems[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'Stem file missing from disk: {file_path}'}), 404

            # Mobile-quality variant (?q=mobile): downsample to 22kHz mono ONLY
            # for LONG songs. The iOS Web Audio crash is driven by DECODED PCM
            # memory = duration × samplerate × channels — it's a function of song
            # LENGTH, not MP3 file size (a 5MB/12min song crashes; a 35MB/2min
            # song is fine). So we gate on the stem's true DURATION via ffprobe,
            # which is bitrate-independent. Downsampling to 22kHz mono cuts
            # decoded memory ~75%. Short songs are served full-quality on mobile
            # too. Threshold tunable via MOBILE_DOWNSAMPLE_SEC env (default 240s
            # = 4 min). Mobile variant generated on first request, cached on disk.
            if request.args.get('q') == 'mobile':
                src = Path(file_path)
                threshold_sec = float(os.environ.get('MOBILE_DOWNSAMPLE_SEC', '240'))
                dur = 0.0
                if shutil.which('ffprobe'):
                    try:
                        pr = subprocess.run(
                            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                             '-of', 'default=noprint_wrappers=1:nokey=1', str(src)],
                            capture_output=True, timeout=20,
                        )
                        dur = float((pr.stdout or b'0').decode(errors='replace').strip() or 0)
                    except Exception:
                        dur = 0.0
                if dur > threshold_sec and shutil.which('ffmpeg'):
                    mobile_path = src.parent / (src.stem + '.m22.mp3')
                    if not mobile_path.exists():
                        try:
                            r = subprocess.run(
                                ['ffmpeg', '-y', '-i', str(src),
                                 '-ar', '22050', '-ac', '1', '-b:a', '96k',
                                 str(mobile_path)],
                                capture_output=True, timeout=120,
                            )
                            if r.returncode != 0:
                                logger.warning(f"mobile stem gen failed for {filename}: {r.stderr.decode(errors='replace')[:300]}")
                                mobile_path = src  # fall back to full quality
                        except subprocess.TimeoutExpired:
                            logger.warning(f"mobile stem gen timed out for {filename}")
                            mobile_path = src
                    logger.info(f"Serving mobile stem {filename} (dur {dur:.0f}s > {threshold_sec:.0f}s threshold)")
                    resp = send_file(str(mobile_path), mimetype='audio/mpeg', conditional=True)
                    resp.headers['Cache-Control'] = 'private, max-age=2592000, immutable'
                    return resp
                # Under threshold (or no ffprobe/ffmpeg) — full quality on mobile too.

            resp = send_file(file_path, mimetype='audio/wav', conditional=True)
            # Stems are immutable per-job — bake aggressive client-side caching
            # so iOS Safari doesn't re-download all 8 stems every page open.
            # 'immutable' tells the browser the file never changes (no revalidation).
            resp.headers['Cache-Control'] = 'private, max-age=2592000, immutable'
            return resp

        elif file_type == 'enhanced':
            if filename not in job.enhanced_stems:
                available = list(job.enhanced_stems.keys())
                return jsonify({'error': f'Enhanced stem not found. Available: {available}'}), 404
            file_path = job.enhanced_stems[filename]
            if not Path(file_path).exists():
                return jsonify({'error': f'Enhanced stem file missing from disk: {file_path}'}), 404
            resp = send_file(file_path, mimetype='audio/wav', conditional=True)
            resp.headers['Cache-Control'] = 'private, max-age=2592000, immutable'
            return resp

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
@auth_required(optional=True)
def download_substem(job_id, skill_id, filename):
    """Download a skill-generated sub-stem"""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    if '..' in filename or '/' in filename or '..' in skill_id or '/' in skill_id:
        return jsonify({'error': 'Invalid filename or skill ID'}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()

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
    if not authorize_job_access(job):
        return forbidden_response()
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
    if not authorize_job_access(job):
        return forbidden_response()

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


@api_bp.route('/api/master/<job_id>', methods=['GET'])
@auth_required(optional=True)
def download_master_mix(job_id):
    """Serve a single pre-mixed master MP3 (sum of the 6 PRIMARY stems) for
    mobile playback.

    Why: iOS Safari cannot decode 8 separate stems into Web Audio without
    crashing the tab (each ~85MB PCM decoded × 8 ≈ 680MB, past the iOS
    memory ceiling — confirmed 2026-05-28 on iOS 26.4 with a long Stones
    track). Mobile plays THIS one native file via a plain <audio> element
    instead. Generated on first request from the on-disk stems, then cached.

    The 6 primaries (vocals, drums, bass, guitar, piano, other) sum to the
    original mix. vocals_lead / vocals_backing are sub-splits of `vocals`
    and are EXCLUDED — including them would double-count the vocal.
    """
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()

    job_dir = OUTPUT_DIR / job_id
    master_path = job_dir / 'master.mp3'

    if not master_path.exists():
        stems_dir = job_dir / 'stems'
        if not stems_dir.exists():
            return jsonify({'error': 'Stems not found for this job'}), 404
        primary = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']
        inputs = [str(stems_dir / f'{n}.mp3') for n in primary if (stems_dir / f'{n}.mp3').exists()]
        if not inputs:
            return jsonify({'error': 'No stems available to build master mix'}), 404
        if not shutil.which('ffmpeg'):
            return jsonify({'error': 'ffmpeg not available on server'}), 500
        cmd = ['ffmpeg', '-y']
        for inp in inputs:
            cmd += ['-i', inp]
        cmd += [
            '-filter_complex',
            f'amix=inputs={len(inputs)}:normalize=0,alimiter=level_in=1:level_out=0.95:limit=0.95',
            '-c:a', 'libmp3lame', '-b:a', '192k',
            str(master_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=180)
            if result.returncode != 0:
                logger.error(f"master mix ffmpeg failed for {job_id}: {result.stderr.decode(errors='replace')[:500]}")
                return jsonify({'error': 'Master mix generation failed'}), 500
        except subprocess.TimeoutExpired:
            logger.error(f"master mix generation timed out for {job_id}")
            return jsonify({'error': 'Master mix generation timed out'}), 500
        logger.info(f"Generated master.mp3 for {job_id} from {len(inputs)} stems")

    resp = send_file(str(master_path), mimetype='audio/mpeg', as_attachment=False, conditional=True)
    resp.headers['Cache-Control'] = 'private, max-age=2592000, immutable'
    return resp


# ============ JOBS LIST ============

@api_bp.route('/api/peaks/<job_id>/<stem_name>', methods=['GET'])
@auth_required(optional=True)
def get_peaks(job_id, stem_name):
    """Return waveform peaks for a stem (used for visual rendering without loading audio)."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()

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
    """List ONLY the caller's own jobs (owner / anonymous-session / admin),
    plus public demo jobs. Previously returned ALL users' jobs to any
    anonymous caller — a data leak of filenames, user_ids, source_urls and
    chord data. Filtered via authorize_job_access, the same ownership rule
    used by every per-job route."""
    return jsonify({
        'jobs': [job.to_dict() for job in jobs.values()
                 if authorize_job_access(job)]
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
@auth_required(optional=True)
def get_chord_chart(job_id):
    """Serve or update manual chord chart JSON for a job."""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    # Authorize on the job — GET respects demo flag (anonymous users view demos),
    # PUT NEVER allows demo-anonymous edits (only owner/admin/session can mutate).
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if request.method == 'PUT':
        # Mutation requires real ownership — demo bypass is read-only.
        if not authorize_job_access(job, allow_demo=False):
            return forbidden_response()
    else:
        if not authorize_job_access(job):
            return forbidden_response()
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
    """Clean up old stem files to save disk space (ADMIN ONLY — deletes ANY
    user's job dirs older than N days, so it must not be caller-triggerable)."""
    from flask import g
    if getattr(getattr(g, 'current_user', None), 'email', None) not in _ADMIN_EMAILS:
        return forbidden_response()
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


_BAND_IMAGE_CACHE_PATH = '/opt/stemscribe/band_images.json'
_band_image_lock = threading.Lock()


@api_bp.route('/api/band-image', methods=['GET'])
def band_image():
    """Album-art image URL for a band name, via the public iTunes Search API.

    Proxied server-side because the site's connect-src CSP doesn't (and
    shouldn't) allow arbitrary client fetches; *.mzstatic.com is already in
    img-src so the returned URL renders directly. Disk-cached per name.
    """
    import json as _json
    import requests as _requests

    name = (request.args.get('name') or '').strip()[:80]
    if not name:
        return jsonify({'url': None})
    key = name.lower()

    with _band_image_lock:
        try:
            with open(_BAND_IMAGE_CACHE_PATH) as f:
                cache = _json.load(f)
        except Exception:
            cache = {}
    if key in cache:
        return jsonify({'url': cache[key]})

    url = None
    # Deezer first — real artist/band PHOTOS (Jeff: not album artwork).
    try:
        r = _requests.get(
            'https://api.deezer.com/search/artist',
            params={'q': name, 'limit': 1},
            timeout=6,
        )
        data = (r.json() or {}).get('data') or []
        if data:
            pic = data[0].get('picture_big') or data[0].get('picture_medium') or ''
            # Deezer returns a gray placeholder for unknown artists — its URL
            # has an empty md5 segment ("/images/artist//"), skip those.
            if pic and '/artist//' not in pic:
                url = pic
    except Exception as e:
        logger.warning(f"band-image Deezer lookup failed for {name!r}: {e}")
    # Fallback: iTunes album art (better than initials).
    if not url:
        try:
            r = _requests.get(
                'https://itunes.apple.com/search',
                params={'term': name, 'entity': 'album', 'limit': 1, 'media': 'music'},
                timeout=6,
            )
            results = (r.json() or {}).get('results') or []
            if results:
                art = results[0].get('artworkUrl100') or ''
                if art:
                    url = art.replace('100x100', '400x400')
        except Exception as e:
            logger.warning(f"band-image iTunes lookup failed for {name!r}: {e}")

    with _band_image_lock:
        try:
            with open(_BAND_IMAGE_CACHE_PATH) as f:
                cache = _json.load(f)
        except Exception:
            cache = {}
        cache[key] = url
        try:
            with open(_BAND_IMAGE_CACHE_PATH + '.new', 'w') as f:
                _json.dump(cache, f)
            os.replace(_BAND_IMAGE_CACHE_PATH + '.new', _BAND_IMAGE_CACHE_PATH)
        except Exception:
            pass
    return jsonify({'url': url})


_USER_TAB_EXTS = {'.gp', '.gp3', '.gp4', '.gp5', '.gpx', '.txt', '.pdf', '.xml', '.musicxml'}
_USER_TAB_MIMES = {
    '.pdf': 'application/pdf', '.txt': 'text/plain; charset=utf-8',
    '.xml': 'application/xml', '.musicxml': 'application/xml',
}


@api_bp.route('/api/user-tab/<job_id>', methods=['POST'])
@auth_required
def upload_user_tab(job_id):
    """Attach the user's own guitar tab (UG download, GP file, text) to a job.

    One tab per job — re-upload replaces. Private to the owner, stored as the
    raw file plus a JSON sidecar; job metadata is untouched (no in-memory
    jobs-dict interaction, works on completed jobs without a restart).
    """
    import json as _json
    from werkzeug.utils import secure_filename

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    if job.user_id is not None and job.user_id != str(user.id):
        return jsonify({'error': 'Forbidden'}), 403

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f or not f.filename:
        return jsonify({'error': 'Empty filename'}), 400
    name = secure_filename(f.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in _USER_TAB_EXTS:
        return jsonify({'error': f'Unsupported file type: {ext or "(none)"}'}), 415
    blob = f.read(10 * 1048576 + 1)
    if len(blob) > 10 * 1048576:
        return jsonify({'error': 'File too large (max 10 MB)'}), 413

    tab_dir = OUTPUT_DIR / job_id
    tab_dir.mkdir(parents=True, exist_ok=True)
    # clear any prior tab (different extension included)
    for old in tab_dir.glob('user_tab.*'):
        try: old.unlink()
        except OSError: pass
    (tab_dir / ('user_tab' + ext)).write_bytes(blob)
    sidecar = {'original_name': name, 'ext': ext, 'size': len(blob)}
    (tab_dir / 'user_tab.json').write_text(_json.dumps(sidecar))
    return jsonify({'ok': True, 'tab': sidecar})


@api_bp.route('/api/user-tab/<job_id>', methods=['GET'])
@auth_required(optional=True)
def get_user_tab(job_id):
    """Serve the attached tab. ?meta=1 returns the sidecar JSON instead.
    Owner-scoped; supports ?token= like stem downloads (PDF <embed> can't
    send headers)."""
    import json as _json

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()

    tab_dir = OUTPUT_DIR / job_id
    side = tab_dir / 'user_tab.json'
    if not side.exists():
        return jsonify({'error': 'No tab attached'}), 404
    meta = _json.loads(side.read_text())
    if request.args.get('meta'):
        return jsonify({'tab': meta})
    path = tab_dir / ('user_tab' + meta['ext'])
    if not path.exists():
        return jsonify({'error': 'No tab attached'}), 404
    mime = _USER_TAB_MIMES.get(meta['ext'], 'application/octet-stream')
    resp = make_response(send_file(str(path), mimetype=mime))
    resp.headers['Content-Disposition'] = 'inline; filename="' + meta['original_name'] + '"'
    return resp


@api_bp.route('/api/user-take/<job_id>', methods=['POST'])
@auth_required
def upload_user_take(job_id):
    """Persist the MY TRACK recorded take (WAV) + its placement so it
    survives leaving the console. One take per job; re-record replaces."""
    import json as _json

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    if job.user_id is not None and job.user_id != str(user.id):
        return jsonify({'error': 'Forbidden'}), 403
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    blob = request.files['file'].read(60 * 1048576 + 1)
    if len(blob) > 60 * 1048576:
        return jsonify({'error': 'Take too large (max 60 MB)'}), 413
    try:
        start_offset = float(request.form.get('start_offset', '0'))
    except ValueError:
        start_offset = 0.0

    take_dir = OUTPUT_DIR / job_id
    take_dir.mkdir(parents=True, exist_ok=True)
    (take_dir / 'user_take.wav').write_bytes(blob)
    (take_dir / 'user_take.json').write_text(_json.dumps({
        'start_offset': start_offset, 'size': len(blob),
    }))
    # The take lands AFTER the song's pipeline backup already ran — re-sync
    try:
        from backup.stem_backup import backup_job
        backup_job(job_id, async_thread=True)
    except Exception:
        pass
    return jsonify({'ok': True})


@api_bp.route('/api/user-take/<job_id>', methods=['GET'])
@auth_required(optional=True)
def get_user_take(job_id):
    """Serve the saved take. ?meta=1 -> sidecar JSON. Owner-scoped."""
    import json as _json

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()
    side = OUTPUT_DIR / job_id / 'user_take.json'
    wav = OUTPUT_DIR / job_id / 'user_take.wav'
    if not side.exists() or not wav.exists():
        return jsonify({'error': 'No take saved'}), 404
    if request.args.get('meta'):
        return jsonify({'take': _json.loads(side.read_text())})
    return send_file(str(wav), mimetype='audio/wav')


@api_bp.route('/api/user-take/<job_id>', methods=['DELETE'])
@auth_required
def delete_user_take(job_id):
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    user = getattr(g, 'current_user', None)
    if not user or (job.user_id is not None and job.user_id != str(user.id)):
        return jsonify({'error': 'Forbidden'}), 403
    for n in ('user_take.wav', 'user_take.json'):
        try: (OUTPUT_DIR / job_id / n).unlink()
        except OSError: pass
    return jsonify({'ok': True})


# ============ FROM-SCRATCH SESSIONS (#30) ============
# A session is a lightweight job (metadata.kind == 'session') with NO stems:
# an empty console the user records into track by track. Tracks are user
# recordings stored as outputs/<job_id>/track_<n>.wav + track_<n>.json
# sidecars {start_offset, name, gain, size} — same pattern as user-take.

_SESSION_TRACK_MAX = 8
_SESSION_TRACK_BYTES = 60 * 1048576


def _session_owner_or_error(job_id):
    """(job, error_response) — load the job + enforce ownership for writes."""
    job = get_job(job_id)
    if not job:
        return None, (jsonify({'error': 'Job not found'}), 404)
    user = getattr(g, 'current_user', None)
    if not user:
        return None, (jsonify({'error': 'Authentication required'}), 401)
    if job.user_id is not None and job.user_id != str(user.id):
        return None, (jsonify({'error': 'Forbidden'}), 403)
    return job, None


def _session_track_num(n):
    try:
        n = int(n)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= _SESSION_TRACK_MAX else None


@api_bp.route('/api/session', methods=['POST'])
@auth_required
def create_session():
    """Create an empty From-Scratch Session: a completed job with no stems."""
    from models.job import save_job_checkpoint
    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    body = request.get_json(silent=True) or {}
    name = sanitize_text(str(body.get('name') or 'Untitled Session')).strip()[:120] or 'Untitled Session'
    try:
        bpm = int(body.get('bpm') or 90)
    except (TypeError, ValueError):
        bpm = 90
    bpm = max(30, min(300, bpm))
    # NOTE: plain uuid4 — a 'sess-' prefix fails validate_job_id (hex+dash,
    # <=36 chars) everywhere. metadata.kind is the discriminator.
    job_id = str(uuid.uuid4())
    job = ProcessingJob(job_id, name)
    job.status = 'completed'
    job.progress = 100
    job.stage = 'Session'
    job.user_id = str(user.id)
    job.metadata = {'kind': 'session', 'bpm': bpm, 'title': name,
                    'artist': 'From-Scratch Session'}
    jobs[job_id] = job
    save_job_checkpoint(job)
    logger.info(f"Created from-scratch session {job_id} for user {job.user_id}")
    return jsonify({'ok': True, 'job_id': job_id})


@api_bp.route('/api/session/<job_id>', methods=['PATCH'])
@auth_required
def update_session(job_id):
    """Update session settings (bpm, name). Owner only."""
    from models.job import save_job_checkpoint
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job, err = _session_owner_or_error(job_id)
    if err:
        return err
    if (job.metadata or {}).get('kind') != 'session':
        return jsonify({'error': 'Not a session'}), 400
    body = request.get_json(silent=True) or {}
    if 'bpm' in body:
        try:
            job.metadata['bpm'] = max(30, min(300, int(body['bpm'])))
        except (TypeError, ValueError):
            pass
    if body.get('name'):
        nm = sanitize_text(str(body['name'])).strip()[:120]
        if nm:
            job.filename = nm
            job.metadata['title'] = nm
    save_job_checkpoint(job)
    return jsonify({'ok': True, 'metadata': job.metadata})


@api_bp.route('/api/session-track/<job_id>/<n>', methods=['POST'])
@auth_required
def upload_session_track(job_id, n):
    """Save one session track (WAV) + its placement sidecar. Re-upload replaces."""
    import json as _json
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    n = _session_track_num(n)
    if n is None:
        return jsonify({'error': f'Track number must be 1-{_SESSION_TRACK_MAX}'}), 400
    job, err = _session_owner_or_error(job_id)
    if err:
        return err
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    blob = request.files['file'].read(_SESSION_TRACK_BYTES + 1)
    if len(blob) > _SESSION_TRACK_BYTES:
        return jsonify({'error': 'Track too large (max 60 MB)'}), 413
    try:
        start_offset = float(request.form.get('start_offset', '0'))
    except ValueError:
        start_offset = 0.0
    try:
        gain = float(request.form.get('gain', '0.8'))
    except ValueError:
        gain = 0.8
    name = sanitize_text(str(request.form.get('name') or f'TRACK {n}')).strip()[:60] or f'TRACK {n}'
    tdir = OUTPUT_DIR / job_id
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f'track_{n}.wav').write_bytes(blob)
    sidecar = {'n': n, 'start_offset': start_offset, 'name': name,
               'gain': gain, 'size': len(blob)}
    (tdir / f'track_{n}.json').write_text(_json.dumps(sidecar))
    # Sessions never run the pipeline, so its post-run R2 backup never fires —
    # protect recordings off-site on every upload (async, rglob catches all).
    try:
        from backup.stem_backup import backup_job
        backup_job(job_id, async_thread=True)
    except Exception:
        pass
    return jsonify({'ok': True, 'track': sidecar})


@api_bp.route('/api/session-track/<job_id>/<n>', methods=['GET'])
@auth_required(optional=True)
def get_session_track(job_id, n):
    """Serve one track's WAV; ?meta=1 -> sidecar JSON. Owner-scoped."""
    import json as _json
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    n = _session_track_num(n)
    if n is None:
        return jsonify({'error': 'Invalid track number'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()
    side = OUTPUT_DIR / job_id / f'track_{n}.json'
    wav = OUTPUT_DIR / job_id / f'track_{n}.wav'
    if not side.exists() or not wav.exists():
        return jsonify({'error': 'No such track'}), 404
    if request.args.get('meta'):
        return jsonify({'track': _json.loads(side.read_text())})
    return send_file(str(wav), mimetype='audio/wav')


@api_bp.route('/api/session-track/<job_id>/<n>', methods=['PATCH'])
@auth_required
def rename_session_track(job_id, n):
    """Rename a session track (updates the sidecar the strips are built from)."""
    import json as _json
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    user = getattr(g, 'current_user', None)
    if not user or (job.user_id is not None and job.user_id != str(user.id)):
        return jsonify({'error': 'Forbidden'}), 403
    name = ((request.get_json(silent=True) or {}).get('name') or '').strip()[:40]
    if not name:
        return jsonify({'error': 'Name required'}), 400
    side = OUTPUT_DIR / job_id / f'track_{n}.json'
    meta = {}
    try:
        meta = _json.loads(side.read_text())
    except Exception:
        pass
    meta['name'] = name
    tmp = str(side) + '.new'
    with open(tmp, 'w') as f:
        _json.dump(meta, f)
    os.replace(tmp, str(side))
    return jsonify({'ok': True, 'name': name})


@api_bp.route('/api/session-track/<job_id>/<n>', methods=['DELETE'])
@auth_required
def delete_session_track(job_id, n):
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    n = _session_track_num(n)
    if n is None:
        return jsonify({'error': 'Invalid track number'}), 400
    job, err = _session_owner_or_error(job_id)
    if err:
        return err
    for fn in (f'track_{n}.wav', f'track_{n}.json'):
        try:
            (OUTPUT_DIR / job_id / fn).unlink()
        except OSError:
            pass
    return jsonify({'ok': True})


@api_bp.route('/api/session-tracks/<job_id>', methods=['GET'])
@auth_required(optional=True)
def list_session_tracks(job_id):
    """All saved tracks' sidecars, ascending by n. Owner-scoped."""
    import json as _json
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()
    out = []
    for i in range(1, _SESSION_TRACK_MAX + 1):
        side = OUTPUT_DIR / job_id / f'track_{i}.json'
        wav = OUTPUT_DIR / job_id / f'track_{i}.wav'
        if side.exists() and wav.exists():
            try:
                out.append(_json.loads(side.read_text()))
            except Exception:
                pass
    return jsonify({'tracks': out})


_chart_track_lock = threading.Lock()


@api_bp.route('/api/chart-track/<job_id>/<int:n>', methods=['POST'])
@auth_required
def chart_session_track(job_id, n):
    """Jeff's 4-track vision: generate a chord chart FROM a session track the
    user recorded (solo guitar/piano = the detector's easiest input). Lite
    pipeline: detection + measured key + Whisper + format_chart on one WAV —
    no stem separation, no cloud GPU. One chart per session (latest wins).
    """
    import json as _json
    from pathlib import Path as _Path

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    user = getattr(g, 'current_user', None)
    if not user or (job.user_id is not None and job.user_id != str(user.id)):
        return jsonify({'error': 'Forbidden'}), 403
    if (job.metadata or {}).get('kind') != 'session':
        return jsonify({'error': 'Charting is for session tracks (v1)'}), 400
    wav = OUTPUT_DIR / job_id / f'track_{n}.wav'
    side = OUTPUT_DIR / job_id / f'track_{n}.json'
    if not wav.exists():
        return jsonify({'error': 'No such track'}), 404
    if not _chart_track_lock.acquire(blocking=False):
        return jsonify({'error': 'Another track is being charted — try again in a minute'}), 429

    track_name = f'TRACK {n}'
    try:
        meta = _json.loads(side.read_text())
        track_name = meta.get('name') or track_name
    except Exception:
        pass

    def _run():
        try:
            job.metadata['charting'] = 'running'
            from models.job import save_job_checkpoint
            from take_chord_detector import detect_take_chords
            from word_timestamps import get_word_timestamps
            from chart_formatter import format_chart
            from chart_library_matcher import _measure_sounding_key, _spell_in_key, _parse_key

            # TAKE-SIZED detector (7/6): the song detectors assume full-band
            # full-length audio and went blind on a clean 19s G-Am-C-D take.
            # Windowed chroma template-match reads solo takes correctly.
            job.chord_progression = detect_take_chords(wav)
            job.detected_key = None

            key = job.detected_key or 'Unknown'
            measured = _measure_sounding_key({'other': str(wav)})
            if measured is not None:
                dk = _parse_key(key)
                minor = dk[1] if dk else False
                key_pc = (measured - 3) % 12 if minor else measured
                key = _spell_in_key(key_pc, measured) + ('m' if minor else '')

            word_ts = []
            try:
                word_ts = get_word_timestamps(str(wav)) or []
            except Exception:
                pass

            chart = format_chart(
                chord_events=job.chord_progression or [],
                word_timestamps=word_ts,
                title=f'{job.filename} — {track_name}',
                artist='',
                key=key,
                grid=(job.metadata or {}).get('grid'),
            )
            if chart and chart.get('sections'):
                chart['charted_from'] = {'track': n, 'name': track_name}
                out = OUTPUT_DIR / job_id / 'chord_chart.json'
                tmp = str(out) + '.new'
                with open(tmp, 'w') as f:
                    _json.dump(chart, f, indent=2)
                os.replace(tmp, str(out))
                job.metadata['charting'] = 'done'
            else:
                job.metadata['charting'] = 'empty'
            save_job_checkpoint(job)
        except Exception as e:
            logger.warning(f'chart-track failed for {job_id}/{n}: {e}')
            job.metadata['charting'] = 'failed'
        finally:
            _chart_track_lock.release()

    t = threading.Thread(target=_run)
    t.daemon = True
    t.start()
    return jsonify({'ok': True, 'charting': n, 'name': track_name})

@api_bp.route('/api/account', methods=['DELETE'])
@auth_required
def delete_account():
    """Apple 5.1.1(v): full in-app account deletion. Removes the user's jobs
    (disk + memory), chart library, edit history, and the user row. The JSON
    body must contain {"confirm": "DELETE"} — a typed confirmation from the
    UI, never a bare click."""
    import shutil
    from db import execute as db_execute, query_all as db_query_all
    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401
    body = request.get_json(silent=True) or {}
    if body.get('confirm') != 'DELETE':
        return jsonify({'error': 'Confirmation required'}), 400
    uid = str(user.id)

    # 1) jobs: disk + memory
    removed = 0
    for job_id, job in list(jobs.items()):
        owns = (getattr(job, 'user_id', None) == uid or
                (isinstance(getattr(job, 'metadata', None), dict) and job.metadata.get('user_id') == uid))
        if owns:
            try:
                d = OUTPUT_DIR / job_id
                if d.exists():
                    shutil.rmtree(d)
            except Exception as e:
                logger.warning(f'account-delete: job dir {job_id}: {e}')
            jobs.pop(job_id, None)
            removed += 1
    # disk-only jobs the memory map missed
    try:
        import json as _json
        for meta_path in OUTPUT_DIR.glob('*/job_metadata.json'):
            try:
                m = _json.loads(meta_path.read_text())
            except Exception:
                continue
            if m.get('user_id') == uid or (m.get('metadata') or {}).get('user_id') == uid:
                shutil.rmtree(meta_path.parent, ignore_errors=True)
                removed += 1
    except Exception as e:
        logger.warning(f'account-delete disk sweep: {e}')

    # 2) database rows (order matters for FKs)
    for sql in [
        "DELETE FROM chart_edit_history WHERE user_id = %s",
        "DELETE FROM chart_library WHERE user_id = %s",
        "DELETE FROM users WHERE id = %s",
    ]:
        try:
            db_execute(sql, (uid,))
        except Exception as e:
            logger.warning(f'account-delete sql ({sql.split()[2]}): {e}')

    logger.info(f'ACCOUNT DELETED: {uid} ({getattr(user, "email", "?")}), {removed} jobs removed')
    return jsonify({'ok': True, 'jobs_removed': removed})
