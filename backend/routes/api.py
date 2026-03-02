"""
Core API routes — upload, url, status, health, download, jobs, cleanup, skills, models, quality.
"""

import os
import re
import uuid
import shutil
import threading
import logging
from pathlib import Path
from flask import Blueprint, request, jsonify, send_file

from models.job import (
    ProcessingJob, jobs, get_job, OUTPUT_DIR, UPLOAD_DIR,
)
from processing.pipeline import process_audio, process_url
from services.url_resolver import (
    is_supported_url, is_streaming_url,
    get_spotify_track_info, get_apple_music_track_info, search_youtube_for_song,
    validate_url_no_ssrf as _validate_url_no_ssrf,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)


# ============ VALIDATION HELPERS ============

def _validate_job_id(job_id: str) -> bool:
    """Validate job_id is a safe hex string (UUID prefix)."""
    return bool(job_id) and len(job_id) <= 36 and re.match(r'^[a-f0-9\\-]+$', job_id)


def _safe_path(base_dir: Path, untrusted_path: str) -> Path:
    """Resolve a path and ensure it stays within base_dir (prevents path traversal)."""
    resolved = (base_dir / untrusted_path).resolve()
    if not str(resolved).startswith(str(base_dir.resolve())):
        raise ValueError(f"Path traversal detected: {untrusted_path}")
    return resolved


# ============ HEALTH ============

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
        'service': 'StemScribe API',
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

    # Create job with skills
    job_id = str(uuid.uuid4())[:8]
    job = ProcessingJob(job_id, file.filename, skills=skills)
    jobs[job_id] = job

    # Save uploaded file (sanitize filename to prevent path traversal)
    from werkzeug.utils import secure_filename
    safe_name = secure_filename(file.filename) or 'upload.wav'
    job_upload_dir = UPLOAD_DIR / job_id
    job_upload_dir.mkdir(exist_ok=True)
    audio_path = job_upload_dir / safe_name
    file.save(str(audio_path))

    mode_str = 'ENSEMBLE' if ensemble_mode else ('MDX' if mdx_model else 'standard')
    logger.info(f"Created job {job_id} for file {file.filename} - mode: {mode_str}, gp_tabs: {gp_tabs}, chord_detection: {chord_detection}")

    # Start processing in background thread
    thread = threading.Thread(target=process_audio, args=(job, audio_path, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode))
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Processing started',
        'filename': file.filename,
        'skills': skills
    })


# ============ URL PROCESSING ============

@api_bp.route('/api/url', methods=['POST'])
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

    # Check if this is a streaming service URL (Spotify/Apple Music)
    streaming_service = is_streaming_url(url)
    original_url = url
    track_info = None

    if streaming_service:
        logger.info(f"Detected {streaming_service} URL, extracting track info...")

        if streaming_service == 'spotify':
            track_info = get_spotify_track_info(url)
        elif streaming_service == 'apple_music':
            track_info = get_apple_music_track_info(url)

        if not track_info:
            return jsonify({
                'error': f'Could not extract track info from {streaming_service}. Try pasting a direct track link.'
            }), 400

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
    enhance_stems = data.get('enhance_stems', False)
    stereo_split = data.get('stereo_split', False)
    gp_tabs = data.get('gp_tabs', True)
    chord_detection = data.get('chord_detection', True)
    mdx_model = data.get('mdx_model', False)
    ensemble_mode = data.get('ensemble', False)

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
    thread = threading.Thread(target=process_url, args=(job, url, enhance_stems, stereo_split, gp_tabs, chord_detection, mdx_model, ensemble_mode))
    thread.daemon = True
    thread.start()

    return jsonify({
        'job_id': job_id,
        'message': 'Download and processing started',
        'url': url,
        'source': streaming_service or 'direct',
        'track_info': track_info
    })


# ============ STATUS ============

@api_bp.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Get the status of a processing job.

    Query params:
        slim=1  Return only {status, progress, stage, error} (~80 bytes)
                instead of the full job dict (~5-10KB). Use this for polling
                during processing; fetch full status once when status='completed'.
    """
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
    return jsonify(job.to_dict())


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


# ============ JOBS LIST ============

@api_bp.route('/api/jobs', methods=['GET'])
def list_jobs():
    """List all jobs"""
    return jsonify({
        'jobs': [job.to_dict() for job in jobs.values()]
    })


# ============ CLEANUP ============

@api_bp.route('/api/cleanup', methods=['POST'])
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
