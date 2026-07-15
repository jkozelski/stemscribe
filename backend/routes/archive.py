"""
Archive.org Live Music API routes.
"""

import re
import uuid
import threading
import logging
from flask import Blueprint, request, jsonify, g

from models.job import ProcessingJob, jobs
from processing.pipeline import process_url
from services.url_resolver import validate_url_no_ssrf as _validate_url_no_ssrf
from auth.middleware import auth_required

logger = logging.getLogger(__name__)

# Maximum concurrent batch processing threads
MAX_BATCH_THREADS = 5
_batch_semaphore = threading.Semaphore(MAX_BATCH_THREADS)

# Conditional imports
try:
    from archive_pipeline import (
        ArchivePipeline, search_archive, get_show_info,  # noqa: F401
        get_pipeline as get_archive_pipeline
    )
    ARCHIVE_PIPELINE_AVAILABLE = True
except ImportError:
    ARCHIVE_PIPELINE_AVAILABLE = False


def _clean_archive_title(raw_name):
    """Taper filenames -> human titles: gd73-07-28s1t06BoxOfRain -> Box Of Rain."""
    name = re.sub(r'\.(mp3|flac|ogg|wav|shn|m4a)$', '', raw_name or '', flags=re.IGNORECASE)
    # Date/track prefixes: "gd69-05-23 t03 ", "gd1972-08-21s1t08", "gd73-07-28s1t06BoxOfRain"
    name = re.sub(r'^[a-z]{2,4}\d{2,4}[-_]\d{2}[-_]\d{2}\s*([sd]\d+)?t?\d*\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^t\d+\s+', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^d\d+t\d+\s*', '', name, flags=re.IGNORECASE)
    # Split CamelCase: BoxOfRain -> Box Of Rain
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', name)
    name = name.replace('_', ' ')
    name = re.sub(r'\s+', ' ', name).strip()
    if name and name.islower():
        name = name.title()
    return name or raw_name


archive_bp = Blueprint("archive", __name__)


@archive_bp.route('/api/archive/search', methods=['GET'])
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
    try:
        rows = max(1, min(int(request.args.get('rows', 25)), 100))
    except (ValueError, TypeError):
        rows = 25

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


@archive_bp.route('/api/archive/collections', methods=['GET'])
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


@archive_bp.route('/api/archive/show/<identifier>', methods=['GET'])
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

    _prefer_format = request.args.get('format', 'mp3')

    try:
        result = get_show_info(identifier)
        if 'error' in result:
            return jsonify(result), 404
        result['available'] = True
        return jsonify(result)
    except Exception as e:
        logger.error(f"Archive show details failed for {identifier}: {e}")
        return jsonify({'error': f'Failed to get show details: {str(e)}'}), 500


@archive_bp.route('/api/archive/process', methods=['POST'])
@auth_required(optional=True)
def archive_process_track():
    """
    Process an Archive.org track through StemScriber's full pipeline.

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

    # Sanitize identifier and filename (prevent path traversal in constructed URLs)
    if identifier and not re.match(r'^[a-zA-Z0-9._\-]+$', identifier):
        return jsonify({'error': 'Invalid archive identifier'}), 400
    if filename and ('..' in filename or filename.startswith('/')):
        return jsonify({'error': 'Invalid filename'}), 400

    # Build the URL if we have identifier + filename but no URL
    if not url and identifier and filename:
        from urllib.parse import quote
        url = f"https://archive.org/download/{quote(identifier)}/{quote(filename)}"
    elif not url and identifier:
        from urllib.parse import quote
        url = f"https://archive.org/details/{quote(identifier)}"

    if not url:
        return jsonify({'error': 'Provide url, or identifier + filename'}), 400

    # Validate constructed URL
    if not _validate_url_no_ssrf(url):
        return jsonify({'error': 'URL not allowed'}), 400

    # Get processing options
    skills = data.get('skills', [])
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(',') if s.strip()]

    enhance_stems = data.get('enhance_stems', False)
    stereo_split = data.get('stereo_split', False)
    gp_tabs = data.get('gp_tabs', True)
    chord_detection = data.get('chord_detection', True)
    mdx_model = data.get('mdx_model', False)
    ensemble_mode = data.get('ensemble', False)

    # Create job — clean up archive filenames for display
    job_id = str(uuid.uuid4())
    # Newer clients send the already-cleaned display title; fall back to filename
    raw_name = (data.get('title') or '').strip() or filename or identifier or 'Archive.org track'
    display_name = _clean_archive_title(raw_name)
    artist = (data.get('artist') or '').strip()
    # Server-side rescue: taper filenames like gd1987-11-15d1t04 carry no song
    # name, and older clients don't send title/artist — ask archive.org for the
    # track list (title) and show creator (artist).
    _needs_title = (display_name == raw_name or
                    re.match(r'^[a-z]{1,4}\d{2,4}[-_.]', display_name, re.IGNORECASE) or
                    # CamelCase band prefixes fused to dates slip the cleaner
                    # ("DeerTick2009-06-16d1t08" -> "Deer Tick2009-06-16d1t08",
                    # cold-run finding 7/5) — any date/track residue means the
                    # name is still a filename, ask archive.org for the title
                    re.search(r'\d{2,4}[-_.]\d{2}[-_.]\d{2}|d\d+t\d+|s\d+t\d+', display_name, re.IGNORECASE))
    if identifier and (_needs_title or not artist):
        try:
            _show_info = get_show_info(identifier)
            if _needs_title and filename:
                for _t in (_show_info.get('tracks') or []):
                    _tf = _t.get('filename') if isinstance(_t, dict) else getattr(_t, 'filename', None)
                    if _tf == filename:
                        _tt = _t.get('title') if isinstance(_t, dict) else getattr(_t, 'title', '')
                        if _tt:
                            display_name = _clean_archive_title(_tt)
                        break
            if not artist:
                _sh = _show_info.get('show') or {}
                artist = ((_sh.get('creator') if isinstance(_sh, dict)
                           else getattr(_sh, 'creator', '')) or '').strip()
        except Exception as _look_err:
            logger.warning(f"Archive title/artist lookup failed (non-fatal): {_look_err}")
    job = ProcessingJob(job_id, display_name, source_url=url, skills=skills)
    # Stamp the logged-in owner so the job SAVES to their library (archive/URL
    # downloads were orphaned with user_id=None → processed fine but never showed
    # up in the user's library — Jeff's "it worked but never saved" bug).
    job.user_id = str(g.current_user.id) if getattr(g, 'current_user', None) else None
    job.session_id = request.cookies.get('session_id') or str(uuid.uuid4())
    jobs[job_id] = job

    # Store archive metadata
    job.metadata['source'] = 'archive.org'
    job.metadata['archive_identifier'] = identifier
    if filename:
        job.metadata['archive_filename'] = filename
    if artist:
        job.metadata['artist'] = artist
    # Keep the cleaned title — the downloader must not stomp it with the raw filename
    job.metadata['display_name_locked'] = True

    # Process in background thread (same pipeline as /api/url)
    thread = threading.Thread(
        target=process_url,
        args=(job, url),
        kwargs={
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

    resp = jsonify({
        'job_id': job_id,
        'message': 'Processing Archive.org track',
        'filename': display_name,
        'source': 'archive.org',
        'identifier': identifier,
        'skills': skills,
    })
    if not request.cookies.get('session_id'):
        # hand the anonymous session back so status polls can authorize
        resp.set_cookie('session_id', job.session_id, httponly=True, max_age=86400, samesite='Lax')
    return resp


@archive_bp.route('/api/archive/batch', methods=['POST'])
@auth_required
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

    identifier = data['identifier'].strip()
    if not re.match(r'^[a-zA-Z0-9._\-]+$', identifier):
        return jsonify({'error': 'Invalid archive identifier'}), 400
    requested_tracks = data.get('tracks', [])
    prefer_format = data.get('format', 'mp3')
    if prefer_format not in ('mp3', 'flac', 'ogg', 'wav'):
        prefer_format = 'mp3'
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
            job_id = str(uuid.uuid4())
            display_name = _clean_archive_title(track.title or track.filename)
            job = ProcessingJob(job_id, display_name, source_url=url, skills=skills)
            job.user_id = str(g.current_user.id) if getattr(g, 'current_user', None) else None
            job.session_id = request.cookies.get('session_id') or str(uuid.uuid4())
            jobs[job_id] = job
            job.metadata['source'] = 'archive.org'
            job.metadata['archive_identifier'] = identifier
            job.metadata['archive_filename'] = track.filename
            job.metadata['archive_track_number'] = track.track_number
            _batch_artist = (data.get('artist') or '').strip()
            if _batch_artist:
                job.metadata['artist'] = _batch_artist
            job.metadata['display_name_locked'] = True

            def _batch_worker(j, u, kw):
                _batch_semaphore.acquire()
                try:
                    process_url(j, u, **kw)
                finally:
                    _batch_semaphore.release()

            thread = threading.Thread(
                target=_batch_worker,
                args=(job, url, {
                    'gp_tabs': data.get('gp_tabs', True),
                    'chord_detection': data.get('chord_detection', True),
                })
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

