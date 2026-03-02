"""
Tab-related routes — find-tabs, download-pro-tabs, download-pro-tab-file.
"""

import re
import logging
from pathlib import Path
from flask import Blueprint, jsonify, send_file

from models.job import get_job

logger = logging.getLogger(__name__)

tabs_bp = Blueprint("tabs", __name__)


def _validate_job_id(job_id: str) -> bool:
    """Validate job_id is a safe hex string (UUID prefix)."""
    return bool(job_id) and len(job_id) <= 36 and re.match(r'^[a-f0-9\\-]+$', job_id)


@tabs_bp.route('/api/find-tabs/<job_id>', methods=['GET'])
def find_tabs(job_id):
    """Find matching tabs on Songsterr and Ultimate Guitar for a job"""
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    title = job.metadata.get('title', '') if job.metadata else ''
    artist = job.metadata.get('artist', '') if job.metadata else ''

    if not title:
        return jsonify({'error': 'No song title available'}), 400

    def slugify(text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s-]', '', text)
        text = re.sub(r'[\s_]+', '-', text)
        return text.strip('-')

    title_slug = slugify(title)
    artist_slug = slugify(artist) if artist else ''
    search_query = f"{title} {artist}".strip().replace(' ', '+')

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


@tabs_bp.route('/api/download-pro-tabs/<job_id>', methods=['POST'])
def download_pro_tabs(job_id):
    """Download professional tabs from Songsterr for this job."""
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

        logger.info(f"Downloaded pro tabs: {tab.title} by {tab.artist}")

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


@tabs_bp.route('/api/download/<job_id>/pro_tabs/<filename>', methods=['GET'])
def download_pro_tab_file(job_id, filename):
    """Download a professional tab file (GP5)"""
    if not _validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    if '..' in filename or '/' in filename:
        return jsonify({'error': 'Invalid filename'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    if not hasattr(job, 'pro_tabs') or not job.pro_tabs:
        return jsonify({'error': 'No professional tabs available'}), 404

    if 'songsterr' in job.pro_tabs:
        tab_path = Path(job.pro_tabs['songsterr']['path'])
        if tab_path.exists():
            return send_file(
                tab_path,
                as_attachment=True,
                download_name=filename
            )

    return jsonify({'error': 'Tab file not found'}), 404
