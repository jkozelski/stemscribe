"""
Track info routes — artist context, learning tips, instrument recommendations.
"""

import logging
from flask import Blueprint, request, jsonify

from models.job import get_job

logger = logging.getLogger(__name__)

info_bp = Blueprint("info", __name__)


@info_bp.route('/api/info/<job_id>', methods=['GET'])
def get_track_info(job_id):
    """Get contextual info about a track (artist bio, learning tips, etc.)"""
    from dependencies import TRACK_INFO_AVAILABLE

    try:
        job = get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        if not TRACK_INFO_AVAILABLE:
            return jsonify({'error': 'Track info module not available'}), 500

        from dependencies import fetch_track_info, get_instrument_tips, should_stereo_split

        # Get track name from job
        track_name = job.filename
        if job.metadata.get('search_query'):
            track_name = job.metadata['search_query']
        elif job.metadata.get('title'):
            track_name = job.metadata['title']

        logger.info(f"Fetching track info for job {job_id}: {track_name}")

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
        return jsonify({
            'error': None,
            'track': job.filename if job else 'Unknown',
            'artist': job.metadata.get('artist') if job else None,
            'bio': 'Track information temporarily unavailable.',
            'learning_tips': 'Use the stem separation to isolate and study individual parts.',
            'fetched_from': ['fallback']
        })


@info_bp.route('/api/info/search', methods=['POST'])
def search_track_info():
    """Search for track info by artist/song name"""
    from dependencies import TRACK_INFO_AVAILABLE

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    if not TRACK_INFO_AVAILABLE:
        return jsonify({'error': 'Track info module not available'}), 500

    from dependencies import fetch_track_info

    track_name = data.get('track', '')
    artist = data.get('artist', '')

    if not track_name and not artist:
        return jsonify({'error': 'Provide track or artist name'}), 400

    info = fetch_track_info(track_name=track_name, artist=artist)
    return jsonify(info)
