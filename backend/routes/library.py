"""
Library routes — browse and manage processed songs.
"""

import shutil
import logging
from flask import Blueprint, jsonify

from models.job import jobs, get_job, OUTPUT_DIR

logger = logging.getLogger(__name__)

library_bp = Blueprint("library", __name__)


@library_bp.route('/api/library', methods=['GET'])
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


@library_bp.route('/api/library/<job_id>', methods=['DELETE'])
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

        logger.info(f"Deleted job {job_id} from library")
        return jsonify({'status': 'deleted', 'job_id': job_id})

    except Exception as e:
        logger.error(f"Failed to delete job {job_id}: {e}")
        return jsonify({'error': str(e)}), 500
