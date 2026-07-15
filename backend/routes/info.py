from urllib.parse import quote
"""
Track info routes — artist context, learning tips, instrument recommendations.
"""

import logging
from flask import Blueprint, request, jsonify

from models.job import get_job
from auth.middleware import auth_required, authorize_job_access, forbidden_response

logger = logging.getLogger(__name__)

info_bp = Blueprint("info", __name__)


@info_bp.route('/api/info/<job_id>', methods=['GET'])
@auth_required(optional=True)
def get_track_info(job_id):
    """Get contextual info about a track (artist bio, learning tips, etc.)"""
    from dependencies import TRACK_INFO_AVAILABLE

    try:
        job = get_job(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        if not authorize_job_access(job):
            return forbidden_response()

        if not TRACK_INFO_AVAILABLE:
            return jsonify({'error': 'Track info module not available'}), 500

        from dependencies import fetch_track_info, get_instrument_tips, should_stereo_split

        # Get track name from job
        track_name = job.filename
        if job.metadata.get('search_query'):
            track_name = job.metadata['search_query']
        elif job.metadata.get('title'):
            track_name = job.metadata['title']

        # Parse real artist from title ("Artist - Song") since YouTube metadata
        # often has the uploader name rather than the actual artist
        artist = job.metadata.get('artist')
        if track_name and ' - ' in track_name:
            parsed_artist = track_name.split(' - ', 1)[0].strip()
            if parsed_artist:
                artist = parsed_artist
                # Also clean the track name to just the song title
                track_name = track_name.split(' - ', 1)[1].strip()

        logger.info(f"Fetching track info for job {job_id}: {track_name} by {artist}")

        # Use embedded track_info from metadata if available (e.g. demo songs)
        embedded = job.metadata.get('track_info')
        if embedded:
            logger.info(f"Using embedded track info for {job_id}")
            info = dict(embedded)
            info.setdefault('artist', artist)
            info.setdefault('track', track_name)
        else:
            info = fetch_track_info(
                track_name=track_name,
                artist=artist,
                source_url=job.source_url
            )
            # Cache the expensive external fetch on the job so repeat "About
            # This Track" views are instant. Without this, every view re-hits
            # Wikipedia + MusicBrainz (6-15s, rate-limited) and can time out.
            try:
                if info and any(info.get(k) for k in ('bio', 'album', 'cover_art_url', 'learning_tips')):
                    job.metadata['track_info'] = dict(info)
                    from models.job import save_job_to_disk
                    save_job_to_disk(job)
            except Exception as _ce:
                logger.debug(f"track_info cache failed: {_ce}")

        # Enrich with job metadata not available from track_info lookup
        if not info.get('thumbnail'):
            raw_thumb = job.metadata.get('thumbnail') or ''
            if raw_thumb and 'ytimg.com' in raw_thumb:
                raw_thumb = '/api/thumbnail?url=' + quote(raw_thumb, safe='')
            info['thumbnail'] = raw_thumb
        if not info.get('album') and job.metadata.get('album'):
            info['album'] = job.metadata['album']
        info['source_url'] = job.metadata.get('webpage_url') or job.source_url or ''

        # Add instrument-specific tips for each stem (skip if embedded info has them)
        if not info.get('stem_tips'):
            info['stem_tips'] = {}
            for stem_name in job.stems.keys():
                info['stem_tips'][stem_name] = get_instrument_tips(stem_name, info.get('style'))

        # Add stereo split recommendation (reuse parsed artist from above)
        artist = artist or info.get('artist')
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




def _gen_full_mix_peaks(output_dir, job_id, N=1400, sr=11025):
    """Sum the primary stems into a full-mix peak map (lazy, cached to peaks.json)."""
    import os, json
    try:
        import numpy as np, librosa
    except Exception:
        return None
    stems_dir = os.path.join(output_dir, job_id, 'stems')
    if not os.path.isdir(stems_dir):
        return None
    mix = None
    for nm in ('vocals', 'drums', 'bass', 'guitar', 'piano', 'other'):
        f = os.path.join(stems_dir, nm + '.mp3')
        if not os.path.exists(f):
            continue
        try:
            y, _ = librosa.load(f, sr=sr, mono=True)
        except Exception:
            continue
        if mix is None:
            mix = y.astype('float32')
        else:
            n = min(len(mix), len(y)); mix = mix[:n] + y[:n].astype('float32')
    if mix is None or len(mix) == 0:
        return None
    win = max(1, len(mix) // N)
    rms = np.array([np.sqrt(np.mean(mix[i:i+win] ** 2)) for i in range(0, len(mix), win)][:N])
    rms = rms / (rms.max() or 1)
    lo, hi = np.percentile(rms, 10), np.percentile(rms, 95)
    env = np.clip((rms - lo) / (hi - lo + 1e-9), 0, 1)
    data = {'peaks': [round(float(x), 4) for x in env], 'duration': round(len(mix) / sr, 2)}
    try:
        json.dump(data, open(os.path.join(output_dir, job_id, 'peaks.json'), 'w'))
    except Exception:
        pass
    return data

@info_bp.route('/api/peaks/<job_id>', methods=['GET'])
@auth_required(optional=True)
def get_peaks(job_id):
    """Precomputed full-mix waveform peak map for the master waveform."""
    import os, json
    from models.job import OUTPUT_DIR
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()
    pf = os.path.join(str(OUTPUT_DIR), job_id, 'peaks.json')
    if not os.path.exists(pf):
        data = _gen_full_mix_peaks(str(OUTPUT_DIR), job_id)
        if not data:
            return jsonify({'peaks': [], 'duration': 0}), 200
        resp = jsonify(data)
        resp.headers['Cache-Control'] = 'public, max-age=3600'
        return resp
    with open(pf) as f:
        data = json.load(f)
    resp = jsonify(data)
    resp.headers['Cache-Control'] = 'public, max-age=3600'
    return resp


@info_bp.route('/api/mix/<job_id>', methods=['GET'])
@auth_required(optional=True)
def get_mix(job_id):
    """Lazy-generated full-mix audio (sum of primary stems) so the master
    waveform can render the REAL, detailed waveform of the whole song."""
    import os, subprocess
    from flask import send_file
    from models.job import OUTPUT_DIR
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if not authorize_job_access(job):
        return forbidden_response()
    jdir = os.path.join(str(OUTPUT_DIR), job_id)
    mixf = os.path.join(jdir, 'mix.mp3')
    if not os.path.exists(mixf):
        stems_dir = os.path.join(jdir, 'stems')
        inputs = []
        for nm in ('vocals', 'drums', 'bass', 'guitar', 'piano', 'other'):
            f = os.path.join(stems_dir, nm + '.mp3')
            if os.path.exists(f):
                inputs += ['-i', f]
        if not inputs:
            return jsonify({'error': 'no stems'}), 404
        n = len(inputs) // 2
        try:
            subprocess.run(
                ['ffmpeg', '-y'] + inputs +
                ['-filter_complex', 'amix=inputs=%d:normalize=1' % n,
                 '-c:a', 'libmp3lame', '-b:a', '96k', mixf],
                capture_output=True, timeout=180, check=True)
        except Exception:
            return jsonify({'error': 'mix failed'}), 500
    return send_file(mixf, mimetype='audio/mpeg', conditional=True)
