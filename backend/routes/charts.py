"""
Chart library routes — Jeff's personal chord-chart library.

PRIVACY RULE: charts are PRIVATE to their owner. Many bodies contain
copyrighted lyrics, so every query is scoped to g.current_user.id and
there is deliberately NO share/public path. Do not add one.
"""

import logging

from flask import Blueprint, jsonify, request, g

from auth.middleware import auth_required
from db import query_all, query_one
from middleware.validation import validate_job_id

logger = logging.getLogger(__name__)

charts_bp = Blueprint('charts', __name__)

MAX_RESULTS = 5000  # library is ~2k charts; cap defensively


def _summary(row):
    return {
        'id': row['id'],
        'title': row['title'],
        'artist': row['artist'],
        'song_key': row['song_key'],
        'source': row['source'],
        'flagged_for_review': row['flagged_for_review'],
        'created_at': row['created_at'].isoformat() if row.get('created_at') else None,
    }


@charts_bp.route('/api/charts', methods=['GET'])
@auth_required
def list_charts():
    """List/search the caller's own charts. ?q= matches title or artist."""
    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    q = (request.args.get('q') or '').strip()
    sql = """
        SELECT id, title, artist, song_key, source, flagged_for_review, created_at
        FROM chart_library
        WHERE user_id = %s
    """
    params = [str(user.id)]
    if q:
        sql += " AND (title ILIKE %s OR artist ILIKE %s)"
        like = '%' + q.replace('%', r'\%').replace('_', r'\_') + '%'
        params += [like, like]
    sql += " ORDER BY title ASC, artist ASC LIMIT %s"
    params.append(MAX_RESULTS)

    rows = query_all(sql, params)
    return jsonify({
        'charts': [_summary(r) for r in rows],
        'count': len(rows),
        'q': q or None,
    })


@charts_bp.route('/api/charts/<int:chart_id>', methods=['GET'])
@auth_required
def get_chart(chart_id):
    """Fetch one chart, body included. Owner-only — 404 for anyone else."""
    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    row = query_one(
        """
        SELECT id, title, artist, song_key, body, source, flagged_for_review,
               created_at, updated_at
        FROM chart_library
        WHERE id = %s AND user_id = %s
        """,
        (chart_id, str(user.id)),
    )
    if not row:
        # 404 (not 403) so non-owners can't probe which ids exist
        return jsonify({'error': 'Chart not found'}), 404

    out = _summary(row)
    out['body'] = row['body']
    out['updated_at'] = row['updated_at'].isoformat() if row.get('updated_at') else None
    return jsonify(out)


@charts_bp.route('/api/charts/<int:chart_id>', methods=['PATCH'])
@auth_required
def update_chart(chart_id):
    """Owner edits their own chart body (the correction flywheel, #13).

    Accepts {body, song_key?, refresh_job?}. Body stays in the chart's own
    key — the display transposition is recomputed per job. When refresh_job
    is a job the caller owns whose chart matched THIS library chart, its
    chord_chart.json is re-rendered immediately so the fix shows on reload
    (aligned section start times are carried over by section index+name).
    """
    import json as _json
    from pathlib import Path

    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json(silent=True) or {}
    body = data.get('body')
    if not isinstance(body, str) or not body.strip():
        return jsonify({'error': 'body required'}), 400
    if len(body) > 100_000:
        return jsonify({'error': 'body too large'}), 413

    row = query_one(
        "SELECT id, title, song_key FROM chart_library WHERE id = %s AND user_id = %s",
        (chart_id, str(user.id)),
    )
    if not row:
        return jsonify({'error': 'Chart not found'}), 404

    song_key = (data.get('song_key') or row['song_key'] or '').strip() or row['song_key']

    from db import get_db
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chart_library SET body = %s, song_key = %s, updated_at = NOW() "
                "WHERE id = %s AND user_id = %s",
                (body, song_key, chart_id, str(user.id)),
            )
            conn.commit()

    refreshed = False
    refresh_err = None
    job_id = (data.get('refresh_job') or '').strip()
    if job_id and validate_job_id(job_id):
        try:
            from models.job import get_job, OUTPUT_DIR
            job = get_job(job_id)
            chart_path = Path(str(OUTPUT_DIR)) / job_id / 'chord_chart.json'
            if job and (job.user_id == str(user.id)) and chart_path.exists():
                old = _json.loads(chart_path.read_text())
                lm = old.get('library_match') or {}
                if lm.get('chart_id') == chart_id:
                    from chart_library_matcher import render_library_chart
                    fresh_row = query_one(
                        "SELECT id, title, artist, song_key, body FROM chart_library WHERE id = %s",
                        (chart_id,),
                    )
                    target = lm.get('display_key') if lm.get('key_source') == 'measured' else None
                    new_chart = render_library_chart(dict(fresh_row), target)
                    new_chart['title'] = old.get('title') or new_chart['title']
                    new_chart['artist'] = old.get('artist') or new_chart.get('artist')
                    # carry aligned section timings over where structure matches
                    old_secs = old.get('sections') or []
                    for i, sec in enumerate(new_chart.get('sections') or []):
                        if i < len(old_secs) and old_secs[i].get('name') == sec.get('name'):
                            if old_secs[i].get('start') is not None:
                                sec['start'] = old_secs[i]['start']
                            if old_secs[i].get('lines') and sec.get('lines') and \
                               len(old_secs[i]['lines']) == len(sec['lines']):
                                for j, ln in enumerate(sec['lines']):
                                    oseg = old_secs[i]['lines'][j].get('segments')
                                    if oseg and ln.get('lyrics') == old_secs[i]['lines'][j].get('lyrics'):
                                        ln['segments'] = oseg
                    lm['edited'] = True
                    new_chart['library_match'] = lm
                    tmp = str(chart_path) + '.new'
                    with open(tmp, 'w') as f:
                        _json.dump(new_chart, f, indent=2)
                    import os as _os
                    _os.replace(tmp, str(chart_path))
                    refreshed = True
        except Exception as e:
            logger.warning(f"chart refresh after edit failed for job {job_id}: {e}")
            refresh_err = str(e)

    return jsonify({'ok': True, 'chart_id': chart_id, 'refreshed_job': refreshed,
                    'refresh_error': refresh_err})
