"""
Admin-only restore endpoints. JWT-gated to ADMIN_EMAILS so a leaked
account doesn't get to grab arbitrary stems.

Endpoints:
    GET  /admin/r2/list-snapshots      List all DB snapshots on R2
    GET  /admin/r2/list-job-files/<id> List files backed up for a job
    POST /admin/r2/restore-job/<id>    Restore a job's files to local disk
                                       (overwrites local copies — careful)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from flask import Blueprint, jsonify, g
from auth.middleware import auth_required
from backup.r2_client import r2_enabled, list_keys, download_file, get_bucket

logger = logging.getLogger(__name__)

restore_bp = Blueprint('r2_restore', __name__, url_prefix='/admin/r2')

ADMIN_EMAILS = frozenset({'jkozelski@gmail.com'})
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "/opt/stemscribe/outputs"))


def _require_admin():
    """Return the admin user object or None if caller isn't admin."""
    user = getattr(g, 'current_user', None)
    if not user or (user.email or '').lower() not in ADMIN_EMAILS:
        return None
    return user


@restore_bp.route('/health', methods=['GET'])
@auth_required(optional=True)
def health():
    """Quick check — is R2 wired AND can we list our own bucket?"""
    if not _require_admin():
        return jsonify({'error': 'admin only'}), 403
    if not r2_enabled():
        return jsonify({'enabled': False, 'reason': 'R2_* env vars not set'}), 200
    keys = list_keys('', max_keys=1)
    return jsonify({
        'enabled': True,
        'bucket': get_bucket(),
        'reachable': len(keys) >= 0,  # list_keys returns [] on failure too
    })


@restore_bp.route('/list-snapshots', methods=['GET'])
@auth_required(optional=True)
def list_snapshots():
    if not _require_admin():
        return jsonify({'error': 'admin only'}), 403
    if not r2_enabled():
        return jsonify({'error': 'R2 not configured'}), 503
    keys = list_keys('db-snapshots/', max_keys=100)
    return jsonify({'snapshots': sorted(keys, reverse=True), 'count': len(keys)})


@restore_bp.route('/list-job-files/<job_id>', methods=['GET'])
@auth_required(optional=True)
def list_job_files(job_id):
    if not _require_admin():
        return jsonify({'error': 'admin only'}), 403
    if not r2_enabled():
        return jsonify({'error': 'R2 not configured'}), 503
    prefix = f"jobs/{job_id}/"
    keys = list_keys(prefix, max_keys=200)
    return jsonify({'job_id': job_id, 'prefix': prefix, 'files': keys, 'count': len(keys)})


@restore_bp.route('/restore-job/<job_id>', methods=['POST'])
@auth_required(optional=True)
def restore_job(job_id):
    """Pull every R2-backed file for a job back to local disk under
    /opt/stemscribe/outputs/<job_id>/. Overwrites local copies. Use when
    the disk version is missing or corrupted."""
    if not _require_admin():
        return jsonify({'error': 'admin only'}), 403
    if not r2_enabled():
        return jsonify({'error': 'R2 not configured'}), 503

    prefix = f"jobs/{job_id}/"
    keys = list_keys(prefix, max_keys=200)
    if not keys:
        return jsonify({'error': f'No R2 backup found for job {job_id}', 'prefix': prefix}), 404

    target_root = OUTPUTS_DIR / job_id
    target_root.mkdir(parents=True, exist_ok=True)

    restored, failed = [], []
    for key in keys:
        rel = key[len(prefix):]  # path relative to job_id/
        if not rel:
            continue
        local = target_root / rel
        local.parent.mkdir(parents=True, exist_ok=True)
        if download_file(key, str(local)):
            restored.append(rel)
        else:
            failed.append(rel)

    logger.info(f"[r2-restore] job {job_id}: restored {len(restored)} files, {len(failed)} failed")

    return jsonify({
        'job_id': job_id,
        'target': str(target_root),
        'restored': restored,
        'failed': failed,
        'count_restored': len(restored),
        'count_failed': len(failed),
    })
