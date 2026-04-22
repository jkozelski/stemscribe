"""
Google Drive export — per-user OAuth flow for on-demand stem uploads to
the user's own Google Drive account.

Endpoints:
    GET  /api/drive/auth         → returns Google OAuth redirect URL
    GET  /api/drive/callback     → handles Google redirect, stores tokens
    GET  /api/drive/status       → { connected: bool, email: str }
    POST /api/drive/export       → uploads a processed job's stems
    POST /api/drive/disconnect   → revokes + removes stored tokens

Env (all read lazily so the module can import cleanly without them):
    GOOGLE_CLIENT_ID       (already set for sign-in)
    GOOGLE_CLIENT_SECRET   (NEW — needed for this flow)
    DRIVE_REDIRECT_URI     (optional; defaults to https://stemscriber.com/api/drive/callback)

Storage: Supabase `user_drive_tokens` table (schema in migrations/drive_tokens.sql).

Scope: only `drive.file` (NON-sensitive — Google doesn't require app verification,
and this app can only see/modify files it created).
"""

import os
import logging
import secrets
from pathlib import Path
from functools import wraps

from flask import Blueprint, request, jsonify, redirect

try:
    from google_auth_oauthlib.flow import Flow
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request as GRequest
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False

from auth.middleware import auth_required
from auth.decorators import get_current_user
from db import query_one, execute
from models.job import get_job

logger = logging.getLogger(__name__)

drive_bp = Blueprint('drive', __name__)

SCOPES = ['https://www.googleapis.com/auth/drive.file']
DRIVE_FOLDER_NAME = 'StemScriber'


def _redirect_uri():
    return os.environ.get('DRIVE_REDIRECT_URI', 'https://stemscriber.com/api/drive/callback')


def _client_config():
    """Build the OAuth client_config dict from env vars (no JSON file needed)."""
    client_id = os.environ.get('GOOGLE_CLIENT_ID')
    client_secret = os.environ.get('GOOGLE_CLIENT_SECRET')
    if not client_id or not client_secret:
        return None
    return {
        'web': {
            'client_id': client_id,
            'client_secret': client_secret,
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'redirect_uris': [_redirect_uri()],
        }
    }


def _env_ok(fn):
    """Short-circuit endpoints if env isn't configured."""
    @wraps(fn)
    def wrapper(*a, **kw):
        if not GOOGLE_AVAILABLE:
            return jsonify({
                'error': 'Google API libraries not installed on server',
            }), 503
        if not _client_config():
            return jsonify({
                'error': 'Google Drive integration not configured. GOOGLE_CLIENT_SECRET missing.',
            }), 503
        return fn(*a, **kw)
    return wrapper


# ---------------------------------------------------------------------------
# Token storage
# ---------------------------------------------------------------------------

def _save_tokens(user_id, creds, email=None):
    execute(
        """
        INSERT INTO user_drive_tokens
            (user_id, access_token, refresh_token, email, expires_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE SET
            access_token  = EXCLUDED.access_token,
            refresh_token = COALESCE(EXCLUDED.refresh_token, user_drive_tokens.refresh_token),
            email         = COALESCE(EXCLUDED.email, user_drive_tokens.email),
            expires_at    = EXCLUDED.expires_at,
            updated_at    = NOW()
        """,
        (str(user_id), creds.token, creds.refresh_token, email, creds.expiry),
    )


def _load_tokens(user_id):
    return query_one(
        "SELECT access_token, refresh_token, email, expires_at "
        "FROM user_drive_tokens WHERE user_id = %s",
        (str(user_id),),
    )


def _delete_tokens(user_id):
    execute("DELETE FROM user_drive_tokens WHERE user_id = %s", (str(user_id),))


def _get_user_creds(user_id):
    """Valid Credentials for this user, refreshing if needed. None if not connected."""
    row = _load_tokens(user_id)
    if not row:
        return None

    cfg = _client_config()
    if not cfg:
        return None

    creds = Credentials(
        token=row['access_token'],
        refresh_token=row['refresh_token'],
        client_id=cfg['web']['client_id'],
        client_secret=cfg['web']['client_secret'],
        token_uri=cfg['web']['token_uri'],
        scopes=SCOPES,
    )

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(GRequest())
            _save_tokens(user_id, creds, email=row.get('email'))
        except Exception as e:
            logger.error(f"Drive token refresh failed for user {user_id}: {e}")
            return None

    return creds


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@drive_bp.route('/api/drive/auth', methods=['GET'])
@_env_ok
@auth_required
def drive_auth():
    """Return the Google OAuth redirect URL for the frontend to open."""
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=_redirect_uri())

    nonce = secrets.token_urlsafe(24)
    state = f"{user.id}:{nonce}"

    execute(
        """
        INSERT INTO user_drive_oauth_state (user_id, nonce, created_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (user_id) DO UPDATE SET nonce = EXCLUDED.nonce, created_at = NOW()
        """,
        (str(user.id), nonce),
    )

    auth_url, _ = flow.authorization_url(
        access_type='offline',
        include_granted_scopes='true',
        prompt='consent',
        state=state,
    )

    return jsonify({'auth_url': auth_url})


@drive_bp.route('/api/drive/callback', methods=['GET'])
@_env_ok
def drive_callback():
    """Google redirects here after the user approves."""
    state = request.args.get('state', '')
    code = request.args.get('code')
    error = request.args.get('error')

    if error:
        return redirect(f'/app?drive=error&reason={error}')

    if not code or ':' not in state:
        return redirect('/app?drive=error&reason=invalid_state')

    user_id, nonce = state.split(':', 1)

    stored = query_one(
        "SELECT nonce FROM user_drive_oauth_state WHERE user_id = %s",
        (user_id,),
    )
    if not stored or stored['nonce'] != nonce:
        return redirect('/app?drive=error&reason=state_mismatch')

    try:
        flow = Flow.from_client_config(_client_config(), scopes=SCOPES, redirect_uri=_redirect_uri())
        flow.fetch_token(code=code)
        creds = flow.credentials

        email = None
        try:
            service = build('oauth2', 'v2', credentials=creds, cache_discovery=False)
            info = service.userinfo().get().execute()
            email = info.get('email')
        except Exception:
            pass

        _save_tokens(user_id, creds, email=email)
        execute("DELETE FROM user_drive_oauth_state WHERE user_id = %s", (user_id,))

        return redirect('/app?drive=connected')
    except Exception:
        logger.exception("Drive OAuth callback failed")
        return redirect('/app?drive=error&reason=callback_failed')


@drive_bp.route('/api/drive/status', methods=['GET'])
@_env_ok
@auth_required
def drive_status():
    user = get_current_user()
    if not user:
        return jsonify({'connected': False})
    row = _load_tokens(user.id)
    if not row:
        return jsonify({'connected': False})
    return jsonify({
        'connected': True,
        'email': row.get('email'),
    })


@drive_bp.route('/api/drive/export', methods=['POST'])
@_env_ok
@auth_required
def drive_export():
    """Upload a processed job's stems to the user's Drive. Body: { job_id }."""
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')
    if not job_id:
        return jsonify({'error': 'job_id required'}), 400

    creds = _get_user_creds(user.id)
    if not creds:
        return jsonify({'error': 'Not connected to Google Drive'}), 401

    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404

    # Authorize — ensure this job belongs to this user if ownership is tracked
    job_user = (job.metadata or {}).get('user_id')
    if job_user and str(job_user) != str(user.id):
        return jsonify({'error': 'Not authorized for this job'}), 403

    try:
        drive = build('drive', 'v3', credentials=creds, cache_discovery=False)
        parent_id = _ensure_folder(drive, DRIVE_FOLDER_NAME)

        song_name = (job.metadata or {}).get('title') or job.filename or job_id
        safe_name = (song_name.replace('/', '-') or job_id).strip()[:120]
        song_folder_id = _ensure_folder(drive, safe_name, parent_id=parent_id)

        uploaded = []
        for stem_name, stem_path in (job.stems or {}).items():
            if not stem_path:
                continue
            p = Path(stem_path)
            if not p.exists():
                continue
            media = MediaFileUpload(str(p), resumable=True)
            f = drive.files().create(
                body={'name': p.name, 'parents': [song_folder_id]},
                media_body=media,
                fields='id, webViewLink',
            ).execute()
            uploaded.append({'stem': stem_name, 'name': p.name, 'id': f.get('id')})

        folder = drive.files().get(fileId=song_folder_id, fields='id, webViewLink').execute()

        return jsonify({
            'status': 'ok',
            'folder_url': folder.get('webViewLink'),
            'uploaded': uploaded,
            'count': len(uploaded),
        })

    except Exception as e:
        logger.exception(f"Drive export failed for job {job_id}")
        return jsonify({'error': f'Export failed: {e}'}), 500


@drive_bp.route('/api/drive/disconnect', methods=['POST'])
@_env_ok
@auth_required
def drive_disconnect():
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Authentication required'}), 401

    row = _load_tokens(user.id)
    if row and row.get('access_token'):
        try:
            import urllib.request
            urllib.request.urlopen(
                f"https://oauth2.googleapis.com/revoke?token={row['access_token']}",
                timeout=5,
            )
        except Exception as e:
            logger.warning(f"Drive revoke call failed (non-fatal): {e}")

    _delete_tokens(user.id)
    return jsonify({'status': 'disconnected'})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_folder(drive, name, parent_id=None):
    """Return the ID of the named folder, creating if missing."""
    safe = name.replace("'", "\\'")
    q = [
        f"name = '{safe}'",
        "mimeType = 'application/vnd.google-apps.folder'",
        "trashed = false",
    ]
    if parent_id:
        q.append(f"'{parent_id}' in parents")

    results = drive.files().list(
        q=' and '.join(q),
        spaces='drive',
        fields='files(id, name)',
        pageSize=1,
    ).execute()

    if results.get('files'):
        return results['files'][0]['id']

    body = {'name': name, 'mimeType': 'application/vnd.google-apps.folder'}
    if parent_id:
        body['parents'] = [parent_id]
    folder = drive.files().create(body=body, fields='id').execute()
    return folder['id']
