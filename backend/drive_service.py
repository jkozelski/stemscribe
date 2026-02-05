"""
Google Drive Integration for StemScribe
Handles uploading transcriptions and cleaning up old files
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    logger.warning("Google API libraries not installed. Run: pip install google-auth-oauthlib google-api-python-client --break-system-packages")

# Scopes for Google Drive
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Paths
SCRIPT_DIR = Path(__file__).parent.parent.absolute()
CREDENTIALS_FILE = SCRIPT_DIR / 'credentials.json'
TOKEN_FILE = SCRIPT_DIR / 'token.json'

# StemScribe folder name in Google Drive
DRIVE_FOLDER_NAME = 'StemScribe Transcriptions'


def get_drive_service():
    """Get authenticated Google Drive service"""
    if not GOOGLE_API_AVAILABLE:
        logger.error("Google API libraries not available")
        return None

    if not CREDENTIALS_FILE.exists():
        logger.error(f"Credentials file not found: {CREDENTIALS_FILE}")
        return None

    creds = None

    # Load existing token if available
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception as e:
            logger.warning(f"Could not load token: {e}")

    # Refresh or get new credentials
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Token refresh failed: {e}")
                creds = None

        if not creds:
            try:
                flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
                creds = flow.run_local_server(port=8090)
            except Exception as e:
                logger.error(f"OAuth flow failed: {e}")
                return None

        # Save token for next time
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('drive', 'v3', credentials=creds)
        return service
    except Exception as e:
        logger.error(f"Failed to build Drive service: {e}")
        return None


def get_or_create_folder(service, folder_name, parent_id=None):
    """Get or create a folder in Google Drive"""
    # Search for existing folder
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    folders = results.get('files', [])

    if folders:
        return folders[0]['id']

    # Create new folder
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        file_metadata['parents'] = [parent_id]

    folder = service.files().create(body=file_metadata, fields='id').execute()
    logger.info(f"Created Drive folder: {folder_name}")
    return folder.get('id')


def upload_file_to_drive(service, file_path, folder_id, filename=None):
    """Upload a file to Google Drive"""
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return None

    filename = filename or file_path.name

    # Determine MIME type
    mime_types = {
        '.mid': 'audio/midi',
        '.midi': 'audio/midi',
        '.musicxml': 'application/vnd.recordare.musicxml+xml',
        '.xml': 'application/xml',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.json': 'application/json',
    }
    mime_type = mime_types.get(file_path.suffix.lower(), 'application/octet-stream')

    file_metadata = {
        'name': filename,
        'parents': [folder_id]
    }

    media = MediaFileUpload(str(file_path), mimetype=mime_type, resumable=True)

    try:
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink'
        ).execute()
        logger.info(f"Uploaded to Drive: {filename}")
        return file
    except Exception as e:
        logger.error(f"Upload failed for {filename}: {e}")
        return None


def upload_job_to_drive(job, keep_stems=False):
    """
    Upload a completed job's transcriptions to Google Drive

    Args:
        job: ProcessingJob instance
        keep_stems: If True, also upload stem audio files (larger)

    Returns:
        dict with upload results or None if failed
    """
    service = get_drive_service()
    if not service:
        logger.error("Could not get Drive service")
        return None

    try:
        # Get or create main StemScribe folder
        main_folder_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)

        # Create folder for this song
        song_name = job.metadata.get('title', job.filename or job.job_id)
        # Sanitize folder name
        song_name = "".join(c for c in song_name if c.isalnum() or c in ' -_')[:50]
        song_folder_id = get_or_create_folder(service, f"{song_name} ({job.job_id})", main_folder_id)

        results = {
            'folder_id': song_folder_id,
            'uploaded_files': [],
            'errors': []
        }

        # Upload MusicXML files (small, notation data)
        for stem_name, xml_path in job.musicxml_files.items():
            file = upload_file_to_drive(service, xml_path, song_folder_id, f"{stem_name}.musicxml")
            if file:
                results['uploaded_files'].append({
                    'type': 'musicxml',
                    'stem': stem_name,
                    'drive_id': file.get('id'),
                    'link': file.get('webViewLink')
                })
            else:
                results['errors'].append(f"Failed to upload {stem_name}.musicxml")

        # Upload MIDI files (tiny)
        for stem_name, midi_path in job.midi_files.items():
            file = upload_file_to_drive(service, midi_path, song_folder_id, f"{stem_name}.mid")
            if file:
                results['uploaded_files'].append({
                    'type': 'midi',
                    'stem': stem_name,
                    'drive_id': file.get('id'),
                    'link': file.get('webViewLink')
                })
            else:
                results['errors'].append(f"Failed to upload {stem_name}.mid")

        # Upload Guitar Pro files (tablature) - these are what you want for learning!
        if hasattr(job, 'gp_files') and job.gp_files:
            for stem_name, gp_path in job.gp_files.items():
                file = upload_file_to_drive(service, gp_path, song_folder_id, f"{stem_name}.gp5")
                if file:
                    results['uploaded_files'].append({
                        'type': 'guitarpro',
                        'stem': stem_name,
                        'drive_id': file.get('id'),
                        'link': file.get('webViewLink')
                    })
                else:
                    results['errors'].append(f"Failed to upload {stem_name}.gp5")

        # Optionally upload stem audio files (larger)
        if keep_stems:
            for stem_name, stem_path in job.stems.items():
                file = upload_file_to_drive(service, stem_path, song_folder_id, f"{stem_name}.mp3")
                if file:
                    results['uploaded_files'].append({
                        'type': 'stem',
                        'stem': stem_name,
                        'drive_id': file.get('id'),
                        'link': file.get('webViewLink')
                    })
                else:
                    results['errors'].append(f"Failed to upload {stem_name}.mp3")

        # Save job metadata
        metadata = {
            'job_id': job.job_id,
            'title': job.metadata.get('title'),
            'artist': job.metadata.get('artist'),
            'source_url': job.source_url,
            'created_at': datetime.now().isoformat(),
            'stems': list(job.stems.keys()),
            'midi_files': list(job.midi_files.keys()),
            'musicxml_files': list(job.musicxml_files.keys()),
            'gp_files': list(job.gp_files.keys()) if hasattr(job, 'gp_files') and job.gp_files else []
        }

        # Write metadata to temp file and upload
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metadata, f, indent=2)
            metadata_path = f.name

        file = upload_file_to_drive(service, metadata_path, song_folder_id, 'metadata.json')
        os.unlink(metadata_path)

        if file:
            results['uploaded_files'].append({
                'type': 'metadata',
                'drive_id': file.get('id'),
                'link': file.get('webViewLink')
            })

        logger.info(f"Uploaded job {job.job_id} to Drive: {len(results['uploaded_files'])} files")
        return results

    except Exception as e:
        logger.error(f"Drive upload failed: {e}")
        return None


def cleanup_old_stems(output_dir, max_age_days=7):
    """
    Delete stem audio files older than max_age_days to save disk space.
    Keeps MIDI and MusicXML files (small).

    Args:
        output_dir: Path to outputs directory
        max_age_days: Delete stems older than this many days

    Returns:
        dict with cleanup results
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return {'deleted': 0, 'freed_bytes': 0}

    cutoff = datetime.now() - timedelta(days=max_age_days)
    deleted = 0
    freed_bytes = 0

    # Find all stem directories
    for job_dir in output_dir.iterdir():
        if not job_dir.is_dir():
            continue

        stems_dir = job_dir / 'stems'
        if not stems_dir.exists():
            continue

        # Check modification time of stems directory
        mtime = datetime.fromtimestamp(stems_dir.stat().st_mtime)
        if mtime < cutoff:
            # Delete all audio files in stems directory
            for stem_file in stems_dir.rglob('*.mp3'):
                try:
                    freed_bytes += stem_file.stat().st_size
                    stem_file.unlink()
                    deleted += 1
                    logger.info(f"Deleted old stem: {stem_file}")
                except Exception as e:
                    logger.error(f"Failed to delete {stem_file}: {e}")

            for stem_file in stems_dir.rglob('*.wav'):
                try:
                    freed_bytes += stem_file.stat().st_size
                    stem_file.unlink()
                    deleted += 1
                except Exception as e:
                    logger.error(f"Failed to delete {stem_file}: {e}")

    logger.info(f"Cleanup complete: deleted {deleted} files, freed {freed_bytes / 1024 / 1024:.1f} MB")
    return {
        'deleted': deleted,
        'freed_bytes': freed_bytes,
        'freed_mb': round(freed_bytes / 1024 / 1024, 1)
    }


def get_drive_stats():
    """Get info about StemScribe folder in Google Drive"""
    service = get_drive_service()
    if not service:
        return None

    try:
        # Find main folder
        query = f"name='{DRIVE_FOLDER_NAME}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        folders = results.get('files', [])

        if not folders:
            return {'exists': False, 'file_count': 0, 'folder_count': 0}

        folder_id = folders[0]['id']

        # Count contents
        query = f"'{folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name, mimeType, size)').execute()
        files = results.get('files', [])

        file_count = sum(1 for f in files if f.get('mimeType') != 'application/vnd.google-apps.folder')
        folder_count = sum(1 for f in files if f.get('mimeType') == 'application/vnd.google-apps.folder')
        total_size = sum(int(f.get('size', 0)) for f in files)

        return {
            'exists': True,
            'folder_id': folder_id,
            'file_count': file_count,
            'folder_count': folder_count,
            'total_size_mb': round(total_size / 1024 / 1024, 1)
        }

    except Exception as e:
        logger.error(f"Failed to get Drive stats: {e}")
        return None


# Test function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    print("Testing Google Drive connection...")
    service = get_drive_service()

    if service:
        print("✓ Connected to Google Drive!")
        stats = get_drive_stats()
        print(f"Stats: {stats}")
    else:
        print("✗ Failed to connect to Google Drive")
