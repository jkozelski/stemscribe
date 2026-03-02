"""
Cloudflare R2 storage client (S3-compatible via boto3).

Handles upload, download (presigned URLs), listing, and deletion of
stems, MIDI, Guitar Pro, and source audio files.

Environment variables:
    R2_ACCESS_KEY_ID      — R2 API token key ID
    R2_SECRET_ACCESS_KEY  — R2 API token secret
    R2_BUCKET_NAME        — Bucket name (default: stemscribe-uploads)
    CF_ACCOUNT_ID         — Cloudflare account ID (for endpoint URL)

Object key layout:
    {job_id}/source.{ext}           — original uploaded audio
    {job_id}/stems/{name}.wav       — separated stems
    {job_id}/midi/{name}.mid        — MIDI transcriptions
    {job_id}/gp/{name}.gp5          — Guitar Pro files
    {job_id}/chords.json            — chord analysis
"""

import os
import logging
import mimetypes
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

_client = None


# ============ CLIENT SETUP ============

def _get_client():
    """Lazily initialize and return the R2 S3 client."""
    global _client
    if _client is not None:
        return _client

    account_id = os.environ.get('CF_ACCOUNT_ID')
    access_key = os.environ.get('R2_ACCESS_KEY_ID')
    secret_key = os.environ.get('R2_SECRET_ACCESS_KEY')

    if not all([account_id, access_key, secret_key]):
        raise RuntimeError(
            "R2 storage not configured. Set CF_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "and R2_SECRET_ACCESS_KEY environment variables."
        )

    _client = boto3.client(
        's3',
        endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name='auto',
        config=Config(
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'},
        ),
    )
    logger.info(f"R2 client initialized (account={account_id[:8]}...)")
    return _client


def _bucket():
    return os.environ.get('R2_BUCKET_NAME', 'stemscribe-uploads')


def is_r2_configured() -> bool:
    """Check if R2 environment variables are set (without initializing client)."""
    return all([
        os.environ.get('CF_ACCOUNT_ID'),
        os.environ.get('R2_ACCESS_KEY_ID'),
        os.environ.get('R2_SECRET_ACCESS_KEY'),
    ])


# ============ KEY BUILDERS ============

def source_key(job_id: str, filename: str) -> str:
    """Key for the original uploaded audio file."""
    ext = Path(filename).suffix or '.mp3'
    return f"{job_id}/source{ext}"


def stem_key(job_id: str, stem_name: str) -> str:
    """Key for a separated stem WAV file."""
    return f"{job_id}/stems/{stem_name}.wav"


def midi_key(job_id: str, stem_name: str) -> str:
    """Key for a MIDI transcription file."""
    return f"{job_id}/midi/{stem_name}.mid"


def gp_key(job_id: str, stem_name: str) -> str:
    """Key for a Guitar Pro file."""
    return f"{job_id}/gp/{stem_name}.gp5"


def chords_key(job_id: str) -> str:
    """Key for the chord analysis JSON."""
    return f"{job_id}/chords.json"


# ============ UPLOAD ============

def upload_file(local_path: str, key: str, content_type: str = None) -> str:
    """Upload a local file to R2.

    Args:
        local_path: Path to the local file.
        key: R2 object key (e.g. "{job_id}/stems/vocals.wav").
        content_type: MIME type. Auto-detected if not provided.

    Returns:
        The object key that was uploaded.

    Raises:
        ClientError: If upload fails.
        FileNotFoundError: If local_path doesn't exist.
    """
    client = _get_client()
    bucket = _bucket()
    path = Path(local_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {local_path}")

    if not content_type:
        content_type, _ = mimetypes.guess_type(str(path))
        content_type = content_type or 'application/octet-stream'

    extra_args = {'ContentType': content_type}

    client.upload_file(
        Filename=str(path),
        Bucket=bucket,
        Key=key,
        ExtraArgs=extra_args,
    )
    logger.info(f"Uploaded {path.name} → r2://{bucket}/{key} ({content_type})")
    return key


def upload_bytes(data: bytes, key: str, content_type: str = 'application/octet-stream') -> str:
    """Upload raw bytes to R2.

    Args:
        data: The bytes to upload.
        key: R2 object key.
        content_type: MIME type.

    Returns:
        The object key.
    """
    client = _get_client()
    bucket = _bucket()

    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    logger.info(f"Uploaded {len(data)} bytes → r2://{bucket}/{key}")
    return key


def upload_job_stems(job_id: str, stems_dir: str) -> dict:
    """Upload all stem WAV files from a local directory to R2.

    Args:
        job_id: The processing job ID.
        stems_dir: Local directory containing stem .wav files.

    Returns:
        Dict mapping stem_name → R2 key, e.g. {"vocals": "{job_id}/stems/vocals.wav"}.
    """
    stems_path = Path(stems_dir)
    uploaded = {}

    for wav_file in sorted(stems_path.glob('*.wav')):
        name = wav_file.stem  # e.g. "vocals", "drums"
        key = stem_key(job_id, name)
        upload_file(str(wav_file), key, content_type='audio/wav')
        uploaded[name] = key

    logger.info(f"Uploaded {len(uploaded)} stems for job {job_id}")
    return uploaded


def upload_job_midis(job_id: str, midi_dir: str) -> dict:
    """Upload all MIDI files from a local directory to R2.

    Args:
        job_id: The processing job ID.
        midi_dir: Local directory containing .mid files.

    Returns:
        Dict mapping stem_name → R2 key.
    """
    midi_path = Path(midi_dir)
    uploaded = {}

    for mid_file in sorted(midi_path.glob('*.mid')):
        name = mid_file.stem
        key = midi_key(job_id, name)
        upload_file(str(mid_file), key, content_type='audio/midi')
        uploaded[name] = key

    return uploaded


def upload_job_gp_files(job_id: str, gp_dir: str) -> dict:
    """Upload all Guitar Pro files from a local directory to R2.

    Args:
        job_id: The processing job ID.
        gp_dir: Local directory containing .gp5 files.

    Returns:
        Dict mapping stem_name → R2 key.
    """
    gp_path = Path(gp_dir)
    uploaded = {}

    for gp_file in sorted(gp_path.glob('*.gp5')):
        name = gp_file.stem
        key = gp_key(job_id, name)
        upload_file(str(gp_file), key, content_type='application/octet-stream')
        uploaded[name] = key

    return uploaded


# ============ PRESIGNED URLs ============

def presigned_upload_url(key: str, content_type: str = 'application/octet-stream',
                         expires_in: int = 300) -> str:
    """Generate a presigned PUT URL for direct client-to-R2 upload.

    Args:
        key: R2 object key.
        content_type: Expected MIME type of the upload.
        expires_in: URL expiry in seconds (default 5 minutes).

    Returns:
        Presigned URL string.
    """
    client = _get_client()
    url = client.generate_presigned_url(
        'put_object',
        Params={
            'Bucket': _bucket(),
            'Key': key,
            'ContentType': content_type,
        },
        ExpiresIn=expires_in,
    )
    return url


def presigned_download_url(key: str, expires_in: int = 3600,
                           filename: str = None) -> str:
    """Generate a presigned GET URL for downloading a file from R2.

    Args:
        key: R2 object key.
        expires_in: URL expiry in seconds (default 1 hour).
        filename: If set, adds Content-Disposition header for browser download.

    Returns:
        Presigned URL string.
    """
    client = _get_client()
    params = {
        'Bucket': _bucket(),
        'Key': key,
    }
    if filename:
        params['ResponseContentDisposition'] = f'attachment; filename="{filename}"'

    url = client.generate_presigned_url(
        'get_object',
        Params=params,
        ExpiresIn=expires_in,
    )
    return url


def download_url(key: str, **kwargs) -> str:
    """Alias for presigned_download_url (matches INFRA_PLAN naming)."""
    return presigned_download_url(key, **kwargs)


# ============ DOWNLOAD TO LOCAL ============

def download_file(key: str, local_path: str) -> str:
    """Download a file from R2 to a local path.

    Args:
        key: R2 object key.
        local_path: Where to save the file locally.

    Returns:
        The local path.
    """
    client = _get_client()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)

    client.download_file(
        Bucket=_bucket(),
        Key=key,
        Filename=local_path,
    )
    logger.info(f"Downloaded r2://{_bucket()}/{key} → {local_path}")
    return local_path


# ============ LIST ============

def list_job_files(job_id: str) -> list[dict]:
    """List all files for a given job in R2.

    Args:
        job_id: The processing job ID.

    Returns:
        List of dicts with 'key', 'size', 'last_modified' for each object.
    """
    client = _get_client()
    prefix = f"{job_id}/"
    result = []

    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
        for obj in page.get('Contents', []):
            result.append({
                'key': obj['Key'],
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat(),
            })

    return result


def file_exists(key: str) -> bool:
    """Check if a file exists in R2."""
    client = _get_client()
    try:
        client.head_object(Bucket=_bucket(), Key=key)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            return False
        raise


def get_file_size(key: str) -> int | None:
    """Get file size in bytes. Returns None if file doesn't exist."""
    client = _get_client()
    try:
        resp = client.head_object(Bucket=_bucket(), Key=key)
        return resp['ContentLength']
    except ClientError:
        return None


# ============ DELETE ============

def delete_file(key: str) -> bool:
    """Delete a single file from R2.

    Returns True if deleted, False if it didn't exist.
    """
    client = _get_client()
    try:
        client.delete_object(Bucket=_bucket(), Key=key)
        logger.info(f"Deleted r2://{_bucket()}/{key}")
        return True
    except ClientError as e:
        logger.warning(f"Failed to delete {key}: {e}")
        return False


def delete_job_files(job_id: str) -> int:
    """Delete ALL files for a job from R2.

    Args:
        job_id: The processing job ID.

    Returns:
        Number of objects deleted.
    """
    client = _get_client()
    files = list_job_files(job_id)

    if not files:
        return 0

    # S3 delete_objects accepts up to 1000 keys at a time
    keys = [{'Key': f['key']} for f in files]
    deleted = 0

    for i in range(0, len(keys), 1000):
        batch = keys[i:i + 1000]
        resp = client.delete_objects(
            Bucket=_bucket(),
            Delete={'Objects': batch, 'Quiet': True},
        )
        deleted += len(batch) - len(resp.get('Errors', []))

    logger.info(f"Deleted {deleted} files for job {job_id}")
    return deleted


# ============ CONVENIENCE: FULL JOB UPLOAD ============

def upload_job_results(job_id: str, output_dir: str) -> dict:
    """Upload all results for a completed job to R2.

    Looks for stems/, midi/, guitarpro/ subdirectories and chords.json
    in the given output directory.

    Args:
        job_id: The processing job ID.
        output_dir: Local output directory (e.g. outputs/{job_id}/).

    Returns:
        Dict with 'stems', 'midi', 'gp', 'chords' keys mapping to R2 keys.
    """
    base = Path(output_dir)
    result = {'stems': {}, 'midi': {}, 'gp': {}, 'chords': None}

    # Upload stems
    stems_dir = base / 'stems'
    if not stems_dir.exists():
        # Some jobs put stems directly in the output dir
        stems_dir = base
    wav_files = list(stems_dir.glob('*.wav'))
    if wav_files:
        result['stems'] = upload_job_stems(job_id, str(stems_dir))

    # Upload MIDI files
    midi_dir = base / 'midi'
    if midi_dir.exists():
        result['midi'] = upload_job_midis(job_id, str(midi_dir))

    # Upload Guitar Pro files
    gp_dir = base / 'guitarpro'
    if gp_dir.exists():
        result['gp'] = upload_job_gp_files(job_id, str(gp_dir))

    # Upload chords.json
    chords_file = base / 'chords.json'
    if chords_file.exists():
        key = chords_key(job_id)
        upload_file(str(chords_file), key, content_type='application/json')
        result['chords'] = key

    total = sum(len(v) if isinstance(v, dict) else (1 if v else 0) for v in result.values())
    logger.info(f"Uploaded {total} files for job {job_id} to R2")
    return result
