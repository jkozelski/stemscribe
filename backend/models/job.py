"""
ProcessingJob model and job persistence.

Single source of truth for job state, persistence to disk, and the global jobs registry.
"""

import json
import time
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ============ SHARED STATE ============

# Directories - relative to script location
SCRIPT_DIR = Path(__file__).parent.parent.parent.absolute()
UPLOAD_DIR = SCRIPT_DIR / 'uploads'
OUTPUT_DIR = SCRIPT_DIR / 'outputs'
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job registry (evicts oldest completed jobs when exceeding MAX_CACHED_JOBS)
MAX_CACHED_JOBS = 100
jobs = {}


# ============ NUMPY SERIALIZATION HELPER ============

def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj


# ============ JOB MODEL ============

class ProcessingJob:
    def __init__(self, job_id, filename, source_url=None, skills=None):
        self.job_id = job_id
        self.filename = filename
        self.source_url = source_url
        self.status = 'pending'
        self.progress = 0
        self.stage = 'Initializing'
        self.stems = {}
        self.enhanced_stems = {}  # Enhanced versions of stems
        self.midi_files = {}
        self.musicxml_files = {}
        self.gp_files = {}       # Guitar Pro files for tablature notation
        self.sub_stems = {}  # Skills-generated sub-stems: {skill_id: {sub_stem_name: path}}
        self.selected_skills = skills or []  # Skills to apply
        self.error = None
        self.created_at = time.time()
        self.metadata = {}  # Store title, artist, duration, etc.
        self.chord_progression = []  # Detected chords with timestamps
        self.detected_key = None     # Detected musical key
        self.articulations = {}      # Per-stem articulation counts
        self.transcription_quality = {}  # Quality scores per stem
        self.pro_tabs = {}           # Professional tabs from Songsterr/UG
        self.transcription_mode = {}  # Per-stem: 'melody', 'enhanced', 'basic_pitch'
        self.user_id = None       # Owner's user ID (for logged-in users)
        self.session_id = None    # Anonymous session tracking cookie

    def to_dict(self):
        """Convert job to dict with numpy types converted for JSON serialization"""
        data = {
            'job_id': self.job_id,
            'filename': self.filename,
            'source_url': self.source_url,
            'status': self.status,
            'progress': self.progress,
            'stage': self.stage,
            'stems': self.stems,
            'enhanced_stems': self.enhanced_stems,
            'midi_files': self.midi_files,
            'musicxml_files': self.musicxml_files,
            'gp_files': self.gp_files,
            'sub_stems': self.sub_stems,
            'selected_skills': self.selected_skills,
            'error': self.error,
            'metadata': self.metadata,
            'chord_progression': self.chord_progression,
            'detected_key': self.detected_key,
            'transcription_quality': self.transcription_quality,
            'pro_tabs': self.pro_tabs,
            'transcription_mode': self.transcription_mode,
            'user_id': self.user_id,
            'session_id': self.session_id,
        }
        return convert_numpy_types(data)


# ============ JOB PERSISTENCE ============

def save_job_to_disk(job):
    """Save job metadata to disk for persistence across server restarts"""
    try:
        job_file = OUTPUT_DIR / job.job_id / 'job_metadata.json'
        job_data = job.to_dict()
        job_data['saved_at'] = time.time()

        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2, default=str)

        logger.info(f"Saved job metadata: {job.job_id}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save job metadata: {e}")
        return False


def save_job_checkpoint(job):
    """Save job state at each stage transition to prevent data loss on crash.
    Cheaper than save_job_to_disk -- skips log line, tolerates missing output dir."""
    try:
        job_dir = OUTPUT_DIR / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / 'job_metadata.json'
        job_data = job.to_dict()
        job_data['saved_at'] = time.time()
        with open(job_file, 'w') as f:
            json.dump(job_data, f, indent=2, default=str)
    except Exception:
        pass  # Non-fatal: best-effort checkpoint


def load_job_from_disk(job_dir: Path):
    """Load a job from disk"""
    try:
        job_file = job_dir / 'job_metadata.json'
        if not job_file.exists():
            return None

        with open(job_file, 'r') as f:
            data = json.load(f)

        # Recreate the job object
        job = ProcessingJob(
            job_id=data['job_id'],
            filename=data.get('filename', 'Unknown'),
            source_url=data.get('source_url')
        )

        # Restore all attributes
        job.status = data.get('status', 'completed')
        job.progress = data.get('progress', 100)
        job.stage = data.get('stage', 'Complete')
        job.stems = data.get('stems', {})
        job.enhanced_stems = data.get('enhanced_stems', {})
        job.midi_files = data.get('midi_files', {})
        job.musicxml_files = data.get('musicxml_files', {})
        job.gp_files = data.get('gp_files', {})
        job.sub_stems = data.get('sub_stems', {})
        job.selected_skills = data.get('selected_skills', [])
        job.error = data.get('error')
        job.created_at = data.get('created_at', time.time())
        job.metadata = data.get('metadata', {})
        job.chord_progression = data.get('chord_progression', [])
        job.detected_key = data.get('detected_key')
        job.transcription_quality = data.get('transcription_quality', {})
        job.pro_tabs = data.get('pro_tabs', {})
        job.user_id = data.get('user_id')
        job.session_id = data.get('session_id')

        # Verify that stem files still exist
        valid_stems = {}
        for stem_name, stem_path in job.stems.items():
            if Path(stem_path).exists():
                valid_stems[stem_name] = stem_path
        job.stems = valid_stems

        # Auto-discover GP files that may have been generated later
        gp_dir = job_dir / 'guitarpro'
        if gp_dir.exists():
            gp_files_found = list(gp_dir.glob('*.gp5'))
            for gp_file in gp_files_found:
                stem_name = gp_file.stem
                if stem_name not in job.gp_files:
                    job.gp_files[stem_name] = str(gp_file)
                    logger.info(f"Auto-discovered GP file: {stem_name}")

        return job
    except Exception as e:
        logger.warning(f"Failed to load job from {job_dir}: {e}")
        return None


def load_all_jobs_from_disk():
    """Load all saved jobs from disk on startup"""
    loaded_count = 0

    if not OUTPUT_DIR.exists():
        return loaded_count

    for job_dir in OUTPUT_DIR.iterdir():
        if job_dir.is_dir():
            job = load_job_from_disk(job_dir)
            if job and job.stems:  # Only load jobs with valid stems
                jobs[job.job_id] = job
                loaded_count += 1

    if loaded_count > 0:
        logger.info(f"Loaded {loaded_count} jobs from library")

    return loaded_count


def _evict_old_jobs():
    """Remove oldest completed jobs from memory when cache is full."""
    if len(jobs) < MAX_CACHED_JOBS:
        return
    completed = [(jid, j) for jid, j in jobs.items() if j.status == 'completed']
    if not completed:
        return
    # Sort by created_at (oldest first) and evict half
    completed.sort(key=lambda x: getattr(x[1], 'created_at', 0))
    evict_count = max(1, len(completed) // 2)
    for jid, _ in completed[:evict_count]:
        del jobs[jid]
    logger.info(f"Evicted {evict_count} completed jobs from memory cache")


def get_job(job_id: str):
    """Get a job from memory, or auto-load from disk if not found"""
    job = None

    # First check memory
    if job_id in jobs:
        job = jobs[job_id]
    else:
        # Try to load from disk
        job_dir = OUTPUT_DIR / job_id
        if job_dir.exists():
            job = load_job_from_disk(job_dir)
            if job:
                _evict_old_jobs()
                jobs[job_id] = job  # Cache it in memory
                logger.info(f"Auto-loaded job {job_id} from disk")

    # Always check for new GP files (they may have been generated after job was loaded)
    if job:
        job_dir = OUTPUT_DIR / job_id
        gp_dir = job_dir / 'guitarpro'
        if gp_dir.exists():
            for gp_file in gp_dir.glob('*.gp5'):
                stem_name = gp_file.stem
                if stem_name not in job.gp_files:
                    job.gp_files[stem_name] = str(gp_file)
                    logger.info(f"Discovered new GP file: {stem_name}")

    return job
