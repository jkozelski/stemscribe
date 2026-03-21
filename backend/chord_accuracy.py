"""
Chord accuracy scoring — compare AI chord detection against ground truth (Songsterr).

Provides:
- score_chord_accuracy(ai_chords, reference_chords) -> accuracy report
- save_accuracy_score(job_id, report) -> persist to disk
- get_accuracy_score(job_id) -> load from disk
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

ACCURACY_FILE = Path(__file__).parent / 'accuracy_scores.json'

# Time tolerance for chord matching (seconds)
TIME_TOLERANCE = 1.0


def _normalize_chord(chord_name: str) -> str:
    """Normalize chord name for comparison.

    - Strip whitespace
    - Map common variants (e.g. Cmaj -> C, Cmin -> Cm)
    - Case-sensitive for note names (C, C#, Db, etc.)
    """
    if not chord_name:
        return ''
    chord = chord_name.strip()

    # Remove 'maj' suffix when it means plain major (Cmaj -> C, but Cmaj7 stays)
    chord = re.sub(r'^([A-G][#b]?)maj$', r'\1', chord)

    # Normalize 'min' to 'm'
    chord = re.sub(r'^([A-G][#b]?)min', r'\1m', chord)

    # Normalize 'mi' to 'm' (Ami -> Am)
    chord = re.sub(r'^([A-G][#b]?)mi$', r'\1m', chord)
    chord = re.sub(r'^([A-G][#b]?)mi(\d)', r'\1m\2', chord)

    return chord


def _find_nearest_chord(time_pos: float, chords: List[Dict], tolerance: float = TIME_TOLERANCE) -> Optional[Dict]:
    """Find the nearest chord in a list within the time tolerance."""
    best = None
    best_dist = float('inf')
    for chord in chords:
        t = chord.get('time', chord.get('start', chord.get('timestamp', 0)))
        dist = abs(t - time_pos)
        if dist < best_dist and dist <= tolerance:
            best = chord
            best_dist = dist
    return best


def score_chord_accuracy(ai_chords: List[Dict], reference_chords: List[Dict],
                         tolerance: float = TIME_TOLERANCE) -> Dict[str, Any]:
    """Compare AI-detected chords against reference (ground truth) chords.

    Args:
        ai_chords: List of dicts with at least 'chord' and 'time'/'start' keys
        reference_chords: List of dicts with at least 'chord' and 'time'/'start' keys
        tolerance: Time tolerance in seconds for position matching

    Returns:
        Dict with accuracy %, correct/wrong/missed/extra counts, and per-chord details.
    """
    correct = []
    wrong = []
    missed = []
    extra = []

    # Track which reference chords have been matched
    matched_ref_indices = set()

    # For each AI chord, try to find a matching reference chord
    for ai_chord in ai_chords:
        ai_name = _normalize_chord(ai_chord.get('chord', ''))
        ai_time = ai_chord.get('time', ai_chord.get('start', ai_chord.get('timestamp', 0)))

        if not ai_name or ai_name.upper() == 'N' or ai_name.upper() == 'NC':
            continue  # Skip "no chord" markers

        nearest_ref = None
        nearest_idx = None
        nearest_dist = float('inf')

        for idx, ref_chord in enumerate(reference_chords):
            if idx in matched_ref_indices:
                continue
            ref_time = ref_chord.get('time', ref_chord.get('start', ref_chord.get('timestamp', 0)))
            dist = abs(ref_time - ai_time)
            if dist < nearest_dist and dist <= tolerance:
                nearest_ref = ref_chord
                nearest_idx = idx
                nearest_dist = dist

        if nearest_ref is not None:
            matched_ref_indices.add(nearest_idx)
            ref_name = _normalize_chord(nearest_ref.get('chord', ''))

            if ai_name == ref_name:
                correct.append({
                    'time': ai_time,
                    'chord': ai_name,
                })
            else:
                wrong.append({
                    'time': ai_time,
                    'ai_chord': ai_name,
                    'reference_chord': ref_name,
                    'time_offset': round(nearest_dist, 3),
                })
        else:
            extra.append({
                'time': ai_time,
                'chord': ai_name,
            })

    # Find reference chords that weren't matched (missed by AI)
    for idx, ref_chord in enumerate(reference_chords):
        if idx in matched_ref_indices:
            continue
        ref_name = _normalize_chord(ref_chord.get('chord', ''))
        ref_time = ref_chord.get('time', ref_chord.get('start', ref_chord.get('timestamp', 0)))
        if ref_name and ref_name.upper() not in ('N', 'NC'):
            missed.append({
                'time': ref_time,
                'chord': ref_name,
            })

    total_reference = len(correct) + len(wrong) + len(missed)
    accuracy_pct = round(100.0 * len(correct) / total_reference, 1) if total_reference > 0 else 0.0

    return {
        'accuracy_percent': accuracy_pct,
        'total_reference_chords': total_reference,
        'total_ai_chords': len(correct) + len(wrong) + len(extra),
        'correct': len(correct),
        'wrong': len(wrong),
        'missed': len(missed),
        'extra': len(extra),
        'tolerance_seconds': tolerance,
        'details': {
            'correct': correct,
            'wrong': wrong,
            'missed': missed,
            'extra': extra,
        },
    }


def save_accuracy_score(job_id: str, report: Dict[str, Any]):
    """Save an accuracy report for a job."""
    try:
        scores = {}
        if ACCURACY_FILE.exists():
            try:
                with open(ACCURACY_FILE, 'r') as f:
                    scores = json.load(f)
            except (json.JSONDecodeError, IOError):
                scores = {}

        scores[job_id] = {
            'scored_at': time.time(),
            **report,
        }

        tmp = ACCURACY_FILE.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            json.dump(scores, f, indent=2, default=str)
        tmp.replace(ACCURACY_FILE)

        logger.info(f"Saved accuracy score for job {job_id}: {report['accuracy_percent']}%")
    except Exception as e:
        logger.error(f"Failed to save accuracy score: {e}")


def get_accuracy_score(job_id: str) -> Optional[Dict]:
    """Load accuracy score for a job from disk."""
    if not ACCURACY_FILE.exists():
        return None
    try:
        with open(ACCURACY_FILE, 'r') as f:
            scores = json.load(f)
        return scores.get(job_id)
    except (json.JSONDecodeError, IOError):
        return None


def get_all_accuracy_scores() -> Dict:
    """Load all accuracy scores."""
    if not ACCURACY_FILE.exists():
        return {}
    try:
        with open(ACCURACY_FILE, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
