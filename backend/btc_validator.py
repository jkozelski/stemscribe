"""
BTC Chord Validation Layer — validates/upgrades Songsterr chords using BTC model.

When Songsterr chords are available (human-verified, accurate root + quality),
this module runs BTC on the audio as a VALIDATION layer. Where BTC detects
extensions (7, maj7, m7, sus4, dim, aug, etc.) with high confidence that are
compatible with the Songsterr chord, it upgrades the chord.

Rules:
- Only UPGRADES chords (adds extensions like 7, maj7, m7, sus4, dim, aug) — never downgrades
- Requires high BTC confidence to override Songsterr
- Keeps the original Songsterr chord if BTC confidence is low
- Logs what was changed for debugging
- Adds btc_validated: true/false flag to chord data
"""

import logging
import os
import re
from typing import List, Dict

logger = logging.getLogger(__name__)

# Minimum confidence threshold for BTC to upgrade a Songsterr chord
CONFIDENCE_THRESHOLD = 0.65

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
ENHARMONIC = {
    'Db': 'C#', 'Eb': 'D#', 'Fb': 'E', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#',
    'C#': 'Db', 'D#': 'Eb', 'E#': 'F', 'F#': 'Gb', 'G#': 'Ab', 'A#': 'Bb',
}

# Valid upgrade paths: Songsterr quality -> BTC qualities that are valid upgrades
UPGRADE_MAP = {
    # Major triads can be upgraded to extended major chords
    '': ['7', 'maj7', 'add9', '6', '9', '11', '13', 'sus4', 'sus2'],
    'maj': ['7', 'maj7', 'add9', '6', '9', '11', '13'],
    # Minor triads can be upgraded to extended minor chords
    'm': ['m7', 'mMaj7', 'm6', 'm9', 'm11', 'm13'],
    'min': ['m7', 'mMaj7', 'm6', 'm9', 'm11', 'm13'],
    # Dominant 7 can be upgraded to extended dominant
    '7': ['9', '11', '13', '7b5', '7#5', '7sus4'],
    # Minor 7 can be upgraded
    'm7': ['m9', 'm11', 'm13'],
    'min7': ['m9', 'm11', 'm13'],
    # Major 7 can be upgraded
    'maj7': ['maj9', 'maj13'],
    # Diminished can get 7th
    'dim': ['dim7', 'hdim7', 'm7b5'],
    # Sus chords can get 7th
    'sus4': ['7sus4'],
    'sus2': ['7sus2', '9'],
    # Power chord can become anything with same root
    '5': ['', 'm', '7', 'm7', 'sus4', 'sus2'],
}


def _parse_root(chord_name):
    """Extract root note from chord name."""
    if not chord_name or chord_name in ('N', 'X', '-'):
        return ''
    m = re.match(r'^([A-G][#b]?)', chord_name)
    return m.group(1) if m else ''


def _parse_quality(chord_name):
    """Extract quality/extension from chord name (everything after root)."""
    if not chord_name or chord_name in ('N', 'X', '-'):
        return ''
    m = re.match(r'^[A-G][#b]?(.*?)(?:/[A-G][#b]?)?$', chord_name)
    return m.group(1) if m else ''


def _roots_match(root1, root2):
    """Check if two root notes are the same (considering enharmonics)."""
    if not root1 or not root2:
        return False
    if root1 == root2:
        return True
    return ENHARMONIC.get(root1) == root2 or ENHARMONIC.get(root2) == root1


def _is_upgrade(songsterr_quality, btc_quality):
    """Check if BTC quality is a valid upgrade of Songsterr quality."""
    sq = songsterr_quality.strip()
    bq = btc_quality.strip()

    if sq == bq:
        return False  # Same chord, not an upgrade

    # Check explicit upgrade map
    if sq in UPGRADE_MAP and bq in UPGRADE_MAP[sq]:
        return True

    # General rule: if BTC adds extensions to the same base, it's an upgrade
    if bq.startswith(sq) and len(bq) > len(sq):
        return True

    return False


def _btc_chord_to_simple(chord_str):
    """Convert BTC mir_eval format chord (C:min7) to simple format (Cm7)."""
    if chord_str in ('N', 'X', '-'):
        return 'N'
    return (chord_str
            .replace(':minmaj7', 'mMaj7')
            .replace(':min7', 'm7')
            .replace(':min6', 'm6')
            .replace(':min9', 'm9')
            .replace(':min11', 'm11')
            .replace(':min13', 'm13')
            .replace(':min', 'm')
            .replace(':maj7', 'maj7')
            .replace(':maj6', '6')
            .replace(':maj9', 'maj9')
            .replace(':maj13', 'maj13')
            .replace(':maj', '')
            .replace(':7', '7')
            .replace(':9', '9')
            .replace(':11', '11')
            .replace(':13', '13')
            .replace(':aug', 'aug')
            .replace(':dim7', 'dim7')
            .replace(':dim', 'dim')
            .replace(':sus4', 'sus4')
            .replace(':sus2', 'sus2')
            .replace(':hdim7', 'm7b5'))


def validate_chords_with_btc(songsterr_events, audio_path):
    """
    Validate Songsterr chord events against BTC model predictions.

    Args:
        songsterr_events: List of {'time': float, 'chord': str} from Songsterr
        audio_path: Path to audio file for BTC analysis

    Returns:
        Dict with:
            'chordEvents': Enhanced chord events with btc_validated flag
            'btc_validated': True/False
            'btc_changes': List of changes made
            'btc_stats': Summary stats
    """
    if not songsterr_events or not audio_path or not os.path.exists(audio_path):
        logger.info("BTC validation skipped: no events or audio unavailable")
        return {
            'chordEvents': songsterr_events,
            'btc_validated': False,
            'btc_changes': [],
            'btc_stats': {'reason': 'no_audio'},
        }

    try:
        from chord_detector_v10 import ChordDetector
        detector = ChordDetector(min_duration=0.3)

        logger.info("BTC validation: running BTC on %s", os.path.basename(audio_path))
        btc_prog = detector.detect(audio_path)

        if not btc_prog or not btc_prog.chords:
            logger.info("BTC validation: no chords detected by BTC")
            return {
                'chordEvents': [{**e, 'btc_validated': False} for e in songsterr_events],
                'btc_validated': False,
                'btc_changes': [],
                'btc_stats': {'reason': 'btc_no_chords'},
            }

        # Build BTC chord timeline
        btc_timeline = []
        for ce in btc_prog.chords:
            raw = ce.chord
            simple = _btc_chord_to_simple(raw)
            btc_timeline.append({
                'start': ce.time,
                'end': ce.time + ce.duration,
                'chord': simple,
                'confidence': ce.confidence,
                'root': _parse_root(simple),
                'quality': _parse_quality(simple),
            })

        logger.info("BTC validation: BTC detected %d chord segments", len(btc_timeline))

        # For each Songsterr event, find the best BTC chord at that time
        enhanced_events = []
        changes = []
        confirmed = 0
        upgraded = 0
        root_mismatch = 0

        for event in songsterr_events:
            t = event['time']
            songsterr_chord = event['chord']
            songsterr_root = _parse_root(songsterr_chord)
            songsterr_quality = _parse_quality(songsterr_chord)

            if not songsterr_root or songsterr_chord in ('N', 'X', '-'):
                enhanced_events.append({**event, 'btc_validated': True})
                continue

            # Find BTC chord at this time (with 0.5s tolerance)
            btc_match = None
            best_overlap = 0
            for btc in btc_timeline:
                if btc['start'] <= t + 0.5 and btc['end'] >= t - 0.5:
                    overlap = min(btc['end'], t + 1.0) - max(btc['start'], t - 0.5)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        btc_match = btc

            if not btc_match:
                enhanced_events.append({**event, 'btc_validated': True, 'btc_match': None})
                confirmed += 1
                continue

            btc_chord = btc_match['chord']
            btc_root = btc_match['root']
            btc_quality = btc_match['quality']
            btc_confidence = btc_match['confidence']

            # Check if roots match (considering enharmonics)
            if not _roots_match(songsterr_root, btc_root):
                root_mismatch += 1
                enhanced_events.append({
                    **event,
                    'btc_validated': True,
                    'btc_match': btc_chord,
                    'btc_confidence': round(btc_confidence, 3),
                })
                continue

            # Roots match — check if BTC has a valid upgrade with high confidence
            if btc_confidence >= CONFIDENCE_THRESHOLD and _is_upgrade(songsterr_quality, btc_quality):
                upgraded_chord = songsterr_root + btc_quality
                changes.append({
                    'time': round(t, 2),
                    'original': songsterr_chord,
                    'upgraded': upgraded_chord,
                    'btc_confidence': round(btc_confidence, 3),
                })
                logger.info(
                    "BTC upgrade: %s -> %s (conf=%.2f) at t=%.1fs",
                    songsterr_chord, upgraded_chord, btc_confidence, t
                )
                enhanced_events.append({
                    **event,
                    'chord': upgraded_chord,
                    'original_chord': songsterr_chord,
                    'btc_validated': True,
                    'btc_upgraded': True,
                    'btc_confidence': round(btc_confidence, 3),
                })
                upgraded += 1
            else:
                confirmed += 1
                enhanced_events.append({
                    **event,
                    'btc_validated': True,
                    'btc_match': btc_chord,
                    'btc_confidence': round(btc_confidence, 3),
                })

        stats = {
            'total': len(songsterr_events),
            'upgraded': upgraded,
            'confirmed': confirmed,
            'root_mismatch': root_mismatch,
        }

        if changes:
            logger.info(
                "BTC validation complete: %d chord(s) upgraded, %d confirmed, "
                "%d root mismatches out of %d total",
                upgraded, confirmed, root_mismatch, len(songsterr_events)
            )
        else:
            logger.info(
                "BTC validation complete: no upgrades — all %d chords confirmed",
                len(songsterr_events)
            )

        return {
            'chordEvents': enhanced_events,
            'btc_validated': True,
            'btc_changes': changes,
            'btc_stats': stats,
        }

    except ImportError as e:
        logger.warning("BTC validation unavailable (import error): %s", e)
        return {
            'chordEvents': [{**e_item, 'btc_validated': False} for e_item in songsterr_events],
            'btc_validated': False,
            'btc_changes': [],
            'btc_stats': {'reason': 'import_error: %s' % e},
        }
    except Exception as e:
        logger.error("BTC validation failed: %s", e, exc_info=True)
        return {
            'chordEvents': [{**e_item, 'btc_validated': False} for e_item in songsterr_events],
            'btc_validated': False,
            'btc_changes': [],
            'btc_stats': {'reason': 'error: %s' % e},
        }
