"""
Chord Detection v10 — BTC (Bi-directional Transformer for Chord Recognition).

Uses the pre-trained BTC-ISMIR19 model for state-of-the-art chord detection.
Falls back to Essentia ChordsDetection if BTC unavailable.

BTC paper: https://arxiv.org/abs/1907.02698
"""

import sys
import os
import json
import numpy as np
import logging
from typing import List
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)

# BTC model directory
BTC_DIR = Path(__file__).parent.parent / "btc_chord"


@dataclass
class ChordEvent:
    time: float
    duration: float
    chord: str
    root: str
    quality: str
    confidence: float


@dataclass
class ChordProgression:
    chords: List[ChordEvent]
    key: str
    tuning_info: dict = None  # Populated by tuning detection post-processing


NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def _parse_chord(name: str) -> tuple:
    """Parse chord name into (root, quality)."""
    if not name or name == 'N':
        return ('N', 'none')
    root = name[0]
    rest = name[1:]
    if rest and rest[0] in ('#', 'b'):
        root += rest[0]
        rest = rest[1:]
    return (root, rest if rest else 'maj')


def _mir_to_simple(chord: str) -> str:
    """Convert mir_eval chord format (C:min7) to simple format (Cm7)."""
    if chord in ('N', 'X'):
        return 'N'
    return (chord
            .replace(':minmaj7', 'mMaj7')
            .replace(':min7', 'm7')
            .replace(':min6', 'm6')
            .replace(':min', 'm')
            .replace(':maj7', 'maj7')
            .replace(':maj6', '6')
            .replace(':maj', '')
            .replace(':7', '7')
            .replace(':aug', 'aug')
            .replace(':dim7', 'dim7')
            .replace(':dim', 'dim')
            .replace(':sus4', 'sus4')
            .replace(':sus2', 'sus2')
            .replace(':hdim7', 'hdim7'))


def detect_chords(audio_path: str, min_duration: float = 0.3, **kwargs) -> list:
    """Convenience function for backward compatibility."""
    detector = ChordDetector(min_duration=min_duration)
    prog = detector.detect(audio_path)
    return [{'time': c.time, 'duration': c.duration, 'chord': c.chord,
             'root': c.root, 'quality': c.quality, 'confidence': c.confidence}
            for c in prog.chords]


ENHARMONIC = {
    'Bb': 'A#', 'A#': 'Bb', 'Db': 'C#', 'C#': 'Db',
    'Eb': 'D#', 'D#': 'Eb', 'Gb': 'F#', 'F#': 'Gb',
    'Ab': 'G#', 'G#': 'Ab',
}

CHORD_DB_PATH = Path(__file__).parent / "training_data" / "chord_db" / "chord_database.json"


def _expand_chord(ch: str) -> set:
    """Generate all equivalent forms of a chord for vocabulary matching."""
    base = ch.split('/')[0] if '/' in ch else ch
    forms = {base}
    for old, new in ENHARMONIC.items():
        if base.startswith(old):
            forms.add(new + base[len(old):])
    if 'add9' in base:
        forms.add(base.replace('add9', ''))
    if 'sus4' in base:
        forms.add(base.replace('sus4', ''))
    if 'sus2' in base:
        forms.add(base.replace('sus2', ''))
    if base.endswith('5') and len(base) >= 2:
        root = base[:-1]
        forms.update({root, root + 'm'})
        for old, new in ENHARMONIC.items():
            if root.startswith(old):
                forms.update({new + root[len(old):], new + root[len(old):] + 'm'})
    if base.endswith('4') and 'sus' not in base and len(base) == 2:
        forms.add(base[0] + 'sus4')
    return forms


def _load_chord_db() -> dict:
    """Load the chord vocabulary database."""
    if CHORD_DB_PATH.exists():
        try:
            return json.loads(CHORD_DB_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_chord_db(db: dict):
    """Save chord vocabulary database."""
    try:
        CHORD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHORD_DB_PATH.write_text(json.dumps(db, indent=2))
    except Exception as e:
        logger.warning(f"Failed to save chord DB: {e}")


def _scrape_ug_chords(artist: str, title: str) -> list | None:
    """Live-scrape chord vocabulary from Ultimate Guitar for a single song."""
    try:
        import html as html_mod
        import re
        import urllib.parse
        from curl_cffi import requests as cffi_requests

        query = f"{artist} {title}"
        encoded = urllib.parse.quote(query)
        search_url = (
            f"https://www.ultimate-guitar.com/search.php?"
            f"search_type=title&value={encoded}&type%5B%5D=300"
        )

        resp = cffi_requests.get(search_url, impersonate="chrome131", timeout=10)
        resp.raise_for_status()

        m = re.search(r'class="js-store"\s+data-content="(.*?)"', resp.text)
        if not m:
            return None

        store = json.loads(html_mod.unescape(m.group(1)))
        results = (store.get("store", {})
                  .get("page", {})
                  .get("data", {})
                  .get("results", []))

        # Find best-rated chord tab
        candidates = []
        for item in results:
            items_to_check = [item] if isinstance(item, dict) else (item if isinstance(item, list) else [])
            for sub in items_to_check:
                if isinstance(sub, dict) and sub.get("type") == "Chords":
                    candidates.append(sub)

        if not candidates:
            return None

        candidates.sort(key=lambda x: x.get("rating", 0), reverse=True)
        tab_url = candidates[0].get("tab_url", "")
        if not tab_url:
            return None

        # Fetch tab page
        resp2 = cffi_requests.get(tab_url, impersonate="chrome131", timeout=10)
        resp2.raise_for_status()

        m2 = re.search(r'class="js-store"\s+data-content="(.*?)"', resp2.text)
        if not m2:
            return None

        tab_store = json.loads(html_mod.unescape(m2.group(1)))
        content = (tab_store.get("store", {})
                  .get("page", {})
                  .get("data", {})
                  .get("tab_view", {})
                  .get("wiki_tab", {})
                  .get("content", ""))

        if not content:
            return None

        # Extract unique chord names
        seen = set()
        chords = []
        for cm in re.finditer(r"\[ch\](.*?)\[/ch\]", content):
            ch = cm.group(1).strip()
            if ch and ch not in seen:
                seen.add(ch)
                chords.append(ch)

        return chords if chords else None

    except Exception as e:
        logger.debug(f"UG scrape failed for {artist} - {title}: {e}")
        return None


def _lookup_vocab(artist: str, title: str, chord_db: dict) -> list | None:
    """Find chord vocabulary for a song. Checks cache first, then scrapes UG live."""
    if not artist or not title:
        return None

    # Check cache
    if chord_db:
        key = f"{artist} - {title}"
        if key in chord_db:
            return chord_db[key].get("chords", [])
        key_lower = key.lower()
        for k, v in chord_db.items():
            if k.lower() == key_lower:
                return v.get("chords", [])
        title_lower = title.lower()
        for k, v in chord_db.items():
            if v.get("song", "").lower() == title_lower:
                return v.get("chords", [])

    # Not cached — scrape UG live
    logger.info(f"Chord DB miss, scraping UG for: {artist} - {title}")
    chords = _scrape_ug_chords(artist, title)

    if chords:
        # Cache for next time
        key = f"{artist} - {title}"
        if chord_db is None:
            chord_db = {}
        chord_db[key] = {
            "artist": artist,
            "song": title,
            "chords": chords,
            "unique_count": len(chords),
        }
        _save_chord_db(chord_db)
        logger.info(f"Cached {len(chords)} chords for {artist} - {title}")
        return chords

    logger.info(f"No UG chords found for {artist} - {title}")
    return None


def _get_diatonic_chords(key: str) -> set:
    """Return the set of diatonic pitch classes for a given key.

    Args:
        key: Key string like 'C', 'Am', 'F#m', 'Bb'

    Returns:
        Set of pitch class indices (0-11) that are diatonic to the key
    """
    if not key or key == "Unknown":
        return set()

    key_root = key[0]
    key_rest = key[1:]
    if key_rest and key_rest[0] in ('#', 'b'):
        key_root = key[:2]
        key_rest = key[2:]
    is_minor = 'm' in key_rest

    FLAT_TO_SHARP = {'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#'}
    key_root_norm = FLAT_TO_SHARP.get(key_root, key_root)

    if key_root_norm not in NOTE_NAMES:
        return set()

    key_idx = NOTE_NAMES.index(key_root_norm)

    if is_minor:
        intervals = [0, 2, 3, 5, 7, 8, 10]
    else:
        intervals = [0, 2, 4, 5, 7, 9, 11]

    return set((key_idx + i) % 12 for i in intervals)


def _root_to_pitch_class(root: str) -> int:
    """Convert a root note name to pitch class (0-11). Returns -1 if unknown."""
    FLAT_TO_SHARP = {'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#'}
    root_norm = FLAT_TO_SHARP.get(root, root)
    if root_norm in NOTE_NAMES:
        return NOTE_NAMES.index(root_norm)
    return -1


def postprocess_chords(events: List[ChordEvent], key: str,
                       audio_path: str = None) -> List[ChordEvent]:
    """Apply post-processing filters to clean up noisy chord detection output.

    Steps applied in order:
      1. Merge consecutive identical chords
      2. Minimum duration filter (tempo-aware, adaptive)
      3. Rare chord filter (adaptive based on unique chord count)
      4. Key-aware confidence weighting (boost diatonic, penalize non-diatonic)
      5. Re-merge after filtering (in case removals created new adjacencies)

    Filters are adaptive: when the raw detection is already sparse (few unique
    chord types), thresholds are relaxed to avoid over-filtering.

    Args:
        events: List of ChordEvent from detection pipeline
        key: Detected key string (e.g. 'Am', 'C', 'F#m')
        audio_path: Path to audio file (for tempo detection). If None, uses 1.0s fallback.

    Returns:
        Cleaned list of ChordEvent
    """
    if not events:
        return events

    logger.info(f"Post-processing {len(events)} chord events (key={key})")

    # Count unique chord types in raw input for adaptive thresholds
    raw_unique = len(set(e.chord for e in events))
    is_sparse = raw_unique <= 8
    logger.info(f"  Raw unique chord types: {raw_unique} (sparse={is_sparse})")

    # --- Step 1: Merge consecutive identical chords ---
    merged = _merge_consecutive(events)
    logger.info(f"  Step 1 (merge consecutive): {len(events)} -> {len(merged)} events")

    # --- Step 2: Minimum duration filter (tempo-aware, adaptive) ---
    min_beat_dur = _get_min_beat_duration(audio_path)
    before_count = len(merged)
    merged = _filter_min_duration_adaptive(merged, min_beat_dur, sparse=is_sparse)
    logger.info(f"  Step 2 (min duration {min_beat_dur:.2f}s, adaptive): {before_count} -> {len(merged)} events")

    # --- Step 3: Rare chord filter (adaptive) ---
    before_count = len(merged)
    merged = _filter_rare_chords_adaptive(merged, raw_unique)
    logger.info(f"  Step 3 (rare chord filter, adaptive): {before_count} -> {len(merged)} events")

    # --- Step 4: Key-aware confidence weighting (with chromatic passing protection) ---
    merged = _apply_key_weighting(merged, key)
    logger.info(f"  Step 4 (key weighting): applied diatonic boost/penalty for key={key}")

    # --- Step 5: Re-merge after filtering (removals may create new adjacencies) ---
    before_count = len(merged)
    merged = _merge_consecutive(merged)
    if len(merged) != before_count:
        logger.info(f"  Step 5 (re-merge): {before_count} -> {len(merged)} events")

    logger.info(f"Post-processing complete: {len(events)} -> {len(merged)} chord events")
    return merged


def _merge_consecutive(events: List[ChordEvent]) -> List[ChordEvent]:
    """Merge consecutive identical chord events into single events with combined duration."""
    if not events:
        return events

    merged = []
    current = events[0]
    conf_sum = current.confidence * current.duration
    total_dur = current.duration

    for ev in events[1:]:
        if ev.chord == current.chord:
            # Same chord — extend duration, accumulate weighted confidence
            total_dur += ev.duration
            conf_sum += ev.confidence * ev.duration
        else:
            # Different chord — emit current and start new
            avg_conf = conf_sum / total_dur if total_dur > 0 else current.confidence
            merged.append(ChordEvent(
                time=current.time,
                duration=total_dur,
                chord=current.chord,
                root=current.root,
                quality=current.quality,
                confidence=avg_conf,
            ))
            current = ev
            conf_sum = ev.confidence * ev.duration
            total_dur = ev.duration

    # Emit last
    avg_conf = conf_sum / total_dur if total_dur > 0 else current.confidence
    merged.append(ChordEvent(
        time=current.time,
        duration=total_dur,
        chord=current.chord,
        root=current.root,
        quality=current.quality,
        confidence=avg_conf,
    ))
    return merged


def _get_min_beat_duration(audio_path: str = None) -> float:
    """Detect tempo and return the duration of 1 beat in seconds.

    Falls back to 1.0 second if tempo detection fails or no audio_path given.
    """
    if not audio_path:
        return 1.0

    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # librosa may return an array; take first element
        if hasattr(tempo, '__len__'):
            tempo = float(tempo[0]) if len(tempo) > 0 else 0.0
        else:
            tempo = float(tempo)
        if tempo > 0:
            beat_dur = 60.0 / tempo
            logger.info(f"  Tempo detected: {tempo:.1f} BPM -> 1 beat = {beat_dur:.2f}s")
            return beat_dur
    except Exception as e:
        logger.warning(f"  Tempo detection failed, using 1.0s fallback: {e}")

    return 1.0


def _filter_min_duration(events: List[ChordEvent], min_dur: float) -> List[ChordEvent]:
    """Remove chord events shorter than min_dur (likely noise)."""
    return [e for e in events if e.duration >= min_dur]


def _filter_min_duration_adaptive(events: List[ChordEvent], min_dur: float,
                                   sparse: bool = False) -> List[ChordEvent]:
    """Remove short chord events with adaptive gap and count protection.

    Adaptive rules:
      - If removing a chord would leave a gap > 2 beats, keep it even if short
      - If total events after filtering would drop below 5, relax threshold to 0.5 beats
      - When sparse (<=8 unique chord types), use relaxed thresholds from the start
    """
    if not events:
        return events

    effective_min = min_dur * 0.5 if sparse else min_dur
    max_gap_beats = 2.0  # max allowed gap in multiples of min_dur

    # First pass: mark which events to keep
    keep = []
    for i, e in enumerate(events):
        if e.duration >= effective_min:
            keep.append(True)
        else:
            # Check if removing would create a gap > 2 beats
            prev_end = (events[i - 1].time + events[i - 1].duration) if i > 0 else e.time
            next_start = events[i + 1].time if i + 1 < len(events) else (e.time + e.duration)
            gap = next_start - prev_end
            if gap > max_gap_beats * min_dur:
                keep.append(True)  # Keep to avoid large gap
            else:
                keep.append(False)

    filtered = [e for e, k in zip(events, keep) if k]

    # Safety net: if we dropped below 5 events, relax to 0.5x threshold and retry
    if len(filtered) < 5 and len(events) >= 5:
        relaxed_min = min_dur * 0.5
        logger.info(f"    Min-duration safety: {len(filtered)} events too few, relaxing to {relaxed_min:.2f}s")
        keep2 = []
        for i, e in enumerate(events):
            if e.duration >= relaxed_min:
                keep2.append(True)
            else:
                prev_end = (events[i - 1].time + events[i - 1].duration) if i > 0 else e.time
                next_start = events[i + 1].time if i + 1 < len(events) else (e.time + e.duration)
                gap = next_start - prev_end
                if gap > max_gap_beats * min_dur:
                    keep2.append(True)
                else:
                    keep2.append(False)
        filtered = [e for e, k in zip(events, keep2) if k]

    return filtered


def _filter_rare_chords(events: List[ChordEvent], min_occurrences: int = 2) -> List[ChordEvent]:
    """Remove chords that appear only once total (likely false positives).

    Exception: chords at the very start or end of the song are kept regardless.
    """
    if not events:
        return events

    # Count occurrences of each chord name
    chord_counts = Counter(e.chord for e in events)

    # Identify boundary chords (first and last)
    first_chord = events[0].chord
    last_chord = events[-1].chord

    filtered = []
    for e in events:
        count = chord_counts[e.chord]
        is_boundary = (e.chord == first_chord or e.chord == last_chord)
        if count >= min_occurrences or is_boundary:
            filtered.append(e)

    return filtered


def _filter_rare_chords_adaptive(events: List[ChordEvent],
                                  raw_unique_count: int) -> List[ChordEvent]:
    """Remove rare chords with adaptive threshold based on unique chord count.

    Threshold rules:
      - raw_unique <= 8 (global sparse check): skip filter entirely
      - raw_unique 7-12: remove chords appearing only once
      - raw_unique > 12: remove chords appearing 1-2 times (original behavior)

    Exception: boundary chords (first/last) are always kept.
    """
    if not events:
        return events

    # Global sparse check: skip entirely if <= 8 unique chord types in raw output
    if raw_unique_count <= 8:
        logger.info(f"    Rare filter skipped: only {raw_unique_count} unique chord types (sparse)")
        return events

    # Determine adaptive threshold
    if raw_unique_count <= 12:
        min_occurrences = 2  # remove chords appearing only once
    else:
        min_occurrences = 3  # remove chords appearing 1-2 times (original behavior)

    logger.info(f"    Rare filter threshold: min_occurrences={min_occurrences} "
                f"(raw_unique={raw_unique_count})")

    chord_counts = Counter(e.chord for e in events)
    first_chord = events[0].chord
    last_chord = events[-1].chord

    filtered = []
    for e in events:
        count = chord_counts[e.chord]
        is_boundary = (e.chord == first_chord or e.chord == last_chord)
        if count >= min_occurrences or is_boundary:
            filtered.append(e)

    return filtered


def _is_chromatic_passing(events: List[ChordEvent], idx: int,
                          diatonic_set: set) -> bool:
    """Check if a non-diatonic chord at index idx is a chromatic passing chord.

    A chromatic passing chord sits between two diatonic chords whose roots are
    a whole step (2 semitones) apart, and the passing chord's root is the
    semitone between them.

    Example: Bm -> Bbm -> Am  (Bbm is chromatic passing between B and A)
    """
    if idx <= 0 or idx >= len(events) - 1:
        return False

    prev_pc = _root_to_pitch_class(events[idx - 1].root)
    curr_pc = _root_to_pitch_class(events[idx].root)
    next_pc = _root_to_pitch_class(events[idx + 1].root)

    if prev_pc < 0 or curr_pc < 0 or next_pc < 0:
        return False

    # Both neighbors must be diatonic
    if prev_pc not in diatonic_set or next_pc not in diatonic_set:
        return False

    # Neighbors must be a whole step apart (2 semitones)
    interval = abs(prev_pc - next_pc)
    if interval > 6:
        interval = 12 - interval  # wrap around
    if interval != 2:
        return False

    # Current chord must be the semitone between them
    lower = min(prev_pc, next_pc)
    upper = max(prev_pc, next_pc)
    # Normal case
    if upper - lower == 2 and curr_pc == lower + 1:
        return True
    # Wrap-around case (e.g., prev=11(B), next=1(C#), curr=0(C))
    if (upper - lower) > 6:
        mid = (upper + 1) % 12
        if curr_pc == mid:
            return True

    return False


def _apply_key_weighting(events: List[ChordEvent], key: str) -> List[ChordEvent]:
    """Boost confidence of diatonic chords and penalize non-diatonic chords.

    - Diatonic chords: +12% confidence (capped at 0.99)
    - Non-diatonic chords: -10% confidence max (reduced from 18%), floored at 0.05
    - Chromatic passing chords: no penalty at all
    """
    diatonic_set = _get_diatonic_chords(key)
    if not diatonic_set:
        return events  # Can't determine key, skip weighting

    DIATONIC_BOOST = 0.12
    NON_DIATONIC_PENALTY = 0.10  # Reduced from 0.18

    weighted = []
    for i, e in enumerate(events):
        pc = _root_to_pitch_class(e.root)
        if pc < 0:
            weighted.append(e)
            continue

        if pc in diatonic_set:
            new_conf = min(0.99, e.confidence + DIATONIC_BOOST)
        elif _is_chromatic_passing(events, i, diatonic_set):
            # Chromatic passing chord -- no penalty
            logger.debug(f"    Chromatic passing chord preserved: {e.chord} at {e.time:.2f}s")
            new_conf = e.confidence
        else:
            new_conf = max(0.05, e.confidence - NON_DIATONIC_PENALTY)

        weighted.append(ChordEvent(
            time=e.time,
            duration=e.duration,
            chord=e.chord,
            root=e.root,
            quality=e.quality,
            confidence=new_conf,
        ))

    return weighted


class ChordDetector:
    """Chord detector using BTC pre-trained model with optional vocabulary constraint."""

    def __init__(self, min_duration: float = 0.3, **kwargs):
        self.min_duration = min_duration
        self._btc_loaded = False
        self._model = None
        self._config = None
        self._mean = None
        self._std = None
        self._idx_to_chord = None
        self._device = None
        self._chord_db = None

    def _load_btc(self):
        """Lazy-load BTC model."""
        if self._btc_loaded:
            return True

        try:
            import torch
            btc_path = str(BTC_DIR)
            if btc_path not in sys.path:
                sys.path.insert(0, btc_path)

            from btc_model import BTC_model
            from utils.hparams import HParams
            from utils.mir_eval_modules import idx2voca_chord

            self._device = torch.device("cpu")
            config = HParams.load(str(BTC_DIR / "run_config.yaml"))
            config.feature['large_voca'] = True
            config.model['num_chords'] = 170

            model = BTC_model(config=config.model).to(self._device)

            finetuned = Path(__file__).parent / "training_data" / "btc_finetune" / "checkpoints" / "btc_finetuned_best.pt"
            original = BTC_DIR / "test" / "btc_model_large_voca.pt"
            model_file = str(finetuned if finetuned.exists() else original)
            logger.info(f"Loading BTC checkpoint: {'fine-tuned' if finetuned.exists() else 'original'}")

            # BTC checkpoint contains numpy arrays (mean/std) — trusted local model
            checkpoint = torch.load(model_file, map_location=self._device, weights_only=False)
            self._mean = checkpoint['mean']
            self._std = checkpoint['std']
            model.load_state_dict(checkpoint['model'])
            model.eval()

            self._model = model
            self._config = config
            self._idx_to_chord = idx2voca_chord()
            self._btc_loaded = True

            # Load chord vocabulary database
            self._chord_db = _load_chord_db()
            db_size = len(self._chord_db) if self._chord_db else 0
            logger.info(f"BTC chord model loaded (chord DB: {db_size} songs)")
            return True

        except Exception as e:
            logger.warning(f"Failed to load BTC model: {e}")
            return False

    def _build_vocab_mask(self, vocab: list):
        """Build a logit mask that constrains output to the given chord vocabulary."""
        import torch
        expanded = set(['N'])
        for ch in vocab:
            expanded.update(_expand_chord(ch))

        allowed = [idx for idx, mn in self._idx_to_chord.items()
                   if _mir_to_simple(mn) in expanded]

        if not allowed:
            return None

        mask = torch.full((170,), float('-inf'), device=self._device)
        mask[torch.tensor(allowed, device=self._device)] = 0
        return mask

    def _validate_vocab(self, unconstrained_chords: List[ChordEvent],
                         vocab: list, artist: str, title: str) -> bool:
        """Validate scraped vocab against unconstrained BTC results.

        Calculates what fraction of BTC's duration-weighted chord roots appear
        in the scraped vocabulary. If overlap is below threshold, the vocab is
        likely from the wrong song.

        Args:
            unconstrained_chords: Chord events from unconstrained BTC run
            vocab: Scraped chord vocabulary to validate
            artist: Artist name (for logging)
            title: Song title (for logging)

        Returns:
            True if vocab is valid (should be used), False if it should be discarded
        """
        OVERLAP_THRESHOLD = 0.30

        if not unconstrained_chords:
            logger.info("Vocab validation: no unconstrained chords, accepting vocab as fallback")
            return True

        # Extract duration-weighted root notes from unconstrained BTC
        root_durations = Counter()
        for ev in unconstrained_chords:
            root = ev.root
            if root and root != 'N':
                root_durations[root] += ev.duration

        # Extract root notes from scraped vocab (with enharmonic equivalents)
        vocab_roots = set()
        for ch in vocab:
            root, _ = _parse_chord(ch)
            if root and root != 'N':
                vocab_roots.add(root)
                if root in ENHARMONIC:
                    vocab_roots.add(ENHARMONIC[root])

        # Calculate overlap: fraction of BTC duration whose roots are in vocab
        total_duration = sum(root_durations.values())
        if total_duration == 0:
            return True

        matched_duration = 0.0
        for root, dur in root_durations.items():
            if root in vocab_roots:
                matched_duration += dur
            elif root in ENHARMONIC and ENHARMONIC[root] in vocab_roots:
                matched_duration += dur

        overlap = matched_duration / total_duration

        logger.info(f"Vocab validation: overlap={overlap:.1%} "
                     f"(BTC roots: {dict(root_durations.most_common(6))}, "
                     f"vocab roots: {sorted(vocab_roots)})")

        if overlap < OVERLAP_THRESHOLD:
            logger.warning(
                f"VOCAB MISMATCH DETECTED for '{artist} - {title}': "
                f"overlap={overlap:.1%} < {OVERLAP_THRESHOLD:.0%} threshold. "
                f"Scraped vocab {vocab} does not match audio. "
                f"Discarding scraped vocab — using unconstrained BTC results."
            )
            return False

        logger.info(f"Vocab validated (overlap={overlap:.1%} >= {OVERLAP_THRESHOLD:.0%}), using constrained BTC")
        return True

    def detect(self, audio_path: str, artist: str = None, title: str = None) -> ChordProgression:
        """Detect chords from audio file. Pass artist/title for vocabulary-constrained mode.

        When BTC is available, also runs Essentia as an ensemble member to improve
        root note accuracy. Low-confidence BTC chords are reconciled with Essentia.

        Vocab validation: runs BTC unconstrained first, then checks if scraped vocab
        matches what BTC actually hears. If overlap < 30%, scraped vocab is discarded
        (likely wrong song on UG). Songsterr chords bypass this check (more reliable).

        Post-processing is applied to all results: merge consecutive, min-duration filter,
        rare chord filter, and key-aware confidence weighting.
        """
        if self._load_btc():
            # Look up vocabulary constraint
            vocab = None
            vocab_source = None
            if artist and title and self._chord_db:
                vocab = _lookup_vocab(artist, title, self._chord_db)
                if vocab:
                    # Determine vocab source from cache metadata
                    key = f"{artist} - {title}"
                    entry = self._chord_db.get(key, {})
                    if not entry:
                        # Try case-insensitive lookup
                        key_lower = key.lower()
                        for k, v in self._chord_db.items():
                            if k.lower() == key_lower:
                                entry = v
                                break
                    vocab_source = entry.get("source", "ug")  # default to UG
                    logger.info(f"Vocab constraint: {len(vocab)} chords for {artist} - {title} (source={vocab_source})")

            # Always run unconstrained BTC first (needed for vocab validation AND tuning detection)
            unconstrained_result = self._detect_btc(audio_path, vocab=None)

            # --- Vocab validation BEFORE tuning detection ---
            # Must validate vocab first, because tuning detector uses vocab as reference.
            # If vocab is from wrong song, tuning detector will compute a bogus offset.
            vocab_discarded = False
            if vocab and vocab_source != "songsterr":
                if not self._validate_vocab(unconstrained_result.chords, vocab, artist, title):
                    vocab = None
                    vocab_discarded = True

            # --- Tuning detection from unconstrained results ---
            # Must run on unconstrained output (consistent pitch-shifted chord names).
            # Running on vocab-constrained output gives a confusing mix of shifted/unshifted.
            tuning_info = self._apply_tuning_correction(
                unconstrained_result, audio_path, artist, title, vocab,
                vocab_discarded=vocab_discarded)
            tuning_offset = tuning_info.get('tuning_offset_semitones', 0)

            if tuning_offset != 0 and vocab:
                # Tuning correction applied — unconstrained chords are now transposed.
                # Use corrected unconstrained result instead of constrained mode
                # (which would fight the tuning shift).
                logger.info(f"Tuning correction applied (offset={tuning_offset:+d}), "
                            f"skipping vocab-constrained mode")
                btc_result = unconstrained_result
            elif vocab and vocab_source == "songsterr":
                # Songsterr vocab — trusted, use directly without validation
                logger.info("Using Songsterr vocab directly (trusted source)")
                btc_result = self._detect_btc(audio_path, vocab=vocab)
            elif vocab:
                # Vocab was validated above and passed — run constrained
                btc_result = self._detect_btc(audio_path, vocab=vocab)
            else:
                btc_result = unconstrained_result

            # Run Essentia ensemble if available (for root accuracy improvement)
            result = btc_result
            try:
                from essentia_chord_detector import (
                    detect_chords_essentia, ensemble_chords, ESSENTIA_AVAILABLE
                )
                if ESSENTIA_AVAILABLE and btc_result.chords:
                    ess_result = detect_chords_essentia(audio_path, min_duration=self.min_duration)
                    if ess_result.chords:
                        merged = ensemble_chords(btc_result.chords, ess_result.chords)
                        logger.info(f"Ensemble: BTC({len(btc_result.chords)}) + Essentia({len(ess_result.chords)}) -> {len(merged)} chords")
                        result = ChordProgression(chords=merged, key=btc_result.key)
            except ImportError:
                pass  # Essentia ensemble not available, use BTC alone
            except Exception as e:
                logger.warning(f"Essentia ensemble failed, using BTC alone: {e}")

            # Apply post-processing
            result.chords = postprocess_chords(result.chords, result.key, audio_path)
            result.tuning_info = tuning_info
            return result
        else:
            result = self._detect_essentia(audio_path)
            # Apply post-processing to Essentia fallback too
            result.chords = postprocess_chords(result.chords, result.key, audio_path)

            # --- Tuning detection & correction (final step) ---
            result.tuning_info = self._apply_tuning_correction(
                result, audio_path, artist, title, vocab=None)
            return result

    def _apply_tuning_correction(self, result: ChordProgression, audio_path: str,
                                    artist: str = None, title: str = None,
                                    vocab: list = None, vocab_discarded: bool = False) -> dict:
        """Apply tuning detection and correction as a final post-processing step.

        Uses tuning_detector.detect_and_correct_tuning() to identify non-standard
        tunings (e.g., Eb tuning on Little Wing) and transpose chord names to what
        guitarists expect (e.g., Ebm -> Em).

        Returns tuning_info dict (always), updates result.chords in-place if correction needed.
        """
        default_info = {
            'tuning_offset_semitones': 0,
            'tuning_name': 'Standard (E)',
            'detection_method': 'none',
            'effective_a4': 440.0,
        }

        if not result.chords:
            return default_info

        try:
            from tuning_detector import detect_and_correct_tuning

            # Use the vocab already looked up for constraint, or look it up now.
            # If vocab was explicitly discarded by validation (vocab_discarded=True),
            # do NOT re-lookup — the scraped vocab is known to be wrong.
            reference_vocab = vocab
            if not reference_vocab and artist and title and not vocab_discarded:
                chord_db = self._chord_db or _load_chord_db()
                reference_vocab = _lookup_vocab(artist, title, chord_db)

            tuning_info = detect_and_correct_tuning(
                chord_events=result.chords,
                reference_vocab=reference_vocab,
                audio_path=audio_path,
            )

            if tuning_info['tuning_offset_semitones'] != 0:
                result.chords = tuning_info['chords']
                logger.info(f"Tuning correction applied in detect(): "
                            f"{tuning_info['tuning_name']} "
                            f"(offset={tuning_info['tuning_offset_semitones']:+d}, "
                            f"method={tuning_info['detection_method']})")
            else:
                logger.info(f"No tuning correction needed (method={tuning_info['detection_method']})")

            return {
                'tuning_offset_semitones': tuning_info['tuning_offset_semitones'],
                'tuning_name': tuning_info['tuning_name'],
                'detection_method': tuning_info['detection_method'],
                'effective_a4': tuning_info['effective_a4'],
            }

        except ImportError:
            logger.debug("tuning_detector not available — skipping tuning correction")
            return default_info
        except Exception as e:
            logger.warning(f"Tuning detection failed in detect(): {e}")
            return default_info

    def _detect_btc(self, audio_path: str, vocab: list = None) -> ChordProgression:
        """Run BTC chord detection with LSTM path and optional vocabulary constraint."""
        import torch
        import librosa

        try:
            config = self._config
            vocab_mask = self._build_vocab_mask(vocab) if vocab else None
            use_constraint = vocab_mask is not None

            original_wav, sr = librosa.load(
                str(audio_path), sr=config.mp3['song_hz'], mono=True)
            duration = len(original_wav) / sr
            logger.info(f"Loaded {duration:.1f}s audio (constrained={use_constraint})")

            # --- FIX 1: Tuning compensation before CQT ---
            tuning_offset = librosa.estimate_tuning(y=original_wav, sr=sr)
            logger.info(f"Estimated tuning offset: {tuning_offset:.3f} semitones")
            if abs(tuning_offset) > 0.05:
                original_wav = librosa.effects.pitch_shift(
                    original_wav, sr=sr, n_steps=-tuning_offset)
                logger.info(f"Applied tuning compensation: {-tuning_offset:+.3f} semitones")

            # Compute CQT features
            inst_len = config.mp3['inst_len']
            song_hz = config.mp3['song_hz']
            feature = None
            current = 0
            while len(original_wav) > current + int(song_hz * inst_len):
                start_idx = int(current)
                end_idx = int(current + song_hz * inst_len)
                tmp = librosa.cqt(
                    original_wav[start_idx:end_idx], sr=sr,
                    n_bins=config.feature['n_bins'],
                    bins_per_octave=config.feature['bins_per_octave'],
                    hop_length=config.feature['hop_length'])
                feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
                current = end_idx

            tmp = librosa.cqt(
                original_wav[int(current):], sr=sr,
                n_bins=config.feature['n_bins'],
                bins_per_octave=config.feature['bins_per_octave'],
                hop_length=config.feature['hop_length'])
            feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
            feature = np.log(np.abs(feature) + 1e-6)

            time_unit = inst_len / config.model['timestep']

            feature = feature.T
            feature = (feature - self._mean) / self._std

            n_timestep = config.model['timestep']
            num_pad = n_timestep - (feature.shape[0] % n_timestep)
            feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
            num_instance = feature.shape[0] // n_timestep

            min_dur = 0.15 if use_constraint else self.min_duration

            events = []
            start_time = 0.0
            prev_chord = None
            prev_conf_sum = 0.0
            prev_conf_count = 0

            with torch.no_grad():
                feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self._device)
                for t in range(num_instance):
                    chunk = feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
                    encoder_output, _ = self._model.self_attn_layers(chunk)

                    # Standard BTC inference: self-attention output -> output_projection
                    # Note: The LSTM in OutputLayer is NOT used in the standard forward pass
                    # (SoftmaxOutputLayer.forward() only calls output_projection + softmax).
                    # Using the LSTM path produces near-uniform softmax (~1.3% per class)
                    # because it was never part of the trained inference pipeline.
                    logits = self._model.output_layer.output_projection(encoder_output).squeeze(0)

                    if use_constraint:
                        logits = logits + vocab_mask

                    # --- FIX 3: Use softmax for real confidence values ---
                    probs = torch.softmax(logits, dim=-1)
                    prediction = logits.argmax(dim=-1)

                    for i in range(n_timestep):
                        current_time = time_unit * (n_timestep * t + i)
                        if current_time > duration:
                            break

                        chord_idx = prediction[i].item()
                        frame_conf = probs[i, chord_idx].item()
                        if prev_chord is None:
                            prev_chord = chord_idx
                            prev_conf_sum = frame_conf
                            prev_conf_count = 1
                            start_time = current_time
                            continue

                        if chord_idx != prev_chord:
                            chord_name = _mir_to_simple(self._idx_to_chord[prev_chord])
                            dur = current_time - start_time
                            if dur >= min_dur and chord_name != 'N':
                                root, quality = _parse_chord(chord_name)
                                avg_conf = prev_conf_sum / max(prev_conf_count, 1)
                                events.append(ChordEvent(
                                    time=float(start_time),
                                    duration=float(dur),
                                    chord=chord_name,
                                    root=root,
                                    quality=quality,
                                    confidence=float(avg_conf),
                                ))
                            start_time = current_time
                            prev_chord = chord_idx
                            prev_conf_sum = frame_conf
                            prev_conf_count = 1
                        else:
                            prev_conf_sum += frame_conf
                            prev_conf_count += 1

            if prev_chord is not None:
                chord_name = _mir_to_simple(self._idx_to_chord[prev_chord])
                dur = duration - start_time
                if dur >= min_dur and chord_name != 'N':
                    root, quality = _parse_chord(chord_name)
                    avg_conf = prev_conf_sum / max(prev_conf_count, 1)
                    events.append(ChordEvent(
                        time=float(start_time),
                        duration=float(dur),
                        chord=chord_name,
                        root=root,
                        quality=quality,
                        confidence=float(avg_conf),
                    ))

            # --- FIX 4: Independent key detection from audio chromagram ---
            key = self._detect_key_from_audio(original_wav, sr)
            chord_key = self._detect_key_from_chords(events)
            logger.info(f"Key detection — audio-based: {key}, chord-based: {chord_key}")
            # Prefer audio-based key; fall back to chord-based if audio gives Unknown
            if key == "Unknown":
                key = chord_key

            # --- FIX 5: Key-constrained tuning drift correction ---
            # If chords are systematically off from the audio-detected key,
            # try shifting by +/-1 semitone to see if diatonic fit improves
            events = self._correct_tuning_drift(events, key)

            logger.info(f"BTC detected {len(events)} chord events, key={key}")
            return ChordProgression(chords=events, key=key)

        except Exception as e:
            logger.error(f"BTC chord detection failed: {e}", exc_info=True)
            return ChordProgression(chords=[], key="Unknown")

    def _detect_key_from_chords(self, events: List[ChordEvent]) -> str:
        """Estimate key from detected chords using duration-weighted root histogram."""
        if not events:
            return "C"

        # Build duration-weighted root histogram
        root_weight = np.zeros(12)
        for e in events:
            if e.root in NOTE_NAMES:
                idx = NOTE_NAMES.index(e.root)
                root_weight[idx] += e.duration

        if np.sum(root_weight) == 0:
            return "C"

        root_weight /= np.sum(root_weight)

        # Krumhansl-Kessler key profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        major_profile /= np.sum(major_profile)
        minor_profile /= np.sum(minor_profile)

        best_corr, best_key = -1, "C"
        for i in range(12):
            rotated = np.roll(root_weight, -i)
            mc = np.corrcoef(rotated, major_profile)[0, 1]
            mic = np.corrcoef(rotated, minor_profile)[0, 1]
            if mc > best_corr:
                best_corr, best_key = mc, NOTE_NAMES[i]
            if mic > best_corr:
                best_corr, best_key = mic, f"{NOTE_NAMES[i]}m"

        return best_key

    def _correct_tuning_drift(self, events: List[ChordEvent], key: str) -> List[ChordEvent]:
        """Correct systematic tuning drift by shifting chord roots if it improves diatonic fit.

        If the BTC model systematically assigns chords that are off by 1-2 semitones
        from the audio-detected key's diatonic set, shift all roots to fix this.
        """
        if not events or key == "Unknown":
            return events

        # Parse key
        key_root = key[0]
        key_rest = key[1:]
        if key_rest and key_rest[0] in ('#', 'b'):
            key_root = key[:2]
            key_rest = key[2:]
        is_minor = 'm' in key_rest

        # Normalize key root (flats to sharps only, since NOTE_NAMES uses sharps)
        FLAT_TO_SHARP = {'Bb': 'A#', 'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#'}
        key_root_norm = FLAT_TO_SHARP.get(key_root, key_root)

        if key_root_norm not in NOTE_NAMES:
            return events

        key_idx = NOTE_NAMES.index(key_root_norm)

        # Build diatonic set for the key
        if is_minor:
            # Natural minor: W-H-W-W-H-W-W (0,2,3,5,7,8,10)
            intervals = [0, 2, 3, 5, 7, 8, 10]
        else:
            # Major: W-W-H-W-W-W-H (0,2,4,5,7,9,11)
            intervals = [0, 2, 4, 5, 7, 9, 11]

        diatonic_set = set((key_idx + i) % 12 for i in intervals)

        def _normalize_root(root: str) -> str:
            """Normalize root to sharp notation used in NOTE_NAMES."""
            return FLAT_TO_SHARP.get(root, root)

        def _diatonic_score(evts: List[ChordEvent]) -> float:
            """Compute duration-weighted fraction of chords whose root is in the diatonic set."""
            total_dur = 0.0
            diatonic_dur = 0.0
            for e in evts:
                root = _normalize_root(e.root)
                if root not in NOTE_NAMES:
                    continue
                root_idx = NOTE_NAMES.index(root)
                total_dur += e.duration
                if root_idx in diatonic_set:
                    diatonic_dur += e.duration
            return diatonic_dur / total_dur if total_dur > 0 else 0

        def _shift_events(evts: List[ChordEvent], shift: int) -> List[ChordEvent]:
            """Shift all chord roots by `shift` semitones."""
            shifted = []
            for e in evts:
                root = _normalize_root(e.root)
                if root not in NOTE_NAMES:
                    shifted.append(e)
                    continue
                root_idx = NOTE_NAMES.index(root)
                new_root_idx = (root_idx + shift) % 12
                new_root = NOTE_NAMES[new_root_idx]
                # Rebuild chord name
                quality_str = e.chord[len(e.root):]  # everything after root
                new_chord = new_root + quality_str
                shifted.append(ChordEvent(
                    time=e.time,
                    duration=e.duration,
                    chord=new_chord,
                    root=new_root,
                    quality=e.quality,
                    confidence=e.confidence,
                ))
            return shifted

        current_score = _diatonic_score(events)
        logger.info(f"Diatonic fit before correction: {current_score:.1%}")

        # Strategy 1: Uniform shift by +/-1 semitone (for systematic tuning errors)
        # Only shift by 1 semitone — larger shifts suggest model confusion, not tuning
        best_shift = 0
        best_score = current_score
        for shift in [-1, 1]:
            shifted = _shift_events(events, shift)
            score = _diatonic_score(shifted)
            if score > best_score:
                best_score = score
                best_shift = shift

        if best_shift != 0 and best_score > current_score + 0.10:
            logger.info(f"Tuning drift correction: shifting roots by {best_shift:+d} semitones "
                        f"(diatonic fit {current_score:.1%} -> {best_score:.1%})")
            return _shift_events(events, best_shift)

        # Strategy 2: Conservative per-chord diatonic snap
        # Only snap non-diatonic chords that are exactly 1 semitone from a diatonic note
        # and only when the non-diatonic root has exactly one diatonic neighbor (unambiguous)
        if current_score < 0.70:
            snapped = []
            snap_count = 0
            for e in events:
                root = _normalize_root(e.root)
                if root not in NOTE_NAMES:
                    snapped.append(e)
                    continue
                root_idx = NOTE_NAMES.index(root)
                if root_idx in diatonic_set:
                    snapped.append(e)
                else:
                    up = (root_idx + 1) % 12
                    down = (root_idx - 1) % 12
                    # Only snap if exactly one neighbor is diatonic (unambiguous)
                    if up in diatonic_set and down not in diatonic_set:
                        new_root = NOTE_NAMES[up]
                        quality_str = e.chord[len(e.root):]
                        snapped.append(ChordEvent(
                            time=e.time, duration=e.duration,
                            chord=new_root + quality_str, root=new_root,
                            quality=e.quality, confidence=e.confidence,
                        ))
                        snap_count += 1
                    elif down in diatonic_set and up not in diatonic_set:
                        new_root = NOTE_NAMES[down]
                        quality_str = e.chord[len(e.root):]
                        snapped.append(ChordEvent(
                            time=e.time, duration=e.duration,
                            chord=new_root + quality_str, root=new_root,
                            quality=e.quality, confidence=e.confidence,
                        ))
                        snap_count += 1
                    else:
                        # Ambiguous or no diatonic neighbor — keep original
                        snapped.append(e)

            snap_score = _diatonic_score(snapped)
            if snap_score > current_score + 0.05:
                logger.info(f"Diatonic snap correction: {snap_count} chords snapped "
                            f"(fit {current_score:.1%} -> {snap_score:.1%})")
                return snapped

        logger.info(f"No tuning drift correction applied (diatonic fit: {current_score:.1%})")
        return events

    def _detect_key_from_audio(self, audio: np.ndarray, sr: int) -> str:
        """Estimate key directly from audio using chromagram + Krumhansl-Schmuckler.

        This is independent of chord detection results, avoiding the circular
        dependency where wrong chords produce a wrong key.
        """
        import librosa
        try:
            # Compute chromagram from audio
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=2048)
            # Average chroma across time to get pitch class distribution
            chroma_avg = np.mean(chroma, axis=1)

            if np.sum(chroma_avg) == 0:
                return "Unknown"

            chroma_avg /= np.sum(chroma_avg)

            # Krumhansl-Kessler key profiles
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                      2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                      2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            major_profile /= np.sum(major_profile)
            minor_profile /= np.sum(minor_profile)

            best_corr, best_key = -1, "C"
            for i in range(12):
                rotated = np.roll(chroma_avg, -i)
                mc = np.corrcoef(rotated, major_profile)[0, 1]
                mic = np.corrcoef(rotated, minor_profile)[0, 1]
                if mc > best_corr:
                    best_corr, best_key = mc, NOTE_NAMES[i]
                if mic > best_corr:
                    best_corr, best_key = mic, f"{NOTE_NAMES[i]}m"

            return best_key
        except Exception as e:
            logger.warning(f"Audio-based key detection failed: {e}")
            return "Unknown"

    def _detect_essentia(self, audio_path: str) -> ChordProgression:
        """Fallback: Essentia ChordsDetection."""
        try:
            import essentia.standard as es

            audio = es.MonoLoader(filename=str(audio_path), sampleRate=44100)()
            sr = 44100
            logger.info(f"Loaded {len(audio)/sr:.1f}s audio (Essentia fallback)")

            key_ext = es.KeyExtractor()
            key_name, scale, _ = key_ext(audio)
            detected_key = f"{key_name}m" if scale == "minor" else key_name

            frame_size = 8192
            hop = 4096
            w = es.Windowing(type='blackmanharris62')
            spectrum = es.Spectrum()
            peaks = es.SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.00001,
                                     minFrequency=40, maxFrequency=5000, maxPeaks=60)
            hpcp = es.HPCP(size=36, referenceFrequency=440, bandPreset=False,
                          minFrequency=40, maxFrequency=5000, weightType='cosine')
            chords_algo = es.ChordsDetection(hopSize=hop, sampleRate=sr)

            hpcp_frames = []
            for fstart in range(0, len(audio) - frame_size, hop):
                frame = audio[fstart:fstart + frame_size]
                h = hpcp(*peaks(spectrum(w(frame))))
                hpcp_frames.append(h)

            if not hpcp_frames:
                return ChordProgression(chords=[], key=detected_key)

            hpcp_array = np.array(hpcp_frames)
            chord_list, strength_list = chords_algo(hpcp_array)

            # Majority vote smoothing
            half = 10
            smoothed = []
            for i in range(len(chord_list)):
                s, e = max(0, i - half), min(len(chord_list), i + half + 1)
                window = [c for c in chord_list[s:e] if c != 'N']
                smoothed.append(Counter(window).most_common(1)[0][0] if window else chord_list[i])

            # Consolidate
            events = []
            current = smoothed[0]
            start_idx = 0
            for i in range(1, len(smoothed)):
                if smoothed[i] != current:
                    t0 = start_idx * hop / sr
                    dur = (i - start_idx) * hop / sr
                    if dur >= self.min_duration and current != 'N':
                        root, quality = _parse_chord(current)
                        events.append(ChordEvent(
                            time=float(t0), duration=float(dur),
                            chord=current, root=root, quality=quality, confidence=0.6))
                    current = smoothed[i]
                    start_idx = i

            logger.info(f"Essentia detected {len(events)} chord events, key={detected_key}")
            return ChordProgression(chords=events, key=detected_key)

        except Exception as e:
            logger.error(f"Essentia fallback failed: {e}", exc_info=True)
            return ChordProgression(chords=[], key="Unknown")
