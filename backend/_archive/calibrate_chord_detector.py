"""
Stem-Aware Chord Detector Calibration Script
=============================================
Evaluates and tunes the stem chord detector against:
1. GuitarSet JAMS annotations (chord naming accuracy)
2. Jeff's songs with known chord charts (full pipeline accuracy)

Usage:
    python calibrate_chord_detector.py --guitarset
    python calibrate_chord_detector.py --songs thunderhead
    python calibrate_chord_detector.py --songs all
    python calibrate_chord_detector.py --separate --max-songs 3
    python calibrate_chord_detector.py --sweep
    python calibrate_chord_detector.py --all
"""

import json
import logging
import sys
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from stem_chord_detector import (
    detect_notes_in_stem, detect_bass_root,
    notes_to_chord, merge_consecutive_chords,
    cross_validate_segments, _dominant_pitch_classes_in_segment,
    _dominant_bass_in_segment, _onset_weighted_pitch_classes_in_segment,
    ChordEvent, ChordProgression,
    NOTE_NAMES, CHORD_INTERVALS, ENHARMONIC, load_v8_classes,
    detect_key_from_chords, segment_by_pitch_change,
)

logger = logging.getLogger(__name__)

# ============ PATHS ============

GUITARSET_ANNOTATIONS = (Path(__file__).parent.parent /
    'train_tab_model' / 'trimplexx' / 'python' / '_mir_datasets_storage' / 'annotation')
CHORD_LIBRARY = Path(__file__).parent / 'chord_library' / 'kozelski'
STEMS_DIR = Path(__file__).parent / 'outputs'

AUDIO_SOURCES = {
    'collector': Path.home() / 'Desktop' / 'MUSIC' / 'Kozelski' / 'Kozelski _ Collector',
    'systematic_static': Path.home() / 'Desktop' / 'MUSIC' / 'Kozelski' / 'Kozelski_ Systematic Static',
    'outervention': Path.home() / 'Desktop' / 'MUSIC' / 'The Outervention' / 'ACCLIMATOR',
}

SONG_AUDIO_MAP = {
    'thunderhead':              ('collector', '01 Thunderhead.m4a'),
    'the-city-gonna-teach-you': ('collector', '02 The Citys Gonna Teach You.m4a'),
    'living-water':             ('collector', '03 Living Water.m4a'),
    'the-blinding-glow':        ('collector', '04 The Blinding Glow.m4a'),
    'cold-dice':                ('collector', '05 Cold Dice.m4a'),
    'mystified':                ('collector', '06 Mystified.m4a'),
    'clever-devil':             ('collector', '07 Clever Devil.m4a'),
    'the-time-comes':           ('collector', '08 The Time Comes.m4a'),
    'solar-connection':         ('collector', '09 Solar Connection.m4a'),
    'born-in-zen':              ('systematic_static', 'Born In Zen.wav'),
    'clandestine-sam':          ('systematic_static', 'Clandestine Sam.wav'),
    'climbing-the-bars':        ('systematic_static', 'Climbing The Bars.wav'),
    'eye-to-eye':               ('systematic_static', 'Eye To Eye.wav'),
    'lady-in-the-stars':        ('systematic_static', 'Lady In The Stars.wav'),
    'natural-grow':             ('systematic_static', 'Natural Grow.wav'),
    'silent-life':              ('systematic_static', 'Silent Life.wav'),
    'simpler-understanding':    ('systematic_static', 'Simpler Understanding.wav'),
}


# ============ JAMS CHORD PARSING ============

JAMS_QUALITY_MAP = {
    'maj': 'maj', 'min': 'min', 'dim': 'dim', 'aug': 'aug',
    'maj7': 'maj7', 'min7': 'min7', '7': '7',
    'maj6': '6', 'min6': 'min6',
    'sus2': 'sus2', 'sus4': 'sus4',
    'dim7': 'dim7', 'hdim7': 'hdim7',
    '9': '9', 'min9': 'min9', 'maj9': 'maj9',
}

JAMS_NOTE_TO_PC = {
    'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6,
    'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
    'B': 11, 'Cb': 11,
}


def parse_jams_chord(chord_str):
    if chord_str in ('N', 'X', '') or not chord_str:
        return (None, None, None, None)
    base = chord_str.split('/')[0]
    parts = base.split(':')
    if len(parts) != 2:
        return (None, None, None, None)
    root_str, quality_str = parts
    root_pc = JAMS_NOTE_TO_PC.get(root_str)
    if root_pc is None:
        return (None, None, None, None)
    quality_base = quality_str.split('(')[0]
    quality = JAMS_QUALITY_MAP.get(quality_base, quality_base)
    template = CHORD_INTERVALS.get(quality)
    if template:
        pitch_classes = set((root_pc + iv) % 12 for iv in template)
    else:
        pitch_classes = {root_pc}
    root_name = NOTE_NAMES[root_pc]
    if quality == 'maj':
        chord_name = root_name
    elif quality == 'min':
        chord_name = f"{root_name}m"
    else:
        chord_name = f"{root_name}{quality}"
    return (chord_name, root_pc, quality, pitch_classes)


def normalize_chord_name(chord):
    if not chord or chord == 'N':
        return ('N', 'none', 'N')
    base = chord.split('/')[0]
    if ':' in base:
        parsed = parse_jams_chord(chord)
        if parsed[0]:
            return (NOTE_NAMES[parsed[1]], parsed[2], parsed[0])
        return ('N', 'none', 'N')
    root = base[0]
    rest = base[1:]
    if rest and rest[0] in '#b':
        root += rest[0]
        rest = rest[1:]
    root_pc = JAMS_NOTE_TO_PC.get(root)
    if root_pc is None:
        return ('N', 'none', 'N')
    root_name = NOTE_NAMES[root_pc]
    qmap = {
        '': 'maj', 'm': 'min', 'min': 'min', 'maj': 'maj',
        '7': '7', 'm7': 'min7', 'min7': 'min7', 'maj7': 'maj7',
        'dim': 'dim', 'dim7': 'dim7', 'aug': 'aug',
        'sus2': 'sus2', 'sus4': 'sus4',
        '6': '6', 'm6': 'min6', 'min6': 'min6',
        '9': '9', 'add9': 'add9', 'hdim7': 'hdim7', 'mMaj7': 'mMaj7',
    }
    quality = qmap.get(rest, rest)
    if quality == 'maj':
        name = root_name
    elif quality == 'min':
        name = f"{root_name}m"
    else:
        name = f"{root_name}{quality}"
    return (root_name, quality, name)


# ============ EVAL RESULT ============

@dataclass
class EvalResult:
    name: str
    total_ground_truth: int = 0
    total_detected: int = 0
    vocab_matched: int = 0
    vocab_missed: int = 0
    vocab_extra: int = 0
    vocab_gt_set: set = field(default_factory=set)
    vocab_det_set: set = field(default_factory=set)
    root_correct_duration: float = 0.0
    root_total_duration: float = 0.0
    quality_correct_duration: float = 0.0
    quality_total_duration: float = 0.0
    processing_time: float = 0.0

    @property
    def vocab_precision(self):
        return self.vocab_matched / max(self.vocab_matched + self.vocab_extra, 1)

    @property
    def vocab_recall(self):
        return self.vocab_matched / max(self.vocab_matched + self.vocab_missed, 1)

    @property
    def vocab_f1(self):
        p, r = self.vocab_precision, self.vocab_recall
        return 2 * p * r / max(p + r, 1e-9)

    @property
    def root_accuracy(self):
        return self.root_correct_duration / max(self.root_total_duration, 1e-9)

    @property
    def quality_accuracy(self):
        return self.quality_correct_duration / max(self.quality_total_duration, 1e-9)

    def summary(self):
        lines = [
            f"  {self.name}:",
            f"    Vocab: recall={self.vocab_recall:.0%} precision={self.vocab_precision:.0%} F1={self.vocab_f1:.2f}",
            f"    Root accuracy:    {self.root_accuracy:.1%}",
            f"    Quality accuracy: {self.quality_accuracy:.1%} (given correct root)",
            f"    Segments: {self.total_detected} detected vs {self.total_ground_truth} GT",
            f"    Time: {self.processing_time:.1f}s",
        ]
        if self.vocab_missed > 0:
            missed = self.vocab_gt_set - self.vocab_det_set
            lines.append(f"    Missed: {', '.join(sorted(missed))}")
        if self.vocab_extra > 0:
            extra = self.vocab_det_set - self.vocab_gt_set
            if len(extra) <= 15:
                lines.append(f"    Extra: {', '.join(sorted(extra))}")
            else:
                lines.append(f"    Extra: {len(extra)} unique spurious chords")
        return '\n'.join(lines)


# ============ EVALUATION FUNCTIONS ============

def evaluate_chord_naming(gt_chords):
    correct = root_correct = quality_correct = total = 0
    mismatches = []
    for gt_name, gt_root_pc, gt_quality, gt_pcs in gt_chords:
        if gt_name is None:
            continue
        det_name, det_root, det_quality, det_bass, det_conf = notes_to_chord(
            gt_pcs, bass_root=gt_root_pc)
        gt_norm = normalize_chord_name(gt_name)
        det_norm = normalize_chord_name(det_name)
        total += 1
        if gt_norm[0] == det_norm[0]:
            root_correct += 1
            if gt_norm[1] == det_norm[1]:
                quality_correct += 1
                correct += 1
            else:
                mismatches.append((gt_name, det_name, 'quality'))
        else:
            mismatches.append((gt_name, det_name, 'root'))
    return {
        'total': total, 'correct': correct,
        'root_correct': root_correct, 'quality_correct': quality_correct,
        'accuracy': correct / max(total, 1),
        'root_accuracy': root_correct / max(total, 1),
        'quality_accuracy': quality_correct / max(root_correct, 1),
        'mismatches': mismatches[:30],
    }


def evaluate_against_chart(detected, chart):
    result = EvalResult(name=chart.get('title', 'Unknown'))
    gt_chords_used = set()
    for c in chart.get('chords_used', []):
        _, _, n = normalize_chord_name(c)
        gt_chords_used.add(n)
    det_chords_used = set()
    for c in detected.chords:
        _, _, n = normalize_chord_name(c.chord.split('/')[0])
        det_chords_used.add(n)
    result.vocab_gt_set = gt_chords_used
    result.vocab_det_set = det_chords_used
    result.vocab_matched = len(gt_chords_used & det_chords_used)
    result.vocab_missed = len(gt_chords_used - det_chords_used)
    result.vocab_extra = len(det_chords_used - gt_chords_used)

    gt_sequence = []
    for section in chart.get('sections', []):
        for line in section.get('lines', []):
            chords_str = line.get('chords', '')
            if chords_str:
                for ch in chords_str.split():
                    _, _, n = normalize_chord_name(ch)
                    gt_sequence.append(n)
    result.total_ground_truth = len(gt_sequence)
    result.total_detected = len(detected.chords)

    total_duration = sum(c.duration for c in detected.chords)
    result.root_total_duration = total_duration
    gt_roots = set(normalize_chord_name(c)[0] for c in chart.get('chords_used', []))

    for c in detected.chords:
        det_root, det_quality, det_name = normalize_chord_name(c.chord.split('/')[0])
        if det_root in gt_roots:
            result.root_correct_duration += c.duration
            result.quality_total_duration += c.duration
            if det_name in gt_chords_used:
                result.quality_correct_duration += c.duration
    return result


# ============ PARAMETER SET ============

@dataclass
class ParamSet:
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_length: float = 80
    min_segment_duration: float = 0.5
    onset_delta: float = 0.15
    onset_wait: int = 8
    pc_voting_threshold: float = 0.25
    bass_conf_threshold: float = 0.3
    merge_min_duration: float = 0.15
    use_pitch_change: bool = True
    smoothing_window: int = 5

    def label(self):
        seg_type = "pc_change" if self.use_pitch_change else f"onset(d={self.onset_delta})"
        return (f"seg={self.min_segment_duration:.2f} {seg_type} "
                f"pcvote={self.pc_voting_threshold:.2f} merge={self.merge_min_duration:.2f} "
                f"bass={self.bass_conf_threshold:.2f} sw={self.smoothing_window}")

    def clone(self, **overrides):
        d = {k: getattr(self, k) for k in [
            'onset_threshold', 'frame_threshold', 'min_note_length',
            'min_segment_duration', 'onset_delta', 'onset_wait',
            'pc_voting_threshold', 'bass_conf_threshold', 'merge_min_duration',
            'use_pitch_change', 'smoothing_window']}
        d.update(overrides)
        return ParamSet(**d)


# ============ DETECTION WITH PARAMS ============

def run_detection_with_params(guitar_path, bass_path, piano_path,
                              params, cached_notes=None):
    import librosa

    guitar_notes = (cached_notes or {}).get('guitar')
    piano_notes = (cached_notes or {}).get('piano')
    bass_frames = (cached_notes or {}).get('bass')

    if guitar_notes is None and guitar_path and Path(guitar_path).exists():
        guitar_notes = detect_notes_in_stem(
            guitar_path, 'guitar',
            onset_threshold=params.onset_threshold,
            frame_threshold=params.frame_threshold,
            min_note_length=params.min_note_length)
    if piano_notes is None and piano_path and Path(piano_path).exists():
        piano_notes = detect_notes_in_stem(
            piano_path, 'piano',
            onset_threshold=params.onset_threshold,
            frame_threshold=params.frame_threshold,
            min_note_length=params.min_note_length)
    if bass_frames is None and bass_path and Path(bass_path).exists():
        bass_frames = detect_bass_root(bass_path)

    cache = {}
    if guitar_notes: cache['guitar'] = guitar_notes
    if piano_notes: cache['piano'] = piano_notes
    if bass_frames: cache['bass'] = bass_frames

    if not guitar_notes and not piano_notes:
        return ChordProgression(chords=[], key='Unknown'), cache

    primary_stem = guitar_path if guitar_notes else piano_path
    primary_notes = guitar_notes if guitar_notes else piano_notes
    y, sr = librosa.load(str(primary_stem), sr=22050, mono=True, duration=None)
    audio_duration = len(y) / sr

    if params.use_pitch_change and primary_notes:
        # Pitch-change segmentation: detects when notes actually change
        onset_times = segment_by_pitch_change(
            primary_notes['pitch_class_frames'],
            min_segment_duration=params.min_segment_duration,
            smoothing_window=params.smoothing_window,
        )
    else:
        # Raw onset detection (over-segments but catches all attacks)
        onset_fr = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=256, backtrack=True, units='frames',
            pre_max=3, post_max=3, pre_avg=3, post_avg=5,
            delta=params.onset_delta, wait=params.onset_wait)
        onset_times = onset_fr * (256 / sr)

        if len(onset_times) > 1:
            filt = [onset_times[0]]
            for t in onset_times[1:]:
                if t - filt[-1] >= params.min_segment_duration:
                    filt.append(t)
            onset_times = np.array(filt)
        else:
            onset_times = np.array([0.0])

        if len(onset_times) == 0 or onset_times[0] > 0.05:
            onset_times = np.concatenate([[0.0], onset_times])
        onset_times = onset_times.tolist()

    if onset_times[-1] < audio_duration - 0.5:
        onset_times.append(audio_duration)

    v8_classes = load_v8_classes()
    chord_events = []

    for i in range(len(onset_times) - 1):
        s, e = onset_times[i], onset_times[i + 1]
        dur = e - s
        if dur < 0.05:
            continue

        g_pcs, g_conf = (set(), 0.0)
        if guitar_notes:
            g_pcs, g_conf = _onset_weighted_pitch_classes_in_segment(
                guitar_notes, s, e,
                threshold=params.pc_voting_threshold)

        p_pcs, p_conf = (set(), 0.0)
        if piano_notes:
            p_pcs, p_conf = _onset_weighted_pitch_classes_in_segment(
                piano_notes, s, e,
                threshold=params.pc_voting_threshold)

        bass_root, bass_conf = None, 0.0
        if bass_frames:
            votes = Counter()
            confs = []
            for t, pc, conf in bass_frames:
                if t < s: continue
                if t >= e: break
                if pc is not None and conf > params.bass_conf_threshold:
                    votes[pc] += 1
                    confs.append(conf)
            if votes:
                bass_root = votes.most_common(1)[0][0]
                bass_conf = float(np.mean(confs))

        combined_pcs, final_bass, combined_conf = cross_validate_segments(
            g_pcs, bass_root, p_pcs, g_conf, p_conf, bass_conf)

        chord_name, root, quality, bass_name, chord_conf = notes_to_chord(
            combined_pcs, final_bass, v8_classes)

        if chord_name == 'N':
            continue

        final_conf = min(0.99, combined_conf * 0.6 + chord_conf * 0.4)
        chord_events.append(ChordEvent(
            time=round(s, 3), duration=round(dur, 3),
            chord=chord_name, root=root, quality=quality,
            confidence=round(final_conf, 3), bass=bass_name))

    chord_events = merge_consecutive_chords(chord_events, min_duration=params.merge_min_duration)
    key = detect_key_from_chords(chord_events)
    return ChordProgression(chords=chord_events, key=key), cache


# ============ GUITARSET EVAL ============

def run_guitarset_naming_eval():
    import jams
    print("\n" + "=" * 70)
    print("GUITARSET CHORD NAMING ACCURACY TEST")
    print("=" * 70)

    if not GUITARSET_ANNOTATIONS.exists():
        print(f"ERROR: Not found: {GUITARSET_ANNOTATIONS}")
        return None

    jams_files = sorted(GUITARSET_ANNOTATIONS.glob('*.jams'))
    print(f"Found {len(jams_files)} JAMS files\n")

    all_gt = []
    quality_counts = Counter()
    for jf in jams_files:
        try:
            jam = jams.load(str(jf))
        except Exception:
            continue
        chord_anns = [ns for ns in jam.annotations if ns.namespace == 'chord']
        if not chord_anns:
            continue
        for obs in chord_anns[0].data:
            parsed = parse_jams_chord(obs.value)
            if parsed[0] is not None:
                all_gt.append(parsed)
                quality_counts[parsed[2]] += 1

    print(f"Total chord segments: {len(all_gt)}")
    print(f"Quality distribution: {dict(quality_counts.most_common(15))}\n")

    results = evaluate_chord_naming(all_gt)
    print(f"Chord Naming Results:")
    print(f"  Total:      {results['total']}")
    print(f"  Exact:      {results['correct']} ({results['accuracy']:.1%})")
    print(f"  Root OK:    {results['root_correct']} ({results['root_accuracy']:.1%})")
    print(f"  Quality OK: {results['quality_correct']}/{results['root_correct']} "
          f"({results['quality_accuracy']:.1%})")

    if results['mismatches']:
        root_mm = [(g, d) for g, d, t in results['mismatches'] if t == 'root']
        qual_mm = [(g, d) for g, d, t in results['mismatches'] if t == 'quality']
        if root_mm:
            print(f"\n  Root errors:")
            for g, d in root_mm[:10]:
                print(f"    {g:12s} -> {d}")
        if qual_mm:
            print(f"\n  Quality errors:")
            for g, d in qual_mm[:10]:
                print(f"    {g:12s} -> {d}")
    return results


# ============ JEFF'S SONGS ============

def load_chart(slug):
    p = CHORD_LIBRARY / f"{slug}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def find_stems(slug):
    if slug == 'thunderhead':
        d = STEMS_DIR / 'modal_test'
        if d.exists() and (d / 'guitar.mp3').exists():
            return {'guitar': str(d / 'guitar.mp3'),
                    'bass': str(d / 'bass.mp3'),
                    'piano': str(d / 'piano.mp3')}
    d = STEMS_DIR / 'calibration' / slug
    if d.exists():
        stems = {}
        for stem in ['guitar', 'bass', 'piano']:
            for ext in ['.mp3', '.wav']:
                p = d / f"{stem}{ext}"
                if p.exists():
                    stems[stem] = str(p)
                    break
        if stems:
            return stems
    return None


def run_song_eval(slug, params=None, cached_notes=None):
    chart = load_chart(slug)
    if not chart:
        print(f"  SKIP {slug}: no chart")
        return None, {}
    stems = find_stems(slug)
    if not stems:
        print(f"  SKIP {slug}: no stems")
        return None, {}
    if params is None:
        params = ParamSet()
    print(f"  {chart['title']}...", end='', flush=True)
    t0 = time.time()
    prog, cache = run_detection_with_params(
        stems.get('guitar'), stems.get('bass'), stems.get('piano'),
        params, cached_notes)
    elapsed = time.time() - t0
    result = evaluate_against_chart(prog, chart)
    result.processing_time = elapsed
    print(f" {elapsed:.1f}s, {len(prog.chords)} chords, F1={result.vocab_f1:.2f}")
    return result, cache


def run_songs_eval(slugs=None, params=None):
    print("\n" + "=" * 70)
    print("JEFF'S SONGS: FULL PIPELINE EVALUATION")
    print("=" * 70)
    if params:
        print(f"Params: {params.label()}")
    if slugs is None or slugs == ['all']:
        slugs = [p.stem for p in sorted(CHORD_LIBRARY.glob('*.json'))]
    results = []
    for slug in slugs:
        r, _ = run_song_eval(slug, params)
        if r:
            results.append(r)
    if not results:
        print("No songs had both charts and stems.")
        return results

    print(f"\n{'='*70}")
    print(f"AGGREGATE ({len(results)} songs)")
    print(f"{'='*70}")
    tm = sum(r.vocab_matched for r in results)
    tgt = sum(r.vocab_matched + r.vocab_missed for r in results)
    tdet = sum(r.vocab_matched + r.vocab_extra for r in results)
    trc = sum(r.root_correct_duration for r in results)
    trt = sum(r.root_total_duration for r in results)
    tqc = sum(r.quality_correct_duration for r in results)
    tqt = sum(r.quality_total_duration for r in results)
    print(f"  Vocab recall:     {tm}/{tgt} = {tm/max(tgt,1):.1%}")
    print(f"  Vocab precision:  {tm}/{tdet} = {tm/max(tdet,1):.1%}")
    print(f"  Root accuracy:    {trc/max(trt,1e-9):.1%}")
    print(f"  Quality accuracy: {tqc/max(tqt,1e-9):.1%}")
    for r in results:
        print()
        print(r.summary())
    return results


# ============ PARAMETER SWEEP ============

def run_parameter_sweep(slugs=None):
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP")
    print("=" * 70)

    if slugs is None:
        slugs = [p.stem for p in sorted(CHORD_LIBRARY.glob('*.json'))]
    available = []
    for slug in slugs:
        chart = load_chart(slug)
        stems = find_stems(slug)
        if chart and stems:
            available.append((slug, chart, stems))

    if not available:
        print("No songs have both charts and stems. Run --separate first.")
        return None, 0

    print(f"Songs: {[s for s, _, _ in available]}")

    # Cache Basic Pitch output (expensive, run once)
    print("\nCaching Basic Pitch output...")
    caches = {}
    for slug, chart, stems in available:
        print(f"  {chart['title']}...", end='', flush=True)
        t0 = time.time()
        _, cache = run_detection_with_params(
            stems.get('guitar'), stems.get('bass'), stems.get('piano'), ParamSet())
        caches[slug] = cache
        print(f" {time.time()-t0:.1f}s")

    # Coordinate descent
    print("\nSweeping parameters (coordinate descent)...")
    grid = {
        'smoothing_window': [3, 5, 8, 12, 20],
        'min_segment_duration': [0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0],
        'pc_voting_threshold': [0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
        'merge_min_duration': [0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0],
        'bass_conf_threshold': [0.2, 0.3, 0.4, 0.5],
    }

    best = ParamSet()
    best_score = -1

    for param_name, values in grid.items():
        print(f"\n  {param_name}: {values}")
        scores = []
        for val in values:
            p = best.clone(**{param_name: val})
            total_f1 = total_root = total_qual = 0
            n = 0
            for slug, chart, stems in available:
                prog, _ = run_detection_with_params(
                    stems.get('guitar'), stems.get('bass'), stems.get('piano'),
                    p, caches[slug])
                ev = evaluate_against_chart(prog, chart)
                total_f1 += ev.vocab_f1
                total_root += ev.root_accuracy
                total_qual += ev.quality_accuracy
                n += 1

            af1 = total_f1 / max(n, 1)
            ar = total_root / max(n, 1)
            aq = total_qual / max(n, 1)
            combined = af1 * 0.4 + ar * 0.3 + aq * 0.3
            scores.append((val, combined, af1, ar, aq))
            print(f"    {param_name}={val}: score={combined:.3f} "
                  f"F1={af1:.2f} root={ar:.1%} qual={aq:.1%}")

        bval, bscore = max(scores, key=lambda x: x[1])[:2]
        setattr(best, param_name, bval)
        if bscore > best_score:
            best_score = bscore
        print(f"  -> Best {param_name} = {bval}")

    print(f"\n{'='*70}")
    print(f"OPTIMAL PARAMETERS (score={best_score:.3f})")
    print(f"{'='*70}")
    for k in ['onset_threshold', 'frame_threshold', 'min_note_length',
              'min_segment_duration', 'onset_delta', 'onset_wait',
              'pc_voting_threshold', 'bass_conf_threshold', 'merge_min_duration']:
        print(f"  {k}: {getattr(best, k)}")

    print(f"\nFinal evaluation with optimal params:")
    run_songs_eval([s for s, _, _ in available], best)
    return best, best_score


# ============ STEM SEPARATION ============

def separate_stems_for_calibration(slugs=None, max_songs=5):
    print("\n" + "=" * 70)
    print("STEM SEPARATION FOR CALIBRATION")
    print("=" * 70)
    if slugs is None:
        slugs = [p.stem for p in sorted(CHORD_LIBRARY.glob('*.json'))]

    to_sep = []
    for slug in slugs:
        chart = load_chart(slug)
        stems = find_stems(slug)
        if chart and not stems and slug in SONG_AUDIO_MAP:
            album, filename = SONG_AUDIO_MAP[slug]
            audio_path = AUDIO_SOURCES[album] / filename
            if audio_path.exists():
                to_sep.append((slug, chart['title'], str(audio_path)))

    if not to_sep:
        print("All charted songs already have stems (or no audio found).")
        return

    print(f"Need separation: {len(to_sep)} songs")
    import subprocess
    cal_dir = STEMS_DIR / 'calibration'
    cal_dir.mkdir(exist_ok=True)

    for slug, title, audio_path in to_sep[:max_songs]:
        out_dir = cal_dir / slug
        out_dir.mkdir(exist_ok=True)
        print(f"\n  Separating: {title}")
        cmd = [sys.executable, '-m', 'demucs', '-n', 'htdemucs_6s',
               '--out', str(cal_dir / '_demucs_tmp'), str(audio_path)]
        t0 = time.time()
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0
            if result.returncode == 0:
                tmp_dir = cal_dir / '_demucs_tmp' / 'htdemucs_6s'
                if tmp_dir.exists():
                    for d in tmp_dir.iterdir():
                        if d.is_dir():
                            for f in d.iterdir():
                                f.rename(out_dir / f.name)
                            try: d.rmdir()
                            except: pass
                    try: tmp_dir.rmdir()
                    except: pass
                    try: (cal_dir / '_demucs_tmp').rmdir()
                    except: pass
                print(f"  Done in {elapsed:.0f}s: {[f.name for f in out_dir.iterdir()]}")
            else:
                print(f"  FAILED: {result.stderr[:300]}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
        except Exception as e:
            print(f"  ERROR: {e}")


# ============ MAIN ============

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Calibrate stem chord detector')
    parser.add_argument('--guitarset', action='store_true')
    parser.add_argument('--songs', nargs='*', default=None)
    parser.add_argument('--sweep', action='store_true')
    parser.add_argument('--separate', nargs='*', default=None)
    parser.add_argument('--max-songs', type=int, default=5)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')

    if args.all or args.guitarset:
        run_guitarset_naming_eval()
    if args.separate is not None:
        separate_stems_for_calibration(args.separate or None, args.max_songs)
    if args.songs is not None:
        run_songs_eval(args.songs or ['all'])
    if args.all or args.sweep:
        run_parameter_sweep()
    if not any([args.all, args.guitarset, args.songs is not None,
                args.sweep, args.separate is not None]):
        parser.print_help()


if __name__ == '__main__':
    main()
