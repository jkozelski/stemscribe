"""
Lyrics Extractor — extracts timestamped lyrics from a vocals stem using faster-whisper.

Produces line-level lyrics with start/end times for use in chord chart assembly,
karaoke mode, and the StemScriber lyrics pipeline.

Uses the isolated vocals stem for much cleaner transcription than full mix.

When audio_path and chords are provided, uses beat detection (librosa) to align
lyric line breaks with musical measure boundaries and chord changes instead of
Whisper's speech pauses.

Usage:
    from lyrics_extractor import extract_lyrics
    lines = extract_lyrics("/path/to/vocals.mp3", "/path/to/output_dir")
    # [{"text": "When the time comes around", "start_time": 12.5, "end_time": 15.2}, ...]

    # Beat-aligned mode (better for chord charts):
    lines = extract_lyrics("/path/to/vocals.mp3", "/path/to/output_dir",
                           audio_path="/path/to/full_mix.mp3",
                           chords=[{"chord": "Am", "time": 0.0}, ...])
"""

import logging
import json
import re
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded whisper model
_model = None
_model_size = "large-v3"


def _get_model():
    """Lazy-load the faster-whisper model. Shares singleton with word_timestamps when possible."""
    global _model
    if _model is not None:
        return _model

    # Try to reuse the already-loaded model from word_timestamps
    try:
        import word_timestamps
        if word_timestamps._model is not None:
            _model = word_timestamps._model
            return _model
    except (ImportError, AttributeError):
        pass

    from faster_whisper import WhisperModel
    logger.info(f"Loading faster-whisper model ({_model_size}) for lyrics extraction...")
    _model = WhisperModel(_model_size, device="cpu", compute_type="float32")
    logger.info("faster-whisper model loaded")
    return _model


# Whisper artifacts to strip (bracketed tags)
_ARTIFACTS = re.compile(
    r'\[(?:Music|Applause|Laughter|Cheering|Silence|Inaudible|'
    r'Background\s*(?:music|noise)|Singing|Instrumental)\]',
    re.IGNORECASE,
)

# Common Whisper hallucinations (full-line patterns)
_HALLUCINATIONS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'^\s*Thank you\.?\s*$',
        r'^\s*Thanks for watching\.?\s*$',
        r'^\s*Subscribe\.?\s*$',
        r'^\s*Please subscribe\.?\s*$',
        r'^\s*Like and subscribe\.?\s*$',
        r'^\s*\.\.\.\s*$',
        r'^\s*\.\s*$',
        r'^\s*♪+\s*$',
    ]
]

# Sentence-ending punctuation
_SENTENCE_END = re.compile(r'[.!?]$')


def _clean_text(text: str) -> str:
    """Clean up common Whisper transcription artifacts."""
    # Remove bracketed artifacts
    text = _ARTIFACTS.sub('', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Check for hallucination patterns
    for pattern in _HALLUCINATIONS:
        if pattern.match(text):
            return ''

    # Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    # Don't end lyrics with a period (looks wrong on chord charts)
    text = text.rstrip('.')

    return text


def _deduplicate_lines(lines: List[Dict]) -> List[Dict]:
    """
    Remove duplicate/near-duplicate consecutive lines.
    Whisper sometimes repeats phrases, especially on sustained notes or choruses.
    """
    if not lines:
        return lines

    result = [lines[0]]
    for line in lines[1:]:
        prev_text = result[-1]['text'].lower().strip()
        curr_text = line['text'].lower().strip()

        # Skip exact duplicates
        if curr_text == prev_text:
            continue

        # Skip if one is a substring of the other and they're close together
        time_gap = line['start_time'] - result[-1]['end_time']
        if time_gap < 2.0:
            if curr_text in prev_text or prev_text in curr_text:
                # Keep the longer one
                if len(curr_text) > len(prev_text):
                    result[-1] = line
                continue

        result.append(line)

    return result


def _detect_beats(audio_path: str) -> Tuple[float, List[float], List[float]]:
    """
    Detect beats and measure boundaries using librosa.

    Args:
        audio_path: Path to audio file (full mix preferred for clearer rhythm)

    Returns:
        (tempo_bpm, beat_times, measure_boundaries)
        - tempo_bpm: estimated BPM
        - beat_times: time in seconds of each beat
        - measure_boundaries: time in seconds of each measure start (every 4 beats for 4/4)
    """
    import librosa
    import numpy as np

    logger.info(f"Detecting beats from {os.path.basename(audio_path)}...")

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    # librosa may return tempo as an array
    if hasattr(tempo, '__len__'):
        tempo = float(tempo[0])
    else:
        tempo = float(tempo)

    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()

    # Group beats into measures (assume 4/4 time)
    beats_per_measure = 4
    measure_boundaries = []
    for i in range(0, len(beat_times), beats_per_measure):
        measure_boundaries.append(beat_times[i])

    logger.info(f"Beat detection: {tempo:.1f} BPM, {len(beat_times)} beats, "
                f"{len(measure_boundaries)} measures")

    return tempo, beat_times, measure_boundaries


def _find_chord_boundaries(chords: List[Dict]) -> List[float]:
    """
    Extract times where chord changes occur.
    Returns sorted list of chord change times.
    """
    if not chords:
        return []
    sorted_chords = sorted(chords, key=lambda c: c.get('time', 0))
    boundaries = []
    prev_chord = None
    for c in sorted_chords:
        if c.get('chord') != prev_chord:
            boundaries.append(c['time'])
            prev_chord = c['chord']
    return boundaries


def _find_best_break_points(measure_boundaries: List[float],
                            chord_boundaries: List[float],
                            tolerance: float = 1.0) -> List[float]:
    """
    Merge measure boundaries and chord change points into a unified set of
    candidate line break points.

    Priority: chord changes that fall near a measure boundary are preferred.
    Chord changes mid-measure are also included as break candidates.

    Returns sorted list of break point times.
    """
    break_points = set()

    # All measure boundaries are candidates
    for mb in measure_boundaries:
        break_points.add(round(mb, 3))

    # Chord changes that don't align with a measure boundary — add them too
    for cb in chord_boundaries:
        # Check if there's already a nearby measure boundary
        near_measure = any(abs(cb - mb) < tolerance for mb in measure_boundaries)
        if not near_measure:
            break_points.add(round(cb, 3))

    return sorted(break_points)


def _collect_words_from_segments(segments: list) -> List[Dict]:
    """
    Flatten all Whisper segments into a single list of word dicts:
    [{"text": "word", "start": float, "end": float}, ...]
    """
    words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                text = w.word.strip()
                if text:
                    words.append({
                        'text': text,
                        'start': w.start,
                        'end': w.end,
                    })
        else:
            # No word timestamps — treat whole segment as one "word"
            text = segment.text.strip()
            if text:
                words.append({
                    'text': text,
                    'start': segment.start,
                    'end': segment.end,
                })
    return words


def _split_lines_on_beats(words: List[Dict],
                          break_points: List[float],
                          chords: Optional[List[Dict]] = None,
                          min_words_per_line: int = 2,
                          max_words_per_line: int = 12) -> List[Dict]:
    """
    Split words into lyric lines aligned to beat/measure/chord boundaries.

    For each break point interval, collect words whose start times fall within it.
    Then merge intervals that have too few words, and split those with too many.

    Args:
        words: flat list of {"text", "start", "end"}
        break_points: sorted candidate break times
        chords: chord list for chord-aware splitting
        min_words_per_line: merge lines shorter than this
        max_words_per_line: split lines longer than this

    Returns:
        List of {"text": str, "start_time": float, "end_time": float}
    """
    if not words or not break_points:
        return []

    # Build intervals from break points
    # Each interval is [bp[i], bp[i+1])
    intervals = []
    for i in range(len(break_points) - 1):
        intervals.append((break_points[i], break_points[i + 1]))
    # Add a final interval from last break point to end of last word + buffer
    if break_points:
        last_word_end = max(w['end'] for w in words)
        intervals.append((break_points[-1], last_word_end + 1.0))

    # Assign each word to the interval where its midpoint falls
    word_assignments = [[] for _ in intervals]
    for w in words:
        w_mid = (w['start'] + w['end']) / 2
        # Find the interval this word belongs to
        assigned = False
        for idx, (start, end) in enumerate(intervals):
            if start <= w_mid < end:
                word_assignments[idx].append(w)
                assigned = True
                break
        if not assigned:
            # Word is before first break point — assign to first interval
            if w_mid < intervals[0][0]:
                word_assignments[0].append(w)
            else:
                # After last interval
                word_assignments[-1].append(w)

    # Build raw lines from intervals that have words
    raw_lines = []
    for idx, assigned_words in enumerate(word_assignments):
        if not assigned_words:
            continue
        text = ' '.join(w['text'] for w in assigned_words)
        text = _clean_text(text)
        if text:
            raw_lines.append({
                'text': text,
                'start_time': round(assigned_words[0]['start'], 3),
                'end_time': round(assigned_words[-1]['end'], 3),
                '_words': assigned_words,
            })

    # Merge short lines with neighbors
    merged = _merge_beat_aligned_lines(raw_lines, min_words_per_line, chords)

    # Split overly long lines at chord changes
    final = _split_long_beat_lines(merged, max_words_per_line, chords)

    # Strip internal _words key
    for line in final:
        line.pop('_words', None)

    return final


def _merge_beat_aligned_lines(lines: List[Dict], min_words: int,
                              chords: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Merge beat-aligned lines that are too short (< min_words) with adjacent lines.
    Prefers merging with the previous line if same chord, otherwise next line.
    """
    if not lines:
        return lines

    def _chord_at_time(t: float) -> Optional[str]:
        if not chords:
            return None
        best = None
        for c in chords:
            if c['time'] <= t:
                best = c.get('chord')
        return best

    merged = [dict(lines[0])]
    for line in lines[1:]:
        word_count = len(line['text'].split())
        prev = merged[-1]
        prev_words = len(prev['text'].split())

        # If this line is very short, try to merge
        if word_count < min_words:
            # Check if merging wouldn't make it too long
            combined_words = prev_words + word_count
            if combined_words <= 12:
                # Merge with previous
                prev['text'] = _clean_text(prev['text'] + ' ' + line['text'])
                prev['end_time'] = line['end_time']
                if '_words' in prev and '_words' in line:
                    prev['_words'] = prev['_words'] + line['_words']
                continue

        # If previous line is also very short, merge it with this one
        if prev_words < min_words and prev_words + word_count <= 12:
            prev['text'] = _clean_text(prev['text'] + ' ' + line['text'])
            prev['end_time'] = line['end_time']
            if '_words' in prev and '_words' in line:
                prev['_words'] = prev['_words'] + line['_words']
            continue

        merged.append(dict(line))

    return merged


def _split_long_beat_lines(lines: List[Dict], max_words: int,
                           chords: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Split lines that are too long, preferring chord change points as split locations.
    """
    if not chords:
        return lines

    result = []
    sorted_chords = sorted(chords, key=lambda c: c.get('time', 0))

    for line in lines:
        line_words = line.get('_words', [])
        word_count = len(line['text'].split())

        if word_count <= max_words or not line_words:
            result.append(line)
            continue

        # Find chord changes within this line's time range
        l_start = line['start_time']
        l_end = line['end_time']
        chord_changes = [c['time'] for c in sorted_chords
                         if l_start < c['time'] < l_end]

        if not chord_changes:
            result.append(line)
            continue

        # Find the best split point: a chord change near the middle of the line
        mid_time = (l_start + l_end) / 2
        best_split_time = min(chord_changes, key=lambda t: abs(t - mid_time))

        # Split words at this point
        before = [w for w in line_words if w['start'] < best_split_time]
        after = [w for w in line_words if w['start'] >= best_split_time]

        if before and after:
            text_before = _clean_text(' '.join(w['text'] for w in before))
            text_after = _clean_text(' '.join(w['text'] for w in after))
            if text_before:
                result.append({
                    'text': text_before,
                    'start_time': round(before[0]['start'], 3),
                    'end_time': round(before[-1]['end'], 3),
                    '_words': before,
                })
            if text_after:
                result.append({
                    'text': text_after,
                    'start_time': round(after[0]['start'], 3),
                    'end_time': round(after[-1]['end'], 3),
                    '_words': after,
                })
        else:
            result.append(line)

    return result


def _split_into_lines(segments: list) -> List[Dict]:
    """
    Split Whisper segments into individual lyric lines.

    Whisper often returns paragraph-sized chunks. We split on:
    - Natural pauses between words (>1.0s gap)
    - Sentence boundaries (period, question mark, exclamation) with a pause
    - Maximum word count (~15 words per line)
    - Maximum duration (~6s per line)
    - Segment boundaries from Whisper itself
    """
    lines = []
    MAX_WORDS = 15
    PAUSE_THRESHOLD = 1.0
    MAX_LINE_DURATION = 6.0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        # If segment has word-level timestamps, use gaps to detect line breaks
        if segment.words:
            current_words = []
            current_start = None

            for i, word in enumerate(segment.words):
                word_text = word.word.strip()
                if not word_text:
                    continue

                if current_start is None:
                    current_start = word.start

                current_words.append(word)

                # Check for line break conditions
                is_last = (i == len(segment.words) - 1)
                should_break = is_last

                if not is_last:
                    next_word = segment.words[i + 1]
                    gap = next_word.start - word.end

                    # Natural pause
                    if gap > PAUSE_THRESHOLD:
                        should_break = True

                    # Sentence ending + pause
                    if _SENTENCE_END.search(word_text) and gap > 0.5:
                        should_break = True

                    # Too many words
                    if len(current_words) >= MAX_WORDS:
                        should_break = True

                    # Too long in duration
                    if word.end - current_start > MAX_LINE_DURATION:
                        should_break = True

                if should_break and current_words:
                    line_text = ' '.join(w.word.strip() for w in current_words)
                    line_text = _clean_text(line_text)
                    if line_text:
                        lines.append({
                            'text': line_text,
                            'start_time': round(current_start, 3),
                            'end_time': round(word.end, 3),
                        })
                    current_words = []
                    current_start = None
        else:
            # No word timestamps — use the whole segment as one line
            cleaned = _clean_text(text)
            if cleaned:
                lines.append({
                    'text': cleaned,
                    'start_time': round(segment.start, 3),
                    'end_time': round(segment.end, 3),
                })

    return lines


def _merge_short_lines(lines: List[Dict], min_words: int = 2, max_gap: float = 1.5) -> List[Dict]:
    """
    Merge very short lines (1 word) into adjacent lines if they're close together.
    Prevents single-word fragments from cluttering the chart.
    """
    if not lines:
        return lines

    merged = []
    i = 0
    while i < len(lines):
        current = dict(lines[i])
        word_count = len(current['text'].split())

        # If this line is very short and the next line is close, merge forward
        if word_count < min_words and i + 1 < len(lines):
            next_line = lines[i + 1]
            gap = next_line['start_time'] - current['end_time']
            if gap < max_gap:
                merged_text = current['text'] + ' ' + next_line['text']
                merged.append({
                    'text': _clean_text(merged_text),
                    'start_time': current['start_time'],
                    'end_time': next_line['end_time'],
                })
                i += 2
                continue

        merged.append(current)
        i += 1

    return merged


def extract_lyrics(vocals_path: str, output_dir: str = None,
                   audio_path: str = None,
                   chords: Optional[List[Dict]] = None) -> List[Dict]:
    """
    Extract timestamped lyrics from an isolated vocals stem.

    Args:
        vocals_path: Path to the isolated vocals audio file (mp3/wav/flac)
        output_dir: Optional directory to save lyrics.json
        audio_path: Optional path to the full mix audio (for beat detection).
                    When provided along with chords, enables beat-aligned line
                    splitting that aligns lyric lines with musical measures.
        chords: Optional list of chord dicts [{"chord": "Am", "time": 0.0}, ...]
                Used with audio_path for beat-aligned splitting.

    Returns:
        List of {"text": str, "start_time": float, "end_time": float}
    """
    if not os.path.exists(vocals_path):
        logger.error(f"Vocals file not found: {vocals_path}")
        return []

    logger.info(f"Extracting lyrics from {os.path.basename(vocals_path)}...")

    model = _get_model()

    # Transcribe with word timestamps for accurate line splitting
    segments, info = model.transcribe(
        vocals_path,
        word_timestamps=True,
        language="en",
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
    )

    # Collect all segments (it's a generator)
    all_segments = list(segments)
    logger.info(f"Whisper returned {len(all_segments)} segments, "
                f"language={info.language} (prob={info.language_probability:.2f})")

    if not all_segments:
        logger.warning("No speech segments detected in vocals")
        return []

    # Choose splitting strategy based on available data
    use_beat_aligned = (
        audio_path and os.path.exists(audio_path) and chords
    )

    if use_beat_aligned:
        logger.info("Using beat-aligned line splitting (audio_path + chords provided)")
        try:
            # Step 1: Get beat/measure positions from the full mix
            tempo, beat_times, measure_boundaries = _detect_beats(audio_path)

            # Step 2: Get chord change boundaries
            chord_boundaries = _find_chord_boundaries(chords)

            # Step 3: Build unified break points
            break_points = _find_best_break_points(measure_boundaries, chord_boundaries)

            # Step 4: Flatten all words from Whisper segments
            words = _collect_words_from_segments(all_segments)

            # Step 5: Split words into lines at break points
            lines = _split_lines_on_beats(words, break_points, chords)
            logger.info(f"Beat-aligned splitting produced {len(lines)} lines")

        except Exception as e:
            logger.warning(f"Beat-aligned splitting failed, falling back to pause-based: {e}")
            lines = _split_into_lines(all_segments)
            lines = _merge_short_lines(lines)
    else:
        # Fallback: original pause-based splitting
        lines = _split_into_lines(all_segments)
        logger.info(f"Split into {len(lines)} raw lyric lines")

        # Merge very short fragments
        lines = _merge_short_lines(lines)
        logger.info(f"After merging short lines: {len(lines)} lines")

    # Deduplicate consecutive repeats
    lines = _deduplicate_lines(lines)
    logger.info(f"After deduplication: {len(lines)} lines")

    # Filter out empty lines
    lines = [l for l in lines if l['text'].strip()]

    # Save if output_dir specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, 'lyrics.json')
        with open(out_path, 'w') as f:
            json.dump(lines, f, indent=2)
        logger.info(f"Saved lyrics to {out_path}")

    return lines


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    if len(sys.argv) < 2:
        print("Usage: python lyrics_extractor.py <vocals_path> [output_dir]")
        sys.exit(1)

    vocals = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else None
    result = extract_lyrics(vocals, outdir)

    print(f"\n{'='*60}")
    print(f"  {len(result)} lyric lines extracted")
    print(f"{'='*60}\n")
    for line in result:
        print(f"[{line['start_time']:6.2f} - {line['end_time']:6.2f}]  {line['text']}")
