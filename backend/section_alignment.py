"""
Section-label alignment (#36) — Phase 2 of the chart-library integration.

When a job matched a chart in the owner's chart_library, align the CHART's
section structure ([Verse]/[Chorus]/... with their lyric lines) to the
recording's timeline by fuzzy-matching each section's opening lyric words
against the Whisper word stream. Fixes two complaints (Jeff, 7/4):
  1. Section labels landing AFTER the section audibly starts.
  2. Sections the user's chart has (a Chorus) going missing entirely from
     the generated chart, because format_chart invents its own sections
     from lyric gaps.

Two integration paths, mirroring chart_library_matcher:
  - RENDER path (degenerate detection -> render_library_chart output):
    sections get a 'start' time; confidently-matched lyric lines get real
    'segments' timing so practice-mode highlighting tracks the audio.
  - SNAP path (good detection -> format_chart output): format_chart's lines
    (which carry real per-chord/per-bar segment timing) are REGROUPED under
    the library chart's section structure, with 4-bar slash-chunk lines
    split at section boundaries. Chords are never altered — chords_used is
    unchanged by alignment.

Matching is monotonic (each section must match after the previous one) so
repeated choruses resolve in order. Lyric-less sections (Solo/Intro/Outro)
are interpolated between their matched neighbors. Every chart section
survives — none are dropped, even when unmatched (flagged in the report).
"""

import logging
import re
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Section-anchor phrase length (words) and acceptance threshold.
_ANCHOR_LEN = 6
_SECTION_THRESHOLD = 0.60
# Per-line matching inside an already-matched section window.
_LINE_ANCHOR_LEN = 4
_LINE_THRESHOLD = 0.55


def _norm_tokens(text):
    """Lowercase word tokens, punctuation stripped ("Ridin'" -> ridin)."""
    return _TOKEN_RE.findall((text or '').lower().replace('’', "'"))


def _is_chordish(lyrics):
    """True when a 'lyrics' line is really a chord row that fell through the
    chord-line detector (render-path Solo rows like 'Am Am(maj7) D9 - F')."""
    from chart_library_matcher import _parse_chord
    toks = (lyrics or '').split()
    if not toks:
        return False
    hits = sum(1 for t in toks
               if _parse_chord(t) or t in ('-', '|', '/', '%')
               or re.fullmatch(r'[xX]\d+', t))
    return hits / len(toks) >= 0.7


def _build_stream(word_ts):
    """Flatten Whisper words into (token, start, end) tuples, normalized."""
    stream = []
    for w in word_ts or []:
        try:
            s, e = float(w.get('start', 0.0)), float(w.get('end', 0.0))
        except (TypeError, ValueError):
            continue
        for tok in _norm_tokens(w.get('word')):
            stream.append((tok, s, e))
    return stream


def _find_phrase(stream, anchor, lo, hi=None, threshold=0.6):
    """Best fuzzy match position of anchor tokens in stream[lo:hi].

    Score = difflib ratio of the joined strings (robust to split/merged
    tokens), with a small bonus when the first token matches exactly.
    Returns (index, score); index is None below threshold.
    """
    n = len(stream)
    if hi is None or hi > n:
        hi = n
    m = len(anchor)
    if m == 0 or lo >= hi:
        return None, 0.0
    a_join = ' '.join(anchor)
    a_set = set(anchor)
    # A sung phrase of m words can't plausibly span more than ~2.5s/word:
    # without this cap a window can bridge a 30s jam (Half-Step matched
    # "on my way ... when your ship" across the chorus-2 -> verse-3 gap).
    span_limit = max(15.0, 2.5 * m)
    best_i, best_s = None, 0.0
    for i in range(lo, hi):
        chunk = stream[i:i + m]
        window = [t for t, _, _ in chunk]
        if not window:
            break
        if chunk[-1][2] - chunk[0][1] > span_limit:
            continue
        # Cheap prefilter: skip windows sharing almost no vocabulary.
        if len(a_set & set(window)) < max(1, len(a_set) // 3):
            continue
        s = SequenceMatcher(None, a_join, ' '.join(window)).ratio()
        if window[0] == anchor[0]:
            s = min(1.0, s + 0.05)
        if s > best_s:
            best_s, best_i = s, i
    if best_i is not None and best_s >= threshold:
        return best_i, best_s
    return None, best_s


def _section_lyric_lines(sec):
    """Real lyric lines of a section — skips parenthetical performance notes
    and chord rows masquerading as lyrics."""
    out = []
    for ln in sec.get('lines') or []:
        ly = (ln.get('lyrics') or '').strip()
        if not ly or ly.startswith('('):
            continue
        if _is_chordish(ly):
            continue
        if not _norm_tokens(ly):
            continue
        out.append(ln)
    return out


def _section_anchor(sec, max_words=_ANCHOR_LEN, skip_lines=0):
    """First ~6 lyric words of a section (optionally skipping leading lines,
    for the second-line fallback when Whisper garbled the opener)."""
    toks = []
    for ln in _section_lyric_lines(sec)[skip_lines:]:
        toks.extend(_norm_tokens(ln.get('lyrics')))
        if len(toks) >= max_words:
            break
    return toks[:max_words]


def _section_token_count(sec):
    return sum(len(_norm_tokens(ln.get('lyrics'))) for ln in _section_lyric_lines(sec))


def _bar_seconds(grid):
    try:
        bpm = float((grid or {}).get('tempo_bpm') or 0)
        if bpm > 0:
            ts = str((grid or {}).get('time_signature') or '4/4')
            beats = int(ts.split('/')[0]) if '/' in ts else 4
            return max(0.5, min(6.0, beats * 60.0 / bpm))
    except (TypeError, ValueError):
        pass
    return None


def _pickup_seconds(grid):
    """Pickup allowance: labels land ~1 bar BEFORE the first matched lyric
    word, so instrumental section tops and sung pickups are covered."""
    b = _bar_seconds(grid)
    return min(3.0, max(1.0, b)) if b else 2.0


def compute_section_times(sections, stream, grid=None):
    """Monotonically match each section's anchor phrase in the word stream.

    Returns a list (same order/length as sections) of report dicts:
      {name, start, method, score, matched_words}
    plus private keys (_idx, _t, _cut) used by the appliers.
    method: 'matched' | 'matched(line2)' | 'interpolated' | 'appended'
    """
    pickup = _pickup_seconds(grid)
    bar = _bar_seconds(grid) or 2.0
    res = []
    cursor = 0
    for sec in sections:
        entry = {'name': sec.get('name'), 'start': None, 'method': 'interpolated',
                 'score': 0.0, 'matched_words': None, '_idx': None, '_t': None}
        anchor = _section_anchor(sec)
        if anchor and stream:
            idx, score = _find_phrase(stream, anchor, cursor,
                                      threshold=_SECTION_THRESHOLD)
            method = 'matched'
            offset_back = 0.0
            if idx is None:
                # Whisper may have garbled the section's first line — try the
                # second lyric line and back off ~2 bars for the missed one.
                alt = _section_anchor(sec, skip_lines=1)
                if alt:
                    idx, score = _find_phrase(stream, alt, cursor,
                                              threshold=_SECTION_THRESHOLD)
                    if idx is not None:
                        method = 'matched(line2)'
                        offset_back = 2.0 * bar
                        anchor = alt
            if idx is not None:
                entry.update(method=method, score=round(score, 3), _idx=idx)
                entry['matched_words'] = ' '.join(
                    t for t, _, _ in stream[idx:idx + len(anchor)])
                entry['_t'] = max(0.0, stream[idx][1] - offset_back)
                cursor = idx + len(anchor)
        res.append(entry)

    _finalize_starts(res, sections, stream, pickup)
    return res


def _finalize_starts(res, sections, stream, pickup):
    """Fill 'start' for every section: matched ones get first-word-minus-
    pickup; unmatched ones interpolate between neighbors (Solo case) or
    append after the last match. Enforces strictly-increasing starts and
    computes '_cut' regroup boundaries at the raw match times."""
    n = len(res)
    matched_is = [i for i, r in enumerate(res) if r['_t'] is not None]

    def end_est(i):
        """Estimated vocal end of matched section i: the timestamp where its
        lyric token count runs out in the stream (clamped before the next
        matched section)."""
        r = res[i]
        count = _section_token_count(sections[i])
        j = r['_idx'] + max(1, count)
        for k in matched_is:
            if k > i:
                j = min(j, res[k]['_idx'] - 1)
                break
        j = max(0, min(j, len(stream) - 1))
        return stream[j][2]

    for i in matched_is:
        res[i]['start'] = max(0.0, res[i]['_t'] - pickup)

    if matched_is:
        # Leading unmatched sections (Intro...) spread across [0, first match)
        first = matched_is[0]
        lead = list(range(first))
        for k, i in enumerate(lead):
            res[i]['start'] = res[first]['start'] * k / len(lead)
        # Gaps between matched sections (the interpolated-Solo case): start
        # at the previous section's estimated vocal end, spread evenly.
        for a, b in zip(matched_is, matched_is[1:]):
            gap = list(range(a + 1, b))
            if not gap:
                continue
            lo = max(res[a]['start'] + 1.0,
                     min(end_est(a) + 0.5, res[b]['start'] - 1.0))
            hi = res[b]['start']
            step = max(0.5, (hi - lo) / len(gap))
            for k, i in enumerate(gap):
                res[i]['start'] = lo + step * k
        # Trailing unmatched sections append after the last match's vocals.
        last = matched_is[-1]
        t0 = end_est(last)
        for k, i in enumerate(range(last + 1, n)):
            res[i]['start'] = t0 + 1.0 + 8.0 * k
            res[i]['method'] = 'appended'
    else:
        for k, r in enumerate(res):
            r['start'] = 8.0 * k

    prev = -0.5
    for r in res:
        if r['start'] is None or r['start'] <= prev:
            r['start'] = prev + 0.5
        r['start'] = round(r['start'], 2)
        prev = r['start']

    # Regroup cut boundaries: just before the raw matched word (NOT the
    # pickup-padded display start, which would steal the previous section's
    # last phrase).
    prev_cut = 0.0
    for r in res:
        cut = (r['_t'] - 0.75) if r['_t'] is not None else r['start']
        cut = max(prev_cut + 0.25, cut)
        r['_cut'] = round(cut, 3)
        prev_cut = cut


def _words_between(word_ts, t0, t1):
    return [{'w': w.get('word', ''), 't': round(float(w.get('start', 0.0)), 3)}
            for w in word_ts
            if t0 <= float(w.get('start', 0.0)) < t1]


def _public_report(res):
    return [{'name': r['name'], 'start': r['start'], 'method': r['method'],
             'score': r['score'], 'matched_words': r['matched_words']}
            for r in res]


# ---------------------------------------------------------------------------
# RENDER path — chart IS the library chart (source: 'Your chart library')
# ---------------------------------------------------------------------------

def align_rendered_chart(chart, word_ts, grid=None):
    """Mutates a render_library_chart output: sections gain 'start', and
    confidently-matched lyric lines gain real 'segments' timing (per-chord
    even split across the line window — approximate but monotonic)."""
    from chart_library_matcher import _parse_chord

    sections = chart.get('sections') or []
    stream = _build_stream(word_ts)
    res = compute_section_times(sections, stream, grid=grid)

    lyric_secs = [s for s in sections if _section_anchor(s)]
    matched = [r for r in res if r['_t'] is not None]
    if not stream or len(matched) < max(1, round(0.5 * len(lyric_secs))):
        return {'sections_aligned': False,
                'reason': f'matched {len(matched)}/{len(lyric_secs)} lyric sections',
                'sections': _public_report(res)}

    bar = _bar_seconds(grid) or 2.0
    matched_idx = [i for i, r in enumerate(res) if r['_t'] is not None]
    for pos, i in enumerate(matched_idx):
        lo = res[i]['_idx']
        hi = res[matched_idx[pos + 1]]['_idx'] if pos + 1 < len(matched_idx) else len(stream)
        seed_line = 1 if res[i]['method'] == 'matched(line2)' else 0
        _time_section_lines(sections[i], stream, word_ts, lo, hi, bar,
                            _parse_chord, seed_line=seed_line)

    for sec, r in zip(sections, res):
        sec['start'] = r['start']

    return {'sections_aligned': True, 'sections': _public_report(res)}


def _time_section_lines(sec, stream, word_ts, lo, hi, bar, parse_chord,
                        seed_line=0):
    """Match each lyric line of a matched section inside its stream window;
    matched lines get segments: chord tokens evenly split across the line's
    sung window, with the Whisper words that fall inside each slice.

    seed_line: the section anchor was BUILT from this lyric line, so that
    line's position is inherited from the section match directly (a re-search
    can run off to a later verse-reprise when Whisper garbled the opener —
    Me And My Uncle heard "me and my auger" at 0:03 and the line matcher
    jumped to the verse repeat at 2:53). A proximity guard stops later lines
    from doing the same jump: a match landing far past the previous timed
    line's end is distrusted and the line stays untimed."""
    lyric_lines = _section_lyric_lines(sec)
    seed_target = lyric_lines[seed_line] if seed_line < len(lyric_lines) else None
    max_gap = max(30.0, 8.0 * bar)
    cursor = lo
    prev_end_t = None
    timed = []  # (line, start_time, token_count, stream_idx)
    for ln in sec.get('lines') or []:
        ly = (ln.get('lyrics') or '').strip()
        toks = _norm_tokens(ly)
        if not toks or _is_chordish(ly):
            continue
        if ln is seed_target:
            idx = min(lo, len(stream) - 1)
        else:
            anchor = toks[:_LINE_ANCHOR_LEN]
            idx, _score = _find_phrase(stream, anchor, cursor, hi,
                                       threshold=_LINE_THRESHOLD)
            if (idx is not None and prev_end_t is not None
                    and stream[idx][1] > prev_end_t + max_gap):
                idx = None
        if idx is None:
            continue
        timed.append((ln, stream[idx][1], len(toks), idx))
        cursor = idx + max(1, min(len(toks), _LINE_ANCHOR_LEN) - 1)
        j = min(idx + max(1, len(toks)) - 1, len(stream) - 1)
        prev_end_t = stream[j][2]

    for k, (ln, t_start, tok_count, idx) in enumerate(timed):
        j = min(idx + max(1, tok_count) - 1, len(stream) - 1)
        sung_end = stream[j][2]
        if k + 1 < len(timed):
            line_end = timed[k + 1][1]
        else:
            line_end = sung_end + 0.5
        line_end = max(line_end, t_start + 0.5)

        chord_toks = [t for t in (ln.get('chords') or '').split() if parse_chord(t)]
        if not chord_toks:
            continue
        span = (line_end - t_start) / len(chord_toks)
        segments = []
        for ci, ch in enumerate(chord_toks):
            s0 = t_start + ci * span
            s1 = t_start + (ci + 1) * span
            segments.append({
                'chord': ch,
                'start': round(s0, 3),
                'end': round(s1, 3),
                'duration': round(s1 - s0, 3),
                'bars': max(1, min(8, round(span / bar))),
                'words': _words_between(word_ts, s0, s1),
            })
        ln['segments'] = segments


# ---------------------------------------------------------------------------
# SNAP path — format_chart output regrouped under the library structure
# ---------------------------------------------------------------------------

def align_snapped_chart(chart, library_info, word_ts, grid=None):
    """Rebuild format_chart's sections from the library chart's structure:
    the chart's section names/order with Whisper-aligned boundaries, the
    formatter's own timed lines regrouped (and slash-chunk lines split)
    into those windows. Chords and chords_used are untouched."""
    from chart_library_matcher import render_library_chart
    from db import query_one

    row = query_one(
        "SELECT id, title, artist, song_key, body FROM chart_library WHERE id = %s",
        [library_info['chart_id']],
    )
    if not row:
        return {'sections_aligned': False, 'reason': 'library chart row missing'}
    struct = render_library_chart(dict(row), None)
    lib_secs = struct.get('sections') or []
    if not lib_secs:
        return {'sections_aligned': False, 'reason': 'library chart has no sections'}

    stream = _build_stream(word_ts)
    res = compute_section_times(lib_secs, stream, grid=grid)
    lyric_secs = [s for s in lib_secs if _section_anchor(s)]
    matched = [r for r in res if r['_t'] is not None]
    # Higher bar than the render path: we are about to REPLACE the
    # formatter's section boundaries, so demand a solid structural lock.
    if len(matched) < 2 or len(matched) < round(0.6 * len(lyric_secs)):
        return {'sections_aligned': False,
                'reason': f'matched {len(matched)}/{len(lyric_secs)} lyric sections',
                'sections': _public_report(res)}

    # Flatten the formatter's lines in chart order with their sung windows.
    flat = []
    for sec in chart.get('sections') or []:
        for ln in sec.get('lines') or []:
            segs = ln.get('segments') or []
            st = segs[0].get('start') if segs else None
            en = segs[-1].get('end') if segs else None
            flat.append({'line': ln, 'start': st, 'end': en})
    if not flat:
        return {'sections_aligned': False, 'reason': 'chart has no lines'}
    # Untimed lines inherit a neighbor's time so ordering survives.
    last = 0.0
    for f in flat:
        if f['start'] is None:
            f['start'] = last
        if f['end'] is None or f['end'] < f['start']:
            f['end'] = f['start']
        last = f['end']

    cuts = [r['_cut'] for r in res]
    flat = _split_chunk_lines_at_cuts(flat, cuts)

    buckets = [[] for _ in lib_secs]
    leading = []
    for f in flat:
        mid = 0.5 * (f['start'] + f['end'])
        k = None
        for i in range(len(cuts) - 1, -1, -1):
            if mid >= cuts[i]:
                k = i
                break
        if k is None:
            leading.append(f['line'])
        else:
            buckets[k].append(f['line'])

    new_sections = []
    if leading:
        first_name = (lib_secs[0].get('name') or '').lower()
        if 'intro' in first_name or cuts[0] <= 4.0:
            buckets[0] = leading + buckets[0]
        else:
            new_sections.append({'name': 'Intro', 'start': 0.0, 'lines': leading})

    report = _public_report(res)
    for i, sec in enumerate(lib_secs):
        entry = {'name': sec.get('name'), 'start': res[i]['start'],
                 'lines': buckets[i]}
        if not buckets[i]:
            report[i]['empty'] = True
        new_sections.append(entry)

    chart['sections'] = new_sections
    return {'sections_aligned': True, 'sections': report}


def _split_chunk_lines_at_cuts(flat, cuts):
    """Split 4-bar slash-notation chunk lines (per-bar segments) whose span
    crosses a section cut, so boundaries land on the nearest bar (~2s) rather
    than mid-chunk (~8s). UG-style vocal lines (chords column-aligned to the
    lyric text) are never split — they move whole by midpoint."""
    try:
        from chart_formatter import _build_slash_chord_line
    except Exception:
        return flat

    out = []
    for f in flat:
        ln = f['line']
        segs = ln.get('segments') or []
        chords = ln.get('chords') or ''
        splittable = ('////' in chords and len(segs) >= 2
                      and all(isinstance(s.get('start'), (int, float)) for s in segs))
        if not splittable:
            out.append(f)
            continue
        inner = [c for c in cuts if f['start'] + 0.5 < c < f['end'] - 0.5]
        if not inner:
            out.append(f)
            continue
        pieces = []
        remaining = list(segs)
        for c in inner:
            k = min(range(len(remaining)),
                    key=lambda j: abs(remaining[j]['start'] - c))
            if 0 < k < len(remaining):
                pieces.append(remaining[:k])
                remaining = remaining[k:]
        pieces.append(remaining)
        pieces = [p for p in pieces if p]
        if len(pieces) < 2:
            out.append(f)
            continue
        for p in pieces:
            words = [w for s in p for w in (s.get('words') or [])]
            lyrics = ' '.join((w.get('w') or '').strip() for w in words).strip()
            new_line = {
                'chords': _build_slash_chord_line([s.get('chord') for s in p]),
                'lyrics': lyrics if lyrics else None,
                'segments': p,
            }
            out.append({'line': new_line,
                        'start': p[0]['start'], 'end': p[-1]['end']})
    return out


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def align_chart_with_library(chart, library_info, word_ts, grid=None):
    """Align a library-matched chart's sections to the recording. Mutates
    `chart` (section starts / regrouped sections / line segments) and
    `library_info` (sections_aligned + section_alignment report). Never
    raises — alignment is strictly best-effort."""
    try:
        if not (chart and chart.get('sections')):
            info = {'sections_aligned': False, 'reason': 'no chart sections'}
        elif not word_ts:
            info = {'sections_aligned': False, 'reason': 'no word timestamps'}
        elif chart.get('source') == 'Your chart library':
            info = align_rendered_chart(chart, word_ts, grid)
        else:
            info = align_snapped_chart(chart, library_info, word_ts, grid)
    except Exception as e:
        logger.warning(f"Section alignment failed (non-fatal): {e}", exc_info=True)
        info = {'sections_aligned': False, 'reason': f'error: {e}'}

    if isinstance(library_info, dict):
        library_info['sections_aligned'] = bool(info.get('sections_aligned'))
        if info.get('sections'):
            library_info['section_alignment'] = info['sections']
        if not info.get('sections_aligned') and info.get('reason'):
            library_info['section_alignment_reason'] = info['reason']
    if info.get('sections_aligned'):
        n_matched = sum(1 for s in info.get('sections', [])
                        if s['method'].startswith('matched'))
        logger.info(f"✓ Section alignment: {n_matched}/{len(info.get('sections', []))} "
                    f"sections matched to the recording")
    else:
        logger.info(f"Section alignment skipped: {info.get('reason')}")
    return info
