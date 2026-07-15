"""
Chart-library correction — the "use the chords I gave it" integration.

When a processed song matches a chart in the OWNER's chart_library, snap the
detected chord events to that chart's chord vocabulary, transposed to the
recording (live tapes often run a half-step off the studio key). The library
chart is treated as ground truth: detector output like A#6/Gmin7/D#maj9 on a
song charted as A/G/Em becomes Bb/Ab/Fm when the tape runs +1 semitone.

PRIVACY: lookups are scoped to the job owner's user_id — same rule as
routes/charts.py. A user's charts never influence anyone else's jobs.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Pitch classes for both spellings
_PC = {
    'C': 0, 'B#': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
    'E': 4, 'Fb': 4, 'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6, 'G': 7,
    'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11, 'Cb': 11,
}
_SHARP_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
_FLAT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
# Major keys conventionally spelled with flats
_FLAT_KEYS = {5, 10, 3, 8, 1, 6}  # F, Bb, Eb, Ab, Db, Gb

_CHORD_RE = re.compile(
    r'^([A-G][#b]?)'                       # root
    r'((?:[Mm]aj|[Mm]in|m(?![Aa]j)|[Dd]im|[Aa]ug|[Ss]us|[Aa]dd|\d|[#b()+°ø/])*)$'  # suffix; detector emits mixed case (EmMaj7)
)

_MINOR_QUAL_RE = re.compile(r'^(m(?![Aa]j)|[Mm]in)')


def _parse_chord(token):
    """Return (root, suffix) or None. Handles slash chords by root only."""
    token = token.strip().rstrip(',')
    if not token or len(token) > 12:
        return None
    base = token.split('/')[0]
    m = _CHORD_RE.match(base)
    if not m or m.group(1) not in _PC:
        return None
    return m.group(1), m.group(2) or ''


def _parse_key(key_str):
    """'A', 'Bb', 'A# major', 'Em' -> (pitch_class, is_minor) or None."""
    if not key_str:
        return None
    s = key_str.strip()
    m = re.match(r'^([A-G][#b]?)\s*(m\b|min|minor)?', s)
    if not m or m.group(1) not in _PC:
        return None
    return _PC[m.group(1)], bool(m.group(2))


def _spell(pc, prefer_flats):
    return (_FLAT_NAMES if prefer_flats else _SHARP_NAMES)[pc % 12]


# Diatonic sharps per sharp major key: G has F#; D adds C#; A adds G#;
# E adds D#; B adds A#. Used so borrowed flat-side chords in non-flat keys
# spell as flats (Bb/Eb/Ab/Db in C) while genuine diatonic sharps survive
# (F# in G, G#m7 in E). pc 6 always spells F# outside the flat keys — the
# #4-of-C / 3-of-D case is far more common than Gb.
_KEY_SHARP_PCS = {
    7: {6},                # G
    2: {6, 1},             # D
    9: {6, 1, 8},          # A
    4: {6, 1, 8, 3},       # E
    11: {6, 1, 8, 3, 10},  # B
}


def _spell_in_key(pc, key_pc):
    """Spell a pitch class sensibly for the given major-key pitch class.
    Flat keys keep all-flats (unchanged behavior); other keys spell the
    borrowed black keys {Bb,Eb,Ab,Db} as flats unless that pc is a
    diatonic sharp of the key (Shakedown-in-C bug 7/5: A#/G#m7 rendered
    where a musician reads Bb/Abm7)."""
    pc %= 12
    if key_pc is None:
        return _SHARP_NAMES[pc]
    key_pc %= 12
    if key_pc in _FLAT_KEYS:
        return _FLAT_NAMES[pc]
    if pc in {10, 3, 8, 1} and pc not in _KEY_SHARP_PCS.get(key_pc, set()):
        return _FLAT_NAMES[pc]
    return _SHARP_NAMES[pc]


def _norm_title(s):
    return re.sub(r'[^a-z0-9]', '', (s or '').lower().replace('&', 'and'))


def _normalize_body(body):
    """Charts imported from old email/Word sources use <br/> line breaks and
    non-breaking spaces — normalize to plain newlines/spaces so chord-line
    detection works (Candyman id 37 case, 7/4)."""
    if not body:
        return ''
    s = body.replace('\xa0', ' ').replace('&nbsp;', ' ')
    s = re.sub(r'<br\s*/?>', '\n', s, flags=re.IGNORECASE)
    s = re.sub(r'<[^>\n]{1,20}>', '', s)  # strip any other stray short tags
    return s


def extract_chart_chords(body):
    """Pull the chord vocabulary out of a chords-over-lyrics chart body.

    A line is a chord line when every non-space token parses as a chord
    (and there is at least one). Parenthetical performance notes are skipped.
    """
    chords = []
    seen = set()
    for line in _normalize_body(body).splitlines():
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('('):
            continue
        tokens = line.split()
        parsed = [_parse_chord(t) for t in tokens]
        if parsed and all(p is not None for p in parsed):
            for root, suffix in parsed:
                label = root + suffix
                if label not in seen:
                    seen.add(label)
                    chords.append((root, suffix))
    return chords


def _relative_major_pc(key_tuple):
    """(pc, is_minor) -> pitch class of the relative major (Em -> G)."""
    pc, minor = key_tuple
    return (pc + 3) % 12 if minor else pc


# Krumhansl-Schmuckler major key profile
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]


def _measure_sounding_key(stem_paths):
    """Independently measure the recording's key (major pitch class) from a
    harmonic stem. The detector's own key call was wrong three times on
    7/4 alone (F-on-C, A-on-C) — chords and key can be *consistently*
    shifted together, so only a measurement from the audio can anchor the
    transposition. Returns pc int or None (non-fatal on any failure).
    """
    try:
        import os as _os
        import numpy as _np
        import librosa as _librosa
        paths = []
        for name in ('other', 'guitar', 'piano', 'vocals'):
            p = (stem_paths or {}).get(name)
            if p and _os.path.isfile(p):
                paths.append(p)
            if len(paths) == 2:
                break
        if not paths:
            return None
        chroma = _np.zeros(12)
        for p in paths:
            y, sr = _librosa.load(p, sr=22050, offset=60, duration=60, mono=True)
            if y is None or len(y) < sr * 5:
                y, sr = _librosa.load(p, sr=22050, duration=60, mono=True)
            chroma += _librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
        prof = _np.array(_KS_MAJOR)
        scores = [_np.corrcoef(_np.roll(prof, i), chroma)[0, 1] for i in range(12)]
        best = int(_np.argmax(scores))
        logger.info(f"Independent key measurement: {_SHARP_NAMES[best]} "
                    f"(r={scores[best]:.3f}, runner-up r={sorted(scores)[-2]:.3f})")
        return best
    except Exception as e:
        logger.warning(f"Independent key measurement failed (non-fatal): {e}")
        return None


def find_library_chart(title, artist, user_id):
    """Best chart_library match for (title, artist) owned by user_id.

    Returns dict {id, title, artist, song_key, body} or None.
    """
    if not user_id or not title:
        return None
    from db import query_all, query_one

    rows = query_all(
        "SELECT id, title, artist, song_key, flagged_for_review "
        "FROM chart_library WHERE user_id = %s",
        [str(user_id)],
    )
    want = _norm_title(title)
    want_artist = _norm_title(artist)
    candidates = []
    for r in rows:
        have = _norm_title(r['title'])
        if have == want or have == _norm_title('the ' + title) or want == _norm_title('the ' + (r['title'] or '')):
            candidates.append(r)
    if not candidates:
        # Substring fallback for messy source titles — YouTube gives
        # "Grateful Dead - Mississippi Half-Step Uptown Toodeloo (Without A
        # Net)"; the chart title lives inside it. Longest chart title wins;
        # minimum 8 normalized chars so "Fire" can't match everything.
        subs = [r for r in rows
                if len(_norm_title(r['title'])) >= 8 and _norm_title(r['title']) in want]
        if subs:
            subs.sort(key=lambda r: -len(_norm_title(r['title'])))
            best_len = len(_norm_title(subs[0]['title']))
            candidates = [r for r in subs if len(_norm_title(r['title'])) == best_len]
    if not candidates:
        return None

    def rank(r):
        artist_hit = want_artist and _norm_title(r['artist']) == want_artist
        return (
            0 if artist_hit else 1,
            0 if not r.get('flagged_for_review') else 1,
            0 if r.get('artist') else 1,
            r['id'],
        )

    best = sorted(candidates, key=rank)[0]
    full = query_one(
        "SELECT id, title, artist, song_key, body FROM chart_library WHERE id = %s",
        [best['id']],
    )
    return dict(full) if full else None


def _simplify_suffix(suffix, minor):
    """Off-vocabulary fallback: collapse jazz extensions to plain triads/7s."""
    if minor:
        return 'm'
    if suffix.startswith('sus'):
        return suffix[:4]
    if suffix.startswith('dim') or suffix.startswith('°'):
        return 'dim'
    if suffix.startswith('aug') or suffix.startswith('+'):
        return 'aug'
    if re.match(r'^(7|9|11|13)', suffix):
        return '7'
    return ''  # maj7/maj9/6/add9/plain -> plain major


def snap_progression(chord_events, library_chords, chart_key, detected_key, offset=None):
    """Rewrite chord events in place to the library vocabulary.

    offset (semitones chart->recording) is normally supplied by the caller
    from the independent key measurement; the detected_key fallback here
    compares relative majors so Am-vs-C doesn't read as a 3-semitone shift.
    Returns info dict, or None when keys can't be parsed.
    """
    ck = _parse_key(chart_key)
    dk = _parse_key(detected_key)
    if not ck or not library_chords:
        return None
    if offset is None:
        offset = ((_relative_major_pc(dk) - _relative_major_pc(ck)) % 12) if dk else 0

    display_key_pc = (ck[0] + offset) % 12
    # Spelling is anchored to the sounding MAJOR key (relative major for
    # minor charts) so borrowed chords read as flats where a player expects
    # them (Bb in C, not A#).
    spell_key_pc = (_relative_major_pc(ck) + offset) % 12
    display_key = _spell_in_key(display_key_pc, spell_key_pc) + ('m' if ck[1] else '')

    # Transposed vocabulary: pitch class -> list of (label, is_minor)
    vocab = {}
    for root, suffix in library_chords:
        pc = (_PC[root] + offset) % 12
        label = _spell_in_key(pc, spell_key_pc) + suffix
        vocab.setdefault(pc, []).append((label, bool(_MINOR_QUAL_RE.match(suffix))))

    snapped = 0
    respelled = 0
    for ev in chord_events:
        parsed = _parse_chord(ev.get('chord') or '')
        if not parsed:
            continue
        root, suffix = parsed
        pc = _PC[root] % 12
        minor = bool(_MINOR_QUAL_RE.match(suffix))
        options = vocab.get(pc)
        if options:
            # Prefer the library chord with matching major/minor flavor;
            # otherwise trust the library outright — it is the ground truth.
            match = next((l for l, m in options if m == minor), options[0][0])
            ev['chord'] = match
            new = _parse_chord(match)
            ev['root'], ev['quality'] = new[0], new[1]
            ev['library_snapped'] = True
            snapped += 1
        else:
            new_root = _spell_in_key(pc, spell_key_pc)
            new_suffix = _simplify_suffix(suffix, minor)
            ev['chord'] = new_root + new_suffix
            ev['root'], ev['quality'] = new_root, new_suffix
            ev['library_snapped'] = False
            respelled += 1

    return {
        'display_key': display_key,
        'semitones': offset if offset <= 6 else offset - 12,
        'snapped': snapped,
        'respelled': respelled,
    }


def _transpose_token(tok, offset, spell_key_pc):
    """Transpose one chord token, handling slash bass (A/C# -> Bb/D).
    spell_key_pc = sounding major-key pitch class driving flat/sharp
    spelling (key-aware since 7/5: Bb-not-A# in C)."""
    if offset == 0:
        return tok
    main, slash, bass = tok.partition('/')
    parsed = _parse_chord(main)
    if not parsed:
        return tok
    root, suffix = parsed
    out = _spell_in_key((_PC[root] + offset) % 12, spell_key_pc) + suffix
    if slash and bass in _PC:
        out += '/' + _spell_in_key((_PC[bass] + offset) % 12, spell_key_pc)
    return out


def is_degenerate_detection(chord_events, min_unique=3):
    """True when the detector heard so few distinct pitch classes the chart
    is useless (audience-tape mud collapsing to the tonic)."""
    pcs = set()
    for ev in chord_events or []:
        parsed = _parse_chord(ev.get('chord') or '')
        if parsed:
            pcs.add(_PC[parsed[0]] % 12)
    return len(pcs) < min_unique


def render_library_chart(chart, detected_key):
    """Render the user's stored chart body directly as a chord-chart dict
    (same JSON shape as chart_formatter.format_chart), transposed to the
    recording. No timing segments — used when detection is too broken to
    correct, so the user still gets THEIR chart instead of a one-chord grid.
    """
    ck = _parse_key(chart.get('song_key'))
    dk = _parse_key(detected_key)
    offset = ((dk[0] - ck[0]) % 12) if (ck and dk) else 0
    display_pc = ((ck[0] + offset) % 12) if ck else (dk[0] if dk else 0)
    # Key-aware spelling (7/5): anchor to the sounding relative-major key so
    # borrowed chords in C read Bb/Ab, while F# in G stays F#.
    spell_key_pc = ((_relative_major_pc(ck) + offset) % 12) if ck else display_pc
    display_key = _spell_in_key(display_pc, spell_key_pc) + ('m' if (ck and ck[1]) else '')

    sections = []
    current = {'name': 'Song', 'lines': []}
    pending_chords = None
    chords_used = []

    def flush_pending():
        nonlocal pending_chords
        if pending_chords is not None:
            current['lines'].append({'chords': pending_chords, 'lyrics': '', 'segments': []})
            pending_chords = None

    def transpose_line(line):
        out = []
        for tok in line.split():
            t = _transpose_token(tok, offset, spell_key_pc)
            out.append(t)
            if t not in chords_used:
                chords_used.append(t)
        return '  '.join(out)

    for raw in _normalize_body(chart.get('body')).splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        m = re.match(r'^\[(.+)\]$', stripped)
        if m:
            flush_pending()
            if current['lines']:
                sections.append(current)
            current = {'name': m.group(1).strip(), 'lines': []}
            continue
        if stripped.startswith('('):
            flush_pending()
            current['lines'].append({'chords': '', 'lyrics': stripped, 'segments': []})
            continue
        tokens = stripped.split()
        if tokens and all(_parse_chord(t) for t in tokens):
            flush_pending()
            pending_chords = transpose_line(stripped)
        else:
            current['lines'].append({
                'chords': pending_chords or '',
                'lyrics': stripped,
                'segments': [],
            })
            pending_chords = None
    flush_pending()
    if current['lines']:
        sections.append(current)

    # Chart convention: chords are written once ("Verse 1") and later verses
    # carry lyrics only. Players want the changes on EVERY verse (Jeff 7/4:
    # "add the chords to the whole song") — propagate each section family's
    # chord lines onto its bare siblings by line position.
    def _fam(name):
        return re.sub(r'\s*\d+\s*$', '', (name or '').strip().lower())

    templates = {}
    for s in sections:
        if any(l['chords'] for l in s['lines']) and _fam(s['name']) not in templates:
            templates[_fam(s['name'])] = [l['chords'] for l in s['lines']]
    for s in sections:
        if any(l['chords'] for l in s['lines']):
            continue
        t = templates.get(_fam(s['name']))
        if not t:
            continue
        for i, l in enumerate(s['lines']):
            if l['lyrics'] and i < len(t) and t[i]:
                l['chords'] = t[i]

    return {
        'title': chart.get('title') or 'Untitled',
        'artist': chart.get('artist') or '',
        'key': display_key,
        'capo': 0,
        'capo_shape_map': None,
        'source': 'Your chart library',
        'chords_used': chords_used,
        'sections': sections,
        'bar_grid': [],
        'grid': None,
    }


def apply_library_chart(job, title, artist):
    """Match + snap for a pipeline job. Returns info dict or None.

    Mutates job.chord_progression labels in place when a match is found.
    """
    user_id = getattr(job, 'user_id', None)
    chart = find_library_chart(title, artist, user_id)
    if not chart:
        return None
    library_chords = extract_chart_chords(chart.get('body'))
    if not library_chords:
        logger.info(f"Library chart {chart['id']} matched but no chords parsed from body")
        return None
    degenerate = is_degenerate_detection(job.chord_progression)

    # Anchor the chart->recording transposition to an independent key
    # measurement of the stems, NOT the detector's key call: the detector's
    # chords and key can be shifted TOGETHER (Half-Step 7/4 — heard the
    # whole tape in A, audio measures C), which no consistency check on its
    # own output can catch.
    ck0 = _parse_key(chart.get('song_key'))
    dk0 = _parse_key(getattr(job, 'detected_key', None))
    if not ck0:
        return None
    measured = _measure_sounding_key(getattr(job, 'stems', None))
    if measured is not None:
        offset = (measured - _relative_major_pc(ck0)) % 12
        key_source = 'measured'
    elif dk0:
        offset = (_relative_major_pc(dk0) - _relative_major_pc(ck0)) % 12
        key_source = 'detected'
    else:
        offset = 0
        key_source = 'none'

    # Coverage test: how much of the user's chart did the detector actually
    # hear? Me And My Uncle ('87 sbd) heard 3 pitch classes, missed the
    # chart's G and A and added an F# the song doesn't have — "correcting"
    # that timeline still yields a wrong chart. Below 60% coverage, treat
    # detection as unreliable and render the user's chart instead.
    chart_pcs = {(_PC[r] + offset) % 12 for r, s in library_chords}
    det_pcs = set()
    for ev in job.chord_progression or []:
        p = _parse_chord(ev.get('chord') or '')
        if p:
            det_pcs.add(_PC[p[0]] % 12)
    coverage = len(det_pcs & chart_pcs) / max(1, len(chart_pcs))
    if coverage < 0.6:
        degenerate = True

    info = snap_progression(
        job.chord_progression, library_chords,
        chart.get('song_key'), getattr(job, 'detected_key', None),
        offset=offset,
    )
    if not info:
        return None
    info['key_source'] = key_source
    info['coverage'] = round(coverage, 2)
    info.update({
        'chart_id': chart['id'],
        'title': chart['title'],
        'artist': chart.get('artist'),
        'chart_key': chart.get('song_key'),
        # Degenerate = detector heard <3 pitch classes (audience-tape mud) while
        # the library chart has a real progression — the pipeline should render
        # the user's chart outright instead of a one-chord grid.
        'degenerate': degenerate and len({r for r, s in library_chords}) >= 3,
    })
    return info


def apply_measured_key_no_library(job):
    """Public path: when NO library chart matched, the detector's key call is
    still unreliable (F-on-C and A-on-C both happened on 7/4-7/5) — measure
    the real key from the stems and respell the detected chords for it.
    Pitch classes never change; only names/spellings. Returns the corrected
    display key string, or None when measurement is unavailable.
    """
    measured = _measure_sounding_key(getattr(job, 'stems', None))
    if measured is None:
        return None
    dk = _parse_key(getattr(job, 'detected_key', None))
    minor = dk[1] if dk else False
    key_pc = (measured - 3) % 12 if minor else measured
    display = _spell_in_key(key_pc, measured) + ('m' if minor else '')

    for ev in job.chord_progression or []:
        label = ev.get('chord') or ''
        main, slash, bass = label.partition('/')
        parsed = _parse_chord(main)
        if not parsed:
            continue
        root, suffix = parsed
        new_root = _spell_in_key(_PC[root] % 12, measured)
        new_label = new_root + suffix
        if slash and bass in _PC:
            new_label += '/' + _spell_in_key(_PC[bass] % 12, measured)
        elif slash:
            new_label += '/' + bass
        if new_label != label:
            ev['chord'] = new_label
            ev['root'] = new_root
    return display


def render_override_for(library_info, detected_key=None):
    """When apply_library_chart flagged 'degenerate', build the full-chart
    replacement from the matched library chart. Separate call so the info
    dict stays backward-compatible.

    detected_key is deliberately IGNORED: a degenerate detection's key
    estimate is unreliable (Candyman '87 "detected" F on a recording
    independently measured in C). When apply_library_chart anchored the
    offset to a real measurement, render at that key; otherwise render the
    chart in its own key.
    """
    from db import query_one
    row = query_one(
        "SELECT id, title, artist, song_key, body FROM chart_library WHERE id = %s",
        [library_info['chart_id']],
    )
    if not row:
        return None
    target = library_info.get('display_key') if library_info.get('key_source') == 'measured' else None
    return render_library_chart(dict(row), target)
