"""
Songsterr integration — search for professional tabs and serve them as GP5 files.
"""

import io
import logging
import re as _re
import urllib.parse
import requests
from flask import Blueprint, request, jsonify, send_file

logger = logging.getLogger(__name__)

songsterr_bp = Blueprint("songsterr", __name__)

SONGSTERR_API = "https://www.songsterr.com/api"
SONGSTERR_CDN = "https://dqsljvtekg760.cloudfront.net"
CHORDPRO_CDN = "https://chordpro1.songsterr.com"


@songsterr_bp.route('/api/songsterr/search', methods=['GET'])
def songsterr_search():
    """Search Songsterr for tabs matching a query."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'No search query provided'}), 400

    try:
        encoded = urllib.parse.quote(query)
        resp = requests.get(
            f"{SONGSTERR_API}/songs?size=8&pattern={encoded}",
            timeout=10
        )
        resp.raise_for_status()
        results = resp.json()

        # Normalize to list (API sometimes returns single object)
        if isinstance(results, dict):
            results = [results]

        songs = []
        for item in results:
            tracks = []
            for t in item.get('tracks', []):
                tracks.append({
                    'instrument': t.get('instrument', ''),
                    'name': t.get('name', ''),
                    'difficulty': t.get('difficulty', 0),
                    'views': t.get('views', 0),
                    'tuning': t.get('tuning', []),
                })
            songs.append({
                'songId': item.get('songId'),
                'artist': item.get('artist', ''),
                'title': item.get('title', ''),
                'hasChords': item.get('hasChords', False),
                'tracks': tracks,
            })

        return jsonify({'results': songs})

    except requests.RequestException as e:
        logger.error(f"Songsterr search failed: {e}")
        return jsonify({'error': 'Search failed', 'details': str(e)}), 502


@songsterr_bp.route('/api/songsterr/tabs/<int:song_id>', methods=['GET'])
def songsterr_tabs(song_id):
    """Fetch Songsterr tabs and return as a GP5 file."""
    track_index = request.args.get('track', None)

    try:
        # 1. Get song metadata (includes revisionId and image hash for CDN)
        meta_resp = requests.get(
            f"{SONGSTERR_API}/meta/{song_id}",
            timeout=10
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        revision_id = meta.get('revisionId')
        image_hash = meta.get('image')
        tracks_meta = meta.get('tracks', [])

        if not revision_id or not image_hash:
            return jsonify({'error': 'Missing revision data'}), 404

        # 2. Fetch tab data for requested tracks
        track_indices = range(len(tracks_meta))
        if track_index is not None:
            track_indices = [int(track_index)]

        track_data = []
        for idx in track_indices:
            url = f"{SONGSTERR_CDN}/{song_id}/{revision_id}/{image_hash}/{idx}.json"
            resp = requests.get(url, headers={
                'Accept-Encoding': 'gzip, deflate',
            }, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            # Merge track metadata
            if idx < len(tracks_meta):
                data['_meta'] = tracks_meta[idx]
            track_data.append(data)

        # 3. Get tempo from first track's automations
        tempo = 120
        if track_data:
            automations = track_data[0].get('automations', {})
            tempo_list = automations.get('tempo', [])
            if tempo_list and isinstance(tempo_list, list):
                tempo = tempo_list[0].get('bpm', 120)

        # 4. Return JSON for HTML tab renderer (default) or GP5 for legacy AlphaTab
        fmt = request.args.get('format', 'json')
        if fmt == 'json':
            return jsonify({
                'title': meta.get('title', 'Unknown'),
                'artist': meta.get('artist', 'Unknown'),
                'tempo': tempo,
                'tracks': track_data,
                'tracks_meta': tracks_meta,
            })

        # GP5 fallback
        from songsterr_to_gp import songsterr_to_gp5
        gp5_bytes = songsterr_to_gp5(
            track_data=track_data,
            title=meta.get('title', 'Unknown'),
            artist=meta.get('artist', 'Unknown'),
            tempo=tempo,
        )
        return send_file(
            io.BytesIO(gp5_bytes),
            mimetype='application/x-guitar-pro',
            as_attachment=True,
            download_name=f"{meta.get('artist', 'tab')} - {meta.get('title', 'tab')}.gp5"
        )

    except requests.RequestException as e:
        logger.error(f"Songsterr tab fetch failed for song {song_id}: {e}")
        return jsonify({'error': 'Failed to fetch tabs', 'details': str(e)}), 502
    except Exception as e:
        logger.error(f"Songsterr tab fetch failed for song {song_id}: {e}", exc_info=True)
        return jsonify({'error': 'Tab conversion failed', 'details': str(e)}), 500


@songsterr_bp.route('/api/songsterr/meta/<int:song_id>', methods=['GET'])
def songsterr_meta(song_id):
    """Get Songsterr song metadata (track list, etc.)."""
    try:
        resp = requests.get(f"{SONGSTERR_API}/meta/{song_id}", timeout=10)
        resp.raise_for_status()
        meta = resp.json()

        tracks = []
        for i, t in enumerate(meta.get('tracks', [])):
            tracks.append({
                'index': i,
                'instrument': t.get('instrument', ''),
                'name': t.get('name', ''),
                'difficulty': t.get('difficulty', 0),
                'views': t.get('views', 0),
                'tuning': t.get('tuning', []),
            })

        return jsonify({
            'songId': meta.get('songId'),
            'artist': meta.get('artist', ''),
            'title': meta.get('title', ''),
            'revisionId': meta.get('revisionId'),
            'hasChords': meta.get('hasChords', False),
            'tracks': tracks,
        })

    except requests.RequestException as e:
        logger.error(f"Songsterr meta fetch failed: {e}")
        return jsonify({'error': 'Metadata fetch failed'}), 502


# ── Chord extraction from tab data ──

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
DEFAULT_TUNING = [64, 59, 55, 50, 45, 40]  # standard guitar MIDI


def _frets_to_notes(frets, tuning):
    """Convert (string_index, fret) pairs to sorted (midi, note_name) list.

    Args:
        frets: list of (string_index, fret) where string_index is 0-based
               (0 = string 1 / high E in standard, 5 = string 6 / low E).
        tuning: list of MIDI note numbers, index 0 = string 1, index 5 = string 6.
                Default standard tuning: [64, 59, 55, 50, 45, 40].

    Returns:
        List of (midi_number, note_name) sorted by pitch (lowest first).
    """
    result = []
    for string_idx, fret in frets:
        string_idx = int(string_idx)
        fret = int(fret)
        if 0 <= string_idx < len(tuning):
            midi = int(tuning[string_idx]) + fret
            result.append((midi, NOTE_NAMES[midi % 12]))
    result.sort()  # lowest pitch first
    return result


def _notes_from_tab(tab_str, tuning=None):
    """Convert a tab fret string to note names, one per string (6 to 1).

    Args:
        tab_str: Fret numbers as a string, one char per string from string 6
                 (low E) to string 1 (high E). Use 'x' or 'X' for muted.
                 Examples: "x02220" (A major), "022100" (E major).
                 For frets >= 10, use a list of ints/strings instead.
        tuning:  List of 6 MIDI note numbers [string6, string5, ..., string1]
                 (low to high, i.e. the musician's perspective).
                 If None, uses standard tuning E-A-D-G-B-E = [40, 45, 50, 55, 59, 64].

    Returns:
        List of note name strings from string 6 to string 1.
        Muted strings are represented as 'x'.
    """
    # Standard tuning in low-to-high order (musician perspective)
    if tuning is None:
        tuning_low_to_high = [40, 45, 50, 55, 59, 64]
    else:
        tuning_low_to_high = list(tuning)

    num_strings = len(tuning_low_to_high)

    # Parse fret data
    if isinstance(tab_str, str):
        fret_chars = list(tab_str)
    else:
        fret_chars = [str(f) for f in tab_str]

    if len(fret_chars) != num_strings:
        raise ValueError(
            f"Tab has {len(fret_chars)} entries but tuning has {num_strings} strings"
        )

    result = []
    for i, fret_val in enumerate(fret_chars):
        if fret_val.lower() == 'x':
            result.append('x')
        else:
            fret = int(fret_val)
            midi = tuning_low_to_high[i] + fret
            result.append(NOTE_NAMES[midi % 12])

    return result


def _identify_chord(midi_notes):
    """Identify chord name from [(midi, note_name), ...] sorted by pitch."""
    if len(midi_notes) < 2:
        return None

    note_names = list(dict.fromkeys(n for _, n in midi_notes))  # unique, pitch-ordered
    bass = note_names[0]

    # Try each note as root, but score bass note higher
    candidates = []
    for root in note_names:
        root_midi = NOTE_NAMES.index(root)
        intervals = set((NOTE_NAMES.index(n) - root_midi) % 12 for n in note_names)

        name = None
        priority = 0  # higher = better; complex chords score higher

        # --- Complex chords first (most intervals → highest priority) ---
        has_fifth = 7 in intervals

        # 7#9 / Hendrix chord: root + major 3rd + minor 7th + #9 (= minor 3rd)
        # 5th often omitted in this voicing
        if {0, 3, 4, 10} <= intervals:
            name = root + '7#9'
            priority = 7 if has_fifth else 6

        # 9th: dominant 7th + major 9th (interval 2)
        elif {0, 2, 4, 7, 10} <= intervals:
            name = root + '9'
            priority = 6

        # 7sus4: sus4 + minor 7th
        elif {0, 5, 7, 10} <= intervals:
            name = root + '7sus4'
            priority = 5

        # m6: minor triad + major 6th (interval 9)
        elif {0, 3, 7, 9} <= intervals:
            name = root + 'm6'
            priority = 5

        # 6th: major triad + major 6th (interval 9)
        elif {0, 4, 7, 9} <= intervals:
            name = root + '6'
            priority = 5

        # add9: major triad + major 9th (interval 2), NO 7th
        elif {0, 2, 4, 7} <= intervals and 10 not in intervals and 11 not in intervals:
            name = root + 'add9'
            priority = 5

        # Major with extensions (5th may be omitted in 7th chords)
        elif {0, 4, 7} <= intervals or ({0, 4} <= intervals and (10 in intervals or 11 in intervals)):
            if 10 in intervals:
                name = root + '7'
                priority = 4
            elif 11 in intervals:
                name = root + 'maj7'
                priority = 4
            elif has_fifth:
                name = root
                priority = 3

        # Minor with extensions (5th may be omitted in m7 chords)
        elif {0, 3, 7} <= intervals or ({0, 3} <= intervals and 10 in intervals):
            if 10 in intervals:
                name = root + 'm7'
                priority = 4
            elif has_fifth:
                name = root + 'm'
                priority = 3

        # Sus4
        elif {0, 5, 7} <= intervals:
            name = root + 'sus4'
            priority = 2
        # Sus2
        elif {0, 2, 7} <= intervals:
            name = root + 'sus2'
            priority = 2
        # Diminished
        elif {0, 3, 6} <= intervals:
            name = root + 'dim'
            priority = 2
        # Augmented
        elif {0, 4, 8} <= intervals:
            name = root + 'aug'
            priority = 2
        # Power chord
        elif intervals == {0, 7}:
            name = root + '5'
            priority = 1

        if name:
            # Bonus priority if root == bass note
            if root == bass:
                priority += 10
            candidates.append((priority, name, root))

    if not candidates:
        return None

    # Pick highest priority (bass-rooted chords win)
    candidates.sort(key=lambda x: -x[0])
    best_name = candidates[0][1]
    best_root = candidates[0][2]

    # If bass note isn't the root, show as slash chord
    if best_root != bass and candidates[0][0] < 10:
        best_name += '/' + bass

    return best_name


def _extract_chords_from_tracks(all_track_data):
    """Extract chord progression from Songsterr track data.

    Strategy:
    1. Try native chord annotations (beat.chord.text) — most accurate
    2. Fall back to identifying chords from the actual fret/note data
    """
    if not all_track_data:
        return []

    max_measures = max(len(t.get('measures', [])) for t in all_track_data)
    result = []
    has_native_chords = False

    # Pass 1: try native chord annotations
    for mi in range(max_measures):
        measure_chords = []
        sig = [4, 4]

        for tdata in all_track_data:
            measures = tdata.get('measures', [])
            if mi >= len(measures):
                continue

            m = measures[mi]
            s = m.get('signature')
            if s and len(s) == 2:
                sig = s

            for v in m.get('voices', []):
                for b in v.get('beats', []):
                    chord_data = b.get('chord')
                    if chord_data and chord_data.get('text'):
                        chord_name = chord_data['text'].strip()
                        if chord_name and (not measure_chords or measure_chords[-1] != chord_name):
                            measure_chords.append(chord_name)
                            has_native_chords = True

        result.append({
            'measure': mi + 1,
            'signature': sig,
            'chords': measure_chords,
        })

    if has_native_chords:
        return result

    # Pass 2: no native annotations — identify chords from notes
    # Pick the guitar track with the MOST chord beats (2+ simultaneous notes)
    best_track = None
    best_chord_count = 0
    for tdata in all_track_data:
        meta = tdata.get('_meta', {})
        inst = (meta.get('instrument') or meta.get('name') or '').lower()
        if 'drum' in inst or 'perc' in inst or 'bass' in inst or 'vocal' in inst or 'voice' in inst:
            continue
        chord_count = 0
        for m in tdata.get('measures', []):
            for v in m.get('voices', []):
                for b in v.get('beats', []):
                    if len(b.get('notes', [])) >= 2:
                        chord_count += 1
        if chord_count > best_chord_count:
            best_chord_count = chord_count
            best_track = tdata
    rhythm_track = best_track

    if not rhythm_track:
        return result

    tuning = (rhythm_track.get('_meta', {}).get('tuning') or DEFAULT_TUNING)
    result = []

    for mi, m in enumerate(rhythm_track.get('measures', [])):
        sig = m.get('signature', [4, 4])
        measure_chords = []

        for v in m.get('voices', []):
            for b in v.get('beats', []):
                notes = b.get('notes', [])
                if len(notes) < 2:
                    continue
                frets = []
                for n in notes:
                    if n.get('string') is not None and n.get('fret') is not None:
                        frets.append((n['string'] - 1, n['fret']))
                if len(frets) >= 2:
                    midi_notes = _frets_to_notes(frets, tuning)
                    chord_name = _identify_chord(midi_notes)
                    if chord_name and (not measure_chords or measure_chords[-1] != chord_name):
                        measure_chords.append(chord_name)

        result.append({
            'measure': mi + 1,
            'signature': sig,
            'chords': measure_chords,
        })

    return result


_CHORDPRO_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/plain, application/json, */*',
    'Accept-Encoding': 'gzip, deflate',
}


def fetch_chordpro(song_id):
    """Fetch ChordPro text from Songsterr's ChordPro CDN.

    Returns a tuple (raw_text, metadata_dict) where metadata_dict has
    artist/title/songId from the chords API, or (None, None) if unavailable.
    """
    try:
        # Step 1: get the chordpro hash and chordsRevisionId
        resp = requests.get(
            f"{SONGSTERR_API}/chords/{song_id}",
            headers=_CHORDPRO_HEADERS,
            timeout=(5, 8),  # (connect, read) timeouts
        )
        if resp.status_code == 404:
            return None, None
        resp.raise_for_status()
        info = resp.json()

        chordpro_hash = info.get('chordpro')
        chords_revision_id = info.get('chordsRevisionId')
        if not chordpro_hash or chords_revision_id is None:
            return None, None

        # Extract metadata from this same response (avoid second API call)
        meta = {
            'title': info.get('title', ''),
            'artist': info.get('artist', ''),
            'songId': song_id,
        }

        # Step 2: fetch the .chordpro file from the CDN
        url = f"{CHORDPRO_CDN}/{song_id}/{chords_revision_id}/{chordpro_hash}.chordpro"
        cp_resp = requests.get(url, headers=_CHORDPRO_HEADERS, timeout=(5, 8))
        if cp_resp.status_code == 404:
            return None, meta
        cp_resp.raise_for_status()

        text = cp_resp.text.strip()
        return (text if text else None), meta

    except requests.RequestException as e:
        logger.debug(f"ChordPro fetch failed for song {song_id}: {e}")
        return None, None


def parse_chordpro(text):
    """Parse ChordPro text into structured data for renderManualChordChart().

    Returns a dict like:
    {
        "capo": 2,
        "tuning": "Standard (EADGBE)",
        "sections": [
            {
                "name": "Verse",
                "lines": [
                    {"chords": "Em7  G        Dsus4     A7sus4", "lyrics": "Today is gonna be the day"}
                ]
            }
        ],
        "source": "songsterr_chordpro"
    }

    Returns None if the ChordPro text has no actual chord content.
    """
    if not text:
        return None

    capo = None
    tuning = None
    sections = []
    current_section = None
    has_any_chords = False

    for raw_line in text.split('\n'):
        line = raw_line.rstrip()

        # Parse directives: {key: value}
        directive_match = _re.match(r'^\{(\w+):\s*(.*?)\}$', line.strip())
        if directive_match:
            key = directive_match.group(1).lower()
            value = directive_match.group(2).strip()
            if key == 'capo':
                try:
                    capo = int(value)
                except ValueError:
                    capo = value
            elif key == 'tuning':
                tuning = value
            elif key == 'section':
                current_section = {'name': value, 'lines': []}
                sections.append(current_section)
            continue

        # Skip empty lines (but preserve structure)
        if not line.strip():
            continue

        # Ensure we have a section container
        if current_section is None:
            current_section = {'name': 'Intro', 'lines': []}
            sections.append(current_section)

        # Parse chord+lyric lines: extract [Chord] markers
        if '[' in line:
            has_any_chords = True
            # Build aligned chord line and lyric line
            chords_line = ''
            lyrics_line = ''
            pos = 0
            first_chord = True

            for match in _re.finditer(r'\[([^\]]+)\]', line):
                # Text before this chord marker (lyrics between chords)
                pre_text = line[pos:match.start()]
                # Remove any chord markers from pre_text (shouldn't be any, but safe)
                clean_pre = _re.sub(r'\[([^\]]*)\]', '', pre_text)
                lyrics_line += clean_pre

                # Pad chord line to current lyrics position
                while len(chords_line) < len(lyrics_line):
                    chords_line += ' '

                # Ensure minimum spacing between chords when separator is whitespace-only
                if not first_chord and len(chords_line) > 0 and chords_line[-1] != ' ':
                    # Separator text was shorter than chord name — add min gap
                    min_gap = max(len(clean_pre), 3) if clean_pre.strip() == '' else 0
                    target = len(chords_line) + min_gap
                    while len(chords_line) < target:
                        chords_line += ' '
                    # Keep lyrics_line in sync so future chord positions align
                    while len(lyrics_line) < len(chords_line):
                        lyrics_line += ' '

                chord_name = match.group(1)
                chords_line += chord_name
                first_chord = False

                pos = match.end()

            # Remaining text after last chord
            remainder = line[pos:]
            clean_remainder = _re.sub(r'\[([^\]]*)\]', '', remainder)
            lyrics_line += clean_remainder

            # Pad chords line if lyrics is longer
            while len(chords_line) < len(lyrics_line):
                chords_line += ' '

            lyrics_stripped = lyrics_line.strip()
            chords_stripped = chords_line.rstrip()

            current_section['lines'].append({
                'chords': chords_stripped,
                'lyrics': lyrics_stripped if lyrics_stripped else None,
            })
        else:
            # Pure lyrics line (no chords)
            current_section['lines'].append({
                'lyrics': line.strip(),
            })

    if not has_any_chords:
        return None

    result = {
        'sections': sections,
        'source': 'songsterr_chordpro',
    }
    if capo is not None:
        result['capo'] = capo
    if tuning:
        result['tuning'] = tuning

    return result


@songsterr_bp.route('/api/songsterr/chordpro/<int:song_id>', methods=['GET'])
def songsterr_chordpro(song_id):
    """Fetch and parse ChordPro data from Songsterr CDN."""
    try:
        raw, meta = fetch_chordpro(song_id)
        if not raw:
            return jsonify({'error': 'No ChordPro data available'}), 404

        parsed = parse_chordpro(raw)
        if not parsed:
            return jsonify({'error': 'ChordPro has no chord content'}), 404

        # Add song metadata (already extracted from the chords API in fetch_chordpro)
        if meta:
            parsed['title'] = meta.get('title', '')
            parsed['artist'] = meta.get('artist', '')
            parsed['songId'] = meta.get('songId', song_id)

        return jsonify(parsed)

    except Exception as e:
        logger.error(f"ChordPro endpoint error for song {song_id}: {e}", exc_info=True)
        return jsonify({'error': 'ChordPro fetch failed', 'details': str(e)}), 500


def fetch_chord_data(song_id):
    """Fetch and compute chord data for a song. Returns a dict (not a Response).

    Shared by the /api/songsterr/chords endpoint and chord_sheet.py.
    Raises ValueError if data is missing, requests.RequestException on network errors.
    """
    import re

    # Get metadata
    meta_resp = requests.get(f"{SONGSTERR_API}/meta/{song_id}", timeout=10)
    meta_resp.raise_for_status()
    meta = meta_resp.json()

    revision_id = meta.get('revisionId')
    image_hash = meta.get('image')
    tracks_meta = meta.get('tracks', [])

    if not revision_id or not image_hash:
        raise ValueError('Missing revision data')

    # Fetch all tracks
    all_tracks = []
    for idx in range(len(tracks_meta)):
        url = f"{SONGSTERR_CDN}/{song_id}/{revision_id}/{image_hash}/{idx}.json"
        resp = requests.get(url, headers={
            'Accept-Encoding': 'gzip, deflate',
        }, timeout=15)
        resp.raise_for_status()
        all_tracks.append(resp.json())

    # Get tempo
    tempo = 120
    if all_tracks:
        automations = all_tracks[0].get('automations', {})
        tempo_list = automations.get('tempo', [])
        if tempo_list and isinstance(tempo_list, list):
            tempo = tempo_list[0].get('bpm', 120)

    # Get section markers from first track
    sections = []
    if all_tracks:
        for mi, m in enumerate(all_tracks[0].get('measures', [])):
            marker = m.get('marker')
            if marker and marker.get('text'):
                sections.append({'measure': mi + 1, 'name': marker['text']})

    # Extract chords (per-measure)
    chords = _extract_chords_from_tracks(all_tracks)

    # Calculate measure start times from tempo + time signatures
    tempo_changes = []
    if all_tracks:
        automations = all_tracks[0].get('automations', {})
        for tc in automations.get('tempo', []):
            tempo_changes.append({
                'measure': tc.get('measure', 0),
                'beat': tc.get('beat', 0),
                'bpm': tc.get('bpm', tempo)
            })

    current_time = 0.0
    current_bpm = tempo
    measure_durations = []
    for mi, m_data in enumerate(chords):
        sig = m_data.get('signature', [4, 4])
        for tc in tempo_changes:
            if tc['measure'] == mi:
                current_bpm = tc['bpm']

        m_data['time'] = round(current_time, 2)
        beats_in_measure = sig[0] * (4.0 / sig[1]) if sig[1] > 0 else 4
        measure_duration = (beats_in_measure * 60.0) / current_bpm
        measure_durations.append(measure_duration)
        current_time += measure_duration

    # Build chord events with timing: space multiple chords within a measure evenly
    chord_events = []
    last_chord = ''
    for mi, m_data in enumerate(chords):
        m_chords = m_data.get('chords', [])
        if not m_chords:
            continue
        m_start = m_data.get('time', 0)
        m_dur = measure_durations[mi] if mi < len(measure_durations) else 3.0
        for ci, ch in enumerate(m_chords):
            # Space chords evenly within the measure
            offset = (ci / max(len(m_chords), 1)) * m_dur
            t = round(m_start + offset, 2)
            if ch != last_chord:
                chord_events.append({'time': t, 'chord': ch})
                last_chord = ch

    # Extract lyrics from any track that has them
    lyrics_text = ''
    synced_lyrics = []
    for tdata in all_tracks:
        for lyric_entry in tdata.get('newLyrics', []):
            text = lyric_entry.get('text', '').strip()
            if text:
                lyrics_text = text
                break
        if lyrics_text:
            break

    # Fallback: fetch lyrics from lrclib.net if Songsterr has none
    if not lyrics_text:
        try:
            title = meta.get('title', '')
            artist = meta.get('artist', '')
            if title and artist:
                lrc_resp = requests.get(
                    'https://lrclib.net/api/search',
                    params={'track_name': title, 'artist_name': artist},
                    timeout=5
                )
                if lrc_resp.ok:
                    lrc_results = lrc_resp.json()
                    for lr in lrc_results:
                        synced = lr.get('syncedLyrics', '').strip()
                        plain = lr.get('plainLyrics', '').strip()
                        if synced:
                            for line in synced.split('\n'):
                                match = re.match(r'\[(\d+):(\d+\.\d+)\]\s*(.*)', line)
                                if match:
                                    mins = int(match.group(1))
                                    secs = float(match.group(2))
                                    text = match.group(3).strip()
                                    time_s = mins * 60 + secs
                                    synced_lyrics.append({'time': round(time_s, 2), 'text': text})
                            if not lyrics_text:
                                lyrics_text = plain or '\n'.join(sl['text'] for sl in synced_lyrics)
                            break
                        elif plain:
                            lyrics_text = plain
                            break
        except Exception as e:
            logger.debug(f"lrclib lyrics fallback failed: {e}")

    return {
        'songId': song_id,
        'title': meta.get('title', ''),
        'artist': meta.get('artist', ''),
        'tempo': tempo,
        'sections': sections,
        'measures': chords,
        'chordEvents': chord_events,
        'lyrics': lyrics_text,
        'syncedLyrics': synced_lyrics,
    }


@songsterr_bp.route('/api/songsterr/chords/<int:song_id>', methods=['GET'])
def songsterr_chords(song_id):
    """Extract chord chart from Songsterr tab data."""
    try:
        data = fetch_chord_data(song_id)
        return jsonify(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except requests.RequestException as e:
        logger.error(f"Songsterr chord extraction failed for song {song_id}: {e}")
        return jsonify({'error': 'Failed to fetch chord data', 'details': str(e)}), 502
    except Exception as e:
        logger.error(f"Chord extraction error for song {song_id}: {e}", exc_info=True)
        return jsonify({'error': 'Chord extraction failed', 'details': str(e)}), 500
