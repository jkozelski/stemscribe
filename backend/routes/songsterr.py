"""
Songsterr integration — search for professional tabs and serve them as GP5 files.
"""

import io
import logging
import urllib.parse
import requests
from flask import Blueprint, request, jsonify, send_file

logger = logging.getLogger(__name__)

songsterr_bp = Blueprint("songsterr", __name__)

SONGSTERR_API = "https://www.songsterr.com/api"
SONGSTERR_CDN = "https://dqsljvtekg760.cloudfront.net"


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
    """Convert (string_index, fret) pairs to sorted (midi, note_name) list."""
    result = []
    for string_idx, fret in frets:
        if string_idx < len(tuning):
            midi = tuning[string_idx] + fret
            result.append((midi, NOTE_NAMES[midi % 12]))
    result.sort()  # lowest pitch first
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
        priority = 0  # higher = better

        # Major
        if {0, 4, 7} <= intervals:
            if 10 in intervals:
                name = root + '7'
            elif 11 in intervals:
                name = root + 'maj7'
            else:
                name = root
            priority = 3
        # Minor
        elif {0, 3, 7} <= intervals:
            if 10 in intervals:
                name = root + 'm7'
            else:
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

    Uses native chord annotations (beat.chord.text) only.
    """
    if not all_track_data:
        return []

    max_measures = max(len(t.get('measures', [])) for t in all_track_data)
    result = []

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

        result.append({
            'measure': mi + 1,
            'signature': sig,
            'chords': measure_chords,
        })

    return result


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
