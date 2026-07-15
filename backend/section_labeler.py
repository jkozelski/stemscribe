"""Claude-mediated SECTION LABELER.

Re-segments an audio-detected chord chart into musical sections
(Intro/Verse/Pre-Chorus/Chorus/Bridge/Solo/Outro) by finding repeated
chord + lyric patterns.

Chords and lyrics are NEVER changed — only the `sections` grouping/labels are
rewritten. This is organization of the user's own audio-detected data
(chords from the audio, lyrics from the user's own Whisper transcription),
not chord recall.

Gated behind ENABLE_SECTION_LABELER (default: false).
"""
import os
import re
import json
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
_CHORDS_PER_LINE = 4

try:
    from chart_formatter import _build_slash_chord_line  # type: ignore
except ImportError:
    from backend.chart_formatter import _build_slash_chord_line  # type: ignore


SECTION_SYSTEM_PROMPT = """You are segmenting a chord chart DETECTED FROM AUDIO into musical sections. You get numbered bars (one chord each) plus lyric lines tagged with their bar position. Some lyric lines are marked [REPEATS] — those exact words recur elsewhere in the song.

These are OUR product conventions:
- CHORUS = the recurring hook — the block that comes back the same. [REPEATS] tags flag likely choruses, but TRUST YOUR EAR too: a hook that recurs is a Chorus even if the words were transcribed a little differently each time, or weren't tagged. Most songs HAVE a chorus — find it. Every chorus gets the label "Chorus".
- VERSE = the sung parts that carry the story forward. They often share a chord progression, and a verse CAN repeat a line — being repetitive does NOT make it a chorus; being the HOOK does. (e.g. Ring of Fire's verses repeat but aren't the chorus.)
- BRIDGE = a sung block that appears ONCE, sounds different, usually late in the song. Label a real bridge — don't fold it into a verse.
- Find the CHORUSES and BRIDGES — they carry the song. Be conservative ONLY with Pre-Chorus.
- PRE-CHORUS = only a short distinct block that repeats DIRECTLY before each Chorus. When unsure whether something is a Pre-Chorus, fold it into the Verse — that's the only place to default to Verse.
- SOLO = a chord-only stretch in the MIDDLE of the song.
- Do NOT worry about Intro/Outro — those are added automatically; just label those edge bars Verse/Solo and we'll fix them.
- Do NOT change chords/lyrics. Do NOT use memory of the real song — segment THIS data.
- Cover EVERY bar in order, no gaps/overlaps. start_bar/end_bar inclusive.

Output ONLY this JSON, no prose:
{"sections":[{"name":"Verse","start_bar":0,"end_bar":15},{"name":"Chorus","start_bar":16,"end_bar":23}]}"""


_SHARP_TO_FLAT = {'A#': 'Bb', 'C#': 'Db', 'D#': 'Eb', 'F#': 'Gb', 'G#': 'Ab'}
_FLAT_TO_SHARP = {v: k for k, v in _SHARP_TO_FLAT.items()}


def _key_prefers_flats(key: Optional[str]) -> bool:
    """Should chords be spelled with flats for this key? (A# vs Bb is the same
    pitch — the KEY decides the correct spelling.) C and flat keys → flats;
    sharp keys → sharps."""
    if not key:
        return False
    k = (str(key).lower()
         .replace('major', '').replace('maj', '')
         .replace('minor', 'm').replace('min', 'm')
         .replace('♯', '#').replace('♭', 'b').strip())
    flat_keys = {'c', 'f', 'bb', 'eb', 'ab', 'db', 'gb',
                 'am', 'dm', 'gm', 'cm', 'fm', 'bbm', 'ebm',
                 'a#', 'd#', 'g#'}
    sharp_keys = {'g', 'd', 'a', 'e', 'b', 'f#', 'c#',
                  'em', 'bm', 'f#m', 'c#m', 'g#m', 'd#m'}
    if k in sharp_keys:
        return False
    return k in flat_keys


def _respell_chord(chord: str, to_flats: bool) -> str:
    """Respell only the accidental note-names in a chord (root + bass), leaving
    the pitch and quality untouched. e.g. 'A#' -> 'Bb', 'D#m7' -> 'Ebm7'."""
    table = _SHARP_TO_FLAT if to_flats else _FLAT_TO_SHARP
    return re.sub(r'[A-G][#b]', lambda m: table.get(m.group(0), m.group(0)), chord or '')


def _respell_chart_for_key(chord_chart: Dict[str, Any]) -> None:
    """Normalize chord spelling to the chart's key (A# vs Bb etc.), in place."""
    to_flats = _key_prefers_flats(chord_chart.get('key'))
    for b in chord_chart.get('bar_grid') or []:
        if b.get('chord'):
            b['chord'] = _respell_chord(b['chord'], to_flats)
    cu = chord_chart.get('chords_used')
    if cu:
        chord_chart['chords_used'] = [_respell_chord(c, to_flats) for c in cu]
    for s in chord_chart.get('sections') or []:
        for ln in s.get('lines') or []:
            if ln.get('chords'):
                ln['chords'] = _respell_chord(ln['chords'], to_flats)
            for seg in ln.get('segments') or []:
                if seg.get('chord'):
                    seg['chord'] = _respell_chord(seg['chord'], to_flats)


def _enabled() -> bool:
    return (os.environ.get("ENABLE_SECTION_LABELER", "") or "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _api_key() -> Optional[str]:
    return os.environ.get("ANTHROPIC_API_KEY") or None


def _line_start_time(line: Dict[str, Any]) -> Optional[float]:
    for s in line.get("segments") or []:
        if s.get("start") is not None:
            return s["start"]
    return None


def _extract_lyric_lines(chord_chart: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Pull (bar_index, lyric_text) pairs from the input chart's sections,
    mapping each lyric line's start time to the nearest bar."""
    bar_grid = chord_chart.get("bar_grid") or []
    bar_starts = [(i, b.get("start_time")) for i, b in enumerate(bar_grid)
                  if b.get("start_time") is not None]
    if not bar_starts:
        return None
    out: List[Dict[str, Any]] = []
    for sec in chord_chart.get("sections") or []:
        for ln in sec.get("lines") or []:
            txt = (ln.get("lyrics") or "").strip()
            if not txt:
                continue
            t = _line_start_time(ln)
            if t is None:
                continue
            idx = min(bar_starts, key=lambda x: abs(x[1] - t))[0]
            out.append({"bar": idx, "text": txt})
    return out or None


def _query_sections(
    bars: List[str],
    title: str,
    artist: str,
    model: str,
    lyric_lines: Optional[List[Dict[str, Any]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    api_key = _api_key()
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    numbered = "\n".join(f"{i}: {ch}" for i, ch in enumerate(bars))
    user = f'Song: "{title}" by {artist}\nTotal bars: {len(bars)}\n\nBars:\n{numbered}'
    if lyric_lines:
        # Mark repeating lyric lines (FUZZY — Whisper transcribes a repeated
        # hook slightly differently each pass, so match on word overlap, not
        # verbatim) so Claude has a real Chorus signal to anchor on.
        def _wset(t: str):
            return set(re.sub(r"[^a-z0-9 ]", "", (t or "").lower()).split())
        wsets = [_wset(l.get("text", "")) for l in lyric_lines]
        repeats = [False] * len(lyric_lines)
        for i in range(len(lyric_lines)):
            ai = wsets[i]
            if len(ai) < 2:
                continue
            for j in range(len(lyric_lines)):
                if i == j:
                    continue
                bj = wsets[j]
                if len(bj) < 2:
                    continue
                union = len(ai | bj)
                if union and len(ai & bj) / union >= 0.6:
                    repeats[i] = True
                    break
        rows = []
        for idx, l in enumerate(lyric_lines):
            txt = l.get("text", "")
            if not txt:
                continue
            tag = " [REPEATS]" if repeats[idx] else ""
            rows.append(f'bar~{int(l.get("bar", -1))}: {txt}{tag}')
        if rows:
            user += (
                "\n\nLyric lines (approx bar position; [REPEATS] = these words "
                "recur elsewhere, a strong Chorus hint):\n" + "\n".join(rows)
            )
    user += "\n\nReturn ONLY the JSON."

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model,
            max_tokens=1500,
            system=[{
                "type": "text",
                "text": SECTION_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user}],
        )
    except Exception as e:
        logger.warning(f"[section_labeler] Anthropic call failed: {e}")
        return None

    text = ""
    for b in getattr(resp, "content", []) or []:
        if getattr(b, "type", None) == "text":
            text = (getattr(b, "text", "") or "").strip()
            break
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        secs = json.loads(m.group(0)).get("sections")
    except Exception:
        return None
    return secs if isinstance(secs, list) and secs else None


def _label_time_ranges(bar_grid: List[Dict], labels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    n = len(bar_grid)
    ranges = []
    for lab in labels:
        try:
            a = max(0, int(lab["start_bar"]))
            b = min(n - 1, int(lab["end_bar"]))
        except (KeyError, ValueError, TypeError):
            continue
        if b < a:
            continue
        t0 = bar_grid[a].get("start_time")
        t1 = bar_grid[b].get("end_time")
        ranges.append({"name": (lab.get("name") or "Section").strip(),
                       "a": a, "b": b, "t0": t0, "t1": t1, "lines": []})
    return ranges


def _regroup_lines(chord_chart: Dict[str, Any], ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Assign the chart's ORIGINAL lines (chords + lyrics intact) to Claude's
    section ranges by time, preserving everything the formatter already built."""
    orig_lines = []
    for sec in chord_chart.get("sections") or []:
        for ln in sec.get("lines") or []:
            orig_lines.append(ln)
    for ln in orig_lines:
        t = _line_start_time(ln)
        target = None
        if t is not None:
            for r in ranges:
                if r["t0"] is not None and r["t1"] is not None and r["t0"] - 0.5 <= t <= r["t1"] + 0.5:
                    target = r
                    break
            if target is None:
                target = min(
                    ranges,
                    key=lambda r: abs(((r["t0"] or 0) + (r["t1"] or 0)) / 2 - t),
                ) if ranges else None
        if target is not None:
            target["lines"].append(ln)
    return [{"name": r["name"], "lines": r["lines"]} for r in ranges if r["lines"]]


def _rebuild_chords_only(bar_grid: List[Dict], ranges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in ranges:
        sec_bars = bar_grid[r["a"]:r["b"] + 1]
        lines = []
        for i in range(0, len(sec_bars), _CHORDS_PER_LINE):
            chunk = sec_bars[i:i + _CHORDS_PER_LINE]
            names = [(x.get("chord") or "").strip() for x in chunk]
            segs = [{"chord": (x.get("chord") or "").strip(),
                     "start": x.get("start_time"), "end": x.get("end_time")} for x in chunk]
            lines.append({"chords": _build_slash_chord_line(names), "lyrics": None, "segments": segs})
        if lines:
            out.append({"name": r["name"], "lines": lines})
    return out


def _carve_leading_intro(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deterministic rule (Jeff 2026-05-29): everything instrumental at the
    very beginning is the Intro; the moment lyrics start, that's the Verse.
    No guessing.

    Pulls all leading no-lyric lines (across however many sections) into a
    single 'Intro', then the first sung line begins the body. If there's no
    instrumental lead-in at all, there's no Intro — and a section the model
    mislabeled 'Intro' that actually starts on a lyric is renamed 'Verse'."""
    if not sections:
        return sections
    intro_lines: List[Dict[str, Any]] = []
    body: List[Dict[str, Any]] = []
    hit_lyric = False
    for sec in sections:
        kept = []
        for ln in sec.get("lines") or []:
            if not hit_lyric and not (ln.get("lyrics") or "").strip():
                intro_lines.append(ln)
            else:
                hit_lyric = True
                kept.append(ln)
        if kept:
            body.append({"name": sec.get("name") or "Section", "lines": kept})
    # The body's first section is the song's first sung part — never 'Intro'.
    if body and (body[0].get("name") or "") == "Intro":
        body[0]["name"] = "Verse"
    if intro_lines:
        return [{"name": "Intro", "lines": intro_lines}] + body
    return body


def _carve_trailing_outro(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Mirror of the intro rule: instrumental lines AFTER the last sung line
    become the Outro."""
    if not sections:
        return sections
    outro_lines: List[Dict[str, Any]] = []
    body: List[Dict[str, Any]] = []
    hit_lyric = False
    for sec in reversed(sections):
        kept = []
        for ln in reversed(sec.get("lines") or []):
            if not hit_lyric and not (ln.get("lyrics") or "").strip():
                outro_lines.append(ln)
            else:
                hit_lyric = True
                kept.append(ln)
        if kept:
            body.append({"name": sec.get("name") or "Section", "lines": list(reversed(kept))})
    body = list(reversed(body))
    outro_lines = list(reversed(outro_lines))
    if outro_lines:
        return body + [{"name": "Outro", "lines": outro_lines}]
    return body


def _merge_consecutive_same_name(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge back-to-back sections that share a name (e.g. Chorus/Chorus/Chorus
    from a repeated outro-chorus) into one block. The renderer already collapses
    identical repeated lines with a x N badge, so the merged block stays clean."""
    out: List[Dict[str, Any]] = []
    for sec in sections:
        nm = sec.get("name") or "Section"
        if out and (out[-1].get("name") or "") == nm:
            out[-1]["lines"] = (out[-1].get("lines") or []) + (sec.get("lines") or [])
        else:
            out.append({"name": nm, "lines": list(sec.get("lines") or [])})
    return out


def label_sections(
    chord_chart: Dict[str, Any],
    *,
    enabled: Optional[bool] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    if enabled is None:
        enabled = _enabled()
    if not enabled or not isinstance(chord_chart, dict):
        return chord_chart

    bar_grid = chord_chart.get("bar_grid") or []
    if not bar_grid:
        return chord_chart

    # Spell chords to the key (A# -> Bb in flat keys, etc.) before anything else,
    # so the whole chart reads correctly. Same pitch, never changes the chord.
    _respell_chart_for_key(chord_chart)

    bars = [(b.get("chord") or "N").strip() for b in bar_grid]
    use_model = model or os.environ.get("SECTION_LABELER_MODEL") or _DEFAULT_MODEL
    title = (chord_chart.get("title") or "Unknown").strip()
    artist = (chord_chart.get("artist") or "Unknown").strip()
    lyric_lines = _extract_lyric_lines(chord_chart)

    labels = _query_sections(bars, title, artist, use_model, lyric_lines)
    if not labels:
        chord_chart.setdefault("section_labeler", {}).update({"status": "skipped"})
        return chord_chart

    ranges = _label_time_ranges(bar_grid, labels)
    if not ranges:
        chord_chart.setdefault("section_labeler", {}).update({"status": "skipped_rebuild"})
        return chord_chart

    # Prefer re-grouping the original lyric'd lines; fall back to chords-only
    # if the time-mapping collapses the structure.
    new_sections = _regroup_lines(chord_chart, ranges)
    used_lyrics = bool(lyric_lines)
    if len(new_sections) < max(2, len(ranges) - 1):
        new_sections = _rebuild_chords_only(bar_grid, ranges)
        used_lyrics = False

    new_sections = _carve_leading_intro(new_sections)
    new_sections = _carve_trailing_outro(new_sections)
    new_sections = _merge_consecutive_same_name(new_sections)

    if not new_sections:
        chord_chart.setdefault("section_labeler", {}).update({"status": "skipped_empty"})
        return chord_chart

    chord_chart["sections"] = new_sections
    chord_chart["section_labeler"] = {
        "status": "applied",
        "model": use_model,
        "n_sections": len(new_sections),
        "used_lyrics": used_lyrics,
    }
    logger.info(
        f"[section_labeler] {title!r}: {len(new_sections)} sections "
        f"({'with lyrics' if used_lyrics else 'chords-only'})"
    )
    return chord_chart
