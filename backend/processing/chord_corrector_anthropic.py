"""
Anthropic-mediated chord chart corrector — drop / replace / full-edit post-processor.

Three correction modes:
  1. drop-only      — remove chords librosa hallucinated; never adds. Safest.
  2. replace        — also swap chord names in bar_grid for Claude's spelling
                      when librosa is in the wrong key (e.g. Every Rose).
  3. full           — also relabel sections (Chorus -> Verse 1) and rebuild
                      chord-line layout per section. Lyrics + bar timing
                      remain audio-derived, untouched.

Hard rules across all modes:
  - LYRICS are read-only. Whisper transcribed them from the user's audio;
    Claude never sees or modifies them. Lyrics ARE protected expression
    (Passman ch.19). This boundary is critical for legal posture.
  - BAR TIMING / STRUCTURE is read-only. Audio is the source of truth for
    when chords change.
  - Chord NAMES and section LABELS are editable music-theory metadata.
  - Nothing is stored. Each correction is a fresh API call per upload.

Per Passman, chord progressions are largely unprotectable expression —
"you can't sue over chord structure, just words and melody." Combined with
the no-storage property, this corrector is structurally distinct from the
chord_recall_index that was deleted per Alexandra's Apr 10, 2026 directive
(which was a stored compilation/derivative work).

Feature flag:  ENABLE_ANTHROPIC_CORRECTION (default: false).
Mode flag:     ANTHROPIC_CORRECTION_MODE in {"drop","replace","full"} (default: drop).
API key:       ANTHROPIC_API_KEY env var, or macOS keychain `anthropic-api-key`.
Model:         claude-sonnet-4-5-20250929 by default (~$0.01-$0.02/song).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Reuse the audit oracle's prompt + normalization. Tries both import paths
# so the module works whether sys.path is rooted at the repo root (local
# tooling) or at backend/ (prod Flask runtime).
try:
    from audit.llm_oracle import SYSTEM_PROMPT, normalize_chord, normalize_chord_v2  # type: ignore
except ImportError:
    from backend.audit.llm_oracle import SYSTEM_PROMPT, normalize_chord, normalize_chord_v2  # type: ignore


_DEFAULT_MODEL = "claude-sonnet-4-5-20250929"


def _v2_enabled() -> bool:
    """V2 corrector prompt is gated behind ANTHROPIC_CORRECTION_V2_PROMPT.

    When enabled, the corrector uses RICH_CORRECTOR_SYSTEM_PROMPT (preserves
    extensions + slash chords) and normalize_chord_v2 (preserves slash bass).
    Default off — flip the env var to "true" / "1" to enable.
    """
    return (os.environ.get("ANTHROPIC_CORRECTION_V2_PROMPT", "") or "").strip().lower() in ("1", "true", "yes", "on")


def _norm(c: str) -> str:
    """Route to v1 or v2 normalizer based on the V2 flag."""
    return normalize_chord_v2(c) if _v2_enabled() else normalize_chord(c)


# V2 corrector prompt — replaces SYSTEM_PROMPT when ANTHROPIC_CORRECTION_V2_PROMPT=true.
# Designed to preserve extensions + slash chords vs the audit oracle prompt's
# "ROOT + QUALITY only" rule. See docs/corrector-v2-proposal-2026-05-12.md.
RICH_CORRECTOR_SYSTEM_PROMPT = """\
You are a music-theory editor producing a RICH, performance-quality chord chart \
for a song, competing head-to-head with Ultimate Guitar's "official" tabs and \
Chordify's pro charts. Your output drives a chord chart that musicians read \
while playing. A chart with the wrong vocabulary level is a failed chart — \
flattening Emmaj7 to Em, or dropping the C/G that defines a descending bassline, \
is the same kind of error as outputting the wrong chord entirely.

<role>
A noisy automatic chord detector has produced a candidate chord set for a song. \
Your job is NOT to clean up uncertainty by dropping things — your job is to \
RETURN THE RICHEST DEFENSIBLE CHORD VOCABULARY the song actually uses, drawn \
from the most-circulated published transcription (Hal Leonard, the artist's \
official folio, the top-voted Ultimate Guitar tab, or a respected songbook).
</role>

<vocabulary_floor>
Preserve, do not flatten:
  - Seventh chords: maj7, m7, dom7. If the published chart uses Emmaj7, output \
    "Emmaj7" — NEVER drop to "Em". If the song has any evidence of a 7th \
    interval on a chord (detected score, published source, or characteristic \
    voicing like a descending bass through the 7th), prefer the 7th-extension \
    chord over the bare triad.
  - Extensions: 9, 11, 13, add9, sus2, sus4. List them when the published \
    chart does.
  - Slash chords (inversions): if the bass note differs from the chord root in \
    the published chart, output the slash form (C/G, Am/F#, D/F#, G/B). \
    Descending or ascending basslines under a held chord almost always imply \
    slash voicings — preserve them.
  - Distinctive variants: Cadd9 stays Cadd9 (not C) if it's the signature \
    voicing.
</vocabulary_floor>

<anti_simplification_rules>
1. "Structural chord set" is NOT "minimal chord set." A song with 9 distinct \
   chords in its published chart should produce 9 entries, not 5.
2. Do NOT remove a chord just because it appears infrequently. Bridge chords, \
   pre-chorus turnarounds, and one-bar passing chords all belong in chord_set.
3. Do NOT collapse a slash chord into its root chord. C and C/G are different \
   entries.
4. Do NOT collapse a 7th-extension chord into its triad. Em and Em7 are \
   different entries. Em7 and Emmaj7 are different entries.
5. The ONLY things to drop are: detector hallucinations (chords with no \
   published support), enharmonic dupes (Bbm vs A#m — pick one per the key \
   signature rule below), and pure voicing variants (G with capo III vs open G \
   — same chord).
</anti_simplification_rules>

<notation>
Format: ROOT [QUALITY] [EXTENSION] [/BASS]
  - Root: A-G with # or b. Sharps in sharp keys (G/D/A/E/B/F#), flats in flat \
    keys (F/Bb/Eb/Ab/Db).
  - Quality: bare = major triad; "m" = minor; "dim", "aug", "sus2", "sus4".
  - Extension: "7" (dominant 7), "maj7", "m7", "9", "maj9", "m9", "11", "13", \
    "add9".
  - Slash bass: "/X" where X is the bass note (A-G with #/b).
Examples of valid tokens: G, Em, F#m7, Cmaj7, A9, Bbm, D7sus4, C/G, Am/F#, \
G/B, Emmaj7, Cadd9, Dsus4/F#.
</notation>

<output_schema>
Return a SINGLE JSON object, no code fences, no prose:
{
  "found": bool,         // true only if you confidently know the published chart
  "key": str,            // concise key, e.g. "G", "Bm", "Eb"
  "chord_set": [str,...] // unique chords as a set; slash chords and extensions REQUIRED when published
  "notes": str           // one line: notable extensions, slash basslines, capo, modal interchange
}
Set found=false (and empty chord_set) when you don't recognize the song with \
high confidence — never fabricate a rich chart.
</output_schema>

<examples>
Song: "Into the Great Wide Open" by Tom Petty
{"found": true, "key": "Em", "chord_set": ["Em", "Emmaj7", "Em7", "Em/C#", "C", "C/G", "Am/F#", "G", "D", "A", "Asus4"], "notes": "Verse rides a descending chromatic bassline under Em (E-D#-D-C#), notated as Em-Emmaj7-Em7-Em/C#; chorus uses C-G-D-Am with C/G and Am/F# passing slashes"}

Song: "Hotel California" by Eagles
{"found": true, "key": "Bm", "chord_set": ["Bm", "F#7", "A", "E7", "G", "D", "Em", "F#m7"], "notes": "Iconic descending-bass progression; F#7 is the V7 dominant, E7 is secondary dominant of A"}

Song: "Free Fallin'" by Tom Petty
{"found": true, "key": "F", "chord_set": ["F", "Bb", "C", "Csus4"], "notes": "Three-chord I-IV-V with Csus4 suspensions on the V"}

Song: "Hotel California" by Some Indie Band
{"found": false, "key": "", "chord_set": [], "notes": "Title matches Eagles song but artist suggests a cover — can't confirm canonical chords"}
</examples>
"""


# ---- quality-flips gate helpers (May 10 2026) ----
# Empirically validated against the 18-song May 8 audit
# (/tmp/audit-may8-results/). quality_flips counts Claude chord_set entries
# that share a root with a librosa chord but flip major<->minor (e.g.
# librosa D, Claude Dm). Pearson r vs delta_F1 = -0.649, the strongest
# pre-oracle signal. See docs/detector-signal-research-2026-05-10.md.

_PITCH_CLASS = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}


def _root_pc(c: str) -> Optional[int]:
    """Return integer pitch class 0..11 of chord root, or None."""
    m = re.match(r"^([A-G])([#b]?)", c or "")
    if not m:
        return None
    base = _PITCH_CLASS[m.group(1)]
    a = m.group(2)
    if a == '#':
        base += 1
    elif a == 'b':
        base -= 1
    return base % 12


def _is_minor(c: str) -> bool:
    """Tag a normalized chord as minor (m, m7, m9, ...) excluding maj7."""
    return 'm' in c and 'maj' not in c.lower()


def _bar_weights(bar_grid: List[Dict]) -> Dict[str, float]:
    """Fraction of bars each (normalized) chord occupies in librosa's bar_grid.
    Used by the retention-by-bar-weight signal: chords that occupy a large
    share of the song are unlikely to be hallucinations even if Claude drops
    them. See docs/detector-signal-research-2026-05-10.md open question #1
    (Hells Bells: librosa Am=27% of bars, Claude dropped it, F1 fell to 0.67)."""
    if not bar_grid:
        return {}
    counts: Dict[str, int] = {}
    total = 0
    for b in bar_grid:
        ch = (b.get("chord") or "").strip()
        if not ch:
            continue
        norm = normalize_chord(ch)
        if not norm:
            continue
        counts[norm] = counts.get(norm, 0) + 1
        total += 1
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _collapse_mode_pairs(
    chord_chart: Dict[str, Any], minority_threshold: float = 0.15
) -> List[Tuple[str, str, float]]:
    """Collapse same-root major/minor pairs when one is rare.

    klang.io head-to-head finding (2026-05-11): our detector over-emits Dm
    next to D, Fm next to F. Same root, different mode. The oracle counts
    the rare one as a precision miss. When one variant is <15% of bars AND
    the other dominates by 2x, the minority is almost certainly detector
    noise — collapse it into the majority's bar slots.

    NEVER adds chords. Only removes minorities and rewrites their bar slots
    to the majority. Safe to run after any correction mode.

    Returns list of (removed_chord, kept_chord, removed_weight) tuples for
    logging / meta. Empty list if nothing collapsed.

    GATED 2026-05-11: default OFF. Audit showed -0.025 F1 — Man in the Box
    dropped 0.80 -> 0.00 because the song legitimately uses both E and Em
    as chord changes, and the rule collapsed the rarer one. Heuristic works
    on most songs but catastrophically wrong on load-bearing-rare-mode
    cases. Re-enable with ANTHROPIC_CORRECTION_COLLAPSE_MODE_PAIRS=true
    only if a smarter rule lands.
    """
    if os.environ.get("ANTHROPIC_CORRECTION_COLLAPSE_MODE_PAIRS", "false").lower() not in (
        "1", "true", "yes"
    ):
        return []
    chords_used = chord_chart.get("chords_used") or []
    if len(chords_used) < 2:
        return []

    weights = _bar_weights(chord_chart.get("bar_grid") or [])
    if not weights:
        return []

    # Group chords by root pitch class
    by_root: Dict[int, List[str]] = {}
    for c in chords_used:
        rp = _root_pc(c)
        if rp is not None:
            by_root.setdefault(rp, []).append(c)

    collapses: List[Tuple[str, str, float]] = []
    for rp, variants in by_root.items():
        if len(variants) < 2:
            continue
        # Sort variants by bar weight descending
        weighted = [(c, weights.get(normalize_chord(c), 0.0)) for c in variants]
        weighted.sort(key=lambda x: -x[1])
        majority_chord, majority_weight = weighted[0]
        majority_norm = normalize_chord(majority_chord)
        for minority_chord, minority_weight in weighted[1:]:
            minority_norm = normalize_chord(minority_chord)
            # Three conditions to collapse:
            #   1. Minority occupies < minority_threshold of bars
            #   2. Majority dominates by 2x (avoids collapsing two near-equal variants)
            #   3. The pair is an actual major↔minor flip (not e.g. C and Cmaj7)
            if minority_weight >= minority_threshold:
                continue
            if majority_weight < minority_weight * 2:
                continue
            if _is_minor(majority_norm) == _is_minor(minority_norm):
                continue  # same mode — not a mode-pair flip
            collapses.append((minority_chord, majority_chord, round(minority_weight, 3)))

    if not collapses:
        return []

    # Apply: remove from chords_used, rewrite bar_grid
    norm_to_target: Dict[str, str] = {
        normalize_chord(minor): major for minor, major, _ in collapses
    }
    chord_chart["chords_used"] = [
        c for c in chords_used if normalize_chord(c) not in norm_to_target
    ]
    for b in chord_chart.get("bar_grid") or []:
        ch = (b.get("chord") or "").strip()
        if not ch:
            continue
        norm = normalize_chord(ch)
        if norm in norm_to_target:
            new_chord = norm_to_target[norm]
            b.setdefault("source_meta", {})["replaced_from"] = ch
            b["source_meta"]["reason"] = "mode-pair-collapse"
            b["chord"] = new_chord

    return collapses


def _quality_flips(librosa_set, claude_set) -> int:
    """Count Claude chords that share a root with a librosa chord but flip
    major<->minor. Excludes exact root+quality matches."""
    lib_by_root: Dict[int, List[str]] = {}
    for c in librosa_set:
        rp = _root_pc(c)
        if rp is not None:
            lib_by_root.setdefault(rp, []).append(c)
    flips = 0
    for cn in claude_set:
        rp = _root_pc(cn)
        if rp is None or rp not in lib_by_root:
            continue
        candidates = lib_by_root[rp]
        if any(o == cn for o in candidates):
            continue  # exact match — not a flip
        if any(_is_minor(o) != _is_minor(cn) for o in candidates):
            flips += 1
    return flips


def _api_key() -> Optional[str]:
    """Pull the Anthropic API key from env first, keychain second."""
    env_key = os.environ.get("ANTHROPIC_API_KEY")
    if env_key and env_key.startswith("sk-"):
        return env_key
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", "anthropic-api-key", "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            key = r.stdout.strip()
            if key.startswith("sk-"):
                return key
    except Exception:
        pass
    return None


def _query_canonical_chords(title: str, artist: str, model: str) -> Optional[Dict[str, Any]]:
    """Ask Claude for the canonical chord set. Returns None on any failure
    (network, parse, low confidence) so callers fall back to librosa's output."""
    api_key = _api_key()
    if not api_key:
        logger.warning("[chord_corrector] no Anthropic API key — skipping correction")
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("[chord_corrector] anthropic SDK not installed — skipping")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    user_msg = (
        f'Song: "{title}"' + (f" by {artist}" if artist else "") + "\n\n"
        "Return ONLY the JSON object. No code fences, no prose."
    )
    # Route system prompt through the V2 flag — RICH version preserves
    # extensions and slash chords; default V1 flattens them.
    active_system_prompt = RICH_CORRECTOR_SYSTEM_PROMPT if _v2_enabled() else SYSTEM_PROMPT
    if _v2_enabled():
        logger.info("[chord_corrector] using V2 rich-vocabulary prompt")
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=600,
            system=[{
                "type": "text",
                "text": active_system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        logger.warning(f"[chord_corrector] Anthropic API call failed: {e}")
        return None

    blocks = getattr(resp, "content", []) or []
    text = ""
    for b in blocks:
        if getattr(b, "type", None) == "text":
            text = (getattr(b, "text", "") or "").strip()
            break
    if not text:
        return None

    # Tolerant JSON extraction — model might wrap in fences despite the prompt
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _scrub_bar_grid(bar_grid: List[Dict], drop_chords: set) -> Tuple[List[Dict], int]:
    """Remove dropped chords from bar_grid by replacing them with the previous
    bar's chord (chord-held). Returns (new_grid, replacement_count)."""
    if not bar_grid or not drop_chords:
        return bar_grid, 0
    out: List[Dict] = []
    last_kept_chord = None
    replaced = 0
    for b in bar_grid:
        ch = (b.get("chord") or "").strip()
        norm = normalize_chord(ch) if ch else ""
        if norm in drop_chords:
            new_chord = last_kept_chord or ""
            new_b = dict(b)
            new_b["chord"] = new_chord
            new_b.setdefault("source_meta", {})["replaced_from"] = ch
            new_b["source_meta"]["reason"] = "anthropic-flagged-hallucination"
            out.append(new_b)
            replaced += 1
        else:
            last_kept_chord = ch
            out.append(b)
    return out, replaced


_FORMAT_SYSTEM_PROMPT = """\
You are a music-theory editor reviewing an automated chord chart for a song. \
You will receive the song's title, artist, and a JSON list of sections. Each \
section has a name and a list of "lines"; each line has a unique "id" (e.g. \
"2:0" = section 2, line 0), a chord string in slash notation, and the \
transcribed lyric text for that line.

Your job: return a corrected sections list. You may
  (a) relabel a section ("Chorus" -> "Verse 1")
  (b) rewrite the chord string of a given line
  (c) MOVE a line to a different section (when the lyric content clearly \
      belongs there — e.g. "desert highway" attached to the Intro when it's \
      really the start of Verse 1 of "Hotel California")

Hard rules
----------
1. LYRICS ARE READ-ONLY. Never modify, paraphrase, abbreviate, expand, \
   correct, or fill in missing words in the lyric text. The user's audio is \
   the source of truth — even when a line is obviously incomplete (e.g. \
   "desert highway" missing "On a dark"), keep it as-is. You can move that \
   line to the right section, but you can't fix the missing prefix.
2. Every line "id" you receive must appear EXACTLY ONCE in your output. \
   Don't duplicate a line. Don't drop a line. Don't invent a new line. The \
   set of ids in your output must equal the set of ids you received.
3. Section count in your output may differ from input (you can collapse \
   adjacent sections of the same type if they were over-split, or split a \
   section if multiple sections were over-merged). But you cannot drop or \
   invent lyric content.
4. Chord notation: ROOT + QUALITY only. No slash bass. Use the same \
   enharmonic conventions as the audit oracle (flats in flat keys, sharps \
   in sharp keys).
5. If you don't recognize the song with confidence, set "found": false and \
   leave everything unchanged.

Output format
-------------
Return a SINGLE JSON object, no fences:
  found     bool  — must be true to apply changes
  key       str   — concise key name
  sections  array — your corrected list. Each section:
                    {"name": str, "line_ids": [str, ...], "chord_overrides": {id: str}}
                    line_ids = ordered list of "id" values from input that go in this section
                    chord_overrides = OPTIONAL map of id -> new chord string. Omit any line whose chords you're not changing.
  notes     str   — one-line note about what you changed and why
"""


def _query_format_correction(
    title: str,
    artist: str,
    sections: List[Dict[str, Any]],
    model: str,
) -> Optional[Dict[str, Any]]:
    """Send the full chart structure (with lyrics, for context only) to Claude
    for editorial review. Claude can relabel sections, rewrite chord strings,
    and reassign lines to different sections. Lyric content is read-only.
    Returns None on any failure."""
    api_key = _api_key()
    if not api_key:
        return None
    try:
        import anthropic
    except ImportError:
        return None

    # Build the section payload with stable line IDs.
    payload_sections: List[Dict[str, Any]] = []
    for sec_idx, s in enumerate(sections):
        lines_out: List[Dict[str, Any]] = []
        for line_idx, ln in enumerate(s.get("lines") or []):
            lines_out.append({
                "id": f"{sec_idx}:{line_idx}",
                "chords": (ln.get("chords") or "").strip(),
                "lyrics": (ln.get("lyrics") or "").strip(),
            })
        payload_sections.append({
            "name": s.get("name") or "",
            "lines": lines_out,
        })

    user_msg = (
        f'Song: "{title}"' + (f" by {artist}" if artist else "") + "\n\n"
        "Sections:\n"
        + json.dumps(payload_sections, indent=2)
        + "\n\nReturn ONLY the JSON object."
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            system=[{
                "type": "text",
                "text": _FORMAT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        logger.warning(f"[chord_corrector] format-correction API failed: {e}")
        return None

    blocks = getattr(resp, "content", []) or []
    text = ""
    for b in blocks:
        if getattr(b, "type", None) == "text":
            text = (getattr(b, "text", "") or "").strip()
            break
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _replace_in_bar_grid(
    bar_grid: List[Dict],
    detected_set: Dict[str, str],
    canonical_set: set,
    canonical_chord_list: List[str],
) -> Tuple[List[Dict], int]:
    """Replace librosa's chord NAMES in bar_grid with the closest canonical
    chord, keeping the original timing. Used by 'replace' and 'full' modes.

    Strategy: for each bar, if the bar's chord normalizes to one of Claude's
    canonical chords, leave it. Otherwise, find the canonical chord with the
    same root if possible, else replace with the previous bar's kept chord.
    """
    if not bar_grid or not canonical_set:
        return bar_grid, 0

    # Build a "what should this become" map. Use _norm so slash chords are
    # preserved in V2 mode (otherwise Em/C# collapses to Em and we lose the
    # bass walk entirely).
    canonical_by_root: Dict[str, str] = {}
    for c in canonical_chord_list:
        n = _norm(c)
        if not n:
            continue
        m = re.match(r"^([A-G][#b]?)", n)
        if m and m.group(1) not in canonical_by_root:
            canonical_by_root[m.group(1)] = c  # keep original spelling

    out: List[Dict] = []
    last_kept = None
    replaced = 0
    for b in bar_grid:
        ch = (b.get("chord") or "").strip()
        norm = _norm(ch)
        if norm in canonical_set:
            out.append(b)
            last_kept = ch
            continue
        # Try same-root match
        m = re.match(r"^([A-G][#b]?)", norm)
        chosen = None
        if m and m.group(1) in canonical_by_root:
            chosen = canonical_by_root[m.group(1)]
        if not chosen:
            chosen = last_kept or (canonical_chord_list[0] if canonical_chord_list else "")
        if chosen and chosen != ch:
            new_b = dict(b)
            new_b["chord"] = chosen
            new_b.setdefault("source_meta", {})["replaced_from"] = ch
            new_b["source_meta"]["reason"] = "anthropic-replace-mode"
            out.append(new_b)
            last_kept = chosen
            replaced += 1
        else:
            out.append(b)
            last_kept = ch
    return out, replaced


def _extract_chord_tokens(chord_text: str) -> List[str]:
    """Pull chord names out of a slash-notation chord line.
    'Bm //// A //// E //// G ////' -> ['Bm', 'A', 'E', 'G'].
    Tokens that are pure slash separators or empty are dropped."""
    if not chord_text:
        return []
    out: List[str] = []
    for tok in chord_text.split():
        tok = tok.strip()
        if not tok:
            continue
        # Skip slash-only tokens (////, /, |)
        if all(c in "/|" for c in tok):
            continue
        # A real chord starts with [A-G]
        if tok and tok[0].upper() in "ABCDEFG":
            out.append(tok)
    return out


def _apply_format_corrections(
    chord_chart: Dict[str, Any],
    correction: Dict[str, Any],
) -> Tuple[int, int, int]:
    """Apply Claude's section-restructure + chord-line corrections to
    chord_chart in place.

    Claude returns each new section as {name, line_ids, chord_overrides}.
    We:
      1. Build a flat map of {line_id: line_dict} from current sections
      2. Validate Claude's line_ids are a permutation of the input set
         (no duplicates, no drops, no invented ids) — refuse to apply if not
      3. Rebuild sections in the order Claude proposed, pulling each line
         from its original location and applying chord_overrides per line.
         Lyric text + segments + bar timing on each line stay untouched.

    Returns (sections_relabeled, chord_lines_rewritten, lines_moved)."""
    if not correction.get("found"):
        return 0, 0, 0

    proposed = correction.get("sections") or []
    current = chord_chart.get("sections") or []

    # Build flat id -> line map and id -> origin-section-name map
    id_to_line: Dict[str, Dict[str, Any]] = {}
    id_to_origin_section: Dict[str, str] = {}
    for sec_idx, s in enumerate(current):
        sec_name = s.get("name") or ""
        for line_idx, ln in enumerate(s.get("lines") or []):
            lid = f"{sec_idx}:{line_idx}"
            id_to_line[lid] = ln
            id_to_origin_section[lid] = sec_name

    # Collect every id Claude proposed; validate strict permutation
    proposed_ids: List[str] = []
    seen: set = set()
    for s in proposed:
        for lid in (s.get("line_ids") or []):
            if not isinstance(lid, str):
                logger.warning(f"[chord_corrector] non-string line_id {lid!r} — refusing to apply")
                return 0, 0, 0
            if lid in seen:
                logger.warning(f"[chord_corrector] duplicate line_id {lid} — refusing to apply")
                return 0, 0, 0
            if lid not in id_to_line:
                logger.warning(f"[chord_corrector] invented line_id {lid} — refusing to apply")
                return 0, 0, 0
            seen.add(lid)
            proposed_ids.append(lid)

    if seen != set(id_to_line.keys()):
        missing = set(id_to_line.keys()) - seen
        logger.warning(
            f"[chord_corrector] response is missing {len(missing)} line(s) — refusing to apply: "
            f"{sorted(missing)[:5]}..."
        )
        return 0, 0, 0

    # Build the new sections list. Each line keeps every field from the
    # original (lyrics, _start, _end, _words, etc.) — only the `chords`
    # string and the per-segment chord names get overridden by Claude.
    new_sections: List[Dict[str, Any]] = []
    rewritten = 0
    relabeled = 0
    moved = 0
    for s in proposed:
        new_name = (s.get("name") or "").strip()
        overrides = s.get("chord_overrides") or {}
        new_lines: List[Dict[str, Any]] = []
        for lid in (s.get("line_ids") or []):
            ln = dict(id_to_line[lid])  # shallow copy so origin map stays intact
            if id_to_origin_section.get(lid) != new_name:
                moved += 1
            if lid in overrides:
                new_chord_str = (overrides[lid] or "").strip()
                if new_chord_str and new_chord_str != (ln.get("chords") or "").strip():
                    ln["chords"] = new_chord_str
                    # Also rewrite line.segments — the practice-page renderer
                    # uses segments[i].chord, NOT the chord text — so we must
                    # keep them in sync. Extract chord tokens from the new
                    # slash-notation string and replay them onto existing
                    # segments (preserving timing). If counts mismatch, do
                    # best-effort length-min replay.
                    if isinstance(ln.get("segments"), list) and ln["segments"]:
                        new_chords = _extract_chord_tokens(new_chord_str)
                        ln["segments"] = [dict(seg) for seg in ln["segments"]]  # don't mutate origin
                        for i, seg in enumerate(ln["segments"]):
                            if i < len(new_chords) and new_chords[i]:
                                seg["chord"] = new_chords[i]
                    rewritten += 1
            new_lines.append(ln)
        new_sections.append({"name": new_name, "lines": new_lines})

    # Count relabels by comparing against original sequence (best-effort —
    # if Claude restructured aggressively this is approximate)
    for i, s in enumerate(new_sections):
        if i < len(current) and current[i].get("name") != s.get("name"):
            relabeled += 1

    chord_chart["sections"] = new_sections
    return relabeled, rewritten, moved


# ---------------------------------------------------------------------------
# RE-RANKER STRATEGY (May 11 2026)
# ---------------------------------------------------------------------------
# Replaces the "generator" contract (Claude invents a chord_set from title +
# artist alone) with "re-ranker": Claude sees librosa's top-K template-match
# candidates per bar with cosine-similarity scores and picks one per bar
# (or abstains). Eliminates the wrong-key failure mode structurally — Claude
# can never pick a chord librosa didn't surface as a candidate.
#
# See docs/reranker-design-2026-05-11.md for the full spec, including the
# token-budget analysis (~3.6K input + ~1.2K output per song, ~$0.03/song
# with ephemeral-cache hits on the system prompt).
# ---------------------------------------------------------------------------


_RERANKER_SYSTEM_PROMPT = """\
You are a chord-chart re-ranker. For each bar of a song, you receive 3-5 \
chord candidates from a librosa template detector, each with a cosine \
similarity score in [0,1]. Pick the single best chord per bar.

Hard rules:
  - You MUST pick one of the provided candidates OR output "abstain".
  - "abstain" means "candidates look noisy, keep librosa's top pick" — \
use it for bars where the top score is < 0.65 or all candidates look \
musically implausible given the song key.
  - DO NOT invent chord names not in the candidate list.
  - DO NOT add extensions (7, maj7, sus, etc.) — the candidate vocabulary \
is maj/min triads only. Note them in `notes` if a chord SHOULD be a 7 \
chord, but pick the triad in the candidates.
  - Prefer chords diatonic to the provided key and consistent with the \
song's most-frequent chord set.
  - When two candidates are within 0.05 score of each other, prefer the \
one that maintains chord continuity with the previous bar (avoid \
one-bar outliers between two identical neighbors).
  - Within a bar's candidates, if `Xm` and `X` (same root, different \
quality) both appear, prefer the one diatonic to the declared key — but \
if neither is, prefer the higher-score one.

Output a JSON object, no fences:
{
  "found": true | false,
  "bars": [
    {"bar": 1, "pick": "Am", "abstain": false},
    {"bar": 2, "pick": null, "abstain": true},
    ...
  ],
  "notes": "one-line summary"
}

Every bar in the input must appear exactly once in "bars". \
`pick` must be null when `abstain` is true and non-null otherwise.
"""


def _extract_top_k_from_bar_grid(bar_grid: List[Dict]) -> Optional[List[Dict[str, Any]]]:
    """Walk bar_grid and pull each bar's source_meta.top_k. Returns
    None when no bar carries a top_k field (e.g. legacy detectors, or
    candidates were filtered out for every bar by min-score floor).
    """
    if not bar_grid:
        return None
    out: List[Dict[str, Any]] = []
    found_any = False
    for b in bar_grid:
        bar_num = b.get("bar")
        if bar_num is None:
            continue
        sm = b.get("source_meta") or {}
        top_k = sm.get("top_k") or []
        if top_k:
            found_any = True
        out.append({
            "bar": bar_num,
            "argmax": b.get("chord"),
            "candidates": top_k,
        })
    if not found_any:
        return None
    return out


def _summarize_chords_used(chord_chart: Dict[str, Any]) -> str:
    """Build a short 'chord: x% of bars' summary for the reranker prompt
    so Claude sees the global chord distribution alongside per-bar candidates.
    """
    bar_grid = chord_chart.get("bar_grid") or []
    if not bar_grid:
        return ""
    weights = _bar_weights(bar_grid)
    if not weights:
        return ""
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:8]
    return ", ".join(f"{c}:{int(round(w * 100))}%" for c, w in ranked)


def _query_reranker(
    title: str,
    artist: str,
    key: str,
    tempo: float,
    bars_payload: List[Dict[str, Any]],
    chord_set_summary: str,
    model: str,
) -> Optional[Dict[str, Any]]:
    """Call Claude with per-bar librosa candidates and ask for per-bar picks.
    Returns None on any failure (no key, network, malformed response)."""
    api_key = _api_key()
    if not api_key:
        logger.warning("[chord_corrector:reranker] no Anthropic API key — skipping")
        return None
    try:
        import anthropic
    except ImportError:
        logger.warning("[chord_corrector:reranker] anthropic SDK not installed — skipping")
        return None

    # Compact JSON for token efficiency — drop spaces, alias keys.
    compact_bars = [
        {
            "bar": b["bar"],
            "candidates": [
                {"c": cd.get("chord"), "s": cd.get("score")}
                for cd in (b.get("candidates") or [])
            ],
        }
        for b in bars_payload
    ]

    user_msg_parts = [f'Song: "{title}"']
    if artist:
        user_msg_parts.append(f"by {artist}")
    header = " ".join(user_msg_parts)
    user_msg = (
        f"{header}\n"
        f"Librosa key: {key or 'unknown'}\n"
        f"Librosa tempo: {tempo:.0f} BPM\n"
        f"Top chords across the whole song (by bar weight): {chord_set_summary or '(unknown)'}\n\n"
        "Bars (top-K candidates per bar, score in 0..1):\n"
        f"{json.dumps(compact_bars, separators=(',', ':'))}\n\n"
        "Return ONLY the JSON object."
    )

    client = anthropic.Anthropic(api_key=api_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0,
            system=[{
                "type": "text",
                "text": _RERANKER_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": user_msg}],
        )
    except Exception as e:
        logger.warning(f"[chord_corrector:reranker] Anthropic API call failed: {e}")
        return None

    blocks = getattr(resp, "content", []) or []
    text = ""
    for b in blocks:
        if getattr(b, "type", None) == "text":
            text = (getattr(b, "text", "") or "").strip()
            break
    if not text:
        return None

    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None


def _validate_reranker_response(
    response: Dict[str, Any],
    bars_payload: List[Dict[str, Any]],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Validate Claude's response shape. Returns {bar_num: {"pick","abstain"}}
    on success or None if the response violates any contract."""
    if not isinstance(response, dict):
        return None
    bars_out = response.get("bars")
    if not isinstance(bars_out, list):
        return None

    expected_bars = {b["bar"] for b in bars_payload}
    candidates_by_bar: Dict[int, set] = {
        b["bar"]: {cd.get("chord") for cd in (b.get("candidates") or []) if cd.get("chord")}
        for b in bars_payload
    }

    seen: Dict[int, Dict[str, Any]] = {}
    for entry in bars_out:
        if not isinstance(entry, dict):
            return None
        bar_num = entry.get("bar")
        try:
            bar_num = int(bar_num)
        except (TypeError, ValueError):
            return None
        if bar_num not in expected_bars or bar_num in seen:
            return None
        abstain = bool(entry.get("abstain"))
        pick = entry.get("pick")
        if abstain:
            if pick is not None:
                return None
        else:
            if not isinstance(pick, str) or not pick.strip():
                return None
            if pick not in candidates_by_bar.get(bar_num, set()):
                return None
        seen[bar_num] = {"pick": pick, "abstain": abstain}

    if set(seen.keys()) != expected_bars:
        return None
    return seen


def _apply_reranker_picks(
    chord_chart: Dict[str, Any],
    picks: Dict[int, Dict[str, Any]],
) -> Dict[str, int]:
    """Walk bar_grid; rewrite bar.chord per Claude's pick when not abstained.
    Records source_meta.replaced_from for any rewritten bar. Returns a dict of
    counters {bars_rewritten, bars_abstained, bars_unchanged}.
    """
    bar_grid = chord_chart.get("bar_grid") or []
    counters = {"bars_rewritten": 0, "bars_abstained": 0, "bars_unchanged": 0}
    new_grid: List[Dict[str, Any]] = []
    for b in bar_grid:
        bar_num = b.get("bar")
        entry = picks.get(bar_num) if bar_num is not None else None
        if entry is None:
            new_grid.append(b)
            counters["bars_unchanged"] += 1
            continue
        if entry["abstain"]:
            new_grid.append(b)
            counters["bars_abstained"] += 1
            continue
        new_chord = entry["pick"]
        old_chord = (b.get("chord") or "").strip()
        if new_chord == old_chord:
            new_grid.append(b)
            counters["bars_unchanged"] += 1
            continue
        new_b = dict(b)
        new_b["chord"] = new_chord
        sm = dict(new_b.get("source_meta") or {})
        sm["replaced_from"] = old_chord
        sm["reason"] = "reranker-rerank"
        new_b["source_meta"] = sm
        new_grid.append(new_b)
        counters["bars_rewritten"] += 1
    chord_chart["bar_grid"] = new_grid
    return counters


def _recompute_chords_used(chord_chart: Dict[str, Any]) -> List[str]:
    """De-dup ordered list of chord names appearing in the final bar_grid."""
    seen: set = set()
    out: List[str] = []
    for b in chord_chart.get("bar_grid") or []:
        ch = (b.get("chord") or "").strip()
        if not ch or ch in seen:
            continue
        seen.add(ch)
        out.append(ch)
    return out


def apply_correction_reranker(
    chord_chart: Dict[str, Any],
    *,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-ranker corrector: Claude picks per bar from librosa's top-K
    candidates. Returns chord_chart unchanged on any failure (no top-K
    available, no API key, network/parse error). NEVER crashes the pipeline.

    Reads from chord_chart["bar_grid"][i]["source_meta"]["top_k"] (populated
    by the librosa detector via chart_formatter._quantize_chords_to_bars).
    When the detector path didn't supply candidates (legacy stem-based path
    or all candidates filtered below min-score), this function is a no-op
    and stamps anthropic_correction.status accordingly.
    """
    if not isinstance(chord_chart, dict):
        return chord_chart

    title = (chord_chart.get("title") or "").strip()
    artist = (chord_chart.get("artist") or "").strip()
    bar_grid = chord_chart.get("bar_grid") or []

    use_model = model or os.environ.get("ANTHROPIC_CORRECTION_MODEL") or _DEFAULT_MODEL

    meta: Dict[str, Any] = {
        "strategy": "reranker",
        "model": use_model,
    }

    bars_payload = _extract_top_k_from_bar_grid(bar_grid)
    if not bars_payload:
        meta["status"] = "skipped_no_top_k"
        chord_chart["anthropic_correction"] = meta
        logger.info(
            f"[chord_corrector:reranker] {title!r}: no top_k in bar_grid; "
            "falling back to detector output unchanged"
        )
        return chord_chart

    if not title:
        meta["status"] = "skipped_no_title"
        chord_chart["anthropic_correction"] = meta
        return chord_chart

    chord_set_summary = _summarize_chords_used(chord_chart)
    key = (chord_chart.get("key") or chord_chart.get("detected_key") or "").strip()
    tempo = 0.0
    try:
        tempo = float(
            chord_chart.get("tempo")
            or chord_chart.get("bpm")
            or (chord_chart.get("metadata") or {}).get("tempo")
            or 0.0
        )
    except (TypeError, ValueError):
        tempo = 0.0

    response = _query_reranker(
        title=title,
        artist=artist,
        key=key,
        tempo=tempo,
        bars_payload=bars_payload,
        chord_set_summary=chord_set_summary,
        model=use_model,
    )
    if response is None:
        meta["status"] = "skipped_no_response"
        chord_chart["anthropic_correction"] = meta
        logger.warning(f"[chord_corrector:reranker] {title!r}: no response from Claude")
        return chord_chart

    if not response.get("found"):
        meta.update({
            "status": "skipped_unrecognized",
            "claude_notes": response.get("notes", ""),
        })
        chord_chart["anthropic_correction"] = meta
        logger.info(
            f"[chord_corrector:reranker] {title!r}: Claude found=false "
            f"(notes: {response.get('notes', '')[:80]})"
        )
        return chord_chart

    picks = _validate_reranker_response(response, bars_payload)
    if picks is None:
        meta["status"] = "reranker_validation_failed"
        meta["claude_notes"] = response.get("notes", "")
        chord_chart["anthropic_correction"] = meta
        logger.warning(
            f"[chord_corrector:reranker] {title!r}: response failed validation; "
            "leaving bar_grid unchanged"
        )
        return chord_chart

    counters = _apply_reranker_picks(chord_chart, picks)
    chord_chart["chords_used"] = _recompute_chords_used(chord_chart)

    meta.update({
        "status": "applied",
        "claude_notes": response.get("notes", ""),
        "bars_total": len(bars_payload),
        **counters,
    })
    chord_chart["anthropic_correction"] = meta
    logger.info(
        f"[chord_corrector:reranker] {title!r}: "
        f"rewrote={counters['bars_rewritten']} "
        f"abstained={counters['bars_abstained']} "
        f"unchanged={counters['bars_unchanged']}"
    )
    return chord_chart


def apply_correction(
    chord_chart: Dict[str, Any],
    *,
    enabled: Optional[bool] = None,
    mode: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Optionally rewrite a chord_chart dict by dropping Anthropic-flagged
    hallucinations. NEVER adds chords. Returns the (possibly mutated) dict.

    On any failure mode (no API key, low confidence, network error) returns
    the input unchanged — librosa's output is always the safe fallback.

    The corrected chart gets a non-load-bearing meta field so the frontend can
    show "edited by AI review" if you want a UI hint. Removed chords are
    listed in `chord_chart["anthropic_correction"]["dropped"]`.

    Env var `ANTHROPIC_CORRECTION_STRATEGY` selects between the original
    "generator" path (Claude invents a chord_set from title + artist) and the
    new "reranker" path (Claude picks per bar from librosa's top-K). Default
    is "generator" to preserve current prod behavior.
    """
    if enabled is None:
        enabled = os.environ.get("ENABLE_ANTHROPIC_CORRECTION", "").lower() in ("1", "true", "yes")
    if not enabled:
        return chord_chart

    if not isinstance(chord_chart, dict):
        return chord_chart

    # Strategy dispatch. Reranker is a fundamentally different contract
    # (per-bar votes vs whole-chord-set rewrite) so it gets its own function.
    strategy = (os.environ.get("ANTHROPIC_CORRECTION_STRATEGY") or "generator").strip().lower()
    if strategy == "reranker":
        try:
            return apply_correction_reranker(chord_chart, model=model)
        except Exception as e:
            # Reranker must never crash the pipeline. Fall through to the
            # generator path so corrections still happen even when the new
            # code path explodes.
            logger.warning(
                f"[chord_corrector] reranker raised {type(e).__name__}: {e} — "
                "falling through to generator strategy"
            )

    title = (chord_chart.get("title") or "").strip()
    artist = (chord_chart.get("artist") or "").strip()
    chords_used = chord_chart.get("chords_used") or []

    if not title or not chords_used:
        return chord_chart

    use_mode = (mode or os.environ.get("ANTHROPIC_CORRECTION_MODE") or "drop").strip().lower()
    if use_mode not in ("drop", "replace", "full"):
        use_mode = "drop"
    use_model = model or os.environ.get("ANTHROPIC_CORRECTION_MODEL") or _DEFAULT_MODEL

    canon = _query_canonical_chords(title, artist, use_model)
    if not canon:
        chord_chart.setdefault("anthropic_correction", {})["status"] = "skipped_no_response"
        return chord_chart

    if not canon.get("found"):
        chord_chart.setdefault("anthropic_correction", {}).update({
            "status": "skipped_unrecognized",
            "mode": use_mode,
            "claude_notes": canon.get("notes", ""),
        })
        logger.info(
            f"[chord_corrector] Claude didn't recognize {title!r} by {artist!r}; "
            f"skipping (mode={use_mode}, notes: {canon.get('notes', '')[:80]})"
        )
        return chord_chart

    canonical_chord_list = list(canon.get("chord_set") or [])
    # V2 mode preserves slash chords through normalization; V1 strips them.
    canonical_set = {_norm(c) for c in canonical_chord_list if c}
    canonical_set.discard("")
    detected_norm = {_norm(c): c for c in chords_used if c}
    detected_norm.pop("", None)

    drop_norm = {n for n in detected_norm if n not in canonical_set}
    drop_ratio = len(drop_norm) / max(1, len(detected_norm))

    meta: Dict[str, Any] = {
        "mode": use_mode,
        "model": use_model,
        "claude_chord_set": sorted(canonical_set),
        "claude_key": canon.get("key", ""),
        "claude_notes": canon.get("notes", ""),
    }

    # ---------- BAR-WEIGHT RETENTION (May 11 2026) ----------
    # When librosa heavily weights a chord (occupies >N% of bars) AND Claude
    # is doing a SURGICAL edit (not a wholesale key-rewrite), trust librosa
    # over Claude on those chords. Targets the Hells Bells failure pattern:
    # librosa had Am at 27% of bars; Claude returned a strict subset that
    # dropped Am; oracle scored F1=0.67. With retention, Am stays.
    #
    # Guardrail: SKIP retention when drop_ratio is high (Claude is doing a
    # legitimate wholesale rewrite — e.g. Bad Company D#->D). Retention is
    # only safe when Claude is selectively pruning, not transposing keys.
    if use_mode in ("replace", "full"):
        retention_threshold = float(
            os.environ.get("ANTHROPIC_CORRECTION_RETENTION_THRESHOLD") or "0.15"
        )
        retention_dropratio_max = float(
            os.environ.get("ANTHROPIC_CORRECTION_RETENTION_DROPRATIO_MAX") or "0.7"
        )
        if drop_ratio < retention_dropratio_max and drop_norm:
            weights = _bar_weights(chord_chart.get("bar_grid") or [])
            retained = []
            for n in drop_norm:
                if weights.get(n, 0.0) >= retention_threshold:
                    canonical_set.add(n)
                    # Preserve the original label (case, accidentals) — pick
                    # whatever librosa had under this normalized form.
                    original_label = detected_norm.get(n) or n
                    if original_label not in canonical_chord_list:
                        canonical_chord_list.append(original_label)
                    retained.append((original_label, round(weights[n], 2)))
            if retained:
                # Drop_norm shrinks by what we restored
                for label, _ in retained:
                    drop_norm.discard(normalize_chord(label))
                drop_ratio = len(drop_norm) / max(1, len(detected_norm))
                meta["retained_high_weight_chords"] = retained
                meta["drop_ratio_after_retention"] = round(drop_ratio, 2)
                logger.info(
                    f"[chord_corrector:retain] {title!r} retained "
                    f"{[r[0] for r in retained]} (weights {[r[1] for r in retained]}) "
                    f"that Claude wanted to drop"
                )

    # ---------- QUALITY-FLIP GATE (May 10 2026) ----------
    # Corrector frequently fails by transposing to the wrong relative key
    # (major<->minor swap on the same root: Bm->Am, D->Dm, F#m->F).
    # Catches Sister Golden Hair / Misunderstood / Wildest Dreams losses
    # without sacrificing Bad Company / Man in the Box / Every Rose / Highway
    # wins (those have quality_flips=0). Predicted mean F1 lift on 18-song
    # audit: 0.768 -> 0.840. Doesn't apply in drop-only mode (drop never
    # introduces a flip — it only removes). See docs/detector-signal-research-2026-05-10.md.
    if use_mode in ("replace", "full"):
        qf = _quality_flips(set(detected_norm.keys()), canonical_set)
        qf_threshold = int(os.environ.get("ANTHROPIC_CORRECTION_QFLIP_GATE") or "2")
        qf_drop_ceiling = float(os.environ.get("ANTHROPIC_CORRECTION_QFLIP_DROP_GATE") or "0.4")
        if qf >= qf_threshold or (qf >= 1 and drop_ratio < qf_drop_ceiling):
            meta.update({
                "status": "skipped_quality_flip",
                "drop_ratio": round(drop_ratio, 2),
                "quality_flips": qf,
                "qf_threshold": qf_threshold,
                "qf_drop_ceiling": qf_drop_ceiling,
                "detector_chord_set": sorted(detected_norm.keys()),
            })
            chord_chart["anthropic_correction"] = meta
            logger.warning(
                f"[chord_corrector:qflip] {title!r}: quality_flips={qf}, "
                f"drop_ratio={drop_ratio:.0%}; likely wrong-key correction, "
                f"leaving detector output untouched"
            )
            return chord_chart

    # ---------- DROP MODE ----------
    if use_mode == "drop":
        if not drop_norm:
            meta["status"] = "no_changes_needed"
            chord_chart["anthropic_correction"] = meta
            return chord_chart
        if drop_ratio >= 0.7:
            meta.update({
                "status": "skipped_too_many_drops",
                "drop_ratio": round(drop_ratio, 2),
                "detector_chord_set": sorted(detected_norm.keys()),
            })
            chord_chart["anthropic_correction"] = meta
            logger.warning(
                f"[chord_corrector:drop] {drop_ratio:.0%} flagged on {title!r} — "
                f"likely wrong-key bug; leaving output untouched"
            )
            return chord_chart
        dropped_labels = sorted(detected_norm[n] for n in drop_norm)
        chord_chart["chords_used"] = [
            c for c in chords_used if normalize_chord(c) not in drop_norm
        ]
        new_grid, n_replaced = _scrub_bar_grid(
            chord_chart.get("bar_grid") or [], drop_norm
        )
        chord_chart["bar_grid"] = new_grid
        mode_pair_collapses = _collapse_mode_pairs(chord_chart)
        meta.update({
            "status": "applied",
            "dropped": dropped_labels,
            "bars_replaced": n_replaced,
        })
        if mode_pair_collapses:
            meta["mode_pair_collapses"] = mode_pair_collapses
        chord_chart["anthropic_correction"] = meta
        logger.info(f"[chord_corrector:drop] {title!r}: dropped {dropped_labels}")
        return chord_chart

    # ---------- REPLACE MODE ----------
    # Same as drop, but also rewrite mismatched chord names in bar_grid to
    # the closest canonical chord (same root if possible, else hold previous).
    if use_mode == "replace":
        new_grid, n_replaced = _replace_in_bar_grid(
            chord_chart.get("bar_grid") or [],
            detected_norm,
            canonical_set,
            canonical_chord_list,
        )
        chord_chart["bar_grid"] = new_grid
        chord_chart["chords_used"] = list(canonical_chord_list)
        mode_pair_collapses = _collapse_mode_pairs(chord_chart)
        meta.update({
            "status": "applied",
            "bars_replaced": n_replaced,
            "drop_ratio": round(drop_ratio, 2),
        })
        if mode_pair_collapses:
            meta["mode_pair_collapses"] = mode_pair_collapses
        chord_chart["anthropic_correction"] = meta
        logger.info(
            f"[chord_corrector:replace] {title!r}: rewrote {n_replaced} bars; "
            f"chords_used now {canonical_chord_list}"
        )
        return chord_chart

    # ---------- FULL MODE ----------
    # 1. Rewrite bar_grid + chords_used (replace-style)
    # 2. Send section names + chord-only lines to Claude for editorial review,
    #    apply section relabeling + chord-line rewrites. Lyrics never sent.
    new_grid, n_replaced = _replace_in_bar_grid(
        chord_chart.get("bar_grid") or [],
        detected_norm,
        canonical_set,
        canonical_chord_list,
    )
    chord_chart["bar_grid"] = new_grid
    chord_chart["chords_used"] = list(canonical_chord_list)

    sections_relabeled = 0
    chord_lines_rewritten = 0
    lines_moved = 0
    fmt = _query_format_correction(title, artist, chord_chart.get("sections") or [], use_model)
    if fmt:
        sections_relabeled, chord_lines_rewritten, lines_moved = _apply_format_corrections(
            chord_chart, fmt
        )
    mode_pair_collapses = _collapse_mode_pairs(chord_chart)
    meta.update({
        "status": "applied",
        "bars_replaced": n_replaced,
        "sections_relabeled": sections_relabeled,
        "chord_lines_rewritten": chord_lines_rewritten,
        "lines_moved": lines_moved,
        "drop_ratio": round(drop_ratio, 2),
        "claude_format_notes": (fmt or {}).get("notes", "") if fmt else "",
    })
    if mode_pair_collapses:
        meta["mode_pair_collapses"] = mode_pair_collapses
    chord_chart["anthropic_correction"] = meta
    logger.info(
        f"[chord_corrector:full] {title!r}: bars={n_replaced} "
        f"sections={sections_relabeled} chord_lines={chord_lines_rewritten} "
        f"moved={lines_moved}"
    )
    return chord_chart
