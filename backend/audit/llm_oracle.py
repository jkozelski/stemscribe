"""LLM-as-oracle audit harness for StemScriber chord detection.

For each chord_chart.json under the given directory, asks Claude what the
canonical chord set should be for (title, artist), normalizes both sides,
and reports precision / recall / F1 against the detector's `chords_used`.

Usage:
    ANTHROPIC_API_KEY=sk-ant-... python -m backend.audit.llm_oracle --dir /tmp/kk_jobs/all
    # or stash key in macOS keychain under service "anthropic-api-key" and the
    # script will pick it up automatically.

The system prompt is cached (`cache_control: ephemeral`) so re-runs across
the 17-job audit set only pay full price for the per-song user turn.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Iterable

import anthropic


SYSTEM_PROMPT = """\
You are a music-theory expert evaluating an automated chord-detection system. \
Your job: given a song's title and artist, return the canonical chord set the \
song uses, drawn from the most commonly-circulated published transcription \
(Hal Leonard chart, an artist's official transcription, a well-sourced \
classic-rock songbook, the consensus version on a major chord-archive site). \
Use the most common version; do not list rare variations or improvised live \
changes.

Output format
-------------
Return a SINGLE JSON object with these fields. Do not include code fences, \
prose, or any other text outside the JSON.

  found      bool  — true only if you know the song with high confidence. \
                     If you're guessing or partially sure, return false. \
                     Don't fabricate chord sets.
  key        str   — concise key name, e.g. "G", "Bm", "F# minor", "Eb".
  chord_set  array — unique chords used, as a SET (no duplicates, order is \
                     irrelevant). See chord notation rules below.
  notes      str   — one short line about anything unusual: capo, modal \
                     interchange, key changes, alternate tunings, common \
                     simplifications. Empty string if nothing notable.

Chord notation rules
--------------------
1. Use ROOT + QUALITY only. No slash-bass, no inversions, no voicings.
   - "G", "Em", "F#m7", "Cmaj7", "A9", "Bbm", "D7sus4" — yes
   - "C/E", "Em/G", "G(add9)/B" — strip to "C", "Em", "Gadd9"
2. Quality forms to use:
   - major triad: bare root ("G", "C")
   - minor triad: lowercase "m" ("Em", "F#m")
   - 7ths: "G7" (dominant), "Cmaj7", "Em7"
   - extensions: "A9", "Cmaj9", "F#m11", "D13"
   - sus: "Asus2", "Dsus4"
   - dim/aug: "Bdim", "Caug"
3. Enharmonics: pick the spelling conventional in the song's key signature.
   Flat keys (F, Bb, Eb, Ab, Db) → use flats: Bb not A#, Eb not D#.
   Sharp keys (G, D, A, E, B, F#) → use sharps: F# not Gb, C# not Db.
4. Don't list every embellishment — return the structural chord set. If the \
   song uses Cadd9 throughout where someone might write C, list "Cadd9" \
   (it's distinctive). If the chart uses C with occasional add9 ornament, \
   just list "C".
5. Drop duplicates. The chord_set is a SET, not a sequence.

When NOT to answer (set found: false)
-------------------------------------
- Songs you don't recognize, even if title is suggestive
- Original/independent music with no published canonical version
- Live or demo versions that differ substantially from the studio cut
- Cover versions where the title doesn't disambiguate

Examples
--------
Song: "Hotel California" by Eagles
{"found": true, "key": "Bm", "chord_set": ["Bm", "F#7", "A", "E", "G", "D", "Em"], "notes": "Acoustic guitar uses capo VII; chords above are concert pitch"}

Song: "Black Cow" by Steely Dan
{"found": true, "key": "A", "chord_set": ["A9", "D", "C9", "E", "Bm7", "Amaj7"], "notes": "Stays mostly on the A9 vamp; descending bassline creates apparent slash chords that some charts label as separate roots"}

Song: "Your Wildest Dreams" by Moody Blues
{"found": true, "key": "G", "chord_set": ["G", "C", "D", "Em", "Bm", "F"], "notes": "F is bVII borrowed from G mixolydian; sometimes notated as G major with a chromatic passing chord"}

Song: "Every Rose Has Its Thorn" by Poison
{"found": true, "key": "G", "chord_set": ["G", "Cadd9", "D", "Em"], "notes": "Recorded with capo IV (effective key B); chords above are open-position shapes"}

Song: "Aja" by Steely Dan
{"found": true, "key": "Bm", "chord_set": ["Bm7", "Cm9", "Gm9", "Am7", "F#m9", "Em9"], "notes": "Modal jazz with extensive minor 9th voicings; bridge has additional changes not in main vamp"}

Song: "Highway to Hell" by AC/DC
{"found": true, "key": "A", "chord_set": ["A", "D", "G", "F#m", "Bm", "E"], "notes": "Classic I-IV-bVII-iii cadence; bVII (G) is borrowed from A mixolydian"}

Song: "Hells Bells" by AC/DC
{"found": true, "key": "Am", "chord_set": ["Am", "G", "D", "C", "E"], "notes": "A natural minor with picardy V (E major as dominant)"}

Song: "Free Fallin'" by Tom Petty
{"found": true, "key": "F", "chord_set": ["F", "Bb", "C"], "notes": "Recorded in F (capo III over D-G-A on guitar); three-chord I-IV-V"}

Song: "Free Fallin'" by Tom Petty And The Heartbreakers
{"found": true, "key": "F", "chord_set": ["F", "Bb", "C"], "notes": "Same arrangement as the studio single — capo III on guitar"}

Song: "Peg" by Steely Dan
{"found": true, "key": "G", "chord_set": ["G", "Cmaj7", "D7", "Bm7", "Em7", "F#m7"], "notes": "Extended jazz harmony; some charts include additional passing chords for the chorus turnaround"}

Song: "Rikki Don't Lose That Number" by Steely Dan
{"found": true, "key": "A", "chord_set": ["A", "E", "D", "F#m", "Bm", "G"], "notes": "Verse cycles A-E-D, chorus adds F#m and Bm; opening 4-bar vamp uses Em-A approach"}

Song: "Hotel California" by Some Indie Band
{"found": false, "key": "", "chord_set": [], "notes": "Title matches Eagles song but artist suggests a cover or different track — can't confirm canonical chords"}
"""


_QUALITY_ALIASES = [
    ("major7", "maj7"), ("major9", "maj9"), ("major11", "maj11"), ("major13", "maj13"),
    ("minor7", "m7"), ("minor9", "m9"), ("minor11", "m11"), ("minor13", "m13"),
    ("minor", "m"), ("major", ""),
    ("min7", "m7"), ("min9", "m9"), ("min11", "m11"), ("min13", "m13"),
    ("min", "m"), ("maj7", "MAJ7"), ("maj9", "MAJ9"), ("maj", ""),
    ("Δ7", "MAJ7"), ("△7", "MAJ7"),
]

_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def normalize_chord(c: str) -> str:
    """Normalize a chord label for set comparison.

    Examples:
      "C/E"     -> "C"
      "Bbm7"    -> "A#m7"
      "Cmaj7"   -> "CM7"
      "Eminor"  -> "Em"
      "G(add9)" -> "Gadd9"
    """
    if not c:
        return ""
    c = c.strip()
    c = c.split("/", 1)[0]
    c = c.replace("(", "").replace(")", "")
    m = re.match(r"^([A-Ga-g])([#b]?)(.*)$", c)
    if not m:
        return c.lower()
    root = m.group(1).upper() + m.group(2)
    root = _FLAT_TO_SHARP.get(root, root)
    quality = m.group(3)
    for old, new in _QUALITY_ALIASES:
        quality = quality.replace(old, new)
    quality = quality.replace("MAJ", "maj")  # restore protected maj7/maj9 markers
    return root + quality


def normalize_chord_v2(c: str) -> str:
    """V2 normalizer: preserves slash bass.

    Used by the corrector's V2 prompt path (ANTHROPIC_CORRECTION_V2_PROMPT=true)
    where slash chords must survive the canonical_set comparison so they can be
    re-applied to the bar grid. Splits on "/", normalizes each side via v1
    rules, then rejoins.

    Examples:
      "C/E"     -> "C/E"     (vs v1: "C")
      "Em/C#"   -> "Em/C#"   (vs v1: "Em")
      "Bbm/F"   -> "A#m/F"   (root normalized, bass enharmonic-normalized too)
      "Cmaj7"   -> "CM7"     (no slash, behaves like v1)
    """
    if not c:
        return ""
    s = c.strip()
    if "/" in s:
        head, bass = s.split("/", 1)
        head_n = normalize_chord(head)
        bass_n = normalize_chord(bass)
        if head_n and bass_n:
            return f"{head_n}/{bass_n}"
        return head_n
    return normalize_chord(s)


def diff_sets(detected: Iterable[str], canonical: Iterable[str]) -> dict:
    d = {normalize_chord(c) for c in detected if c} - {""}
    g = {normalize_chord(c) for c in canonical if c} - {""}
    tp = d & g
    fp = d - g
    fn = g - d
    precision = len(tp) / len(d) if d else 0.0
    recall = len(tp) / len(g) if g else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision, "recall": recall, "f1": f1,
        "tp": sorted(tp), "fp": sorted(fp), "fn": sorted(fn),
        "detected_norm": sorted(d), "canonical_norm": sorted(g),
    }


_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "key": {"type": "string"},
        "chord_set": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["found", "key", "chord_set", "notes"],
    "additionalProperties": False,
}


def _api_key_from_keychain() -> str | None:
    try:
        r = subprocess.run(
            ["security", "find-generic-password", "-s", "anthropic-api-key", "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            key = r.stdout.strip()
            return key if key.startswith("sk-") else None
    except Exception:
        pass
    return None


def _load_oracle_response(client: anthropic.Anthropic, model: str, title: str, artist: str):
    user_msg = (
        f'Song: "{title}"' + (f" by {artist}" if artist else "") + "\n\n"
        "Return ONLY the JSON object. No code fences, no prose."
    )
    return client.messages.create(
        model=model,
        max_tokens=600,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_msg}],
        output_config={"format": {"type": "json_schema", "schema": _RESPONSE_SCHEMA}},
    )


def _extract_json(text: str) -> dict | None:
    text = text.strip()
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", default="/tmp/kk_jobs/all",
                   help="Directory of chord_chart.json files (one per job/song)")
    p.add_argument("--model", default="claude-opus-4-7")
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of songs (0 = all)")
    p.add_argument("--json-out", default=None,
                   help="Write per-song results as JSONL to this path")
    args = p.parse_args(argv)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        key = _api_key_from_keychain()
        if key:
            os.environ["ANTHROPIC_API_KEY"] = key
        else:
            print(
                "ERROR: ANTHROPIC_API_KEY not found in env or macOS keychain.\n"
                "  Either: export ANTHROPIC_API_KEY=sk-ant-...\n"
                "  Or:     security add-generic-password -s anthropic-api-key "
                "-a $USER -w sk-ant-...",
                file=sys.stderr,
            )
            return 2

    files = sorted(glob.glob(os.path.join(args.dir, "*.json")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"No *.json files in {args.dir}", file=sys.stderr)
        return 1

    client = anthropic.Anthropic()

    header = f'{"job":<10} {"title":<30} {"artist":<22} {"P":>5} {"R":>5} {"F1":>5}  diff'
    print(header)
    print("-" * len(header))

    rows = []
    cache_announced = False
    out_fh = open(args.json_out, "w") if args.json_out else None

    for f in files:
        jid = os.path.splitext(os.path.basename(f))[0]
        try:
            data = json.load(open(f))
        except Exception as e:
            print(f"{jid[:8]:<10} <load error: {e}>")
            continue
        title = (data.get("title") or "").strip()
        artist = (data.get("artist") or "").strip()
        detected = data.get("chords_used") or []
        if not title:
            print(f"{jid[:8]:<10} <no title in chord_chart>")
            continue
        if not detected:
            print(f"{jid[:8]:<10} {title[:29]:<30} {artist[:21]:<22} <no chords_used in chart>")
            continue

        try:
            resp = _load_oracle_response(client, args.model, title, artist)
        except anthropic.APIError as e:
            print(f"{jid[:8]:<10} {title[:29]:<30} <API error: {e.__class__.__name__}: {e}>")
            continue

        if not cache_announced:
            u = resp.usage
            cw = getattr(u, "cache_creation_input_tokens", 0) or 0
            cr = getattr(u, "cache_read_input_tokens", 0) or 0
            print(
                f"# first-call usage: input={u.input_tokens} cache_write={cw} "
                f"cache_read={cr} output={u.output_tokens}",
                file=sys.stderr,
            )
            cache_announced = True

        text = next((b.text for b in resp.content if b.type == "text"), "")
        obj = _extract_json(text)
        if obj is None:
            print(f"{jid[:8]:<10} {title[:29]:<30} {artist[:21]:<22} <oracle returned non-JSON: {text[:80]}>")
            continue
        if not obj.get("found"):
            print(f"{jid[:8]:<10} {title[:29]:<30} {artist[:21]:<22} <oracle declined: {obj.get('notes','')[:60]}>")
            continue

        canonical = obj.get("chord_set") or []
        result = diff_sets(detected, canonical)
        rows.append({"jid": jid, "title": title, "artist": artist,
                     "oracle_key": obj.get("key", ""),
                     "oracle_notes": obj.get("notes", ""),
                     "detected": sorted(set(detected)),
                     "canonical": sorted(set(canonical)),
                     **result})

        diff_str = (
            f"hit={','.join(result['tp']) or '-'} | "
            f"extra={','.join(result['fp']) or '-'} | "
            f"miss={','.join(result['fn']) or '-'}"
        )
        print(f"{jid[:8]:<10} {title[:29]:<30} {artist[:21]:<22} "
              f"{result['precision']:>5.2f} {result['recall']:>5.2f} {result['f1']:>5.2f}  {diff_str}")

        if out_fh:
            out_fh.write(json.dumps(rows[-1]) + "\n")

    if out_fh:
        out_fh.close()

    print()
    if rows:
        n = len(rows)
        avg_p = sum(r["precision"] for r in rows) / n
        avg_r = sum(r["recall"] for r in rows) / n
        avg_f = sum(r["f1"] for r in rows) / n
        skipped = len(files) - n
        print(f"Scored {n}/{len(files)} songs (skipped {skipped}). "
              f"Avg P={avg_p:.2f} R={avg_r:.2f} F1={avg_f:.2f}")
    else:
        print("No songs scored.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
