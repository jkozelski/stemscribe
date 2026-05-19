# Anthropic chord-corrector V2 prompt — proposal

**Status:** Drafted, not deployed. Awaiting scoreboard data + Jeff review.

## Why

The current corrector prompt (`backend/audit/llm_oracle.py:29-118`) was
written to **score** detector output, then reused as the **corrector**'s
prompt. The scoring prompt explicitly says:

- **line 55:** "Use ROOT + QUALITY only. No slash-bass, no inversions, no voicings." `C/E → C`, `Em/G → Em`
- **line 68:** "Don't list every embellishment — return the structural chord set." `C with occasional add9 → just C`

That's perfect for scoring against a noisy detector. It's terrible as a
corrector prompt because it actively **flattens** the chord vocabulary.

The reranker path (`chord_corrector_anthropic.py:644`) is even more
restrictive: "DO NOT add extensions (7, maj7, sus, etc.) — the candidate
vocabulary is maj/min triads only."

External validation:
- Wu et al. 2025 (arXiv:2509.18700) uses **301 chord classes** including
  inverted triads, 7ths, 9ths, 11ths, 13ths, slash chords. They preserve
  Harte shorthand through all stages of LLM correction.
- ChordMini, Party4Bread/noten, and every published LLM-music system newer
  than 2025 preserves slash + extension chords.

StemScriber is the outlier for compressing vocabulary at the LLM stage.

## What changes (three coordinated edits)

### 1. New prompt in `chord_corrector_anthropic.py`

Add a module-level constant `RICH_CORRECTOR_SYSTEM_PROMPT`:

```python
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
```

### 2. New normalizer `normalize_chord_v2` in `llm_oracle.py`

Currently `normalize_chord` strips slash bass at line 146:
```python
c = c.split("/", 1)[0]  # ← drops everything after /
```

New version preserves it:
```python
def normalize_chord_v2(c: str) -> str:
    """Like normalize_chord but preserves slash-bass.

    Examples:
      "C/E"     -> "C/E"     (was "C")
      "Bbm7"    -> "A#m7"
      "Em/C#"   -> "Em/C#"
      "Cmaj7"   -> "CM7"
    """
    if not c: return ""
    c = c.strip()
    if "/" in c:
        head, bass = c.split("/", 1)
        return normalize_chord(head) + "/" + normalize_chord(bass)
    return normalize_chord(c)  # delegates to existing v1
```

### 3. `_replace_in_bar_grid` in `chord_corrector_anthropic.py:441`

Currently only swaps when root matches (it can replace `Em` with `E` but
not `Em` with `Em7`). New logic: if V2 mode is on and Claude's chord
shares the same ROOT, apply Claude's full label (`Em7`, `Em/C#`, etc.)
over the librosa triad.

### Gate: `ANTHROPIC_CORRECTION_V2_PROMPT` env flag

All three changes hide behind one flag. Default off. Set to `true` to
enable. Allows clean A/B with current production behavior.

## Regression risks

- **Aja must stay 226/226.** Current corrector is doing GOOD work on jazz —
  flattening hallucinations. V2's "produce richer output" could backfire
  if Claude hallucinates richness Aja doesn't actually have.
- **Hells Bells stays 1.00.** Power-chord rock doesn't need 7ths; V2 must
  recognize that absence of evidence = bare triad.
- **The `qf_threshold` may need bumping from 2 to ~4.** A richer
  chord_set means more legitimate same-root flips (Em + Em7 + Emmaj7 all
  share root) — current threshold could over-trigger the qflip gate.

## Expected lift

- **Tom Petty** (current F1 = 0.62): could jump to 0.85+ — Claude already
  knows this song, just needs prompt permission to emit Emmaj7/Em/C#/Asus4
- **Hotel California** (current F1 = 0.92): minor lift — was already strong
- **Songs Claude doesn't recognize** (`found: false`): no change
- **Average F1 across cohort**: estimated +0.10 to +0.15 lift on
  recognizable songs, near-zero on unrecognized

## Decision gate

**If 14-song scoreboard average ≥ 0.80:** V2 becomes post-launch work.
**If average is 0.70-0.79:** V2 is the highest-leverage pre-launch fix,
ship behind flag, A/B for one week, flip on if avg improves.
**If average < 0.70:** V2 is urgent. Ship aggressively.

## Files this touches

- `backend/processing/chord_corrector_anthropic.py` — new prompt constant + gate logic + bar-grid replacement update
- `backend/audit/llm_oracle.py` — new `normalize_chord_v2` function (keep v1 unchanged)
- `backend/tests/` — add regression tests against Aja, Hells Bells, Hotel California fixtures

## Estimated effort

4-6 hours coding + 30 min A/B testing + regression check.
