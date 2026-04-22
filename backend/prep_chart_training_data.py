"""
Prepare training data for chart formatter QLoRA fine-tuning.

Converts 15,437 chord library JSONs into input/output training pairs.
- Input: synthetic chord_events + word_timestamps (simulating what the listening model produces)
- Output: formatted chart in three formats (JSON, slash notation, ChordPro)

Augments with transpositions: each chart x 12 keys = ~185K examples.
Splits out Kozelski charts as validation set (NEVER trained on).

Run on CPU:
    cd ~/stemscribe/backend && ../venv311/bin/python prep_chart_training_data.py
"""

import json
import os
import random
import hashlib
from pathlib import Path
from typing import Optional

# ─── Constants ───────────────────────────────────────────────────────────────

CHORD_LIBRARY = Path(__file__).parent / "chord_library"
OUTPUT_DIR = Path(__file__).parent / "training_data" / "chart_formatter"
KOZELSKI_DIR = CHORD_LIBRARY / "kozelski"

# Chromatic scale for transposition
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ENHARMONIC = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B", "E#": "F", "B#": "C",
}
# Reverse for natural output
PREFER_FLAT = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}

SECTION_TYPES = {
    "intro", "verse", "pre-chorus", "chorus", "bridge", "solo",
    "outro", "interlude", "instrumental", "guitar solo", "keyboard solo",
    "refrain", "coda", "break", "hook",
}

# Format distribution per training plan: 60% JSON, 25% slash, 15% ChordPro
FORMAT_WEIGHTS = {"json": 0.60, "slash": 0.25, "chordpro": 0.15}


# ─── Chord Transposition ────────────────────────────────────────────────────

def parse_chord_root(chord: str) -> tuple[str, str]:
    """Extract root note and suffix from a chord symbol. Returns (root, suffix)."""
    if not chord:
        return ("", "")

    # Handle slash chords: take everything before the slash for root parsing
    slash_bass = ""
    if "/" in chord:
        parts = chord.split("/", 1)
        # Check if it's a slash chord (bass note) vs chord quality like m7b5
        if parts[1] and len(parts[1]) <= 3 and parts[1][0].isupper():
            slash_bass = "/" + parts[1]
            chord = parts[0]

    if len(chord) >= 2 and chord[1] in ("#", "b"):
        root = chord[:2]
        suffix = chord[2:]
    else:
        root = chord[:1]
        suffix = chord[1:]

    return (root, suffix + slash_bass)


def transpose_note(note: str, semitones: int) -> str:
    """Transpose a single note by semitones."""
    # Normalize enharmonics
    normalized = ENHARMONIC.get(note, note)
    if normalized not in NOTES:
        return note  # Can't transpose unknown notes
    idx = NOTES.index(normalized)
    new_idx = (idx + semitones) % 12
    return NOTES[new_idx]


def transpose_chord(chord: str, semitones: int) -> str:
    """Transpose a full chord symbol by semitones."""
    if not chord or chord in ("N.C.", "NC", "N/C", "x", "X", "-", "/"):
        return chord

    root, suffix = parse_chord_root(chord)
    if not root:
        return chord

    new_root = transpose_note(root, semitones)

    # Handle slash bass note
    if "/" in suffix:
        parts = suffix.rsplit("/", 1)
        bass = parts[1]
        new_bass = transpose_note(bass, semitones)
        suffix = parts[0] + "/" + new_bass

    return new_root + suffix


def transpose_chord_line(chord_line: str, semitones: int) -> str:
    """Transpose all chords in a space-separated chord line."""
    if not chord_line:
        return chord_line or ""
    chords = chord_line.split()
    return "  ".join(transpose_chord(c, semitones) for c in chords)


def transpose_chart(chart: dict, semitones: int) -> dict:
    """Transpose an entire chart by semitones. Returns a new dict."""
    if semitones == 0:
        return chart

    new_chart = dict(chart)

    # Transpose key
    if chart.get("key"):
        root, suffix = parse_chord_root(chart["key"])
        new_chart["key"] = transpose_note(root, semitones) + suffix

    # Transpose chords_used
    if chart.get("chords_used"):
        new_chart["chords_used"] = [transpose_chord(c, semitones) for c in chart["chords_used"]]

    # Transpose sections
    new_sections = []
    for section in chart.get("sections", []):
        new_section = dict(section)
        new_lines = []
        for line in section.get("lines", []):
            new_line = dict(line)
            if line.get("chords"):
                new_line["chords"] = transpose_chord_line(line["chords"], semitones) or line["chords"]
            new_lines.append(new_line)
        new_section["lines"] = new_lines
        new_sections.append(new_section)
    new_chart["sections"] = new_sections

    return new_chart


# ─── Synthetic Timestamp Generation ─────────────────────────────────────────

def generate_synthetic_input(chart: dict, tempo: int = None) -> dict:
    """
    Generate synthetic chord_events and word_timestamps from a chart JSON,
    simulating what the listening model would produce.
    """
    if tempo is None:
        tempo = chart.get("tempo", random.choice([80, 90, 100, 110, 120, 130, 140]))

    beat_duration = 60.0 / tempo  # seconds per beat
    bar_duration = beat_duration * 4  # assume 4/4

    time_sig = chart.get("time_signature", [4, 4])
    if isinstance(time_sig, list) and len(time_sig) == 2:
        beats_per_bar = time_sig[0]
    else:
        beats_per_bar = 4
    bar_duration = beat_duration * beats_per_bar

    chord_events = []
    word_timestamps = []
    beat_times = []
    downbeat_times = []

    current_time = 0.0

    for section in chart.get("sections", []):
        for line in section.get("lines", []):
            chords_str = line.get("chords", "")
            if not chords_str:
                continue

            chords = chords_str.split()
            num_chords = len(chords)

            if num_chords == 0:
                continue

            # Determine bars for this line
            # Heuristic: if chords <= 4, one bar per chord; if > 4, distribute
            if num_chords <= beats_per_bar:
                # Each chord gets equal beats within one bar
                beats_per_chord = max(1, beats_per_bar // num_chords)
                total_beats = beats_per_bar
            else:
                # Multiple bars: each chord gets one beat, rounded up to full bars
                beats_per_chord = 1
                total_beats = num_chords
                # Round up to full bars
                bars_needed = (total_beats + beats_per_bar - 1) // beats_per_bar
                total_beats = bars_needed * beats_per_bar

            line_start = current_time

            # Generate chord events
            for i, chord in enumerate(chords):
                chord_start = current_time + i * beats_per_chord * beat_duration
                chord_dur = beats_per_chord * beat_duration
                # Add slight timing noise (real detectors aren't perfect)
                noise = random.uniform(-0.05, 0.05)
                conf = random.uniform(0.75, 0.98)

                chord_events.append({
                    "time": round(chord_start + noise, 3),
                    "duration": round(chord_dur, 3),
                    "chord": chord,
                    "confidence": round(conf, 2),
                })

            # Generate beat times for this line
            for b in range(total_beats):
                bt = line_start + b * beat_duration
                beat_times.append(round(bt, 3))
                if b % beats_per_bar == 0:
                    downbeat_times.append(round(bt, 3))

            # Generate word timestamps from lyrics
            lyrics = line.get("lyrics")
            if lyrics:
                words = lyrics.split()
                if words:
                    line_dur = total_beats * beat_duration
                    # Spread words across 70-90% of the line duration
                    word_spread = random.uniform(0.7, 0.9)
                    word_start_offset = random.uniform(0.0, line_dur * 0.1)
                    word_dur = (line_dur * word_spread) / len(words)

                    for j, word in enumerate(words):
                        ws = line_start + word_start_offset + j * word_dur
                        we = ws + word_dur * random.uniform(0.6, 0.95)
                        word_timestamps.append({
                            "word": word,
                            "start": round(ws, 3),
                            "end": round(we, 3),
                        })

            current_time += total_beats * beat_duration

    return {
        "title": chart.get("title", "Unknown"),
        "artist": chart.get("artist", "Unknown"),
        "tempo_bpm": tempo,
        "time_signature": time_sig if isinstance(time_sig, list) else [4, 4],
        "key": chart.get("key", "C"),
        "duration_seconds": round(current_time, 2),
        "chord_events": chord_events,
        "word_timestamps": word_timestamps,
        "beat_times": beat_times[:50],  # Truncate for prompt size
        "downbeat_times": downbeat_times[:20],
    }


# ─── Output Format Converters ───────────────────────────────────────────────

def chart_to_slash(chart: dict) -> str:
    """Convert chart JSON to slash notation (Jeff's handwritten style)."""
    lines_out = []

    title = chart.get("title", "Unknown")
    artist = chart.get("artist", "Unknown")
    key = chart.get("key", "")
    tempo = chart.get("tempo", chart.get("tempo_bpm", ""))

    lines_out.append(f"{title} - {artist}")
    meta_parts = []
    if key:
        meta_parts.append(f"Key: {key}")
    if tempo:
        meta_parts.append(f"Tempo: {tempo} BPM")
    meta_parts.append("Time: 4/4")
    lines_out.append("  ".join(meta_parts))
    lines_out.append("")

    for section in chart.get("sections", []):
        name = section.get("name", "Section")
        section_lines = section.get("lines", [])
        repeat = section.get("repeat", 0)

        # Detect repeated lines within section
        chord_seqs = [l.get("chords", "") for l in section_lines]
        all_same = len(set(chord_seqs)) == 1 and len(chord_seqs) > 1

        header = name
        if all_same and len(chord_seqs) > 1:
            header += f" (x{len(chord_seqs)})"
        elif repeat and repeat > 1:
            header += f" (x{repeat})"

        lines_out.append(header)

        if all_same and len(chord_seqs) > 1:
            # Write once with repeat signs
            line = section_lines[0]
            chords = (line.get("chords") or "").split()
            chord_row = "||: " + "  -  ".join(f"{c:8s}" for c in chords).rstrip() + "  :||"
            slash_row = "    " + "  ".join("/ / / /" for _ in chords)
            lines_out.append(chord_row)
            lines_out.append(slash_row)
            # Add first lyrics line if present
            if line.get("lyrics"):
                lines_out.append("    " + line["lyrics"])
        else:
            use_repeats = repeat and repeat > 1
            for line in section_lines:
                chords = (line.get("chords") or "").split()
                if use_repeats:
                    chord_row = "||: " + "  -  ".join(f"{c:8s}" for c in chords).rstrip() + "  :||"
                else:
                    chord_row = "  " + "  -  ".join(f"{c:8s}" for c in chords).rstrip()
                slash_row = "    " + "  ".join("/ / / /" for _ in chords)
                lines_out.append(chord_row)
                lines_out.append(slash_row)
                if line.get("lyrics"):
                    lines_out.append("    " + line["lyrics"])

        lines_out.append("")

    return "\n".join(lines_out).rstrip()


def chart_to_chordpro(chart: dict) -> str:
    """Convert chart JSON to ChordPro format."""
    lines_out = []

    lines_out.append(f"{{title: {chart.get('title', 'Unknown')}}}")
    lines_out.append(f"{{artist: {chart.get('artist', 'Unknown')}}}")
    if chart.get("key"):
        lines_out.append(f"{{key: {chart['key']}}}")
    tempo = chart.get("tempo", chart.get("tempo_bpm"))
    if tempo:
        lines_out.append(f"{{tempo: {tempo}}}")
    lines_out.append("{time: 4/4}")
    lines_out.append("")

    for section in chart.get("sections", []):
        name = section.get("name", "Section")
        section_type = name.lower().split()[0]
        if section_type in ("verse", "chorus", "bridge", "intro", "outro", "solo"):
            directive = section_type
        else:
            directive = "verse"

        lines_out.append(f"{{start_of_{directive}: {name}}}")

        for line in section.get("lines", []):
            chords_str = line.get("chords") or ""
            chords = chords_str.split()
            lyrics = line.get("lyrics")

            if lyrics and chords:
                # Interleave chords with lyrics
                words = lyrics.split()
                result = ""
                words_per_chord = max(1, len(words) // max(1, len(chords)))

                for ci, chord in enumerate(chords):
                    word_start = ci * words_per_chord
                    word_end = word_start + words_per_chord if ci < len(chords) - 1 else len(words)
                    segment = " ".join(words[word_start:word_end])
                    if segment:
                        result += f"[{chord}]{segment} "
                    else:
                        result += f"[{chord}] "

                lines_out.append(result.rstrip())
            elif chords:
                # No lyrics - slash notation
                slash_line = " | ".join(f"[{c}] / / / /" for c in chords)
                lines_out.append(slash_line)

        lines_out.append(f"{{end_of_{directive}}}")
        lines_out.append("")

    return "\n".join(lines_out).rstrip()


# ─── Prompt Template ────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a professional chord chart formatter. Given raw chord detection "
    "data with timestamps, format a clean, readable chord chart."
)

def build_training_example(chart: dict, fmt: str, synthetic_input: dict) -> dict:
    """Build a single training example in chat format."""

    # Truncate chord_events and word_timestamps for prompt size
    ce = synthetic_input["chord_events"][:100]  # max 100 chord events
    wt = synthetic_input["word_timestamps"][:200]  # max 200 words

    format_names = {"json": "StemScriber JSON", "slash": "slash notation", "chordpro": "ChordPro"}

    user_msg = f"""Format this as a {format_names[fmt]} chord chart:

Title: {synthetic_input['title']}
Artist: {synthetic_input['artist']}
Key: {synthetic_input.get('key', 'C')}
Tempo: {synthetic_input['tempo_bpm']} BPM
Time Signature: {synthetic_input['time_signature'][0]}/{synthetic_input['time_signature'][1]}

Chord events:
{json.dumps(ce, separators=(',', ':'))}

Lyrics with timing:
{json.dumps(wt, separators=(',', ':'))}"""

    # Generate output
    if fmt == "json":
        output = json.dumps(chart, indent=2, ensure_ascii=False)
    elif fmt == "slash":
        output = chart_to_slash(chart)
    elif fmt == "chordpro":
        output = chart_to_chordpro(chart)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": output},
        ]
    }


# ─── Main Pipeline ──────────────────────────────────────────────────────────

def load_all_charts() -> tuple[list[dict], list[dict]]:
    """Load all charts, splitting Kozelski into validation set."""
    train_charts = []
    val_charts = []

    for artist_dir in sorted(CHORD_LIBRARY.iterdir()):
        if not artist_dir.is_dir():
            continue

        is_kozelski = artist_dir.name == "kozelski"

        for chart_file in sorted(artist_dir.glob("*.json")):
            try:
                with open(chart_file) as f:
                    chart = json.load(f)

                # Validate minimum structure
                if not chart.get("sections"):
                    continue
                if not any(line.get("chords") for s in chart["sections"] for line in s.get("lines", [])):
                    continue

                chart["_source_file"] = str(chart_file)

                if is_kozelski:
                    val_charts.append(chart)
                else:
                    train_charts.append(chart)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  Skipping {chart_file}: {e}")

    return train_charts, val_charts


def pick_format() -> str:
    """Pick output format according to weighted distribution."""
    r = random.random()
    if r < FORMAT_WEIGHTS["json"]:
        return "json"
    elif r < FORMAT_WEIGHTS["json"] + FORMAT_WEIGHTS["slash"]:
        return "slash"
    else:
        return "chordpro"


def main():
    random.seed(42)

    print("=" * 60)
    print("StemScriber Chart Formatter — Training Data Preparation")
    print("=" * 60)

    # Load charts
    print("\n[1/4] Loading chord library...")
    train_charts, val_charts = load_all_charts()
    print(f"  Training charts: {len(train_charts)}")
    print(f"  Validation charts (Kozelski): {len(val_charts)}")

    # Generate training examples with transposition augmentation
    print("\n[2/4] Generating training examples with 12-key augmentation...")
    train_examples = []

    for i, chart in enumerate(train_charts):
        if i % 1000 == 0:
            print(f"  Processing chart {i}/{len(train_charts)} ({len(train_examples)} examples so far)")

        for semitones in range(12):
            transposed = transpose_chart(chart, semitones)
            tempo = transposed.get("tempo", random.choice([80, 100, 120, 140]))
            synthetic_input = generate_synthetic_input(transposed, tempo)
            fmt = pick_format()
            example = build_training_example(transposed, fmt, synthetic_input)
            train_examples.append(example)

    print(f"  Total training examples: {len(train_examples)}")

    # Generate validation examples (all 3 formats, original key only)
    print("\n[3/4] Generating validation examples...")
    val_examples = []

    for chart in val_charts:
        synthetic_input = generate_synthetic_input(chart)
        for fmt in ["json", "slash", "chordpro"]:
            example = build_training_example(chart, fmt, synthetic_input)
            example["_song"] = chart.get("title", "Unknown")
            example["_format"] = fmt
            val_examples.append(example)

    print(f"  Total validation examples: {len(val_examples)}")

    # Shuffle training data
    random.shuffle(train_examples)

    # Save to JSONL files
    print("\n[4/4] Saving to disk...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"

    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Stats
    train_size_mb = train_path.stat().st_size / (1024 * 1024)
    val_size_mb = val_path.stat().st_size / (1024 * 1024)

    print(f"\n{'=' * 60}")
    print(f"DONE!")
    print(f"  Training: {train_path} ({len(train_examples)} examples, {train_size_mb:.1f} MB)")
    print(f"  Validation: {val_path} ({len(val_examples)} examples, {val_size_mb:.2f} MB)")
    print(f"{'=' * 60}")

    # Print format distribution
    fmt_counts = {"json": 0, "slash": 0, "chordpro": 0}
    for ex in train_examples[:10000]:  # Sample first 10K
        # Detect format from output
        output = ex["messages"][2]["content"]
        if output.startswith("{"):
            fmt_counts["json"] += 1
        elif output.startswith("{title:") or "{title:" in output[:50]:
            fmt_counts["chordpro"] += 1
        else:
            # Check user message for format hint
            user = ex["messages"][1]["content"]
            if "slash notation" in user:
                fmt_counts["slash"] += 1
            elif "ChordPro" in user:
                fmt_counts["chordpro"] += 1
            else:
                fmt_counts["json"] += 1

    total_sampled = sum(fmt_counts.values())
    if total_sampled > 0:
        print(f"\n  Format distribution (first 10K):")
        for fmt, count in fmt_counts.items():
            print(f"    {fmt}: {count} ({100*count/total_sampled:.1f}%)")


if __name__ == "__main__":
    main()
