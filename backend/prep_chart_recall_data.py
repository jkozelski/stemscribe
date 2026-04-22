"""
StemScriber — Chord Recall Training Data Preparation (Stage 2)

Generates ~107,000 recall training examples from 15,437 chord chart JSONs.
This is CPU-only data prep ($0 cost). No GPU training is triggered.

For songs WITH lyrics (6 variants each):
  1. Full lyrics — complete lyrics as input, correct chart as output
  2. Partial lyrics (verse only) — just verse 1, full chart as output
  3. Partial lyrics (chorus only) — just chorus lyrics, full chart as output
  4. Whisper error sim 1 — ~10% of words misspelled/garbled
  5. Whisper error sim 2 — ~20% of words missing
  6. Metadata hint — title + artist + first line of lyrics, full chart as output

For songs WITHOUT lyrics (instrumental, 2 variants each):
  1. Title + artist + detected notes
  2. Title + artist only

Also generates 5,000 "unknown song" negative examples.

Validation set: All Kozelski songs (excluded from training).

Usage:
    cd ~/stemscribe/backend
    ../venv311/bin/python prep_chart_recall_data.py
"""

import json
import os
import glob
import random
import re
import hashlib
from pathlib import Path

random.seed(42)

# ─── Paths ───────────────────────────────────────────────────────────────────

CHORD_LIBRARY = Path(__file__).parent / "chord_library"
OUTPUT_DIR = Path(__file__).parent / "training_data" / "chart_recall"
TRAIN_PATH = OUTPUT_DIR / "recall_train.jsonl"
VAL_PATH = OUTPUT_DIR / "recall_val.jsonl"

# ─── Constants ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a chord chart expert with encyclopedic knowledge of 15,000+ songs. "
    "Given lyrics and optional detected notes, identify the song and recall the "
    "correct chord chart from your training. If the song is not recognized, say so."
)

# Phonetic neighbor substitutions for Whisper error simulation
PHONETIC_SUBS = {
    "night": ["knight", "nite", "nigh"],
    "right": ["rite", "write", "wright"],
    "boys": ["boy's", "bois"],
    "hear": ["here", "ear"],
    "there": ["their", "they're"],
    "your": ["you're", "yer"],
    "you": ["ya", "u"],
    "to": ["too", "2"],
    "for": ["4", "fer"],
    "the": ["da", "tha", "de"],
    "and": ["an", "n"],
    "love": ["luv"],
    "know": ["no", "kno"],
    "one": ["won", "1"],
    "see": ["sea"],
    "be": ["bee"],
    "heart": ["hart"],
    "time": ["tyme"],
    "down": ["doun"],
    "another": ["anudder", "anuther"],
    "going": ["goin", "gonna"],
    "running": ["runnin"],
    "coming": ["comin"],
    "nothing": ["nothin", "nuthin"],
    "something": ["somethin", "sumthin"],
    "everything": ["everythin"],
    "want": ["wanna", "wan"],
    "because": ["cuz", "cause", "coz"],
    "with": ["wit", "wif"],
    "what": ["wut", "wat"],
    "just": ["jus", "juss"],
    "that": ["dat"],
    "this": ["dis"],
    "them": ["em"],
    "about": ["bout"],
    "little": ["lil", "litle"],
    "baby": ["babe", "babey"],
    "never": ["neva"],
    "ever": ["eva"],
    "together": ["togetha"],
    "where": ["were", "wear"],
    "through": ["thru", "threw"],
    "light": ["lite"],
    "away": ["aweigh"],
    "world": ["worl"],
    "girl": ["grl", "gurl"],
    "would": ["wud"],
    "could": ["cud"],
    "should": ["shud"],
    "doing": ["doin"],
    "having": ["havin"],
    "being": ["bein"],
    "feeling": ["feelin"],
    "looking": ["lookin"],
    "talking": ["talkin"],
    "walking": ["walkin"],
    "thinking": ["thinkin"],
}

FILLER_WORDS = ["uh", "oh", "um", "yeah", "ah", "hmm", "like"]

# ─── Helpers ─────────────────────────────────────────────────────────────────


def extract_root(chord_name: str) -> str:
    """Extract root note from chord symbol: 'Am7' -> 'A', 'Bbmaj7' -> 'Bb'."""
    if not chord_name:
        return ""
    m = re.match(r'^([A-G][#b]?)', chord_name)
    return m.group(1) if m else chord_name[:1]


def chord_to_notes(chord_name: str) -> list:
    """Approximate pitch classes from a chord symbol."""
    root = extract_root(chord_name)
    if not root:
        return []

    NOTE_MAP = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4, 'F': 5, 'E#': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11, 'Cb': 11,
    }
    NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']

    root_val = NOTE_MAP.get(root)
    if root_val is None:
        return [root]

    quality = chord_name[len(root):]
    intervals = [0, 4, 7]  # default major triad

    if 'm' in quality and 'maj' not in quality:
        intervals = [0, 3, 7]
    if '7' in quality:
        if 'maj7' in quality:
            intervals.append(11)
        else:
            intervals.append(10)
    if '9' in quality:
        intervals.append(2)  # add 9th (=2nd)
    if 'dim' in quality:
        intervals = [0, 3, 6]
    if 'aug' in quality:
        intervals = [0, 4, 8]
    if 'sus2' in quality:
        intervals = [0, 2, 7]
    if 'sus4' in quality:
        intervals = [0, 5, 7]

    notes = []
    for i in set(intervals):
        note_idx = (root_val + i) % 12
        notes.append(NAMES[note_idx])
    return sorted(set(notes))


def chords_to_pitch_classes(chords_used: list) -> list:
    """Extract unique root pitch classes from chord symbols."""
    roots = set()
    for chord in chords_used:
        r = extract_root(chord)
        if r:
            roots.add(r)
    return sorted(roots)


def chords_to_full_notes(chords_used: list) -> list:
    """Get all pitch classes from all chords (not just roots)."""
    notes = set()
    for chord in chords_used:
        notes.update(chord_to_notes(chord))
    return sorted(notes)


def extract_all_lyrics(chart: dict) -> str:
    """Extract all lyrics from a chart, joined by newlines."""
    lines = []
    for sec in chart.get("sections", []):
        for line in sec.get("lines", []):
            if line.get("lyrics"):
                lines.append(line["lyrics"].strip())
    return "\n".join(lines)


def extract_section_lyrics(chart: dict, section_name_pattern: str) -> str:
    """Extract lyrics from sections matching a name pattern."""
    lines = []
    for sec in chart.get("sections", []):
        name = sec.get("name", "").lower()
        if re.search(section_name_pattern, name, re.IGNORECASE):
            for line in sec.get("lines", []):
                if line.get("lyrics"):
                    lines.append(line["lyrics"].strip())
    return "\n".join(lines)


def get_first_lyric_line(chart: dict) -> str:
    """Get the very first line with lyrics."""
    for sec in chart.get("sections", []):
        for line in sec.get("lines", []):
            if line.get("lyrics"):
                return line["lyrics"].strip()
    return ""


def chart_to_slash_notation(chart: dict) -> str:
    """Convert chart JSON to readable slash notation string (the output format)."""
    parts = []
    parts.append(f"Title: {chart.get('title', 'Unknown')}")
    parts.append(f"Artist: {chart.get('artist', 'Unknown')}")
    if chart.get("key"):
        parts.append(f"Key: {chart['key']}")
    if chart.get("capo"):
        parts.append(f"Capo: {chart['capo']}")
    parts.append(f"Chords used: {', '.join(chart.get('chords_used', []))}")
    parts.append("")

    for sec in chart.get("sections", []):
        sec_name = sec.get("name", "Section")
        notes = ""
        # Check if first line has notes
        if sec.get("lines") and sec["lines"][0].get("notes"):
            notes = f" ({sec['lines'][0]['notes']})"
        parts.append(f"[{sec_name}]{notes}")

        for line in sec.get("lines", []):
            chord_line = line.get("chords", "")
            lyrics = line.get("lyrics")
            if chord_line:
                parts.append(f"  {chord_line}")
            if lyrics:
                parts.append(f"  {lyrics}")
            if line.get("notes") and line != sec["lines"][0]:
                parts.append(f"  ({line['notes']})")
        parts.append("")

    return "\n".join(parts).strip()


def simulate_whisper_errors(text: str, error_rate: float = 0.10) -> str:
    """Simulate Whisper transcription errors on lyrics text."""
    if not text:
        return text
    words = text.split()
    if len(words) < 3:
        return text

    augmented = []
    for word in words:
        r = random.random()
        word_lower = word.lower().rstrip(".,!?;:'\"")

        if r < error_rate * 0.4:
            # Phonetic substitution
            if word_lower in PHONETIC_SUBS:
                sub = random.choice(PHONETIC_SUBS[word_lower])
                augmented.append(sub)
            else:
                # Random character swap/drop for unknown words
                if len(word) > 3 and random.random() < 0.5:
                    idx = random.randint(1, len(word) - 2)
                    mutated = word[:idx] + word[idx + 1:]
                    augmented.append(mutated)
                else:
                    augmented.append(word)
        elif r < error_rate * 0.6:
            # Word deletion
            continue
        elif r < error_rate * 0.7:
            # Filler insertion
            augmented.append(random.choice(FILLER_WORDS))
            augmented.append(word)
        else:
            augmented.append(word)

    return " ".join(augmented)


def simulate_word_dropout(text: str, drop_rate: float = 0.20) -> str:
    """Simulate Whisper dropping ~drop_rate of words."""
    if not text:
        return text
    words = text.split()
    if len(words) < 3:
        return text
    kept = [w for w in words if random.random() > drop_rate]
    if not kept:
        kept = [words[0]]  # keep at least one word
    return " ".join(kept)


def build_detected_notes_str(chart: dict) -> str:
    """Build a 'detected notes' string simulating pitch class detection."""
    chords_used = chart.get("chords_used", [])
    if not chords_used:
        return ""

    roots = chords_to_pitch_classes(chords_used)
    all_notes = chords_to_full_notes(chords_used)

    # Format: pipe-separated groups simulating per-section detection
    sections = chart.get("sections", [])
    if len(sections) <= 1:
        return " ".join(all_notes)

    # Group notes by section
    note_groups = []
    for sec in sections[:6]:  # cap at 6 sections
        sec_chords = set()
        for line in sec.get("lines", []):
            if line.get("chord_beats"):
                for cb in line["chord_beats"]:
                    sec_chords.add(cb["name"])
            elif line.get("chords"):
                # Parse from chord string
                for token in line["chords"].split():
                    if token and token[0].isupper():
                        sec_chords.add(token)
        sec_notes = set()
        for c in sec_chords:
            sec_notes.update(chord_to_notes(c))
        if sec_notes:
            note_groups.append(" ".join(sorted(sec_notes)))

    if note_groups:
        return " | ".join(note_groups)
    return " ".join(all_notes)


def make_recall_input(lyrics: str, detected_notes: str, detected_key: str) -> str:
    """Build the <recall> formatted input block."""
    parts = ["<recall>"]
    if lyrics:
        parts.append(f"Lyrics: {lyrics}")
    if detected_notes:
        parts.append(f"Detected notes: {detected_notes}")
    if detected_key:
        parts.append(f"Detected key: {detected_key}")
    parts.append("</recall>")
    return "\n".join(parts)


def make_recall_input_with_meta(title: str, artist: str, lyrics: str,
                                 detected_notes: str, detected_key: str) -> str:
    """Build input with title/artist metadata."""
    parts = ["<recall>"]
    if title:
        parts.append(f"Title: {title}")
    if artist:
        parts.append(f"Artist: {artist}")
    if lyrics:
        parts.append(f"Lyrics: {lyrics}")
    if detected_notes:
        parts.append(f"Detected notes: {detected_notes}")
    if detected_key:
        parts.append(f"Detected key: {detected_key}")
    parts.append("</recall>")
    return "\n".join(parts)


def make_output(chart: dict, confidence: float) -> str:
    """Build structured recall output."""
    title = chart.get("title", "Unknown")
    artist = chart.get("artist", "Unknown")
    chart_text = chart_to_slash_notation(chart)

    return (
        f'Song identified: "{title}" by {artist}\n'
        f"Confidence: {confidence:.2f}\n\n"
        f"{chart_text}"
    )


def make_negative_output() -> str:
    """Output for unrecognized songs."""
    return json.dumps({
        "recognized": False,
        "confidence": 0.0,
        "message": "Song not recognized. Using audio detection."
    })


def make_example(user_content: str, assistant_content: str,
                 song_id: str = "", variant: str = "") -> dict:
    """Create a training example in Phi-3 chat format."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Identify this song and recall the correct chord chart:\n\n{user_content}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "_song": song_id,
        "_variant": variant,
    }


# ─── Negative Example Generation ────────────────────────────────────────────

FAKE_WORDS = [
    "moonlight", "shadow", "river", "fire", "dancing", "broken", "whisper",
    "midnight", "thunder", "crystal", "golden", "silver", "ocean", "forest",
    "diamond", "velvet", "burning", "falling", "rising", "shining", "dreaming",
    "running", "flying", "crying", "laughing", "spinning", "floating", "diving",
    "crawling", "bleeding", "breathing", "screaming", "singing", "praying",
    "waiting", "sleeping", "waking", "chasing", "hiding", "searching",
    "wandering", "drifting", "crashing", "burning", "freezing", "melting",
    "fading", "glowing", "shaking", "breaking", "making", "taking",
    "love", "heart", "soul", "mind", "eyes", "hands", "arms", "lips",
    "blood", "tears", "rain", "snow", "wind", "storm", "star", "sun",
    "moon", "sky", "earth", "sea", "road", "door", "wall", "window",
    "the", "a", "my", "your", "our", "in", "on", "at", "by", "to",
    "and", "but", "or", "so", "if", "when", "where", "how", "why",
    "I", "you", "we", "they", "she", "he", "it", "me", "us", "them",
    "don't", "can't", "won't", "isn't", "wasn't", "couldn't", "wouldn't",
    "never", "always", "sometimes", "forever", "again", "away", "down",
    "up", "out", "back", "here", "there", "now", "then", "still", "just",
]

FAKE_ARTISTS = [
    "The Wanderlust", "Broken Compass", "Velvet Void", "Neon Dusk",
    "Glass Harbor", "Iron Petals", "Hollow Crown", "Driftwood Saints",
    "The Midnight Bloom", "Echo Valley", "Crimson Tide Orchestra",
    "Paper Wolves", "Static Garden", "Luna Bridge", "The Fading Hours",
]

NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def generate_fake_lyrics(num_lines: int = 4, words_per_line: int = 8) -> str:
    """Generate plausible but non-matching lyrics."""
    lines = []
    for _ in range(num_lines):
        line = " ".join(random.choices(FAKE_WORDS, k=words_per_line))
        lines.append(line)
    return "\n".join(lines)


def generate_fake_notes() -> str:
    """Generate random pitch class groups."""
    num_groups = random.randint(1, 4)
    groups = []
    for _ in range(num_groups):
        num_notes = random.randint(2, 5)
        notes = random.sample(NOTE_NAMES, num_notes)
        groups.append(" ".join(sorted(notes)))
    return " | ".join(groups)


def generate_negative_examples(n: int = 5000) -> list:
    """Generate negative (unknown song) training examples."""
    examples = []
    for i in range(n):
        num_lines = random.randint(2, 6)
        words_per_line = random.randint(5, 12)
        fake_lyrics = generate_fake_lyrics(num_lines, words_per_line)
        fake_notes = generate_fake_notes()
        fake_key = random.choice(NOTE_NAMES) + random.choice(["", "m"])

        # Vary input format
        r = random.random()
        if r < 0.5:
            # Lyrics + notes
            user_input = make_recall_input(fake_lyrics, fake_notes, fake_key)
        elif r < 0.8:
            # Lyrics only
            user_input = make_recall_input(fake_lyrics, "", fake_key)
        else:
            # With fake metadata
            fake_title = " ".join(random.choices(FAKE_WORDS, k=random.randint(1, 3))).title()
            fake_artist = random.choice(FAKE_ARTISTS)
            user_input = make_recall_input_with_meta(
                fake_title, fake_artist, fake_lyrics, fake_notes, fake_key
            )

        examples.append(make_example(
            user_input, make_negative_output(),
            song_id=f"negative_{i}", variant="negative"
        ))
    return examples


# ─── Main Data Generation ───────────────────────────────────────────────────

def load_all_charts() -> list:
    """Load all chart JSONs from the chord library."""
    charts = []
    for fpath in sorted(glob.glob(str(CHORD_LIBRARY / "*/*.json"))):
        try:
            with open(fpath) as f:
                chart = json.load(f)
            chart["_path"] = fpath
            artist_dir = os.path.basename(os.path.dirname(fpath))
            chart["_artist_dir"] = artist_dir
            charts.append(chart)
        except Exception as e:
            print(f"  WARN: Failed to load {fpath}: {e}")
    return charts


def has_lyrics(chart: dict) -> bool:
    """Check if a chart has any lyrics."""
    for sec in chart.get("sections", []):
        for line in sec.get("lines", []):
            if line.get("lyrics"):
                return True
    return False


def generate_variants_with_lyrics(chart: dict) -> list:
    """Generate 6 variants for a song with lyrics."""
    title = chart.get("title", "Unknown")
    artist = chart.get("artist", "Unknown")
    key = chart.get("key", "")
    song_id = f"{artist} - {title}"
    chords_used = chart.get("chords_used", [])

    all_lyrics = extract_all_lyrics(chart)
    verse_lyrics = extract_section_lyrics(chart, r"verse\s*1|verse$")
    chorus_lyrics = extract_section_lyrics(chart, r"chorus|refrain|hook")
    first_line = get_first_lyric_line(chart)
    detected_notes = build_detected_notes_str(chart)

    output_text = chart_to_slash_notation(chart)
    examples = []

    # 1. Full lyrics
    conf = round(random.uniform(0.95, 0.99), 2)
    user_input = make_recall_input(all_lyrics, detected_notes, key or "")
    examples.append(make_example(
        user_input, make_output(chart, conf), song_id, "full_lyrics"
    ))

    # 2. Partial lyrics (verse only)
    verse_text = verse_lyrics if verse_lyrics else all_lyrics.split("\n")[0] if all_lyrics else ""
    if verse_text:
        conf = round(random.uniform(0.85, 0.94), 2)
        user_input = make_recall_input(verse_text, detected_notes, key or "")
        examples.append(make_example(
            user_input, make_output(chart, conf), song_id, "verse_only"
        ))

    # 3. Partial lyrics (chorus only)
    chorus_text = chorus_lyrics if chorus_lyrics else ""
    if not chorus_text:
        # Fallback: use middle section lyrics
        all_lines = all_lyrics.split("\n")
        mid = len(all_lines) // 2
        chorus_text = "\n".join(all_lines[max(0, mid - 1):mid + 2])
    if chorus_text:
        conf = round(random.uniform(0.70, 0.84), 2)
        user_input = make_recall_input(chorus_text, detected_notes, key or "")
        examples.append(make_example(
            user_input, make_output(chart, conf), song_id, "chorus_only"
        ))

    # 4. Whisper error sim 1 (~10% errors)
    garbled_1 = simulate_whisper_errors(all_lyrics, error_rate=0.10)
    conf = round(random.uniform(0.75, 0.88), 2)
    user_input = make_recall_input(garbled_1, detected_notes, key or "")
    examples.append(make_example(
        user_input, make_output(chart, conf), song_id, "whisper_error_10pct"
    ))

    # 5. Whisper error sim 2 (~20% word dropout)
    garbled_2 = simulate_word_dropout(all_lyrics, drop_rate=0.20)
    conf = round(random.uniform(0.60, 0.80), 2)
    user_input = make_recall_input(garbled_2, detected_notes, key or "")
    examples.append(make_example(
        user_input, make_output(chart, conf), song_id, "whisper_dropout_20pct"
    ))

    # 6. Metadata hint (title + artist + first line)
    conf = round(random.uniform(0.90, 0.99), 2)
    user_input = make_recall_input_with_meta(
        title, artist, first_line, "", key or ""
    )
    examples.append(make_example(
        user_input, make_output(chart, conf), song_id, "metadata_hint"
    ))

    return examples


def generate_variants_instrumental(chart: dict) -> list:
    """Generate 2 variants for a song without lyrics."""
    title = chart.get("title", "Unknown")
    artist = chart.get("artist", "Unknown")
    key = chart.get("key", "")
    song_id = f"{artist} - {title}"
    detected_notes = build_detected_notes_str(chart)

    examples = []

    # 1. Title + artist + detected notes
    conf = round(random.uniform(0.85, 0.95), 2)
    user_input = make_recall_input_with_meta(
        title, artist, "", detected_notes, key or ""
    )
    examples.append(make_example(
        user_input, make_output(chart, conf), song_id, "instrumental_with_notes"
    ))

    # 2. Title + artist only
    conf = round(random.uniform(0.90, 0.99), 2)
    user_input = make_recall_input_with_meta(
        title, artist, "", "", key or ""
    )
    examples.append(make_example(
        user_input, make_output(chart, conf), song_id, "instrumental_meta_only"
    ))

    return examples


def main():
    print("=" * 60)
    print("StemScriber — Chord Recall Training Data Prep (Stage 2)")
    print("=" * 60)

    # Load all charts
    print("\n[1/5] Loading chord library...")
    charts = load_all_charts()
    print(f"  Loaded {len(charts)} charts from {CHORD_LIBRARY}")

    # Split Kozelski (validation) vs rest (training)
    kozelski_charts = [c for c in charts if c.get("_artist_dir", "").lower() == "kozelski"]
    train_charts = [c for c in charts if c.get("_artist_dir", "").lower() != "kozelski"]

    kozelski_with = [c for c in kozelski_charts if has_lyrics(c)]
    kozelski_without = [c for c in kozelski_charts if not has_lyrics(c)]
    train_with = [c for c in train_charts if has_lyrics(c)]
    train_without = [c for c in train_charts if not has_lyrics(c)]

    print(f"\n  Training artists: {len(set(c.get('_artist_dir') for c in train_charts))}")
    print(f"  Training songs: {len(train_charts)} ({len(train_with)} with lyrics, {len(train_without)} without)")
    print(f"  Validation (Kozelski): {len(kozelski_charts)} ({len(kozelski_with)} with lyrics, {len(kozelski_without)} without)")

    # Generate training examples
    print("\n[2/5] Generating training examples...")
    train_examples = []
    variant_counts = {}

    for i, chart in enumerate(train_with):
        variants = generate_variants_with_lyrics(chart)
        for ex in variants:
            v = ex["_variant"]
            variant_counts[v] = variant_counts.get(v, 0) + 1
        train_examples.extend(variants)
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i + 1}/{len(train_with)} songs with lyrics...")

    for i, chart in enumerate(train_without):
        variants = generate_variants_instrumental(chart)
        for ex in variants:
            v = ex["_variant"]
            variant_counts[v] = variant_counts.get(v, 0) + 1
        train_examples.extend(variants)

    print(f"  Generated {len(train_examples)} positive examples")

    # Generate negative examples
    print("\n[3/5] Generating 5,000 negative (unknown song) examples...")
    negative_examples = generate_negative_examples(5000)
    variant_counts["negative"] = len(negative_examples)
    train_examples.extend(negative_examples)

    # Generate validation examples (Kozelski)
    print("\n[4/5] Generating Kozelski validation examples...")
    val_examples = []
    for chart in kozelski_with:
        val_examples.extend(generate_variants_with_lyrics(chart))
    for chart in kozelski_without:
        val_examples.extend(generate_variants_instrumental(chart))

    print(f"  Generated {len(val_examples)} validation examples from {len(kozelski_charts)} Kozelski songs")

    # Shuffle training set
    random.shuffle(train_examples)

    # Write output
    print("\n[5/5] Writing JSONL files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_size = 0
    with open(TRAIN_PATH, "w") as f:
        for ex in train_examples:
            line = json.dumps(ex, ensure_ascii=False) + "\n"
            f.write(line)
            train_size += len(line.encode("utf-8"))

    val_size = 0
    with open(VAL_PATH, "w") as f:
        for ex in val_examples:
            line = json.dumps(ex, ensure_ascii=False) + "\n"
            f.write(line)
            val_size += len(line.encode("utf-8"))

    # Stats
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\n  Training file: {TRAIN_PATH}")
    print(f"  Training size: {train_size / 1e6:.1f} MB ({train_size / 1e9:.2f} GB)")
    print(f"  Training examples: {len(train_examples):,}")
    print(f"\n  Validation file: {VAL_PATH}")
    print(f"  Validation size: {val_size / 1e6:.2f} MB")
    print(f"  Validation examples: {len(val_examples):,}")

    print(f"\n  Total examples: {len(train_examples) + len(val_examples):,}")

    print("\n  Variant breakdown (training set):")
    for variant, count in sorted(variant_counts.items()):
        print(f"    {variant}: {count:,}")

    print(f"\n  Songs with lyrics: {len(train_with) + len(kozelski_with):,}")
    print(f"  Songs without lyrics (instrumental): {len(train_without) + len(kozelski_without):,}")

    # Verify format compatibility
    print("\n  Format check:")
    sample = train_examples[0]
    assert "messages" in sample, "Missing 'messages' key"
    assert len(sample["messages"]) == 3, "Expected 3 messages (system/user/assistant)"
    assert sample["messages"][0]["role"] == "system"
    assert sample["messages"][1]["role"] == "user"
    assert sample["messages"][2]["role"] == "assistant"
    print("    OK — messages format matches train_chart_formatter.py expectations")
    print("    OK — system/user/assistant roles present")
    print("    OK — _song and _variant metadata fields present")

    # Sample output
    print("\n  Sample training example (first 200 chars of user input):")
    print(f"    {sample['messages'][1]['content'][:200]}...")
    print(f"\n  Sample assistant output (first 200 chars):")
    print(f"    {sample['messages'][2]['content'][:200]}...")

    print("\n  READY FOR TRAINING.")
    print("  When Phi-3 formatter finishes, combine with:")
    print(f"    training_data/chart_formatter/train.jsonl (formatter)")
    print(f"    training_data/chart_recall/recall_train.jsonl (recall)")
    print("  Then run: ../venv311/bin/python -m modal run train_chart_formatter.py")


if __name__ == "__main__":
    main()
