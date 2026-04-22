# Chord Recall Training Plan — Song Recognition + Chart Memory

**Date:** 2026-04-04
**Status:** Design Only (DO NOT TRAIN YET)
**Depends on:** Phi-3 chart formatter training (in progress)
**Goal:** Train a model that recognizes songs from lyrics/stems and recalls the correct chart from memory

---

## 1. The Core Concept

This is fundamentally different from chord detection or chart formatting:

| Capability | What It Does | Status |
|------------|-------------|--------|
| **Chord Detection** (BTC) | Listens to audio, outputs raw chord names + timestamps | Existing |
| **Chart Formatting** (Phi-3) | Takes raw chords + lyrics + timing, formats a clean chart | Training now |
| **Chart Recall** (THIS PLAN) | Recognizes a song from lyrics/notes, recalls the correct chart from memory | Design phase |

The recall model has **memorized 15,437 songs** during training. At inference time:
1. Whisper transcribes lyrics from the vocal stem
2. Stem-aware detector picks up pitch classes from guitar/bass/piano
3. The model recognizes the song from lyrics (primary) + detected notes (secondary)
4. The model **recalls** the correct chord chart from its training weights
5. It cross-references recalled chords with detected chords for validation
6. Outputs a complete, correctly formatted chart

This is analogous to how a session musician works: they hear the lyrics, recognize the song, and play from memory -- they don't re-figure-out the chords by ear every time.

---

## 2. Why Recall Beats Detection for Known Songs

**Detection-only accuracy** (current BTC pipeline): ~70-80% on common chords, worse on extended chords (9ths, 11ths, dim7, etc.)

**Recall accuracy** (this approach): Theoretically ~95-100% for recognized songs, because the model learned the exact correct chart during training.

The win is clearest for:
- Extended/jazz chords that BTC struggles with (E9, Bbmaj7, F#m7b5)
- Songs with unusual voicings or detuned guitars
- Songs where the mix makes chord detection ambiguous
- Songs with rapid chord changes that BTC smears

The 15,437 charts in the library represent the most popular songs that users will upload. A recall model that nails these covers the majority of real-world usage.

---

## 3. Architecture Decision: Same Model or Separate?

### Option A: Extend the Phi-3 Formatter (RECOMMENDED)

Add recall as a second capability to the same Phi-3-mini model already being trained for formatting.

**Pros:**
- Single model to deploy and maintain on Modal
- Shared understanding of chart structure and formatting conventions
- The model already learns what a correct chart looks like during formatter training
- Recall is essentially "given context, produce a chart" -- same output format
- Saves ~$20-40 in separate training costs

**Cons:**
- Larger training dataset (formatter examples + recall examples)
- Risk of catastrophic forgetting if not balanced properly
- Training takes longer (but still under 24 hours on A10G)

**Implementation:** Two task prefixes in the prompt:

```
Task 1 (Formatting): "Format this chord chart from raw detection data: ..."
Task 2 (Recall): "Identify this song and recall the correct chord chart: ..."
```

The model learns both skills. At inference, we choose the task based on whether the song is recognized.

### Option B: Separate Recall Model

Train a second Phi-3-mini exclusively on recall.

**Pros:**
- No interference with formatter training
- Can optimize recall-specific hyperparameters independently
- Easier to debug and iterate on each model separately

**Cons:**
- Two models to deploy on Modal (2x VRAM, 2x cost, 2x latency)
- Duplicated understanding of chart formatting
- More complex pipeline orchestration

### Recommendation: Option A (Combined)

Train recall and formatting as two tasks in the same Phi-3-mini model. Use a 60/40 split (60% formatter examples, 40% recall examples) to prevent one task from dominating. If quality degrades on either task, split into separate models as a fallback.

**Wait for the formatter training to complete, then add recall examples and retrain.**

---

## 4. Training Data Preparation

### 4.1 Source Data

- **15,437 chord chart JSONs** at `backend/chord_library/` (759 artists)
- Each contains: title, artist, key, capo, chords_used, sections with chords + lyrics
- Example (Thunderhead by Kozelski):
  ```json
  {
    "title": "Thunderhead",
    "artist": "Kozelski",
    "key": "Am",
    "chords_used": ["Am7", "Bm7", "Cm7", "E9", "Bm6", "Dm7", "Gmaj7", "G#7", "Em7", "G#maj7"],
    "sections": [
      {"name": "Intro Pickup", "lines": [{"chords": "Cm7       Bm7", "lyrics": null}]},
      {"name": "Verse 1", "lines": [{"chords": "Am7  Bm7  Cm7  Bm7  E9  Bm6", "lyrics": "Another night stirring with the boys in a cloud"}]}
    ]
  }
  ```

### 4.2 Recall Training Example Format

Each of the 15,437 songs becomes multiple training examples:

**Primary example (full lyrics):**
```
<|system|>
You are a chord chart expert with encyclopedic knowledge of 15,000+ songs.
Given lyrics and optional detected notes, identify the song and recall the
correct chord chart from your training.
<|user|>
Identify this song and recall the correct chord chart:

Lyrics (from Whisper):
"Another night stirring with the boys in a cloud
Where the pain runs deep and the music plays loud
Through the haze and the smoke another round on the house
And the whole damn room raises up with a shout"

Detected pitch classes per section:
- Section 1: A, B, C, E (root notes)

Detected key: Am

<|assistant|>
Song identified: "Thunderhead" by Kozelski
Confidence: 0.97

{full chart JSON output}
```

**Augmented example (partial lyrics / Whisper errors):**
```
<|user|>
Identify this song and recall the correct chord chart:

Lyrics (from Whisper):
"Another night stirring with the boys in the cloud
Where the pain runs deep and the music plays loud"

Detected pitch classes per section:
- Section 1: A, B, C, E (root notes)

Detected key: Am

<|assistant|>
Song identified: "Thunderhead" by Kozelski
Confidence: 0.85

{full chart JSON output}
```

### 4.3 Training Example Categories

For each of the 15,437 songs, generate these variants:

| Variant | Count per Song | Purpose | Total Examples |
|---------|---------------|---------|----------------|
| Full lyrics, all sections | 1 | Baseline recall | 15,437 |
| Verse 1 + chorus only | 1 | Partial lyrics (common with Whisper) | 15,437 |
| Chorus only | 1 | Minimal lyrics recognition | 15,437 |
| Full lyrics with simulated Whisper errors | 2 | Error robustness | 30,874 |
| Lyrics + key mismatch (transposed) | 1 | Capo/tuning differences | 15,437 |
| Title + artist hint only (no lyrics) | 1 | Metadata-based recall | 15,437 |

**Total recall examples: ~107,000**

Combined with ~185,000 formatter examples already prepared: **~292,000 total training examples.**

### 4.4 Data Augmentation Details

**Simulated Whisper errors:**
- Random word substitution (10-20% of words replaced with phonetically similar alternatives)
- Word deletion (5-10% of words dropped)
- Word insertion (occasional filler words: "uh", "oh", "yeah")
- Homophone confusion: "night" -> "knight", "boys" -> "boy's", "hear" -> "here"
- Slurred words: "another" -> "anudder", "gonna" -> "going to"

Script approach:
```python
import random

def simulate_whisper_errors(lyrics_text, error_rate=0.15):
    words = lyrics_text.split()
    augmented = []
    for word in words:
        r = random.random()
        if r < error_rate * 0.4:  # substitution
            augmented.append(get_phonetic_neighbor(word))
        elif r < error_rate * 0.6:  # deletion
            continue
        elif r < error_rate * 0.7:  # insertion
            augmented.append(random.choice(["uh", "oh", "the"]))
            augmented.append(word)
        else:
            augmented.append(word)
    return " ".join(augmented)
```

**Transposition handling:**
- If the training chart is in key C and the detected key is D, the recall output should note the transposition
- Training example: input key=D, recalled chart key=C, output includes note: "Detected key (D) differs from chart key (C) -- song may be performed with capo or alternate tuning"
- The recalled chart remains in its original key (the correct one from the library)

**Partial lyrics generation:**
- Extract only specific sections from the full chart lyrics
- Simulate realistic Whisper behavior: Whisper often gets the chorus right but mangles quiet verses
- Create examples where only 30-50% of lyrics are provided

### 4.5 Pitch Class Simulation

For each chart, generate simulated "detected pitch classes" from the chord symbols:

```python
def chords_to_pitch_classes(chords_used):
    """Extract root pitch classes from chord symbols."""
    roots = []
    for chord in chords_used:
        root = extract_root(chord)  # "Am7" -> "A", "Bbmaj7" -> "Bb"
        roots.append(root)
    return list(set(roots))

# Example: ["Am7", "Bm7", "Cm7", "E9"] -> ["A", "B", "C", "E"]
```

This simulates what the stem-aware detector would output: it can identify root notes from the bass/guitar stems even if it cannot determine the exact chord quality.

---

## 5. Confidence Scoring and Fallback

### 5.1 Three-Tier Confidence System

| Tier | Confidence | Trigger | Action |
|------|-----------|---------|--------|
| **High** | > 0.85 | Lyrics match a known song clearly | Recall chart directly, minor cross-reference with detection |
| **Medium** | 0.50 - 0.85 | Lyrics partially match, or multiple possible songs | Recall chart as hypothesis, heavily cross-reference with stem detection |
| **Low / No Match** | < 0.50 | Song not in training data | Fall back entirely to BTC detection + Phi-3 formatter |

### 5.2 How Confidence Is Computed

The model outputs a confidence score as part of its structured response. During training, confidence is assigned based on the augmentation level:

| Training Input Quality | Assigned Confidence |
|----------------------|-------------------|
| Full lyrics, exact match | 0.95 - 0.99 |
| Partial lyrics (verse + chorus) | 0.85 - 0.94 |
| Chorus only | 0.70 - 0.84 |
| Lyrics with 15%+ Whisper errors | 0.60 - 0.80 |
| Only title + artist metadata | 0.90 - 0.99 (if metadata matches) |
| No recognizable lyrics | 0.10 - 0.30 |

The model learns to calibrate its confidence based on input quality.

### 5.3 Cross-Reference Validation

When confidence is medium (0.50-0.85), cross-reference the recalled chart with stem detection:

```python
def cross_reference(recalled_chart, detected_chords, detected_key):
    """Validate recall against detection."""
    recalled_roots = set(extract_roots(recalled_chart["chords_used"]))
    detected_roots = set(extract_roots(detected_chords))

    overlap = recalled_roots & detected_roots
    overlap_ratio = len(overlap) / max(len(recalled_roots), 1)

    if overlap_ratio > 0.7:
        # Detection confirms recall — boost confidence
        return recalled_chart, "confirmed"
    elif overlap_ratio > 0.4:
        # Partial match — use recall for known chords, detection for gaps
        return merge_charts(recalled_chart, detected_chords), "partial"
    else:
        # Recall and detection disagree — fall back to detection
        return None, "rejected"
```

### 5.4 Fallback: Unknown Songs

When the model does not recognize the song (confidence < 0.50):

```
Upload audio
  -> Stem separation (existing)
  -> BTC chord detection on harmonic stems (existing)
  -> Whisper lyrics from vocal stem (existing)
  -> Recall model says: "Song not recognized (confidence: 0.23)"
  -> FALLBACK: Use BTC chords + Whisper lyrics
  -> Phi-3 formatter produces chart from raw detection data
  -> Output chart with source: "StemScriber AI (detected)"
```

vs. recognized song:

```
Upload audio
  -> Stem separation (existing)
  -> BTC chord detection (for cross-reference only)
  -> Whisper lyrics from vocal stem
  -> Recall model says: "Thunderhead by Kozelski (confidence: 0.94)"
  -> Output recalled chart with source: "StemScriber AI (recalled)"
```

The user sees a different badge: "recalled" (high accuracy) vs. "detected" (standard accuracy). This sets expectations.

---

## 6. Partial Match and Fuzzy Lyrics Handling

### 6.1 The Whisper Error Problem

Whisper on isolated vocal stems typically achieves 85-95% word accuracy. Common failure modes:
- Soft/mumbled passages: words dropped or garbled
- Backing vocal bleed from imperfect separation: phantom words added
- Song-specific vocabulary: proper nouns, slang, made-up words
- Repeated sections: Whisper sometimes merges or skips repeats

### 6.2 How the Model Handles Partial Matches

The model is trained on degraded lyrics (Section 4.3 augmentation), so it learns to recognize songs from partial/noisy input. The key insight: **most songs are identifiable from just the chorus lyrics**, even with 20% word errors.

Training ensures:
- Given just "Is this the real life is this just fantasy" (even without punctuation or capitalization), the model recalls Bohemian Rhapsody
- Given "Another night stirrin with da boys in a cloud", it still recalls Thunderhead despite the errors
- Given only "I miss you when the lights go out", it narrows to Adele's "I Miss You"

### 6.3 Disambiguation

When lyrics could match multiple songs (common phrases like "I love you" or "baby come back"):

1. **Detected notes break the tie.** If lyrics match both Song A (key of C) and Song B (key of G), and detected root notes are C, F, G, Am -- Song A wins.
2. **Detected tempo helps.** If Song A is 120 BPM and Song B is 80 BPM, and detected tempo is ~118 BPM -- Song A wins.
3. **If still ambiguous**, return both candidates with confidence scores and let the cross-reference validation (Section 5.3) pick the winner.

Training data includes disambiguation examples where the model must choose between similar songs.

---

## 7. Training Configuration

### 7.1 Combined Model Config (Phi-3-mini with Recall + Formatting)

```yaml
# Base model
base_model: microsoft/Phi-3-mini-128k-instruct
quantization: 4bit (bitsandbytes nf4)

# LoRA config
lora_r: 64
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Training
learning_rate: 1.5e-4  # slightly lower than formatter-only since more data
warmup_ratio: 0.03
lr_scheduler: cosine
num_epochs: 3
batch_size: 4
gradient_accumulation: 8  # effective batch size 32
max_seq_length: 4096

# Data mix
formatter_examples: 185,000 (60%)
recall_examples: 107,000 (40% — weighted sampling)

# Hardware
gpu: Modal A10G (24GB VRAM)
estimated_time: 12-18 hours
estimated_cost: $25-40 on Modal
```

### 7.2 Training Prompt Templates

**Task 1: Chart Formatting (existing)**
```
<|system|>
You are a professional chord chart formatter. Given raw chord detection
data with timestamps, format a clean, readable chord chart.
<|user|>
Format this as a StemScriber JSON chord chart:
Title: {title} ...
Chord events: [{...}]
Lyrics with timing: [{...}]
<|assistant|>
{formatted chart JSON}
```

**Task 2: Song Recall (NEW)**
```
<|system|>
You are a chord chart expert with encyclopedic knowledge of 15,000+ songs.
Given lyrics and optional detected notes, identify the song and recall the
correct chord chart from your training.
<|user|>
Identify this song and recall the correct chord chart:

Lyrics (from Whisper):
"{lyrics_text}"

Detected pitch classes per section:
{pitch_classes}

Detected key: {key}

<|assistant|>
Song identified: "{title}" by {artist}
Confidence: {confidence}

{full chart JSON}
```

**Task 3: Unknown Song Fallback (implicit)**
```
<|user|>
Identify this song and recall the correct chord chart:

Lyrics (from Whisper):
"Some lyrics the model has never seen before..."

Detected pitch classes per section:
- A, D, E

Detected key: A

<|assistant|>
Song not recognized.
Confidence: 0.15

Falling back to detected chords. Please use the chord detection pipeline output.
```

The model learns to output "Song not recognized" when it genuinely does not know the song. This is trained by including ~5,000 examples with synthetic/random lyrics that do not match any song in the library.

### 7.3 Training in Two Stages

**Stage 1: Formatter training (ALREADY IN PROGRESS)**
- 185,000 formatter examples
- Phi-3-mini QLoRA fine-tune
- Estimated: 4-8 hours, $10-20

**Stage 2: Add recall capability (THIS PLAN)**
- Load the Stage 1 checkpoint (formatter-trained model)
- Continue training with mixed data: 60% formatter + 40% recall
- 3 additional epochs on the combined dataset
- Estimated: 8-12 hours, $15-25

This staged approach means:
- The formatter is usable immediately after Stage 1
- Recall gets added without restarting from scratch
- If recall training degrades formatting quality, we can revert to the Stage 1 checkpoint

---

## 8. Inference Pipeline Integration

### 8.1 Updated Pipeline Flow

```
User uploads audio
  |
  v
[Stem Separation] — Modal A10G (~60s)
  |
  +---> [BTC Chord Detection] on harmonic mix (~5s)
  |       Output: raw chord events with timestamps
  |
  +---> [Whisper] on vocal stem (~10s)
  |       Output: lyrics with word timestamps
  |
  v
[Recall Check] — Phi-3-mini on Modal (~1-2s)
  |  Input: Whisper lyrics + BTC root notes + detected key
  |  Output: {song_id, confidence, recalled_chart} OR "not recognized"
  |
  v
[Decision Gate]
  |
  +-- confidence >= 0.85 --> Use recalled chart
  |     Cross-reference with BTC (validation only)
  |     Output: chart with badge "recalled"
  |
  +-- 0.50 <= confidence < 0.85 --> Merge recalled + detected
  |     Use recalled chart structure, verify chords against BTC
  |     Output: chart with badge "recalled (verified)"
  |
  +-- confidence < 0.50 --> Fall back to detection
        Use BTC chords + Whisper lyrics
        Run through Phi-3 formatter (Task 1)
        Output: chart with badge "detected"
```

### 8.2 Latency Impact

| Step | Current | With Recall |
|------|---------|-------------|
| Stem separation | ~60s | ~60s (unchanged) |
| BTC chord detection | ~5s | ~5s (unchanged, runs in parallel) |
| Whisper lyrics | ~10s | ~10s (unchanged, runs in parallel) |
| Recall check | N/A | ~1-2s (new, runs after Whisper) |
| Chart formatting | ~0.5s | ~0.5s (only if fallback needed) |
| **Total wall time** | ~65s | ~66-67s |

The recall check adds only 1-2 seconds because:
- It runs AFTER Whisper completes (needs lyrics as input)
- It is a single forward pass through Phi-3-mini
- The A10G is already warm from stem separation

### 8.3 Cost Impact

| Component | Current Cost/Song | With Recall |
|-----------|------------------|-------------|
| Stem separation (Modal A10G) | $0.06 | $0.06 |
| BTC chord detection | ~$0.001 | ~$0.001 |
| Whisper | ~$0.002 | ~$0.002 |
| Recall check (Phi-3) | N/A | ~$0.001 |
| Formatter (Phi-3, fallback) | ~$0.001 | ~$0.001 (only if needed) |
| **Total** | ~$0.064 | ~$0.065 |

Negligible cost increase. The recall model and formatter share the same Phi-3 weights loaded on the same GPU.

---

## 9. Evaluation Plan

### 9.1 Recall Accuracy Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Song ID accuracy (full lyrics)** | Given complete lyrics, correctly identify the song | > 98% |
| **Song ID accuracy (partial lyrics)** | Given chorus only or verse + chorus | > 90% |
| **Song ID accuracy (noisy lyrics)** | Given lyrics with 15% Whisper errors | > 85% |
| **False positive rate** | Incorrectly identifying an unknown song as a known one | < 2% |
| **Chart recall fidelity** | Recalled chart matches training data exactly (chord-for-chord) | > 95% |
| **Confidence calibration** | High-confidence predictions are actually correct at the stated rate | Within 5% |

### 9.2 Validation Sets

| Set | Size | Source | Purpose |
|-----|------|--------|---------|
| Kozelski songs | 20 | `chord_library/kozelski/` | Gold standard -- Jeff knows these are correct |
| Random holdout | 500 | Random 3% from library, excluded from training | Generalization test |
| Unknown songs | 100 | Synthetic lyrics not in library | False positive detection |
| Degraded lyrics | 200 | Holdout songs with simulated Whisper errors | Robustness test |

### 9.3 End-to-End Test

Upload actual audio for songs in the library (e.g., "The Time Comes" demo WAV) and verify:
1. Whisper produces lyrics
2. Recall model identifies the song correctly
3. Recalled chart matches the library JSON
4. Cross-reference with BTC detection confirms chords
5. Frontend renders the chart correctly

---

## 10. Data Prep Script Outline

```python
#!/usr/bin/env python3
"""
chord_recall_data_prep.py

Generates training examples for chord recall from the chord library.
Run from: ~/stemscribe/backend/
Output: training_data/chord_recall/train.jsonl, val.jsonl
"""

import json
import os
import random
from pathlib import Path

CHORD_LIBRARY = Path("chord_library")
OUTPUT_DIR = Path("training_data/chord_recall")
HOLDOUT_RATIO = 0.03  # 3% for validation

SYSTEM_PROMPT = (
    "You are a chord chart expert with encyclopedic knowledge of 15,000+ songs. "
    "Given lyrics and optional detected notes, identify the song and recall the "
    "correct chord chart from your training."
)

def extract_all_lyrics(chart_json):
    """Extract all lyrics text from a chart."""
    lines = []
    for section in chart_json.get("sections", []):
        for line in section.get("lines", []):
            if line.get("lyrics") and line["lyrics"].strip():
                lines.append(line["lyrics"].strip())
    return lines

def extract_section_lyrics(chart_json, section_types):
    """Extract lyrics from specific section types (e.g., ['Verse 1', 'Chorus'])."""
    lines = []
    for section in chart_json.get("sections", []):
        name = section.get("name", "").lower()
        if any(st.lower() in name for st in section_types):
            for line in section.get("lines", []):
                if line.get("lyrics") and line["lyrics"].strip():
                    lines.append(line["lyrics"].strip())
    return lines

def chords_to_roots(chords_used):
    """Extract root note names from chord symbols."""
    roots = set()
    for chord in chords_used:
        root = ""
        if len(chord) >= 1:
            root = chord[0]
        if len(chord) >= 2 and chord[1] in ("#", "b"):
            root += chord[1]
        if root:
            roots.add(root)
    return sorted(roots)

def simulate_whisper_errors(text, error_rate=0.15):
    """Add realistic Whisper transcription errors."""
    words = text.split()
    result = []
    for word in words:
        r = random.random()
        if r < error_rate * 0.4:
            # Phonetic substitution (simplified)
            result.append(word[:-1] + random.choice("aeiouns") if len(word) > 2 else word)
        elif r < error_rate * 0.6:
            continue  # Drop word
        elif r < error_rate * 0.7:
            result.append(random.choice(["uh", "oh", "the"]))
            result.append(word)
        else:
            result.append(word)
    return " ".join(result)

def make_recall_example(chart, lyrics_text, pitch_classes, confidence, include_metadata=False):
    """Create a single training example in chat format."""
    user_parts = ["Identify this song and recall the correct chord chart:\n"]
    
    if include_metadata:
        user_parts.append(f"Metadata hint: {chart['title']} by {chart['artist']}\n")
    
    user_parts.append(f"Lyrics (from Whisper):\n\"{lyrics_text}\"\n")
    user_parts.append(f"Detected pitch classes per section:\n- {', '.join(pitch_classes)}\n")
    user_parts.append(f"Detected key: {chart.get('key', 'Unknown')}")
    
    # Build assistant response
    assistant_parts = [
        f"Song identified: \"{chart['title']}\" by {chart['artist']}",
        f"Confidence: {confidence:.2f}",
        "",
        json.dumps(chart, indent=2)
    ]
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_parts)},
            {"role": "assistant", "content": "\n".join(assistant_parts)}
        ]
    }

def make_unknown_song_example():
    """Create a training example for an unrecognized song."""
    fake_lyrics = generate_random_lyrics()  # Random word sequences
    fake_roots = random.sample(["A", "B", "C", "D", "E", "F", "G"], 3)
    fake_key = random.choice(["C", "G", "D", "A", "E", "Am", "Em", "Dm"])
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Identify this song and recall the correct chord chart:\n\n"
                f"Lyrics (from Whisper):\n\"{fake_lyrics}\"\n\n"
                f"Detected pitch classes per section:\n- {', '.join(fake_roots)}\n\n"
                f"Detected key: {fake_key}"
            )},
            {"role": "assistant", "content": (
                "Song not recognized.\n"
                "Confidence: 0.10\n\n"
                "Falling back to detected chords. Please use the chord detection pipeline output."
            )}
        ]
    }

def process_all_charts():
    """Main processing loop."""
    all_examples = []
    holdout_songs = set()
    
    # Collect all chart files
    chart_files = list(CHORD_LIBRARY.rglob("*.json"))
    random.shuffle(chart_files)
    
    # Reserve holdout
    holdout_count = int(len(chart_files) * HOLDOUT_RATIO)
    holdout_files = set(chart_files[:holdout_count])
    train_files = chart_files[holdout_count:]
    
    for chart_path in train_files:
        chart = json.loads(chart_path.read_text())
        lyrics_lines = extract_all_lyrics(chart)
        if not lyrics_lines:
            continue  # Skip instrumental-only charts
        
        full_lyrics = "\n".join(lyrics_lines)
        roots = chords_to_roots(chart.get("chords_used", []))
        
        # Variant 1: Full lyrics
        all_examples.append(make_recall_example(chart, full_lyrics, roots, 0.95 + random.uniform(0, 0.04)))
        
        # Variant 2: Verse + chorus only
        partial = extract_section_lyrics(chart, ["Verse", "Chorus"])
        if partial:
            all_examples.append(make_recall_example(chart, "\n".join(partial), roots, 0.85 + random.uniform(0, 0.09)))
        
        # Variant 3: Chorus only
        chorus = extract_section_lyrics(chart, ["Chorus"])
        if chorus:
            all_examples.append(make_recall_example(chart, "\n".join(chorus), roots, 0.70 + random.uniform(0, 0.14)))
        
        # Variant 4: Whisper errors (x2)
        for _ in range(2):
            noisy = simulate_whisper_errors(full_lyrics, error_rate=random.uniform(0.10, 0.25))
            all_examples.append(make_recall_example(chart, noisy, roots, 0.60 + random.uniform(0, 0.20)))
        
        # Variant 5: Title + artist hint
        all_examples.append(make_recall_example(chart, full_lyrics[:50] + "...", roots, 0.92 + random.uniform(0, 0.07), include_metadata=True))
    
    # Add unknown song examples
    for _ in range(5000):
        all_examples.append(make_unknown_song_example())
    
    # Write output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.shuffle(all_examples)
    
    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")
    
    # Write holdout validation
    val_examples = []
    for chart_path in holdout_files:
        chart = json.loads(chart_path.read_text())
        lyrics_lines = extract_all_lyrics(chart)
        if not lyrics_lines:
            continue
        roots = chords_to_roots(chart.get("chords_used", []))
        val_examples.append(make_recall_example(chart, "\n".join(lyrics_lines), roots, 0.97))
    
    with open(OUTPUT_DIR / "val.jsonl", "w") as f:
        for ex in val_examples:
            f.write(json.dumps(ex) + "\n")
    
    print(f"Training examples: {len(all_examples)}")
    print(f"Validation examples: {len(val_examples)}")

if __name__ == "__main__":
    process_all_charts()
```

---

## 11. Training Script Outline

```python
#!/usr/bin/env python3
"""
chord_recall_train.py

Fine-tunes Phi-3-mini with combined formatter + recall capability.
Designed to run on Modal A10G.

Run: cd ~/stemscribe/backend && ../venv311/bin/python -m modal deploy chord_recall_train.py
"""

import modal

app = modal.App("stemscriber-chord-recall-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1",
        "transformers>=4.38",
        "peft>=0.8",
        "bitsandbytes>=0.42",
        "datasets>=2.16",
        "trl>=0.7",
        "accelerate>=0.26",
        "wandb",
    )
)

volume = modal.Volume.from_name("stemscriber-training", create_if_missing=True)

@app.function(
    image=image,
    gpu=modal.gpu.A10G(),
    timeout=86400,  # 24 hours max
    volumes={"/training": volume},
)
def train():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset
    
    # Load base model (or Stage 1 checkpoint if formatter is done)
    MODEL_ID = "microsoft/Phi-3-mini-128k-instruct"
    STAGE1_CHECKPOINT = "/training/formatter-checkpoint"  # if exists, start from here
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load from Stage 1 checkpoint if available, otherwise from base
    import os
    if os.path.exists(STAGE1_CHECKPOINT):
        print("Loading from Stage 1 (formatter) checkpoint...")
        model = AutoModelForCausalLM.from_pretrained(
            STAGE1_CHECKPOINT, quantization_config=bnb_config, device_map="auto"
        )
    else:
        print("Loading from base Phi-3-mini...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, quantization_config=bnb_config, device_map="auto"
        )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # Load combined dataset (formatter + recall)
    dataset = load_dataset("json", data_files={
        "train": [
            "/training/chart_formatter/train.jsonl",  # formatter examples
            "/training/chord_recall/train.jsonl",      # recall examples
        ],
        "validation": "/training/chord_recall/val.jsonl",
    })
    
    training_args = SFTConfig(
        output_dir="/training/combined-checkpoint",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1.5e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_steps=500,
        eval_steps=500,
        eval_strategy="steps",
        bf16=True,
        max_seq_length=4096,
        dataset_text_field=None,  # using chat format
        report_to="wandb",
    )
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Save final model
    trainer.save_model("/training/combined-final")
    tokenizer.save_pretrained("/training/combined-final")
    
    # Merge LoRA weights for inference
    merged = model.merge_and_unload()
    merged.save_pretrained("/training/combined-merged")
    
    volume.commit()
    print("Training complete. Model saved to /training/combined-merged")
```

---

## 12. Cost and Timeline Estimate

### 12.1 Data Preparation

| Task | Time | Cost |
|------|------|------|
| Write `chord_recall_data_prep.py` (based on outline above) | 2-3 hours | $0 |
| Run data prep on 15,437 charts | ~10 minutes (CPU) | $0 |
| Generate ~107,000 recall examples + 5,000 unknown examples | ~10 minutes (CPU) | $0 |
| Validate data quality (spot-check examples) | 1 hour | $0 |

### 12.2 Training

| Task | Time | Cost |
|------|------|------|
| Wait for Stage 1 (formatter) to complete | In progress | Already budgeted |
| Stage 2: Combined training (~292K examples, 3 epochs) | 12-18 hours | $25-40 Modal A10G |
| Evaluation on holdout set | 1 hour | ~$2 |
| Iteration (if needed, 1-2 retraining runs) | 12-18 hours each | $25-40 each |

### 12.3 Integration

| Task | Time | Cost |
|------|------|------|
| Add recall inference endpoint to Modal | 2-3 hours | $0 |
| Wire recall into `app.py` pipeline | 2-3 hours | $0 |
| Cross-reference validation logic | 1-2 hours | $0 |
| End-to-end testing | 2-3 hours | ~$5 Modal |

### 12.4 Total

| | Optimistic | Realistic |
|--|-----------|-----------|
| **Dev time** | 2 days | 4 days |
| **Modal GPU cost** | $30 | $85 |
| **Calendar time** | 3 days (after formatter done) | 7 days |

---

## 13. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Model memorizes lyrics verbatim** (copyright concern) | High | High | The model necessarily memorizes lyrics to do recall. Mitigation: never serve recalled lyrics to users -- only serve recalled CHORDS. Re-generate lyrics from Whisper on every request. The chord progressions themselves are not copyrightable (factual). |
| **Recall degrades formatting quality** (catastrophic forgetting) | Medium | Medium | Stage 2 includes 60% formatter examples to prevent forgetting. If quality drops, keep models separate. |
| **15K songs insufficient for robust recognition** | Low | Medium | Most uploaded songs will be popular -- the library covers 759 artists with their top songs. Unknown songs fall back gracefully to detection. |
| **Whisper error patterns differ from training augmentation** | Medium | Low | Monitor real-world Whisper outputs and add augmentation patterns as needed. The error simulation is a starting point, not final. |
| **Confidence scores poorly calibrated** | Medium | Medium | Use a temperature-scaled calibration step on the validation set. Adjust confidence thresholds based on real-world performance. |
| **Two songs with identical chorus lyrics** | Low | Low | Detected pitch classes + key break the tie. If still ambiguous, return both candidates. |
| **Model too large for single A10G with both tasks** | Very Low | High | Phi-3-mini (3.8B) with QLoRA uses ~6GB VRAM. Even with both tasks, the model size does not change -- only the training data mix changes. |

---

## 14. Legal Considerations

**Critical: The recall model will have song lyrics baked into its weights.**

This is architecturally unavoidable -- the model recognizes songs BY their lyrics. However:

1. **Chord progressions are not copyrightable.** The recalled chords are factual information.
2. **The model never outputs lyrics from memory.** Lyrics in the output come from Whisper transcription of the user's own audio upload.
3. **The training data (chord library) can be deleted after training.** The model weights encode patterns, not verbatim text (though this is a gray area for LLMs).
4. **Discuss with Alexandra at Morris Music Law** before training. The key question: "Does training a model on copyrighted chord charts + lyrics, where the model only outputs chord progressions (not lyrics) at inference time, constitute fair use?"

**Recommendation:** Raise this in the lawyer call (week of Apr 14). It is likely defensible under fair use (transformative use, factual output, no market substitution), but get legal sign-off before training.

---

## 15. Open Questions for Jeff

1. **Priority vs. formatter?** The formatter is actively training. Should recall be the immediate next step, or are there higher-priority features (vocal onset detection, mobile optimization, etc.)?

2. **Confidence threshold tuning?** The three tiers (0.85/0.50) are starting points. Jeff should test the system and adjust based on what feels right -- when should StemScriber say "I know this song" vs. "I'm guessing"?

3. **Badge UI?** Should the frontend show "recalled" vs. "detected" badges? This is a product decision -- it could build user trust ("this model knows this song") or confuse users.

4. **Chordonomicon license?** The 666K-song Chordonomicon dataset (CC-BY-NC-4.0) could be used for additional recall training. But NC = non-commercial. Need legal clarity.

5. **User correction feedback loop?** When users correct a recalled chart, should those corrections feed back into the model (fine-tuning on corrections)? This would improve accuracy over time but adds complexity.

---

## 16. File References

| File | Purpose |
|------|---------|
| `backend/chord_library/` | 15,437 training charts (759 artists) |
| `backend/training_data/chord_vocabulary_index.json` | 2,364 unique chord symbols |
| `backend/training_data/chart_formatter/train.jsonl` | 184,968 formatter training examples (existing) |
| `backend/training_data/chart_formatter/val.jsonl` | 57 formatter validation examples (existing) |
| `docs/chart-writing-training-plan.md` | Companion plan for Phi-3 chart formatting |
| `docs/ai-chord-model-plan.md` | Companion plan for BTC chord detection |
| `docs/chart-formatter-design.md` | Design doc for existing rule-based formatter |
| `backend/chord_detector_v10.py` | Current BTC chord detection pipeline |
| `backend/chart_formatter.py` | Current rule-based formatter |
| `backend/chart_assembler.py` | Current chord+lyric assembly |
| `backend/word_timestamps.py` | Whisper word-level timing |

---

## 17. Summary: What This Gets Us

**Before (detection only):**
- Upload song -> stem separation -> BTC guesses chords from audio -> 70-80% accuracy
- Extended chords (9ths, dims, aug) often wrong
- No song awareness -- treats every upload as an unknown

**After (recall + detection):**
- Upload song -> Whisper + BTC -> model recognizes song -> recalls correct chart -> ~95%+ accuracy for known songs
- Extended chords are correct because they were memorized, not detected
- Unknown songs still work via detection fallback
- Same Phi-3 model handles both recall and formatting
- +1-2 seconds latency, +$0.001/song cost

The 15,437 charts in the library represent the long tail of songs that real users will upload. A recall model that nails these covers the vast majority of real-world usage, with graceful degradation for everything else.
