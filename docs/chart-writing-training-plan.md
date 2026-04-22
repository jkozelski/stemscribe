# Chart Writing Training Plan

**Date:** 2026-04-04
**Status:** Research & Planning
**Goal:** Train an AI to write properly formatted chord charts -- not just detect chords, but format them like a professional musician would

---

## 1. The Two Skills (Separation of Concerns)

| Skill | What It Does | Status |
|-------|-------------|--------|
| **Listening** | Stem-aware chord detection from audio (BTC fine-tune, spectrograms) | Separate effort -- `ai-chord-model-plan.md` |
| **Writing** | Takes raw chord events + timing + lyrics and produces a clean, readable chart | **THIS PLAN** |

The writing model receives structured input (chord names, timestamps, beat positions, lyrics with word timing, tempo, time signature) and outputs a properly formatted chart. It never touches audio.

---

## 2. What "Correct" Chart Writing Looks Like

### 2.1 Jeff's Handwritten Charts (Gold Standard)

Analyzed 21 JPEG scans at `~/Downloads/KB Chord Charts/`. Key formatting conventions observed:

| Convention | Example from Charts |
|------------|-------------------|
| **Section labels** underlined | _Intro_, _Verse_, _Chorus_, _Bridge_, _Outro_ |
| **Slash notation** for beat counting | `/ / / /` under each chord = 4 beats |
| **Bar lines** separating measures | `|` between groups of 4 slashes |
| **Repeat signs** | `||:` ... `:||` for repeating sections |
| **Repeat counts** | `x4` at the end of a repeated line |
| **Chords above slashes** | Chord name sits directly over its first beat slash |
| **Dashes between chords** | `-` connects chords within a bar or across bars |
| **Multi-bar chords** | `/ / / / - / / / /` = 8 beats on one chord |
| **Numbered endings** | Bracket with `1` and `2` for first/second endings |
| **BPM marking** | `120 BPM` at top left (Climbing The Bars) |
| **Section references** | "Verse 2 -> Chorus 2", "Solo over verse" |
| **Extended chords** | `Amaj7`, `G#m7`, `Bbm(11)`, `F#m7b5`, `C7`, `G#dim` |
| **Slash chords** | `B/C#`, `E/D`, `C/B` |
| **No lyrics on chart** | Rhythm slashes only -- lyrics are in the singer's head |

### 2.2 Formatting Rules (Extracted from Handwritten Charts)

1. **4 slashes per measure** in 4/4 time (3 in 3/4, 6 in 6/8, etc.)
2. **Chord name above the first slash** of the beat where it starts
3. **Dashes** (`-`) between chords in the same line = visual separation
4. **Bar lines** implied by slash grouping (groups of 4 slashes = one bar)
5. **Repeat signs** (`||:` ... `:||`) when a section plays the same progression multiple times
6. **x4, x8** repeat counts when the same 1-2 bar pattern loops
7. **Section labels** always present: Intro, Verse, Pre-Chorus, Chorus, Bridge, Solo, Outro
8. **Numbered sections** when different content: Verse 1, Verse 2 (but NOT Chorus 1, Chorus 2 if same chords)
9. **First/second endings** with bracket notation for variations
10. **No wasted space** -- if a section repeats identically, write it once with repeat signs

### 2.3 Industry Standard Formats

**Ultimate Guitar format:**
```
[Verse 1]
Am                C
Hello darkness my old friend
G                          Am
I've come to talk with you again
```
- Chords above lyrics, positioned at the syllable where the chord change happens
- Section labels in square brackets
- No beat counting -- chord position relative to text implies rhythm

**ChordPro format (v6):**
```
{title: Sound of Silence}
{artist: Simon & Garfunkel}
{key: Am}
{start_of_verse: Verse 1}
[Am]Hello darkness my [C]old friend
[G]I've come to talk with you a[Am]gain
{end_of_verse}
```
- Inline chord brackets before the syllable
- Metadata in curly braces
- Section directives: `{start_of_verse}` / `{end_of_verse}`
- Portable, machine-readable, widely supported

**iReal Pro format:**
```
irealbook://Song=Artist==Style=C==T44*A{C7 |F7 |C7 |C7 |F7 |F7 |C7 |A7 |D-7 |G7 |C7 |G7 }
```
- URL-encoded string with 16 cells per line, max 12 lines
- Chord symbols: `^7` = maj7, `-7` = min7, `o` = dim, `h` = half-dim
- Bar lines: `|`, repeat signs: `{` / `}`, rehearsal marks: `*A`, `*B`
- No lyrics, no melody -- rhythm section reference only
- Can export to MusicXML and MIDI

**MusicXML (W3C standard):**
- `<harmony>` element encodes chord symbols with `<root>`, `<kind>`, `<degree>` sub-elements
- 33 built-in chord kinds plus arbitrary degree alterations
- `<frame>` element for chord diagrams (string/fret positions)
- XML-verbose but the universal interchange format

**Real Book conventions (jazz standard):**
- Melody in standard notation on a single staff
- Chord symbols above the staff, aligned to the beat
- 4 bars per system, 4-8 systems per page
- Rehearsal letters (A, B, C) for sections
- Repeat signs and codas for navigation
- Key/time signature at the beginning only

---

## 3. Training Data Inventory

### 3.1 What We Have

| Source | Count | Format | Contains | Location |
|--------|-------|--------|----------|----------|
| Chord library JSONs | 15,437 | JSON (sections/lines/chords/lyrics) | Section names, chord-over-lyric alignment, chord vocabulary | `backend/chord_library/` (759 artists) |
| Kozelski handwritten charts | 21 | JPEG scans | Slash notation, repeats, section labels, beat counts, BPM | `~/Downloads/KB Chord Charts/` |
| Kozelski library JSONs | 20 | JSON (same schema as library) | Machine-readable versions of the handwritten charts | `backend/chord_library/kozelski/` |
| Chord vocabulary index | 1 | JSON | 2,364 unique chord symbols across all 15,437 songs | `backend/training_data/chord_vocabulary_index.json` |
| Billboard paired audio+labels | 988 | Audio + lab files | Timestamped chord annotations (Harte syntax) | `backend/training_data/btc_finetune/` |
| JAAH jazz annotations | 109 | Lab files | Jazz chord annotations with timestamps | `backend/training_data/jaah/` |

### 3.2 What We Can Get (Open Source, CC-Licensed)

| Source | Count | Format | License | Value |
|--------|-------|--------|---------|-------|
| **Chordonomicon** | 666,000 songs | Harte syntax chords + section labels + genre metadata | CC-BY-NC-4.0 | Massive chord progression + structure training data |
| **McGill Billboard** | ~740 songs | Lab format (timestamped chords + sections) | Research use | Gold-standard human annotations with timing |
| **JAAH (Jazz Audio-Aligned Harmony)** | 113 songs | JSON + lab | CC-BY-NC-SA-4.0 | Rich jazz harmony annotations |
| **Jazznet** | 162,520 patterns | Audio + MIDI | Open | Chords, arpeggios, scales in all keys |
| **iReal Pro forums** | ~4,500 songs | iReal Pro URL format | User-contributed | Jazz standards with section structure, no lyrics |
| **EWLD (Enhanced Wikifonia)** | 5,000+ lead sheets | MusicXML | Research use | Full lead sheets with melody + chords + lyrics |

### 3.3 Jeff's iReal Pro Charts

Jeff has charts in iReal Pro. These can be exported as:
- **MusicXML** (via Export > Chord Chart > MusicXML) -- preserves chords, structure, key, time signature
- **iReal Pro URL strings** -- compact text encoding with all chart data

Both formats are parseable and can become training/validation data.

---

## 4. The Formatting Task (Input/Output Specification)

### 4.1 Input (What the Listening Skill Produces)

```json
{
  "title": "The Time Comes",
  "artist": "Kozelski",
  "tempo_bpm": 120,
  "time_signature": [4, 4],
  "key": "F#m",
  "duration_seconds": 240.0,
  "chord_events": [
    {"time": 0.0, "duration": 3.8, "chord": "F#m", "confidence": 0.92},
    {"time": 3.8, "duration": 3.9, "chord": "A", "confidence": 0.88},
    {"time": 7.7, "duration": 3.7, "chord": "Bm", "confidence": 0.85},
    {"time": 11.4, "duration": 4.1, "chord": "E", "confidence": 0.90}
  ],
  "word_timestamps": [
    {"word": "You", "start": 15.5, "end": 15.8},
    {"word": "say", "start": 15.9, "end": 16.3},
    {"word": "you", "start": 16.4, "end": 16.6},
    {"word": "wanted", "start": 16.7, "end": 17.2}
  ],
  "beat_times": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
  "downbeat_times": [0.0, 2.0, 4.0, 6.0]
}
```

### 4.2 Output Option A: StemScriber JSON (Current Library Format)

```json
{
  "title": "The Time Comes",
  "artist": "Kozelski",
  "key": "F#m",
  "capo": 0,
  "source": "StemScriber AI",
  "chords_used": ["F#m", "A", "Bm", "E", "B", "D", "C#m", "G"],
  "sections": [
    {
      "name": "Intro",
      "lines": [
        {"chords": "F#m  A  Bm  E", "lyrics": null}
      ],
      "repeat": 2
    },
    {
      "name": "Verse",
      "lines": [
        {"chords": "F#m  A  F#m  B", "lyrics": "You say you wanted someone to take you away"}
      ],
      "repeat": 4
    }
  ]
}
```

### 4.3 Output Option B: Slash Notation (Jeff's Handwritten Style)

```
The Time Comes - Kozelski
Key: F#m  Tempo: 120 BPM  Time: 4/4

Intro
||: F#m    -  A     -  Bm    -  E       :||
    / / / /   / / / /  / / / /  / / / /

Verse (x4)
||: F#m    -  A     -  F#m   -  B       :||
    / / / /   / / / /  / / / /  / / / /

Pre-Chorus
  D         -  C#m      Bm
  / / / /      / / / /  / / / / - / / / /
  D         -  C#m      G
  / / / /      / / / /  / / / / - / / / /

Chorus (x4)
||: F#m    -  A  -  E     -  B          :||
    / / / /   / / / /  / / / /  / / / /
```

### 4.4 Output Option C: ChordPro (Interop Format)

```
{title: The Time Comes}
{artist: Kozelski}
{key: F#m}
{tempo: 120}
{time: 4/4}
{start_of_verse: Intro}
||: [F#m] / / / / | [A] / / / / | [Bm] / / / / | [E] / / / / :||
{end_of_verse}
{start_of_verse: Verse 1}
[F#m]You say you [A]wanted someone to [F#m]take you a[B]way
{end_of_verse}
```

### 4.5 Recommendation: Train on All Three

The model should learn to produce all three output formats from the same input. The StemScriber JSON is the internal format (what the frontend renders). The slash notation is the "musician's chart" (what Jeff writes by hand). ChordPro is the interop format (export to other apps). A single model can handle this via a format token at the start of the output: `<json>`, `<slash>`, `<chordpro>`.

---

## 5. Training Approach

### 5.1 Why an LLM, Not a Rule-Based Formatter

We already have a rule-based formatter (`chart_formatter.py` + `chart_assembler.py`). It works but has known limitations:

1. **Section detection is fragile** -- uses chord progression fingerprinting (interval sequences), fails on songs with similar progressions in verse and chorus
2. **No repeat detection** -- never outputs repeat signs or x4 counts; writes every repetition out fully
3. **Beat quantization is approximate** -- snaps chords to nearest word by timestamp, not to actual beat grid
4. **No awareness of musical conventions** -- doesn't know that a 4-bar intro usually repeats, or that bridges are typically 8 bars
5. **Chord-lyric alignment is purely positional** -- places chord at character offset, doesn't understand phrase structure

An LLM trained on 15,000+ correctly formatted charts will internalize these conventions. It will learn that:
- Intros are usually 2-4 bars and often repeat
- Verses have the same chords with different lyrics
- Pre-choruses are short transition sections
- Choruses are typically louder/higher energy with different harmonic rhythm
- Bridges appear once, usually 8 bars
- The pattern `|: Am | G | F | C :|` with x4 is more readable than writing it out 4 times

### 5.2 Model Selection

| Model | Size | Context | Training Cost | Inference Cost | Recommendation |
|-------|------|---------|--------------|----------------|----------------|
| **Phi-3-mini (3.8B)** | 3.8B | 128K | ~$20-40 on Modal A10G | ~0.5s/chart on A10G | **Top pick** -- small, fast, long context for full songs |
| **Llama-3.2-3B** | 3B | 128K | ~$15-30 on Modal A10G | ~0.4s/chart on A10G | Strong alternative, good instruction-following |
| **Qwen2.5-3B** | 3B | 32K | ~$15-30 | ~0.4s/chart | Good multilingual base, 32K context sufficient |
| **Gemma-2-2B** | 2.6B | 8K | ~$10-20 | ~0.3s/chart | Smallest, but 8K context may be too short for long songs |
| **Mistral-7B** | 7B | 32K | ~$40-80 | ~1s/chart | Larger = better quality but 2x cost |

**Recommendation: Phi-3-mini (3.8B) with QLoRA fine-tuning.**

Rationale:
- 128K context handles any song length
- 3.8B parameters is large enough to learn formatting conventions but small enough to run on a single A10G
- QLoRA (4-bit quantization + LoRA adapters) reduces VRAM to ~6GB, fits easily on Modal A10G (24GB)
- Microsoft's Phi-3 family has shown strong performance on structured output tasks
- Inference at ~0.5s/chart means negligible added cost to the pipeline ($0.0003/chart)

### 5.3 Training Data Preparation

#### Phase 1: Convert Existing Library to Training Pairs (15,437 examples)

Each chord library JSON becomes a training pair:

**Input construction (synthetic):**
- Parse the JSON to extract chord sequence and section structure
- Generate synthetic `chord_events` with plausible timestamps (based on typical tempo + beat grid)
- Generate synthetic `word_timestamps` from lyrics (estimate ~3 words/second)
- Add tempo/key/time_signature metadata

**Output:** The original JSON as-is (it's already correctly formatted by humans)

```python
# Pseudo-code for data prep
for song_json in chord_library:
    synthetic_input = generate_synthetic_timestamps(song_json)
    training_pair = {
        "input": json.dumps(synthetic_input),
        "output": json.dumps(song_json)  # human-formatted chart
    }
```

This gives us 15,437 input/output pairs where the output is known-correct formatting.

#### Phase 2: Augment with Chordonomicon (up to 666K additional)

Download the Chordonomicon dataset from Hugging Face (`ailsntua/Chordonomicon`). It includes:
- Chord progressions in Harte syntax
- Section labels (verse, chorus, bridge, etc.)
- Genre and sub-genre metadata
- Release year

Convert each to our training format. The Chordonomicon has section structure but no lyrics, so these examples train section detection and repeat pattern recognition specifically.

#### Phase 3: Handwritten Chart Validation Set (20 examples)

The 20 Kozelski handwritten charts + their 20 JSON counterparts become the validation/test set. The model's output should match Jeff's handwriting conventions:
- Correct section labels
- Correct repeat signs and counts
- Correct beat slashes per chord
- Correct chord names

These 20 examples are NEVER used for training -- they are the gold standard for evaluation.

#### Phase 4: Billboard + JAAH Timing-Accurate Examples (1,097 examples)

The Billboard (988) and JAAH (109) datasets have real audio timestamps. Convert their lab-format annotations to training pairs where the input timestamps are real (not synthetic). These examples teach the model to handle real-world timing noise.

### 5.4 Training Prompt Template

```
<|system|>
You are a professional chord chart formatter. Given raw chord detection
data with timestamps, format a clean, readable chord chart.
<|user|>
Format this as a {format_type} chord chart:

Title: {title}
Artist: {artist}
Key: {key}
Tempo: {tempo} BPM
Time Signature: {time_sig}

Chord events:
{chord_events_json}

Lyrics with timing:
{word_timestamps_json}

Beat grid:
{beat_times}
<|assistant|>
{formatted_chart_output}
```

### 5.5 QLoRA Training Configuration

```yaml
# QLoRA config for Phi-3-mini
base_model: microsoft/Phi-3-mini-128k-instruct
quantization: 4bit (bitsandbytes nf4)
lora_r: 64
lora_alpha: 16
lora_dropout: 0.1
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

# Training
learning_rate: 2e-4
warmup_ratio: 0.05
lr_scheduler: cosine
num_epochs: 3
batch_size: 4
gradient_accumulation: 4
max_seq_length: 4096  # most charts fit in 2K tokens; 4K for safety

# Hardware
gpu: Modal A10G (24GB VRAM)
estimated_time: ~4 hours for 15K examples, ~24 hours with Chordonomicon
estimated_cost: $8-50 on Modal
```

### 5.6 Multi-Format Training Strategy

Include all three output formats in training, with a format token:

- 60% StemScriber JSON format (`<json>`) -- primary output for the app
- 25% Slash notation format (`<slash>`) -- matches Jeff's handwriting
- 15% ChordPro format (`<chordpro>`) -- interop/export

The model learns all three, selected at inference by the format token.

---

## 6. Evaluation Protocol

### 6.1 Automated Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| **Section label accuracy** | Does the model correctly identify Intro/Verse/Chorus/Bridge/Outro? | > 90% |
| **Repeat detection F1** | Does it correctly identify repeating sections and use repeat signs? | > 85% |
| **Beat count accuracy** | Do the slash counts match the actual beat durations? | > 95% |
| **Chord placement accuracy** | Are chords above the correct beat/word? | > 90% |
| **JSON validity** | Does the JSON output parse without errors? | 100% |
| **Chord vocabulary preservation** | Are all input chords present in the output? No hallucinated chords? | 100% |
| **ChordPro validity** | Does ChordPro output parse in a ChordPro renderer? | 100% |

### 6.2 Kozelski Validation (Human Eval)

For each of the 20 Kozelski songs:
1. Feed the model the chord events + timing from the JSON
2. Generate slash notation output
3. Compare side-by-side with Jeff's handwritten JPEG scan
4. Score on: section labels match, repeat signs match, beat counts match, chord names match, overall layout similarity

Target: the AI's output for "The Time Comes" should look functionally identical to the handwritten chart at `~/Downloads/KB Chord Charts/The TIme Comes.jpeg`.

### 6.3 A/B Testing in Production

Once deployed, show users both:
- (A) Rule-based formatter output (`chart_formatter.py`)
- (B) LLM formatter output

Collect preference clicks. The LLM output should win > 70% of the time to justify the added inference cost.

---

## 7. Integration Plan

### 7.1 Pipeline Position

```
Upload audio
  -> Stem separation (Modal A10G)                    [existing]
  -> BTC chord detection on harmonic mix              [existing]
  -> Whisper lyrics from vocal stem                   [existing]
  -> Beat/downbeat detection (madmom or librosa)      [ADD THIS]
  -> Chart Writing LLM (Phi-3-mini QLoRA)             [ADD THIS]
  -> Serve to frontend                                [existing]
```

### 7.2 Inference on Modal

Deploy the fine-tuned Phi-3-mini on Modal alongside the existing stem separation:
- Same A10G GPU class
- Load model once, keep warm for subsequent requests
- Inference time: ~0.5s per chart (negligible vs. 30-60s stem separation)
- Cost: ~$0.0003 per chart (vs. $0.06 for stem separation)

### 7.3 Fallback

If the LLM produces invalid JSON or takes > 5s, fall back to the existing `chart_formatter.py` rule-based output. Log the failure for debugging.

### 7.4 Export Formats

With the multi-format model, StemScriber can offer:
- **View in app** -- StemScriber JSON rendered by frontend
- **Download ChordPro** -- `.cho` file compatible with OnSong, SongBook, BandHelper
- **Download PDF** -- slash notation rendered to PDF (via a simple text-to-PDF converter)
- **Download iReal Pro** -- convert slash notation to iReal Pro URL format (straightforward mapping)
- **Download MusicXML** -- for Finale, Sibelius, MuseScore

---

## 8. Research Insights

### 8.1 Representation Format Matters

The Phi-3-MusiX paper found that **compact symbolic formats significantly outperform verbose JSON** for music notation tasks in LLMs. This means:
- For internal training, keep the compact chord library JSON format (it's already concise)
- Do NOT expand to full MusicXML during training -- too verbose, wastes context
- Consider a kern+-like compact notation for the intermediate representation

### 8.2 ChatMusician Approach (Relevant Precedent)

ChatMusician (LLaMA2-7B fine-tuned on ABC notation) demonstrated that:
- LLMs can learn music notation as a "second language"
- Continual pre-training on music text helps (not just fine-tuning)
- 1.1M training samples were used (we have 15K-666K depending on Chordonomicon)
- LoRA with r=64, alpha=16, dropout=0.1 worked well (we'll use same config)

### 8.3 Chordonomicon as Augmentation

The Chordonomicon (666K songs, CC-BY-NC-4.0) provides:
- Chord progressions in standardized Harte syntax
- Section structure labels
- Genre metadata (can train genre-aware formatting)
- 20x larger than our chord library

Caveat: Chordonomicon has chord progressions but NO lyrics. It trains section detection and repeat pattern recognition, not chord-lyric alignment. Use it as augmentation, not primary data.

### 8.4 iReal Pro as Validation Source

If Jeff exports his iReal Pro charts, they provide:
- Beat-accurate chord placement (4 cells per bar)
- Section markers and repeat signs
- Professional-grade formatting
- A second validation set beyond the handwritten charts

---

## 9. Implementation Timeline

| Phase | What | Duration | Cost |
|-------|------|----------|------|
| **Phase 1: Data Prep** | Convert 15,437 library JSONs to training pairs. Parse Kozelski handwritten charts for validation set. Download Chordonomicon. | 2-3 days | $0 |
| **Phase 2: Baseline Training** | QLoRA fine-tune Phi-3-mini on 15K library examples. Evaluate against Kozelski validation set. | 1 day | ~$10 Modal |
| **Phase 3: Augmented Training** | Add Chordonomicon examples. Add Billboard/JAAH timing-accurate examples. Retrain. | 1 day | ~$30 Modal |
| **Phase 4: Multi-Format** | Add slash notation and ChordPro output formats. Retrain with format tokens. | 1 day | ~$15 Modal |
| **Phase 5: Integration** | Deploy on Modal. Wire into pipeline after chord detection + Whisper. Add fallback to rule-based formatter. | 1-2 days | ~$5 Modal |
| **Phase 6: Evaluation** | A/B test against rule-based formatter. Collect user preferences. Iterate. | Ongoing | Minimal |

**Total estimated cost: $50-60 in Modal GPU time.**
**Total estimated time: 7-10 days.**

---

## 10. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| LLM hallucinates chords not in input | Medium | High | Post-process: validate all output chords exist in input chord_events. Strip any hallucinated chords. |
| JSON output is malformed | Low | Medium | Constrained decoding (force valid JSON tokens). Fallback to rule-based formatter. |
| Section labels wrong (calls Verse a Chorus) | Medium | Medium | Train on section detection specifically with Chordonomicon's 666K labeled examples. |
| Repeat detection over-aggressive | Medium | Low | Conservative repeat threshold: only merge if progression matches exactly >= 3 times. |
| Model too slow for real-time | Low | High | Phi-3-mini at 3.8B is fast (~0.5s). If needed, distill to 1B or use vLLM for batched inference. |
| Chordonomicon license (CC-BY-NC-4.0) | Low | Medium | StemScriber is a commercial product. NC = non-commercial. Use Chordonomicon for research/validation only, not as primary training data, OR contact authors for commercial license. |
| 15K examples insufficient for good formatting | Medium | Medium | Augment with synthetic variations (transpose all songs to all 12 keys = 185K examples). Add Chordonomicon if license permits. |

---

## 11. Open Questions

1. **Should we add beat detection to the pipeline?** Madmom or librosa beat tracking would give us downbeat times, enabling proper measure-level quantization. The handwritten charts are all beat-aligned -- the model needs beat input to produce beat-aligned output.

2. **Should the model also learn capo suggestions?** Many charts include capo position. The model could learn: "if the song is in Bb and uses barre chords, suggest capo 1 and play in A shapes."

3. **Should we support Nashville Number System?** Nashville charts use numbers instead of chord names (1, 4, 5 instead of C, F, G). Popular with session musicians. Easy to add as a fourth output format.

4. **iReal Pro export priority?** If Jeff uses iReal Pro, direct export to iReal Pro URL format would be a differentiating feature. The URL format is simple enough to generate with a template (no LLM needed).

5. **Chordonomicon commercial license?** Need to verify whether CC-BY-NC-4.0 allows using the data to train a model deployed in a commercial product. If not, contact the authors (NTUA Athens) for a commercial license or use only the 15K library + Billboard + JAAH data.

---

## 12. File References

| File | Purpose |
|------|---------|
| `backend/chart_formatter.py` | Existing rule-based formatter (to be replaced/augmented) |
| `backend/chart_assembler.py` | Existing chord+lyric assembly (to be replaced/augmented) |
| `backend/chord_library/` | 15,437 training examples (759 artists) |
| `backend/chord_library/kozelski/` | 20 Kozelski charts (validation set) |
| `~/Downloads/KB Chord Charts/*.jpeg` | 21 handwritten gold-standard charts |
| `backend/training_data/chord_vocabulary_index.json` | 2,364 unique chord symbols |
| `backend/training_data/btc_finetune/` | 988 Billboard audio+label pairs |
| `backend/training_data/jaah/` | 109 JAAH jazz annotations |
| `docs/ai-chord-model-plan.md` | Companion plan for chord detection (listening skill) |
| `docs/chart-formatter-design.md` | Design doc for existing rule-based formatter |
| `docs/btc-finetune-status.md` | Status of BTC chord detection fine-tuning |
