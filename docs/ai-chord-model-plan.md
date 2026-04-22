# AI Chord Model Plan — StemScriber

**Date:** 2026-04-04
**Status:** Research / Architecture Decision

---

## 1. Current State

### What We Have
- **15,417 chord chart JSONs** (108MB) at `backend/chord_library/` covering 758 artists
- **BTC (Bi-directional Transformer for Chords)** — pre-trained model already integrated (`chord_detector_v10.py`), 12MB model, 25-chord vocabulary (small) or 170-chord vocabulary (large)
- **988 paired audio+label training examples** already prepared at `backend/training_data/btc_finetune/` from the Billboard dataset
- **faster-whisper** for word-level lyrics timestamps (`word_timestamps.py`)
- **User correction pipeline** already collecting chord and lyrics corrections (`routes/feedback.py`)
- **Chart assembler** that combines chords + lyrics + structure into renderable JSON (`chart_assembler.py`)
- **Modal A10G GPU** already in use for stem separation ($0.000575/sec = ~$2.07/hr)

### Current Pipeline Flow
```
Upload audio
  -> Stem separation (Modal A10G)
  -> Mix harmonic stems (guitar + piano + bass)
  -> BTC chord detection on harmonic mix
  -> Post-processing (merge, min-duration, rare-chord filter, key-weighting)
  -> Whisper lyrics from vocal stem
  -> Chart assembler merges chords + lyrics
  -> Serve to frontend
```

### The Problem
The chord library contains copyrighted lyrics served as stored files. The legal requirement is to stop serving stored charts and instead generate everything from the user's uploaded audio in real time. The library becomes training data only, then gets deleted from the production server.

### Chord Library Data Format
Each JSON contains:
```json
{
  "title": "Play That Funky Music",
  "artist": "Wild Cherry",
  "key": null,
  "capo": 0,
  "chords_used": ["E9", "G9", "Bb7", "Em", "D", "Bm", "A7", "A9"],
  "sections": [
    {
      "name": "Verse 1",
      "lines": [
        {"chords": "E9", "lyrics": "Hey once I was a boogie singer..."},
        {"chords": null, "lyrics": "I never had no problems..."}
      ]
    }
  ]
}
```

Key observations:
- Chords are **not timestamped** — they are positionally aligned with lyrics text
- Section names are human-labeled (Verse, Chorus, Pre-Chorus, Solo, Bridge, Outro, etc.)
- `chords_used` gives the full chord vocabulary for each song
- Many songs use extended/jazz chords (E9, Bbmaj7, D7+5, C#dim, etc.)
- Some lines have chords but no lyrics (instrumentals), some have lyrics but no chords

---

## 2. Architecture Decision: Fine-Tune BTC, Not Train From Scratch

### Recommendation: Fine-tune BTC-ISMIR19 on chord library vocabulary

**Why not train from scratch:**
- BTC is already state-of-the-art for chord recognition (2019 ISMIR paper)
- It already understands audio-to-chord mapping from CQT spectrograms
- 15K text-only charts cannot train an audio model — there is no paired audio
- Training a new audio model from scratch would need 50K+ paired audio/label examples and weeks of GPU time

**Why fine-tuning BTC is the right move:**
- BTC's small vocabulary (25 chords: maj, min, 7, maj7, min7, dim, aug, sus2, sus4, N for 12 roots) misses the extended chords in the library (9ths, 11ths, add9, dim7, hdim7, 6ths, etc.)
- The large vocabulary model (170 chords) exists but has lower accuracy
- We can fine-tune on the 988 Billboard pairs we already have, plus create new pairs
- Fine-tuning is 10-100x cheaper than training from scratch

### Three-Phase Approach

---

## 3. Phase 1: Expand BTC Vocabulary via Fine-Tuning

**Goal:** Improve chord detection accuracy, especially for extended chords.

### 3a. Build a Chord Vocabulary from the Library

Extract the complete chord vocabulary across all 15,417 charts:

```python
# Pseudocode — data prep script
all_chords = set()
for json_file in chord_library:
    chart = json.load(json_file)
    all_chords.update(chart.get("chords_used", []))
```

Expected result: ~200-400 unique chord types. Map these to BTC's 170-chord large vocabulary classes, identify gaps, and define a target vocabulary (likely 170-250 classes).

### 3b. Create Training Pairs from Existing Data

We already have 988 Billboard audio+label pairs. To expand:

1. **Billboard dataset (already done):** 988 songs with `.wav` + `.lab` files (start_time, end_time, chord label format). Ready to use.

2. **JAAH dataset (already downloaded):** Jazz audio with annotations at `training_data/jaah/`. Convert annotations to BTC `.lab` format.

3. **Synthesize training audio from chord charts:** For the 15,417 library charts where we do NOT have audio:
   - Use a MIDI synthesizer to generate chord audio (piano voicings for each chord)
   - FluidSynth or similar can render chords with proper voicings
   - This gives us timestamped chord audio paired with known labels
   - Quality is lower than real recordings, but useful for vocabulary expansion
   - Estimate: 15K synthetic training pairs in ~4 hours of generation

4. **User uploads with correction feedback:** As users submit chord corrections via the existing feedback system, these become high-value training pairs (real audio + human-verified chords).

### 3c. Fine-Tuning Procedure

```
Input:  CQT spectrogram features (144 bins, 24 bins/octave)
Model:  BTC transformer (8 layers, 4 heads, 128 hidden)
Output: Chord class per frame (expanded vocabulary)

Training:
  - Start from pre-trained btc_model_large_voca.pt (170 classes)
  - Add new output classes for chords not in vocabulary
  - Freeze early transformer layers, fine-tune last 2-3 layers + new output head
  - Learning rate: 1e-5 (10x lower than original 1e-4)
  - Batch size: 128, max epochs: 50
  - Data: 988 Billboard + JAAH + synthetic chord audio
```

### Cost Estimate — Phase 1
- **Data prep:** ~8 hours developer time (scripting)
- **Synthetic audio generation:** ~4 hours on CPU (FluidSynth)
- **Fine-tuning on Modal A10G:** ~2-4 hours ($4-8)
- **Total:** ~$10 compute + 2 days dev time

---

## 4. Phase 2: Song Recognition + Chord Vocabulary Constraint

**Goal:** When a user uploads a known song, use the library's chord vocabulary to constrain and correct BTC output.

This is the highest-value use of the 15,417 charts — not as training data for the audio model, but as a **knowledge base** for post-processing.

### 4a. Song Fingerprinting / Matching

When a user uploads audio:
1. Extract audio fingerprint (Chromaprint/AcoustID or spectral embedding)
2. Compare against a fingerprint database built from the library metadata
3. If matched: retrieve the known `chords_used` list for that song

**Simpler alternative (already partially built):** The existing `_lookup_vocab()` function in `chord_detector_v10.py` already does artist+title matching against a chord database AND scrapes Ultimate Guitar as fallback. This can be enhanced:
- Build a title/artist lookup index from all 15,417 charts
- Use fuzzy matching (already extracting artist/title from upload metadata)
- Store only `{title, artist, chords_used, key, sections[].name}` — no lyrics

### 4b. Vocabulary-Constrained Decoding

When we know the song's chord vocabulary (say `["E9", "G9", "Bb7", "Em", "D", "Bm", "A7"]`):

1. Run BTC normally to get frame-level chord probabilities
2. **Mask** the output softmax to only allow chords in the known vocabulary (+ "N" for no chord)
3. This dramatically reduces errors — instead of choosing from 170+ chords, BTC chooses from ~8

This is already partially implemented in `chord_detector_v10.py`'s vocabulary lookup system. The enhancement is:
- Pre-build the vocabulary index from the full 15,417-chart library
- Store as a lightweight JSON lookup (title+artist -> chords_used), ~2MB
- Delete the full chord charts (with lyrics) after extracting this index

### 4c. Section Structure Prior

The library also gives us section structure knowledge:
- "Verse 1 typically has chords X, Y, Z; Chorus has A, B, C"
- Section ordering patterns (Intro -> Verse -> Chorus -> Verse -> Chorus -> Bridge -> Chorus -> Outro)

This can improve the chart assembler's section detection — currently it just groups lines by count.

### Cost Estimate — Phase 2
- **Vocabulary index extraction:** 1 hour script
- **Fingerprint database (if doing audio matching):** ~$5 on Modal (process library metadata)
- **Constrained decoding implementation:** 1 day dev
- **Total:** ~$5 compute + 1.5 days dev time

---

## 5. Phase 3: Lyrics Correction via Contextual LLM

**Goal:** Use an LLM to clean up Whisper transcription errors using song context.

### The Problem with Whisper Alone
- Whisper is excellent at general speech but makes errors on:
  - Song-specific vocabulary ("funky music" might become "monkey music")
  - Mumbled/stylized vocals
  - Backing vocal bleed
  - Repeated refrains (sometimes merges or drops repeats)

### Solution: Post-Processing with a Small LLM

After Whisper transcribes the vocal stem, run a lightweight LLM to:
1. Fix common Whisper errors using song context (title, artist, detected chords)
2. Align lyrics to detected section structure
3. Format into the chord chart JSON structure

**Model options:**
- **Phi-3-mini (3.8B)** or **Llama-3.2-3B**: Small enough to run on Modal A10G in ~2-3 seconds per song
- **Fine-tuned on our data:** Use the 15,417 charts as training data for lyrics formatting (the LLM learns section naming conventions, chord-lyric alignment patterns, and common song structures)
- **No lyrics stored:** The LLM learns *patterns* (verse structures, chorus repetitions) not specific copyrighted lyrics

### Training Data Preparation for LLM

From each chart, create training examples:
```
Input:  "Title: Play That Funky Music | Artist: Wild Cherry | 
         Key: E | Chords: E9 G9 Bb7 Em D Bm A7 |
         Raw Whisper: [timestamped word list from vocal stem]"

Output: "Formatted chord chart JSON with sections, chord-lyric alignment"
```

Since we don't have the original audio for most library songs, the LLM training focuses on:
- Section labeling given lyrics patterns (verse vs chorus detection)
- Chord-lyric alignment formatting
- Common correction patterns (proper nouns, musical terms)

### Cost Estimate — Phase 3
- **Data prep:** 2 days (format 15K charts into LLM training pairs)
- **Fine-tune Phi-3-mini on Modal A10G:** ~8-12 hours ($17-25)
- **Inference cost per song:** ~$0.002 (2-3 seconds on A10G)
- **Total:** ~$25 compute + 3 days dev time

---

## 6. Unified Runtime Pipeline (Post-Training)

```
User uploads audio
  |
  v
[Stem Separation] — Modal A10G (existing, ~60s)
  |
  v
[Chord Detection] — Fine-tuned BTC (existing pipeline, ~5s)
  |                  + vocabulary constraint if song recognized
  v
[Lyrics Extraction] — faster-whisper on vocal stem (existing, ~10s)
  |
  v
[Chart Assembly] — Fine-tuned LLM post-processor (~3s)
  |                 Corrects Whisper errors
  |                 Assigns section labels  
  |                 Aligns chords with lyrics
  v
[Output] — Chord chart JSON (same format as current library)
```

**Total added latency:** ~3 seconds (LLM step only — everything else already runs)
**Total added cost per song:** ~$0.002

---

## 7. Data Extraction Before Library Deletion

Before deleting the chord library from production, extract and retain:

| Keep | Delete |
|------|--------|
| `{title, artist, chords_used, key, capo}` per song (vocabulary index, ~2MB) | Full lyrics text |
| Section name patterns (Intro/Verse/Chorus/Bridge/Solo/Outro frequencies) | Chord-lyric alignment |
| Chord transition statistics (what chord follows what) | Any copyrighted content |
| Trained model weights (encode patterns, not content) | Raw JSON files |

The vocabulary index is purely factual (chord names are not copyrightable) and serves as the constraint database for Phase 2.

---

## 8. Implementation Timeline

| Phase | What | Dev Time | Compute Cost | Dependency |
|-------|------|----------|-------------|------------|
| **0** | Extract vocabulary index + chord stats from library | 0.5 days | $0 | None |
| **1** | Fine-tune BTC on expanded vocabulary | 2 days | ~$10 | Phase 0 |
| **2** | Song recognition + constrained decoding | 1.5 days | ~$5 | Phase 0 |
| **3** | LLM lyrics post-processor | 3 days | ~$25 | Phase 1 |
| **4** | Delete chord library from production | 0.5 days | $0 | Phases 1-3 |
| **5** | User feedback loop (ongoing) | ongoing | ~$0 | Phase 1 |

**Total: ~7.5 dev days, ~$40 compute**

### Priority Order
1. **Phase 0 + Phase 2 first** — Highest immediate value. Extract the vocabulary index, wire up constrained decoding. This alone will significantly improve chord accuracy for known songs with zero training cost.
2. **Phase 1 second** — Fine-tune BTC for better base accuracy on extended chords.
3. **Phase 3 third** — LLM lyrics post-processing. This is the most complex but also the most valuable for the legal requirement (generating lyrics from audio rather than serving stored text).
4. **Phase 4 last** — Delete the library only after all phases are validated.

---

## 9. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Synthetic chord audio doesn't transfer well to real recordings | Phase 1 fine-tuning less effective | Use Billboard+JAAH real audio as primary training data; synthetic as supplement only |
| LLM hallucinates lyrics not in the audio | Legal liability — serving made-up lyrics | Constrain LLM to only edit Whisper output, never generate from scratch; confidence thresholds |
| Vocabulary index still considered derivative work | Legal challenge | Chord names are factual/functional (not copyrightable); discuss with Alexandra |
| BTC fine-tuning overfits on small dataset | Worse accuracy on unseen songs | Early stopping, validation split, data augmentation (pitch shift, tempo change) |
| Latency increase from LLM step | User experience regression | LLM runs in parallel with MIDI transcription (already ~30s); net zero added wall time |

---

## 10. What NOT To Do

- **Do NOT train a new chord detection model from scratch.** BTC is battle-tested and the marginal gain from a custom architecture does not justify the cost.
- **Do NOT try to use the 15K text-only charts as audio training data.** There is no paired audio. The charts are valuable as vocabulary constraints and section structure priors, not as audio model training data.
- **Do NOT serve the vocabulary index to users.** It stays server-side for model inference only.
- **Do NOT fine-tune Whisper itself.** Whisper is already excellent at transcription; the errors are best fixed with post-processing context, not by retraining a 1.5B parameter model on insufficient data.
- **Do NOT store generated lyrics persistently.** Generate on every request from the user's uploaded audio. Cache only within the session/job lifetime.
