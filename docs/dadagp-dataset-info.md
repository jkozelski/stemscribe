# DadaGP Dataset Research

**Date:** 2026-04-04
**Purpose:** Evaluate DadaGP for training guitar tab transcription models (trimplexx CRNN, MIDI-to-Tab)

---

## 1. What Is DadaGP?

DadaGP is a symbolic music dataset of **26,181 GuitarPro song scores** across **739 musical genres**, released alongside an encoder/decoder that converts GuitarPro files to/from a token-sequence format designed for generative language models (GPT-2, TransformerXL, etc.).

- **Paper:** "DadaGP: A Dataset of Tokenized GuitarPro Songs for Sequence Models" (ISMIR 2021)
- **Authors:** Pedro Sarmento, Adarsh Kumar, CJ Carr, Zack Zukowski, Mathieu Barthet, Yi-Hsuan Yang
- **Institutions:** Queen Mary University of London + Dadabots
- **Paper PDF:** https://archives.ismir.net/ismir2021/paper/000076.pdf
- **GitHub (encoder/decoder):** https://github.com/dada-bots/dadaGP
- **arXiv:** https://arxiv.org/abs/2107.14653

---

## 2. Download and Access

**The dataset is NOT freely downloadable.** Access is by request only.

- **How to request:** Contact Dadabots or Pedro Sarmento via email or Twitter (@dadabots / @umpedronosapato)
- **Hosted on:** Zenodo (on application by request)
- **Dataset size:** Not publicly documented in GB. The tokenized text is ~116 million tokens (roughly comparable to WikiText-103 in size). The GuitarPro binary files are likely in the low single-digit GB range.
- **Encoder/decoder code:** Publicly available on GitHub (MIT License)

---

## 3. File Formats

DadaGP includes **two representations** of the same songs:

### a) GuitarPro binary files (.gp3, .gp4, .gp5)
- Standard GuitarPro format, readable by Guitar Pro software, TuxGuitar, PyGuitarPro
- Contains full multi-instrument scores with tab notation

### b) Tokenized text files (.txt)
- One token per line, event-based encoding inspired by MIDI event sequences
- **116 million tokens** total across the dataset
- Avg song length: ~2:45, most songs have ~4 instrumental parts (2 guitars, bass, drums)

**NO audio files are included.** This is notation-only.

---

## 4. Token Format

The DadaGP token vocabulary includes:

| Token Type | Example | Meaning |
|-----------|---------|---------|
| **Pitched note** | `distorted_guitar:note:3:5` | instrument:note:string:fret |
| **Rest** | `clean_guitar:note:rest` | instrument:note:rest |
| **Drum note** | `drums:note:snare` | drums:note:type |
| **Wait (timing)** | `wait:480` | Time gap in ticks (960 ticks = quarter note) |
| **Measure** | `new_measure` | Bar boundary |
| **Tempo** | `tempo:120` | BPM change |
| **Note effects** | `nfx:trill:fret5:duration8` | Trill with target fret and duration |
| **Beat effects** | `bfx:palmMute` | Beat-level articulation |
| **Metadata** | `artist:metallica` | Song metadata |
| **Tuning** | `downtune:1` | Half-steps down from standard |
| **Structure** | `start`, `end` | Song boundaries |

**Key details:**
- Tick resolution: 960 ticks per quarter note
- Eighth notes separated by `wait:480`, sixteenth notes by `wait:240`
- Note durations are implicit (derived from wait tokens between events)
- String numbering: 0-indexed from highest string
- Fret numbering: standard (0 = open, 1-24)
- Rock subset vocabulary: ~2,104 unique tokens

---

## 5. Data Source

The GuitarPro files were sourced from **Ultimate Guitar**, which hosts over 200,000 user-submitted GuitarPro transcriptions. Genre metadata was acquired via the **Spotify Web API** (artist + title lookup).

This means the underlying data is **crowd-sourced transcriptions of copyrighted songs** -- the same legal territory as Songsterr's data.

---

## 6. License and Commercial Use

**This is the critical question for StemScriber.**

| Component | License | Commercial OK? |
|-----------|---------|----------------|
| **Encoder/decoder code** (dadagp.py) | MIT License | Yes |
| **Dataset itself** (26K GP files + tokens) | No explicit license stated | Unclear |
| **Access model** | By request, "for research purposes" | Suggests research-only intent |

### Legal analysis:

1. **The code is MIT** -- we can freely use the encoder/decoder for any purpose.

2. **The dataset has no formal license.** The paper and README say "contact us for research purposes." This is NOT a permissive license. Using it for commercial model training without explicit permission would be legally risky.

3. **The underlying songs are copyrighted.** The GuitarPro files are user transcriptions of copyrighted music (same issue as training on Songsterr data). The paper itself raises the question: "Is it wrong to train machine learning models on copyrighted music?" and references OpenAI's fair-use position, but does not resolve it.

4. **Fair use for model training** is an evolving legal area. As of early 2026, US courts have issued mixed rulings. The Copyright Office issued guidance in 2025 suggesting fair use is NOT automatic for commercial AI training.

### Recommendation:
- **For research/prototyping:** Request access, train models, validate approach. Zero risk.
- **For commercial deployment (StemScriber):** Discuss with Alexandra at Morris Music Law before shipping any model trained on DadaGP. The risk is similar to Songsterr bulk-scraping -- we're training on transcriptions of copyrighted songs. Frame it as: "We want to train our own transcription model. What data can we legally use?"
- **Safest commercial path:** Train on GuitarSet (CC BY 4.0) for initial model, then use DadaGP-trained model only after legal guidance.

---

## 7. Audio: Notation Only (Must Synthesize)

DadaGP contains **zero audio files**. To create (audio, tab) training pairs, we must synthesize audio from the notation.

### Option A: DIY synthesis with FluidSynth
- Decode DadaGP tokens back to .gp5 using `dadagp.py decode`
- Export MIDI from GP5 using PyGuitarPro
- Render with FluidSynth + guitar SoundFont
- Cost: free, quality: mediocre (MIDI synth sound)

### Option B: Use SynthTab (pre-built, better quality)
**SynthTab** (ICASSP 2024) already did this work:
- Derived from DadaGP, synthesized using commercial guitar VSTs
- **60,000 tracks**, **13,113 hours of audio**, 23 timbral profiles (acoustic + electric)
- Includes per-string annotations in JAMS format
- **License: CC BY-NC 4.0** (NonCommercial -- cannot use for commercial training)
- **Size: ~2 TB** total
- **Download:** Available via UR Box (Rochester)
- **GitHub:** https://github.com/yongyizang/SynthTab
- **Paper:** https://arxiv.org/abs/2309.09085

**SynthTab is ideal for research/prototyping but its NC license blocks commercial use**, same problem as DadaGP itself.

### Option C: Render our own audio from DadaGP with Neural Amp Modeler
- Use DadaGP notation + our own synthesis pipeline
- Neural Amp Modeler (NAM) for realistic amp tones
- More work but we own the audio output
- The legal question is still about the notation input, not the audio

---

## 8. How to Convert DadaGP to trimplexx CRNN Training Pairs

The trimplexx CRNN expects: **(CQT spectrogram, per-frame string+fret labels)**

Pipeline to get there from DadaGP:

```
DadaGP tokens (.txt)
    │
    ▼  dadagp.py decode
GuitarPro file (.gp5)
    │
    ▼  PyGuitarPro 0.6
Parse tracks → extract per-note: [onset_tick, string, fret, duration_ticks]
    │
    ├──▶ Convert ticks to seconds using tempo map
    │    → Ground truth labels: [(time, string, fret, duration_sec), ...]
    │
    ▼  FluidSynth / NAM / SynthTab
Synthesized audio (.wav)
    │
    ▼  librosa CQT
CQT spectrogram frames
    │
    ▼  Align labels to CQT frames
Per-frame annotation: frame → [(string, fret, active/onset)]
    │
    ▼  
Training pair ready: (CQT_frames, labels)
```

### Key conversion steps:

1. **Decode tokens to GP5:** `python dadagp.py decode song.tokens.txt song.gp5`
   - Requires PyGuitarPro 0.6 specifically (not newer versions)

2. **Parse GP5 for note events:**
   ```python
   import guitarpro
   song = guitarpro.parse('song.gp5')
   for track in song.tracks:
       for measure in track.measures:
           for voice in measure.voices:
               for beat in voice.beats:
                   for note in beat.notes:
                       # note.string, note.fret, beat.start, beat.duration
                       # + note.effect (bend, slide, hammer, etc.)
   ```

3. **Build tempo map:** Extract tempo changes from GP5 to convert tick positions to wall-clock seconds.

4. **Synthesize audio:** FluidSynth with a guitar SoundFont, or export MIDI and render through a VST.

5. **Compute CQT:** Same parameters as trimplexx uses on GuitarSet (hop_length, n_bins, etc.)

6. **Quantize labels to frames:** For each CQT frame, determine which (string, fret) pairs are active.

### Estimated effort:
- Build converter pipeline: 4-6 hours dev
- Render 26K tracks audio: 8-12 hours compute (~$5 Modal CPU)
- Total: ~2 days including testing

---

## 9. Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| Songs | 26,181 |
| Genres | 739 |
| Total tokens | ~116 million |
| Avg song duration | ~2:45 |
| Avg tracks per song | ~4 (2 guitars, bass, drums) |
| Total music duration | ~1,200 hours |
| File formats | GP3/GP4/GP5 + tokenized text |
| Audio included | No |
| Source | Ultimate Guitar (user transcriptions) |
| Genre metadata source | Spotify Web API |
| Encoder/decoder | MIT licensed, requires PyGuitarPro 0.6 |
| Dataset access | By request (email/Twitter) |

---

## 10. Related Datasets Comparison

| Dataset | Size | Audio? | License | Commercial? | Best for |
|---------|------|--------|---------|-------------|----------|
| **DadaGP** | 26K tracks, ~116M tokens | No | By request (research) | Unclear | Notation pretraining |
| **SynthTab** | 60K tracks, 13K hours, ~2TB | Yes (synthesized) | CC BY-NC 4.0 | No | Research pretraining |
| **GuitarSet** | 360 clips, 3 hours | Yes (hexaphonic) | CC BY 4.0 | Yes | Gold-standard fine-tuning |
| **ProgGP** | 173 prog metal songs | No | Research | Unclear | Niche genre supplement |
| **IDMT-SMT-Guitar** | 4,700 isolated notes | Yes | Academic | Check terms | Single-note training |
| **Slakh2100** | 2,100 tracks | Yes (MIDI-synth) | CC BY 4.0 | Yes | Multi-instrument training |

---

## 11. Action Items

1. **Request DadaGP access** -- email Pedro Sarmento / @dadabots on Twitter. Mention research use for guitar transcription model training. Ask explicitly about commercial training rights.
2. **Ask Alexandra** (Morris Music Law) about training on crowd-sourced transcriptions of copyrighted songs -- this applies to both DadaGP and any Songsterr data.
3. **Start with GuitarSet** (CC BY 4.0, commercially safe) for MVP by May 5.
4. **Use DadaGP for Phase A2** fine-tuning (post-MVP) once we have legal clarity.
5. **Do NOT download SynthTab** for commercial training (CC BY-NC 4.0 blocks this). Fine for research experiments only.
6. **Consider DIY synthesis** from DadaGP GP5 files using our own rendering pipeline -- the audio we create is ours, but the underlying notation copyright question remains.
