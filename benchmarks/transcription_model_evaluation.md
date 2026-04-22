# StemScriber Transcription Model Evaluation

## Date: 2026-02-27
## Author: Pipeline Agent

---

## 1. Current Transcription Architecture

StemScriber uses a tiered fallback system per instrument type:

### Guitar
1. **GuitarTabTranscriber** (Basic Pitch + post-processing) -- `guitar_tab_transcriber.py`
   - Basic Pitch inference with guitar-tuned parameters
   - Range filtering (E2=40 to E6=88)
   - Polyphony limiting (max 6 simultaneous)
   - Context-aware string/fret assignment via FretMapper
   - Short note cleanup, velocity dynamics
   - Checkpoint: `best_tab_model.pt` (57MB) -- BUT this is not a custom CRNN; the transcriber uses Basic Pitch directly
2. **MelodyExtractor** (pyin-based monophonic) -- `melody_transcriber.py`
3. **EnhancedTranscriber** (Basic Pitch + articulation detection) -- `transcriber_enhanced.py`
4. **Basic Pitch** raw (fallback)

### Bass
1. **BassTranscriber** (Basic Pitch + post-processing) -- `bass_transcriber.py`
   - Octave correction (fixes harmonic confusion)
   - 4-string assignment
   - Checkpoint: `best_bass_model.pt` (113MB) -- same pattern as guitar, uses Basic Pitch directly
2. MelodyExtractor / EnhancedTranscriber / Basic Pitch fallbacks

### Piano
1. **PianoTranscriber** (Custom CRNN) -- `piano_transcriber.py`
   - Trained on MAESTRO v3.0.0 (198 hours)
   - Mel -> Conv2D -> BiLSTM -> onset/frame/velocity
   - 88-key polyphonic, onset-conditioned frames
   - Checkpoint: `best_piano_model.pt` (138MB)
   - **This is a genuine neural model, not a Basic Pitch wrapper**
2. MelodyExtractor / EnhancedTranscriber / Basic Pitch fallbacks

### Drums
1. **NeuralDrumTranscriber** (Custom CRNN) -- `drum_nn_transcriber.py`
   - Trained on E-GMD (444 hours)
   - 8 drum classes, onset/frame/velocity
   - Checkpoint: `best_drum_model.pt` (109MB)
   - **Genuine neural model**
2. **OaFDrumTranscriber** (Magenta OaF) -- `oaf_drum_transcriber.py`
3. **EnhancedDrumTranscriber** (spectral v2) -- `drum_transcriber_v2.py`
4. Spectral v1 fallback

---

## 2. Key Finding: Guitar and Bass Are NOT Neural Models

Despite having checkpoint files (`best_tab_model.pt`, `best_bass_model.pt`), the guitar and bass transcribers in production actually use **Basic Pitch** as their core inference engine. The "neural" aspect is the post-processing (range filtering, string/fret assignment, octave correction).

Looking at `guitar_tab_transcriber.py` line 283:
```python
model_output, midi_data, note_events = predict(
    str(audio_path),
    onset_threshold=0.45,
    frame_threshold=0.30,
    ...
)
```

And `bass_transcriber.py` line 273:
```python
model_output, midi_data, note_events = predict(
    str(audio_path),
    onset_threshold=0.50,
    frame_threshold=0.35,
    ...
)
```

Both call `basic_pitch.inference.predict()` directly. The `.pt` checkpoint files appear to be from an earlier CRNN training attempt that was abandoned.

The task description mentions "guitar/bass CRNN training failed (F1 only 0.052 after 107 epochs, too few training samples)." This confirms the custom CRNN approach was tried and abandoned in favor of Basic Pitch + post-processing.

---

## 3. Basic Pitch Strengths and Weaknesses

### Strengths
- **Lightweight**: ~15MB model, fast inference
- **Polyphonic**: Handles multiple simultaneous notes
- **Cross-instrument**: Works on guitar, bass, piano, vocals
- **Pitch bend detection**: Detects continuous pitch changes
- **MIT licensed, actively maintained by Spotify**

### Weaknesses for Guitar/Bass
- **No guitar-specific training**: Trained on general audio, not guitar-specific datasets
- **Chord voicing confusion**: Struggles with dense guitar chords (especially distorted)
- **Harmonic confusion for bass**: Often detects harmonics as fundamental (octave errors)
- **No tablature awareness**: Can't distinguish between same pitch on different strings
- **Limited articulation**: Can't detect hammer-ons, pull-offs, slides, bends natively
- **Timing imprecision**: Onset detection is adequate but not precise for fast passages

---

## 4. Alternative Models Evaluated

### 4.1 YourMT3+ (Multi-Task Multitrack)

**What:** Enhanced MT3 with hierarchical attention transformer + MoE
**Source:** https://github.com/mimbres/YourMT3
**Status:** Research code, not pip-installable

**Pros:**
- Multi-instrument aware (reduces cross-instrument confusion)
- Trained on multiple datasets with cross-stem augmentation
- State-of-the-art on 2025 AMT Challenge benchmarks
- Handles piano, guitar, bass, drums, vocals simultaneously

**Cons:**
- Heavy model (requires significant GPU memory)
- Research code quality -- not production-ready
- Requires JAX/Flax (not PyTorch) for original MT3
- YourMT3 is PyTorch but requires significant setup
- No pip install; must clone and configure

**Verdict:** Too heavy for local real-time use. Best suited as a cloud/Colab option for premium quality.

### 4.2 Omnizart

**What:** Multi-task transcription (music, drums, chords, beat tracking)
**Source:** https://github.com/Music-and-Culture-Technology-Lab/omnizart
**Status:** pip installable (`pip install omnizart`)

**Pros:**
- Dedicated drum transcription module
- Chord detection module
- Beat tracking
- Multiple output formats

**Cons:**
- **NOT compatible with ARM Mac (Apple Silicon)** -- TensorFlow 1.x dependency
- Last significant update was 2022
- Intel Mac or Linux only
- Performance on guitar is mediocre (not guitar-specific training)

**Verdict:** Not viable on M3 Max. Could work in a Docker container with x86 emulation, but adds complexity for marginal improvement.

### 4.3 Klangio Guitar2Tabs (Commercial)

**What:** Best-in-class guitar-specific transcription
**Source:** https://klang.io/guitar2tabs/
**Status:** Commercial SaaS, no self-hosted option

**Pros:**
- Handles distorted tones, solos, chords exceptionally well
- Exports TAB, Sheet Music, MIDI, MusicXML, GuitarPro
- Understands guitar fingering positions
- Trained specifically on guitar audio

**Cons:**
- Paid: $14.99/mo or $5.99/mo annual
- No self-hosted option
- API not publicly documented
- Dependency on external service

**Verdict:** Best quality available, but requires commercial API integration. Good candidate for a "premium transcription" tier.

### 4.4 Google MT3 (Multi-Task Multitrack)

**What:** Google Research's transformer-based AMT
**Source:** https://github.com/magenta/mt3
**Status:** Research code, Colab notebook available

**Pros:**
- Handles multiple instruments simultaneously
- Used as benchmark in 2025 AMT Challenge
- Good documentation

**Cons:**
- JAX/Flax dependency (not PyTorch)
- Heavy inference requirements
- Research-quality code
- Not guitar-specific

**Verdict:** Similar to YourMT3+ -- good as a cloud option, too heavy for local use.

### 4.5 MR-MT3 (Memory Retaining MT3)

**What:** MT3 variant that mitigates instrument leakage
**Source:** https://github.com/gudgud96/MR-MT3
**Status:** Research code

**Pros:**
- Specifically addresses cross-instrument bleed in transcription
- Memory mechanism for longer-context understanding

**Cons:**
- Same JAX/Flax requirements as MT3
- Research code, not production-ready

**Verdict:** Interesting for reducing cross-instrument artifacts, but same deployment challenges as MT3.

### 4.6 A-star-Guitar (Tablature Optimization)

**What:** A* pathfinding for optimal guitar tablature
**Source:** Academic paper
**Status:** Research

**Pros:**
- Finds physically optimal fingerings for any MIDI
- Considers string/fret transitions, hand positions

**Cons:**
- Not a transcription model -- requires MIDI input
- Would complement, not replace, pitch detection

**Verdict:** Could improve StemScriber's FretMapper (`_assign_string_fret`) algorithm, which already does a simpler version of this.

---

## 5. Comparison Matrix

| Model | Guitar Quality | Bass Quality | Piano Quality | Drums | Local M3 Max | Pip Install | License |
|-------|---------------|-------------|---------------|-------|-------------|-------------|---------|
| Basic Pitch (current) | Fair | Fair | Fair | Poor | Yes | Yes | Apache 2.0 |
| Piano CRNN (current) | N/A | N/A | Good | N/A | Yes | Built-in | Custom |
| Drum CRNN (current) | N/A | N/A | N/A | Good | Yes | Built-in | Custom |
| YourMT3+ | Good | Good | Very Good | Good | Marginal | No | MIT |
| Omnizart | Fair | Fair | Fair | Good | No (Intel only) | Yes | MIT |
| Klangio | Excellent | Good | N/A | N/A | N/A (SaaS) | No | Commercial |
| MT3 | Good | Good | Very Good | Good | No (too heavy) | No | Apache 2.0 |

---

## 6. Recommendations

### Short Term (Immediate Improvements)

1. **Keep Basic Pitch as the core engine** for guitar and bass.
   - It's lightweight, fast, and "good enough" for many use cases.
   - The post-processing in `guitar_tab_transcriber.py` and `bass_transcriber.py` already adds significant value.

2. **Tune Basic Pitch parameters per genre/style.**
   - Current params are one-size-fits-all. Could benefit from:
     - Clean guitar: lower onset_threshold (0.35), wider freq range
     - Distorted guitar: higher onset_threshold (0.55), narrower freq range
     - Fingerpicked acoustic: lower frame_threshold (0.25)
   - Could auto-detect style from audio features (spectral centroid, RMS energy)

3. **Improve the FretMapper** using A*-inspired pathfinding.
   - Current `_assign_string_fret` uses greedy local optimization
   - A* global optimization would produce more realistic tab positions
   - Especially important for passages spanning multiple positions

### Medium Term (1-2 weeks)

4. **Train a custom guitar CRNN on better data.**
   - The previous attempt failed with 0.052 F1 due to insufficient training data
   - **GuitarSet** (3.2 hours, 360 recordings, hexaphonic pickup) is the gold standard dataset
   - **IDMT-SMT-Guitar** (4.5 hours) provides isolated guitar stems
   - Combined: ~8 hours. The piano model trained successfully on 198 hours (MAESTRO)
   - Need 50-100x more data than the failed attempt
   - **Transfer learning from piano model** could bootstrap -- same CRNN architecture

5. **Add ensemble transcription option.**
   - Run both Basic Pitch and melody extractor (pyin)
   - Merge results: use pyin for timing accuracy, Basic Pitch for pitch accuracy
   - The `melody_transcriber.py` already does this partially (ensemble=True flag)

### Long Term (Future Features)

6. **Klangio API integration** as a premium tier.
   - Best guitar transcription quality available
   - Freemium model (first 30 seconds free)
   - Position as "AI-enhanced" transcription upgrade

7. **YourMT3+ as a cloud processing option.**
   - Deploy on GPU cloud for users who want maximum quality
   - Too heavy for local M3 Max real-time processing
   - Could run async with job queue

---

## 7. Training Data Gap Analysis

The core issue for guitar/bass CRNN training:

| Model | Training Data | Hours | Result |
|-------|--------------|-------|--------|
| Piano CRNN | MAESTRO v3.0.0 | 198 hrs | Working (val_loss converges) |
| Drum CRNN | E-GMD | 444 hrs | Working (val_loss 0.0335) |
| Guitar CRNN | Unknown (likely too small) | ~? hrs | FAILED (F1 0.052) |
| Bass CRNN | Unknown (likely too small) | ~? hrs | FAILED (assumed) |

**Available guitar datasets:**
- GuitarSet: 3.2 hours (hexaphonic, clean guitar)
- IDMT-SMT-Guitar: 4.5 hours (isolated stems)
- MusicNet (guitar portions): ~2 hours
- MUSDB18 guitar stems: ~10 hours (but with separation artifacts)

**Minimum viable dataset:** ~50 hours of paired audio + MIDI
**Current gap:** Need 40+ more hours of guitar training data

**Possible solutions:**
1. Synthesize training data from MIDI + guitar soundfonts (data augmentation)
2. Use StemScriber's own separation to create guitar stems from MUSDB18/full songs
3. Fine-tune Basic Pitch on guitar-specific data (transfer learning)
4. Train a Mel-Band-RoFormer model specifically for guitar transcription (the `train_guitar_model/` directory has the infrastructure for this)

---

## 8. Quick Wins Checklist

- [ ] Remove unused `.pt` checkpoints for guitar/bass if they're not actually loaded
- [ ] Add genre-adaptive Basic Pitch parameters (clean vs distorted vs acoustic)
- [ ] Improve FretMapper with lookahead window (2-3 notes ahead) for better position choices
- [ ] Add confidence thresholding: if Basic Pitch confidence < 0.4, mark notes as uncertain in GP output
- [ ] Document the actual transcription model used for each stem in the job metadata
- [ ] Benchmark current output quality against GuitarSet reference MIDI (compute F1, precision, recall)

---

*Evaluation compiled: 2026-02-27*
