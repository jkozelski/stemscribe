# StemScriber Transcription Pipeline Upgrade Plan

**Date:** 2026-04-04
**Author:** Claude (research agent)
**Status:** Research complete — ready for implementation

---

## 1. Current Pipeline Audit

### Architecture
```
Audio File
  ↓
Stem Separation (htdemucs_6s on Modal A10G, or BS-RoFormer+Demucs locally)
  ↓
Per-Stem Transcription:
  - Guitar: Guitar v3 NN → Basic Pitch + GuitarTabTranscriber → MelodyExtractor → EnhancedTranscriber
  - Bass:   Bass v3 NN → Basic Pitch BassTranscriber → MelodyExtractor → EnhancedTranscriber
  - Drums:  Neural CRNN → OaF → v2 Spectral → v1
  - Piano:  Neural CRNN → MelodyExtractor → EnhancedTranscriber
  ↓
MIDI Quantization (16th note grid, min 50ms notes)
  ↓
MIDI → GP5 Conversion (midi_to_gp.py — FretMapper, articulations, chord voicing)
```

### Key Files
| File | Lines | Role |
|------|-------|------|
| `modal_separator.py` | 201 | Modal GPU deployment — htdemucs_6s via CLI subprocess |
| `processing/separation.py` | 763 | Local separation — 4 modes (demucs, roformer, mdx, ensemble) |
| `enhanced_separator.py` | ~200 | Local audio-separator wrapper (BS-RoFormer, MDX23C) |
| `processing/transcription.py` | ~650 | Transcription dispatcher — fallback chains per stem type |
| `guitar_tab_transcriber.py` | ~400 | Basic Pitch + guitar post-processing (key filter, FretMapper) |
| `midi_to_gp.py` | 697 | MIDI → GP5 converter (FretMapper, articulations, chord voicing) |

### Quality Bottlenecks Identified

#### Bottleneck 1: Separation Artifacts (Modal = htdemucs_6s)
- **Problem:** Modal deployment uses htdemucs_6s (SDR ~6-7 dB for guitar), while local already has BS-RoFormer+Demucs ensemble. Guitar bleed from vocals/drums corrupts downstream transcription.
- **Impact:** HIGH — separation quality is the #1 factor. Bad stems = bad tabs regardless of transcription quality.
- **Location:** `modal_separator.py:80-91` — demucs CLI subprocess call

#### Bottleneck 2: Basic Pitch Has No String/Fret Awareness
- **Problem:** Basic Pitch outputs MIDI pitches only. The `GuitarTabTranscriber` applies heuristic post-processing: key filtering, polyphony limiting, and FretMapper scoring. But it **cannot distinguish** between equivalent voicings (e.g., fret 5 on A string vs open D — both MIDI 50).
- **Impact:** MEDIUM-HIGH — tabs are "playable" but often use wrong positions (e.g., putting a blues lick on the wrong string, or assigning chords to awkward voicings).
- **Location:** `guitar_tab_transcriber.py:6-18`, `midi_to_gp.py:48-128` (FretMapper scoring heuristic)

#### Bottleneck 3: No Articulation Detection from Audio
- **Problem:** Bends, slides, hammer-ons are only detected if the MelodyExtractor produces CC#20 markers + pitch bend events. Basic Pitch path has **zero** articulation data. The GP5 converter supports bends/slides (`midi_to_gp.py:551-586`) but rarely gets the data.
- **Impact:** MEDIUM — tabs lack expression, feel "robotic"

#### Bottleneck 4: Duration Quantization Is Lossy
- **Problem:** `midi_to_gp.py:528-537` uses a simple if/elif chain to snap durations to whole/half/quarter/eighth/sixteenth. No dotted notes, no triplets, no ties. A note lasting 1.5 beats becomes a quarter note (1 beat).
- **Impact:** LOW-MEDIUM — rhythm feels "off" on complex passages

#### Bottleneck 5: Chord Voicing Is Approximate
- **Problem:** `FretMapper.map_chord()` in `midi_to_gp.py:236-308` uses brute-force search with a 5-fret span limit. Doesn't know about common chord shapes (open chords, barre chords). Can produce physically awkward voicings.
- **Impact:** LOW-MEDIUM — chords are technically correct but not how a guitarist would play them

---

## 2. Upgrade 1: BS-RoFormer-SW on Modal (Separation)

### Why
BS-RoFormer-SW produces dramatically better stems than htdemucs_6s:

| Stem | BS-RoFormer-SW | htdemucs_6s | Delta |
|------|---------------|-------------|-------|
| Vocals | 11.30 dB | ~9.7 dB | +1.6 dB |
| Bass | 14.62 dB | ~10.0 dB | +4.6 dB |
| Drums | 14.11 dB | ~8.5 dB | +5.6 dB |
| **Guitar** | **9.05 dB** | **~6-7 dB** | **+2-3 dB** |
| Piano | 7.83 dB | Poor | Massive improvement |

A +2-3 dB improvement on guitar separation means **significantly fewer** vocal/drum artifacts leaking into the guitar transcription.

### Model Details
- **Checkpoint:** `BS-Rofo-SW-Fixed.ckpt` (699 MB) from HuggingFace `jarredou/BS-ROFO-SW-Fixed`
- **Config:** `BS-Rofo-SW-Fixed.yaml`
- **Architecture:** BS-RoFormer, dim=256, depth=12, 8 heads, flash_attn
- **Output stems:** bass, drums, other, vocals, guitar, piano (same 6 as htdemucs_6s)
- **Sample rate:** 44100 Hz
- **Estimated VRAM:** 3-5 GB peak → A10G (24 GB) has massive headroom

### Feasibility: CONFIRMED
- `audio-separator` package (v0.44.1) fully supports this model
- You already use `audio-separator` locally in `enhanced_separator.py`
- Same 6 stems = zero downstream changes needed
- Model auto-downloads on first use; cache in Modal Volume

### Code Changes Required

#### `modal_separator.py` — Image
```python
# BEFORE
stemscribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "sox")
    .pip_install(
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "demucs==4.0.1",
        "numpy<2",
        "scipy",
        "soundfile",
        "pydub",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
)

# AFTER
stemscribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "git", "sox")
    .pip_install(
        "audio-separator[gpu]",
        "numpy<2",
        "scipy",
        "soundfile",
        "pydub",
    )
)
```

#### `modal_separator.py` — Function Body
```python
@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/model-cache": model_volume},
    memory=16384,
)
@modal.concurrent(max_inputs=3)  # Reduced from 5 — BS-RoFormer uses more VRAM
def separate_stems_gpu(audio_bytes: bytes, filename: str = "input.mp3") -> dict:
    import tempfile
    from pathlib import Path
    from audio_separator.separator import Separator

    work_dir = Path(tempfile.mkdtemp())
    input_path = work_dir / filename
    input_path.write_bytes(audio_bytes)

    output_dir = work_dir / "separated"
    output_dir.mkdir()

    separator = Separator(
        model_file_dir='/model-cache/audio-separator-models',
        output_dir=str(output_dir),
        output_format='MP3',
        output_bitrate='320k',
    )
    separator.load_model(model_filename='BS-Rofo-SW-Fixed.ckpt')
    output_files = separator.separate(str(input_path))

    # Parse output files — audio-separator embeds stem names in parens
    stems_result = {}
    stem_name_map = {
        'vocals': 'vocals', 'bass': 'bass', 'drums': 'drums',
        'guitar': 'guitar', 'piano': 'piano', 'other': 'other',
    }
    for f in output_files:
        f = Path(f) if Path(f).is_absolute() else output_dir / f
        fname_lower = f.stem.lower()
        for key, stem_name in stem_name_map.items():
            if key in fname_lower:
                stems_result[stem_name] = f.read_bytes()
                break

    model_volume.commit()
    return stems_result
```

### Gotchas
1. **Cold start:** First invocation downloads ~700 MB. Modal Volume caches it after.
2. **flash_attn:** Model config uses `flash_attn: true`. audio-separator handles fallback to standard attention if flash-attn isn't installed, but will be slower.
3. **Concurrency:** Reduce `max_inputs` from 5 to 3 initially. Monitor VRAM via Modal dashboard.
4. **Processing time:** Expect 30-60s per song on A10G (comparable to demucs).

---

## 3. Upgrade 2: trimplexx CRNN for Guitar Transcription

### Why
Basic Pitch is a general-purpose pitch detector. It outputs MIDI notes with **no string/fret information**. StemScriber's `FretMapper` heuristic guesses string positions, but it's fundamentally limited — it can't know that the guitarist played fret 7 on the G string vs fret 3 on the B string.

The trimplexx CRNN (`github.com/trimplexx/music-transcription`) is trained specifically for **guitar tablature** and outputs **string + fret positions directly**.

### Benchmark Scores (GuitarSet)
| Model | Multi-Pitch F1 | Tablature Detection Rate |
|-------|---------------|------------------------|
| Basic Pitch (general) | ~0.79 (estimated) | N/A (no tab output) |
| **trimplexx CRNN** | **0.8736** | **0.8569** |
| FretNet | 0.818 | 0.727 |

### Architecture
- 5-layer CNN (32→64→128→128→128) + 2-layer bidirectional GRU (hidden 768)
- Two output heads: `onset_fc` (per-string onsets, [T, 6]) and `fret_fc` (per-string fret classification, [T, 6, 22])
- Input: CQT spectrogram (168 bins, 24 bins/octave, fmin=E2, hop=512, sr=22050)
- Estimated model size: 15-25 MB

### Integration Plan

#### Step 1: Train the Model (weights not in repo)
```bash
# Clone repo
cd ~/stemscribe/backend
git clone https://github.com/trimplexx/music-transcription trimplexx_transcription

# Install dependencies
pip install mirdata nnAudio noisereduce

# Download GuitarSet via mirdata
python -c "import mirdata; ds = mirdata.initialize('guitarset'); ds.download()"

# Train (their pipeline handles feature extraction + training)
cd trimplexx_transcription/python
python train.py --config config.py  # ~3-4 hours on GPU
# Best checkpoint saved to runs/run_XX/best_model.pt
```

#### Step 2: Create `trimplexx_transcriber.py` Wrapper
```python
# New file: backend/trimplexx_transcriber.py
# - Load CQT features (168 bins, 24 bins/octave, fmin=E2, hop=512, sr=22050)
# - Run model inference → onset + fret predictions per string
# - Decode frame predictions to note events with string/fret/onset/offset
# - Write MIDI with string assignment metadata
# - Output: TabTranscriptionResult compatible with existing pipeline
```

#### Step 3: Wire into Transcription Fallback Chain
In `processing/transcription.py`, add as highest-priority guitar transcriber:
```
Guitar: trimplexx CRNN → Guitar v3 NN → Basic Pitch + GuitarTabTranscriber → ...
```

### Key Advantage
Eliminates the entire `FretMapper` heuristic layer. String+fret assignments come directly from the model, trained on real guitar recordings with ground-truth tablature.

### Risks
- **No pretrained weights distributed.** Must train on GuitarSet (~3-4 hours on GPU, or request from author).
- **GuitarSet is small** (360 30-second clips). Model may not generalize well to distorted electric guitar, metal, etc.
- **6-star repo** — less community vetting than established tools.

---

## 4. Upgrade 3: FretNet for Continuous Pitch (Bends/Vibrato)

### Why
FretNet (`github.com/cwitkowitz/guitar-transcription-continuous`) is an ICASSP 2023 paper that outputs **continuous-valued pitch contours per string+fret pair**. This means it can detect:
- **Bends** (pitch deviation above nominal fret pitch)
- **Vibrato** (periodic pitch oscillation)
- **Slides** (continuous pitch change between frets)

This data maps directly to GP5 bend/vibrato effects in `midi_to_gp.py:551-586`.

### Benchmark Scores (GuitarSet, 6-fold CV)
| Metric | FretNet | TabCNN (baseline) |
|--------|---------|-------------------|
| Tablature F1 | 0.727 | 0.717 |
| Multi-pitch F1 | 0.818 | 0.820 |
| String-dependent Note F1 | **0.506** | 0.430 |
| String-agnostic Note F1 | **0.664** | 0.583 |

FretNet's discrete tablature accuracy is lower than trimplexx (0.727 vs 0.857), but it provides **articulation data that no other model offers**.

### Architecture
- 3 conv blocks (16/32/48 filters) + 3 prediction heads (tablature, pitch deviation, onsets)
- Input: Harmonic CQT (4 octaves, 3 bins/semitone, 6 harmonic channels)
- Output: Frame-level activations + pitch deviations per string/fret
- Estimated model size: 2-5 MB

### Integration Strategy: Complementary to trimplexx

**Recommended approach:** Use trimplexx for note detection + string/fret assignment, then run FretNet on the same audio to extract pitch deviation data and annotate the trimplexx output with bend/vibrato markers.

```
Guitar Stem Audio
  ↓
trimplexx CRNN → note events with string + fret
  ↓
FretNet → continuous pitch contours per string
  ↓
Merge: annotate trimplexx notes with FretNet pitch deviations
  ↓
MIDI with bend/vibrato CC data → midi_to_gp.py → GP5 with articulations
```

### Integration Steps
1. `pip install amt-tools>=0.3.1 muda>=0.4.1`
2. Install `guitar-transcription-with-inhibition` from GitHub
3. Download/train FretNet checkpoint on GuitarSet
4. Write `fretnet_articulation.py` — runs FretNet, extracts pitch deviation per note, returns bend/vibrato annotations
5. Merge annotations into trimplexx output before GP5 conversion

### Risks
- **Heavy dependency chain:** `amt-tools` + `guitar-transcription-with-inhibition` — potential version conflicts
- **README says "TODO" for usage section** — incomplete documentation
- **Lower complexity priority** — trimplexx alone is a bigger win; FretNet adds articulation polish

---

## 5. Additional Improvements (Quick Wins)

### 5a. Fix Duration Quantization in midi_to_gp.py
**Current** (`midi_to_gp.py:528-537`): Simple if/elif snapping to power-of-2 durations.
**Fix:** Add dotted notes and triplets:
```python
# Add after the existing duration logic:
# Check for dotted notes (1.5x base duration)
if note_data['duration_beats'] >= 6:
    beat.duration.value = 1; beat.duration.isDotted = True  # Dotted whole
elif note_data['duration_beats'] >= 3:
    beat.duration.value = 2; beat.duration.isDotted = True  # Dotted half
elif note_data['duration_beats'] >= 1.5:
    beat.duration.value = 4; beat.duration.isDotted = True  # Dotted quarter
elif note_data['duration_beats'] >= 0.75:
    beat.duration.value = 8; beat.duration.isDotted = True  # Dotted eighth
# ... then fall through to existing logic for non-dotted
```

### 5b. Improve Chord Voicing in FretMapper
Add common open chord shape awareness to `midi_to_gp.py:236-308`:
- Detect when input pitches match a known chord (Am, C, G, D, E, Em, etc.)
- Prefer the standard open voicing when fret span allows
- This makes rhythm guitar parts much more realistic

### 5c. Add Tied Notes for Sustained Pitches
When a note spans a barline, create a tie to the next measure instead of truncating. Requires tracking notes across measure boundaries in the GP5 converter.

---

## 6. Implementation Priority & Roadmap

### Phase 1: BS-RoFormer-SW on Modal (1-2 days)
**Impact: HIGH | Effort: LOW**
- Swap modal_separator.py image + function body
- Test with "The Time Comes" demo song
- Compare guitar stem quality (A/B listen test)
- Deploy: `modal deploy modal_separator.py`

### Phase 2: trimplexx CRNN Guitar Transcriber (3-5 days)
**Impact: HIGH | Effort: MEDIUM**
- Clone repo, train on GuitarSet (~4 hours)
- Write `trimplexx_transcriber.py` wrapper
- Wire into transcription.py fallback chain
- Run against test songs, compare tab accuracy vs Basic Pitch
- Optionally train on Modal A10G for speed

### Phase 3: Duration Quantization Fix (0.5 day)
**Impact: MEDIUM | Effort: LOW**
- Add dotted note support to midi_to_gp.py
- Update existing tests (28 passing)

### Phase 4: FretNet Articulation Layer (3-5 days)
**Impact: MEDIUM | Effort: HIGH**
- Install dependency chain (amt-tools, etc.)
- Train/download FretNet checkpoint
- Write articulation extraction + merge logic
- Wire into pipeline between transcription and GP5 conversion

### Phase 5: Chord Voicing Improvements (1 day)
**Impact: LOW-MEDIUM | Effort: LOW**
- Add open chord shape library to FretMapper
- Test with acoustic guitar songs

---

## 7. Model Downloads & Dependencies

### BS-RoFormer-SW
```bash
# Auto-downloaded by audio-separator on first use
# Or manual:
pip install huggingface_hub
huggingface-cli download jarredou/BS-ROFO-SW-Fixed BS-Rofo-SW-Fixed.ckpt
huggingface-cli download jarredou/BS-ROFO-SW-Fixed BS-Rofo-SW-Fixed.yaml
```

### trimplexx CRNN
```bash
pip install mirdata nnAudio noisereduce
git clone https://github.com/trimplexx/music-transcription ~/stemscribe/backend/trimplexx_transcription
# Then train — see Section 3, Step 1
```

### FretNet
```bash
pip install amt-tools>=0.3.1 muda>=0.4.1
pip install git+https://github.com/cwitkowitz/guitar-transcription-with-inhibition
git clone https://github.com/cwitkowitz/guitar-transcription-continuous ~/stemscribe/backend/fretnet
```

---

## 8. Testing Plan

### A/B Test Framework
For each upgrade, compare against the current pipeline using the demo song ("The Time Comes" by Kozelski):

1. **Separation quality:** Listen to isolated guitar stem — fewer artifacts = better
2. **Note accuracy:** Compare transcribed MIDI against manual transcription (count correct/missed/extra notes)
3. **Tab accuracy:** Load GP5 in AlphaTab — are fret positions where a guitarist would actually play?
4. **Articulation accuracy:** Do bends/slides appear where expected?
5. **Rhythm accuracy:** Do dotted notes and triplets render correctly?

### Automated Metrics
- MIDI note count per stem (sanity check)
- Pitch range distribution (should cluster in guitar range)
- Polyphony histogram (shouldn't exceed 6 for guitar)
- GP5 file validity (loads without error in AlphaTab)

---

## Summary

| Upgrade | Impact | Effort | Priority |
|---------|--------|--------|----------|
| BS-RoFormer-SW on Modal | HIGH | LOW | **Phase 1** |
| trimplexx CRNN guitar | HIGH | MEDIUM | **Phase 2** |
| Duration quantization fix | MEDIUM | LOW | **Phase 3** |
| FretNet articulations | MEDIUM | HIGH | Phase 4 |
| Chord voicing improvements | LOW-MEDIUM | LOW | Phase 5 |

**Biggest bang for buck:** Phase 1 (separation) + Phase 2 (transcription) together eliminate the two largest quality bottlenecks. The BS-RoFormer swap is a ~30-line code change with massive quality impact.
