# trimplexx CRNN Integration Audit for StemScriber

**Date:** 2026-04-05
**Status:** Research complete -- no code changes made
**Repo:** https://github.com/trimplexx/music-transcription

---

## Executive Summary

The trimplexx CRNN is a strong candidate for replacing Basic Pitch as StemScriber's guitar transcription engine. Its key advantage is **native string+fret output** (eliminating the FretMapper heuristic entirely), with benchmark scores of 0.8736 MPE F1 and 0.8569 TDR F1 on GuitarSet. The main blocker is that **no pretrained weights are distributed** -- you must train from scratch on GuitarSet (~3-4 hours on A10G GPU). The model is ~81 MB (21.3M parameters), runs comfortably on Modal's A10G, and can run on CPU with acceptable latency for a background processing pipeline (~15-45 seconds per song). Integration requires a new wrapper file and a one-line addition to the guitar fallback chain in `transcription.py`.

---

## 1. Current Guitar Pipeline

### Flow
```
Guitar stem (WAV/MP3)
  --> Basic Pitch (general AMT, ICASSP 2022 model)
  --> GuitarTabTranscriber post-processing:
      - Key detection + spurious note filtering
      - Guitar range filter (E2=40 to E6=88)
      - Polyphony limit (max 6 simultaneous)
      - FretMapper heuristic (chord-shape-aware scoring)
      - Short note cleanup (<30ms)
  --> MIDI with guitar note numbers (tuning[string] + fret)
  --> midi_to_gp.py --> GP5
```

### Fallback chain (transcription.py lines 328-441)
```
Guitar v3 NN --> Basic Pitch + GuitarTabTranscriber --> ChordToTabGenerator --> MelodyExtractor --> EnhancedTranscriber --> raw Basic Pitch
```

### Key limitations of Basic Pitch path
1. **No string/fret awareness** -- outputs MIDI pitches only; FretMapper guesses positions
2. **No articulation data** -- bends, slides, hammer-ons are absent
3. **Voicing ambiguity** -- cannot distinguish fret 5/A-string vs open D (both MIDI 50)

---

## 2. trimplexx CRNN: Technical Details

### Architecture
| Component | Details |
|-----------|---------|
| CNN encoder | 5 layers: 1->32->64->128->128->128 channels, 3x3 kernels, BatchNorm+ReLU+MaxPool |
| RNN | 2-layer bidirectional GRU, hidden=768, dropout=0.5 |
| Onset head | Linear(1536, 6) -- per-string binary onset predictions |
| Fret head | Linear(1536, 132) -- per-string, 22-class fret classification (0-20 + silence) |
| Total params | ~21.3M |
| Model size | ~81 MB (float32) |

### Input format
| Parameter | Value |
|-----------|-------|
| Audio sample rate | 22,050 Hz |
| Transform | CQT (Constant-Q Transform) |
| CQT bins | 168 (7 octaves x 24 bins/octave) |
| Hop length | 512 samples (~23ms resolution) |
| F_min | E2 (82.4 Hz) |
| CQT library | nnAudio (GPU-accelerated) |

### Output format
- **Frame-level predictions:**
  - `onset_logits`: shape [batch, T_reduced, 6] -- per-string onset probability
  - `fret_logits`: shape [batch, T_reduced, 6, 22] -- per-string fret class logits
- **Post-processing** (`frames_to_notes_for_eval`): converts to note list with fields:
  - `start_time`, `end_time` (seconds)
  - `pitch_midi` (integer MIDI note)
  - `string` (0-5, low E to high E)
  - `fret` (0-20)

This output format is **directly compatible** with StemScriber's `midi_to_gp.py` pipeline -- it provides the exact string+fret data that FretMapper currently has to guess.

### Dependencies (requirements.txt)
```
librosa==0.11.0
mirdata==0.3.9
matplotlib==3.9.4
numpy==1.26.3
scikit-learn==1.6.1
tqdm==4.67.1
pretty_midi>=0.2.10
jams~=0.3.4
nnAudio~=0.3.3
torch~=2.5.1+cu121
noisereduce~=3.0.3
torchaudio~=2.5.1+cu121
```

**Conflicts with StemScriber's existing stack:** Minimal. StemScriber already uses torch, librosa, numpy, pretty_midi, torchaudio. The new additions are `nnAudio`, `mirdata`, `jams`, and `noisereduce` (only mirdata and jams are needed for training, not inference).

**Inference-only dependencies:** torch, librosa (or nnAudio for CQT), numpy, pretty_midi.

### License
MIT (stated in README badge, though no LICENSE file exists in the repo). Low legal risk.

### Benchmark scores (GuitarSet)
| Metric | trimplexx CRNN | Basic Pitch (estimated) | FretNet |
|--------|---------------|------------------------|---------|
| Multi-Pitch F1 (MPE) | **0.8736** | ~0.79 | 0.818 |
| Tab Detection Rate (TDR) | **0.8569** | N/A (no tab) | 0.727 |
| Onset F1 | ~0.85 | ~0.70 | -- |

### Pretrained weights
**NOT INCLUDED in the repository.** The `.gitignore` excludes the `python/data/` directory where trained checkpoints would live. You must train from scratch. The author's best run is "run_72" with TDR F1 = 0.8569.

---

## 3. CPU vs GPU Viability

### Model characteristics for inference estimation
- 21.3M parameters (medium-small by modern standards)
- Bidirectional GRU requires full sequence in memory
- CQT computation is the expensive part on CPU (nnAudio uses GPU; librosa fallback on CPU)
- A 3-minute song at 22,050 Hz / hop 512 = ~7,759 frames; after CNN pooling (~16x reduction on freq axis) the RNN processes ~7,759 time steps

### Estimated inference times

| Platform | CQT Time | Model Inference | Total (3-min song) |
|----------|----------|----------------|-------------------|
| **Modal A10G (GPU)** | <1s (nnAudio) | 2-5s | **3-8 seconds** |
| **Hetzner VPS (4 CPU, 8GB RAM)** | 3-8s (librosa) | 10-30s | **15-45 seconds** |
| **M3 Max (MPS)** | <1s (nnAudio/MPS) | 3-8s | **4-10 seconds** |

### Memory requirements
- Model weights: ~81 MB
- CQT features for 3-min song: ~168 x 7,759 x 4 bytes = ~5 MB
- GRU hidden states: ~2 x 768 x 7,759 x 4 = ~47 MB
- **Peak RAM estimate: ~200 MB** -- well within Hetzner's 8 GB

### Recommendation
**Run on Modal A10G** alongside stem separation for best latency (3-8s). The Hetzner VPS can serve as a CPU fallback (15-45s) if Modal is unavailable. The model is small enough that CPU inference is viable for a background processing pipeline where users expect 1-3 minutes total processing time anyway.

---

## 4. Integration Plan

### What gets replaced
| Current Component | Action |
|-------------------|--------|
| `guitar_tab_transcriber.py` (GuitarTabTranscriber) | **Demoted to fallback** -- kept as backup |
| FretMapper heuristic in `midi_to_gp.py` | **Bypassed for trimplexx output** -- string+fret come from model |
| Basic Pitch guitar path | **Demoted to lower priority** in fallback chain |

### What stays the same
| Component | Why |
|-----------|-----|
| Stem separation (Modal/local) | Upstream -- no change |
| `midi_to_gp.py` GP5 conversion | Downstream -- still consumes MIDI with string/fret data |
| `quantize_midi()` | Still useful for rhythmic cleanup |
| Drum/bass/piano transcription | Different stems -- untouched |
| `transcription.py` dispatcher | Just adds a new entry to the guitar fallback chain |

### New file: `backend/trimplexx_transcriber.py` (to be created)

Responsibilities:
1. Load trained CRNN checkpoint + config
2. Accept audio path, produce CQT features (librosa or nnAudio)
3. Run inference -> onset_logits + fret_logits
4. Post-process with `frames_to_notes_for_eval()` to get note list with string+fret
5. Convert note list to MIDI (using pretty_midi, embedding string/fret as metadata)
6. Return a result object compatible with the existing pipeline (midi_path, quality_score, etc.)

### Fallback chain change (transcription.py)
```
BEFORE:
  Guitar v3 NN -> Basic Pitch + GuitarTabTranscriber -> ...

AFTER:
  trimplexx CRNN -> Guitar v3 NN -> Basic Pitch + GuitarTabTranscriber -> ...
```

This is a ~15-line addition to `transcription.py` (lines 328-365 pattern), following the exact same try/except/quality-check/continue pattern used by every other transcriber.

### Input/output format compatibility

**Input:** The trimplexx model expects mono audio at 22,050 Hz. StemScriber's stem files are typically 44,100 Hz stereo MP3. Conversion needed:
```python
import librosa
audio, sr = librosa.load(stem_path, sr=22050, mono=True)
```

**Output:** The `frames_to_notes_for_eval()` function returns a list of dicts:
```python
{'start_time': float, 'end_time': float, 'pitch_midi': int, 'string': int, 'fret': int}
```

This maps directly to pretty_midi notes. The string+fret data can be passed through to `midi_to_gp.py` either via:
- MIDI note metadata (program change per string, or custom CC messages)
- A sidecar JSON file alongside the MIDI
- Direct integration where `midi_to_gp.py` accepts a note list instead of a MIDI file

The cleanest approach is to write a MIDI file with 6 tracks (one per string), where each track's notes carry the fret assignment. This is already how `midi_to_gp.py` expects multi-string input.

---

## 5. Training Requirements

### Must train from scratch
No pretrained weights are distributed. Training requires:

1. **Dataset:** GuitarSet (360 30-second clips, ~180 minutes total)
   - Downloaded via `mirdata`: `mirdata.initialize('guitarset').download()` (~2 GB)
   - Hexaphonic recordings with JAMS annotations (precise per-string ground truth)

2. **Training compute:**
   - **Modal A10G:** ~3-4 hours for 300 epochs with early stopping
   - **M3 Max local:** ~6-8 hours (MPS acceleration)
   - **Hetzner VPS (CPU):** Not recommended (~24+ hours)

3. **Training procedure:**
   - The repo provides a Jupyter notebook (`guitar_ATM.ipynb`) and programmatic pipeline (`training/pipeline.py`)
   - Best config: GRU, hidden=768, 2 layers, bidirectional, lr=0.0003, onset_loss_weight=9.0
   - Data augmentation: time stretch, noise, reverb, EQ, clipping, SpecAugment
   - Early stopping patience: 25 epochs, checkpoint on val_tdr_f1

4. **Output:** `best_model.pt` checkpoint (~81 MB) + `run_config.json`

### Training cost estimate
- Modal A10G at ~$1.10/hr x 4 hours = **~$4.40** for one training run
- May want 2-3 runs to verify reproducibility = ~$13

---

## 6. Risk Assessment

### HIGH risk
- **GuitarSet generalization gap:** GuitarSet contains only clean acoustic guitar recordings (360 clips from 6 players). It includes rock, jazz, funk, bossa nova, and singer-songwriter but **no distorted electric guitar, metal, or heavily processed tones**. StemScriber users will upload all genres. The model may perform poorly on distorted/effects-heavy guitar.
  - **Mitigation:** Keep Basic Pitch as fallback; implement quality-score threshold to auto-fallback when trimplexx confidence is low.

### MEDIUM risk
- **No pretrained weights:** Must train before any testing. Cannot evaluate real-world performance until training completes.
  - **Mitigation:** Training is cheap ($4-5 on Modal) and reproducible (hyperparams documented).

- **Small community (6 stars):** Minimal external validation or bug reports. Code is a university thesis project.
  - **Mitigation:** The codebase is clean, well-structured (~1,200 lines of Python), and the architecture is standard (CNN+GRU). Low risk of hidden bugs.

### LOW risk
- **Dependency conflicts:** New deps (nnAudio, mirdata, jams) are lightweight and unlikely to conflict with existing stack. mirdata and jams are training-only.
- **License:** MIT (per README badge). Safe for commercial use.
- **Model size:** 81 MB is trivial for Modal deployment (cached in Volume). Acceptable for Hetzner VPS.

---

## 7. Recommended Next Steps

### Phase 1: Train and evaluate (1-2 days)
1. Clone repo to Modal or local dev environment
2. Download GuitarSet via mirdata
3. Train best config (run_72 hyperparams) on Modal A10G (~4 hours, ~$5)
4. Evaluate on held-out test set -- verify TDR F1 > 0.85

### Phase 2: Build wrapper and integrate (1-2 days)
1. Create `backend/trimplexx_transcriber.py`:
   - `TrimplexxTranscriber` class with `.transcribe(audio_path, output_dir, ...)`
   - CQT feature extraction (librosa for CPU, nnAudio for GPU)
   - Model inference + `frames_to_notes_for_eval()` post-processing
   - MIDI output with per-string tracks
2. Wire into `transcription.py` as highest-priority guitar transcriber
3. Add quality score calculation based on onset confidence distribution

### Phase 3: A/B test (1 day)
1. Process "The Time Comes" demo with both pipelines
2. Compare GP5 output in AlphaTab:
   - Are fret positions more realistic?
   - Are chord voicings natural?
   - Is note accuracy improved?
3. Test with 2-3 additional songs spanning genres

### Phase 4: Production deploy (0.5 day)
1. Bundle trained checkpoint in Modal image (or cache in Modal Volume)
2. Deploy alongside BS-RoFormer separation upgrade
3. Monitor quality scores and fallback rates

---

## 8. Key Files Reference

### StemScriber (existing)
- `/Users/jeffkozelski/stemscribe/backend/processing/transcription.py` -- dispatcher, fallback chains
- `/Users/jeffkozelski/stemscribe/backend/guitar_tab_transcriber.py` -- Basic Pitch + FretMapper
- `/Users/jeffkozelski/stemscribe/backend/midi_to_gp.py` -- MIDI to GP5 converter
- `/Users/jeffkozelski/stemscribe/docs/transcription-upgrade-plan.md` -- prior research

### trimplexx (cloned to /tmp/trimplexx-eval2)
- `python/model/architecture.py` -- GuitarTabCRNN model definition
- `python/config.py` -- all hyperparameters and constants
- `python/training/pipeline.py` -- training loop
- `python/training/note_conversion_utils.py` -- frame-to-note post-processing (key for inference)
- `python/model/utils.py` -- checkpoint loading utility
- `python/evaluation/tablature_export.py` -- ASCII tab + MIDI export
- `python/requirements.txt` -- dependencies
