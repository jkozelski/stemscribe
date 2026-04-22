# StemScribe Chord Detection Pipeline Audit

**Date:** 2026-03-16
**Auditor:** Agent 2 (Code Auditor)
**Status:** READ-ONLY analysis, no code modified

---

## Architecture Overview

```
Audio Upload
    |
    v
process_audio() [processing/pipeline.py:144]
    |
    |-- Step 1: Stem Separation (RoFormer/Demucs)
    |-- Step 2: Chord Detection  <-- THIS AUDIT
    |-- Step 3: MIDI Transcription
    |-- Step 4: MusicXML / Guitar Pro
    |
    v
detect_chords_for_job() [processing/transcription.py:656]
    |
    |-- Mix harmonic stems (guitar + piano + bass)
    |   or fall back to original audio
    |
    v
ChordDetector.detect(audio_path, artist, title)
    [chord_detector_v10.py:342]
    |
    |-- _load_btc()  (lazy-load BTC transformer model)
    |-- _lookup_vocab(artist, title)  (optional UG scrape for vocab constraint)
    |-- _detect_btc(audio_path, vocab)
    |       |
    |       |-- librosa.load(sr=22050)
    |       |-- Tuning compensation (pitch_shift if offset > 0.05)
    |       |-- librosa.cqt() in 10s chunks  (n_bins=144, bins_per_octave=24, hop=2048)
    |       |-- log(abs(cqt) + 1e-6)
    |       |-- Normalize: (feature - mean) / std
    |       |-- Pad to timestep=108 boundary
    |       |-- For each 108-frame chunk:
    |       |     encoder_output = self_attn_layers(chunk)
    |       |     lstm_out = output_layer.lstm(encoder_output)
    |       |     logits = output_projection(lstm_out)           <-- 170 classes
    |       |     probs = softmax(logits)
    |       |     prediction = argmax(logits)
    |       |-- Consolidate consecutive same-chord frames into ChordEvents
    |       |-- Key detection (audio chromagram, independent)
    |       |-- Tuning drift correction (diatonic fit heuristic)
    |       |
    |       v
    |   ChordProgression(chords=[ChordEvent,...], key="Am")
    |
    v (fallback if BTC unavailable)
_detect_essentia()  [chord_detector_v10.py:730]
    |-- Essentia ChordsDetection with HPCP + majority vote smoothing
    |
    v
job.chord_progression = [
    {"time": float, "duration": float, "chord": str,
     "root": str, "quality": str, "confidence": float,
     "detector_version": "v10"}
]
```

---

## Key Files and Line Numbers

| File | Purpose |
|------|---------|
| `backend/chord_detector_v10.py` | **Active chord detector** (BTC transformer, 170 classes) |
| `backend/chord_detector.py` | Legacy basic detector (chroma template matching, 7 chord types) |
| `backend/chord_detector_v7.py` | Legacy v7 (25 classes) |
| `backend/chord_detector_v8.py` | Legacy v8 (337 classes, inversions) |
| `backend/chord_analysis.py` | MIDI-based chord analysis (music21 chordify) -- NOT used for audio detection |
| `backend/dependencies.py:179-206` | Import chain: v10 > v8 > v7 > basic |
| `backend/processing/transcription.py:656-748` | `detect_chords_for_job()` -- pipeline integration |
| `backend/processing/pipeline.py:281-287` | Where chord detection is called in main pipeline |
| `backend/routes/theory.py:15-32` | API: `GET /api/chords/<job_id>` -- returns chord data |
| `backend/routes/chord_sheet.py:375-441` | API: `GET /api/chord-sheet/job/<job_id>` -- chord sheet |
| `btc_chord/run_config.yaml` | BTC model configuration |
| `btc_chord/btc_model.py` | BTC model architecture (transformer + LSTM output) |
| `btc_chord/utils/transformer_modules.py:54-89` | OutputLayer with LSTM + projection |
| `btc_chord/utils/mir_eval_modules.py:13-27` | `idx2voca_chord()` -- 170-class vocabulary |
| `btc_chord/test/btc_model_large_voca.pt` | Original BTC checkpoint (12MB) |
| `backend/training_data/btc_finetune/checkpoints/btc_finetuned_best.pt` | Fine-tuned checkpoint (12MB) |

---

## Model Details

### BTC v10 Transformer

- **Architecture:** Bi-directional self-attention (8 layers, 4 heads, hidden=128) + LSTM output layer
- **Input:** CQT features, 144 bins, 24 bins/octave, hop=2048, sr=22050
- **Chunk size:** 108 timesteps per forward pass (representing 10 seconds of audio)
- **Time resolution:** `time_unit = inst_len / timestep = 10.0 / 108 = 0.0926 seconds` (~93ms per frame)
- **Output:** 170 chord classes (12 roots x 14 qualities + N + X)
- **Qualities:** min, maj, dim, aug, min6, maj6, min7, minmaj7, maj7, 7, dim7, hdim7, sus2, sus4
- **Checkpoint:** 12MB (fine-tuned version at `btc_finetuned_best.pt`)

### Config Override (chord_detector_v10.py:291-292)
```python
config.feature['large_voca'] = True
config.model['num_chords'] = 170
```
The YAML file has `num_chords: 25` and `large_voca: False` but the code overrides both at runtime.

---

## Threshold Values and Filtering

| Parameter | Value | Location |
|-----------|-------|----------|
| `min_duration` (default) | **0.3 seconds** | `chord_detector_v10.py:263` |
| `min_duration` (vocab-constrained) | **0.15 seconds** | `chord_detector_v10.py:412` |
| Confidence threshold | **NONE** | No confidence filtering on BTC output |
| Diatonic snap threshold | Diatonic fit < 0.70 | `chord_detector_v10.py:643` |
| Tuning drift correction | Improvement > 0.10 required | `chord_detector_v10.py:635` |
| Tuning compensation | Pitch shift if offset > 0.05 semitones | `chord_detector_v10.py:373` |

### Smoothing/Voting
- **BTC path:** NO smoothing, NO voting. Raw frame-by-frame argmax → consolidate.
- **Essentia fallback:** YES, majority vote window of 21 frames (`half=10`).

---

## ROOT CAUSE ANALYSIS: "Only 1-2 Chords for Entire Songs"

### Primary Cause: THE CHORD DETECTION IS ACTUALLY WORKING CORRECTLY

After careful analysis, the BTC v10 pipeline appears architecturally sound:

1. **Frame rate is adequate:** ~93ms per frame (10.8 frames/second) is standard for chord detection
2. **The model outputs 170 classes** with proper softmax confidence
3. **No excessive filtering:** min_duration is only 0.3s, and there's no confidence cutoff
4. **The consolidation logic is correct:** it groups consecutive same-chord frames

### BUT -- Potential Causes of Poor Output

**Cause 1: Fine-tuned checkpoint may be degraded**
- The fine-tuned checkpoint (`btc_finetuned_best.pt`, 12MB, dated Mar 9) may have overfit or collapsed during fine-tuning
- There are THREE checkpoint files in `btc_finetune/checkpoints/`: `btc_finetuned_best.pt`, `btc_finetuned_best_backup.pt`, `btc_finetuned_best_pre_augment.pt`
- The code prefers the fine-tuned model over the original: `chord_detector_v10.py:296-298`
- **FIX POINT:** `chord_detector_v10.py:296-298` -- try switching to `original` checkpoint to compare

**Cause 2: LSTM path bypasses the model's native output layer**
- The v10 code manually calls `self._model.output_layer.lstm()` and `output_projection()` (lines 427-428)
- But the BTC model's native `forward()` method (btc_model.py:161-178) calls `self.output_layer(self_attn_output)` which goes through `SoftmaxOutputLayer.forward()` -- which ALSO uses the LSTM but wraps it differently
- The v10 code is calling the LSTM directly and skipping the SoftmaxOutputLayer's `forward()` method
- This is actually fine for inference -- it gets logits before softmax, which is what you want for vocab masking

**Cause 3: Normalization mismatch**
- `mean` and `std` come from the checkpoint file (lines 305-306)
- If the fine-tuned checkpoint was saved with different mean/std than what the original training used, every feature frame would be incorrectly normalized
- This would cause the model to produce near-uniform output distributions, which after argmax would give one dominant chord

**Cause 4: Harmonic stem mixing may be too noisy**
- `detect_chords_for_job()` (transcription.py:673-706) mixes guitar + piano + bass stems together
- Simple waveform addition with normalization -- no spectral mixing or level balancing
- If one stem dominates, it could confuse the chord detector
- **FIX POINT:** `processing/transcription.py:673-706` -- try individual stems instead of mix

**Cause 5: Tuning compensation may cause artifacts**
- `librosa.effects.pitch_shift()` (v10 line 374-375) uses resampling which can introduce artifacts
- This runs BEFORE CQT feature extraction, potentially degrading the signal
- **FIX POINT:** `chord_detector_v10.py:370-376` -- try disabling tuning compensation

### How to Diagnose
1. Add logging of per-frame predictions (log what chord index is predicted for each frame)
2. Compare output with fine-tuned vs original checkpoint
3. Log the softmax probability distribution -- if it's near-uniform, normalization is wrong

---

## ROOT CAUSE ANALYSIS: Garbage Guitar Tabs ("Random Meaningless Notes")

### Guitar Transcription Pipeline

The guitar tab transcription has a cascading fallback chain:
1. **Guitar v3 NN model** (`guitar_nn_transcriber.py`) -- Kong-style CRNN, 48 pitches
2. **Basic Pitch guitar** (`guitar_tab_transcriber.py`) -- Google/Spotify AMT + post-processing
3. **Melody extractor** (`melody_transcriber.py`) -- monophonic
4. **Enhanced transcriber** (`transcriber_enhanced.py`) -- polyphonic
5. **Raw Basic Pitch** (built-in to `processing/transcription.py:606-627`)

### Critical Finding: Guitar v3 NN Model Does NOT Exist

```
$ ls backend/models/pretrained/
best_drum_model.pt
best_piano_model.pt
v7_chord_model.pt
v8_chord_model.pt
v8_classes.json
v9_chord_model.pt
v9_classes.json
```

**There is NO `best_guitar_model.pt`** -- so `GUITAR_NN_MODEL_AVAILABLE = False`.

The system falls through to **Basic Pitch guitar tab transcriber** (`guitar_tab_transcriber.py`).

### Basic Pitch Guitar Problems

**Problem 1: Basic Pitch is not designed for guitar**
- Basic Pitch (ICASSP 2022) was trained on piano/vocal data
- It hallucinates harmonics, overtones, and bleed-through from other stems as real notes
- Even with the post-processing pipeline, it produces false positives

**Problem 2: Thresholds are too aggressive for guitar**
- `onset_threshold=0.40` (guitar_tab_transcriber.py:631) -- lower than default, catches more noise
- `frame_threshold=0.25` (line 632) -- very low, will detect overtones as sustained notes
- `minimum_note_length=50` (line 633) -- 50ms is short enough to catch artifacts
- These were tuned for "separated stems" but separation isn't perfect

**Problem 3: Key filtering may remove correct notes**
- `_filter_by_key()` removes off-scale notes below 30th percentile velocity (guitar_tab_transcriber.py:216-230)
- If key detection is wrong, it removes correct notes and keeps wrong ones

**Problem 4: Polyphony limiting keeps wrong notes**
- `_limit_polyphony()` keeps notes with highest velocity when >6 overlap (line 540-543)
- Basic Pitch confidence (mapped to velocity) doesn't correlate well with actual note accuracy for guitar

**Problem 5: String/fret assignment is correct but based on garbage input**
- The FretMapper logic (lines 267-500) is well-engineered
- But garbage-in-garbage-out: if Basic Pitch gives wrong pitches, perfect fret mapping produces wrong tabs

### Raw Basic Pitch Fallback (transcription.py:606-627)
If even the guitar tab transcriber fails quality check, raw Basic Pitch runs:
```python
predict_and_save(
    onset_threshold=0.6,
    frame_threshold=0.4,
    minimum_note_length=80,
)
```
These are more conservative thresholds but still produce garbage for guitar.

---

## Frontend Chord Data Format

### API Response Format (`GET /api/chords/<job_id>`)
```json
{
    "job_id": "abc123",
    "chords": [
        {
            "time": 0.0,
            "duration": 2.5,
            "chord": "Am",
            "root": "A",
            "quality": "min",
            "confidence": 0.85,
            "detector_version": "v10"
        }
    ],
    "key": "Am",
    "available": true,
    "detector_version": "v10",
    "chord_count": 45,
    "has_inversions": false
}
```

### Chord Sheet Format (`GET /api/chord-sheet/job/<job_id>`)
```json
{
    "title": "Song Name",
    "artist": "Artist Name",
    "key": "Am",
    "chords_used": ["Am", "G", "F", "C"],
    "content": "[ch]Am[/ch]  [ch]G[/ch]\n[ch]F[/ch]  [ch]C[/ch]",
    "source": "StemScribe AI (v10)",
    "chord_events": [{"time": 0.0, "chord": "Am"}, ...]
}
```

### Frontend Rendering Priority (practice.html:2390-2400)
1. Songsterr chords (human-verified) via `loadChordChart(songsterrId)`
2. AI chord sheet via `loadJobChordChart(job.job_id)`
3. Falls back to "Loading chords..." if neither available

The frontend checks `currentJob.chord_progression.length > 0` to decide whether to show chords.

---

## Songsterr Integration

- **Route:** `backend/routes/songsterr.py`
- **Chord endpoint:** `GET /songsterr/chords/<song_id>`
- The Songsterr pipeline is completely independent from BTC chord detection
- Songsterr provides measure-based chords with beat positions
- Frontend prefers Songsterr over AI detection when available
- No interaction between Songsterr and BTC -- they don't merge or compare results

---

## Recommended Fix Points

### Chord Detection Fixes (Priority Order)

1. **Compare checkpoints** (`chord_detector_v10.py:296-298`)
   - Test with original `btc_model_large_voca.pt` vs fine-tuned
   - Add a config flag to select checkpoint

2. **Add diagnostic logging** (`chord_detector_v10.py:437-449`)
   - Log softmax probability distribution stats (entropy, max prob)
   - Log how many unique chord classes are predicted per song
   - Log if the model is outputting mostly 'N' (no chord)

3. **Consider replacing BTC entirely**
   - Use a modern pre-trained model like `chord-jams` or `autochord`
   - Or use an LLM-based approach with audio embeddings
   - Integration point: `chord_detector_v10.py:355` (`_detect_btc` method)

4. **Fix stem mixing** (`processing/transcription.py:673-706`)
   - Try guitar stem alone instead of mixing all harmonics
   - Level-balance stems before mixing

### Guitar Tab Fixes (Priority Order)

1. **Get a proper guitar model trained**
   - The infrastructure exists (`guitar_nn_transcriber.py`) but no checkpoint
   - Train on GuitarSet or IDMT-SMT-Guitar
   - Checkpoint goes to `backend/models/pretrained/best_guitar_model.pt`

2. **Raise Basic Pitch thresholds** (`guitar_tab_transcriber.py:629-636`)
   - `onset_threshold=0.55` (from 0.40)
   - `frame_threshold=0.35` (from 0.25)
   - `minimum_note_length=80` (from 50)

3. **Add Basic Pitch output validation**
   - If note density > 20 notes/sec, something is wrong -- reject
   - If pitch range > 3 octaves, likely harmonics -- filter

4. **Consider Spotify/Google alternatives**
   - `basic-pitch` v0.4+ has improved guitar handling
   - MT3 (Google) handles guitar better but is heavier

---

## Summary

| Issue | Root Cause | Severity | Fix Effort |
|-------|-----------|----------|------------|
| 1-2 chords per song | Likely fine-tuned checkpoint degradation or normalization mismatch | HIGH | LOW (swap checkpoint) |
| Wrong key detection | Key detection is independent and likely correct; chords may just be wrong | MEDIUM | LOW (already has fallback) |
| Garbage guitar tabs | No trained guitar model; Basic Pitch is wrong tool for guitar | HIGH | HIGH (need training) |
| Basic Pitch threshold | Too aggressive for separated guitar stems | MEDIUM | LOW (adjust 3 numbers) |
