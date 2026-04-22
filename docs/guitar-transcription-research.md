# Guitar Transcription Research
**Date:** 2026-03-17
**Status:** Research complete, chord-to-tab fallback implemented

## The Problem
Basic Pitch (Google/Spotify) is a piano-centric AMT model. When applied to guitar stems, it produces:
- Wrong pitches (particularly for distorted/overdriven guitar)
- Ghost notes and harmonics misidentified as fundamentals
- Wrong rhythms (note durations wildly off)
- No awareness of guitar idiom (open strings, chord shapes, fret positions)

The existing `best_guitar_model.pt` (Kong-style CRNN fine-tuned on GuitarSet) does not exist -- training on RunPod was attempted but no viable checkpoint was produced.

## Models Surveyed

### 1. TabCNN (Wiggins, ISMIR 2019)
- **Repo:** https://github.com/andywiggins/tab-cnn
- **Architecture:** CNN on CQT spectrograms, outputs fret labels per string
- **Dataset:** GuitarSet (solo acoustic guitar only)
- **Pros:** Purpose-built for guitar tablature, lightweight
- **Cons:** No pretrained weights for download; trained only on solo acoustic guitar (GuitarSet is tiny: 360 recordings, ~3 hours); would need retraining; unlikely to generalize to electric guitar, band mixes
- **MPS compatible:** Yes (simple CNN, PyTorch)
- **Verdict:** Would need training from scratch, GuitarSet too small for production quality

### 2. FretNet (Cwitkowitz et al., 2022)
- **Repo:** https://github.com/cwitkowitz/guitar-transcription-continuous
- **Architecture:** Deeper TabCNN variant with continuous-valued pitch output
- **Dataset:** GuitarSet (cross-validation)
- **Pros:** Handles pitch bends/vibrato, guitar-specific output layer
- **Cons:** No pretrained weights available for download; requires full training pipeline setup; GuitarSet-only
- **MPS compatible:** Yes (PyTorch)
- **Verdict:** Best research architecture for guitar but needs training infrastructure

### 3. Guitar Transcription with Inhibition (Cwitkowitz et al., 2022)
- **Repo:** https://github.com/cwitkowitz/guitar-transcription-with-inhibition
- **Architecture:** TabCNN + inhibition-based output layer using DadaGP statistics
- **Pros:** Uses real tablature statistics to bias toward playable outputs
- **Cons:** No pretrained weights; requires DadaGP data processing; GuitarSet-only evaluation
- **MPS compatible:** Yes
- **Verdict:** Interesting constraint mechanism but same training problem

### 4. trimplexx/music-transcription (2024)
- **Repo:** https://github.com/trimplexx/music-transcription
- **Architecture:** CRNN (Conv + BiGRU + multi-task heads)
- **Performance:** 0.87 MPE F1 on GuitarSet (SOTA for GuitarSet-only training)
- **Cons:** No pretrained weights published; GuitarSet-only; unclear license
- **MPS compatible:** Yes (PyTorch)
- **Verdict:** Best reported GuitarSet performance but no downloadable model

### 5. MT3 (Google Magenta, 2022)
- **Repo:** https://github.com/magenta/mt3
- **Architecture:** T5-based transformer, multi-instrument
- **Pros:** Handles multiple instruments including guitar; pretrained models available
- **Cons:** Uses T5X/JAX (not PyTorch); massive model; designed for Colab/TPU; no Apple MPS support; complex dependency chain (TensorFlow + JAX); guitar quality is OK but not specialized
- **MPS compatible:** No (JAX/TPU-oriented)
- **Verdict:** Too heavy and wrong framework for local deployment

### 6. YourMT3+ (Cheuk et al., 2024)
- **Repo:** https://github.com/mimbres/YourMT3
- **Architecture:** Enhanced MT3 with hierarchical attention + MoE
- **Pros:** Multi-instrument, pretrained checkpoints, PyTorch; better than MT3
- **Cons:** Very large model (transformer); complex setup; not guitar-specialized; would need significant integration work
- **MPS compatible:** Potentially (PyTorch), but untested
- **Verdict:** Most promising "real model" option but high integration effort

### 7. Omnizart (Music-and-Culture-Technology-Lab)
- **Repo:** https://github.com/Music-and-Culture-Technology-Lab/omnizart
- **Architecture:** Multi-task transcription (piano, drums, chord, vocal)
- **Cons:** Incompatible with ARM MacOS (Apple Silicon); no guitar-specific module; uses TensorFlow 2.x
- **MPS compatible:** No (ARM incompatible per GitHub issue #38)
- **Verdict:** Not viable on M3 Max

## Recommendation

### Short-term (implemented): Chord-to-Tab Generation
Since no production-ready guitar transcription model exists that:
1. Has downloadable pretrained weights
2. Runs on Apple MPS
3. Handles real-world guitar recordings (not just solo acoustic GuitarSet)

The best immediate approach is **chord-to-tab generation**:
- Use the already-detected chord progression (from BTC/Essentia ensemble)
- Map each chord to a standard guitar voicing from CHORD_SHAPES
- Place chord shapes at beat positions using tempo/beat tracking
- Use onset detection to determine strum rhythm
- This produces a **rhythm chart** (like what guitarists actually use) rather than note-for-note transcription
- Falls back to Basic Pitch for sections with no chord data

### Medium-term: Fine-tune on larger dataset
- Collect guitar tablature data from DadaGP (700K+ tabs in GuitarPro format)
- Use the FretNet or trimplexx CRNN architecture
- Train on M3 Max with MPS acceleration
- Would take 2-4 weeks of engineering + training time

### Long-term: YourMT3+ integration
- Most promising general-purpose model
- Would need PyTorch MPS testing and integration wrapper
- Could replace Basic Pitch for all instruments, not just guitar

## Implementation Status
- [x] Research documented
- [x] Chord-to-tab fallback implemented in `guitar_tab_transcriber.py`
- [x] Uses CHORD_VOICINGS dict with standard guitar shapes
- [x] Onset detection for strum rhythm
- [x] Integrates with existing pipeline (falls through when Basic Pitch quality is low)
- [x] All tests passing
