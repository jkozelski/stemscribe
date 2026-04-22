# StemScriber Transcription Failure Inventory

**Purpose:** Brief incoming research agents on what we've already tried that didn't work, so we stop looping.
**Date compiled:** 2026-04-17

---

## Environment

- **Stack:** Python/Flask backend, Modal (serverless GPU A10G/A100), Hetzner VPS ($8/mo 4CPU/8GB RAM, NO GPU)
- **Stem separation (UPSTREAM, WORKING):** BS-RoFormer-SW on Modal → 6 stems (vocals, guitar, bass, drums, piano, other)
- **Chord detection (WORKING):** stem-aware, bass-root-first, 95% quality after Apr 16 bleed-simplification pass
- **Launch date:** 2026-05-12 (22 days)

## What's failed (do NOT re-try these without new justification)

### 1. Trimplexx custom CRNN for guitar tab (Apr ~1-13, 2026)
- **Model:** `trimplexx_guitar_model.pt` — 79MB, 2-layer bidirectional GRU (hidden 768), CQT input @ 22050Hz, 168 bins, trained on GuitarSet
- **Reported metrics (test set):** TDR-F1 0.85, MPE-F1 0.87, onset-F1 0.81 — looked great on paper
- **Real-world behavior:** complete nonsense on BS-RoFormer guitar stems from real songs. User calls output "wrong AI guesses" — 2 months of this.
- **Root cause suspected:** GuitarSet is ~60 pieces of clean isolated guitar. BS-RoFormer stems have bleed, compression artifacts, distortion → severe domain gap. Augmentations (reverb, EQ, clipping) during training weren't enough.
- **Runtime:** `backend/trimplexx_transcriber.py` + model in `backend/models/pretrained/`

### 2. CRNN models for drums/piano/bass (Apr, 2026)
- **Files:** `best_drum_model.pt` (109MB), `best_piano_model.pt` (138MB), `best_bass_model.pt` (92MB), `best_guitar_model.pt` (71MB)
- **Sources:** Drums trained from scratch, Piano domain-adapted from Kong piano, Guitar/Bass transfer from piano
- **Real-world behavior:** nonsense output on real BS-RoFormer stems
- **Root cause:** same domain gap + training data volume issue. Our datasets aren't large enough for polyphonic CRNN generalization.

### 3. Phi-3 QLoRA chart formatter (~Apr 10-16, 2026)
- **Goal:** fine-tune Phi-3 to format detected chords into lead-sheet JSON
- **Training runs:**
  - Run 1: 1 epoch, A10G — output essay prose instead of JSON
  - Run 2: 3 epochs, A100-80GB, 6 hours, timed out at step 2500/2907 (86%)
- **Both adapters:** repetition-loop hallucination, filled lyric fields with "Title / by Artist / from the album" infinitely
- **Root cause:** contaminated training data in `backend/training_data/chart_formatter/train.jsonl`
- **Status:** OFFICIALLY ABANDONED. Rule-based `chart_formatter.py` ships instead. Code scaffolding left in place, commented out. ~$24 Modal spent.

### 4. BTC + V8 chord models
- Not a failure — these WORK for chord detection at 95% quality. Listed for completeness. `btc_finetuned_best.pt` (170 chords) + `v8_chord_model.pt` (337 jazz voicings). The simplification pass in `backend/stem_chord_detector.py` `_simplify_bleed_extensions()` is what got chord quality from 15% → 95%.

## Lesson pattern (what keeps repeating)

We keep going: "let's train our own model" → 2-4 weeks → ships nonsense on real audio. Then "let's use a pretrained model" → some progress → back to training. User is tired of the loop. **This round MUST end with a deployed solution using existing pretrained models, not a new training project.**

## Constraints for the answer

1. **Must ship before May 12, 2026 (22 days).** No 6-week ML training projects.
2. **Must run on Modal** (GPU available, serverless) OR on the Hetzner VPS (CPU-only, 8GB RAM).
3. **License must be commercial-usable** (MIT, Apache-2.0, or similar). GPL viral licenses NO.
4. **Input is ALREADY a clean isolated stem** (BS-RoFormer output). Don't re-invent separation.
5. **Output must be MIDI** (pipeline already has `midi_to_notation.py` → MusicXML → OSMD/alphaTab rendering).

## External research already done (don't redo)

A prior agent surveyed the SOTA 2025-2026 and recommended:

| Task | Recommended model | License | Notes |
|------|------|------|------|
| Guitar tab | `xavriley/HighResolutionGuitarTranscription` (pretrained) + separate CP tab arranger | Apache-2.0-ish | Best zero-shot on GuitarSet, runs on Modal GPU |
| Bass | Spotify Basic Pitch | MIT | Already works for guitar in our pipeline; bass is monophonic-friendly |
| Piano | Sony `hFT-Transformer` | MIT | SOTA on MAESTRO: 97.44 note-F1 (Toyama et al., ISMIR 2023) |
| Drums | `MZehren/ADTOF` | MIT | Standard 5-class kit, paper-grade results |

The specialists assigned to each task should verify/challenge this and produce a deployable integration plan with real risks, not rubber-stamp it.
