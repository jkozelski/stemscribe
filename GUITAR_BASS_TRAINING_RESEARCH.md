# Guitar/Bass CRNN Training Failure: Research & Solutions

## ML Research Agent Report — February 2026

---

## 1. Root Cause Analysis

### What Failed
- **Guitar CRNN** trained on RunPod (RTX 4090) with Dice Loss + Focal Loss
- F1 reached only **0.052 after 107 epochs** — effectively a failed training
- **Bass CRNN v1** collapsed to predicting all zeros (plain BCELoss)
- Bass v2 with Focal Loss + pos_weight was created but training results unknown

### Root Causes Identified

| Problem | Details |
|---------|---------|
| **Insufficient data** | GuitarSet has only **288 samples** (3 hours, 6 guitarists). This is far too few for a 5M parameter CRNN model. |
| **Extreme class imbalance** | Onset positive rate is ~0.055% (1:1835 ratio). BCELoss causes collapse to all-zeros since predicting "no onset" is 99.945% accurate. |
| **Wrong task framing** | The `train_guitar_model/` directory contains a **MelBand-Roformer source separation** config (lead vs rhythm guitar), NOT a transcription model. This is a fundamentally different problem. |
| **No pre-training or transfer learning** | Training from random initialization on 288 samples has near-zero chance of converging for a complex CRNN. |
| **Weak augmentation** | Only gain jitter (0.7-1.3x) and noise addition. No pitch shifting, time stretching, reverb, EQ, or SpecAugment. |

---

## 2. State of the Art: What Actually Works (2024-2026)

### A. Domain Adaptation from Piano (BEST APPROACH)

**Paper:** "High Resolution Guitar Transcription via Domain Adaptation" (Riley et al., ICASSP 2024)

**Key idea:** Take a pre-trained high-resolution **piano transcription model** (Kong et al., ~20M params trained on MAESTRO) and fine-tune it on a small guitar dataset.

**Results on GuitarSet:**
- Zero-shot (no GuitarSet training): **87.3% onset F1**
- With GuitarSet fine-tuning: **89.7% onset F1**

**Training procedure:**
- Base model: Kong et al. piano transcription model (pre-trained on MAESTRO)
- Fine-tuning data: 79 jazz guitar recordings (4 hours), commercially transcribed
- Learning rate: 1e-5 (very small — preserving pre-trained features)
- Batch size: 4
- Steps: 100K (~10 epochs) with 0.9 LR decay every 10K steps
- Augmentation: Random transposition +/- 2 semitones

**Why this works for StemScribe:**
- MAESTRO piano model is freely available
- Only needs ~4 hours of guitar data for fine-tuning
- The model already understands note onsets, frames, velocity from piano
- Guitar transcription is close enough to piano that features transfer well

### B. GAPS Dataset + CRNN Benchmark

**Paper:** "GAPS: A Large and Diverse Classical Guitar Dataset" (Riley et al., ISMIR 2024)

**Dataset:** 14 hours of classical guitar, 200+ performers, note-level MIDI alignments
- Available on Zenodo (CC license): https://zenodo.org/records/13962272

**Results on GuitarSet:**
- Zero-shot (trained on GAPS only): **88.1% F1**
- With GuitarSet: **91.2% F1** (current SOTA)

**Architecture:** CRNN with:
- Log mel-spectrogram input (10ms resolution)
- Conv layers across frequency dimension only (preserves time resolution)
- GRU recurrent layers
- Outputs: onset, offset, frame activity, velocity per pitch per time step
- Augmentation: Pitch shifting +/- 3 semitones

### C. SynthTab: Synthesized Pre-Training Data

**Paper:** "SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription" (ICASSP 2024)

**Key idea:** Generate massive synthetic guitar audio from DadaGP tablature collection:
- 11,132 acoustic guitar songs (3,511 hours)
- 38,838 electric guitar songs (9,601 hours)
- Uses professional guitar VST plugins for synthesis
- Perfectly aligned audio + tablature annotations

**Training procedure:**
1. Pre-train on SynthTab (thousands of hours)
2. Fine-tune on real GuitarSet data (3 hours)
3. Cross-dataset experiments show pre-training significantly mitigates overfitting

**GitHub:** https://github.com/yongyizang/SynthTab

### D. Aggressive Data Augmentation (trimplexx CRNN)

**Project:** https://github.com/trimplexx/music-transcription

**Result:** 0.8736 MPE F1 on GuitarSet using ONLY GuitarSet data (no external datasets)

**Architecture:**
- CNN: 5 layers (32 -> 64 -> 128 -> 128 channels) with BatchNorm + ReLU + MaxPool
- RNN: 2-layer Bidirectional GRU, hidden_size=768, dropout=0.5
- CQT input: 168 bins (7 octaves x 24 bins/octave), hop=512, sr=22050
- Multi-task: onset [T, 6] + fret classification [T, 6, 22]

**Critical insight: Augmentation was the "decisive factor":**
- Time stretching (60% prob, rate 0.8-1.2)
- Noise addition (70%, level 0.001-0.01)
- Random gain (70%, factor 0.6-1.4)
- Reverb (40%, decay 0.1-0.45s)
- EQ filtering (50%, bandpass 250Hz-4.5kHz)
- Clipping (30%, threshold 0.5-0.9)
- SpecAugment (time: 40 masks, frequency: 26 masks)

**Training hyperparameters:**
- Learning rate: 0.0003
- Onset loss weight: 9.0
- Onset positive class weight: 6.0

### E. New Datasets Available (2024-2025)

| Dataset | Size | Type | License | URL |
|---------|------|------|---------|-----|
| **GAPS** | 14 hours | Classical guitar, 200+ performers | CC | zenodo.org/records/13962272 |
| **GOAT** | 5.9 hours (29.5h augmented) | Electric guitar, tablature annotations | CC | arxiv.org/abs/2509.22655 |
| **Guitar-TECHS** | Multi-hour | Electric guitar techniques, 3 guitarists | CC BY 4.0 | zenodo.org/records/14963133 |
| **SynthTab** | 13,000+ hours | Synthesized from DadaGP tablatures | CC BY-NC 4.0 | github.com/yongyizang/SynthTab |
| **GuitarSet** | 3 hours | 6 guitarists, hexaphonic pickup | CC | github.com/marl/GuitarSet |
| **Slakh2100** | 145 hours total | Multi-instrument (has guitar stems) | CC | zenodo.org/records/4599666 |

---

## 3. Recommended Strategy for StemScribe

### Short-Term: Use Pre-Trained Models (Weeks, not months)

**Option A: Domain Adaptation (RECOMMENDED)**
1. Download Kong et al. pre-trained piano transcription model
2. Fine-tune on GAPS (14h) + GuitarSet (3h) + Guitar-TECHS
3. Expected result: ~88-91% onset F1 (matches SOTA)
4. Training time: ~4-8 hours on RunPod RTX 4090
5. This replaces Basic Pitch for guitar transcription entirely

**Option B: Use trimplexx-style CRNN from scratch on GuitarSet**
1. Implement their exact architecture (5-layer CNN + 2-layer BiGRU, hidden=768)
2. Apply their full augmentation suite (this was the decisive factor)
3. Expected result: ~87% F1 on GuitarSet
4. Risk: still limited by GuitarSet's 288 samples for generalization

### Medium-Term: Full Training Pipeline

1. **Download GAPS + GOAT + Guitar-TECHS** (20+ hours of real annotated guitar)
2. **Generate SynthTab data** for pre-training (hundreds of hours, free)
3. Pre-train CRNN on SynthTab
4. Fine-tune on real data (GAPS + GOAT + Guitar-TECHS + GuitarSet)
5. Expected result: >91% onset F1 with strong generalization

### For Bass

The bass training is actually in much better shape than guitar:
- **Slakh2100 already provides 1000+ bass stems** with aligned MIDI
- The v2 training script (`train_bass_runpod.py`) correctly uses Focal Loss + pos_weight
- **Key fix needed:** The bass CRNN returns logits but the inference code in `bass_transcriber.py` currently uses Basic Pitch, not the trained model
- Once the bass model trains successfully on Slakh2100, integrate it to replace Basic Pitch for bass

### What NOT To Do

- Do NOT continue training from scratch on GuitarSet alone with the current architecture
- Do NOT use the MelBand-Roformer config for transcription (that's for source separation)
- Do NOT use plain BCELoss (use Focal + pos_weight or BCEWithLogitsLoss + pos_weight)
- Do NOT skip augmentation — it's the single biggest factor for small datasets

---

## 4. Specific Code Changes

### Guitar Transcription: Replace Basic Pitch

The current `guitar_tab_transcriber.py` uses Basic Pitch which achieves ~66-79% F1 on guitar.
Replacing it with a domain-adapted piano model would jump to ~88-91% F1.

**Implementation plan:**
1. Download Kong et al. model checkpoint
2. Write a `guitar_nn_transcriber.py` (analogous to `drum_nn_transcriber.py`)
3. Fine-tune on GAPS + GuitarSet
4. Keep Basic Pitch as fallback when model unavailable

### Bass Transcription: Deploy Trained CRNN

The current `bass_transcriber.py` uses Basic Pitch but a CRNN architecture already exists.
Once trained on Slakh2100 with v2 loss functions, deploy it:

1. Run `train_bass_runpod.py` on RunPod (expects ~50 epochs, 4-8 hours on A100)
2. Copy `best_bass_model.pt` to `backend/models/pretrained/`
3. Update `bass_transcriber.py` to load the CRNN model
4. Keep Basic Pitch as fallback

### Loss Function Fix (Critical)

For any future CRNN training, use this pattern:
```python
# For onsets (extremely sparse: ~0.05% positive)
onset_loss = FocalBCEWithLogitsLoss(pos_weight=50.0, gamma=2.0)

# For frames (less sparse: ~1-5% positive)
frame_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))

# Track F1/precision/recall, NOT just loss
# Collapse detection: if recall < 0.01 for 10 epochs, stop
```

### Augmentation Pipeline (Critical)

```python
# Minimum viable augmentation for guitar/bass training
augmentations = {
    'pitch_shift': {'prob': 0.5, 'range': (-3, 3)},      # semitones
    'time_stretch': {'prob': 0.6, 'range': (0.8, 1.2)},
    'gain': {'prob': 0.7, 'range': (0.6, 1.4)},
    'noise': {'prob': 0.7, 'range': (0.001, 0.01)},
    'reverb': {'prob': 0.4, 'decay_range': (0.1, 0.45)},
    'eq_filter': {'prob': 0.5, 'band': (250, 4500)},
    'spec_augment': {'time_masks': 40, 'freq_masks': 26},
}
```

---

## 5. References

1. Riley et al. "High Resolution Guitar Transcription via Domain Adaptation" ICASSP 2024
   - https://arxiv.org/abs/2402.15258
2. Riley et al. "GAPS: A Large and Diverse Classical Guitar Dataset" ISMIR 2024
   - https://arxiv.org/abs/2408.08653
3. Zang et al. "SynthTab: Leveraging Synthesized Data for Guitar Tablature Transcription" ICASSP 2024
   - https://arxiv.org/abs/2309.09085
4. trimplexx/music-transcription (0.87 F1 CRNN on GuitarSet)
   - https://github.com/trimplexx/music-transcription
5. "GOAT: A Large Dataset of Paired Guitar Audio Recordings and Tablatures" ISMIR 2025
   - https://arxiv.org/abs/2509.22655
6. "Guitar-TECHS: An Electric Guitar Dataset" ICASSP 2025
   - https://arxiv.org/abs/2501.03720
7. "Exploring Procedural Data Generation for Guitar Fingerpicking Transcription" 2025
   - https://arxiv.org/abs/2508.07987
8. Kong et al. "High-Resolution Piano Transcription with Pedals" (base model for domain adaptation)
   - Used as pre-training base in [1] and [2]
