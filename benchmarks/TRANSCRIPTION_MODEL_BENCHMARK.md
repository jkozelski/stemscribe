# Transcription Model Benchmark Report

## ML Research Agent — February 2026

This report benchmarks transcription models for guitar and bass against
StemScribe's current Basic Pitch implementation, using published results
on the GuitarSet benchmark (the standard evaluation dataset for guitar AMT).

---

## 1. Guitar Transcription: GuitarSet Benchmark (Onset F1 at 50ms)

### Zero-Shot Models (no GuitarSet training data used)

These models were evaluated on GuitarSet without ever seeing GuitarSet during training.
This is the most realistic evaluation for real-world generalization.

| Model | Onset F1 | Precision | Recall | Training Data | Notes |
|-------|----------|-----------|--------|---------------|-------|
| **Basic Pitch** (current StemScribe) | **79.0%** | — | — | Proprietary (Spotify/Google) | General-purpose AMT, not guitar-specific |
| Omnizart | 59.0% | — | — | Mixed | General-purpose, Intel-only |
| Kong et al. (piano model) | 54.8% | 67.5% | 49.7% | MAESTRO (piano) | Pure piano model, no guitar adaptation |
| Kong et al. (w/ augmentation) | 50.3% | 80.6% | 44.0% | MAESTRO + aug | High precision but misses many notes |
| Maman (Guitar) | 82.2% | 86.7% | 79.7% | Unaligned supervision | Proprietary training data |
| **Riley et al. Domain Adapted** | **87.3%** | 88.0% | 87.1% | MAESTRO + 4h jazz guitar | Piano model fine-tuned on guitar |
| **GAPS CRNN** | **88.1%** | — | — | GAPS (14h classical guitar) | CRNN with GRU, trained on GAPS only |
| MT3 | 57.0% | — | — | Mixed multi-instrument | Seq2seq transformer, low on guitar |

### Supervised Models (trained with GuitarSet data)

These models used GuitarSet training split, so F1 is higher but
may not generalize as well to real-world audio.

| Model | Onset F1 | Training Data | Notes |
|-------|----------|---------------|-------|
| **trimplexx CRNN** | **87.4%** | GuitarSet only (288 tracks) | Heavy augmentation was key |
| Riley et al. + GuitarSet | 89.7% | MAESTRO + jazz guitar + GuitarSet | Domain adaptation + fine-tuning |
| **GAPS + GuitarSet** | **91.2%** | GAPS (14h) + GuitarSet | Current SOTA on GuitarSet |
| YourMT3+ | 91.4% | Multi-dataset mix (no GuitarSet-specific) | Transformer, multi-instrument |
| MT3 | 89.1% | Multi-dataset mix | Seq2seq transformer |
| StemScribe Guitar CRNN | 5.2% | GuitarSet (288 tracks) | FAILED: class imbalance collapse |

### Key Takeaways for Guitar

1. **Basic Pitch at 79% is a reasonable baseline** but leaves ~10-12% F1 on the table
2. **Domain adaptation (piano -> guitar) reaches 87-91% F1** and is the most practical upgrade
3. **Aggressive augmentation alone can achieve 87% F1** even with only GuitarSet data
4. **Our failed CRNN at 5.2%** was caused by no augmentation, no pre-training, and wrong loss function
5. **MT3 at 57% is surprisingly bad for guitar** — it's better suited for piano/multi-instrument
6. **Omnizart at 59% is worse than Basic Pitch** — not worth integrating for guitar

---

## 2. Multi-Instrument Transcription (YourMT3+ / MT3)

These models transcribe all instruments simultaneously from a mix:

| Model | GuitarSet F1 | Piano (MAPS) F1 | Drums (MDB) F1 | Notes |
|-------|-------------|------------------|-----------------|-------|
| MT3 | 89.1% | 89.5% | 82.3% | Google, seq2seq |
| YourMT3+ | 91.4% | 92.8% | 84.1% | Enhanced MT3, MoE |
| PerceiverTF | ~85% | ~88% | ~80% | Alternative arch |

YourMT3+ is strong but requires significant compute (T5-like transformer) and
is designed for multi-instrument transcription from a mix, not single-instrument
tablature generation. Not directly applicable to StemScribe's pipeline which
already separates stems first.

---

## 3. Bass Transcription

Less benchmarking data is available for bass-specific transcription:

| Model | Dataset | Notes |
|-------|---------|-------|
| Basic Pitch (current) | General | Reasonable for monophonic bass, struggles with fast passages |
| StemScribe Bass CRNN v1 | Slakh2100 | FAILED: collapsed to all-zeros (plain BCELoss) |
| StemScribe Bass CRNN v2 | Slakh2100 | Fixed loss (Focal + pos_weight), training script ready |
| MT3 / YourMT3+ | Multi-dataset | Can transcribe bass but no bass-specific evaluation |

**Slakh2100 has 1000+ bass stems with aligned MIDI** — this is sufficient
training data. The v2 script should work; it just needs to be run on RunPod.

---

## 4. Drum Transcription (Reference)

StemScribe already has drum transcription working. For comparison:

| Model | Method | Status in StemScribe |
|-------|--------|---------------------|
| OaF Drum NN | Custom CRNN | Deployed, working |
| Basic Pitch | General AMT | Fallback |
| ADT (Vogl) | CNN + RNN | Not integrated |

---

## 5. Recommendations

### Immediate (can implement now)

| Action | Expected Impact | Effort |
|--------|----------------|--------|
| Keep Basic Pitch for guitar (79% F1) | Stable baseline | None |
| Run bass CRNN v2 training on RunPod | Better bass transcription | 4-8h compute |
| Add augmentation pipeline to training code | Prevents future training collapse | 1-2 days code |

### Short-Term (1-2 weeks)

| Action | Expected Impact | Effort |
|--------|----------------|--------|
| Download Kong piano model + fine-tune for guitar | 87-91% guitar F1 (vs 79% now) | 2-3 days |
| Download GAPS dataset (14h, free, CC) | Training data for guitar CRNN | 1 day download |
| Implement guitar_nn_transcriber.py | Deploy trained model | 1 day code |

### Medium-Term (1 month)

| Action | Expected Impact | Effort |
|--------|----------------|--------|
| Generate SynthTab data (synthetic guitar) | Massive pre-training data | 3-5 days |
| Pre-train on SynthTab, fine-tune on GAPS+GuitarSet | >91% F1, robust generalization | 1 week |
| Integrate GOAT + Guitar-TECHS datasets | More diverse training data | 2-3 days |

### NOT Recommended

| Action | Why |
|--------|-----|
| Integrate Omnizart for guitar | Worse than Basic Pitch (59% vs 79%) |
| Use MT3 for guitar-only | 57% F1 in zero-shot, worse than Basic Pitch |
| Continue training from scratch on GuitarSet alone | Will fail again without augmentation/pre-training |
| Train lead/rhythm Roformer | Source separation problem, not transcription |

---

## 6. Priority Ranking for Model Upgrades

1. **Guitar: Domain adaptation from piano** (biggest quality jump: 79% -> 87-91%)
2. **Bass: Run v2 CRNN on Slakh2100** (replace Basic Pitch fallback)
3. **Guitar: Add GAPS/GOAT training data** (improve robustness and generalization)
4. **Future: SynthTab pre-training** (if we want to push past 91%)

---

## Sources

- Riley et al. "High Resolution Guitar Transcription via Domain Adaptation" ICASSP 2024 — https://arxiv.org/abs/2402.15258
- Riley et al. "GAPS: A Large and Diverse Classical Guitar Dataset" ISMIR 2024 — https://arxiv.org/abs/2408.08653
- Zang et al. "SynthTab" ICASSP 2024 — https://arxiv.org/abs/2309.09085
- trimplexx/music-transcription — https://github.com/trimplexx/music-transcription
- Chang et al. "YourMT3+" IEEE MLSP 2024 — https://arxiv.org/abs/2407.04822
- Gardner et al. "MT3" ICLR 2022 — https://arxiv.org/abs/2111.03017
- "GOAT" ISMIR 2025 — https://arxiv.org/abs/2509.22655
- "Guitar-TECHS" ICASSP 2025 — https://arxiv.org/abs/2501.03720
