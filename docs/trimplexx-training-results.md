# trimplexx CRNN Training Results

**Date:** 2026-04-12
**Platform:** Modal A10G GPU (NVIDIA A10G, 22.1GB VRAM)
**Training time:** 96.7 minutes (111 epochs, early stopped at patience=25)
**Cost:** ~$1.78 (96.7 min at $1.10/hr)

---

## Model Configuration (run_72 hyperparameters)

| Parameter | Value |
|-----------|-------|
| Architecture | CNN (5-layer) + Bidirectional GRU |
| RNN type | GRU |
| RNN hidden size | 768 |
| RNN layers | 2 |
| RNN dropout | 0.5 |
| Total parameters | 20,673,354 |
| Model size | 78.9 MB (float32) |
| Learning rate | 0.0003 (AdamW) |
| Onset loss weight | 9.0 |
| Onset pos weight | 6.0 |
| Batch size | 2 |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.2) |
| Early stopping | patience=25 on val_tdr_f1 |

## Data Augmentation

Full augmentation pipeline with raw audio (matching original run_72):
- Time stretch (p=0.6, range 0.8-1.2)
- Gaussian noise (p=0.7, range 0.001-0.01)
- Random gain (p=0.7, range 0.6-1.4)
- Reverb (p=0.4, scipy convolution)
- EQ bandpass (p=0.5, scipy IIR filter)
- Clipping (p=0.3, threshold 0.5-0.9)
- SpecAugment (time mask=40, freq mask=26)

## Dataset

GuitarSet: 358 tracks (2 problematic excluded), split 80/10/10
- Train: 286 tracks
- Validation: 36 tracks
- Test: 36 tracks

---

## Test Set Results

### Tab Detection Rate (TDR) -- note-level with string+fret match
| Metric | Score |
|--------|-------|
| **TDR Precision** | **0.8795** |
| **TDR Recall** | **0.8255** |
| **TDR F1** | **0.8499** |

### Multi-Pitch Estimation (MPE) -- frame-level
| Metric | Score |
|--------|-------|
| MPE Precision | 0.8784 |
| MPE Recall | 0.8584 |
| **MPE F1** | **0.8683** |

### Onset Detection -- event-level
| Metric | Score |
|--------|-------|
| Onset Precision | 0.9554 |
| Onset Recall | 0.7280 |
| Onset F1 | 0.8079 |

---

## Comparison with Reference Scores

| Metric | Our Run | Original run_72 | FretNet |
|--------|---------|-----------------|---------|
| TDR F1 | **0.8499** | 0.8569 | 0.727 |
| MPE F1 | **0.8683** | 0.8736 | 0.818 |

Our training achieved 99.2% of the original TDR F1 and 99.4% of the original MPE F1. The small gap is within normal variance for different training runs with data augmentation randomness.

---

## Validation Best Scores

| Metric | Score | Epoch |
|--------|-------|-------|
| Best val TDR F1 | 0.8227 | 86 |
| Best val MPE F1 | 0.8440 | 111 |

---

## CPU Inference Performance

| Platform | Time per 30s clip | Total for 36 test clips |
|----------|-------------------|------------------------|
| M3 Max (CPU) | ~1.2 seconds | 43 seconds |
| Hetzner VPS (estimated) | ~2-4 seconds | ~2 minutes |
| Modal A10G (estimated) | ~0.3-0.5 seconds | ~15 seconds |

---

## Files

| File | Path |
|------|------|
| Trained model | `backend/models/pretrained/trimplexx_guitar_model.pt` (78.9 MB) |
| Run config | `backend/models/pretrained/trimplexx_run_config.json` |
| Training log | `backend/models/pretrained/trimplexx_training_log.txt` |
| Test metrics | `backend/models/pretrained/trimplexx_test_metrics.json` |
| Training script | `train_tab_model/modal_train_trimplexx.py` |

---

## Key Findings

1. **TDR F1 = 0.85 achieved** -- model correctly identifies string+fret+onset for 85% of notes. This is far better than Basic Pitch + FretMapper heuristic (which had no string/fret awareness at all).

2. **Onset precision is very high (0.955)** -- when the model detects a note onset, it is almost always correct. Recall is lower (0.728), meaning some quieter notes are missed.

3. **CPU inference is viable** -- ~1.2 seconds per 30-second clip on M3 Max. Acceptable for background processing pipeline.

4. **Training is cheap and fast** -- $1.78 on Modal A10G. Reproducible with the same script.

5. **Model generalizes well within GuitarSet** -- test TDR F1 (0.85) is close to validation TDR F1 (0.82), suggesting minimal overfitting.

---

## Next Steps

1. **Wire into StemScriber** -- create `backend/trimplexx_transcriber.py` wrapper
2. **Add to guitar fallback chain** in `processing/transcription.py`
3. **A/B test** on "The Time Comes" demo song
4. **Bundle in Modal image** for production inference
5. **Consider fine-tuning** on distorted/electric guitar samples if available
