# Bass Transcription Training Results
**Date:** 2026-04-04 (in progress)
**Script:** train_bass_model/modal_train_bass.py (v3 transfer learning)

## Configuration
- **GPU:** Modal A10G (24GB VRAM) -- NVIDIA A10
- **Model:** BassTranscriptionModel_v3 (piano CNN transfer learning)
- **Dataset:** Slakh2100-redux (bass stems, GM programs 32-39)
- **Train tracks:** ~1500+ (Slakh train split)
- **Val tracks:** ~300+ (Slakh validation + test splits)
- **Epochs:** 40 (3-phase: adapter warmup -> full -> CNN fine-tune)
- **Batch size:** 16
- **Pitch range:** E1 (MIDI 28) to G4 (MIDI 67) = 40 keys
- **Spectrogram:** Mel (sr=16000, hop=256, n_mels=229, matching piano model)

## Training Infrastructure
- **Data download:** aria2c with 16 parallel connections from Zenodo
- **Download time:** 43 minutes (104.3 GB archive, ~37 MiB/s)
- **Data extraction:** ~30 min
- **Epoch duration:** ~30 min (685 batches x ~2.5s/batch)
- **Estimated total:** ~20-22 hours for 40 epochs
- **Volume persistence:** bass-training-data (Slakh bass stems), bass-training-results (checkpoints)

## Early Results (Epoch 1)
| Epoch | Phase | Train Loss | Val Loss | Val F1 | Val P | Val R |
|-------|-------|-----------|---------|--------|-------|-------|
| 1 | Phase1 | 0.3688 | 0.2189 | 0.5763 | 0.4436 | 0.8222 |

## Status: IN PROGRESS
Training is running on Modal. Monitor with:
```bash
tail -f ~/stemscribe/train_bass_model/modal_training.log
```

Or check Modal dashboard for the `bass-transcription-training` app.

When complete, the script will automatically:
1. Save best_bass_model.pt to ~/stemscribe/backend/models/pretrained/best_bass_model.pt
2. Update this file with final results

## Architecture Details
- Piano CNN (frozen in Phase 1-2, top blocks unfrozen in Phase 3)
- Domain adapter: Linear(1792->1024) + LayerNorm + GELU + residual
- Onset BiLSTM(1024, 128) -> 40 keys
- Frame BiLSTM(1024+40, 128) -> 40 keys (onset-conditioned)
- Velocity head: Linear(256, 40) + Sigmoid
- Loss: Focal BCE + Soft Dice (onset), BCE (frame), MSE (velocity)
- Scheduler: Cosine warmup (3 epochs)
- Phase 1 (ep 1-8): adapter warmup
- Phase 2 (ep 9-30): full training
- Phase 3 (ep 31-40): CNN fine-tuning (if F1 > 0.3)

## Notes
- Slakh2100-redux bass data is cached on Modal volume `bass-training-data` for future reruns
- Piano checkpoint cached on same volume (145 MB)
- Zenodo download was throttled with wget (229 KB/s) but aria2c with 16 connections achieved 37 MiB/s
- Checkpoints committed to results volume every 5 epochs for crash recovery
