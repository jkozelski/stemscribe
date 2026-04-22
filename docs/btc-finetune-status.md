# BTC Fine-Tune Status

**Date:** 2026-04-04
**Phase:** 1 (Expand BTC Vocabulary via Fine-Tuning)

---

## Current Checkpoint Status

### btc_finetuned_best.pt (RESTORED)
- **Epoch:** 39
- **Val accuracy:** 79.92%
- **Val loss:** 0.776
- **Training data:** 988 songs (879 Billboard + 109 JAAH)
- **Status:** This is the best checkpoint from the first fine-tuning run. It was accidentally overwritten with original weights by `btc_retrain_augmented.py` (which loads from original and saves to the same path). **Restored from backup on 2026-04-04.**

### btc_finetuned_best_backup.pt
- Same as above (the source of the restore)

### btc_finetuned_best_pre_augment.pt
- **Epoch:** 1 (basically untrained)
- **Val accuracy:** 59.4%
- **Status:** Failed augmented retrain attempt. Only epoch 1 saved, immediately worse than original.

### btc_model_large_voca.pt (original pre-trained)
- **Val accuracy:** Unknown (no metadata), but published BTC paper reports ~80% on standard benchmarks
- **170 chord classes:** 12 roots x 14 qualities + N + X

---

## What the Restored Checkpoint Does Well

The epoch 39 checkpoint was trained with:
- Direct fine-tuning (no augmentation, no class weighting)
- 988 songs of Billboard (pop/rock) + JAAH (jazz)
- Standard cross-entropy loss

It likely improved on the original for the Billboard/JAAH distribution but may have overfitted to those genres.

---

## What Still Needs Improvement

1. **Pitch bias:** The non-augmented training likely biased the model toward common keys (C, G, D, A). Sharps/flats (F#, C#, Db, Ab) are underrepresented in Billboard.

2. **Extended chord accuracy:** The training data has solid extended chord coverage (347 unique chord types, including dim7, hdim7, maj6, min6, sus4, aug) but no special weighting was applied to boost these.

3. **The augmented retrain failed:** `btc_retrain_augmented.py` had the right idea (pitch augmentation + class weighting) but:
   - It overwrote the good checkpoint
   - Only saved epoch 1 (59.4%) as pre_augment backup
   - The "best" file ended up with original weights

---

## Training Data Inventory

| Source | Songs | Annotations | Extended Chords |
|--------|-------|-------------|-----------------|
| Billboard | 879 | ~120K | maj6, min6, sus4, dim7, hdim7, aug |
| JAAH Jazz | 109 | ~22K | maj7, min7, dim7, hdim7, aug (rich jazz harmony) |
| **Total** | **988** | **142K** | **347 unique chord types** |

Top extended chords by count: Bb:maj6 (490), C:sus4 (358), Db:maj6 (265), Eb:maj6 (234), E:dim7 (205), Ab:maj6 (202)

---

## Recommended Next Step: Modal GPU Training

### Why retrain?
The restored epoch 39 model is decent (79.9%) but was trained without augmentation or class weighting. A proper retrain should:
- Start from original pre-trained weights (not from the fine-tuned checkpoint, to avoid compounding any overfitting)
- Apply 12-semitone pitch augmentation (12x effective data diversity)
- Apply class-weighted loss with 1.5x boost for extended chord qualities
- Optionally freeze early transformer layers (layers 0-5) to prevent catastrophic forgetting

### Script ready
`~/stemscribe/btc_chord/modal_train_btc.py` is ready to go:

```bash
# 1. Upload training data to Modal volume
cd ~/stemscribe/btc_chord && ../venv311/bin/python modal_train_btc.py --upload

# 2. Run training on Modal A10G (~1-2 hours, ~$2-4)
cd ~/stemscribe/btc_chord && ../venv311/bin/python -m modal run modal_train_btc.py

# 3. Download trained checkpoint
cd ~/stemscribe/btc_chord && ../venv311/bin/python modal_train_btc.py --download
```

### Expected cost
- A10G at $0.000575/sec = ~$2.07/hr
- 60 epochs on 988 songs ~= 1-2 hours
- **Total: $2-4**

---

## Assessment: Skip Retrain or Do It?

**Recommendation: Do the retrain.** The restored 79.9% checkpoint is usable as a fallback, but:
1. It has no pitch augmentation -> biased toward common keys
2. It has no class weighting -> extended chords (the whole point of Phase 1) are underserved
3. The Modal training will cost ~$3 and take ~1 hour
4. The augmented approach should reach 82-85% with better extended chord accuracy

However, **Phase 2 (constrained decoding) will give bigger accuracy gains for known songs** since it reduces the output space from 170 to ~8 chords. Consider running Phase 1 and Phase 2 in parallel.

---

## Production Integration

`chord_detector_v10.py` already:
- Prefers `btc_finetuned_best.pt` over the original (line 685-687)
- Falls back to original if fine-tuned doesn't exist
- Has vocabulary-constrained decoding via `_build_vocab_mask()` (line 712-727)
- Validates scraped vocab against unconstrained BTC output (line 729+)

No production code changes needed -- just drop in a better checkpoint.
