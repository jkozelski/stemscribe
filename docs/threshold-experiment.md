# StemScriber v10 Chord Detection Threshold Experiment

**Date:** 2026-03-16
**Investigator:** Agent 4 (Threshold Fixer)

---

## Executive Summary

The v10 BTC chord detector had no confidence threshold to tune -- but the investigation uncovered a critical bug: **the LSTM inference path was destroying model confidence**. The v10 code was routing inference through an LSTM layer that is not part of the BTC model's standard forward pass, producing near-uniform softmax distributions (~1.3% per class instead of ~84%).

**FIX APPLIED:** Removed the LSTM from the inference path in `chord_detector_v10.py`. The standard path (self-attention -> output_projection) produces:
- **84.4% mean confidence** (was 1.3%)
- **75.3% mean gap** between #1 and #2 predictions (was 0.1%)
- **Correct chord predictions** (Farmhouse: C, G, F -- matches the actual song)

---

## The Bug

### What Was Happening (BEFORE fix)

The v10 code manually ran inference through three layers:
```python
encoder_output, _ = self._model.self_attn_layers(chunk)
lstm_out, _ = self._model.output_layer.lstm(encoder_output)      # <-- BUG
logits = self._model.output_layer.output_projection(lstm_out)
```

The LSTM layer exists in the `OutputLayer` base class but is **never used** in `SoftmaxOutputLayer.forward()`. The standard BTC inference path goes directly from self-attention output to the output projection:
```python
logits = self.output_projection(hidden)  # No LSTM
```

The LSTM was essentially a randomly-initialized (or poorly-fitted) passthrough that collapsed all logits to near-zero, making every chord class equally likely.

### Evidence

Comparison on Farmhouse (Phish) guitar stem, first 108 frames:

| Metric | Standard Path (FIXED) | LSTM Path (BROKEN) |
|--------|----------------------|-------------------|
| Mean max probability | 0.844 | 0.013 |
| Max probability | 0.981 | 0.017 |
| Mean gap #1 vs #2 | 0.753 | 0.001 |
| Logit std dev | 5.655 | 0.412 |
| Frame agreement with BTC forward() | 100% | 0% |

Sample predictions at Frame 20:
- Standard: **C (97.8%)** | G (0.7%) | Cmaj7 (0.5%)
- LSTM: **F#7 (1.2%)** | F (1.2%) | Fmaj7 (1.1%)

The standard path correctly identifies C major (Farmhouse is in C). The LSTM path returns garbage.

---

## Change Made

**File:** `/Users/jeffkozelski/stemscribe/backend/chord_detector_v10.py`

**Lines 424-428 changed from:**
```python
encoder_output, _ = self._model.self_attn_layers(chunk)
# Use LSTM path for better temporal modeling
lstm_out, _ = self._model.output_layer.lstm(encoder_output)
logits = self._model.output_layer.output_projection(lstm_out).squeeze(0)
```

**To:**
```python
encoder_output, _ = self._model.self_attn_layers(chunk)
# Standard BTC inference: self-attention output -> output_projection
# Note: The LSTM in OutputLayer is NOT used in the standard forward pass
# (SoftmaxOutputLayer.forward() only calls output_projection + softmax).
# Using the LSTM path produces near-uniform softmax (~1.3% per class)
# because it was never part of the trained inference pipeline.
logits = self._model.output_layer.output_projection(encoder_output).squeeze(0)
```

**Tests:** All 138 tests pass after the change.

---

## Original Investigation: Threshold Analysis

### No Confidence Threshold Exists

The v10 detector uses argmax on logits -- every frame gets a chord prediction. The only filters are:
- `min_duration` = 0.3s (removes chord events shorter than 0.3 seconds)
- `chord != 'N'` (removes "no chord" predictions)

### min_duration Effect (with the LSTM bug)

| min_duration | Farmhouse events | Zeppelin events | Dylan events |
|-------------|-----------------|----------------|-------------|
| 0.05 | 260 (25 unique) | 232 (22 unique) | 277 (26 unique) |
| 0.15 | 229 (20 unique) | 203 (21 unique) | 239 (25 unique) |
| 0.30 | 175 (17 unique) | 170 (21 unique) | 168 (16 unique) |
| 0.50 | 140 (13 unique) | 133 (17 unique) | 135 (14 unique) |

With the fix applied, the min_duration of 0.3s is reasonable -- the model now produces much more stable predictions with fewer spurious chord changes.

### Potential Future Improvement: Confidence-Based Filtering

Now that the model produces real confidence values (84% mean), a confidence threshold could actually be useful:
- Frames with confidence < 0.3 could be labeled as uncertain
- This would allow the UI to show confidence-based highlighting
- Not recommended as a hard filter yet -- better to ship the fix and evaluate

---

## Files

- **Fixed:** `/Users/jeffkozelski/stemscribe/backend/chord_detector_v10.py` (line 424-430)
- **Experiment scripts:** `/Users/jeffkozelski/stemscribe/scripts/threshold_experiment.py`, `/Users/jeffkozelski/stemscribe/scripts/threshold_compare_paths.py`
- **BTC model definition:** `/Users/jeffkozelski/stemscribe/btc_chord/btc_model.py`
- **SoftmaxOutputLayer:** `/Users/jeffkozelski/stemscribe/btc_chord/utils/transformer_modules.py` (line 70-84)
