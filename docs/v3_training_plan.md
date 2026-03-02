# StemScribe v3 Training Plan: Guitar & Bass Transcription via Piano Transfer Learning

**Author:** ML Architect (Claude)
**Date:** 2026-02-27
**Status:** PROPOSAL — requires review before implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Failure Analysis: Why v1/v2 Failed](#2-failure-analysis)
3. [Architecture Comparison: Piano vs Guitar/Bass](#3-architecture-comparison)
4. [v3 Architecture Design](#4-v3-architecture-design)
5. [Transfer Learning Strategy](#5-transfer-learning-strategy)
6. [Loss Function Design](#6-loss-function-design)
7. [Data Strategy](#7-data-strategy)
8. [Curriculum Learning Plan](#8-curriculum-learning-plan)
9. [Training Pipeline](#9-training-pipeline)
10. [GPU Requirements & Timeline](#10-gpu-requirements--timeline)
11. [Evaluation Protocol](#11-evaluation-protocol)
12. [Risk Mitigation](#12-risk-mitigation)

---

## 1. Executive Summary

The piano and drum CRNNs work well. Guitar tab v1/v2 and bass v1 collapsed to all-zeros
or produced garbage (97% recall, 0.8% precision). The root causes are:

1. **Extreme class imbalance** (~0.03% positive onsets) combined with naive loss functions
2. **Insufficient data** (288 GuitarSet samples for 120-output-neuron model)
3. **Wrong spectrogram representation** (CQT without harmonic awareness)
4. **No pre-trained feature extractor** (training CNN from scratch on tiny data)
5. **Output dimensionality mismatch** (6x20=120 outputs vs piano's 88 flat outputs)

v3 solves all five via transfer learning from the piano CRNN, a pitch-first architecture
(predict MIDI pitch, then assign string/fret), better loss functions, multi-dataset
training, and curriculum learning.

**Key insight:** Guitar and bass produce pitched notes in overlapping frequency ranges
with piano. The piano CNN already knows how to extract pitched features from mel
spectrograms. We should reuse that knowledge rather than training from scratch on
tiny datasets.

---

## 2. Failure Analysis

### 2.1 Guitar Tab v1 (BCELoss Collapse)

- **Architecture:** CQT(84 bins) -> 3 CNN blocks (1->32->64->128) -> BiLSTM(1280->256x2) -> onset/frame heads -> 6x20=120 outputs
- **Loss:** `nn.BCELoss()` (unweighted)
- **Data:** GuitarSet (360 WAV files -> ~288 matched samples -> 230 train / 58 val)
- **Result:** Model predicted all zeros. Val loss ~0.05 looked "good" but onset recall = 0.0
- **Root cause:** With 0.03% positive rate, predicting all-zeros achieves 99.97% accuracy.
  Unweighted BCE has no incentive to find the rare positives.

### 2.2 Guitar Tab v2 (Focal+Dice Overcorrection)

- **Architecture:** Same as v1, but returns logits (no sigmoid)
- **Loss:** `FocalBCEWithLogitsLoss(pos_weight=500, gamma=2) + DiceLoss(weight=0.5)`
- **Data:** Same GuitarSet (288 samples)
- **Result:** 97% recall but 0.8% precision — model predicted everything as onset
- **Root cause:** pos_weight=500 on 0.03% positive rate means the effective weight
  ratio is 500:1 for positives vs negatives. Combined with Focal Loss gamma=2 that
  further suppresses easy negatives, the model learned to over-predict. Also, 288
  samples is far too few for a 120-output model with 5.7M parameters.

### 2.3 Bass v1 (BCELoss Collapse)

- **Architecture:** CQT(84 bins) -> 3 CNN blocks -> BiLSTM -> onset/frame/velocity -> 4x24=96 outputs
- **Loss:** `nn.BCELoss()` (unweighted, same mistake as guitar v1)
- **Data:** Slakh2100 bass stems (~1000+ tracks, much more data)
- **Result:** Collapsed to all-zeros after 11 epochs, same as guitar v1
- **Root cause:** Same unweighted BCE problem. The Slakh data is better but the loss
  function kills training before the model can learn.

### 2.4 Bass v2 (Not Yet Run)

- **Architecture:** Same as v1 but returns logits
- **Loss:** FocalBCE(pw=50, gamma=2) + weighted BCE(pw=10)
- **Status:** Script written but not executed on RunPod yet

### 2.5 Common Problems Across All Failed Attempts

| Problem | Guitar v1 | Guitar v2 | Bass v1 |
|---------|-----------|-----------|---------|
| Unweighted loss on sparse targets | Yes | Fixed | Yes |
| Over-aggressive pos_weight | No | Yes (500x) | No |
| Insufficient training data | Yes (288) | Yes (288) | No (~1000) |
| CQT instead of Mel spectrogram | Yes | Yes | Yes |
| No transfer learning | Yes | Yes | Yes |
| 6x20 / 4x24 output structure | Yes | Yes | Yes |
| No curriculum learning | Yes | Yes | Yes |

---

## 3. Architecture Comparison

### 3.1 Working Piano Model (145 MB, 50 epochs MAESTRO v3)

```
Input:  Mel spectrogram (229 bins, fmin=30, fmax=8000, sr=16000, hop=256)

CNN:    Conv2d(1->48, 3x3) + BN + ReLU
        Conv2d(48->48, 3x3) + BN + ReLU + MaxPool(2,1) + Dropout(0.25)
        Conv2d(48->64, 3x3) + BN + ReLU + MaxPool(2,1) + Dropout(0.25)
        Conv2d(64->96, 3x3) + BN + ReLU + MaxPool(2,1) + Dropout(0.25)
        Conv2d(96->128, 3x3) + BN + ReLU + MaxPool(2,1) + Dropout(0.25)
        Output: (batch, 128, 14, time) -> flatten to 1792 per timestep

LSTM:   onset_lstm: BiLSTM(1792, 256, 2 layers) -> 512-dim per timestep
        frame_lstm: BiLSTM(1792+88, 256, 2 layers) -> 512-dim (onset-conditioned)

Heads:  onset_head: Linear(512 -> 88)
        frame_head: Linear(512 -> 88)
        velocity_head: Linear(512 -> 88) + Sigmoid

Architecture: Onset-and-Frames (OaF) with onset conditioning for frames
```

**Why it works:**
- 229 mel bins provide high frequency resolution for pitched content
- 4 pooling stages reduce 229 -> 14 frequency bins (good compression)
- OaF onset conditioning prevents frame predictions from drifting
- MAESTRO v3 has 198 hours of perfectly aligned audio + MIDI
- 88-key flat output (no string/fret decomposition)
- Trained for 50 epochs with proper BCE loss (piano has ~5-10% positive rate
  in active regions, not 0.03%)

### 3.2 Working Drum Model (best_drum_model.pt, 50 epochs E-GMD)

```
Input:  Mel spectrogram (128 bins, sr=16000, hop=256)

CNN:    3 blocks: 1->32->64->128, pool 128->32 freq
        Output: 128 * 32 = 4096 per timestep

LSTM:   onset_lstm: BiLSTM(4096, 128, 2 layers) -> 256-dim
        frame_lstm: BiLSTM(4096+8, 128, 2 layers) -> 256-dim

Heads:  onset/frame/velocity -> 8 classes
```

**Why it works:**
- Only 8 output classes (not 120)
- E-GMD has 444 hours of data
- Drums have clear transients (easier onset detection)
- Class balance is better (~1-3% positive rate)

### 3.3 Failed Guitar/Bass Models

```
Input:  CQT (84 bins, 7 octaves x 12 bins/octave, sr=22050, hop=256)

CNN:    3 blocks: 1->32->64->128, pool 84->10 freq
        Output: 128 * 10 = 1280 per timestep

LSTM:   BiLSTM(1280, 256, 2 layers) -> 512-dim (guitar)
        OR onset/frame split LSTM (bass v1)

Heads:  onset/frame -> 6 x 20 = 120 (guitar)
        onset/frame/velocity -> 4 x 24 = 96 (bass)
```

**Why they fail:**
- CQT has 84 bins vs piano's 229 mel bins — less frequency resolution
- 1280-dim CNN output vs piano's 1792 — less capacity
- 120/96 multi-dimensional outputs vs 88 flat — harder learning problem
- String/fret decomposition adds ambiguity (same note on multiple string/fret combos)
- Training data: 288 samples (guitar) or ~1000 (bass) vs 198+ hours (piano/drums)

---

## 4. v3 Architecture Design

### 4.1 Core Idea: Pitch-First, Then String/Fret

Instead of predicting string/fret directly from audio (a 120-output problem with
ambiguous targets), v3 uses a **two-stage approach**:

**Stage 1 — Pitch Transcription (transfer from piano):**
Predict which MIDI pitches are active at each frame. This is the same task the piano
model already solves. Guitar range (E2=40 to E6=88) is a subset of piano range
(A0=21 to C8=108).

**Stage 2 — String/Fret Assignment (lightweight head or post-processing):**
Given predicted pitches, assign them to strings and frets. This can be:
- (a) A learned head that considers context (nearby notes, hand position)
- (b) The existing `_assign_string_fret()` heuristic from guitar_tab_transcriber.py

Stage 2(b) is already implemented and working in the current codebase. The bottleneck
has always been Stage 1 (accurate pitch detection), which is exactly what we can
transfer from the piano model.

### 4.2 Model Architecture: GuitarTranscriptionModel_v3

```python
class GuitarTranscriptionModel_v3(nn.Module):
    """
    Transfer-learned guitar transcriber.

    Piano CNN (frozen) -> Adapter -> Guitar LSTM -> Pitch heads (49 keys)

    Input:  Mel spectrogram (229 bins, fmin=30, fmax=8000)  # SAME as piano
    Output: onset    (batch, 49, time)  — E2(40) to E6(88)
            frame    (batch, 49, time)
            velocity (batch, 49, time)
    """

    def __init__(self, num_keys=49, hidden_size=192, num_layers=2, dropout=0.3):
        super().__init__()
        self.num_keys = num_keys

        # === FROZEN: Piano CNN feature extractor ===
        # Loaded from best_piano_model.pt, all gradients disabled
        # Input: (batch, 1, 229, time) -> Output: (batch, 128, 14, time)
        self.cnn = <load_piano_cnn_frozen>()  # 4 blocks, ~1.2M params frozen

        # === TRAINABLE: Domain adaptation layer ===
        # Adapts piano CNN features for guitar timbre
        # 1792 -> 1024 with residual connection
        self.adapter = nn.Sequential(
            nn.Linear(1792, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.adapter_residual = nn.Linear(1792, 1024)  # Skip connection

        # === TRAINABLE: Multi-head self-attention ===
        # Captures harmonic relationships between frequency bins
        self.attention = nn.MultiheadAttention(
            embed_dim=1024, num_heads=8, dropout=dropout, batch_first=True
        )
        self.attn_norm = nn.LayerNorm(1024)

        # === TRAINABLE: Onset and Frame LSTMs ===
        self.onset_lstm = nn.LSTM(
            input_size=1024, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout
        )

        self.frame_lstm = nn.LSTM(
            input_size=1024 + num_keys, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout
        )

        lstm_dim = hidden_size * 2  # 384

        # === TRAINABLE: Output heads ===
        self.onset_head = nn.Linear(lstm_dim, num_keys)
        self.frame_head = nn.Linear(lstm_dim, num_keys)
        self.velocity_head = nn.Sequential(
            nn.Linear(lstm_dim, num_keys),
            nn.Sigmoid()
        )

        # Initialize onset head bias for sparse targets
        nn.init.constant_(self.onset_head.bias, -3.0)  # sigmoid(-3) ~ 0.05

    def forward(self, mel):
        batch, n_mels, time = mel.shape

        # Frozen CNN features
        with torch.no_grad():
            cnn_out = self.cnn(mel.unsqueeze(1))  # (B, 128, 14, T)

        # Flatten CNN output
        cnn_features = cnn_out.permute(0, 3, 1, 2).reshape(batch, time, -1)  # (B, T, 1792)

        # Domain adapter with residual
        adapted = self.adapter(cnn_features) + self.adapter_residual(cnn_features)  # (B, T, 1024)

        # Self-attention for harmonic context
        attn_out, _ = self.attention(adapted, adapted, adapted)
        adapted = self.attn_norm(adapted + attn_out)  # (B, T, 1024)

        # Onset prediction
        onset_out, _ = self.onset_lstm(adapted)  # (B, T, 384)
        onset_logits = self.onset_head(onset_out)
        onset_pred = torch.sigmoid(onset_logits)

        # Frame prediction (onset-conditioned)
        frame_input = torch.cat([adapted, onset_pred.detach()], dim=-1)
        frame_out, _ = self.frame_lstm(frame_input)
        frame_logits = self.frame_head(frame_out)

        # Velocity
        velocity_pred = self.velocity_head(onset_out)

        return (
            onset_pred.permute(0, 2, 1),    # (B, 49, T)
            torch.sigmoid(frame_logits).permute(0, 2, 1),
            velocity_pred.permute(0, 2, 1),
        )
```

### 4.3 Bass Model: BassTranscriptionModel_v3

Same architecture with these differences:
- `num_keys=40` — E1(28) to G4(67), covering the full bass range
- Smaller LSTM: `hidden_size=128` (bass is more monophonic)
- No attention layer (bass has simpler harmonic structure)
- Lower dropout (more data available from Slakh)

### 4.4 Parameter Counts

| Component | Guitar v3 | Bass v3 | Piano (reference) |
|-----------|-----------|---------|-------------------|
| CNN (frozen) | 1.2M (0 trainable) | 1.2M (0 trainable) | 1.2M |
| Adapter | 3.9M | 3.9M | — |
| Attention | 4.2M | — | — |
| LSTMs | 5.1M | 2.6M | 9.2M |
| Heads | 0.06M | 0.03M | 0.13M |
| **Total trainable** | **13.3M** | **6.5M** | **~15M** |
| **Total params** | **14.5M** | **7.7M** | **~15M** |

Compare to failed guitar v2: 5.7M (all trainable, no frozen features) on 288 samples.
v3 has more capacity but the frozen CNN provides a massive head start.

---

## 5. Transfer Learning Strategy

### 5.1 Which Layers to Transfer

From `best_piano_model.pt` (145 MB checkpoint):

| Piano Layer | Transfer To | Freeze? | Rationale |
|------------|-------------|---------|-----------|
| `cnn` (4 Conv2d blocks) | `cnn` | **Yes (frozen)** | Mel spectrogram features are instrument-agnostic for pitch detection. Low-level features (harmonics, onset transients) are shared across piano/guitar/bass. |
| `onset_lstm` | — | **No (discard)** | Piano LSTM is tuned for 88-key output width and piano-specific temporal patterns (sustain pedal, etc.). Guitar/bass need different temporal modeling. |
| `frame_lstm` | — | **No (discard)** | Same reasoning — piano frame patterns differ from guitar. |
| `onset_head` / `frame_head` / `velocity_head` | — | **No (discard)** | Output dimensions differ (88 vs 49/40). |

### 5.2 Transfer Procedure

```python
def load_piano_cnn_for_transfer(piano_checkpoint_path):
    """Extract and freeze the CNN from the piano model."""
    checkpoint = torch.load(piano_checkpoint_path, map_location='cpu', weights_only=False)
    piano_state = checkpoint['model_state_dict']

    # Build a fresh piano model to get the CNN architecture
    piano_model = PianoTranscriptionModel()
    piano_model.load_state_dict(piano_state)

    # Extract CNN and freeze all parameters
    cnn = piano_model.cnn
    for param in cnn.parameters():
        param.requires_grad = False

    return cnn
```

### 5.3 Three-Phase Fine-Tuning Schedule

**Phase 1 — Adapter warmup (epochs 1-10):**
- CNN: frozen
- Adapter + Attention + LSTMs + Heads: trainable
- LR: 3e-4 with linear warmup over 3 epochs
- Purpose: Let the adapter learn to translate piano features to guitar/bass domain

**Phase 2 — Full training (epochs 11-40):**
- CNN: frozen
- All other layers: trainable
- LR: 1e-4, cosine annealing to 1e-6
- Purpose: Main training phase, LSTM learns guitar/bass temporal patterns

**Phase 3 — CNN fine-tuning (epochs 41-60):**
- CNN blocks 3-4 (last 2 blocks): **unfrozen**, LR=1e-5 (10x lower than other layers)
- CNN blocks 1-2: still frozen (these learn universal audio features)
- All other layers: trainable, LR=1e-5
- Purpose: Fine-tune high-level CNN features for guitar/bass timbre
- **Only if** Phase 2 achieves onset F1 > 0.3 (otherwise more data/architecture work needed)

### 5.4 Why Transfer from Piano Specifically

1. **Frequency overlap:** Guitar E2-E6 (82-1319 Hz) and bass E1-G4 (41-392 Hz) overlap
   significantly with piano A0-C8. The piano CNN has already learned to separate pitched
   content across this entire range.

2. **Same spectrogram:** Piano uses 229-bin mel (fmin=30, fmax=8000). Guitar and bass
   fundamentals fall entirely within this range. No spectrogram reconfiguration needed.

3. **Onset detection transfer:** Piano onset detection (hammer strike) is acoustically
   similar to guitar picking and bass plucking — sharp transients followed by decay.

4. **Proven architecture:** The OaF (Onset and Frames) pattern with onset-conditioned
   frame prediction works for piano. The same principle applies to guitar/bass.

---

## 6. Loss Function Design

### 6.1 The Sparsity Problem (Why v1/v2 Failed)

For a 5-second guitar chunk at 62.5 fps with 49 pitch outputs:
- Total elements: 49 * 312 = 15,288
- Typical active onsets: ~8-15 per chunk
- Positive rate: ~0.05-0.1% for onsets, ~1-3% for frames

This is the same order of magnitude as the v1/v2 attempts. The difference is that
49 outputs (pitch-only) has much better positive density than 120 outputs (string x fret).

**Improvement from architecture change alone:**
- v2 guitar: 6 x 20 x 312 = 37,440 elements, ~8 positive onsets -> 0.02% positive
- v3 guitar: 49 x 312 = 15,288 elements, ~8 positive onsets -> 0.05% positive
- That's 2.5x better positive rate just from removing string/fret decomposition

### 6.2 v3 Loss Function: Asymmetric Focal + Soft Dice

```python
class v3OnsetLoss(nn.Module):
    """
    Three-component onset loss designed for sparse pitch detection.

    1. Focal BCE (pos_weight=20, gamma=2): Handles class imbalance without
       the extreme overcorrection of pos_weight=500
    2. Soft Dice: Class-imbalance invariant, focuses on overlap quality
    3. Temporal smoothing penalty: Penalizes isolated frame predictions
    """

    def __init__(self, pos_weight=20.0, gamma=2.0, dice_weight=0.3):
        super().__init__()
        self.focal = FocalBCEWithLogitsLoss(pos_weight=pos_weight, gamma=gamma)
        self.dice = SoftDiceLoss()
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal = self.focal(logits, targets)
        dice = self.dice(logits, targets)
        return (1 - self.dice_weight) * focal + self.dice_weight * dice
```

**Why pos_weight=20 instead of 500:**
- v2 used pos_weight=500 on 0.03% positives -> effective ratio 500 * (0.9997/0.0003) ~ 1.6M:1
- v3 uses pos_weight=20 on 0.05% positives -> effective ratio 20 * (0.9995/0.0005) ~ 40,000:1
- This is aggressive enough to prevent collapse but not so extreme that it floods predictions

### 6.3 Frame Loss

```python
frame_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
```

Frames have ~1-3% positive rate (much better than onsets), so modest pos_weight=5 suffices.

### 6.4 Velocity Loss

```python
# MSE only at onset positions (same as working bass v2 approach)
onset_mask = onset_target > 0.5
if onset_mask.any():
    vel_loss = F.mse_loss(vel_pred[onset_mask], vel_target[onset_mask])
```

### 6.5 Total Loss

```python
total_loss = onset_loss + frame_loss + 0.5 * velocity_loss
```

### 6.6 Collapse Detection (Improved)

```python
# Track running precision/recall over last 5 epochs
# If recall < 0.02 AND precision < 0.02 for 5 consecutive epochs: collapsed
# If recall > 0.5 AND precision < 0.01 for 5 consecutive epochs: over-predicting

# Auto-adjust: if collapse detected, halve pos_weight and restart from last good checkpoint
# Auto-adjust: if over-predicting, double pos_weight and restart
```

---

## 7. Data Strategy

### 7.1 Guitar Datasets

| Dataset | Size | Audio | Labels | Quality |
|---------|------|-------|--------|---------|
| **GuitarSet** | 360 tracks (~3 hours) | Mono mic + hex pickup | JAMS (per-string MIDI) | High (real recordings) |
| **IDMT-SMT-Guitar** | 4700 samples | Isolated notes + licks | XML (pitch + onset) | Medium (controlled) |
| **GuitarPro Synthesized** | Unlimited | Synthesized from GP tabs | Perfect alignment | Low (synthetic) |
| **MAESTRO subset** | ~20 hours | Piano (for pre-training) | Already done | N/A |
| **MusicNet guitar** | ~12 hours | Guitar chamber music | Note-level | Medium |

**v3 training data plan:**
1. **Primary:** GuitarSet (360 tracks) — real audio with perfect labels
2. **Augmentation:** IDMT-SMT-Guitar (4700 samples) — expand training set
3. **Synthetic pre-training:** Generate guitar audio from GuitarPro MIDI using
   FluidSynth + diverse SoundFont guitars — gives us 10,000+ training samples
4. **Pitch-shifting augmentation:** Shift each sample by -2 to +2 semitones
   (5x data expansion with minimal distortion)

**Effective training set size:** ~15,000-20,000 samples (vs 288 in v1/v2)

### 7.2 Bass Datasets

| Dataset | Size | Audio | Labels | Quality |
|---------|------|-------|--------|---------|
| **Slakh2100-redux** | ~1000+ bass stems | Synthesized | Perfect MIDI alignment | Medium (synthetic) |
| **IDMT-SMT-Bass** | 2600+ samples | Real bass recordings | Pitch annotations | High |
| **GuitarPro Synthesized** | Unlimited | Synthesized from GP tabs | Perfect alignment | Low |

**v3 bass training data plan:**
1. **Primary:** Slakh2100 bass stems (~1000 tracks, ~50+ hours)
2. **Augmentation:** IDMT-SMT-Bass (2600 samples)
3. **Synthetic expansion:** FluidSynth bass SoundFonts from MIDI

Bass already has adequate data volume from Slakh. The main issue was the loss function.

### 7.3 Spectrogram Matching

**Critical:** v3 uses the EXACT same mel spectrogram as the piano model:

```python
SAMPLE_RATE = 16000   # Not 22050!
HOP_LENGTH = 256
N_MELS = 229
N_FFT = 2048
FMIN = 30.0
FMAX = 8000.0
```

Previous guitar/bass attempts used CQT at 22050 Hz with 84 bins. This is incompatible
with the piano CNN. v3 switches to mel spectrograms at 16000 Hz to reuse the frozen CNN
features directly.

**This is non-negotiable for transfer learning to work.** The piano CNN expects
(batch, 1, 229, time) mel input. Any deviation will produce garbage features.

### 7.4 Data Augmentation Pipeline

```python
class GuitarAugmentation:
    """Applied to raw audio before spectrogram computation."""

    def __call__(self, audio, onset_targets, frame_targets):
        # 1. Gain jitter (0.7x to 1.3x) — 50% probability
        # 2. Additive noise (SNR 30-50 dB) — 30% probability
        # 3. Pitch shift (-2 to +2 semitones) — 40% probability
        #    IMPORTANT: must also shift target MIDI note numbers!
        # 4. Time stretch (0.9x to 1.1x) — 20% probability
        #    IMPORTANT: must also stretch onset/frame targets
        # 5. Random EQ (gentle low/mid/high boost/cut) — 30% probability
        # 6. Reverb (small room convolution) — 20% probability
```

---

## 8. Curriculum Learning Plan

### 8.1 Stage A — Monophonic (Epochs 1-15)

**Goal:** Get the model reliably detecting single notes before tackling polyphony.

- **Data filter:** Only include chunks where max polyphony <= 2
- **Output masking:** During loss computation, only penalize the top-1 prediction
  per frame (argmax over pitch dimension)
- **Expected outcome:** Onset F1 > 0.5 on monophonic segments

**Rationale:** v2's failure was partly because the model had to simultaneously learn
onset detection, pitch identification, AND polyphonic reasoning. Monophonic training
lets it focus on onset+pitch first.

### 8.2 Stage B — Low Polyphony (Epochs 16-35)

**Goal:** Handle 2-3 simultaneous notes (power chords, double stops).

- **Data filter:** Chunks with max polyphony <= 4
- **Loss:** Full multi-label loss on all outputs
- **Expected outcome:** Onset F1 > 0.4 on polyphonic segments

### 8.3 Stage C — Full Polyphony (Epochs 36-60)

**Goal:** Handle full 6-string strumming and complex chords.

- **Data filter:** All data, including full chords
- **Additional loss:** Chord consistency penalty (all notes in a chord should have
  onsets at the same frame +/- 1)
- **Expected outcome:** Onset F1 > 0.35 overall (benchmark: Basic Pitch gets ~0.3
  on guitar audio)

### 8.4 Bass Curriculum (Simpler)

Bass is predominantly monophonic, so curriculum is simpler:
- **Stage A (1-10):** Monophonic only (most bass lines)
- **Stage B (11-40):** Include double stops and chords
- **No Stage C needed** — bass rarely exceeds 2 simultaneous notes

---

## 9. Training Pipeline

### 9.1 Guitar Training Script Outline

```python
# train_guitar_v3.py

# 1. Load piano checkpoint and extract frozen CNN
piano_cnn = load_piano_cnn_frozen('best_piano_model.pt')

# 2. Build v3 model with piano CNN
model = GuitarTranscriptionModel_v3(num_keys=49)
model.cnn = piano_cnn  # Frozen, no gradients

# 3. Optimizer: different LR for different parameter groups
optimizer = torch.optim.AdamW([
    {'params': model.adapter.parameters(), 'lr': 3e-4},
    {'params': model.attention.parameters(), 'lr': 3e-4},
    {'params': model.onset_lstm.parameters(), 'lr': 1e-4},
    {'params': model.frame_lstm.parameters(), 'lr': 1e-4},
    {'params': model.onset_head.parameters(), 'lr': 3e-4},
    {'params': model.frame_head.parameters(), 'lr': 3e-4},
    {'params': model.velocity_head.parameters(), 'lr': 3e-4},
], weight_decay=1e-4)

# 4. Scheduler: cosine annealing with warmup
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

# 5. Loss functions
onset_criterion = v3OnsetLoss(pos_weight=20, gamma=2, dice_weight=0.3)
frame_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5.0]))
velocity_criterion = nn.MSELoss()

# 6. Training loop with curriculum stages
for epoch in range(60):
    if epoch < 15:
        train_monophonic(model, ...)     # Stage A
    elif epoch < 35:
        train_low_polyphony(model, ...)  # Stage B
    else:
        train_full(model, ...)           # Stage C

    # Evaluate on full validation set every epoch
    val_f1, val_prec, val_rec = evaluate(model, val_loader)

    # Collapse detection and auto-recovery
    check_collapse(val_prec, val_rec, ...)

    # Phase 3: unfreeze CNN blocks 3-4 after epoch 40
    if epoch == 40 and best_val_f1 > 0.3:
        unfreeze_cnn_top_blocks(model)
```

### 9.2 Key Differences from v1/v2

| Aspect | v1/v2 | v3 |
|--------|-------|-----|
| Spectrogram | CQT 84 bins @ 22050 Hz | Mel 229 bins @ 16000 Hz |
| CNN | Random init, 3 blocks | Piano pretrained, 4 blocks, frozen |
| Output | 6x20=120 (string/fret) | 49 (pitch only) |
| String/fret | End-to-end | Post-processing (existing code) |
| LSTM | Single shared | Separate onset/frame (OaF) |
| Attention | None | Multi-head self-attention |
| Loss | BCE / Focal(pw=500) | Focal(pw=20)+Dice |
| Data | 288 samples | 15,000+ samples |
| Curriculum | None | Mono -> Low poly -> Full |
| Transfer learning | None | Piano CNN frozen |

---

## 10. GPU Requirements & Timeline

### 10.1 Memory Estimates

| Component | Memory (float32) |
|-----------|-----------------|
| Model parameters (14.5M) | 55 MB |
| Gradients (trainable only, 13.3M) | 51 MB |
| Optimizer states (Adam, 2x) | 102 MB |
| Activations (batch=16, 5s chunks) | ~2 GB |
| Mel spectrogram cache | ~500 MB |
| **Total per GPU** | **~3 GB** |

Any GPU with 8+ GB VRAM will work. A100/V100 are ideal but even a T4 (16 GB) is fine.

### 10.2 Training Time Estimates

| GPU | Guitar (60 epochs, 15K samples) | Bass (40 epochs, 8K samples) |
|-----|--------------------------------|------------------------------|
| A100 (80 GB) | ~4-6 hours | ~2-3 hours |
| V100 (16 GB) | ~8-12 hours | ~4-6 hours |
| T4 (16 GB) | ~16-24 hours | ~8-12 hours |
| RTX 3090 | ~6-10 hours | ~3-5 hours |

### 10.3 Recommended Compute

**RunPod:**
- GPU Pod with 1x A100 (cheapest per-hour for this workload)
- ~$2/hour -> Guitar: ~$10, Bass: ~$6
- Total budget: ~$20-30 for full training including reruns

**Alternative:** Google Colab Pro with A100 runtime (~$10/month)

### 10.4 Phased Timeline

| Phase | Duration | What |
|-------|----------|------|
| Week 1 | 2-3 days | Implement v3 architecture, data pipeline, loss functions |
| Week 1 | 1 day | Download datasets, generate synthetic data |
| Week 1-2 | 1-2 days | Guitar v3 training (Phases 1-3 on A100) |
| Week 2 | 1 day | Evaluate guitar v3, tune thresholds |
| Week 2 | 1 day | Bass v3 training (A100) |
| Week 2 | 1 day | Integration testing with StemScribe backend |
| **Total** | **~8-10 days** | |

---

## 11. Evaluation Protocol

### 11.1 Metrics

**Primary (must improve over Basic Pitch baseline):**
- **Onset F1** @ 50ms tolerance (mir_eval note_onset)
- **Note F1** @ 50ms onset + 50ms offset tolerance (mir_eval note)
- **Frame-level F1** (per-frame pitch activation accuracy)

**Secondary:**
- **Precision/Recall** breakdown (detect both collapse AND over-prediction)
- **Polyphony accuracy** (correct number of simultaneous notes)
- **Pitch accuracy** (% of detected notes with correct MIDI pitch)

### 11.2 Baselines to Beat

| Method | Guitar Onset F1 | Bass Onset F1 | Source |
|--------|----------------|---------------|--------|
| Basic Pitch (current) | ~0.30-0.40 | ~0.35-0.45 | Our inference code |
| Guitar v2 (failed) | 0.016 | — | Over-predicting |
| Piano model on guitar | ~0.20-0.25 | ~0.15-0.20 | Expected (untrained) |
| **v3 target** | **>0.50** | **>0.55** | Goal |
| State-of-art (TabCNN) | ~0.65-0.70 | — | Literature |

### 11.3 Test Protocol

1. **GuitarSet held-out test split** (20% of 360 tracks = 72 tracks)
   - Per-track F1, averaged
   - Report by playing style (comp, solo, fingerpicking)

2. **Real-world test** on StemScribe user audio (5-10 diverse songs)
   - Qualitative: does the MIDI sound right?
   - Guitar Pro export: are the tabs playable?

3. **A/B comparison** vs Basic Pitch on same audio
   - Side-by-side MIDI visualization
   - Note count comparison
   - Listening test (MIDI playback)

---

## 12. Risk Mitigation

### 12.1 Risk: Transfer doesn't help (piano features too different)

**Mitigation:** Run a quick probe experiment first. Take the frozen piano CNN, attach
a single linear layer, and train on GuitarSet for 5 epochs. If the linear probe
achieves onset F1 > 0.1, the CNN features contain useful information for guitar.
If F1 stays at 0 even with the linear probe, the features don't transfer and we
need a different approach (e.g., pre-train on a larger multi-instrument dataset).

**Estimated time for probe:** 30 minutes on any GPU.

### 12.2 Risk: Still collapses despite better loss

**Mitigation:**
1. Start with pos_weight=20 and monitor precision/recall every epoch
2. Implement auto-adjust: if recall < 0.02 for 3 epochs, increase pos_weight by 50%
3. If precision < 0.01 for 3 epochs, decrease pos_weight by 50%
4. Fall back to class-balanced sampling (oversample positive frames)

### 12.3 Risk: 288 GuitarSet samples still insufficient

**Mitigation:**
1. Synthetic data from GuitarPro MIDI + FluidSynth (10,000+ samples)
2. IDMT-SMT-Guitar (4700 additional samples)
3. Cross-instrument pre-training: train on Slakh2100 guitar+bass+keys stems first
   (thousands of tracks), then fine-tune on GuitarSet

### 12.4 Risk: Mel spectrogram loses guitar-specific frequency info vs CQT

**Mitigation:** The piano mel spectrogram (229 bins, fmin=30, fmax=8000) actually has
better frequency resolution than the CQT (84 bins) used in v1/v2. And since 16 kHz
sample rate captures fundamentals up to 8 kHz, it covers guitar's entire fundamental
range. Harmonics above 8 kHz are not essential for pitch detection.

If mel proves insufficient, we can add a parallel CQT branch that feeds into the
adapter alongside the frozen CNN features. But try mel-only first.

### 12.5 Risk: Post-processing string/fret assignment produces unplayable tabs

**Mitigation:** The existing `_assign_string_fret()` heuristic in
`guitar_tab_transcriber.py` already handles this with playability scoring. If pitch
prediction is accurate, the heuristic will work. For further improvement, we can train
a small learned string/fret head as a Phase 2 project, using GuitarSet's per-string
annotations as supervision.

---

## Appendix A: File Inventory

### Models and Checkpoints
- `backend/models/pretrained/best_piano_model.pt` — 145 MB, source for CNN transfer
- `backend/models/pretrained/best_drum_model.pt` — 115 MB, reference only
- `backend/models/pretrained/best_tab_model.pt` — 59 MB, v1 guitar (collapsed)
- `backend/models/pretrained/best_bass_model.pt` — 119 MB, v1 bass (collapsed)

### Working Inference Code (to be updated)
- `backend/piano_transcriber.py` — PianoTranscriptionModel (transfer source)
- `backend/drum_nn_transcriber.py` — DrumTranscriptionModel (reference)
- `backend/guitar_tab_transcriber.py` — Current Basic Pitch fallback (keep as backup)
- `backend/bass_transcriber.py` — Current Basic Pitch fallback (keep as backup)

### Previous Training Scripts
- `train_tab_model/Guitar_Tab_Training.ipynb` — v1 (collapsed)
- `train_tab_model/Guitar_Tab_Training_v2.ipynb` — v2 (over-predicted)
- `train_tab_model/train_guitar_runpod.py` — v2 RunPod script
- `train_bass_model/Bass_Transcription_Training.ipynb` — v1 (collapsed)
- `train_bass_model/Bass_Transcription_Training_v2.ipynb` — v2 (not run)
- `train_bass_model/train_bass_runpod.py` — v2 RunPod script

### New v3 Files to Create
- `train_tab_model/train_guitar_v3.py` — Main training script
- `train_bass_model/train_bass_v3.py` — Bass training script
- `backend/guitar_nn_transcriber.py` — v3 guitar inference
- `backend/bass_nn_transcriber.py` — v3 bass inference
- `stemscribe/docs/v3_training_plan.md` — This document

---

## Appendix B: Quick-Start Experiment

Before committing to the full v3 training, run this 30-minute probe to validate
that piano CNN features are useful for guitar pitch detection:

```python
# probe_piano_features.py — run on any GPU

import torch
from piano_transcriber import PianoTranscriptionModel

# Load piano CNN
ckpt = torch.load('best_piano_model.pt', map_location='cpu', weights_only=False)
piano = PianoTranscriptionModel()
piano.load_state_dict(ckpt['model_state_dict'])

# Extract CNN, freeze it
cnn = piano.cnn
for p in cnn.parameters():
    p.requires_grad = False

# Simple linear probe: CNN features -> 49 pitch outputs
class LinearProbe(torch.nn.Module):
    def __init__(self, cnn):
        super().__init__()
        self.cnn = cnn
        self.head = torch.nn.Linear(1792, 49)

    def forward(self, mel):
        with torch.no_grad():
            features = self.cnn(mel.unsqueeze(1))
        B, C, F, T = features.shape
        features = features.permute(0, 3, 1, 2).reshape(B, T, -1)
        return self.head(features).permute(0, 2, 1)  # (B, 49, T)

# Train on GuitarSet for 5 epochs with BCEWithLogitsLoss(pos_weight=20)
# If onset F1 > 0.1 after 5 epochs: TRANSFER WORKS, proceed with v3
# If onset F1 == 0: features don't transfer, need different approach
```

This probe validates the entire transfer learning hypothesis with minimal investment.
