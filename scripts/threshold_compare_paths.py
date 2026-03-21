#!/usr/bin/env python3
"""
Compare BTC model inference paths:
1. Standard: self_attn -> output_projection (no LSTM)
2. v10 manual: self_attn -> LSTM -> output_projection
3. Full model forward pass

This tests whether the LSTM path in v10 is degrading confidence.
"""

import sys
import os
import numpy as np
import logging
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.chdir(str(backend_dir))

BTC_DIR = backend_dir.parent / "btc_chord"
sys.path.insert(0, str(BTC_DIR))

import torch
import librosa
from btc_model import BTC_model
from utils.hparams import HParams
from utils.mir_eval_modules import idx2voca_chord
from chord_detector_v10 import _mir_to_simple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load model
device = torch.device("cpu")
config = HParams.load(str(BTC_DIR / "run_config.yaml"))
config.feature['large_voca'] = True
config.model['num_chords'] = 170

model = BTC_model(config=config.model).to(device)

finetuned = backend_dir / "training_data" / "btc_finetune" / "checkpoints" / "btc_finetuned_best.pt"
original = BTC_DIR / "test" / "btc_model_large_voca.pt"
model_file = str(finetuned if finetuned.exists() else original)
print(f"Loading: {'fine-tuned' if finetuned.exists() else 'original'} checkpoint")

checkpoint = torch.load(model_file, map_location=device, weights_only=False)
mean = checkpoint['mean']
std = checkpoint['std']
model.load_state_dict(checkpoint['model'])
model.eval()

idx_to_chord = idx2voca_chord()

# Load one test audio
audio_path = str(backend_dir.parent / "outputs/908afb4a/stems/htdemucs_6s/Farmhouse/guitar.mp3")
original_wav, sr = librosa.load(audio_path, sr=config.mp3['song_hz'], mono=True)
duration = len(original_wav) / sr
print(f"Audio: {duration:.1f}s")

# Compute features (same as v10)
inst_len = config.mp3['inst_len']
song_hz = config.mp3['song_hz']
feature = None
current = 0
while len(original_wav) > current + int(song_hz * inst_len):
    start_idx = int(current)
    end_idx = int(current + song_hz * inst_len)
    tmp = librosa.cqt(
        original_wav[start_idx:end_idx], sr=sr,
        n_bins=config.feature['n_bins'],
        bins_per_octave=config.feature['bins_per_octave'],
        hop_length=config.feature['hop_length'])
    feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
    current = end_idx

tmp = librosa.cqt(
    original_wav[int(current):], sr=sr,
    n_bins=config.feature['n_bins'],
    bins_per_octave=config.feature['bins_per_octave'],
    hop_length=config.feature['hop_length'])
feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
feature = np.log(np.abs(feature) + 1e-6)

feature = feature.T
feature = (feature - mean) / std

n_timestep = config.model['timestep']
num_pad = n_timestep - (feature.shape[0] % n_timestep)
feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)

# Run on first chunk
with torch.no_grad():
    feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
    chunk = feature_tensor[:, :n_timestep, :]

    # Path 1: Self-attention only (standard SoftmaxOutputLayer path)
    encoder_output, _ = model.self_attn_layers(chunk)
    logits_standard = model.output_layer.output_projection(encoder_output).squeeze(0)
    probs_standard = torch.softmax(logits_standard, dim=-1)

    # Path 2: v10 manual LSTM path
    lstm_out, _ = model.output_layer.lstm(encoder_output)
    logits_lstm = model.output_layer.output_projection(lstm_out).squeeze(0)
    probs_lstm = torch.softmax(logits_lstm, dim=-1)

    # Path 3: Use the SoftmaxOutputLayer forward method directly
    # This calls output_projection then softmax then topk
    predictions_forward, second_forward = model.output_layer(encoder_output)

print(f"\n{'='*70}")
print(f"  INFERENCE PATH COMPARISON (first {n_timestep} frames)")
print(f"{'='*70}")

print(f"\n--- Path 1: Standard (self_attn -> output_projection, NO LSTM) ---")
max_probs_std = probs_standard.max(dim=-1).values
print(f"  Max probability per frame: mean={max_probs_std.mean():.4f}, min={max_probs_std.min():.4f}, max={max_probs_std.max():.4f}")
top2_std = torch.topk(probs_standard, k=2, dim=-1)
gap_std = (top2_std.values[:, 0] - top2_std.values[:, 1])
print(f"  Gap (#1 - #2): mean={gap_std.mean():.4f}")
print(f"  Logit range: min={logits_standard.min():.4f}, max={logits_standard.max():.4f}, std={logits_standard.std():.4f}")

# Show sample predictions
for i in [0, 20, 40, 60, 80]:
    top_val, top_idx = torch.topk(probs_standard[i], k=3)
    chords = [f"{_mir_to_simple(idx_to_chord[top_idx[j].item()])}({top_val[j]:.3f})" for j in range(3)]
    print(f"  Frame {i}: {' | '.join(chords)}")

print(f"\n--- Path 2: v10 LSTM path (self_attn -> LSTM -> output_projection) ---")
max_probs_lstm = probs_lstm.max(dim=-1).values
print(f"  Max probability per frame: mean={max_probs_lstm.mean():.4f}, min={max_probs_lstm.min():.4f}, max={max_probs_lstm.max():.4f}")
top2_lstm = torch.topk(probs_lstm, k=2, dim=-1)
gap_lstm = (top2_lstm.values[:, 0] - top2_lstm.values[:, 1])
print(f"  Gap (#1 - #2): mean={gap_lstm.mean():.4f}")
print(f"  Logit range: min={logits_lstm.min():.4f}, max={logits_lstm.max():.4f}, std={logits_lstm.std():.4f}")

for i in [0, 20, 40, 60, 80]:
    top_val, top_idx = torch.topk(probs_lstm[i], k=3)
    chords = [f"{_mir_to_simple(idx_to_chord[top_idx[j].item()])}({top_val[j]:.3f})" for j in range(3)]
    print(f"  Frame {i}: {' | '.join(chords)}")

print(f"\n--- Path 3: SoftmaxOutputLayer.forward() (standard BTC) ---")
predictions_forward = predictions_forward.squeeze(0)
second_forward = second_forward.squeeze(0)
for i in [0, 20, 40, 60, 80]:
    pred = _mir_to_simple(idx_to_chord[predictions_forward[i].item()])
    sec = _mir_to_simple(idx_to_chord[second_forward[i].item()])
    print(f"  Frame {i}: {pred} (2nd: {sec})")

# Check if standard path agrees with LSTM path
agree_std_lstm = (probs_standard.argmax(dim=-1) == probs_lstm.argmax(dim=-1)).float().mean()
agree_std_fwd = (probs_standard.argmax(dim=-1) == predictions_forward).float().mean()
print(f"\n--- Agreement ---")
print(f"  Standard vs LSTM: {agree_std_lstm:.1%} of frames agree")
print(f"  Standard vs Forward: {agree_std_fwd:.1%} of frames agree")

print(f"\n--- KEY FINDING ---")
if max_probs_std.mean() > max_probs_lstm.mean() * 2:
    print(f"  Standard path has {max_probs_std.mean()/max_probs_lstm.mean():.1f}x higher confidence!")
    print(f"  The LSTM path is degrading model confidence.")
    print(f"  RECOMMENDATION: Remove LSTM from inference path in chord_detector_v10.py")
elif max_probs_lstm.mean() > max_probs_std.mean() * 2:
    print(f"  LSTM path has {max_probs_lstm.mean()/max_probs_std.mean():.1f}x higher confidence!")
    print(f"  The LSTM path is correct.")
else:
    print(f"  Both paths produce similar confidence levels.")
    print(f"  Standard: {max_probs_std.mean():.4f}, LSTM: {max_probs_lstm.mean():.4f}")
    if max_probs_std.mean() < 0.05:
        print(f"  BOTH paths have near-uniform confidence -- the issue is deeper (model or features).")
