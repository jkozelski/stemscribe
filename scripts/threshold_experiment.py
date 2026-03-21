#!/usr/bin/env python3
"""
Threshold Experiment for StemScribe v10 Chord Detection
========================================================
Investigates whether the BTC model is filtering out useful signal via:
1. min_duration thresholds
2. 'N' chord filtering
3. Softmax confidence distribution

Tests multiple audio files and reports raw model output statistics.
"""

import sys
import os
import json
import logging
from pathlib import Path
from collections import Counter

# Setup path so we can import backend modules
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))
os.chdir(str(backend_dir))

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Test audio files
TEST_FILES = {
    "Farmhouse (Phish)": str(backend_dir.parent / "outputs/908afb4a/stems/htdemucs_6s/Farmhouse/guitar.mp3"),
    "Houses of the Holy (Zeppelin)": str(backend_dir.parent / "outputs/8b57e908-df5e-496e-bd95-ab008d4c92dc/stems/htdemucs_6s/Led_Zeppelin_-_Houses_of_the_Holy_Remaster_Officia/guitar.mp3"),
    "The Time Comes (Kozelski)": str(backend_dir.parent / "outputs/b8548ca7-d176-485a-a06c-265942ec52ef/stems/htdemucs_6s/Kozelski_-_The_Time_Comes__-__Lyric_Video/guitar.mp3"),
    "Tangled Up In Blue (Dylan)": str(backend_dir.parent / "outputs/54d475f4-3936-4947-9126-7d187ecd6457/stems/htdemucs_6s/Bob_Dylan_-_Tangled_Up_In_Blue_Official_HD_Video/guitar.mp3"),
}

# Filter to existing files only
TEST_FILES = {k: v for k, v in TEST_FILES.items() if Path(v).exists()}


def run_raw_model_analysis(audio_path: str, song_name: str):
    """Run BTC model and analyze raw outputs before any filtering."""
    import torch
    import librosa

    # Import BTC components
    BTC_DIR = backend_dir.parent / "btc_chord"
    btc_path = str(BTC_DIR)
    if btc_path not in sys.path:
        sys.path.insert(0, btc_path)

    from btc_model import BTC_model
    from utils.hparams import HParams
    from utils.mir_eval_modules import idx2voca_chord
    from chord_detector_v10 import _mir_to_simple, _parse_chord

    device = torch.device("cpu")
    config = HParams.load(str(BTC_DIR / "run_config.yaml"))
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170

    model = BTC_model(config=config.model).to(device)

    # Load checkpoint (prefer fine-tuned)
    finetuned = backend_dir / "training_data" / "btc_finetune" / "checkpoints" / "btc_finetuned_best.pt"
    original = BTC_DIR / "test" / "btc_model_large_voca.pt"
    model_file = str(finetuned if finetuned.exists() else original)

    checkpoint = torch.load(model_file, map_location=device, weights_only=False)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    model.eval()

    idx_to_chord = idx2voca_chord()

    # Load audio
    original_wav, sr = librosa.load(str(audio_path), sr=config.mp3['song_hz'], mono=True)
    duration = len(original_wav) / sr

    # Tuning compensation (same as v10)
    tuning_offset = librosa.estimate_tuning(y=original_wav, sr=sr)
    if abs(tuning_offset) > 0.05:
        original_wav = librosa.effects.pitch_shift(original_wav, sr=sr, n_steps=-tuning_offset)

    # Compute CQT features
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

    time_unit = inst_len / config.model['timestep']
    feature = feature.T
    feature = (feature - mean) / std

    n_timestep = config.model['timestep']
    num_pad = n_timestep - (feature.shape[0] % n_timestep)
    feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
    num_instance = feature.shape[0] // n_timestep

    # Collect ALL frame-level predictions with confidence
    all_frames = []  # (time, chord_idx, chord_name, confidence, top3_confs)

    with torch.no_grad():
        feature_tensor = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
        for t in range(num_instance):
            chunk = feature_tensor[:, n_timestep * t:n_timestep * (t + 1), :]
            encoder_output, _ = model.self_attn_layers(chunk)
            lstm_out, _ = model.output_layer.lstm(encoder_output)
            logits = model.output_layer.output_projection(lstm_out).squeeze(0)
            probs = torch.softmax(logits, dim=-1)

            for i in range(n_timestep):
                frame_time = time_unit * (n_timestep * t + i)
                if frame_time > duration:
                    break

                # Top predictions
                top_vals, top_idxs = torch.topk(probs[i], k=5)
                chord_idx = top_idxs[0].item()
                chord_name = _mir_to_simple(idx_to_chord[chord_idx])
                conf = top_vals[0].item()

                top3 = [(
                    _mir_to_simple(idx_to_chord[top_idxs[j].item()]),
                    top_vals[j].item()
                ) for j in range(min(5, len(top_vals)))]

                all_frames.append({
                    'time': frame_time,
                    'chord': chord_name,
                    'confidence': conf,
                    'top5': top3
                })

    return all_frames, duration, tuning_offset


def analyze_frames(frames, song_name, duration):
    """Analyze raw frame predictions for threshold insights."""
    results = {}

    # Basic stats
    total_frames = len(frames)
    confidences = [f['confidence'] for f in frames]
    chord_names = [f['chord'] for f in frames]

    n_chord_count = sum(1 for c in chord_names if c == 'N')
    non_n_confidences = [f['confidence'] for f in frames if f['chord'] != 'N']
    n_confidences = [f['confidence'] for f in frames if f['chord'] == 'N']

    print(f"\n{'='*70}")
    print(f"  {song_name} ({duration:.1f}s)")
    print(f"{'='*70}")

    print(f"\n--- Frame-Level Statistics ---")
    print(f"  Total frames: {total_frames}")
    print(f"  'N' (no chord) frames: {n_chord_count} ({100*n_chord_count/total_frames:.1f}%)")
    print(f"  Chord frames: {total_frames - n_chord_count} ({100*(total_frames-n_chord_count)/total_frames:.1f}%)")

    print(f"\n--- Confidence Distribution (all frames) ---")
    print(f"  Mean:   {np.mean(confidences):.3f}")
    print(f"  Median: {np.median(confidences):.3f}")
    print(f"  Std:    {np.std(confidences):.3f}")
    print(f"  Min:    {np.min(confidences):.3f}")
    print(f"  Max:    {np.max(confidences):.3f}")

    # Confidence histogram
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(confidences, bins=bins)
    print(f"\n--- Confidence Histogram ---")
    for i in range(len(bins)-1):
        bar = '#' * (hist[i] * 50 // max(hist))
        print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]:5d} {bar}")

    if non_n_confidences:
        print(f"\n--- Confidence Distribution (non-N frames only) ---")
        print(f"  Mean:   {np.mean(non_n_confidences):.3f}")
        print(f"  Median: {np.median(non_n_confidences):.3f}")

    if n_confidences:
        print(f"\n--- 'N' chord confidence ---")
        print(f"  Mean:   {np.mean(n_confidences):.3f}")
        print(f"  Median: {np.median(n_confidences):.3f}")

    # Chord variety at different thresholds
    print(f"\n--- Chord Variety by Min Duration ---")
    for min_dur in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        events = consolidate_frames(frames, min_dur, include_n=False)
        unique = set(e['chord'] for e in events)
        print(f"  min_dur={min_dur:.2f}: {len(events):3d} events, {len(unique):3d} unique chords: {sorted(unique)}")

    # What if we INCLUDE 'N' chord events?
    print(f"\n--- Effect of Including 'N' (no chord) Events ---")
    events_no_n = consolidate_frames(frames, 0.3, include_n=False)
    events_with_n = consolidate_frames(frames, 0.3, include_n=True)
    n_events = [e for e in events_with_n if e['chord'] == 'N']
    print(f"  Without N: {len(events_no_n)} events")
    print(f"  With N:    {len(events_with_n)} events ({len(n_events)} are N)")
    if n_events:
        print(f"  N events represent {sum(e['duration'] for e in n_events):.1f}s of audio")

    # Low-confidence frames -- what chords are they?
    print(f"\n--- Low-Confidence Frames (conf < 0.3) ---")
    low_conf = [f for f in frames if f['confidence'] < 0.3 and f['chord'] != 'N']
    if low_conf:
        low_chords = Counter(f['chord'] for f in low_conf)
        print(f"  {len(low_conf)} frames with conf < 0.3 (non-N):")
        for ch, count in low_chords.most_common(10):
            print(f"    {ch}: {count} frames")
    else:
        print(f"  None found")

    # Second-choice analysis: when model picks chord X, what's #2?
    print(f"\n--- Second-Choice Analysis (top 5 per frame, sample) ---")
    sample_indices = list(range(0, len(frames), max(1, len(frames)//8)))[:8]
    for idx in sample_indices:
        f = frames[idx]
        top5_str = " | ".join(f"{ch}({c:.2f})" for ch, c in f['top5'])
        print(f"  t={f['time']:6.1f}s: {top5_str}")

    # Confidence gap analysis: how confident is model in #1 vs #2?
    gaps = []
    for f in frames:
        if len(f['top5']) >= 2:
            gaps.append(f['top5'][0][1] - f['top5'][1][1])

    print(f"\n--- Confidence Gap (#1 - #2) ---")
    print(f"  Mean gap:   {np.mean(gaps):.3f}")
    print(f"  Median gap: {np.median(gaps):.3f}")
    print(f"  Frames with gap < 0.1: {sum(1 for g in gaps if g < 0.1)} ({100*sum(1 for g in gaps if g < 0.1)/len(gaps):.1f}%)")
    print(f"  Frames with gap < 0.05: {sum(1 for g in gaps if g < 0.05)} ({100*sum(1 for g in gaps if g < 0.05)/len(gaps):.1f}%)")

    return results


def consolidate_frames(frames, min_duration, include_n=False):
    """Consolidate frame predictions into chord events with min_duration filter."""
    if not frames:
        return []

    events = []
    prev_chord = frames[0]['chord']
    start_time = frames[0]['time']
    conf_sum = frames[0]['confidence']
    conf_count = 1

    for f in frames[1:]:
        if f['chord'] != prev_chord:
            dur = f['time'] - start_time
            if dur >= min_duration and (include_n or prev_chord != 'N'):
                events.append({
                    'chord': prev_chord,
                    'time': start_time,
                    'duration': dur,
                    'confidence': conf_sum / conf_count
                })
            prev_chord = f['chord']
            start_time = f['time']
            conf_sum = f['confidence']
            conf_count = 1
        else:
            conf_sum += f['confidence']
            conf_count += 1

    # Last event
    if frames:
        dur = frames[-1]['time'] - start_time
        if dur >= min_duration and (include_n or prev_chord != 'N'):
            events.append({
                'chord': prev_chord,
                'time': start_time,
                'duration': dur,
                'confidence': conf_sum / conf_count
            })

    return events


def main():
    if not TEST_FILES:
        print("ERROR: No test audio files found!")
        return

    print(f"Found {len(TEST_FILES)} test files")

    all_results = {}

    for song_name, audio_path in TEST_FILES.items():
        print(f"\nProcessing: {song_name}...")
        try:
            frames, duration, tuning = run_raw_model_analysis(audio_path, song_name)
            print(f"  Tuning offset: {tuning:.3f} semitones")
            results = analyze_frames(frames, song_name, duration)
            all_results[song_name] = {
                'frames': len(frames),
                'duration': duration,
                'tuning': tuning,
            }
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY & RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
KEY FINDING: The v10 BTC chord detector has NO confidence threshold.
It uses argmax on logits -- every frame gets a chord prediction regardless
of confidence. The only filters are:

1. min_duration (0.3s default) -- removes very short chord events
2. 'N' chord filtering -- removes "no chord" predictions

This means the question "is useful signal being filtered out?" comes down to:
- Is min_duration=0.3 too aggressive? (see per-song analysis above)
- Are 'N' predictions hiding real chords?
- Are the softmax confidence values informative?

The confidence gap analysis above shows whether the model is "sure" about
its predictions or if #2 is close to #1 (indicating uncertainty).
""")


if __name__ == '__main__':
    main()
