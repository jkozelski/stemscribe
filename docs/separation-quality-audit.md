# Stem Separation Quality Audit — StemScriber

**Date:** 2026-04-05
**Model:** BS-RoFormer-SW (BS-Roformer-SW.ckpt via audio-separator >=0.30)
**Infrastructure:** Modal A10G (24GB VRAM), 5 concurrent, 600s timeout
**Test songs:** Deacon Blues, The Time Comes (Kozelski)
**Test stems location:** `~/stemscribe/outputs/modal_test/`

---

## 1. Test Stem Inspection

All 6 stems produced successfully:

| Stem | File Size | Duration | Bitrate | Format |
|------|-----------|----------|---------|--------|
| vocals.mp3 | 8.38 MB | 219.5s | 320 kbps | MP3 CBR |
| guitar.mp3 | 8.38 MB | 219.5s | 320 kbps | MP3 CBR |
| bass.mp3 | 8.38 MB | 219.5s | 320 kbps | MP3 CBR |
| drums.mp3 | 8.38 MB | 219.5s | 320 kbps | MP3 CBR |
| piano.mp3 | 8.38 MB | 219.5s | 320 kbps | MP3 CBR |
| other.mp3 | 8.38 MB | 219.5s | 320 kbps | MP3 CBR |

All stems are full-length, 320 kbps CBR, 44100 Hz — production quality encoding. No truncation or corruption detected.

---

## 2. BS-RoFormer-SW Quality Assessment

### SDR Benchmarks (MVSEP Multisong dataset)

| Stem | BS-RoFormer-SW | htdemucs_6s | Delta |
|------|---------------|-------------|-------|
| Vocals | 11.30 | ~9.7 | +1.6 dB |
| Bass | 14.62 | ~10.0 | +4.6 dB |
| Drums | 14.11 | ~8.5 | +5.6 dB |
| Guitar | 9.05 | ~7-8 | +1-2 dB |
| Piano | 7.83 | ~5-6 | +2-3 dB |
| Other | 8.71 | ~5.6 | +3.1 dB |

**BS-RoFormer-SW is substantially better than htdemucs_6s across every stem.** The upgrade from Demucs was the right call.

### Strengths
- **Bass is excellent** (SDR 14.62) — band-split architecture handles low frequencies very well
- **Drums are clean** (SDR 14.11) — minimal bleed, good transient preservation
- **Vocals are strong** (SDR 11.30) — 16% improvement over htdemucs_6s
- **Single-pass 6-stem output** — no cascading/hierarchical pipeline needed
- **Fits comfortably on A10G** — ~8-12 GB VRAM, well within 24 GB budget

### Weaknesses
- **Piano is the weakest stem** (SDR 7.83) — spectral overlap with guitars, vocals, and "other" makes piano the hardest to isolate. This is a known limitation across all current models.
- **Guitar is mediocre** (SDR 9.05) — second-weakest stem; bleeds into piano and vice versa
- **"Other" is a catch-all** (SDR 8.71) — inherently variable quality since it captures everything else
- **Provenance concern** — model shared via Discord, rehosted by jarredou on HuggingFace. Training data/methodology unknown.
- **Capacity trade-off** — dedicated 2-stem models (e.g., Mel-Band RoFormer vocals-only) achieve ~12.9 SDR for vocals vs 11.30 for this 6-stem model

### Expected Bleed Artifacts
- Piano ↔ Guitar: Most likely source of bleed (spectral overlap)
- Vocals → Other: Some vocal remnants may appear in "other" stem
- Bass → Drums: Minimal bleed expected (well-separated frequency ranges)
- Vocals: Cleanest stem overall

---

## 3. Ensemble Approaches

### audio-separator Native Ensemble Support (v0.30+)

The package supports ensembling out of the box:

```python
separator = Separator(ensemble_algorithm='avg_wave', ensemble_weights=[2.0, 1.0])
separator.load_model(model_filename=[model1, model2])
```

**Available algorithms:** `avg_wave`, `median_wave`, `min_wave`, `max_wave`, `avg_fft`, `median_fft`, `uvr_max_spec`, `uvr_min_spec`, `ensemble_wav`

**Available presets:** `vocal_balanced`, `instrumental_clean`, `karaoke`

### Would BS-RoFormer + Mel-Band RoFormer Help?

Research (arXiv:2410.20773, Oct 2024) shows:
- Mel-Band RoFormer is selected 97% of the time for **vocals** in per-stem ensembles
- BS-RoFormer is better for **bass** (mel-scale mapping is less effective at low frequencies)
- SCNet Large dominates for bass (93%) and drums (67%)
- Best approach: **per-stem model selection** rather than simple averaging

**The catch:** BS-RoFormer-SW outputs 6 stems in one pass. Standard BS-RoFormer and Mel-Band RoFormer are 2-stem models (vocal/instrumental). To ensemble them for 6-stem output, you'd need either:
1. Find 6-stem variants of each model, OR
2. Ensemble only the vocal/instrumental split and use a separate pipeline for instrument sub-separation

MVSEP's production ensemble (BS-RoFormer + MelBand RoFormer + SCNet XL IHF) achieves:
- Vocals SDR: 11.93 (vs 11.30 for BS-RoFormer-SW alone)
- Average SDR: 13.67

### Recommendation

**Quickest win:** Ensemble BS-RoFormer-SW + htdemucs_6s with `avg_wave` using native audio-separator support. Both produce 6 stems, so they can be blended directly. This would likely smooth out artifacts without changing the pipeline architecture.

**Maximum quality:** Per-stem model selection — use BS-RoFormer-SW for bass, Mel-Band RoFormer for vocals, etc. This adds 2-3x processing time and significant complexity. Worth considering as a "Premium" tier later.

**For now:** BS-RoFormer-SW alone is the sweet spot. It's already substantially better than the old Demucs pipeline, and the quality is competitive with commercial services.

---

## 4. Lead/Backing Vocal Separation

### Two-Pass Approach (Proven Method)

1. **Pass 1:** Extract vocals with BS-RoFormer-SW (already done)
2. **Pass 2:** Run vocals through `UVR_MDXNET_KARA_2.onnx` → lead vocals + backing vocals

```python
from audio_separator.separator import Separator

# Already have vocals.mp3 from BS-RoFormer-SW pass
separator = Separator(output_dir=output_dir)
separator.load_model('UVR_MDXNET_KARA_2.onnx')
results = separator.separate(vocals_path)
# results[0] = lead vocals
# results[1] = backing vocals
```

### Key Points
- Model auto-downloads on first use, no manual setup
- Running KARA on pre-extracted vocals produces **much better** results than running on the full mix
- Processing time: ~30-60 seconds additional
- Community-validated workflow — standard approach in UVR/audio-separator ecosystem

### Feature Idea
Add an opt-in "Split Vocals" toggle that runs this second pass. Useful for:
- Karaoke practice (lead only or backing only)
- Harmony analysis
- Vocal arrangement study

---

## 5. Post-Processing Capabilities

### Built-in to audio-separator

| Feature | Architecture | Flag | Default |
|---------|-------------|------|---------|
| Artifact removal | VR only | `--vr_enable_post_process` | Off |
| Post-process threshold | VR only | `--vr_post_process_threshold` | 0.2 |
| High-end restoration | VR only | `--vr_high_end_process` | Off |
| Test-time augmentation | VR only | `--vr_enable_tta` | Off |
| Denoise | MDX only | `--mdx_enable_denoise` | Off |
| Normalization | All | `--normalization` | 0.9 |
| Spectral inversion | All | `--invert_spect` | Off |

**None of these apply to RoFormer models.** The built-in post-processing is architecture-specific (VR and MDX only).

### What's NOT Built-in
- **No de-reverb** — requires a separate model pass (UVR-DeEcho-DeReverb, VR architecture)
- **No spectral cleanup** — no noise gate, spectral subtraction
- **No bleed reduction** between stems

### De-Reverb Pipeline (Manual)

For clean dry vocals, the community workflow is:
1. Separate vocals with BS-RoFormer (done)
2. Run vocals through `UVR-DeEcho-DeReverb` model (separate pass)
3. Optionally follow with aggressive de-echo model

Each step is a full `separator.load_model()` + `separator.separate()` call.

### Existing Custom Post-Processing

The worktree ensemble code at `backend/separation/ensemble.py` already has a custom `PostProcessor` with bleed reduction, noise removal, and phase alignment. This is the right approach since audio-separator doesn't provide these for RoFormer models natively.

---

## 6. Summary & Recommendations

### Current State: Good
BS-RoFormer-SW on Modal A10G is a solid production setup. Quality significantly exceeds Demucs across all stems. The ~$0.06/song processing cost is excellent.

### Priority Improvements

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| 1 | Lead/backing vocal split (KARA 2-pass) | Low | High — big UX win for practice mode |
| 2 | Ensemble BS-RoFormer-SW + htdemucs_6s | Medium | Medium — smooths artifacts |
| 3 | De-reverb pass on vocals | Low | Medium — cleaner vocal practice |
| 4 | Per-stem model selection | High | Medium — diminishing returns |

### Known Limitations to Communicate to Users
- Piano stems may contain some guitar bleed (and vice versa) — this is a limitation of all current AI separation models
- "Other" stem is a catch-all — quality varies by song
- Complex arrangements (orchestra, dense mixes) will produce lower quality stems than sparse arrangements

### No Action Required
- Current BS-RoFormer-SW quality is competitive with commercial separation services
- The Modal deployment is correctly configured (A10G, 44100Hz, autocast, model caching)
- 320 kbps MP3 output is appropriate for practice use

---

*Sources: MVSEP benchmarks, arXiv:2410.20773, audio-separator GitHub, UVR community discussions, HuggingFace model pages*
