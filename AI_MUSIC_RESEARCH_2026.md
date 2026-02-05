# StemScribe AI Music Research - February 2026

## Executive Summary

The AI music landscape has evolved dramatically. What seemed like "years away" is now available through various tools and models. This document outlines findings and a practical integration roadmap for StemScribe.

---

## 🎯 Key Findings

### 1. **Stem Separation** (MAJOR UPGRADE AVAILABLE)

#### Current State: Demucs htdemucs_6s
- Separates: vocals, drums, bass, guitar, piano, other
- Limitation: Original maintainer left Meta, repo archived Jan 2025

#### New Options:

**A. audio-separator (Python Package)** ⭐ RECOMMENDED
```bash
pip install audio-separator
```
- Wraps ALL UVR models (MDX-Net, VR Arch, Demucs, MDXC, BS-Roformer)
- Best vocal model: `model_bs_roformer_ep_317_sdr_12.9755` (SDR 12.9)
- Auto-downloads models on first use
- MIT Licensed, actively maintained

**B. BS-Roformer / Mel-Roformer** (State-of-the-Art)
```bash
pip install bs-roformer
# or
pip install melband-roformer-infer
```
- ByteDance's SOTA architecture
- Better than Demucs for vocals (SDR 12.9 vs ~10)
- Frequency-aware band splitting

**C. PolUVR** (Enhanced Fork)
```bash
pip install PolUVR
```
- Adds CUDA & CoreML (Apple Silicon) acceleration
- Gradio UI included

---

### 2. **Lead vs Backing Vocals** ✅ SOLVED

#### Method 1: Two-Pass UVR (FREE, Local)
1. Run through `UVR-MDX-NET Karaoke 2` → get Vocals
2. Run Vocals through same model again:
   - Output "Vocals" = Lead Vocals
   - Output "Instrumentals" = Backing Vocals

#### Method 2: HP-Karaoke-UVR Model
- Run through Kim Vocals 2 first
- Then HP-Karaoke-UVR model
- Outputs lead and backing separately

#### Implementation for StemScribe:
```python
from audio_separator.separator import Separator

separator = Separator()
# First pass - extract all vocals
separator.load_model('UVR-MDX-NET-Voc_FT.onnx')
vocals = separator.separate('song.wav')

# Second pass - split lead/backing
separator.load_model('UVR_MDXNET_KARA_2.onnx')
lead, backing = separator.separate(vocals)
```

---

### 3. **Lead vs Rhythm Guitar** ⚠️ CHALLENGING

#### Available Solutions:

**A. Moises.ai (Commercial API)** - BEST OPTION
- GraphQL API available
- Specifically trained for lead/rhythm separation
- Also separates acoustic/electric guitar
- Pricing: API access required (paid)

**B. Stereo Panning (Already in StemScribe)**
- Works for classic rock (guitars panned L/R)
- Free, local processing
- Limited to panned recordings

**C. Music.ai API**
- Similar to Moises
- Can separate rhythm/solo guitar parts

**D. AudioCleaner.ai**
- Can detect and separate guitar solos
- Works with lead, rhythm, acoustic, electric

#### Recommendation:
For now, keep stereo splitting. Consider Moises.ai API integration as a premium feature.

---

### 4. **Music Transcription** (MAJOR UPGRADE AVAILABLE)

#### Current: Basic Pitch
- Good for monophonic
- Struggles with chords/polyphony
- No guitar-specific training

#### Better Options:

**A. Omnizart** ⭐ RECOMMENDED (Free, Open Source)
```bash
pip install omnizart
omnizart download-checkpoints

# Usage
omnizart music transcribe song.wav   # Pitched instruments
omnizart drum transcribe song.wav    # Drums
omnizart chord transcribe song.wav   # Chords
```
- Multi-instrument transcription
- Drum transcription
- Chord detection
- Beat tracking
- ⚠️ Not compatible with ARM Mac (Intel only)

**B. Google MT3** (Multi-Task Multitrack)
- Research-grade quality
- Handles multiple instruments simultaneously
- Colab notebook available
- Used as benchmark in 2025 AMT Challenge

**C. Klangio Guitar2Tabs** (Commercial)
- BEST guitar transcription available
- Handles distorted tones, solos, chords
- Exports: TAB, Sheet Music, MIDI, MusicXML, GuitarPro
- Pricing: $14.99/mo or $5.99/mo annual
- No self-hosted option

**D. Songscription AI**
- Full transcription to notation
- Detects fingering positions
- Standard tuning only

---

### 5. **Chord Detection** (UPGRADE AVAILABLE)

#### Current: Custom chord_detector.py
- Basic frequency analysis

#### Better Options:

**A. Omnizart Chord Module**
```bash
omnizart chord transcribe song.wav
```

**B. Chordify Pro AI+** (Commercial)
- Industry standard
- Complex chord detection

**C. autochord** (Python)
```bash
pip install autochord
```

---

## 🚀 Integration Roadmap

### Phase 1: Quick Wins (1-2 days)

1. **Replace Demucs with audio-separator**
   - Better models (BS-Roformer)
   - Same interface, drop-in replacement
   - Better vocal quality

2. **Add Lead/Backing Vocal Split**
   - Two-pass UVR method
   - Use existing stereo_splitter pattern

### Phase 2: Enhanced Transcription (3-5 days)

1. **Integrate Omnizart** (if not on ARM Mac)
   - Better multi-instrument transcription
   - Built-in chord detection
   - Drum transcription

2. **Add MT3 Option**
   - For complex polyphonic material
   - Research-grade accuracy

### Phase 3: Premium Features (Optional)

1. **Moises.ai API Integration**
   - Lead/rhythm guitar separation
   - Acoustic/electric detection
   - Requires API key

2. **Klangio Integration**
   - Best-in-class guitar TABs
   - Freemium (first 30 seconds free)

---

## 📦 Recommended Dependencies

```bash
# Core separation (replaces demucs)
pip install audio-separator

# Advanced transcription
pip install omnizart  # Note: Intel Mac only

# Alternative transcription
pip install basic-pitch  # Keep as fallback

# Roformer models (SOTA)
pip install bs-roformer
pip install melband-roformer-infer
```

---

## 🔧 Code Examples

### Using audio-separator for Better Stem Separation
```python
from audio_separator.separator import Separator

def separate_with_roformer(audio_path, output_dir):
    """Use BS-Roformer for better vocal separation"""
    separator = Separator(output_dir=output_dir)

    # Best vocal/instrumental model
    separator.load_model('model_bs_roformer_ep_317_sdr_12.9755.ckpt')
    output_files = separator.separate(audio_path)

    return output_files
```

### Two-Pass Vocal Split
```python
def split_lead_backing_vocals(vocals_path, output_dir):
    """Split vocals into lead and backing"""
    separator = Separator(output_dir=output_dir)
    separator.load_model('UVR_MDXNET_KARA_2.onnx')

    # Second pass splits lead from backing
    results = separator.separate(vocals_path)
    # results[0] = lead vocals
    # results[1] = backing vocals
    return results
```

### Omnizart Transcription
```python
import subprocess

def transcribe_with_omnizart(audio_path, output_dir):
    """Better transcription with Omnizart"""
    # Music (pitched instruments)
    subprocess.run([
        'omnizart', 'music', 'transcribe',
        audio_path, '-o', f'{output_dir}/music.mid'
    ])

    # Drums
    subprocess.run([
        'omnizart', 'drum', 'transcribe',
        audio_path, '-o', f'{output_dir}/drums.mid'
    ])

    # Chords
    subprocess.run([
        'omnizart', 'chord', 'transcribe',
        audio_path, '-o', f'{output_dir}/chords.json'
    ])
```

---

## 📊 Model Comparison

| Task | Current | Recommended | Improvement |
|------|---------|-------------|-------------|
| Vocal separation | Demucs (SDR ~10) | BS-Roformer (SDR 12.9) | +29% quality |
| Lead/backing vocals | None | UVR two-pass | NEW FEATURE |
| Guitar transcription | Basic Pitch | Omnizart/MT3 | Polyphonic support |
| Chord detection | Custom | Omnizart | More accurate |
| Drum transcription | Basic Pitch | Omnizart | Purpose-built |

---

## 🔗 Sources

- [audio-separator on PyPI](https://pypi.org/project/audio-separator/)
- [BS-Roformer GitHub](https://github.com/lucidrains/BS-RoFormer)
- [Omnizart Documentation](https://music-and-culture-technology-lab.github.io/omnizart-doc/)
- [Google MT3](https://github.com/magenta/mt3)
- [UVR GUI GitHub](https://github.com/Anjok07/ultimatevocalremovergui)
- [Moises.ai API](https://developer-legacy.moises.ai/)
- [Klangio Guitar2Tabs](https://klang.io/guitar2tabs/)
- [UVR Best Models 2025](https://vocalremover.cloud/blog/uvr-best-model-aug-2025)

---

## ✅ Next Steps

1. Test `audio-separator` with BS-Roformer on existing stems
2. Implement two-pass vocal splitting
3. Test Omnizart on Intel Mac or Linux
4. Evaluate Moises.ai API pricing for lead/rhythm guitar
5. Update StemScribe backend to use new models

---

*Research compiled: February 5, 2026*
