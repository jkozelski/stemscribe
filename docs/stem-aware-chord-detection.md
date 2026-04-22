# Stem-Aware Chord Detection: Design Document

**Date:** 2026-04-04
**Status:** Research / Architecture Design
**Author:** Jeff Kozelski + Claude

---

## 1. The Insight

Every chord detection system in production today -- BTC, Chordino, Essentia, our V8 model -- works the same way: take a spectrogram of the full mix (or a harmonic remix), pattern-match to chord labels. This is fundamentally limited because the model must simultaneously untangle which instruments are playing, what notes each is contributing, and what chord those notes form.

StemScriber already separates audio into clean per-instrument stems (guitar, bass, piano, drums, vocals, other) using BS-RoFormer-SW before chord detection even begins. Nobody else feeds individually separated stems into multi-pitch estimation and assembles chords from the detected notes. That is our edge.

**Current approach (pattern matching):**
```
full_mix_audio -> spectrogram -> neural_net -> "Am7" (pattern match)
```

**Proposed approach (note assembly):**
```
guitar_stem -> multi-pitch -> {A, C, E, G}  \
bass_stem   -> pitch track -> {A}            |-> assemble -> Am7 (confidence: 0.92)
piano_stem  -> multi-pitch -> {A, C, E}     /
```

---

## 2. Current System Analysis

### 2.1 What We Have Now

The current pipeline (`chord_detector_v10.py`) is a three-layer ensemble:

1. **BTC (primary):** Bi-directional Transformer, 170 chord classes. Runs CQT on a harmonic stem mix (guitar + piano + bass, no drums/vocals). Uses vocabulary constraint from UG/Songsterr scraping when available.

2. **V8 Transformer (fallback):** 337 chord classes including inversions (144 slash chords), mMaj7, dim7, hdim7, add9, sus2/sus4, 6, min6, 9. Consulted when BTC confidence < 0.35.

3. **Essentia (ensemble):** ChordsDetection with HPCP features. Used as a second opinion for root accuracy.

Post-processing includes: consecutive merge, adaptive min-duration filter, rare chord filter, key-aware confidence weighting (with chromatic passing chord protection), tuning detection/correction, and vocab validation against UG scrapes.

### 2.2 Where It Fails

All three models share the same fundamental limitation -- they look at a spectrogram and pattern-match to chord labels:

- **Extended voicings:** E9 vs E7 -- the difference is a single F# note that the spectrogram buries in overtone noise
- **Passing chords:** A 1-beat Bm7 between Am7 and Cm7 gets swallowed by the consolidation window (hop_length=8192 at 22050 Hz = 372ms per frame)
- **Subtle distinctions:** Bm6 vs Bm7 (G# vs A in the voicing) -- the chroma bins for these are adjacent and bleed into each other
- **Inversions:** C/E vs C -- the bass note E is critical but gets mixed into the chroma vector

### 2.3 Existing Assets

| Asset | Location | Notes |
|-------|----------|-------|
| Basic Pitch | `basic_pitch.inference.predict` | Frame-level multi-pitch, 88 keys, ~11.6ms resolution |
| Guitar CRNN | `models/pretrained/best_guitar_model.pt` | Kong-style, 48 pitches (MIDI 40-87), 16kHz/160hop |
| Bass CRNN | `models/pretrained/best_bass_model.pt` | Same architecture, bass range |
| Piano CRNN | `models/pretrained/best_piano_model.pt` | Same architecture, 88 keys |
| V8 classes | `models/pretrained/v8_classes.json` | 337 chord types with inversions |
| Chord vocab index | `training_data/chord_vocabulary_index.json` | 15,416 songs, 2,364 unique chords |
| Separated stems | BS-RoFormer-SW output | guitar, bass, piano, drums, vocals, other |
| librosa | Already in venv | CQT, chroma, onset detection, pitch tracking |

---

## 3. Proposed Architecture

### 3.1 Overview

```
                        SEPARATED STEMS (from BS-RoFormer-SW)
                               |
              +----------------+----------------+
              |                |                |
         guitar.mp3       bass.mp3        piano.mp3
              |                |                |
    [Multi-Pitch Est.]  [Pitch Tracking]  [Multi-Pitch Est.]
              |                |                |
    Frame-level notes    Frame-level root  Frame-level notes
    {A4, C5, E5, G5}    {A2}             {A3, C4, E4}
              |                |                |
              +-------+--------+--------+-------+
                      |                 |
              [Onset Detection]  [Onset Detection]
              per stem               per stem
                      |                 |
                      +--------+--------+
                               |
                    [Note-to-Chord Assembly]
                    pitch_class_set -> chord_name
                               |
                    [Cross-Stem Validation]
                    guitar_chord + bass_root + piano_chord
                               |
                    [Post-Processing Pipeline]
                    (existing v10 filters apply here)
                               |
                         CHORD PROGRESSION
```

### 3.2 Stage 1: Per-Stem Multi-Pitch Estimation

**For guitar and piano stems: Basic Pitch**

Basic Pitch is the right tool. It returns:
- `model_output['note']`: shape (N_frames, 88) -- frame-level activation per MIDI pitch
- `model_output['onset']`: shape (N_frames, 88) -- onset probability per MIDI pitch
- `model_output['contour']`: shape (N_frames, 264) -- fine-grained pitch contour
- `note_events`: list of (start_time, end_time, midi_pitch, amplitude, pitch_bends)

Time resolution: ~11.6ms per frame (256-sample hop at 22050 Hz). This is 32x finer than the current V8/BTC frame rate (372ms). That alone will catch 1-beat passing chords.

For guitar specifically, we already have the Guitar CRNN (`best_guitar_model.pt`) which outputs frame-level activations for 48 pitches (MIDI 40-87) at 10ms resolution (160-sample hop at 16kHz). This model was trained on GuitarSet with domain adaptation from Kong's piano checkpoint, so it may outperform Basic Pitch on isolated guitar. **Recommendation: run both, take the intersection for high confidence.**

For piano, we have the Piano CRNN (`best_piano_model.pt`). Same recommendation.

**For bass stem: Basic Pitch (monophonic mode)**

Bass is almost always monophonic. Basic Pitch handles this, but we can also use:
- `librosa.pyin()` -- probabilistic YIN, excellent for monophonic pitch tracking
- The Bass CRNN (`best_bass_model.pt`) -- frame-level bass note detection

Bass provides the most critical single piece of information: the root note (or bass note for inversions). We should use all three sources and take the consensus.

**Why not CREPE or SPICE?**

- **CREPE** is monophonic-only. Good for bass, useless for guitar/piano polyphony. Basic Pitch is literally built on top of CREPE with added polyphony support, so it strictly dominates.
- **SPICE** (Google) is also monophonic, lower accuracy than CREPE on benchmarks, and not well-maintained.
- **RMVPE** (Robust Model for Vocal Pitch Estimation) -- designed for vocals, not instruments.

**Verdict:** Basic Pitch for multi-pitch on guitar/piano, with CRNN models as secondary validators. librosa.pyin + Basic Pitch + Bass CRNN for bass root detection.

### 3.3 Stage 2: Frame-Level Note Extraction

Convert Basic Pitch's continuous activations into discrete per-frame pitch class sets.

```python
def extract_pitch_classes(model_output, onset_threshold=0.4, frame_threshold=0.3):
    """
    Convert Basic Pitch frame activations to pitch class sets.
    
    Returns:
        List of (time, pitch_class_set, confidences) per frame
        pitch_class_set: frozenset of ints 0-11 (C=0, C#=1, ... B=11)
    """
    note_frames = model_output['note']    # (N, 88)
    onset_frames = model_output['onset']  # (N, 88)
    
    results = []
    for frame_idx in range(note_frames.shape[0]):
        # Active notes: high frame activation OR recent onset
        active = note_frames[frame_idx] > frame_threshold
        onsets = onset_frames[frame_idx] > onset_threshold
        
        # Get MIDI pitches of active notes
        midi_pitches = np.where(active | onsets)[0] + 21  # Basic Pitch offset
        
        # Convert to pitch classes (mod 12)
        pitch_classes = frozenset(p % 12 for p in midi_pitches)
        
        # Confidence = mean activation of detected notes
        if len(midi_pitches) > 0:
            conf = float(np.mean(note_frames[frame_idx, midi_pitches - 21]))
        else:
            conf = 0.0
        
        time = frame_idx * (256 / 22050)  # ~11.6ms per frame
        results.append((time, pitch_classes, conf))
    
    return results
```

### 3.4 Stage 3: Onset-Based Chord Change Detection

Instead of fixed-size frames, detect chord changes at actual musical boundaries.

```python
def detect_chord_boundaries(stem_path, pitch_class_frames):
    """
    Find exact chord change points using onset detection + pitch class changes.
    
    Two signals combined:
    1. Spectral onset detection (librosa.onset.onset_detect) -- catches strums/attacks
    2. Pitch class set changes -- catches harmonic shifts even without a clear onset
    """
    y, sr = librosa.load(stem_path, sr=22050)
    
    # Onset detection tuned for guitar/piano (not drums)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=256,           # Match Basic Pitch resolution
        backtrack=True,           # Snap to nearest preceding minimum
        units='frames',
        pre_max=3, post_max=3,    # Onset peak picking
        pre_avg=3, post_avg=5,
        delta=0.07,               # Sensitivity (lower = more onsets)
        wait=4                    # Min frames between onsets (~46ms)
    )
    
    # Also detect pitch class set changes
    pc_change_frames = []
    prev_pc = frozenset()
    for i, (time, pc_set, conf) in enumerate(pitch_class_frames):
        if pc_set != prev_pc and len(pc_set) > 0:
            pc_change_frames.append(i)
            prev_pc = pc_set
    
    # Merge: union of onset frames and pitch-class change frames
    # Deduplicate within 50ms window
    all_boundaries = sorted(set(onset_frames) | set(pc_change_frames))
    merged = [all_boundaries[0]] if all_boundaries else []
    for b in all_boundaries[1:]:
        if (b - merged[-1]) * (256 / 22050) > 0.05:  # 50ms minimum gap
            merged.append(b)
    
    return merged
```

This catches the 1-beat passing chord that the current system misses. At 120 BPM, one beat = 500ms = ~43 frames at Basic Pitch resolution. The current system's 372ms hop makes this almost invisible. The new system's 11.6ms resolution with onset-aligned segmentation will nail it.

### 3.5 Stage 4: Note-to-Chord Assembly

This is the core innovation. Given a set of detected pitch classes, identify the chord.

#### 3.5.1 Interval-Based Chord Lookup

Build a chord dictionary mapping pitch class interval sets to chord names:

```python
# Chord definitions: intervals from root (in semitones)
CHORD_INTERVALS = {
    'maj':    {0, 4, 7},
    'min':    {0, 3, 7},
    '7':      {0, 4, 7, 10},
    'maj7':   {0, 4, 7, 11},
    'min7':   {0, 3, 7, 10},
    'mMaj7':  {0, 3, 7, 11},
    'dim':    {0, 3, 6},
    'dim7':   {0, 3, 6, 9},
    'hdim7':  {0, 3, 6, 10},
    'aug':    {0, 4, 8},
    'sus2':   {0, 2, 7},
    'sus4':   {0, 5, 7},
    '6':      {0, 4, 7, 9},
    'min6':   {0, 3, 7, 9},
    '9':      {0, 2, 4, 7, 10},      # includes the 9th
    'add9':   {0, 2, 4, 7},
    'min9':   {0, 2, 3, 7, 10},
    'maj9':   {0, 2, 4, 7, 11},
    '7#9':    {0, 3, 4, 7, 10},      # Hendrix chord
    '7b9':    {0, 1, 4, 7, 10},
    '11':     {0, 2, 4, 5, 7, 10},
    '13':     {0, 2, 4, 7, 9, 10},
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def identify_chord(pitch_classes: set, bass_pitch_class: int = None) -> tuple:
    """
    Given detected pitch classes and optional bass note, identify the chord.
    
    Returns: (chord_name, root, quality, bass, confidence)
    
    Algorithm:
    1. Try each pitch class as potential root
    2. Compute intervals from that root
    3. Find best matching chord template
    4. If bass_pitch_class differs from root, it is an inversion
    """
    if not pitch_classes:
        return ('N', 'N', 'none', None, 0.0)
    
    best_match = None
    best_score = -1
    
    for candidate_root in pitch_classes:
        # Intervals relative to this root
        intervals = frozenset((pc - candidate_root) % 12 for pc in pitch_classes)
        
        for quality, template in CHORD_INTERVALS.items():
            # Score = Jaccard similarity (intersection / union)
            intersection = len(intervals & template)
            union = len(intervals | template)
            score = intersection / union if union > 0 else 0
            
            # Bonus for exact match
            if intervals == template:
                score = 1.0
            
            # Bonus if bass confirms this root
            if bass_pitch_class is not None and bass_pitch_class == candidate_root:
                score += 0.15
            
            if score > best_score:
                best_score = score
                root_name = NOTE_NAMES[candidate_root]
                
                # Determine bass/inversion
                bass_name = None
                if bass_pitch_class is not None and bass_pitch_class != candidate_root:
                    bass_name = NOTE_NAMES[bass_pitch_class]
                
                # Format chord name
                if quality == 'maj':
                    chord_name = root_name
                elif quality == 'min':
                    chord_name = f"{root_name}m"
                else:
                    chord_name = f"{root_name}{quality}"
                
                if bass_name:
                    chord_name = f"{chord_name}/{bass_name}"
                
                best_match = (chord_name, root_name, quality, bass_name, best_score)
    
    return best_match if best_match else ('N', 'N', 'none', None, 0.0)
```

#### 3.5.2 Handling Ambiguity

The same set of notes can spell different chords depending on context:

- `{A, C, E, G}` = Am7 OR C6 (if bass is C)
- `{C, E, G, B}` = Cmaj7 OR Em/C (if bass is C but melody emphasizes E)
- `{D, F#, A, C}` = D7 OR F#dim/D

Resolution strategy (in priority order):

1. **Bass note is king.** If bass stem clearly plays A, and guitar has {A, C, E, G}, it is Am7. If bass plays C, it is C6/Am7 -- report as C6.
2. **Root position preference.** When ambiguous and no clear bass, prefer the interpretation where the lowest detected note is the root.
3. **Key context.** If the detected key is Am, prefer Am7 over C6 (Am7 is i7, C6 is bIII6 -- the former is far more common).
4. **Vocabulary constraint.** If UG/Songsterr says the song uses Am7 but not C6, prefer Am7.
5. **Temporal context.** If the previous chord was Dm7 and the next is Em7, Am7 fits the ii-V-i progression; C6 does not.

#### 3.5.3 Mapping to V8's 337-Class Vocabulary

The V8 model's 337 classes are the target output vocabulary. After note-to-chord assembly produces a candidate, we verify it exists in `v8_classes.json`. If not, we map to the nearest match:

```python
def map_to_v8_class(chord_name: str, v8_classes: list) -> str:
    """Map a detected chord to the closest V8 class."""
    if chord_name in v8_classes:
        return chord_name
    
    # Try without inversion
    base = chord_name.split('/')[0]
    if base in v8_classes:
        return base
    
    # Try enharmonic equivalent
    for old, new in ENHARMONIC.items():
        if chord_name.startswith(old):
            alt = new + chord_name[len(old):]
            if alt in v8_classes:
                return alt
    
    # Simplify: drop extensions (9 -> 7, add9 -> maj, etc.)
    simplifications = [
        ('add9', ''), ('9', '7'), ('11', '7'), ('13', '7'),
        ('min9', 'min7'), ('maj9', 'maj7'),
    ]
    for src, dst in simplifications:
        if src in chord_name:
            simplified = chord_name.replace(src, dst)
            if simplified in v8_classes:
                return simplified
    
    return chord_name  # Keep as-is if no match (extend vocabulary)
```

### 3.6 Stage 5: Cross-Stem Validation

The power of having multiple stems is redundancy. Each stem provides an independent vote.

```python
def cross_validate_chord(guitar_result, bass_result, piano_result, 
                         time_window=0.05):
    """
    Combine chord evidence from multiple stems.
    
    Confidence boosting rules:
    - Guitar chord + bass confirms root: +0.15 confidence
    - Piano chord agrees with guitar: +0.10 confidence
    - All three agree: +0.20 confidence (cap at 0.99)
    - Bass contradicts guitar root: flag for review, use bass root
    - Piano contradicts guitar quality: use higher-confidence source
    """
    # Extract pitch classes from each source at this time point
    guitar_pcs = guitar_result['pitch_classes']
    bass_pc = bass_result['pitch_class']  # single note
    piano_pcs = piano_result['pitch_classes']
    
    # Union all detected pitch classes
    all_pcs = guitar_pcs | piano_pcs
    if bass_pc is not None:
        all_pcs.add(bass_pc)
    
    # Identify chord from combined evidence
    chord = identify_chord(all_pcs, bass_pitch_class=bass_pc)
    
    # Compute cross-stem agreement score
    agreement = 0.0
    
    # Check if guitar's implied root matches bass
    guitar_chord = identify_chord(guitar_pcs)
    guitar_root_pc = NOTE_NAMES.index(guitar_chord[1]) if guitar_chord[1] != 'N' else None
    if guitar_root_pc is not None and bass_pc == guitar_root_pc:
        agreement += 0.15
    
    # Check if piano agrees with guitar
    piano_chord = identify_chord(piano_pcs)
    if piano_chord[0] != 'N' and piano_chord[2] == guitar_chord[2]:  # same quality
        agreement += 0.10
    
    # Apply agreement bonus
    final_confidence = min(0.99, chord[4] + agreement)
    
    return (*chord[:4], final_confidence)
```

### 3.7 Stage 6: Temporal Smoothing and Output

Apply the existing v10 post-processing pipeline (it is well-tested and handles edge cases):

1. Merge consecutive identical chords
2. Adaptive min-duration filter (tempo-aware)
3. Rare chord filter (adaptive based on unique count)
4. Key-aware confidence weighting (diatonic boost, chromatic passing protection)
5. Re-merge after filtering
6. Tuning detection and correction

The key difference: the input to this pipeline will be much higher quality because chord identifications come from actual detected notes rather than spectrogram pattern matching.

---

## 4. Chroma Features vs CQT vs Constant-Q Chromagram

For the note detection stage on separated stems, we have three feature options:

| Feature | Resolution | Best For | Drawback |
|---------|-----------|----------|----------|
| **CQT** (librosa.cqt) | Log-frequency, ~3 bins/semitone | Raw pitch content | High-dimensional, needs reduction |
| **Chroma CQT** (librosa.feature.chroma_cqt) | 12 pitch classes | Chord template matching | Loses octave info, not useful for multi-pitch |
| **Constant-Q Chromagram** | 12 bins, CQT-derived | Quick chord overview | Same octave-folding problem |
| **Basic Pitch model_output['note']** | 88 MIDI pitches, ~11.6ms | Individual note detection | Neural net, not pure DSP |

**Verdict for this design:** We do NOT need chroma features for the note detection stage. Chroma collapses octave information, which is exactly what we are trying to preserve. Basic Pitch's 88-pitch frame output is the right representation -- it tells us "A4 is active, C5 is active, E5 is active" rather than "pitch classes A, C, E are active somewhere."

Chroma features remain useful only for the key detection step (where octave information is irrelevant) and as a fallback if Basic Pitch is unavailable.

---

## 5. Prototype Plan: Thunderhead Guitar Stem

### 5.1 Test Data

The Thunderhead separated stems are at `/Users/jeffkozelski/stemscribe/backend/outputs/modal_test/`. Known chords for the song: Am7, Bm7, Cm7, E9, Bm6, Dm7, Gmaj7, Em7, G#7, G#maj7.

### 5.2 Prototype Steps

```python
# Step 1: Run Basic Pitch on guitar stem
from basic_pitch.inference import predict
model_output, midi_data, note_events = predict(
    'outputs/modal_test/guitar.mp3',
    onset_threshold=0.4,
    frame_threshold=0.3,
    minimum_note_length=80,
    minimum_frequency=80.0,   # E2
    maximum_frequency=1200.0  # ~D#6
)

# Step 2: Extract frame-level pitch class sets
# model_output['note'] is (18879, 88) -- one row per ~11.6ms frame

# Step 3: Run onset detection on the guitar stem
import librosa
y, sr = librosa.load('outputs/modal_test/guitar.mp3', sr=22050)
onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=256, units='time')

# Step 4: Segment by onsets, extract dominant pitch class set per segment

# Step 5: Run identify_chord() on each segment

# Step 6: Run Basic Pitch on bass stem for root confirmation
bass_output, _, _ = predict('outputs/modal_test/bass.mp3',
    onset_threshold=0.5, frame_threshold=0.4,
    minimum_frequency=30.0, maximum_frequency=350.0)

# Step 7: Cross-reference guitar chords with bass roots

# Step 8: Compare against known chord list
known = ['Am7', 'Bm7', 'Cm7', 'E9', 'Bm6', 'Dm7', 'Gmaj7', 'Em7', 'G#7', 'G#maj7']
```

### 5.3 What We Are Measuring

1. **Note detection accuracy:** Does Basic Pitch correctly identify the notes in each guitar chord voicing?
2. **E9 vs E7:** Can we detect the F# (9th) that distinguishes E9 from E7?
3. **Bm6 vs Bm7:** Can we distinguish G# (6th) from A (7th)?
4. **Passing chord capture:** Do onset detection + pitch class change detection catch 1-beat chords?
5. **Bass root accuracy:** Does the bass stem clearly show the root note at each chord change?
6. **Cross-stem agreement:** Do guitar and bass agree on chord identity?

---

## 6. Compute Requirements

### 6.1 Basic Pitch Performance

Basic Pitch is a lightweight CNN (ICASSP 2022 model). From our test run on the Thunderhead guitar stem (3.7 minutes):
- Produced 18,879 frames and 2,984 note events
- Runs on CPU without issue
- Model size: ~20MB

Running Basic Pitch on 3 stems (guitar, bass, piano) sequentially would take roughly 3x the single-stem time. On the M3 Max this is fast. On the Hetzner VPS (4 CPU, 8GB RAM), it will be slower but should complete within 30-60 seconds for a 4-minute song.

### 6.2 CRNN Models

The Guitar/Bass/Piano CRNN models are also CPU-compatible:
- Guitar model: ~114MB, Kong-style CNN+GRU
- Processes in 10-second chunks
- Already runs on the VPS for transcription

### 6.3 librosa Onset Detection

Pure CPU, negligible compute. The CQT computation for onset_strength is the heaviest part and takes <2 seconds per stem.

### 6.4 Chord Assembly Logic

Pure Python/numpy, microseconds per frame.

### 6.5 Total Estimated Time

| Step | M3 Max (local) | Hetzner VPS (CPU) |
|------|----------------|-------------------|
| Basic Pitch x3 stems | ~15s | ~45s |
| CRNN models x3 | ~10s | ~30s |
| Onset detection x3 | ~3s | ~8s |
| Chord assembly | <1s | <1s |
| Post-processing | <1s | <1s |
| **Total** | **~30s** | **~85s** |

**Verdict: This runs on the Hetzner VPS (8GB RAM, CPU). No Modal GPU needed.** The total adds ~85 seconds to the pipeline on VPS, which is acceptable given that stem separation (the bottleneck) takes 2-3 minutes on Modal GPU. Chord detection runs concurrently after separation completes.

Memory footprint: Basic Pitch model (~20MB) + CRNN models (~350MB total if all loaded) + audio buffers (~50MB) = ~420MB. Well within 8GB.

---

## 7. Integration Strategy

### 7.1 Recommended Approach: Replace BTC/V8, Keep Post-Processing

The stem-aware detector should **replace** the BTC and V8 pattern-matching models, not run alongside them. Reasons:

1. The stem-aware approach is strictly more informative (it has the same audio data plus structural separation)
2. Running BTC+V8+stem-aware would triple compute time for diminishing returns
3. The post-processing pipeline (v10's merge/filter/key-weighting/tuning) is model-agnostic and should be preserved

### 7.2 Fallback Chain

```
IF separated stems available:
    -> Stem-Aware Chord Detection (new)
ELIF original audio available:
    -> BTC + V8 ensemble (current v10)
ELSE:
    -> Essentia fallback (current)
```

This maintains backward compatibility. Songs processed before the upgrade keep working. The stem-aware path only activates when stems exist (which they always do for new uploads, since separation runs first).

### 7.3 Pipeline Integration Point

In `processing/transcription.py`, `detect_chords_for_job()` currently:
1. Mixes harmonic stems into `harmonic_mix.wav`
2. Runs `ChordDetector().detect(harmonic_mix.wav)`

The new approach changes step 1:
1. Pass individual stem paths (guitar, bass, piano) to the new detector
2. The new detector runs multi-pitch estimation on each stem separately
3. Assembles chords from detected notes
4. Applies existing post-processing

```python
# New entry point in detect_chords_for_job():
from stem_chord_detector import StemAwareChordDetector

detector = StemAwareChordDetector()
progression = detector.detect(
    guitar_path=job.stems.get('guitar'),
    bass_path=job.stems.get('bass'),
    piano_path=job.stems.get('piano'),
    artist=artist,
    title=title,
)
```

### 7.4 Module Structure

New file: `backend/stem_chord_detector.py`

```
class StemAwareChordDetector:
    def __init__(self, min_duration=0.3)
    def detect(self, guitar_path, bass_path, piano_path, artist, title) -> ChordProgression
    
    # Internal stages
    def _detect_notes_basic_pitch(self, stem_path, instrument) -> FrameNotes
    def _detect_notes_crnn(self, stem_path, instrument) -> FrameNotes
    def _detect_bass_root(self, bass_path) -> FrameRoots
    def _detect_onsets(self, stem_path) -> list[float]
    def _segment_by_onsets(self, frame_notes, onsets) -> list[Segment]
    def _assemble_chord(self, pitch_classes, bass_root, key) -> ChordEvent
    def _cross_validate(self, guitar_chords, bass_roots, piano_chords) -> list[ChordEvent]
```

Reuses from v10:
- `postprocess_chords()` (all 5 steps)
- `_lookup_vocab()` / `_validate_vocab()` (UG/Songsterr constraint)
- `detect_and_correct_tuning()` (tuning correction)
- `ChordEvent` and `ChordProgression` dataclasses

---

## 8. Known Risks and Mitigations

### 8.1 Basic Pitch Accuracy on Separated Stems

**Risk:** Basic Pitch was trained on full-mix polyphonic audio. Separated stems have different spectral characteristics (missing overtones, separation artifacts). It may underperform on very clean stems.

**Mitigation:** Run both Basic Pitch and the instrument-specific CRNN model. Take the intersection (notes both agree on) for high confidence, and the union with lower confidence for uncertain notes. The CRNN models were trained specifically on single-instrument audio.

### 8.2 Separation Artifacts

**Risk:** BS-RoFormer sometimes bleeds vocals or drums into the guitar stem, creating phantom notes.

**Mitigation:**
- Apply frequency bandpass filtering before note detection (guitar: 80-1200 Hz, bass: 30-350 Hz)
- Use amplitude thresholding to ignore low-energy bleed
- Cross-stem validation naturally filters artifacts (a bleed artifact in guitar won't appear in bass/piano)

### 8.3 Sparse Piano/Guitar

**Risk:** Many songs have guitar but no piano (or vice versa). The cross-validation loses one input.

**Mitigation:** The system degrades gracefully. Guitar + bass alone is still much better than spectrogram pattern matching. Even guitar alone (multi-pitch on a clean stem) outperforms full-mix chroma template matching. Piano is a bonus when present.

### 8.4 Complex Voicings and Open Strings

**Risk:** Guitar voicings often include open strings that ring across chord changes, creating "smeared" pitch class sets.

**Mitigation:**
- Use onset detection to identify the attack of each new chord voicing
- Apply a short decay window: only consider notes active within 100ms after an onset
- Weight recently-attacked notes higher than sustaining notes

### 8.5 Chord Vocabulary Explosion

**Risk:** Note assembly can produce chords outside the V8 337-class vocabulary (e.g., Cmaj7#11, F13b9).

**Mitigation:** Map to nearest V8 class using the simplification ladder (Section 3.5.3). Extended chords that simplify cleanly (Cmaj9 -> Cmaj7, D13 -> D7) are handled. Truly exotic voicings fall back to the closest practical chord name.

---

## 9. Related Work and Novel Aspects

### 9.1 Prior Art

- **"Training chord recognition models on artificially generated audio" (2025, arxiv 2508.05878):** Uses separated tracks through a 301-class chord recognizer. Closest to our approach, but still uses pattern-matching on each track rather than assembling chords from individual notes.

- **"Multi-pitch estimation with polyphony per instrument" (2025, Springer):** U-Net model for per-instrument multi-pitch estimation. Focuses on estimation accuracy rather than chord assembly.

- **"Enhancing Automatic Chord Recognition through LLM Chain-of-Thought Reasoning" (2025, arxiv 2509.18700):** Uses LLMs to improve chord recognition via reasoning chains. Orthogonal to our approach -- could be combined.

- **musicpy (Rainbow-Dreamer):** Python library with a note-to-chord identification algorithm based on interval analysis. Similar logic to our Stage 4, but designed for symbolic (MIDI) input, not audio.

### 9.2 What Makes This Novel

No published system combines all three of these:
1. Real-time source separation producing clean per-instrument stems
2. Multi-pitch estimation on each separated stem independently
3. Chord assembly from detected individual notes with cross-stem validation

The closest work runs chord recognition models on separated tracks (still pattern matching). Our approach goes one level deeper: detect the actual notes, then apply music theory to name the chord. This is how a human musician identifies chords -- hear the notes, name the chord -- and it is only possible when you have clean separated stems.

---

## 10. Implementation Phases

### Phase 1: Prototype (1-2 days)
- Run Basic Pitch on Thunderhead guitar/bass/piano stems
- Implement `identify_chord()` function
- Compare detected chords against known chord list
- Measure accuracy on E9 vs E7, Bm6 vs Bm7

### Phase 2: Core Module (2-3 days)
- Build `stem_chord_detector.py` with full pipeline
- Integrate onset detection for chord boundary detection
- Add cross-stem validation
- Wire into `detect_chords_for_job()`

### Phase 3: Validation (1-2 days)
- Test on 10+ songs with known chords from the chord vocabulary index
- Compare accuracy against current BTC+V8 ensemble
- Tune thresholds (onset sensitivity, frame thresholds, confidence weights)

### Phase 4: Production (1 day)
- Deploy to Hetzner VPS
- Monitor processing times and memory usage
- Verify fallback to BTC+V8 when stems unavailable

**Total estimated effort: 5-8 days**

---

## 11. Summary

The stem-aware chord detection system replaces spectrogram-to-label pattern matching with a fundamentally different approach: detect individual notes from clean separated stems, then assemble chords from the detected notes using music theory. This leverages StemScriber's unique advantage (real-time stem separation at processing time) to achieve what no other chord detection system can -- identification of extended voicings, passing chords, and subtle harmonic distinctions that spectral pattern matching misses.

The system runs entirely on CPU (Hetzner VPS compatible, no Modal GPU needed), adds ~85 seconds to the pipeline, and degrades gracefully when stems are partially available. It reuses the battle-tested v10 post-processing pipeline and integrates cleanly into the existing processing flow.
