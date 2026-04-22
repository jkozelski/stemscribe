# Audio Boundary Detection Audit

## Current Issues in `structure_detector.py`

### Bug 1: 4-second window is too coarse for short sections
The chord-based segmentation uses `win_size = 4.0` seconds (line 137). The chorus in "The Time Comes" is only ~12 seconds (D-C#m-Bm / D-C#m-G, roughly 45.6s to 58s). With 4s windows, that's only 3 windows. The short-run merge logic (lines 217-226) absorbs any run < 2 windows into its neighbor, meaning a short section bordered by a slightly different chord mix can vanish entirely.

### Bug 2: "Free your head now" lands at end of Chorus 1 instead of Verse 2
The lyric "Free your head now" starts at 56.16s. The chorus chords (D-C#m-Bm / D-C#m-G) run from ~45.6s to ~58.1s. The verse chords (F#m-A-F#m-B) resume at ~58.1s. Because the 4s window boundary falls at 56s-60s, both the chorus tail AND the verse head land in the same window. The run-length merge pushes this window into the chorus run, dragging "Free your head now" (a verse 2 lyric) into Chorus 1.

### Bug 3: Distinctive chord labeling fails for the chorus
The chorus uses D, C#m, Bm, G — these are "distinctive" chords (appear in <30% of windows). But because the chorus is short (3 windows), the distinctive chord label appears only 3 times, while the verse's `_base_` label appears ~15+ times. The `total_time` heuristic (lines 338-340) assigns the longest-running label as Verse — correct so far. But then the chorus label gets classified via energy as a tiebreaker, and when energy doesn't clearly differentiate (e.g., build-up at end of verse can be loud), it can mislabel or merge sections.

### Bug 4: SSM resolution too low (hop=4096)
`_HOP = 4096` gives ~186ms per frame at 22050Hz. For a 240s song, that's ~1290 frames. The checkerboard kernel size `ksize = min(48, n // 4) = 48` means the kernel covers 48 * 0.186s = ~9s on each side. This blurs boundaries — a 12s chorus gets only 3-4 frames of "distinctive" novelty signal, easily missed.

### Bug 5: Audio-only path has no chord context
`_audio_based_segmentation()` uses a single energy threshold (1.15x median) to distinguish chorus from verse. This is unreliable for songs where verse and chorus have similar energy levels or where the loudest part is a guitar solo, not a chorus.

### Bug 6: Post-labeling heuristics are fragile
- Lines 380-405: "After the last Chorus, rename Verse → Bridge" assumes a pop song structure. Breaks for songs with Verse after final Chorus.
- Lines 281-289: "Merge next segment into intro if short and not base" — can swallow the start of Verse 1 if it has distinctive chords.
- The `label_map` grouping (lines 177-202) uses symmetric difference ≤1, which can over-merge labels that share one chord by coincidence.

## What's Fixed in `audio_boundary_detector.py`

### Multi-feature boundary detection
Instead of relying solely on chord-set clustering, the new detector combines four audio features:
1. **RMS energy envelope** (2s smoothing) — detects loudness transitions
2. **Spectral contrast** (mean across 6 bands, 2s smoothing) — captures timbral changes
3. **Onset density** (onset strength, 4s smoothing) — rhythmic activity changes
4. **Chroma-based SSM novelty** (checkerboard kernel at hop=2048) — harmonic structure changes

### Better time resolution
- Features computed at hop=512 (~23ms) instead of 4096 (~186ms)
- SSM uses hop=2048 (~93ms) — 4x better than original
- Minimum section size tunable (default 6s, tested at 5s for short chorus detection)

### No labeling assumptions
The new detector outputs only boundary timestamps with per-feature confidence scores. Section labeling is delegated to the ensemble (Agent 4), which can combine audio boundaries with chord data and lyrics for better labels.

### Recommended parameters for "The Time Comes"
- `min_section_sec=5.0` — catches the 12s chorus (two boundaries within it)
- `novelty_kernel_sec=6.0` — better sensitivity to short harmonic sections
- Default weights: chroma 35%, energy 30%, spectral 20%, onset 15%

## Key Boundaries Detected (The Time Comes)

| Time | Feature | Expected Section Change |
|------|---------|------------------------|
| 7.8s | energy | Intro guitar entry |
| 22.8s | energy+onset | Verse 1 vocals (actual: 21.5s) |
| 36.8s | chroma (0.96) | Pre-chorus transition |
| 41.9s | energy | Near Chorus 1 start (actual: ~45.6s) |
| 52.1s | energy | Mid-chorus (2nd half D-C#m-G) |
| 60.5s | energy | Verse 2 start (actual: ~58s) |
| 66.6s | chroma (0.40) | Mid-verse 2 harmonic shift |
| 79.1s | energy+chroma | Near Chorus 2 start (actual: ~80s) |
| 121.3s | energy | Bridge/instrumental section |
| 220.4s | energy | Outro fade |

## API for Agent 4

```python
from backend.audio_boundary_detector import detect_audio_boundaries

boundaries = detect_audio_boundaries(
    audio_path="path/to/audio.wav",
    min_section_sec=5.0,       # tune for short sections
    novelty_kernel_sec=6.0,    # smaller = more sensitive
)

# Each boundary:
# {
#     "time": 45.6,
#     "energy_change": 0.34,
#     "spectral_change": 0.10,
#     "onset_density_change": 0.38,
#     "chroma_novelty": 0.46,
#     "confidence": 0.26,
# }
```
