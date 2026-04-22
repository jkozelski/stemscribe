# A*-Guitar Fret Assignment — Implementation Results

**Date:** 2026-04-04
**Status:** Implemented, tested, ready for review
**Files changed:**
- `backend/midi_to_gp.py` — replaced heuristic FretMapper with A*-Guitar search
- `backend/tests/test_midi_to_gp.py` — updated tests, added A* test suite

---

## What Changed

### Removed
- **Old greedy heuristic** in `midi_note_to_fret()` — replaced scoring with cleaner `_transition_cost()` + `_position_cost()` functions
- **Old `FretMapper._map_note_lead()`** — lead mode now uses same A* path with bend-aware cost penalties
- **Dead `FretMapper.map_chord()` method** — was never called by the converter (confirmed by audit)
- **`position_history` list** in FretMapper — no longer needed; A* handles lookahead natively

### Added
- **`astar_fret_search()`** — core A* pathfinding function. Treats the note sequence as a layered graph where each layer is the set of candidate (string, fret) positions for one note. Edges are weighted by hand-movement cost. Uses windowed A* (default 12-note lookahead) to keep complexity bounded.
- **`_astar_segment()`** — inner A* search over a segment with priority queue, state pruning, and admissible heuristic (h=0).
- **`_get_candidates()`** — factored out candidate generation for reuse.
- **`_transition_cost()`** — models fret distance (quadratic penalty beyond hand span), string jumps, and bend constraints.
- **`_position_cost()`** — light intrinsic cost for fret zone preference. Much lighter than old heuristic since A* handles path optimization.
- **`FretMapper.map_sequence()`** — batch entry point that runs A* over the full note sequence. The converter now calls this instead of note-by-note `map_note()`.

### Converter Integration
- `convert_midi_to_gp()` now pre-computes all fret assignments via `fret_mapper.map_sequence()` before the measure-building loop
- Bend flags are computed once from articulation markers and passed to A*
- Drum tracks still use the simplified per-note mapping (unchanged)
- All articulation logic (bends, slides, hammer-ons, vibrato) is preserved exactly as before

---

## Test Results

```
36 passed in 0.06s
```

| Test Class | Tests | Status |
|------------|-------|--------|
| TestMidiNoteToFret | 8 | All pass |
| TestFretMapper | 8 | All pass (3 old + 4 new sequence tests + reset) |
| TestInstrumentDetection | 8 | All pass |
| TestGPFileGeneration | 4 | All pass |
| TestAStarFretSearch | 7 | All pass |

### Key test validations
- **Pentatonic box fits in 7 frets** — A* keeps Am pentatonic (57-69) within a single position
- **Bend avoids open strings** — E4 with bend flag gets fret >= 2 (not open high E)
- **200-note sequence in < 5s** — actual: ~0.01s (windowed A* is fast)
- **A* beats greedy** — on a sequence with position jumps (60-64, 72-76, 60-64), A* produces less total hand movement than note-by-note greedy
- **Bass open strings** — E A D G on bass all correctly map to fret 0

---

## Algorithm Details

### Cost Model
```
transition_cost(prev, cur) =
    fret_delta * 1.0                          if delta <= 4 (hand span)
    4.0 + (delta - 4)^2                       if delta > 4 (quadratic shift penalty)
  + string_delta * 0.5                        (string jump cost)
  + 20 if has_bend and fret < 2               (bend constraint)
  + 10 if has_bend and fret > 17              (high-fret bend penalty)
```

### Windowed Search
- Default lookahead: 12 notes
- Each window runs full A* with state pruning (best-cost-to-state tracking)
- Windows process sequentially; last assigned position carries forward as context
- Handles unmappable notes (out of range) gracefully — skipped in the graph

### Complexity
- Per window: O(W * C^2 * log(W * C)) where W = window size, C = avg candidates per note
- Typical C for guitar = 2-4, so each window is very fast
- Total: O(N/W * W * C^2 * log(W * C)) = O(N * C^2 * log(W * C))
- In practice: 200 notes processes in ~10ms

---

## What's NOT Changed
- GP5 file structure and output format
- Articulation effects (bends, slides, hammer-ons, vibrato)
- Velocity-based ghost/accent notes
- Drum track mapping
- Tuning definitions and instrument detection
- `convert_job_midis_to_gp()` batch API
- All existing public function signatures (backward compatible)
