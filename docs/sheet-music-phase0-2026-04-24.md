# Phase 0 — Sheet-music rendering prototype

**Date:** 2026-04-24
**Subject:** Is the existing Basic Pitch MIDI good enough to drive real musical notation via `midi_to_musicxml` → OSMD?
**Gate:** MusicXML renders as readable music (not random dots).
**Verdict: PASS. Proceed to Phase 1.**

---

## TL;DR

Pulled the three archived `.mid` files from yesterday's MIDI-detector corpus (`~/stemscribe_archive/2026-04-23-midi-phase0-corpus/bp_out/`). Ran each through the existing `backend/midi_to_notation.py::midi_to_musicxml(...)` unchanged, then loaded the resulting MusicXML in OSMD (the same library the practice page already ships). All three stems render as real music — title, composer, tempo, clef, time signature, key, dynamics, measure numbers, beamed note groupings.

Basic Pitch MIDI quality is sufficient for notation. The music21 path we already have is ready. Nothing needs to be regenerated.

## Inputs

From the archive:

| File | Size | Notes |
|---|---:|---|
| `piano.mid` | 12.6 KB | 2,002 note events |
| `guitar.mid` | 15.1 KB | 820 note events |
| `bass.mid` | 20.9 KB | 1,048 note events |

(Sizes from the 2026-04-23 corpus — unchanged from yesterday.)

## Outputs

Produced by the existing `midi_to_musicxml(quantize=True, stem_type=<name>, title="Alright", artist="Jamiroquai")`:

| File | Size | Parts | Measures | Clef | Time | Tempo | Notes |
|---|---:|---:|---:|---|---|---:|---:|
| `alright_piano.musicxml` | 1.65 MB | 1 (Piano) | 132 | Treble (grand staff in render) | 4/4 | 120 BPM | 2,002 |
| `alright_guitar.musicxml` | 0.91 MB | 1 (Acoustic Guitar) | 132 | Treble | 4/4 | 120 BPM | 820 |
| `alright_bass.musicxml` | 0.91 MB | 1 (Electric Bass) | 130 | Bass | 4/4 | 120 BPM | 1,048 |

Per-stem duration distribution (quantized) — top five:

```
piano:   eighth 597, quarter 497, 16th 496, half 320, whole 75
guitar:  16th 374,   eighth 371,  quarter 62, half 10,  32nd 3
bass:    eighth 520, 16th 450,   quarter 50, half 11,  whole 11
```

Sensible rhythmic vocabulary for a 120-BPM funk track. No 128th-note noise; the quantizer is doing its job.

Scratch dir: `/tmp/phase0_sheet/` (converter, the three `.musicxml` files, a minimal `view.html` that loads OSMD from CDN, and a local Python `http.server` recipe for local preview).

## Visual verification (OSMD render, Chrome)

Loaded `view.html` (an 50-line HTML page that points OSMD at each MusicXML in turn) from a local `http.server` on port 8765. Screenshots:

### Piano
- Header: "Alright - Piano" (title), subtitle "Alright - Piano", composer "Jamiroquai" aligned right.
- Tempo indicator ♩ = 120 at the start of measure 1.
- Part label "Piano" on the brace.
- Grand staff — treble + bass — rendered cleanly.
- Opening bars show rests, then eighth/quarter/half-note figures with a sustained chord and an `mp` dynamic marking.
- Visible at the top of measure 3: a new line starting with a rest and a flurry of sixteenths — matches Alright's right-hand piano figurations.
- Beams are grouped as expected (four sixteenths per beam, eighth pairs beamed).

### Guitar
- Header: "Alright - Guitar" title, composer "Jamiroquai".
- Part label "Acoustic Guitar".
- Treble clef, 1 flat (F major / D minor per music21's auto-detect — see "Known non-blockers" below).
- `p` dynamic on opening. Measure 1 mostly empty (correctly — guitar doesn't enter at bar 1), then a chord stack and single-note figures.
- Note stems and beaming are legitimate notation — a player could sight-read this.

### Bass
- Header: "Alright - Bass" with "Electric Bass" label.
- **Bass clef — correct.** Auto-assigned by the `_apply_instrument_settings` helper.
- 1 sharp key signature (G major per music21 — again, auto-detect).
- `f` dynamic. Measure numbers 3, 5, 7 visible across the first system.
- Bass figures sit below the staff with a low ledger-line region as expected (MIDI range 28-67, ~E1 to G4).
- Rhythmic figures look like a funk bass line — syncopated eighths and sixteenths with occasional quarter/half holds.

## Known non-blockers

These are cosmetic or first-pass issues that do NOT affect the gate:

1. **Key signatures auto-detected by music21 are wrong.** Alright is ground-truth Cm (3 flats); music21 returned F major for piano, D minor for guitar, G major for bass. Music21's Krumhansl-Schmuckler on single-voice MIDI is known to miss on modal/funk material. Phase 1 can pass the detected song key (which we now have from `chord_detector_v10` and the chart_formatter) into `midi_to_musicxml` as an override instead of relying on the auto-detect.
2. **Dynamics (mp/p/f) vary across stems.** Derived from MIDI velocity averages; not musically authored. Fine for launch — removes blank-page feel without being actively wrong.
3. **Title rendered twice** (as both title and subtitle). The `_apply_instrument_settings` pipeline appends the stem name to the title, and music21 then echoes it as a subtitle. Cosmetic — fix in Phase 1 if trivial.
4. **Piano part reported as a single Treble part in music21** but OSMD renders the grand staff because music21 writes voices spanning both clef ranges. Visual is correct; the library is doing the right thing automatically.
5. **Some measures look busy** (lots of sixteenths beamed together). This matches the actual busy-ness of a Jamiroquai funk track; it isn't noise. A real lead-sheet pass could simplify, but that's a tuning knob for later.

None of the above threaten the sheet-music feature. All are tunable in the existing pipeline.

## Gate decision

**PASS.** The MusicXML is readable as music, not random dots. Basic Pitch MIDI is sufficient quality for notation rendering. Proceed to Phase 1.

## Proceed to Phase 1

Next step:
- Wire Basic Pitch into the pipeline as a MIDI producer (behind `ENABLE_BASIC_PITCH_MIDI=false` default).
- Save per-stem MIDI to `outputs/<job_id>/midi/*.mid`.
- Populate `job.midi_files` so the existing `convert_midi_to_musicxml(job)` call triggers and writes MusicXML to `outputs/<job_id>/musicxml/*.musicxml`.
- If the MIDI detector is also on, share the cached Basic Pitch output — do not invoke it twice.

No code changes this phase.
