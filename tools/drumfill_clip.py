#!/usr/bin/env python3
"""Guess the Fill: slice a window from a drums MIDI and render it to audio.

Usage:
  drumfill_clip.py <job_id-or-midi-path> [--start S --end S | --auto] [--out NAME]

  --auto       pick the densest 7-second window (usually the fill/busiest part)
  --start/--end  explicit window in seconds
  --out        output basename (default: fill_<jobprefix>)

Outputs <out>.mid, <out>.wav, <out>.mp3 in /opt/stemscribe/gtf/.
"""
import argparse, os, subprocess, sys
from pathlib import Path

import pretty_midi

SF2 = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
OUTDIR = Path("/opt/stemscribe/gtf")


def find_midi(arg: str) -> Path:
    p = Path(arg)
    if p.exists():
        return p
    cand = Path(f"/opt/stemscribe/outputs/{arg}/midi/drums_transcribed.mid")
    if cand.exists():
        return cand
    sys.exit(f"No MIDI found for '{arg}'")


def densest_window(notes, span=7.0):
    """Return start time of the window containing the most notes."""
    if not notes:
        sys.exit("MIDI has no notes")
    starts = sorted(n.start for n in notes)
    best_start, best_count = starts[0], 0
    j = 0
    for i, t in enumerate(starts):
        while starts[j] < t - span:
            j += 1
        count = i - j + 1
        if count > best_count:
            best_count, best_start = count, max(0.0, t - span)
    return best_start


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--start", type=float)
    ap.add_argument("--end", type=float)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--tail", type=float, default=1.5, help="ring-out seconds")
    a = ap.parse_args()

    midi_path = find_midi(a.source)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    drums = [i for i in pm.instruments if i.is_drum] or pm.instruments
    notes = sorted((n for i in drums for n in i.notes), key=lambda n: n.start)

    if a.auto or a.start is None:
        start = densest_window(notes)
        end = start + 7.0
        print(f"auto window: {start:.1f}s – {end:.1f}s")
    else:
        start, end = a.start, a.end if a.end else a.start + 7.0

    sel = [n for n in notes if start <= n.start < end]
    if not sel:
        sys.exit(f"No notes in window {start}-{end}s")
    print(f"{len(sel)} notes in window")

    out_pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for n in sel:
        inst.notes.append(pretty_midi.Note(
            velocity=max(n.velocity, 55),  # floor quiet ghost notes so the render is audible
            pitch=n.pitch,
            start=n.start - start,
            end=max(n.end - start, n.start - start + 0.05),
        ))
    out_pm.instruments.append(inst)

    OUTDIR.mkdir(exist_ok=True)
    base = a.out or f"fill_{Path(a.source).name[:8]}"
    mid = OUTDIR / f"{base}.mid"
    wav = OUTDIR / f"{base}.wav"
    mp3 = OUTDIR / f"{base}.mp3"
    out_pm.write(str(mid))

    dur = (end - start) + a.tail
    subprocess.run(
        ["fluidsynth", "-ni", "-g", "1.4", "-F", str(wav), "-r", "44100", SF2, str(mid)],
        check=True, capture_output=True,
    )
    # trim to length, normalize, gentle fade-out
    subprocess.run(
        ["sox", str(wav), str(wav) + ".t.wav", "trim", "0", f"{dur}", "gain", "-n", "-1", "fade", "t", "0", f"{dur}", "0.4"],
        check=True, capture_output=True,
    )
    os.replace(str(wav) + ".t.wav", wav)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-b:a", "192k", str(mp3)],
        check=True,
    )
    print(f"WROTE {mid}\nWROTE {wav}\nWROTE {mp3}")


if __name__ == "__main__":
    main()
