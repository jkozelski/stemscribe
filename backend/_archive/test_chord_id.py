#!/usr/bin/env python3
"""
Test script for _notes_from_tab() and chord identification from tab data.

Validates fret-to-note conversion for:
  - Standard tuning chords (A major, E major, E7#9, etc.)
  - Open tuning support
  - Muted string handling
  - Integration with _identify_chord() for full chord-from-tab pipeline

Run:  cd ~/stemscribe && ./venv311/bin/python backend/test_chord_id.py
"""

import sys
import os

# Ensure the backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from routes.songsterr import (
    _notes_from_tab,
    _frets_to_notes,
    _identify_chord,
    NOTE_NAMES,
    DEFAULT_TUNING,
)

passed = 0
failed = 0


def check(label, actual, expected):
    global passed, failed
    if actual == expected:
        print(f"  PASS  {label}")
        passed += 1
    else:
        print(f"  FAIL  {label}")
        print(f"        expected: {expected}")
        print(f"        got:      {actual}")
        failed += 1


def notes_to_chord(tab_str, tuning=None):
    """Helper: run _notes_from_tab then _identify_chord."""
    notes = _notes_from_tab(tab_str, tuning=tuning)
    # Build (string_idx, fret) pairs for _frets_to_notes (needs internal tuning order)
    if tuning is None:
        tuning_low_to_high = [40, 45, 50, 55, 59, 64]
    else:
        tuning_low_to_high = list(tuning)
    # _frets_to_notes expects 0-indexed where 0=string1, so reverse
    internal_tuning = list(reversed(tuning_low_to_high))
    frets = []
    if isinstance(tab_str, str):
        chars = list(tab_str)
    else:
        chars = [str(f) for f in tab_str]
    for i, c in enumerate(chars):
        if c.lower() != 'x':
            # i is string index from low (6) to high (1)
            # internal index: string 6 = index 5, string 1 = index 0
            internal_idx = len(chars) - 1 - i
            frets.append((internal_idx, int(c)))
    midi_notes = _frets_to_notes(frets, internal_tuning)
    return _identify_chord(midi_notes)


# ────────────────────────────────────────────
print("=== _notes_from_tab: Standard Tuning ===")
# ────────────────────────────────────────────

# A major: x02220 (strings 6-5-4-3-2-1)
check(
    "A major (x02220)",
    _notes_from_tab("x02220"),
    ['x', 'A', 'E', 'A', 'C#', 'E'],
)

# E major: 022100
check(
    "E major (022100)",
    _notes_from_tab("022100"),
    ['E', 'B', 'E', 'G#', 'B', 'E'],
)

# C major: x32010
check(
    "C major (x32010)",
    _notes_from_tab("x32010"),
    ['x', 'C', 'E', 'G', 'C', 'E'],
)

# G major: 320003
check(
    "G major (320003)",
    _notes_from_tab("320003"),
    ['G', 'B', 'D', 'G', 'B', 'G'],
)

# D major: xx0232
check(
    "D major (xx0232)",
    _notes_from_tab("xx0232"),
    ['x', 'x', 'D', 'A', 'D', 'F#'],
)

# E7#9 (Hendrix chord): 076780 — but string 1 muted in practice, use 07678x
check(
    "E7#9 Hendrix (07678x)",
    _notes_from_tab("07678x"),
    ['E', 'E', 'G#', 'D', 'G', 'x'],
)

# Am: x02210
check(
    "Am (x02210)",
    _notes_from_tab("x02210"),
    ['x', 'A', 'E', 'A', 'C', 'E'],
)

# F barre: 133211
check(
    "F barre (133211)",
    _notes_from_tab("133211"),
    ['F', 'C', 'F', 'A', 'C', 'F'],
)

# All muted
check(
    "All muted (xxxxxx)",
    _notes_from_tab("xxxxxx"),
    ['x', 'x', 'x', 'x', 'x', 'x'],
)

# Open strings: 000000
check(
    "Open strings (000000)",
    _notes_from_tab("000000"),
    ['E', 'A', 'D', 'G', 'B', 'E'],
)

# ────────────────────────────────────────────
print("\n=== _notes_from_tab: Open G Tuning ===")
# ────────────────────────────────────────────
# Open G: D-G-D-G-B-D (strings 6-1)
OPEN_G = [38, 43, 50, 55, 59, 62]

# Open strum in Open G: 000000
check(
    "Open G strum (000000)",
    _notes_from_tab("000000", tuning=OPEN_G),
    ['D', 'G', 'D', 'G', 'B', 'D'],
)

# G major barre at fret 5 in Open G? No — open is already G major.
# Fret 5 across all = D-C-G-C-E-G (a C chord shape)
check(
    "Open G fret 5 barre (555555)",
    _notes_from_tab("555555", tuning=OPEN_G),
    ['G', 'C', 'G', 'C', 'E', 'G'],
)

# ────────────────────────────────────────────
print("\n=== Chord Identification from Tab ===")
# ────────────────────────────────────────────

check("A major chord ID (x02220)", notes_to_chord("x02220"), "A")
check("E major chord ID (022100)", notes_to_chord("022100"), "E")
check("C major chord ID (x32010)", notes_to_chord("x32010"), "C")
check("G major chord ID (320003)", notes_to_chord("320003"), "G")
check("D major chord ID (xx0232)", notes_to_chord("xx0232"), "D")
check("Am chord ID (x02210)", notes_to_chord("x02210"), "Am")
check("E7#9 chord ID (07678x)", notes_to_chord("07678x"), "E7#9")
check("F barre chord ID (133211)", notes_to_chord("133211"), "F")

# ────────────────────────────────────────────
print("\n=== Edge Cases ===")
# ────────────────────────────────────────────

# Single note (not a chord)
check("Single note → None", notes_to_chord("x0xxxx"), None)

# Power chord E5: 02xxxx
check("E5 power chord (02xxxx)", notes_to_chord("02xxxx"), "E5")

# Verify _notes_from_tab rejects wrong-length input
try:
    _notes_from_tab("012")  # only 3 chars for 6-string tuning
    print("  FAIL  Wrong-length tab should raise ValueError")
    failed += 1
except ValueError:
    print("  PASS  Wrong-length tab raises ValueError")
    passed += 1

# ────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
if failed:
    print("SOME TESTS FAILED")
    sys.exit(1)
else:
    print("ALL TESTS PASSED")
    sys.exit(0)
