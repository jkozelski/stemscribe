# Diagnostic Ground-Truth Set

Bench of real songs used to honestly score the chord detector. Each fixture
isolates ONE failure mode ("diagnostic axis") so a regression shows up on the
axis it actually broke instead of being averaged away.

**Rebuilt 2026-05-19.** The original untracked diagnostic GT set was lost to a
git-clean incident. Seven songs were reconstructed from scratch against >=3
independent reputable chart sources each (Ultimate Guitar, Songsterr,
Hooktheory, Heartwood, e-chords, lesson sites). Three fixtures survived the
incident and were NOT touched: Positively 4th Street, In My Life, Don't Know
Why.

## Conventions

- **Concert pitch, `capo: 0`** for every fixture (the convention of the three
  survivors). Where a song is universally played capo'd, the concert key is
  stated in `_recovery_note` and the chords are written at concert pitch.
- **Schema:** `{song, artist, key, capo, _recovery_note, sections:[{name,
  lines:[[bar,...],...]}]}` — one line cell == one bar. Identical to the
  surviving fixtures and to what both scorers flatten.
- **Validation floor:** every fixture scores **1.0 on every axis of both
  scorers GT-vs-GT** (verified — see "Validation" below). A fixture that
  cannot score itself 1.0 is broken.

## Scoring commands

v1 (bag-of-chords, vocabulary signal):

```
./venv311/bin/python audit/score_chord_chart.py \
  audit/fixtures/ground_truth/<slug>.json <detector_chart.json> --json
```

v2 (order-aware root / placement / flavor axes — the honest scorer):

```
./venv311/bin/python audit/score_chord_chart_v2.py \
  audit/fixtures/ground_truth/<slug>.json <detector_chart.json>
```

`<detector_chart.json>` is the served `chord_chart.json` for the song's audio
run through the pipeline (scp from `/opt/stemscribe/outputs/<job_id>/` on the
VPS, or any local path). With GT only, v2 mechanics validate via GT-vs-GT.

## The bench

### Rock-solid (root + key + the diagnostic flavor are authoritative)

| Song | Artist | Recording pinned | Key/Capo | Diagnostic axis |
|---|---|---|---|---|
| Free Fallin' | Tom Petty | Full Moon Fever (1989) studio | F / capo 0 | **Suspensions** — IV/V voiced as sus2/sus4 (concert `Bbsus2`/`Csus4`); a detector that flattens sus to plain triads must lose the flavor axis here |
| Folsom Prison Blues | Johnny Cash | 1955 Sun Records studio single | F / capo 0 | **Dominant-7 blues/shuffle** — 12-bar, V is `C7`; dropping the 7th is the documented librosa failure this exposes |
| House of the Rising Sun | The Animals | 1964 MGM single (Alan Price arr.) | Am / capo 0 | **Minor key + borrowed major IV/V** — `D` (not Dm) and `E` (not Em) in A minor; detector must not "correct" them to diatonic minors |
| A Whiter Shade of Pale | Procol Harum | 1967 Deram single | C / capo 0 | **Descending-bass slash chords** — `C C/B Am Am/G F F/E Dm Dm/C ...`; slash is scorer-directional (weight 0) so this stresses root stability under a moving bass |
| Friend of the Devil | Grateful Dead | American Beauty (1970) studio | G / capo 0 | **Walking bass under a static chord** — Garcia's descending G run; the passing bass tones are deliberately NOT charted as chords (the detector trap) |
| Sultans of Swing | Dire Straits | Dire Straits (1978) studio | Dm / capo 0 | **Long instrumental outro** — extended Knopfler solo coda vamping `Dm C Bb C` to the fade with no vocal anchor |
| Hallelujah | Jeff Buckley | Grace (1994) studio | C / capo 0 | **Loose / rubato tempo** — Buckley's free no-click delivery; a bar-locked detector smears chords across the elastic timing. `E7` (V/vi, "the minor fall, the major lift") is the load-bearing flavor |

### Directional caveats (root/key authoritative; the noted dimension is NOT)

- **All 7:** section/verse/chorus *repeat counts* are directional. Bar-indexed
  GT has no timestamps; the harmonic blocks (one verse cycle, one 12-bar, the
  outro progression) are authoritative, total length is an approximation. v2's
  `hold_invariant` placement read is the recommended one for these (it is
  repeat-count-invariant); `strict_bar` will read low purely from length
  mismatch, which is expected and stated in each `_recovery_note`.
- **Free Fallin':** the record is one I-IV-I-V cycle; the sus character lives on
  IV/V (concert `Bbsus2`/`Csus4`). The "D / Dsus4" strum-variant some charts
  show is a different acoustic voicing, not the recording's harmony.
- **House of the Rising Sun:** turnaround `E` vs `E7` — sources split; plain
  `E` chosen to stay conservative.
- **A Whiter Shade of Pale:** slash spelling — sources split between the
  guitar C-side reading (`C/B`, `Am/G`) and the organ-line reading (`Em/B`,
  `C/G`) of the *same* descending bass. C-side encoded; slash is scorer-
  directional anyway (flavor weight 0).
- **Sultans of Swing:** outro is a long fade — vamp-repeat count is directional
  by design (8 `Dm C Bb C` cycles written to exercise the long-instrumental
  axis).

### Excluded

None. All 7 target songs have an unambiguous concert key and root-level
progression backed by >=3 independent sources, so none were dropped for
root-level ambiguity.

## Validation (2026-05-19)

All 7 rebuilt fixtures scored GT-vs-GT through both scorers:

- v1: `root`, `full`, `pcs` F1 = **1.0** for every fixture
- v2: `root`, `placement.strict_bar`, `placement.hold_invariant`,
  `flavor.weighted_flavor`, `composite` = **1.0** for every fixture
- `backend/tests/test_score_chord_chart_v2.py` (8 tests) +
  `test_score_chord_chart_v1_recovery.py` still pass — no regression.

See `MANIFEST.json` for the machine-readable index.
