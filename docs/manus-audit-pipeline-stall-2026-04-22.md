# Manus 1.7 Audit Briefing — Pipeline Stall Fix

**Date:** 2026-04-22
**Asker:** Jeff Kozelski (StemScriber founder, solo)
**For:** Manus 1.7 red-team review
**What to do:** read everything below, then stress-test the Proposed Fix and the Alternatives. Tell us what breaks, what we missed, what's premature. Launch is 20 days out (2026-05-12) so be ruthless about anything that makes the fix worse than the status quo.

---

## Project Context

**StemScriber** (stemscriber.com): consumer web app for musicians. Upload a song (MP3 or YouTube URL) → backend returns 6 stems (vocals, bass, drums, guitar, piano, other) + a chord chart + practice mode (per-stem mute/solo/volume, 50–200% speed, A-B loop).

**Launch scope for 2026-05-12:** stems + chord chart + practice mode. Tab/sheet-music view is CUT. Karaoke is CUT pending lyrics licensing.

**What's shipping** (verified working):
- BS-RoFormer-SW stem separation on Modal (A10G GPU, ~$0.06/song)
- Stem-aware chord detection — bass-root-first, BTC + V8 hybrid (337 classes), 95% quality post-April 16 bleed-simplification pass
- Rule-based chart formatter (Phi-3 fine-tune was abandoned April 16 after $24 burned on contaminated training data)
- Practice mode
- Upload consent popup, DMCA registration, Vapi voice agent "Alex"

**Stack:**
- Backend: Flask (Python 3.11), modularized into blueprints
- Frontend: static HTML/JS/CSS served by Flask
- Infra: Hetzner VPS ($8/mo, **4 CPU / 8 GB RAM**, NO GPU), Cloudflare Tunnel, Modal for GPU work
- Transcription: `faster-whisper medium @ int8` for word timestamps (CPU)
- Chord detection: `btc_finetuned_best.pt` + `v8_chord_model.pt`
- Jobs run in background threads inside the Flask process

---

## THE PROBLEM

We uploaded 4 test songs to verify a newly-shipped render-layer fix. All 4 jobs reached `Generating chord chart` at progress **59 %** — and then sat there for ~10 minutes while the background pipeline actually made real progress internally (Whisper ran, bar grid got built, sections got computed).

The **watchdog thread** (`backend/processing/watchdog.py`) checks every 30 s whether `job.progress` has changed. Its stall threshold was **600 s (10 min)**. When the number didn't move for 10 min, it fired a retry. The retry hit the exact same stall. Infinite loop.

Net effect: **4-out-of-4 jobs looped on retry forever, chord chart never delivered, launch regression test blocked.**

### Symptoms — actual journalctl excerpts

```
Apr 23 00:16:05 stemscribe python: INFO:faster_whisper:Processing audio with duration 05:10.627
Apr 23 00:16:05 stemscribe python: INFO:faster_whisper:VAD filter removed 02:58.915 of audio
Apr 23 00:20:30 stemscribe python: INFO:processing.pipeline:Got 237 word timestamps from vocal stem
Apr 23 00:20:30 stemscribe python: INFO:chart_formatter:chart_formatter: bar grid built (108 bars, 14 quality-smoothed)
Apr 23 00:24:51 stemscribe python: INFO:processing.watchdog:Watchdog [stall_detected] job=<id> {'stalled_at_progress': 59, 'stalled_at_stage': 'Generating chord chart', 'stalled_for_seconds': 629}
Apr 23 00:24:51 stemscribe python: INFO:processing.watchdog:Watchdog [retry] job=<id> {'retry_number': 1, ...}
```

- Whisper (5:10 song) took **4 min 25 s** on CPU with 4 concurrent jobs sharing
- Grid extraction ~45 s
- Bass-root extraction ~20 s
- Rule-based chart formatter itself ~7 s
- Total chord-chart phase: 5–6 min for a single song in isolation; under 4-way concurrency, frequently > 10 min

Watchdog's stall threshold fired before the phase completed, triggering retry, triggering stall again. Classic.

---

## Architecture at the Critical Code

Code layout (post-April-22 modularization):

```
backend/
├── app.py                               # Flask app factory
├── processing/
│   ├── pipeline.py                      # process_audio() main job flow
│   │   └── stages: Separation → Chord Detection → Chord Chart → Transcription
│   ├── watchdog.py                      # background stall-retry thread
│   ├── tempo_beats.py                   # librosa beat/downbeat grid
│   ├── bass_root_extraction.py          # pyin bass pitch → bar-root
│   └── separation.py                    # Modal separation entrypoint
├── chart_formatter.py                   # rule-based chord chart assembler
├── stem_chord_detector.py               # bass-root-first detection
├── chord_detector_v10.py                # BTC primary + V8 fallback
└── models/job.py                        # ProcessingJob, global jobs dict
```

The "Generating chord chart" stage was **one monolithic block** in `pipeline.py` that called, in order:
1. Grid extraction from drums stem
2. Bass-root extraction from bass stem
3. Whisper word timestamps from vocals stem (the slow one)
4. `chart_formatter.format_chart(chords, grid, bass_roots, words)`
5. `lead_sheet_generator` pass

…while `job.stage` stayed as the string `"Generating chord chart"` and `job.progress` stayed at `59` the whole time.

Meanwhile the watchdog at `watchdog.py:_check_jobs()` only watched `job.progress` (the integer) for change. Stage-text change was not considered.

---

## Proposed Fix (already implemented, commit `8128a2a`, not pushed)

### Change 1 — split the long phase in `pipeline.py`

Replace the monolithic 59% block with 5 distinct, monotonically-increasing stages:

| Stage text | progress |
|---|---|
| `Analyzing tempo and beats` | 60 |
| `Extracting bass roots` | 62 |
| `Transcribing lyrics` | 64 |
| `Generating chord chart` | 68 |
| `Generating lead sheet` | 70 |

Each sub-step sets both `job.stage` and `job.progress` before executing, so the UI sees real motion and the watchdog sees repeated changes.

### Change 2 — watchdog resets timer on stage change OR progress change

`watchdog.py:_check_jobs()` now resets `last_check` timestamp when **either** `job.progress` OR `job.stage` changes (not just progress). Defense in depth: future sub-steps that forget to bump `progress` still won't false-retry as long as the stage text changes.

### Change 3 — revert the band-aid

`STALL_THRESHOLD_SECONDS` goes back to **600 s** (was band-aid-bumped to 1800). No longer needed once progress actually advances.

### Change 4 — new regression test

`backend/tests/test_self_healing.py` adds a test covering "stage-change resets the stall timer." 369 pass (was 368).

### Files changed
```
backend/processing/pipeline.py
backend/processing/watchdog.py
backend/tests/test_self_healing.py        (new test case added)
```

### Deploy status
SCP'd to VPS at `/opt/stemscribe/backend/processing/` 00:34 UTC. **Service not yet restarted** — 4 test jobs are still mid-flight using the in-memory band-aid threshold (1800 s) so they won't false-retry. Jeff restarts when queue clears.

---

## Alternatives Considered + Why Rejected

### A) Callback-through-`format_chart()`

Pass a progress callback `(stage: str, pct: int) -> None` into `chart_formatter.format_chart()`; every internal step inside the formatter updates `job.progress`.

**Rejected because:**
- Would require threading a callback through every `_quantize_chords_to_bars`, `_bar_grid_to_chord_events`, `_consolidate_chord_events`, `_simplify_uncommon_quality`, `_rebuild_section_chords_from_bar_grid` helper. Signature change propagates.
- `format_chart()` itself only takes ~7 s of the 10-min phase. The slow work (Whisper, grid, bass roots) already lives *outside* `format_chart()` in `pipeline.py`, where it's easy to split at the pipeline layer.
- Heavier change, more regression risk 20 days before launch.

### B) Just bump the threshold forever

Keep `STALL_THRESHOLD_SECONDS = 1800` (30 min) and call it done.

**Rejected because:**
- Masks stalls. If Whisper genuinely hangs (model load failure, subprocess deadlock), we now wait 30 min instead of 10. The watchdog exists specifically to unstick hangs — blunting it for "legitimate slow work" means real hangs go undetected.
- Tech debt that'll bite us when we add the next long stage (Modal separation of a 20-minute podcast episode?).

### C) Kill the watchdog entirely

Remove the stall-retry mechanism.

**Rejected because:**
- It's caught real hangs in prod before (Apr 15-16 OOM kills where jobs froze — watchdog restarted them cleanly). Useful safety net.

---

## Risks / Caveats (our own red-team)

1. **Hardcoded progress values.** If Whisper is skipped (e.g., instrumental with no vocal stem), progress jumps `62 → 68` directly. Forward motion = watchdog still happy, but the UI briefly shows `Transcribing lyrics 64%` even when no lyrics are being transcribed. Minor UX bug, not a correctness bug.

2. **Exception path doesn't advance progress.** If `get_word_timestamps()` throws, the `Transcribing lyrics` stage is set but the caught exception returns control to the outer `try`/`except` without advancing to 68. The next stage change (`Generating lead sheet` 70) still breaks the stall — safe, but the UI sits at 64% for a moment during recovery.

3. **Coupling.** `pipeline.py` now "knows" that `format_chart()` takes ~7 s vs Whisper's ~4 min. If Whisper moves inside `format_chart()` later (via some refactor), the progress split becomes stale and we're back to a monolithic 59%. Mitigation: comment in both files cross-references the coupling. Better long-term mitigation: callback approach (A).

4. **Concurrency limit unaddressed.** The real underlying issue is that **4 concurrent jobs share 4 CPUs on 8 GB RAM**. Whisper takes 2× longer under contention. We haven't added a concurrency cap (e.g., `BoundedSemaphore(2)` for chart-gen). The fix addresses the *symptom* (false-retry) not the *cause* (CPU contention). For launch, we accept this — most user sessions are one upload at a time. Post-launch work: serialize chord-chart phase or add a concurrency cap.

5. **Not pushed + not active.** Code is committed locally + SCP'd to VPS, but the live Python process is still running the OLD code. Until Jeff runs `systemctl restart stemscribe` (with queue empty), the fix is dormant.

---

## Open Questions for Manus

Please specifically try to break this:

1. **Is there a race condition** between the main processing thread writing `job.progress`/`job.stage` and the watchdog thread reading them? The watchdog is a separate daemon thread and `ProcessingJob` attrs are not behind a lock. Python's GIL makes single-attr reads/writes atomic in practice but we'd love a second opinion.

2. **Are the progress values future-proof?** 60/62/64/68/70 assumes the next stage (post-chord-chart, currently "Finalizing outputs" at ~85%) doesn't collide. Confirm from the pipeline.py linear flow.

3. **Stage-change-resets-timer** — can you construct a realistic scenario where stage text changes rapidly (fast oscillating) masking a true hang? E.g., retries inside Whisper's subprocess that flip stage to "Retrying Whisper" and back?

4. **What would a truly hung Whisper subprocess look like** after this fix? Would the watchdog ever catch it? The subprocess doesn't write to `job.stage` from inside its own process — only the calling Python thread does. So if the caller is blocked on `subprocess.wait()` forever, we'd be stuck in `Transcribing lyrics` at 64% with no stage change. MAX_RETRIES = 3 still bounds it after 30 min though.

5. **Is the 600 s threshold even right now?** With the split, the longest single stage is `Transcribing lyrics` (~5 min under contention). 600 s = 10 min gives 2× headroom. Too tight? Too loose?

6. **Anything in `chart_formatter.py` we should have split further?** We didn't touch chart_formatter internals because the rule-based formatter is fast. Are we missing a latent slow path there?

7. **The existing `MANUS_AUDIT_BRIEFING.md`** at the repo root is stale (references Demucs and Python 3.14, neither of which is current). We plan to archive it. Is there any risk to the audit process if old briefings linger and new ones don't obsolete them?

---

## How to grade this audit

Please respond with:
- **BLOCKING issues** — ship-stoppers. We fix before deploying.
- **Non-blocking issues** — worth addressing post-launch. Add to punch list.
- **Alternatives we should reconsider** — did we miss a better fix?
- **Meta-feedback** — was the briefing itself good? What context did we leave out?

Ground truth: it's 20 days to launch, solo founder, one-man show, already-hardened codebase with real paying-attention-to-code debt. Perfect is the enemy of shipped.
