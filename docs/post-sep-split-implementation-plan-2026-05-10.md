# Post-Separation Semaphore Split — Implementation Plan

**Date:** 2026-05-10
**Author:** implementation-planning agent
**Predecessor doc:** `docs/scaling-pipeline-2026-05-09.md` §2 Recommendation #1
**Scope:** Concrete plan to release `_post_separation_semaphore` at `chord_chart.json` write so MIDI/MusicXML/Guitar Pro/Songsterr/email/URL-cache run outside the cap-4 slot.
**Audience:** Jeff and the next agent picking this up.

---

## 1. File-by-file change list

### `backend/processing/pipeline.py`

1. **Add module-level rollback flag** near the existing semaphore declaration (around `pipeline.py:40-41`). Read `os.environ.get('POST_SEP_RELEASE_AT_CHART', 'true').lower() in ('1', 'true', 'yes')` once at import time into a module constant `_RELEASE_SLOT_AT_CHART`. Default ON so the new behavior ships, but a `false` flip restores legacy behavior without a code change.

2. **Extract lines 574-692 into a new helper `_run_post_deliverable(job, audio_path, gp_tabs)` in `pipeline.py`** placed immediately above `process_audio` (current line 159). The helper does, in order: lead-sheet generation (formerly 574-589), MIDI transcription block (591-649), `job.progress = 100` / status flip / completed_at (651-657), email notification (663-669), URL cache (674-689), `save_job_to_disk(job)` (692). The helper takes its own `try/except` so a crash inside it sets `job.status = 'failed'`, sets `job.error`, logs to `error_tracker`, and **never re-raises** (caller is a daemon thread).

3. **Add a new helper `_release_slot_once(job, acquired_flag_holder)`** — see §2 below for the invariant. This is the single point that touches `_post_separation_semaphore.release()`.

4. **Modify `process_audio` (159-727)**:
   - Replace the local `post_sep_acquired = False` bool (161) with a 1-element list `slot_state = [False]` so the helper can mutate it from a nested scope.
   - Slot acquire (224-229) sets `slot_state[0] = True`.
   - **Immediately after the chord-chart write completes successfully (line 561)**, insert a new block, guarded by `if _RELEASE_SLOT_AT_CHART and slot_state[0]:`:
     - Set `job.status = 'ready_for_practice'` (new milestone — see §2 of the 2026-05-09 scaling doc, recommendation #1 option (a)). Call `save_job_checkpoint(job)` so the frontend sees it.
     - Call `_release_slot_once(job, slot_state)`. This flips `slot_state[0] = False` AND releases the semaphore.
     - Spawn a `threading.Thread(target=_run_post_deliverable, args=(job, audio_path, gp_tabs), daemon=True, name=f'post-deliv-{job.job_id}').start()`.
     - `return` from `process_audio` — the main thread is done.
   - The legacy path (everything 574-692 still inline) stays compiled in but only runs when `_RELEASE_SLOT_AT_CHART` is False (the rollback case). Wrap 574-692 in `if not _RELEASE_SLOT_AT_CHART:` rather than deleting, so the rollback flag truly restores prior behavior byte-for-byte.

5. **Modify the outer `finally:` (723-727)** so it calls `_release_slot_once(job, slot_state)` instead of inlining the release. The helper is idempotent: if `slot_state[0]` is already False, it does nothing. This handles every failure mode in §2.

6. **Make sure the `except Exception` block at 694-721** runs *before* slot release if the failure happens after the chord-chart write but before the thread spawn returns. The existing structure already handles this — the `finally` runs after `except`.

### `backend/models/job.py`

7. **Add a `_save_lock = threading.Lock()` module-level lock** (around line 28, next to `jobs = {}`). Wrap the body of `save_job_to_disk` (111-125) and `save_job_checkpoint` (128-140) in `with _save_lock:`. This prevents the main thread and the post-deliverable thread from interleaving writes to the same `job_metadata.json`.

8. **Add a `status_priority` helper** so the post-deliverable thread doesn't downgrade a status. If a user cancels a job (future feature) or the watchdog flips it to `failed`, the post-deliverable thread should not overwrite `failed` with `completed`. Map: `ready_for_practice < completed < failed`. Inside `_run_post_deliverable`, before setting `job.status = 'completed'`, check `if job.status != 'failed': job.status = 'completed'`.

### `backend/routes/api.py` (no edits — just verify)

9. Confirm `/api/status/<job_id>` returns `job.status` without filtering. The frontend already polls this and must learn to render the new `'ready_for_practice'` value. **Frontend change is out of scope of this doc** but flagged here as a dependency: `frontend/practice.html` and `frontend/progress.js` must accept `ready_for_practice` as a "show practice mode" status. Optional polish from §2 of scaling doc.

### `backend/tests/` (new + modified — see §3)

Files added: `test_post_sep_split.py`. Files modified: `test_pipeline.py` if any existing tests assert `job.status == 'completed'` immediately after `process_audio` returns — these now race with the post-deliverable thread.

---

## 2. Slot-leak guard pattern

**Invariant:** The semaphore is released **exactly once per `process_audio` invocation that acquired it**, regardless of:
- Success (chart written, post-deliverable spawned).
- Failure before chart write (separation crashes, chord detection raises, etc.).
- Failure inside the post-deliverable thread (MIDI crashes — irrelevant to the semaphore, since the slot was already released).
- Process kill (`kill -9`) — accepted to leak; systemd restart re-acquires fresh semaphore on import.

**Pattern:**

- `slot_state` is a 1-element list (mutable closure variable).
- `_release_slot_once(job, slot_state)` is the **only function** that calls `.release()`. Body, paraphrased: if `slot_state[0]` is True, set it to False and call `_post_separation_semaphore.release()`. Else no-op. Wrap in a `try/except` and log if release raises (it can if the semaphore is corrupted — extremely rare, but the log is forensic gold).
- The release happens **before the thread is spawned**, not inside the thread. The thread never touches the semaphore. This is the critical rule: if we released *inside* the thread's finally, a thread spawn failure (rare but possible — `RuntimeError: can't start new thread` under thread exhaustion) would leak the slot forever.
- The outer `finally` calls `_release_slot_once` too. Since it's idempotent, calling it twice (once at chart-write, once in finally) is safe. This is belt-and-suspenders.
- The post-deliverable thread sets `job.status` and `job.error` but **does not own the slot** and cannot release it.

**Failure modes covered:**
| Scenario | What releases the slot |
|---|---|
| Normal happy path | Inline release after chord-chart write |
| Exception during separation | Never acquired; nothing to release |
| Exception between acquire (229) and chart write (561) | Outer `finally` calls `_release_slot_once` |
| Exception inside `_run_post_deliverable` thread | Slot already released; thread crash logged only |
| Thread spawn fails (`RuntimeError`) | Outer `finally` catches; `_release_slot_once` is idempotent — but slot was already released inline, so no-op. The job is marked failed. |
| `kill -9` of Flask process | Slot leaks; accepted (systemd restart resets the semaphore via fresh import) |
| Anthropic corrector hangs forever | Slot held; covered by a separate timeout add (out of scope for this doc, flagged in scaling doc §4 #8) |

---

## 3. Test plan

New file: `backend/tests/test_post_sep_split.py`. All tests use `threading` directly and a fake `ProcessingJob` populated with the minimum fields the helpers read.

1. **`test_five_jobs_all_reach_chart_write_under_cap4`** — the headline test. Build 5 fake jobs. Mock `separate_stems_modal` to set 4 stems and return True instantly. Mock the heavy stages (Whisper word ts, chord detection, formatter) to sleep 0.5s and write a stub `chord_chart.json`. Mock `_run_post_deliverable` to sleep 5s. Spawn 5 worker threads calling `process_audio`. Assert: by t=2s, all 5 jobs have `chord_chart.json` on disk AND `job.status == 'ready_for_practice'`. (Pre-fix, slot 5 would not reach chart write until at least one of slots 1-4 finishes the full 5s, so it'd take >5s.)

2. **`test_slot_released_when_midi_crashes`** — Spawn 1 job. In the mocked `transcribe_to_midi`, raise `RuntimeError("synthetic midi failure")`. Assert: `_post_separation_semaphore._value == _POST_SEPARATION_MAX_CONCURRENT` after the post-deliverable thread joins. (Slot was released BEFORE thread spawn; thread crash cannot affect the semaphore.)

3. **`test_slot_released_when_chart_write_succeeds_but_lead_sheet_crashes`** — Same shape as #2 but make `generate_lead_sheet_for_job` raise. Assert: semaphore value is restored, job.status flips to `completed` anyway (lead sheet is already non-fatal in current code — verify the post-deliverable thread preserves that), no log of "slot leaked".

4. **`test_slot_released_when_chord_detection_crashes_before_chart_write`** — Failure in the in-slot section. Mock `detect_chords_for_job` to raise. Assert: outer `finally` triggers; semaphore restored; `job.status == 'failed'`; post-deliverable thread was NEVER spawned (check via a sentinel — e.g., mock `_run_post_deliverable` and `.assert_not_called()`).

5. **`test_status_transitions_no_race`** — Spawn 1 job. After chord-chart write, main thread sets `'ready_for_practice'`. Post-deliverable thread sets `'completed'`. Sample `job.status` 100 times from a third thread during the run; assert the sequence is monotone in the priority order `processing` → `ready_for_practice` → `completed` (never goes backward). No `'processing'` observation after `'ready_for_practice'`.

6. **`test_save_job_to_disk_lock_prevents_corruption`** — Two threads call `save_job_to_disk(job)` 100 times each with different `job.progress` values. After: `json.load(job_metadata.json)` must succeed (not be half-written / truncated). Without the lock added in change #7, this currently CAN produce a truncated file on the VPS under load — the test asserts the fix.

7. **`test_post_deliverable_does_not_downgrade_failed_status`** — Pre-set `job.status = 'failed'` (simulating watchdog or cancel) before the post-deliverable thread runs. Run `_run_post_deliverable(job, ...)`. Assert: `job.status == 'failed'` afterwards (the `status_priority` guard in change #8 prevents the downgrade).

8. **`test_rollback_flag_restores_legacy_behavior`** — Set `os.environ['POST_SEP_RELEASE_AT_CHART'] = 'false'` and re-import `pipeline`. Run a single job. Assert: slot is held through MIDI transcription (probe `_post_separation_semaphore._value` mid-MIDI; it should be 3, not 4). Assert no `_run_post_deliverable` thread was spawned.

9. **`test_thread_spawn_failure_does_not_leak_slot`** — Monkey-patch `threading.Thread` to raise on `.start()`. Run 1 job through to the spawn point. Assert: semaphore restored (outer finally fires + idempotent release), `job.status == 'failed'`.

10. **Existing test fixup** — grep `backend/tests/` for assertions of the form `assert job.status == 'completed'` immediately after a `process_audio` call. Each one needs to either (a) `time.sleep(N)` to wait for the post-deliverable thread, or better (b) poll `job.status` with a timeout. List them in the PR description.

---

## 4. Load-test procedure

**Goal:** 25 simultaneous uploads, all reaching `chord_chart.json` write within the time it takes 4 of them to finish the in-slot work — not the 25-of-25 worst case.

**Setup (local dev — Jeff's Mac):**

- `cd ~/stemscribe && ./venv311/bin/python backend/app.py` in one terminal (port 5555).
- Pre-stage 25 short test audio files (~1 min each) in `/tmp/load_test/song_{01..25}.mp3`. Reuse the audit corpus.
- Auth: use Jeff's beta JWT or hit a no-auth dev endpoint.

**Command:**

```bash
ls /tmp/load_test/*.mp3 | xargs -P 25 -I {} curl -s -X POST \
  http://localhost:5555/api/upload \
  -F "audio=@{}" -H "Authorization: Bearer $JWT" \
  > /tmp/load_test_results.jsonl
```

**Measurement script (separate terminal):** poll `/api/jobs` every 2s, log timestamps for each `job_id` when it first hits `ready_for_practice` and again when it hits `completed`. Plot the distribution.

**Pass criteria (all must hold):**
1. All 25 jobs reach `status == 'ready_for_practice'` (chord chart on disk).
2. The 5th job hits `ready_for_practice` within **30s** of the 4th job (proves slot 5 wasn't blocked behind a full post-deliverable pipeline — only behind the in-slot work).
3. No slot leak: after all 25 are `completed`, `_post_separation_semaphore._value == 4`. Verify via `/admin.html` or a debug endpoint that exposes the semaphore counter.
4. No truncated `job_metadata.json` files.
5. CPX41 stand-in: if testing on the VPS, RAM stays under 14 GB and no OOM-killer events in `dmesg`.

**Fail = rollback:** flip `POST_SEP_RELEASE_AT_CHART=false`, restart `stemscribe`, re-test to confirm legacy behavior. Diagnose.

---

## 5. Rollback plan

**Single env var:** `POST_SEP_RELEASE_AT_CHART` in `/opt/stemscribe/backend/.env` on the VPS (and same name in `.env.example` for the repo).

**Default:** `true` (new behavior ships ON).

**Rollback:** set `POST_SEP_RELEASE_AT_CHART=false`, run `systemctl restart stemscribe`. The conditional at the top of `pipeline.py` reads the env var once at import; restart re-reads it. Behavior reverts to the current code path byte-for-byte because change #4 above wraps the legacy 574-692 block in `if not _RELEASE_SLOT_AT_CHART:` rather than deleting.

**Wired in:** `pipeline.py` module scope, right next to `_POST_SEPARATION_MAX_CONCURRENT = 4` declaration (around line 40). One read at import — cheap, no per-job env lookup.

---

## 6. Risk register

Ranked by likelihood × impact. H/M/L scales.

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | `save_job_to_disk` race between main and post-deliverable threads → corrupt `job_metadata.json` | **H** | **H** (data loss, job vanishes) | `_save_lock` in `models/job.py` (change #7). Test #6 covers it. |
| 2 | Frontend polling sees `ready_for_practice` and breaks because it only knows `processing`/`completed`/`failed` | **H** | **M** (UI shows "unknown status") | Frontend tweak flagged in change #9. Until frontend ships, the rollback flag stays available. Alternative: skip introducing the new status, just release the slot and keep `processing` until the thread finishes (recommendation #1 option (b)). |
| 3 | Slot leak from a code path that doesn't go through `_release_slot_once` | M | **H** (deadlocks queue at 4 forever) | The pattern in §2 funnels every release through the helper. Test #9 + test #2. Add a dead-man systemd timer that logs the semaphore counter every 5 min and SMS-alerts if it stays at 0 with no active jobs for 15 min. |
| 4 | Post-deliverable thread crashes silently — user thinks job is `completed` but MIDI/GP files never appear | M | M | The helper has its own `try/except` → sets `job.status = 'failed'`, logs to `error_tracker`. Test #2. Frontend should show the deliverables it has + a "MIDI/GP failed to generate" badge if status flips to `failed` after `ready_for_practice`. |
| 5 | `job.status = 'completed'` from post-deliverable thread overwrites a `'failed'` set by a watchdog or future cancel feature | M | M | `status_priority` guard in change #8. Test #7. |
| 6 | Thread spawn fails under load (Python thread limit ≈ 1000 on Linux default) | L | M | At cap=4 + post-deliverable threads, we'd need ~hundreds of long-running threads to hit this. Test #9 covers the failure handling. Long-term fix is RQ (recommendation #3 in scaling doc). |
| 7 | `save_job_checkpoint` from post-deliverable thread overwrites stage info the main thread just wrote for a *different* job (impossible — they share the same `job` object) — but interleaving overwrites the SAME job's stage backward | L | L | The helper only writes its own `job.stage` values. Worst case: brief flicker in UI between "Generating MIDI" and "ready_for_practice" — cosmetic only. |
| 8 | RAM pressure from running 4× in-slot pipelines + N× post-deliverable threads simultaneously | M | M | Pre-fix, slot 5 waits. Post-fix, slot 5 can enter while 4 post-deliverable threads still chug. MIDI transcription per stem is ~1-2 GB peak. Worst-case 4 in-slot + 4 post = ~8-10 GB. CPX41 has 16. Watch the load test. If RAM is tight, add a *second* semaphore capping post-deliverable threads (e.g., cap 4) — defense in depth. |
| 9 | `audio_path` deleted by retention sweeper between chart write and MIDI transcription | L | L | Retention is 48h for uploads; post-deliverable finishes in minutes. Not realistic. |
| 10 | Anthropic corrector still holds the slot for minutes if API hangs | M | H | Out of scope of this doc but flagged in scaling doc §4 #8. Adding a 15s timeout to the corrector is a separate 1-line change that should ship the same week. |
| 11 | Memory cache eviction (`_evict_old_jobs`) removes a job whose post-deliverable thread is still running | L | L | Thread holds a Python ref to the `job` object; eviction only removes the dict key. `get_job` from a later request will reload from disk. Eviction only fires when `len(jobs) >= 100` AND there are `completed` jobs — and we just confirmed `ready_for_practice` is not in the evict set (`models/job.py:228` filters `status == 'completed'`). Safe. |
| 12 | `URL cache` write at the end of post-deliverable runs after retention deletes the output dir | L | L | URL cache writes a tiny JSON row, not a reference to the output dir. Safe. |

---

## Open questions for Jeff before coding

1. Two-stage `ready_for_practice` + `completed` (option a) vs. single `completed` flipped at the end while just releasing the slot earlier (option b)? Recommend (a) — bigger perceived-speed win, frontend change is small.
2. Should the dead-man semaphore monitor (risk #3 mitigation) be in this PR or a follow-up? Recommend follow-up — keeps this PR focused on the split itself.
3. Anthropic corrector timeout: bundle into this PR or ship as a separate one-liner? Recommend separate.
