# Pipeline + Concurrency Scaling Plan — Pre-Refinery Soft Launch

**Date:** 2026-05-09
**Author:** scaling-research agent
**Audience:** Jeff (solo dev) + future agents picking up this work
**Trigger:** 2026-05-09 17-song audit. Modal separations finished fast in parallel; the post-separation slot semaphore (cap=4) stayed locked through MIDI/MusicXML stages (~95% progress) while chord chart was already on disk by ~68%. Result: 20+ min of "Queued for processing 40%" backup on 17 songs. Refinery soft launch is 2026-06-20 — backstage cohort could push 50–150 concurrent uploads. Current architecture dies at that load.
**Scope:** Recommendation only. No code changes outside this doc.

---

## §1 — Bottleneck Diagnosis

### What actually runs serially today

The pipeline has **two** concurrency primitives, in order:

1. **`_separation_semaphore` (cap = 1)** at `processing/separation.py:54`. Bypassed entirely on the Modal cloud-GPU path: `separate_stems_modal` (`separation.py:693-762`) does NOT acquire it (comment at `:701` says "No semaphore needed — this does not use local GPU resources"). Modal is the production path (`pipeline.py:189-192`). This semaphore is effectively dead in prod.

2. **`_post_separation_semaphore` (cap = 4)** at `pipeline.py:40-41`. Acquired at `pipeline.py:224-229`, released in `finally` at `pipeline.py:723-727`. Held through **everything** that happens after Modal returns stems.

The lock window for the post-separation semaphore covers, in order, all of these stages (`pipeline.py:230-657`):

| # | Stage | File:line | Wall time on CPX41 (est.) | CPU/IO profile |
|---|-------|-----------|--------------------------|----------------|
| a | Stem enhancement (off by default) | `pipeline.py:232-241` | 0 | — |
| b | Vocal lead/backing split (BS-Roformer, audio-separator local) | `pipeline.py:245-268` | **30–90s** | Heavy CPU + RAM (loads a full model) |
| c | Guitar lead/rhythm split (MelBand-RoFormer local) | `pipeline.py:271-315` | **30–60s** | Heavy CPU + RAM |
| d | Smart deep extraction | `pipeline.py:319-329` | 5–60s | Variable |
| e | Vocal onset detection (ffmpeg + numpy) | `pipeline.py:333-377` | 2–5s | Light CPU |
| f | Tempo + beat grid extraction | `pipeline.py:390-408` | 5–15s | Light CPU |
| g | Bass root extraction | `pipeline.py:418-432` | 10–20s | Light CPU |
| h | **Chord detection** (librosa now, was stem-aware) | `pipeline.py:438-456` | 20–60s | Medium CPU |
| i | Whisper word timestamps (faster-whisper int8 CPU) | `pipeline.py:476-486` | **120–300s** | Heaviest CPU stage |
| j | Chart formatter (rule-based) | `pipeline.py:528-537` | 5–10s | Light CPU |
| k | **Anthropic chord corrector** (network) | `pipeline.py:546-549` | 5–20s | **IO-bound, sleeps a thread** |
| l | Chart write to disk → user-visible chord_chart.json | `pipeline.py:551-561` | <1s | IO |
| m | Lead sheet (music21 MusicXML, bass only) | `pipeline.py:574-589` | 10–30s | Medium CPU |
| n | **MIDI transcription per stem** (`transcribe_to_midi`) | `pipeline.py:594` → `transcription.py:181` | **60–180s** | Medium CPU, fans out per stem |
| o | MIDI → MusicXML (bass only per `transcription.py:1017-1019`) | `pipeline.py:599` → `transcription.py:967` | 5–15s | Light CPU |
| p | MIDI → Guitar Pro | `pipeline.py:604-614` | 10–30s | Light CPU |
| q | Songsterr pro-tab fetch | `pipeline.py:619-647` | 5–30s network | IO-bound |
| r | Set status='completed', save, notify | `pipeline.py:651-692` | <1s | IO |

**The user-visible chord chart is written at step (l).** Everything after (l) — lead sheet, MIDI transcription per stem, MusicXML, Guitar Pro, Songsterr — is **post-deliverable enrichment** that holds the slot.

Stages (n) and (i) together are roughly **half the slot-held wall time**. Stage (i) has to be inside the slot because the chart needs lyrics. Stage (n) doesn't.

### What "success" looked like in the audit

For a 4-min song on Modal:
- t=0 → upload accepted, thread starts
- t≈30s → Modal stems return (steps 1–2 of pipeline complete, slot acquired)
- t≈30s + 120–200s = **t≈150–230s** → chord_chart.json on disk, user could practice
- t≈230s + 60–180s = **t≈290–410s** → MIDI/MusicXML/GP done, slot released, status flips to "completed"

So the slot is held for ~3–4 minutes per job after Modal returns, but the user-facing primary deliverable (chord chart + practice-mode) is ready about **halfway through** the slot hold. With cap=4 and 17 songs, that's ceil(17/4) × ~3.5 min = ~15 min of theoretical queue, lining up with the observed 20+ min once you add per-job variance and the CPX41's CPU saturating during the lead/backing split stages on adjacent jobs.

### Why this dies at 50–150 concurrent

At launch-day load with cap=4:
- 50 jobs: ceil(50/4) × 3.5 min ≈ **44 min** for the last job to even start its post-separation work, then another 3.5 min to finish — **~47 min total queue tail**.
- 150 jobs: ~131 min queue tail (**>2 hours** to drain).

The user-facing UI just spins on `Queued for processing 40%` (`pipeline.py:225`). There is no ETA shown, no queue-position number, and no "you'll get an email when it's ready" cue. People will refresh, hit upload again, and double the queue.

### Architecture risks beyond throughput

1. **In-process state is the ONLY source of truth for live jobs.** `jobs[job_id] = job` in `models/job.py:28` and `routes/api.py:180`. The dict lives in the Flask process. If the systemd unit restarts (deploy, OOM kill, crash):
   - Jobs that were mid-processing have `save_job_checkpoint` written every few stages (`pipeline.py:164, 175, 226, 249, 275, 321, 393, 421, 478, 495, 580, 593`), so disk has the LAST stage transition, but the worker thread is dead.
   - There is **no resume-on-restart logic.** `get_job` (`models/job.py:239-265`) auto-loads from disk if missing — but only on demand, and it doesn't restart processing. The job will sit forever at whatever stage it was at, status="processing", until 7d retention deletes it.
   - The user gets no error, no notification, just a job that never finishes.

2. **`threading.Thread` with `daemon=True`** (`routes/api.py:231-233`, `:387-389`). On a graceful systemd restart, daemon threads get killed mid-step. Same outcome as #1.

3. **Memory cache caps at 100 jobs** (`models/job.py:27, MAX_CACHED_JOBS = 100`). At 150 concurrent on launch day, the eviction path (`_evict_old_jobs` referenced from `:236`) fires while jobs are still active. If an active job gets evicted, the in-memory `job` reference held by the worker thread is fine (Python ref counts), but `get_job` from another request might re-load a stale copy from disk and the two go out of sync. Low-probability, but a real footgun under load.

4. **No CPU/RAM guard on stages (b) and (c).** The vocal-split and guitar-split steps load full audio-separator models inside the slot. With 4 simultaneous jobs each in stage (b) + Modal also returning new jobs hitting the slot wait queue, the box goes RAM-tight. CPX41 has 16 GB; each loaded BS-Roformer instance is ~3–4 GB. We've never observed the OOM but we've also never load-tested at 4× simultaneous (b)+(c).

5. **The Anthropic corrector has no timeout shown in the call site** (`pipeline.py:546-549`). If the Anthropic API hangs, the slot is held until whatever the SDK's default timeout is. Worth verifying — at 150 concurrent, even one stuck call wastes a slot for minutes.

---

## §2 — Top Recommendations (ranked)

### #1 — Move post-deliverable stages OUT of the post-separation semaphore

**Hypothesis.** Steps (m–r) after `chord_chart.json` is written do not need the 4-slot lock. Release the slot at line `pipeline.py:561` (right after the JSON is written), and let MIDI/MusicXML/GP/Songsterr run in a separate thread without the lock. Effective throughput doubles — at minimum.

**What goes in the slot (must, because of CPU+RAM contention):**
- Vocal split (b) — heavy CPU+RAM, local model
- Guitar split (c) — heavy CPU+RAM, local model
- Smart extraction (d) — variable but CPU-heavy
- Whisper word timestamps (i) — heaviest CPU stage, needed for chart
- Grid + bass roots + chord detection (f, g, h) — chart depends on these
- Chart formatter + Anthropic correction (j, k) — produces the user deliverable
- **Release here.**

**What moves OUT of the slot (post-deliverable enrichment):**
- Lead sheet (m) — bass-only, music21, ~10–30s CPU
- MIDI transcription (n) — per-stem, ~60–180s CPU, the biggest win
- MIDI → MusicXML (o) and MIDI → GP (p) — depend on (n)
- Songsterr fetch (q) — pure network IO
- Status flip + email notify (r) — must wait until enrichment is done OR we mark the chord-chart deliverable as a separate "ready" milestone

**Two ways to model the user-visible status:**

(a) **Two-stage completion.** Add `job.status = 'ready_for_practice'` at the slot release point. Practice mode loads from `chord_chart.json` only — UI can already display this. Keep `'completed'` as the final flip after MIDI/GP/MusicXML are done. Gates that fire on `'completed'` (email notification at `pipeline.py:663-669`, retention timestamp at `:656`, URL cache write at `:674-689`) stay where they are. The UI checks `status in ('ready_for_practice', 'completed')` to show practice mode, and shows a tiny "Generating MIDI/Guitar Pro… (optional, you can practice now)" badge. Best UX. Frontend work: ~2 hr.

(b) **Just release the slot, leave status='processing' until the end.** Simpler. UI doesn't change. But the user still sees "processing 70%" while practice mode would already work. We get the queue-throughput win but lose the perceived-speed win.

**Time estimate.** 4–6 hr including:
- Refactor `pipeline.py` to release the semaphore at the chart-written point and run (m–r) outside it.
- Decide (a) vs (b). Recommend (a).
- Make sure `finally:` block doesn't double-release the slot.
- Add tests for the new state machine. The 484-test suite must stay green.

**Expected throughput improvement.** ~2× sustained throughput on the slot, maybe more. Stages (m–r) total ~90–250s of slot-holding wall time. Removing them turns a 3.5-min slot hold into ~1.5–2 min. Cap=4 → effective concurrent steady-state ~6–8 jobs/min instead of ~3. At 50 concurrent: queue tail drops from ~47 min to ~20 min. At 150 concurrent: from ~131 min to ~55 min. Still not great at 150, but launch-day survivable, and recommendation #2 stacks.

**Risk.**
- A bug in the slot-release split could cause a slot leak (acquired and never released), which deadlocks the queue at 4 forever. Tests + a dead-man timer (release if held > 10 min) mitigate.
- Post-deliverable thread crashes silently. Already partially the case; just need to make sure errors are still logged + tracked via `error_tracker`.
- Status-machine tests downstream of "completed" (notifications, retention, URL cache) need to still fire. Easy to verify.

**Dependencies.** None. Self-contained change in `pipeline.py` plus optional UI tweak.

---

### #2 — Increase post-separation cap from 4 → 6, behind a config flag

**Hypothesis.** Cap=4 was set on 2026-04-25 when stages (n) and (i) were the bottleneck and we observed watchdog-stall at cap=6 on CPX41. Once recommendation #1 ships (MIDI moves outside the slot), the heavy-CPU stages remaining inside are (b), (c), and (i). With the lighter slot, cap=6 should fit on 8 vCPU / 16 GB, leaving ~1.3 vCPU per slot — enough for librosa + faster-whisper int8.

**Time estimate.** 1 hr including a load test. Add `POST_SEPARATION_MAX_CONCURRENT` env var, default 4, set to 6 on the VPS, monitor RAM + CPU during a 12-song re-audit.

**Expected throughput improvement.** ~1.5×. Stacks with #1: 50 jobs at cap=6 + 1.5–2 min per slot ≈ 13 min queue tail. 150 jobs ≈ 38 min.

**Risk.** RAM pressure during simultaneous (b)+(c). Mitigation: if RAM > 14 GB sustained, cap stays at 4 and we either:
  - Drop the local guitar-split (c) — it's a marginal feature, gated by `GUITAR_SEPARATOR_AVAILABLE`. Could move to Modal too, see #4.
  - Run lead/rhythm + lead/backing splits on Modal instead of the VPS (single Modal function, sequential per call, ~$0.02 extra each).

**Dependencies.** #1 ships first.

---

### #3 — Add a persistent queue layer (Redis + RQ) with disk-backed job state

**Hypothesis.** The in-process `jobs` dict + daemon threads are not safe for production load. A restart loses every in-flight job. RQ (Redis Queue) is the lowest-friction persistent-queue option for Flask — minutes-not-days to add, doesn't require Celery's beat scheduler / broker complexity, runs on a single Redis instance.

**What changes:**
- One Redis instance on the VPS (or Hetzner managed Redis if available; Upstash free tier also works at this scale, ~50 ops/sec).
- `routes/api.py` `upload_audio` enqueues a job to RQ instead of calling `threading.Thread(target=process_audio, ...)`.
- One or more RQ workers run as separate systemd units. Each worker pulls one job at a time and runs `process_audio` in-process.
- `_post_separation_semaphore` becomes irrelevant — concurrency is set by the RQ worker count (e.g., 4 workers = 4 simultaneous jobs after-Modal). The semaphore stays as defense-in-depth.
- Job state lives in Redis (RQ status + custom hash for progress/stage), and on disk (existing `save_job_checkpoint`). Frontend `/api/status/<job_id>` reads from Redis (fast slim path) or disk (slower full path).
- Worker crashes → RQ requeues the job. Restart-safe by construction.

**Time estimate.** 1.5–2 days. Including:
- Install Redis on VPS (~30 min).
- `pip install rq`, write `worker.py`, define `process_audio_task(job_id)` wrapper.
- Migrate `upload_audio` and `process_url_endpoint` enqueue paths.
- Update `/api/status` to read job state from Redis + disk fallback.
- systemd unit for `rq worker` × N processes. Restart policy.
- Test: graceful shutdown mid-job, kill -9 worker, full VPS reboot — all should resume.

**Expected throughput improvement.** Same throughput as #1 + #2 (this is about reliability, not speed). At 150 concurrent: zero jobs lost on a deploy, zero ghost-stuck jobs at "processing 70%" forever.

**Risk.**
- RQ has known issues with long-running jobs (>30 min) — the default worker timeout needs to be raised. Easy.
- Forking-mode workers don't play nicely with PyTorch/MPS on Mac dev, but production is Linux + we're using Modal for the GPU stage anyway, so this doesn't bite.
- One extra system-level dependency (Redis) to monitor and back up. Backups: Redis on the same VPS that's already snapshotted by Hetzner backups (€5/mo, already paid). Monitoring: simple `redis-cli ping` in the existing queue-monitor systemd timer.

**Dependencies.** None mechanically, but should ship after #1 so the migration target is the cleaner pipeline shape.

---

### #4 — Move local CPU-heavy stages (b, c) to Modal

**Hypothesis.** Vocal split and guitar split are the hungriest in-slot CPU+RAM consumers. Modal already runs the primary separation; bundle the lead/backing and lead/rhythm splits into the same Modal function (or two new ones) so they run on the A10G alongside, and the VPS only does Whisper, librosa chord detection, formatter, and chart write inside the slot.

**Time estimate.** 1–1.5 days. Including:
- Add `split_lead_backing_vocals` and `split_lead_rhythm_guitar` Modal functions. Same image as `stemscribe-separator`, just additional model loads.
- Modify `pipeline.py:245-315` to call Modal versions when MODAL_AVAILABLE.
- Audit: extra Modal cost per song = +$0.02–0.04 (two more A10G calls of ~10–20s each). At 5,000 songs/mo that's +$100–200/mo. Within the $50–200 budget if we're lucky on volume; over budget at 10,000+/mo.

**Expected throughput improvement.** Big effect on RAM headroom — frees ~6–8 GB of VPS RAM during peak. Allows cap to safely hit 8 instead of 6. Combined with #1 + #2: 50 jobs ≈ 10 min tail, 150 jobs ≈ 30 min tail.

**Risk.**
- Cost. Need to monitor Modal monthly burn and set a hard alert. Currently $0.95 in May (from `stemscriber_full_state.md` May 6), so headroom is fine, but launch traffic will change that.
- Modal cold-start. New functions may have 30s cold-start on first call. Mitigation: warm pool of 1 (already costs ~$5/mo).

**Dependencies.** None. Independent of #1–3.

---

### #5 — Graceful UI degradation: queue ETA + email-when-ready CTA

**Hypothesis.** Even with #1+#2+#3, at 150 simultaneous uploads some users will wait >30 min. The current UI shows nothing useful during a queue wait. Add:
- "You're #N in queue, estimated wait ~M min" on the upload result screen, computed from `len([j for j in jobs if j.status in ('processing', 'queued')])` divided by current throughput.
- "Don't want to wait? Drop your email and we'll send you the link when it's ready" — checkbox on upload, server captures, sends notification when status flips to `ready_for_practice` (recommendation #1's new milestone).
- Keep the rock trivia rotator that's already in `progress.js` — Jeff has called it out as good UX glue.

**Time estimate.** 4–6 hr. Frontend addition + reuse the existing `notifications.send_job_complete_email` from `pipeline.py:664-666`. Need to flip `ENABLE_JOB_EMAILS=true` (already in the launch-day flag-flip list per memory).

**Expected throughput improvement.** Zero technical throughput. **Massive perceived-throughput improvement.** People who close the tab don't refresh-spam, which actually helps real throughput.

**Risk.** Email deliverability. Already on Resend (per memory references). Already-tested path.

**Dependencies.** Recommendation #1's `ready_for_practice` milestone is ideal but not strictly required (could fire email on `'completed'` even today).

---

## §3 — Phased Plan

### This week (May 9–May 16)

Goal: ship #1 + #5. These get us to "launch-survivable at 50 concurrent" without architectural risk.

- **#1 split the post-separation semaphore.** Two-stage completion model. Frontend update for `ready_for_practice`. Tests green.
- **#5 graceful UI:** queue position estimate + opt-in email-when-ready. Flip `ENABLE_JOB_EMAILS=true` on the VPS at the same time.
- Run a 25-song concurrent-upload load test (use the existing audit scripts, fire all 25 at once via `xargs -P 25`). Validate cap=4 with the new slot scope, measure new wall time per job.
- Document the rollback: each ships behind a config flag (`POST_SEP_RELEASE_AT_CHART=true`, `UI_QUEUE_ETA=true`).

### Before launch (May 17 – June 13)

Goal: ship #2 + #4 + soft-bake #3.

- **#2 cap=6 load test** under the new slot scope. Bake at cap=6 in prod for 2 weeks of normal (low) traffic. Watch RAM and CPU graphs (need a basic dashboard — see "what we're explicitly NOT doing" #4 below).
- **#4 move splits to Modal.** Big architectural lift but it's mechanical: Modal infra is already deployed. Test with same 25-song load test. If RAM headroom looks great, push cap to 8.
- **#3 RQ + Redis** in a feature branch. Spend 2 days on it. Stage on a separate systemd unit (`stemscribe-rq.service`, port 5556) that's a copy of prod with the queue swap. Run the 25-song load test against it. Don't promote to main until it's stable for 3 days under real backstage-cohort beta traffic from June 14–18.
- **June 14–18 Refinery rehearsal week:** flip Redis-backed pipeline ON for backstage testers + Jeff. If anything breaks, the flag goes back to threading and we live with #1+#2+#4 only.

### Post-launch (after June 20)

- Real load data tells us whether #3 actually mattered. If we made it through launch on threading, leave Redis path in feature flag for a quiet rollout in early July.
- Add Sentry (the MCP is already authenticated per the May 7 spit-balling capture, just never wired). 1-day task. This is the real production-monitoring gap, much more than a dashboard.
- Consider GPU-side queue: if Modal becomes the bottleneck at >100 concurrent because separation is itself the long pole, scale Modal concurrency limits or pre-warm more containers. Modal handles this with `@app.function(concurrency_limit=N)`. Tunable, costs nothing until activated.

---

## §4 — What we are explicitly NOT doing, and why

1. **Not migrating to Celery.** RQ does the same job for our scale (single-digit workers, single-Redis), with one-tenth the configuration burden. Celery's killer features (chains, chords, periodic schedules) we don't need; we'd just be paying complexity tax. Celery is the right answer at 10+ workers across 3+ machines, which is several years away.

2. **Not migrating to AWS / GCP managed queues (SQS, Cloud Tasks).** Cross-cloud network from Hetzner to AWS is meaningful latency, billing complexity, and a whole new IAM/auth surface. Hetzner-local Redis is faster, simpler, and the user explicitly said no AWS migration.

3. **Not refactoring `process_audio` into a state machine / DAG framework (Airflow, Prefect, Dagster).** These are the right tools at 50+ pipeline steps and a real ops team. We have ~12 stages and one dev. The cost of operating Airflow exceeds any benefit at this scale — Jeff would spend more time fighting the scheduler than fixing actual bugs. Revisit if we ever cross 30 stages or hire a second dev.

4. **Not building a custom monitoring dashboard.** The May 7 spit-balling capture (`#83`, `#84`) flagged this, but a custom-monitoring side project is exactly the thing that doesn't help on launch day. Wire Sentry (already MCP-available) for errors. Lean on the existing `stemscribe-queue-monitor.timer` SMS path for queue depth. UptimeRobot free tier for outside-in liveness. Total ops setup: under a day. Custom dashboards can come post-launch when we know what metrics actually matter.

5. **Not pre-launch-loading model weights into shared memory.** Was tempted to suggest a model-warmth pool to skip the per-job model-load cost in stages (b, c). At only 2 stages and ~5s of cold-load each, the engineering cost (multi-process model sharing in Python is painful; mmap'd torch tensors work but are fragile) is not worth the 10s saved per job. Recommendation #4 (move splits to Modal) sidesteps this anyway.

6. **Not increasing concurrency to >8 on the current CPX41.** RAM ceiling. Even with #4 freeing 6–8 GB, faster-whisper + librosa + the chart formatter pipeline is ~1.5 GB per slot. CPX41 holds 8 of those plus OS + Flask + idle workers. Above 8, we're rolling dice on OOM. If launch demand justifies it, the next move is CPX51 (16 vCPU / 32 GB, ~$50/mo) — fits under the $50–200 budget. Don't pre-upgrade; load-test first.

7. **Not refactoring chord detection or any of the detector code.** Per the May 7 chord-research doc and the Phase 2 librosa replacement already shipped 2026-05-06, the detector path is its own work track. This document is exclusively about the pipeline shell + concurrency. Any change here must be detector-agnostic.

8. **Not changing the Anthropic chord corrector path** beyond adding a hard timeout (15s, with a retry, then skip — non-fatal already). Per the constraint, the corrector is core. But a hung corrector holding a slot at 150 concurrent can take down the box; the timeout is a one-line defensive add, not an architectural change.

---

## Open questions for Jeff

1. Is the two-stage `ready_for_practice` → `completed` status model OK, or do you want to keep one terminal state and just release the slot earlier internally? (Recommendation #1 (a) vs (b).)
2. Are you willing to spend +$100–200/mo on Modal at peak to move the local splits off the VPS? (Recommendation #4.)
3. Is the 25-song concurrent load test something you want to run on prod, or should we stand up a staging copy on a small Hetzner instance for it? Prod is fine for a short burst — backups are on, 17 songs already exercised it.
4. Sentry: ship now or post-launch? It's flagged as launch-prep in `#83` but it's a 1-day item and would cover us during the soft launch. Recommend now.
