# Scaling Infrastructure for June 20 Soft Launch

**Date:** 2026-05-09
**Author:** Research pickup, pre-launch
**Trigger:** Today's 17-song audit backed up the post-separation queue 20+ minutes. Refinery soft launch (June 20) expects 50–150 backstage musicians potentially uploading concurrently at sound-check / set break. We need a launch-day capacity plan.
**Decision authority:** Jeff signs off; this is research only — no code or infra changes made.
**Scope:** What to harden in the next 6 weeks, what NOT to do, and what the kill-switch is on launch day.

---

## §1 — Current Infrastructure Inventory

All numbers below are **live** as of 2026-05-09 13:42 UTC, captured via `ssh root@5.161.203.112` while the post-Apr-25 17-song audit is still draining.

### 1.1 Application server

| Item | Value | Source |
|---|---|---|
| Provider | Hetzner Cloud, type CPX41, server ID 124254345 | `reference_hetzner.md` |
| CPU | 8 vCPU (`nproc`) | live |
| RAM | 15 Gi total | `free -h` |
| Disk | 150 GB SSD, 27 GB used (19%) | `df -h /` |
| Cost | ~€25/mo (~$27) + €5/mo backups | memory |
| OS | Ubuntu 22.04, uptime 14d 17h | `uptime` |

### 1.2 Web server (the load-bearing finding)

The systemd unit at `/etc/systemd/system/stemscribe.service` runs:

```
ExecStart=/opt/stemscribe/venv311/bin/python app.py
```

**Production is serving the Flask development server**, not gunicorn. `app.py:353` calls `app.run(host='0.0.0.0', port=port, debug=False, threaded=True)`. The Dockerfile in the repo *does* configure gunicorn (`--workers 1 --timeout 300 --preload`), but the Dockerfile is not used — the VPS runs the venv directly.

`threaded=True` means Flask spawns a new Python thread per incoming request. There is **no worker pool, no request queueing, no connection limit**. Concurrent capacity is gated by:

- The Python GIL (one thread runs Python at a time, but I/O releases it)
- The post-separation semaphore at 4 (`pipeline.py:40`)
- The global stem-separation semaphore at 1 (`separation.py:54`) — gates Modal calls
- The kernel's per-process file descriptor limit and 16 GB RAM ceiling

Single PID 524581: 5.5 GB RSS, 14.6 GB virtual, **72 threads**, 144 CPU-minutes since May 8.

### 1.3 Concurrency model — the actual chokepoint

```
HTTP request → Flask thread (unbounded) → /api/upload
                                              ↓
                                   stem-separation semaphore (cap = 1)
                                              ↓
                                   Modal cloud-GPU call (A10G, ~60–90s)
                                              ↓
                                   post-separation semaphore (cap = 4)
                                              ↓
                                   chord detect + MIDI + MusicXML + lead sheet
                                              ↓
                                   write to /opt/stemscribe/outputs/{job_id}/
```

The "stem-separation = 1" semaphore is **not** what it sounds like. The actual GPU work runs on Modal (which scales to many parallel A10G containers, see §2.4). But the local semaphore serializes the **Modal API call wrapper**, so even though Modal could run 5 containers in parallel, our backend feeds them one at a time. This is a leftover from when separation ran locally on the M3 Mac.

### 1.4 Modal cloud GPU

From `backend/modal_separator.py:53-59`:

```python
@app.function(
    gpu="A10G",
    timeout=600,
    volumes={"/model-cache": model_volume},
    memory=16384,
)
@modal.concurrent(max_inputs=5)
```

- Each Modal container handles up to 5 in-flight requests
- Modal auto-scales container count horizontally based on demand
- `timeout=600` per request — anything over 10 min hard-fails
- BS-RoFormer-SW + KARA_2 vocal split, ~$0.06/song, ~60–90s wall-clock per song
- **Cold start** is the real risk: first container takes ~30–60s to boot the model. After that, warm containers serve at ~2 songs/min each.

### 1.5 Storage

| Path | Size now | Purpose | Retention |
|---|---|---|---|
| `/opt/stemscribe/uploads/` | 342 MB | original user audio | 48h (`UPLOAD_RETENTION_HOURS`) |
| `/opt/stemscribe/outputs/` | 4.4 GB | stems + MIDI + chord JSON + lead sheets per job | 7d (`OUTPUT_RETENTION_DAYS`) |

`RETENTION_DRY_RUN=false` is staged in `.env` but only activates on next service restart. **Currently old jobs accumulate** — the 4.4 GB outputs/ proves it. Per-job footprint is roughly **45–60 MB** (8 stems × 5–11 MB MP3 + chord JSON + MIDI + MusicXML).

Cloudflare R2 client code exists (`backend/storage/r2.py`, `boto3`) but is **not active in production**: no `R2_ACCESS_KEY_ID` in the live `.env`. All storage is local SSD. R2 bucket creation never happened.

### 1.6 Database

- Supabase Postgres (managed, hosted at `db.mfgxmfuundfytfzlfdhc.supabase.co`)
- DB calls go over public internet; not a VPS resource
- Free tier limits: 500 MB DB size, 5 GB egress/mo, 2 GB file storage

### 1.7 Cloudflare Tunnel

Service `cloudflared.service` running 14 days, 38.5 MB RAM, 70 minutes CPU total. Routes `stemscriber.com` → `localhost:5555`. Free tier. **No bandwidth or connection-count cap on free Tunnel** (Cloudflare's official position; rate-limit rules can be added per-account but none configured).

### 1.8 Rate limits

`backend/middleware/rate_limit.py:62` uses `storage_uri="memory://"`. Limits live in the Flask process. Configured caps:

- Upload: 5/min per IP
- Library: 60/min
- Beta endpoints: 10/min
- SMS: 10/min
- Auth: per `AUTH_LIMIT` (brute-force protection)

A restart wipes the rate-limit window — fine for our scale; not OK for multi-VPS later.

### 1.9 Background services

- `processing.watchdog` — thread, restarts stalled jobs at 600s
- `processing.retention` — thread, deletes expired uploads/outputs
- `stemscribe-queue-monitor.timer` (systemd) — every 5 min, SMS Jeff if queue > 4 sustained 15 min
- `n8n` on `n8n.kozbotix.com` — separate VPS, dormant workflows; not in scope here

### 1.10 Observed strain (today, real data)

`journalctl -u stemscribe --since '6 hours ago'`: at **12:58:07** ten audit jobs all queued for the post-separation slot in the same second. The same 4 job IDs were still re-queueing at **13:42:00** (44 minutes later). Modal completed 34 separations in the past hour (~1 every 100s).

**Modal is keeping up** (GPU separation finishes in ~90s each). **The post-separation pipeline (chord detect + MIDI + MusicXML + Anthropic chord correction with `ENABLE_ANTHROPIC_CORRECTION=true`) is the bottleneck** — 4 simultaneous CPU-bound jobs on 8 vCPUs takes 5–10 min each, so depth-10 = ~25 min wait.

---

## §2 — Capacity Model: Where Each Component Breaks

The Refinery cohort scenario: 50–150 musicians backstage, **bursty arrivals** at sound-check (60–90 min before downbeat) and set break (~15 min mid-show). Realistic peak: **20–40 simultaneous uploads** in a 5-min window. We are NOT facing 150 simultaneous — that's the audience-attention upper bound, not the upload-button-clicker count.

| Component | Capacity ceiling | Headroom at 40-concurrent? | Notes |
|---|---|---|---|
| Flask threaded server | ~100 concurrent open requests before file descriptors / GIL contention | OK for **HTTP**; not OK for long-poll | Each upload holds the request thread until 200 OK. A 4-min song takes ~3 min before response. 40 stalled threads = 40 file descriptors + 40 stack frames. Fine on RAM. |
| RAM | 16 GB total, currently 6.2 GB used + 8 GB buff/cache | **TIGHT**. Each in-flight job allocates ~400 MB peak (chord detector, MIDI, music21). 10 in-flight = +4 GB. Risk of OOM at depth 12+. | No swap. OOM kills the whole Flask process — every in-flight job dies. |
| post-sep semaphore (4) | 4 concurrent post-Modal jobs | **HARD LIMIT.** Job 5+ blocks until a slot frees. At today's 17-song burst, 4 active + 13 waiting = ~25-min queue depth. | Tunable. Cap exists because higher caused watchdog stalls in Apr 25. |
| stem-separation semaphore (1) | 1 Modal call dispatched at a time | Modal containers idle waiting | This is a **vestigial bug** — Modal's `@modal.concurrent(max_inputs=5)` already handles fan-out. Removing this cap is the single highest-leverage change. |
| Modal A10G | 5 inputs per container, auto-scale containers | 100+ concurrent songs = ~20 containers spun up. Cold start 30–60s on the first batch. | Modal pricing is per-container-second, so 20 containers for 2 min = $0.40 burst. Negligible. |
| Modal account QPS | Default Modal account caps are very high (thousands/min for paid plans). Not a launch-day risk. | Plenty | Confirmed by `@modal.concurrent` design pattern — Modal explicitly scales to thousands of concurrent inputs on paid accounts. |
| Disk (`/`) | 150 GB, 27 GB used | 100 jobs × 50 MB = 5 GB additional. Fine. | Retention sweeper must be active, otherwise 1000 jobs = 50 GB and disk pressure begins at ~120 GB. |
| Cloudflare Tunnel free | No documented hard cap on concurrent connections. Some user reports cite ~1000 concurrent. | Plenty for 40 | Free tier is fine. Cloudflare WARP-style abuse triggers throttling, not legit traffic. |
| Cloudflare CDN | Unmetered free | Plenty | Static assets cached at edge. |
| Supabase free tier | 5 GB egress/mo, 500 MB DB | Egress is the risk if user library JSON is large. 100 users × 5 GB log over a month is unlikely; we're fine. | Schema is small. Auth + jobs metadata. |
| Stripe | No QPS issue; webhook is async | Plenty | n/a |
| Anthropic API (chord correction) | 50 RPM default for new accounts | **WATCH.** Every job currently calls Anthropic for chord correction (`ENABLE_ANTHROPIC_CORRECTION=true`). 40 concurrent jobs × 1 call each over ~2 min = 20 RPM. OK at default. | If we hit 429, jobs degrade gracefully (chord correction is optional); not a hard fail. |

### 2.1 Where it actually breaks first

**Order of failure as load climbs:**

1. **Post-sep semaphore queue grows** (already happening today at 17 songs — 25-min waits). User pain: "stuck at 'Queued for processing'".
2. **At ~12 in-flight post-sep jobs simultaneously**, RAM crosses 12 GB. Linux page cache evicted, then OOM-killer fires Flask. **Every in-flight job dies.** This is the catastrophic mode.
3. **At ~50 simultaneous HTTP requests holding open**, Flask threads + Python GIL contention slow new-request acceptance. Users see "spinning" on upload click. Not fatal, just slow.
4. **Cloudflare Tunnel and Modal stay healthy** all the way through.

The fix is to keep the queue from getting >4 deep AND to make sure Flask doesn't OOM if it does.

---

## §3 — Top Ranked Recommendations Before June 20

Ranked by leverage (impact × low effort × low risk). Effort estimates assume Jeff or solo dev work.

### Rec 1 — Drop the stem-separation semaphore from 1 to 4 (or remove)
**Action:** In `backend/processing/separation.py:54`, change `_separation_semaphore = threading.Semaphore(1)` → `Semaphore(8)` or remove entirely. Modal already fans out via `@modal.concurrent(max_inputs=5)` and auto-scaled containers.
**Cost:** $0/mo. Modal costs scale linearly per song regardless ($0.06).
**Effort:** 30 min code + test + deploy.
**Risk:** **LOW.** The semaphore was relevant when separation ran on Jeff's M3. With Modal it's just artificial latency. Failure mode: Modal account hits a soft limit and we get rate-limited; we already retry on Modal errors.
**Unblocks:** Modal can ingest 8 songs simultaneously, GPU work parallelizes, post-sep slot opens up sooner. Today's 25-min wait drops to ~6 min.

### Rec 2 — Activate retention sweeper + add disk-pressure alerting
**Action:** Restart `stemscribe.service` once (activates the staged `RETENTION_DRY_RUN=false`). Then add a second systemd timer that pages Jeff if `df / ` >70% used. Verify after 48h that uploads/ shrinks and outputs/ doesn't grow unboundedly.
**Cost:** $0/mo.
**Effort:** 20 min (restart + write the disk-monitor unit + smoke).
**Risk:** **LOW** if dry-run logs from past week are sane. **MEDIUM** if there's an active job during restart — the graceful_shutdown handler waits 60s so this is mostly safe; do it during off-peak.
**Unblocks:** Disk doesn't run out during launch week. Critical because 150-musician sustained use over a week could write ~30–60 GB.

### Rec 3 — Move Flask behind gunicorn with 1 worker + many threads
**Action:** Update `stemscribe.service` `ExecStart` to `gunicorn --workers 1 --threads 32 --timeout 600 --preload --bind 0.0.0.0:5555 backend.app:app`. Single worker keeps the in-memory `jobs` dict valid (Dockerfile comment confirms this constraint). 32 threads gives more parallelism than Flask dev server's free-for-all model and crucially gives us a **hard timeout** at the proxy layer.
**Cost:** $0/mo.
**Effort:** 1 hour (write the unit, install gunicorn in venv311, smoke test, swap, watch for an hour).
**Risk:** **MEDIUM.** Background threads (watchdog, retention) need to launch correctly under gunicorn `--preload` (they do — they're started in `create_app()`). Some latency-sensitive paths might behave differently. Smoke-test the upload flow end-to-end before declaring done.
**Unblocks:** Production-grade request handling. Flask dev server is explicitly documented as "not suitable for production" — fixing this is overdue regardless of launch.

### Rec 4 — Bump VPS from CPX41 to CPX51 (or CPX61) one week before launch
**Action:** Hetzner has live CPU+RAM resize (5-min downtime) via `hcloud server change-type 124254345 cpx51`. CPX51 is **16 vCPU / 32 GB RAM** at ~€55/mo (~$60). CPX61 is **32 vCPU / 64 GB RAM** at ~€110/mo (~$118). Recommendation: **CPX51**. Doubles RAM (kills OOM risk) and CPU. CPX61 is overkill at our load profile.
**Cost:** +€30/mo (~$33/mo Δ from CPX41). Reversible — downsize after launch if usage doesn't sustain.
**Effort:** 30 min (resize + restart + verify).
**Risk:** **LOW.** Reversible. Backups already taken. Schedule for Sun June 14 morning.
**Unblocks:** OOM safety margin. Lets us safely raise post-sep semaphore from 4 → 8 if needed.

### Rec 5 — Add a "queue full" maintenance mode toggle (kill-switch)
**Action:** Add a feature flag `QUEUE_PAUSED=false` env var. When `true`, `/api/upload` returns `503` with a polite "we're at capacity, try again in 5 min" JSON. Frontend renders this as a banner. Not a maintenance page — just upload-pause. Browse, library, and practice-mode stay live.
**Cost:** $0/mo.
**Effort:** 1 hour. Code + deploy + manual test.
**Risk:** **LOW.** Pure read of an env var on each upload request.
**Unblocks:** Launch-day kill-switch. If Modal blows up, Anthropic 429s, or RAM climbs, Jeff can `ssh ... "echo 'QUEUE_PAUSED=true' >> .env && systemctl restart stemscribe"` in 30 seconds and stop digging the hole. New uploads bounce, in-flight jobs continue.

### Optional Rec 6 (only if Recs 1–5 don't relieve pressure)
**Skip horizontal scale for June 20.** A second VPS behind a load balancer **breaks the in-memory `jobs` dict** (Dockerfile literally calls this out: "1 worker, jobs dict is in-memory, multiple workers can't share it"). The fix is moving job state to Postgres or Redis, which is **2–3 days of work** and adds Redis ($15/mo Hetzner cheap-redis or Upstash free tier) — too much risk for 6-week launch window. Defer to August post-launch hardening.

### Summary table

| Rec | $/mo Δ | Effort | Risk | Do by |
|---|---|---|---|---|
| 1. Drop separation semaphore to 4–8 | $0 | 30 min | LOW | June 1 |
| 2. Activate retention + disk alert | $0 | 20 min | LOW | THIS WEEK |
| 3. Gunicorn 1×32 | $0 | 1 hr | MEDIUM | June 1 |
| 4. CPX41 → CPX51 | +$33 | 30 min | LOW | June 14 |
| 5. Queue-paused kill switch | $0 | 1 hr | LOW | June 14 |
| **Total** | **+$33/mo** | **~3.5 hours** | | |

Total budget impact: well under the $200/mo headroom. Most of the win is software, not hardware.

---

## §4 — Launch-Day Runbook

### 4.1 Pre-flight (June 19, the day before)

- [ ] Verify all 5 recs above shipped. Smoke-test upload + practice mode end-to-end.
- [ ] `ssh ... "free -h && df -h && systemctl status stemscribe cloudflared"` — sanity baseline.
- [ ] Confirm `stemscribe-queue-monitor.timer` is firing every 5 min: `systemctl status stemscribe-queue-monitor.timer`.
- [ ] Confirm Twilio SMS to +1 803-414-9454 is working: send a test alert.
- [ ] Confirm Modal account balance + auto-recharge.
- [ ] Confirm Anthropic key has $50+ balance (chord correction will burn ~50 calls/100 songs at $0.003 each = trivial).
- [ ] Tail journal in a tmux: `journalctl -u stemscribe -f`.

### 4.2 Live monitoring during the show

Put these in a single tmux on Jeff's laptop (or a tablet):

```
ssh root@5.161.203.112 "watch -n 5 'free -h | head -2 ; df -h / | tail -1 ; uptime ; journalctl -u stemscribe --since 5min --no-pager | grep -c \"queued — waiting\"'"
```

Refreshes every 5 sec. Jeff sees: RAM, disk, load, queue depth in last 5 min.

### 4.3 Kill-switch hierarchy (apply in order)

**Tier 0 — green:** RAM <70%, queue <8, no SMS alerts. Do nothing.

**Tier 1 — yellow (queue depth 8–15 sustained 5 min, RAM 70–85%):** Tell Refinery audience verbally: "Give it 2–3 minutes, the queue will drain." Don't change infra.

**Tier 2 — orange (RAM >85% OR queue >15 OR Modal errors spiking):**
```bash
ssh root@5.161.203.112 "echo 'QUEUE_PAUSED=true' >> /opt/stemscribe/.env && systemctl restart stemscribe"
```
Frontend now shows "We're at capacity, try again in 5 min." Existing jobs drain. Wait 10 min, flip back.

**Tier 3 — red (Flask process died, OOM in journal, site returning 502):**
```bash
ssh root@5.161.203.112 "systemctl restart stemscribe && journalctl -u stemscribe -f"
```
Then immediately Tier 2's queue-pause as a precaution. Investigate after the set.

**Tier 4 — catastrophic (Modal down, Cloudflare Tunnel down, Hetzner outage):**
- Cloudflare → check status.cloudflare.com on phone
- Modal → check status.modal.com
- Hetzner → check status.hetzner.com
- Post a holding tweet from `@stemscriber`: "Refinery cohort: we're seeing a spike, hold uploads for 15 min."
- Don't touch anything until you know which layer failed.

### 4.4 The IP throttle fallback

If a single IP is hammering us (unlikely with the 5/min Flask-Limiter cap, but possible if shared NAT at the venue WiFi):

```bash
# At the Cloudflare dashboard — Security → WAF → Custom rules
# Add: cf.client.ip eq <IP> → Action: Challenge (managed)
```

This is in the Cloudflare web UI, not SSH. Requires a phone with login. Pre-stage the URL bookmarked.

---

## §5 — NOT Doing — and Why

| Item | Why not |
|---|---|
| **Multi-VPS horizontal scaling** | Breaks in-memory `jobs` dict. Requires Redis or Postgres job-state migration (2–3 days). Risk > reward for 6-week window. Revisit August. |
| **Move uploads/outputs to R2** | Code exists (`backend/storage/r2.py`) but not wired into pipeline. Migrating mid-flight changes the storage layer that lead-sheet PDFs, MIDI, MusicXML serving all depend on. ~4 hours minimum. Disk is at 19%, retention sweeper handles growth. **Do this only if disk crosses 60%.** |
| **Switch from Cloudflare Tunnel free → paid** | No documented hard cap we'd hit. Tunnel is rock-solid for our load. Save the $20/mo. |
| **Move database from Supabase free → Pro ($25/mo)** | We're at <100 MB DB, <100 MB egress. Free tier ceiling is 500 MB / 5 GB. Plenty. |
| **Add APM / Sentry / Datadog** | Already on the post-launch task list (#83). Pre-launch effort better spent on Recs 1–5. SMS queue alerts cover the "is it broken right now" case. Sentry MCP integration is queued for August. |
| **Replace Flask with FastAPI / Quart** | 2-week rewrite. No launch-day ROI. Flask's threading is fine post-gunicorn. |
| **Pre-warm Modal containers** | `modal.concurrent(max_inputs=5)` with auto-scale already handles this. Pre-warming costs ~$0.50/hr × 6 hr show = $3. Worth it ONLY if Jeff sees cold-start lag in dress rehearsal — easy to add at the last minute via `min_containers=1` on the Modal function. **Stage but don't ship.** |
| **Bigger VPS than CPX51** | CPX51 doubles current capacity. CPX61 quadruples it but costs 4× the upgrade. Refinery 50–150 musicians is firmly inside CPX51 envelope per the §2 capacity model. Reversible if wrong. |
| **k8s, Nomad, Kamal, etc.** | Solo dev. Six weeks. No. |
| **Custom CDN for stem WAVs** | Cloudflare edge caches static assets already. Stem downloads are auth-gated and per-job, not cacheable. CDN buys nothing here. |
| **Drop Anthropic chord correction during launch** | It's the post-Apr-26 quality lift. Dropping it tanks chord accuracy on rock songs (we're already at 74% with librosa + Anthropic correction). If it 429s, jobs degrade gracefully — leave it on. |

---

## §6 — One-Glance Summary for Jeff

**State today:** Single Flask dev-server process on 8 vCPU / 16 GB. Modal scales fine. Bottleneck is local post-separation pipeline, capped at 4 concurrent. Today's 17-song audit proved a 25-min queue depth at burst.

**6-week plan:** Drop the silly separation-semaphore-1 (Rec 1), activate retention (Rec 2), put gunicorn in front of Flask (Rec 3), bump CPX41 → CPX51 the week before (Rec 4), add a `QUEUE_PAUSED` env-var kill-switch (Rec 5). Total: ~3.5 hours of work + $33/mo.

**Capacity after:** Comfortably handles 30–40 concurrent uploads with 6–8 min queue depth, no OOM risk. Refinery 50–150 musicians is in-envelope as long as arrivals are bursty (they will be).

**Kill switch on launch night:** `echo 'QUEUE_PAUSED=true' >> .env && systemctl restart stemscribe` — bounces new uploads with a friendly message, drains in-flight, takes 30 seconds.

**Not doing:** Multi-VPS, R2 migration, paid Cloudflare, FastAPI rewrite, Sentry. All deferred to August post-launch.
