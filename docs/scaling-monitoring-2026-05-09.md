# StemScriber pre-launch monitoring & abuse protection plan

**Date:** 2026-05-09
**Author:** Claude (research-only — no code committed)
**Trigger:** May 7 spit-balling session (#83 monitoring & scaling). 17-song audit on May 5 strained the box. Refinery soft launch on June 20 brings 50–150 backstage musicians uploading concurrently. Single-signal queue-depth SMS is the only thing watching prod today. If anything else breaks at the show, we won't know until customers tell us.
**Constraints:** $0–50/mo budget. Solo dev. 6 weeks. Each item ≤8 hr unless value is unambiguous. Plausible already in place (don't replace).
**Alert routing rule:** SMS or email OK to +1-844-791-5323 / jkozelski@gmail.com / support@stemscriber.com. **Never call 803-414-9454.** (per `feedback_escalation_preference.md`)

---

## §1. Current monitoring & abuse-protection inventory

### What EXISTS (verified May 9 against prod 5.161.203.112 + repo)

| Surface | Status | Where | Notes |
|---|---|---|---|
| **Queue-depth alerting** | ✅ Live | `stemscribe-queue-monitor.timer` (5-min cadence) → `backend/scripts/monitor_queue.py` | Threshold 4 sustained over 3 samples (15 min). Cooldown 30 min. SMS to +18034149454 from +18447915323. |
| **Concurrency cap** | ✅ Live | systemd-side post-separation semaphore | Cap = 4. Job 5+ shows "Queued for processing". |
| **Request-level rate limits** | ✅ Live | `middleware/rate_limit.py` + `app.py:243-246` | Global 60/min/IP. **Upload 5/min/IP applied to `api.upload_audio` + `api.process_url_endpoint`** via `limiter.limit(UPLOAD_LIMIT)` in app factory. Auth 5/min/IP on login/register/forgot. Memory-backed (not Redis) — resets on restart. |
| **Plan-level quota gate** | ✅ Live | `enforce_plan_limits` decorator → `auth/decorators.py` | Free 5/mo, Pro 50/mo, Lifetime 50/mo. Anonymous tracked by IP hash. |
| **Cloudflare Tunnel front** | ✅ Live | stemscriber.com → CF Tunnel → 5.161.203.112 | Free DDoS layer + bot-management heuristics. CF-Connecting-IP correctly read by limiter (`rate_limit.py:50-52`). |
| **Health endpoint** | ✅ Live | `GET /health` returns 200 `{status:"ok"}` | Trivial — does NOT verify Modal, DB, R2, or queue. |
| **Hetzner backups** | ✅ Live | CPX41 €5/mo, 7-day retention | Re-image recovery only — no DB-level rollback. |
| **Plausible analytics** | ✅ Live | All public pages | Traffic + funnel only. NOT an alerting channel. |
| **Vapi cost cap** | ✅ Live (May 7) | `maxDurationSeconds=600` + `silenceTimeoutSeconds=30` on the only remaining assistant | Was uncapped — could have burned $5–15/hr on a runaway call. |
| **Plan/duration enforcement** | ✅ Live | `enforce_duration_limit` (5/15/30 min by tier) | Returns 413 if oversized. |
| **journalctl on VPS** | ✅ Default | systemd | Local only. No retention beyond systemd defaults. No grep alerts. |

### What is MISSING

| Gap | Failure mode it leaves uncovered |
|---|---|
| **Uptime monitoring** | Cloudflare Tunnel down, Hetzner reboot, certs expire, Flask process crashed → site is dead and Jeff finds out from a tweet. |
| **Error tracking (Sentry)** | A 500 in `/api/upload` or `/api/status/` ships a generic error to the user. Nothing surfaces stacktrace, frequency, or affected user to Jeff. Sentry MCP is wired in `claude mcp list` but flagged "Needs authentication" — never been logged in. |
| **5xx-rate alerting** | Modal flake or DB connection drop spikes 5xx. Queue depth stays normal because nothing is making it through. Current monitor would never fire. |
| **Latency monitoring** | `/api/upload` taking 30s (Modal A10G cold-start) vs the usual ~3s — no signal. Affects perceived reliability at Refinery where everyone is uploading simultaneously. |
| **Disk-fill alert** | Currently 27G/150G (19%). Retention sweeper recently activated. If a bug regresses retention, disk fills silently → uploads start failing with cryptic "no space left on device" errors mid-pipeline. |
| **RAM/load alert** | 5.2G/15G used now, load avg 1.14/1.93/2.37 (8 vCPU box). A leak or runaway worker would OOM with no warning. |
| **Modal spend cap** | $0.06/song × abuse loop = unbounded burn. Modal has account-level alerts but no per-app hard stop. |
| **Anonymous-upload abuse mitigation** | `/api/upload` has `@auth_required(optional=True)` — unauthenticated POST is ALLOWED. The 5/min upload limit is the **only** floor. An attacker behind 1 IP burns $0.06 × 5 × 60 = $18/hr; a botnet across 100 IPs is $1,800/hr against Modal. Plan-quota gate catches an authenticated free-tier user but **not anonymous IP-hash users** until 5/month is exceeded. |
| **Log aggregation / search** | Anything not in journalctl on this single box is gone. No long-term retention. No "show me every 500 in the last 7 days" tool. |
| **GitHub Actions deploy verification** | No CI smoke test. Bad deploy → site dead → only signal is users complaining. |

### Already-logged but-not-yet-acted-on tasks (memory file)
- **#83** Production monitoring + scaling readiness (this doc resolves)
- **#84** GitHub Actions monitoring → markdown health journal (5-min, auto-commit)
- **#85** n8n alternative on n8n.kozbotix.com (same-VPS caveat — needs hybrid with external uptime)
- **#86** Self-healing agent system multi-tier (Tier 1 watchdog + Tier 4 SMS for launch MVP)

---

## §2. Top 5 ranked recommendations

Each item lists: tool/action, time-to-implement, monthly cost, and the specific failure mode it covers.

### 🥇 Rec 1 — Wire Sentry via the MCP (free tier)
- **What:** Authenticate the existing Sentry MCP (`claude mcp` shows `sentry: ! Needs authentication`), create a single `stemscriber-prod` project, install `sentry-sdk[flask]` in `venv311`, init it in `backend/app.py` factory immediately after Flask creation, set `traces_sample_rate=0.05` (low — we want errors, not full APM), set `environment="prod"`, deploy.
- **Time:** 1.5–2 hr (auth flow + SDK install + redeploy + verify with a forced exception).
- **Cost:** $0 (free tier = 5K errors/mo + 10K performance events/mo + 1 team member). For a 50-150-musician soft launch this is **comfortably enough** — even a really bad launch with 1K errors fits in <20% of the cap.
- **Covers:** Every uncaught exception in the Flask process, including Modal client failures, R2 upload failures, DB connection drops, JWT decode errors. Stacktrace + breadcrumbs + affected user_id auto-captured. **Highest single-point ROI of anything in this doc** because it transforms "user complains, nothing in journalctl" into "alert lands in inbox with line number."
- **Alert routing:** Sentry sends email to jkozelski@gmail.com on first occurrence + threshold spikes. No SMS — email is fine for error-tracking cadence per `feedback_escalation_preference.md`.
- **Threshold to set:** Alert on >5 events/min sustained 5 min, OR any new issue type. Disable noisy issue groups after first launch day if needed.

### 🥈 Rec 2 — UptimeRobot 5-minute external probe on `/health` and `/`
- **What:** UptimeRobot free-tier monitor hitting `https://stemscriber.com/health` every 5 min (the maximum frequency on free tier). Add a second monitor on `https://stemscriber.com/` (root) to catch Cloudflare Tunnel issues that don't break Flask but break user-facing routing.
- **Time:** 30 min total (sign up + add 2 monitors + plug Jeff's email + SMS-via-email-gateway).
- **Cost:** $0 (50 monitors free, 5-min interval, email + webhook alerts). If we want 1-min interval upgrade is $7/mo (Solo plan) — **defer until post-launch unless we hit a real outage**.
- **Covers:** Cloudflare Tunnel collapse, certificate expiry, Hetzner reboot, Flask crash that systemd doesn't auto-restart, DNS misconfiguration. **Externally-anchored** — survives the case where the VPS itself is unreachable.
- **Alert routing:** Email to jkozelski@gmail.com + support@stemscriber.com. Add Twilio SMS via UptimeRobot's webhook → existing `/api/sms/*` is overkill; use UptimeRobot's email-to-SMS bridge or just rely on email for the launch (per the routing rule, email is fine).
- **Why not Cloudflare Health Checks:** Same feature exists in Cloudflare's $5/mo Pro plan, but UptimeRobot free covers it. Skip Cloudflare here.
- **Why not BetterUptime:** Cleaner UI, public status page included, but free tier is 3-min interval and only 10 monitors. UptimeRobot's free is sufficient for v1; revisit if we need a public status page.

### 🥉 Rec 3 — Tighten anonymous upload abuse floor (per-IP daily Modal-spend cap)
- **What:** Add a daily-cap enforcement layer for anonymous IP-hashed users at `/api/upload` and `/api/url`. Currently the only floor is "5 requests per minute." That allows 5 × 60 × 24 = **7,200 uploads/day per IP** in theory, capped only by Flask-Limiter (which resets on restart and is memory-backed). Add a hard daily ceiling of **3 successful submissions per IP-hash per UTC day** for unauthenticated requests, enforced in `enforce_plan_limits` *before* the song-quota check. Authenticated users are unaffected (their plan-quota already covers it).
- **Time:** 2–3 hr (modify `auth/decorators.py:check_rate_limit` to add daily-window bucket for anon path; add migration for new `anon_daily_count` column or reuse existing `usage_events` table with date filter; write unit test; deploy).
- **Cost:** $0 (in-process, uses existing DB).
- **Covers:** Single-IP attacker burning Modal credit. Caps abuse at $0.06 × 3 = **$0.18/day/IP**. A 1,000-IP botnet still maxes at $180/day, which is detectable in Modal billing same-day. Combined with 5/min limiter, attacker can't even reach 3/day in a sustained burst — they hit the per-minute wall first.
- **Why this number:** Free plan is 5 songs/MONTH for *signed-in* users. Anonymous getting 3/DAY is already MORE generous than the signed-in free plan over a month. The right behavior is to drive anonymous users to sign up after their daily cap, not to subsidize anonymity.
- **Side benefit:** Makes the Refinery cohort see "Sign in to upload more" as the natural next step — turns rate-limit friction into a sign-up funnel.

### Rec 4 — GitHub Actions deploy smoke test (pairs with #84)
- **What:** A workflow on `push` to main that, after deploy, does:
  1. `curl -fsS https://stemscriber.com/health` (must return 200)
  2. `curl -fsS https://stemscriber.com/api/library | head -c 200` (auth-anon path returns valid JSON)
  3. Posts result as a commit-status check + emails on red.
- **Time:** 1 hr (write the workflow YAML + test on a dry-run branch).
- **Cost:** $0 (GitHub Actions free for public/private up to 2,000 min/mo — this is a 30-second job).
- **Covers:** Bad deploy that leaves the service running but broken (e.g., import error caught by a try/except elsewhere, missing env var, wrong route registered). Today there's nothing — Jeff deploys with `scp` and prays.
- **NOT to expand into:** The full "markdown health journal" #84 idea adds 5-min recurring writes + auto-commits to a `monitoring/` directory. Defer that — it's nice-to-have but creates merge noise and journalctl already has the data. Build the deploy-smoke-test piece, skip the recurring-journal piece for launch.

### Rec 5 — `node_exporter` + 1 Grafana Cloud free dashboard (disk/RAM/CPU)
- **What:** Install `prometheus-node-exporter` on the VPS (one apt install). Sign up for Grafana Cloud free tier (10K series, 14-day retention, 50GB logs). Point a single scrape at the node-exporter on a Cloudflare-tunnel-protected port (or Tailscale). Set 3 alerts: disk >85% full, free RAM <500MB sustained 10 min, load >12 sustained 10 min (8 vCPU box, 1.5× cores is the warning line).
- **Time:** 2.5–3 hr (apt install + Grafana Cloud signup + agent config + 3 alert rules + verify each fires by simulating).
- **Cost:** $0 (Grafana Cloud free tier).
- **Covers:** Disk-fill (regression in retention sweeper), OOM, runaway worker. Today none of these have alerting.
- **Alert routing:** Grafana Cloud → email to jkozelski@gmail.com. SMS not needed for these — disk/RAM is "you have hours to fix this," not "the show is happening RIGHT NOW."
- **Defer-able alternative:** If the 3-hour budget is tight, skip Grafana Cloud and use a 20-line bash cron that runs `df`/`free`/`uptime` every 15 min and SMSes if thresholds breach. Crude but it works. Saves ~2 hr.

---

## §3. Concrete pre-launch checklist

### Wired by June 20 (must-do — total ~7 hr work)
- [ ] **Sentry initialized** in `backend/app.py` with free-tier DSN. Verify with forced exception. (2 hr)
- [ ] **UptimeRobot** monitors live on `/health` + `/`. Test by stopping `stemscribe.service` and confirming alert lands in <10 min. (30 min)
- [ ] **Anonymous daily-cap (3/day)** added to `enforce_plan_limits`. Unit test added. Deploy. (2.5 hr)
- [ ] **GitHub Actions deploy smoke test** workflow committed. Verified by intentionally breaking a deploy. (1 hr)
- [ ] **`/health` upgraded** to verify (a) Modal client can authenticate, (b) DB query succeeds, (c) outputs dir is writable. Returns 503 if any check fails so UptimeRobot trips correctly. (1 hr)

### Wired by June 20 if bandwidth (nice-to-have — total ~3.5 hr)
- [ ] **Grafana Cloud + node_exporter** for disk/RAM/CPU dashboard + 3 alerts. (3 hr)
- [ ] **Modal account-level spend alert** at $50/day, $200/week (Modal dashboard, no code). (15 min)
- [ ] **Cloudflare WAF** "Bot Fight Mode" turned on for stemscriber.com (free tier, dashboard toggle). (15 min)

### Defer to post-launch (post June 20)
- [ ] **#84 markdown health journal** — recurring journalctl-to-git-commit workflow. Adds noise, journalctl already has the data, Sentry covers errors. Reconsider only if Jeff hits a recurring debug pattern that needs cross-day aggregation.
- [ ] **#86 Tier 1 self-healing watchdog** — auto-restart on health-check failure. systemd already has `Restart=on-failure`; verify that's set, then this becomes a no-op. Tiers 2–3 (Claude triage + remediation) are post-launch experimentation, not launch-critical.
- [ ] **Loki/Grafana log aggregation** — `journalctl + grep` is sufficient at 50–150 musicians. Solo dev. Revisit at 1,000 DAU.
- [ ] **Public status page** — needs ≥1 real outage of customer signal first. Premature for soft launch.
- [ ] **Per-IP CAPTCHA** on /api/upload — reserve as kill-switch if abuse hits despite rec 3.
- [ ] **Distributed rate-limit storage (Redis)** — only matters if we run multiple Flask workers. Single VPS = memory-backed limiter is fine.

---

## §4. Abuse model — top 3 vectors

### Vector 1 — Modal-budget burn via anonymous uploads (HIGHEST risk)
- **The path:** Attacker scripts curl POST to `/api/upload` with a real audio file from rotating IPs. Each one costs StemScriber **$0.06 on Modal** + R2 storage + bandwidth. No login, no payment, no deterrent. Today's per-IP-per-minute floor is 5; per-IP-per-day is unbounded.
- **Worst case today:** Single IP × 5/min × 60 min × 8 hr = 2,400 jobs = **$144 in one workday from one IP**. A 100-IP botnet for 24 hours = **$8,640**.
- **Mitigation (Rec 3 above):** Anonymous daily cap of 3/IP/UTC-day. Worst case becomes $0.18/IP/day × N IPs. Combined with **Modal account-level $50/day hard alert** (Rec 3.5 free, takes 15 min), Jeff sees the abuse same-day at $50 spent and can flip the env flag `DISABLE_ANONYMOUS_UPLOADS=true` (which he should add as a feature flag during the same change — 30 extra min).
- **Residual risk:** Authenticated abuse — someone signs up, eats their 5/month, gets banned. Caught by plan-quota gate already. Not a $-loss vector.

### Vector 2 — Bandwidth/disk burn via large file uploads
- **The path:** Attacker POSTs a 100MB file (likely the `validate_file_upload` ceiling — verify in `middleware/validation.py`). Each request fills disk + chews bandwidth even if it never reaches Modal. Cloudflare Tunnel passes large bodies through.
- **Worst case:** 5 uploads/min × 100MB = 500MB/min ingress. VPS has 150GB disk currently 81% free; sustained 4 hours fills the box.
- **Mitigation:**
  1. `validate_file_upload` should already reject >100MB — **verify the actual ceiling** during Sentry-rollout work. If higher than 50MB for free/anon, lower it.
  2. Retention sweeper deletes uploads at 48h — already live (verify it ran in last 24h: check journalctl for `retention` keyword).
  3. Disk-fill alert (Rec 5) catches the failure mode regardless of source.

### Vector 3 — Credential/login brute force
- **The path:** Attacker scripts POST to `/auth/login` with credential-stuffing list against known-popular emails.
- **Today's defense:** `AUTH_LIMIT = 5/min/IP` already wired (`app.py:211-213`). Cloudflare bot-fight mode (free, 1 toggle) adds heuristic blocking.
- **Mitigation gap:** No per-account lockout. 100-IP attacker can do 500/min against one email. Defer to post-launch — Supabase JWT means sessions are short-lived and there's no "password reset to take over" path that doesn't email the real owner.
- **What to add by June 20:** Cloudflare Bot Fight Mode toggle (free, in §3 nice-to-haves). That's it.

### Out-of-scope abuse vectors (not addressing pre-launch)
- **Stem download scraping** — outputs are job-id-scoped UUIDs, no public listing. Low risk.
- **Library scraping** — only 20 Kozelski charts remain (15K deleted Apr 16). No commercial value to scrape.
- **Vapi prank-call burn** — capped at 600s/call after May 7. Can refine later.

---

## §5. NOT-doing list — with reasons

| Item | Why we're skipping |
|---|---|
| **Datadog / New Relic full APM** | $15+/host/mo. Sentry + UptimeRobot + node-exporter cover 90% of value at $0. Revisit at >5 paying customers. |
| **Replacing Plausible** | Already in. Already paid for. Already shows funnel. Don't fix what isn't broken. |
| **Loki / ELK self-hosted log aggregation** | Adds another service to monitor. Solo dev. journalctl + grep is fine until daily volume justifies the operational tax. Sentry handles errors specifically, which is the >90% case. |
| **Custom Slack webhook for every alert** | No Slack workspace for StemScriber. Email + SMS is the established channel per memory. Adding Slack is friction without value. |
| **PagerDuty / Opsgenie** | Solo dev with one phone. Email + SMS to Jeff IS the on-call rotation. PagerDuty's value is escalation chains across teams. |
| **#84 in full (markdown health journal auto-commit every 5 min)** | Creates merge-conflict surface area. journalctl + Sentry + UptimeRobot already capture the underlying signal. Build only the deploy-smoke-test piece (Rec 4) — skip the recurring journal. |
| **#86 Tiers 2-3 (Claude triage + auto-remediation)** | Genuinely premature for a soft launch. Tier 1 (watchdog auto-restart) reduces to "verify systemd `Restart=on-failure`". Save the Claude-driven remediation experiment for Q3 once we have real failure patterns to learn from. |
| **Multi-region failover** | Single VPS is correct for current scale. Hetzner backups + 7-day retention is the recovery story. Revisit at first paying-customer SLA conversation. |
| **Redis-backed Flask-Limiter** | Memory-backed limiter is fine on a single Flask process. Reset-on-restart is acceptable — restarts are rare and the per-IP daily cap (Rec 3) is DB-backed, so the persistent layer is already there for the abuse-critical path. |
| **WAF / Cloudflare Pro plan ($20/mo)** | Free Cloudflare tier + Bot Fight Mode toggle is sufficient. Pro adds image optimization + page rules we don't currently need. |
| **Custom CAPTCHA on /api/upload** | Friction for legitimate Refinery users. Hold in reserve as "kill-switch if Rec 3 gets bypassed." |
| **OpenTelemetry tracing** | Sentry's traces (5% sampling) is sufficient for solo-dev launch. Full OTel is premature. |

---

## Appendix A — Specific thresholds & alert routing summary

| Alert | Threshold | Channel | Where |
|---|---|---|---|
| Queue depth | >4 sustained 15 min | SMS to +18034149454 | Already live (`monitor_queue.py`) |
| Site down | `/health` not 200 for 5 min | Email + SMS | UptimeRobot (Rec 2) |
| New error type | Any | Email | Sentry (Rec 1) |
| Error rate spike | >5 events/min sustained 5 min | Email | Sentry (Rec 1) |
| Anonymous abuse | Single IP-hash >3 uploads/UTC day | 429 returned to client; Sentry breadcrumb | Backend code (Rec 3) |
| Modal spend | >$50/day | Email | Modal dashboard config (15 min) |
| Disk full | >85% used | Email | Grafana/node-exporter (Rec 5) |
| Free RAM | <500MB sustained 10 min | Email | Grafana/node-exporter (Rec 5) |
| Load average | >12 sustained 10 min | Email | Grafana/node-exporter (Rec 5) |
| Deploy broken | Smoke-test fails | Email + GitHub Actions red badge | GitHub Actions (Rec 4) |

**Routing rule reminder:** SMS is reserved for "the show is happening RIGHT NOW" alerts (queue + uptime). Everything else goes to email. Never call 803-414-9454.

---

## Appendix B — Pre-launch implementation order (recommended)

**Week 1 (May 9–15):** Rec 1 (Sentry) + Rec 2 (UptimeRobot). Highest ROI, lowest risk, ~2.5 hr total. Both can be done in one focused afternoon. Verify both fire on synthetic failures before moving on.

**Week 2 (May 16–22):** Rec 3 (anonymous daily cap) + Rec 4 (deploy smoke test). Backend code change + CI workflow. ~4 hr. Test against a fresh anon session to confirm cap at 3rd request.

**Week 3 (May 23–29):** Rec 5 (Grafana + node-exporter) IF bandwidth. ~3 hr. Otherwise ship the bash-cron crude version (~30 min) and call it good.

**Week 4 (May 30–Jun 5):** Cloudflare Bot Fight Mode toggle + Modal spend alert (~30 min total). Then **lock the surface** — no more monitoring changes after this point. Use remaining 2 weeks for UI polish per Jeff's launch-bandwidth priorities.

**Weeks 5–6:** Soak. Watch the alerts fire on synthetic load. Tune thresholds. Don't add new tools.

**Total monitoring/abuse investment to ship:** ~7–10 hr over 4 weeks. Well within the 8-hr-per-item / 6-week budget. Rolling out gradually means each piece gets verified before the next is added.

---

## Appendix C — One-paragraph summary for Jeff

You have queue-depth alerting and rate limiting today. You don't have uptime monitoring or error tracking. The single highest-ROI move is wiring Sentry through the MCP that's already installed but unauthenticated — 2 hours, free tier covers the soft launch comfortably, and it transforms every uncaught exception from "user complains" to "stack-trace email lands in Gmail." Pair that with UptimeRobot (free, 30 min) and tightening anonymous-upload abuse to 3/IP/day (2.5 hr) and you have launch-grade monitoring at $0/mo. Total launch-critical work is about 7 hours. Don't build #84's auto-commit health journal or #86's Claude-driven self-healing — they're shiny premature for soft launch. Build them post-launch when you have real failure patterns to learn from.
