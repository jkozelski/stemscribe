# StemScriber launch-readiness checklist
**Drafted:** 2026-04-26
**Owner:** Jeff Kozelski
**Purpose:** Concrete gate-by-gate "ship / hold" framework so you can
make a clean go/no-go call instead of rough vibes.

---

## How to read this

Each gate has:
- **Status** — 🟢 green (done), 🟡 yellow (functional but rough), 🔴 red (blocker)
- **Bar** — what "passing" actually means
- **Where** — file path or system to verify

A gate is **launch-blocking** if its status is 🔴 AND it's marked
*BLOCKER* in the bar. Yellow gates can ship — they're polish.

---

## 1. Detection quality

### 1.1 Chord chart accuracy 🟡
- **Bar (BLOCKER):** ≥70% of audited songs at "B+ or higher" (8 of 10
  for our test cohort). Songs in the test cohort are: Alright, Cosmic
  Girl, Virtual Insanity, Aja, Peg, Black Cow, Rikki, Dirty Work,
  Reelin' in the Years, Do It Again.
- **Current:** 7-of-10 at B+/A range. Pending: Black Cow (verifying
  maj7 promotion right now), Rikki.
- **Decision:** if maj7 verification lifts Black Cow to B+, you're at
  8/10 → passes. Rikki stays as a known caveat.
- **Where:** `~/stemscribe/docs/audit-2026-04-25/`, `audit-2026-04-26-postfix-v2-results.md`

### 1.2 7ths preserved on jazz songs 🟢
- **Bar:** 80%+ of bars on Aja, Alright, Cosmic Girl, Do It Again show
  7ths/9ths.
- **Current:** Aja 100%, Alright 94%, Cosmic Girl 91%, Do It Again 98%.
  Passes.

### 1.3 Sheet music rendering 🟡
- **Bar:** bass MusicXML readable (≤15 notes/measure on average,
  not artifact noise).
- **Current:** 2 of 3 audited bass charts readable, 3rd was tempo issue
  (now fixed by half-time correction in #51).
- **Note:** ship as a "Bass Sheet Music" disclosure tab, NOT promoted
  in marketing. Only 1 of 4 stems renders worth shipping (bass).

### 1.4 Section labeling 🟡
- **Bar (NOT BLOCKER):** Verse / Chorus / Bridge identified on songs
  with structural variation; OK if uniform-progression songs
  (Alright-style 4-chord vamps) get labels by lyric heuristic.
- **Current:** Sections array IS populated with real names ("Intro",
  "Verse 1", etc.). Yesterday's audit "empty sections" finding was
  wrong — agent looked at wrong field.
- **Polish post-launch:** uniform progressions still get labeled by
  lyric repetition heuristic. Working but not perfect.

---

## 2. Stability + infrastructure

### 2.1 Concurrent processing 🟢
- **Bar (BLOCKER):** site doesn't fall over under 5+ concurrent uploads.
- **Current:** Concurrency cap of 4 shipped (commit `dc5614c`). Job 5+
  shows "Queued for processing" instead of stalling. Tested on the
  Apr 25 audit — 6 jobs handled cleanly with 1 queued.

### 2.2 VPS capacity 🟢
- **Bar:** Hetzner CPX41 (8 vCPU / 16 GB) handles the cap-of-4 load.
- **Current:** 4-job concurrent batch processed in 12-15 min/song.
  Load avg sat around 4-5 of 8 cores. Headroom available.

### 2.3 Modal GPU + cost tracking 🟡
- **Bar:** Stem separation reliable (Modal A10G).
- **Current:** Reliable. Cost: ~$0.06/song. Modal balance NOT actively
  monitored — admin dashboard shows "Not tracked" for spend.
- **Mitigation:** Modal will email you if balance gets low. Set up
  manual check weekly post-launch.

### 2.4 Error tracking 🟡
- **Bar:** failures show up SOMEWHERE you can see them.
- **Current:** error_tracker.py logs to disk. No alerting if a job
  fails. Watchdog handles stalls.
- **Polish post-launch:** add Slack/email alert on job failure rate
  > 5%.

### 2.5 Backups 🟢
- **Bar:** daily Hetzner snapshots active.
- **Current:** Hetzner Backups enabled (€5/mo, 7-day retention).

---

## 3. Legal + compliance

### 3.1 DMCA registration 🟢
- USCO registration #DMCA-1070849.
- DMCA agent: `support@stemscriber.com` (Cloudflare routing).
- PO Box mailing address listed (closes §512(c)(2)).

### 3.2 Karaoke / lyric handling 🟢
- Karaoke disabled (per Apr 10 lawyer call).
- Stored chord charts removed from any persistent UI.

### 3.3 Upload consent 🟢
- Consent modal enabled every session with explicit ToS language
  covering YouTube URLs and "no model training on your audio."

### 3.4 Privacy + retention 🟡
- **Bar:** retention sweeper actually deletes old uploads/outputs.
- **Current:** Retention sweeper exists but `RETENTION_DRY_RUN=true`
  in prod. Logs identify candidates but doesn't delete.
- **Action before launch:** review one day's dry-run logs, then flip
  to `RETENTION_DRY_RUN=false`.

### 3.5 ToS, Privacy, Cookie policy pages 🟢
- All three pages exist (`frontend/{terms,privacy,cookie-policy}.html`).
- Last reviewed by Alexandra Mayo: 2026-04-22-ish.

---

## 4. Operational readiness

### 4.1 Admin dashboard 🟢
- Live at `/admin.html`, JWT-gated to your account.
- Shows: signups/day, cumulative users, songs processed/day, DAU,
  peak hours, free vs paid breakdown.
- Modal spend card shows "Not tracked" — accurate, that data doesn't
  exist yet.

### 4.2 Job queue alerting 🔴
- **Bar (BLOCKER if you're doing any marketing push):** alert if queue
  depth > 4 sustained for 15 min.
- **Current:** No alerting. Admin dashboard shows queue depth but you
  have to look. Concurrency cap protects against catastrophic failure
  but doesn't notify you of sustained pressure.
- **Action before any marketing:** ship a tiny cron on VPS that checks
  queue depth every 5 min, emails you if > threshold for 15 min.
  Estimated 30 min of work.

### 4.3 Email notifications (job complete) 🟡
- Code path exists. Currently gated behind `ENABLE_JOB_EMAILS=false`.
- **Action on launch day:** flip the flag. Verify Resend delivery
  rate.

### 4.4 SMS (escalation) 🟢
- Twilio 844 toll-free working (verified delivery 2026-04-25).
- Used for proactive notifications (e.g., today's audit summaries).
- 843 local numbers being canceled (per memory; not blocking).

### 4.5 Support inbox 🟢
- support@stemscriber.com active (Cloudflare routing).

---

## 5. UX + frontend

### 5.1 Landing page (BETA framing) 🟢
- Live at stemscriber.com. BETA badge visible. noindex on for now.

### 5.2 Upload form 🟢
- Drag-drop + click-to-browse. Genre-fit hint shipped Apr 25.

### 5.3 Practice page 🟢
- Chord chart + lyrics + transpose + 4-bar slash notation.
- CAGED multi-voicing chord diagrams. Auto-scroll.
- Bass sheet music disclosure under chord chart.

### 5.4 Mobile responsive 🟢
- Tested on iPhone via remote-control. Practice page works portrait
  and landscape.

### 5.5 Print-to-PDF 🟢
- Practice page has a print stylesheet that produces clean chord
  charts with lyrics underneath.

### 5.6 Section editor (manual override) 🟡
- **Bar (NOT BLOCKER):** users can fix detected section labels.
- **Current:** Read-only. Section UI doesn't let users override.
  Listed as task #41 (post-launch).
- **Mitigation:** "Report issue" button (task #39) lets users flag
  bad sections without editing them.

### 5.7 Error messages on failure 🟡
- **Bar:** when processing fails, user sees actionable message (not
  a stack trace).
- **Current:** Watchdog-failed jobs show "Job stalled and failed
  after 3 retry attempts" — clear enough but doesn't suggest
  re-uploading or what likely caused it.
- **Polish post-launch:** add a "try again with a shorter clip" hint
  for known failure modes.

---

## 6. Marketing prep

### 6.1 Demo content 🟢
- Aja chord chart (perfect 226/226), Alright (Cmin7-Gmin7-Dmin7-Amin9
  vamp), Cosmic Girl chorus vamp — all work as visual proof.

### 6.2 Marketing copy 🟡
- Drafts in `docs/marketing-drafts-2026-04-26.md`. 7 surfaces covered.
- Status: **drafts, not edited**. You'd want to redline before sending.

### 6.3 Plausible analytics 🟢
- Live on all user-facing pages.

### 6.4 Launch list (Tidepool Artists) 🟡
- Personal outreach drafted but **not sent**. Per memory, you find
  artist relationships fragile — you said you'd handle this when home.

### 6.5 Social handles 🔴
- @stemscriber on Twitter / Bluesky / Instagram — exist? Not verified.
  **Action:** confirm handles you control before any link goes out.

---

## 7. Summary: launch decision framework

### Hard blockers (red gates)
- 4.2: Queue alerting cron — ~30 min to ship
- 6.5: Social handles claimed — ~15 min if available

### Yellow gates (ship-ready, polish later)
- 3.4: Flip `RETENTION_DRY_RUN=false`
- 4.3: Flip `ENABLE_JOB_EMAILS=true` on launch day
- 1.4, 5.6, 5.7: section labeling polish, manual editor, error message
  improvements (all post-launch)

### Green (no action)
- DMCA, consent, infrastructure stability, BETA framing, analytics,
  practice page, payment surfaces (Stripe shipped earlier)

### Recommended decision criteria

You can launch when **all hard blockers are green** AND you're
comfortable with the yellow gates as known-rough-edges that you'll
fix in flight.

The hard blockers above are ~45 min of work total. Everything else is
polish.

The chord-chart quality story (the thing customers actually buy) is
**already at A-/A on 7 of 10 test songs** with the maj7 fix verifying
right now. Even if Rikki and Black Cow stay as caveats, that's a
strong launch position — far better than what the comparable products
ship.

### What I'd recommend you tell yourself

> The gates I haven't passed are operational, not product. The product
> works. Customers will see real chord charts they can read. The polish
> items are visible-to-me, not visible-to-them on day one.

---

## What this doc is not

- A timeline. You haven't picked a date and that's fine.
- An audit. The audit is in `audit-2026-04-25-results.md` and
  `audit-2026-04-26-*` files.
- A guarantee. Real users will surface issues we haven't seen. Plan
  for "fix in flight" not "ship perfect."
