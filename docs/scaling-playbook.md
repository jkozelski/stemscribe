# StemScribe Scaling & Launch Playbook
**Last updated:** March 2026

---

## Current State Snapshot

| Component | Current | Notes |
|-----------|---------|-------|
| Server | Flask on Mac M3 Max (48GB RAM, MPS GPU) | Single process, `threaded=True` |
| Tunnel | Cloudflare Tunnel → stemscribe.io | No load balancing |
| Separation | RoFormer via EnhancedSeparator, Demucs fallback | 2-5 min per song, semaphore limits to 1 concurrent job |
| Storage | Local disk + Cloudflare R2 (configured, boto3 client ready) | ~500MB per song with all stems |
| Auth | Beta code system (no user accounts) | JWT + auth blueprint exists but not wired to real user DB |
| Payments | Stripe live — Pro $10/mo, Premium $20/mo | Price IDs in env vars, webhook handler exists |
| Database | JSON files on disk (`load_all_jobs_from_disk()`) | No PostgreSQL/SQLite |
| Queue | `threading.Semaphore(1)` — one separation at a time | No Redis, no Celery |
| CDN | Cloudflare (via tunnel for HTML/JS, R2 presigned URLs for stems) | |
| Users | 10 beta testers | Beta code gating |

---

## Phase 1: 10-50 Users (Current → Next Month)

### What Works As-Is

The M3 Max can handle 10-50 users **if they are not all processing simultaneously**. Key math:

- **Separation time:** 2-5 min per song (average 3.5 min)
- **Semaphore:** 1 concurrent job → max throughput = ~17 songs/hour
- **Daily capacity:** If users spread across 12 active hours = ~200 songs/day
- **50 users × 1 song/day avg = 50 songs/day** → well within capacity
- **Peak concern:** 5+ users hitting "process" within the same 10-minute window → queue backs up to 15-25 min wait

The M3 Max has 48GB unified memory. RoFormer uses ~4-6GB during separation. You have headroom.

### Bottlenecks to Watch

| Bottleneck | Threshold | What Happens |
|------------|-----------|--------------|
| Concurrent separations | 3+ queued | Wait times exceed 10 min, users think it's broken |
| Disk space | ~25GB of stems (50 songs) | Local disk fills if no cleanup |
| Memory | 48GB - 6GB (model) - 4GB (OS) = ~38GB free | Not a concern at this scale |
| Flask threads | 10+ concurrent HTTP requests | Static file serving slows down |
| Upload bandwidth | Cloudflare Tunnel upstream | Tunnel is the bottleneck — typical 50-100 Mbps up |

### Quick Fixes to Implement Now

**1. Request queue with position visibility (1 day)**

The current semaphore blocks silently. Add a visible queue:

```python
# In processing/separation.py
import queue
_job_queue = queue.Queue()
_queue_position = {}  # job_id → position

def get_queue_position(job_id: str) -> int:
    return _queue_position.get(job_id, 0)
```

Add a `/api/queue-position/<job_id>` endpoint so the frontend can poll and show "You are #3 in line, estimated wait: 10 minutes."

**2. Automatic stem cleanup (2 hours)**

The `storage/cleanup.py` module exists. Wire it up:

- Delete local stems older than 7 days (keep R2 copies)
- Run cleanup on a schedule (daily cron or background thread)
- Target: keep local disk under 50GB at all times

```bash
# Add to crontab
0 3 * * * cd ~/stemscribe/backend && ../venv311/bin/python -c "from storage.cleanup import cleanup_old_stems; cleanup_old_stems(days=7)"
```

**3. Basic monitoring (2 hours)**

Add a `/health` endpoint that reports:
- Current queue depth
- Disk space remaining
- Memory usage
- Jobs processed in last 24h
- Active separation (yes/no)

The `routes/health.py` blueprint already exists — extend it with these metrics.

**4. Gunicorn instead of Flask dev server (30 minutes)**

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5555 --timeout 600 app:app
```

4 workers handle static file requests while separation runs in the background. The `--timeout 600` prevents worker kills during long separations.

**Note:** Gunicorn doesn't work natively on macOS with `fork()` + MPS. Use `--preload` and `--worker-class=gthread --threads=4` instead of multiple workers:

```bash
gunicorn -w 1 --threads 8 -b 0.0.0.0:5555 --timeout 600 --preload app:app
```

### When to Worry: Signs the M3 Is Struggling

- Queue consistently has 3+ jobs waiting during peak hours
- Users complain about 15+ minute waits
- Memory usage exceeds 35GB (swap starts thrashing)
- Disk fills past 80%
- `Activity Monitor` shows sustained 100% GPU utilization for hours

**Decision point:** When you see 3+ jobs queued regularly during peak hours (typically evenings/weekends), it's time for Phase 2. At 50 active users processing 2+ songs/day each, you'll hit this.

---

## Phase 2: 50-500 Users (1-3 Months)

### Cloud GPU: Provider Comparison

Stem separation with RoFormer requires a GPU with at least 8GB VRAM. Processing time scales roughly:

| GPU | VRAM | Time per Song | Notes |
|-----|------|---------------|-------|
| M3 Max (current) | 48GB unified | 2-5 min | MPS, slower than CUDA |
| NVIDIA T4 | 16GB | 3-4 min | Cheapest cloud GPU, adequate |
| NVIDIA A10G | 24GB | 1.5-2.5 min | Good price/performance |
| NVIDIA A100 (40GB) | 40GB | 0.5-1.5 min | Overkill for single songs |
| NVIDIA L4 | 24GB | 1.5-2 min | Newer, efficient |

#### Provider Pricing (as of March 2026)

| Provider | GPU | $/hr (on-demand) | $/hr (spot/preemptible) | Cost per Song | Best For |
|----------|-----|-------------------|------------------------|---------------|----------|
| **RunPod** | A10G | $0.37/hr | $0.15/hr | $0.01-0.02 | Cheapest GPU-on-demand, serverless option |
| **RunPod Serverless** | A10G | $0.00038/sec (~$1.37/hr) | N/A | $0.03-0.06 | Zero idle cost, pay per second |
| **Modal** | A10G | $0.000306/sec (~$1.10/hr) | N/A | $0.03-0.05 | Best DX, auto-scaling, cold start ~30s |
| **Modal** | T4 | $0.000164/sec (~$0.59/hr) | N/A | $0.02-0.03 | Budget option |
| **Replicate** | A40 | ~$0.000575/sec | N/A | $0.07-0.17 | Easiest setup, highest per-song cost |
| **AWS g5.xlarge** | A10G (24GB) | $1.006/hr | ~$0.30/hr (spot) | $0.03-0.08 | Enterprise, reserved pricing available |
| **AWS g4dn.xlarge** | T4 (16GB) | $0.526/hr | ~$0.16/hr (spot) | $0.02-0.04 | Budget AWS |
| **GCP g2-standard-4** | L4 (24GB) | $0.838/hr | ~$0.25/hr (spot) | $0.02-0.05 | Google ecosystem |
| **Lambda Labs** | A10 | $0.60/hr | N/A | $0.02-0.04 | Simple, GPU-focused |

**Cost per song calculation:** (GPU hours per song) × (hourly rate). A 4-minute song on an A10G takes ~2 min of GPU time = 0.033 hrs × $0.37 = $0.012 on RunPod.

#### Recommendation: RunPod Serverless or Modal

**For 50-200 users → RunPod Serverless**
- Zero cost when idle (no one is processing)
- Pay per second of actual GPU use
- Cold start of ~30-60 seconds (acceptable — users already wait 2-5 min)
- Deploy your Docker container with RoFormer model baked in
- Cost at 100 users × 1 song/day: ~$3-6/day = **$90-180/month**

**For 200-500 users → Modal**
- Better auto-scaling (0 to N GPUs in seconds)
- Python-native — deploy functions, not containers
- Built-in queue management
- Cost at 500 users × 1 song/day: ~$15-30/day = **$450-900/month**

**Avoid Replicate** at scale — per-prediction pricing is 3-5x more expensive than RunPod/Modal.

**Avoid raw AWS/GCP** unless you want to manage instances yourself. The overhead of autoscaling groups, AMIs, health checks, etc. is not worth it below 1000 users.

### Database: Migrate from JSON to PostgreSQL

**Why now:** JSON files don't support concurrent writes, queries, or user accounts. At 50+ users, you need:
- User accounts with subscription status
- Job history with search/filter
- Usage tracking (songs processed per month per user)
- Billing state synced from Stripe webhooks

**Option A: Managed PostgreSQL (recommended)**

| Provider | Free Tier | Paid | Notes |
|----------|-----------|------|-------|
| **Neon** | 0.5GB storage, 1 project | $19/mo (10GB, autoscale) | Serverless, scales to zero |
| **Supabase** | 500MB, 2 projects | $25/mo (8GB) | PostgreSQL + auth + realtime |
| **Railway** | $5 trial credit | ~$5-15/mo | Simple, good DX |
| **PlanetScale** | Deprecated MySQL | N/A | Don't use — MySQL, and they killed free tier |
| **AWS RDS** | 750 hrs/mo free (12 months) | ~$15-30/mo (db.t3.micro) | Enterprise, more setup |

**Recommendation: Neon or Supabase**

Neon is the best fit:
- Serverless PostgreSQL — scales to zero when idle
- Free tier is enough for Phase 2
- Branch databases for testing
- $19/mo when you outgrow free tier

**Migration plan:**
1. Define schema: `users`, `jobs`, `subscriptions`, `usage` tables
2. Write a migration script that reads all JSON job files and inserts into PostgreSQL
3. Update `models/job.py` to use SQLAlchemy or raw psycopg2 instead of JSON
4. Keep JSON as fallback for 2 weeks, then remove

**Schema sketch:**

```sql
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    stripe_customer_id TEXT UNIQUE,
    plan TEXT DEFAULT 'free',  -- 'free', 'pro', 'premium'
    songs_this_month INT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE jobs (
    id TEXT PRIMARY KEY,  -- your existing job IDs
    user_id UUID REFERENCES users(id),
    title TEXT,
    artist TEXT,
    status TEXT DEFAULT 'pending',  -- pending, processing, complete, failed
    duration_sec FLOAT,
    stems_r2_keys JSONB,  -- {"vocals": "...", "drums": "..."}
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE subscriptions (
    id TEXT PRIMARY KEY,  -- Stripe subscription ID
    user_id UUID REFERENCES users(id),
    plan TEXT NOT NULL,
    status TEXT NOT NULL,  -- active, canceled, past_due
    current_period_end TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

### User Accounts: Google OAuth vs Email/Password

**Recommendation: Start with Google OAuth, add email/password later**

Why Google OAuth first:
- 80%+ of musicians have a Google account
- No password reset flows to build
- No email verification to implement
- 2 hours to implement with Flask-Dance or Authlib
- Your JWT auth blueprint already exists — just add Google as an identity provider

Implementation:
```
pip install Authlib
```
- Register at console.cloud.google.com → OAuth consent screen → credentials
- Callback URL: `https://stemscribe.io/auth/google/callback`
- On successful auth, create user in PostgreSQL, issue JWT
- Store Google `sub` (unique ID) in users table

Add email/password later (Phase 3) for users who don't want Google.

**Alternative: Supabase Auth**
If you pick Supabase for the database, their auth is free and includes Google OAuth, email/password, magic links, and row-level security. This bundles two problems into one solution.

### File Storage: Cloudflare R2

**You already have this built.** The `storage/r2.py` module is complete with upload, download, presigned URLs, and cleanup. Current R2 pricing:

| R2 Component | Price | At 500 users |
|--------------|-------|--------------|
| Storage | $0.015/GB/month | 500 users × 10 songs × 500MB = 2.5TB → **$37.50/mo** |
| Class A ops (writes) | $4.50/million | ~5,000 songs/mo → **$0.02/mo** |
| Class B ops (reads) | $0.36/million | ~50,000 reads/mo → **$0.02/mo** |
| Egress | **Free** | $0 (R2's killer feature) |

**Action items:**
1. Enable R2 upload in the processing pipeline (it's coded but may not be wired into the main flow)
2. After upload to R2, delete local files after 24 hours
3. Serve stems via R2 presigned URLs instead of local Flask file serving
4. Set R2 lifecycle rules: delete stems older than 90 days for free users, keep indefinitely for paid

### Job Queue: Redis + Celery vs Simple Threading

**At 50-200 users: upgrade the threading approach (don't need Redis yet)**

Current semaphore works. Enhance it:
```python
import queue
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class QueuedJob:
    job_id: str
    priority: int = 0  # 0=free, 1=pro, 2=premium
    queued_at: datetime = field(default_factory=datetime.now)

# Priority queue: premium users skip ahead
_job_queue = queue.PriorityQueue()
```

**At 200-500 users: add Redis + RQ (simpler than Celery)**

```bash
pip install rq redis
brew install redis  # or use Upstash for managed Redis
```

RQ (Redis Queue) is simpler than Celery for this use case:
- One queue per priority level
- Workers pull from premium queue first
- Failed jobs automatically retry
- Dashboard via `rq-dashboard`

**Don't use Celery** unless you need scheduled tasks, complex routing, or multiple broker support. RQ does everything StemScribe needs with 1/3 the complexity.

**Managed Redis options:**
- Upstash: Free tier (10,000 commands/day), $0.2/100K commands after
- Railway: ~$5/mo
- Redis Cloud: 30MB free, $5/mo for 250MB

### Monthly Cost Estimates

#### At 100 Users (assuming 60% free, 30% pro, 10% premium)

| Item | Cost |
|------|------|
| GPU (RunPod Serverless, ~100 songs/day) | $90-180/mo |
| R2 storage (500GB growing) | $7.50/mo |
| Database (Neon free tier) | $0/mo |
| Redis (Upstash free tier) | $0/mo |
| Domain + Cloudflare | $0/mo (already have) |
| **Total infrastructure** | **$100-190/mo** |

| Revenue | Monthly |
|---------|---------|
| 30 Pro users × $10 | $300 |
| 10 Premium users × $20 | $200 |
| **Total revenue** | **$500/mo** |
| **Net** | **+$310-400/mo** |

#### At 250 Users

| Item | Cost |
|------|------|
| GPU (RunPod, ~250 songs/day) | $225-450/mo |
| R2 storage (1.25TB) | $18.75/mo |
| Database (Neon $19/mo) | $19/mo |
| Redis (Upstash or Railway) | $5/mo |
| **Total infrastructure** | **$270-495/mo** |

| Revenue (60/30/10 split) | Monthly |
|---------------------------|---------|
| 75 Pro × $10 | $750 |
| 25 Premium × $20 | $500 |
| **Total revenue** | **$1,250/mo** |
| **Net** | **+$755-980/mo** |

#### At 500 Users

| Item | Cost |
|------|------|
| GPU (Modal, ~500 songs/day) | $450-900/mo |
| R2 storage (2.5TB) | $37.50/mo |
| Database (Neon $19/mo) | $19/mo |
| Redis (managed) | $10/mo |
| Monitoring (Sentry free tier) | $0/mo |
| **Total infrastructure** | **$520-970/mo** |

| Revenue (60/30/10 split) | Monthly |
|---------------------------|---------|
| 150 Pro × $10 | $1,500 |
| 50 Premium × $20 | $1,000 |
| **Total revenue** | **$2,500/mo** |
| **Net** | **+$1,530-1,980/mo** |

---

## Phase 3: 500-5,000 Users (3-6 Months)

### Auto-Scaling GPU Workers

**Modal or RunPod Serverless handles this automatically.** Key configuration:

- **Min workers:** 0 (scale to zero when idle — saves money at night)
- **Max workers:** 10 (caps costs, prevents runaway scaling)
- **Scale-up trigger:** Queue depth > 3 or wait time > 2 minutes
- **Scale-down delay:** 5 minutes idle before terminating worker
- **Cold start mitigation:** Keep 1 warm worker during peak hours (9am-11pm user's timezone)

Modal example:
```python
@app.function(
    gpu="A10G",
    image=stem_image,
    min_containers=0,
    max_containers=10,
    scaler="queue_depth",
    timeout=600,
)
def separate_stems(audio_bytes: bytes) -> dict:
    # RoFormer separation logic here
    pass
```

At 1,000 users processing 1 song/day average:
- Peak hour (8-10pm): ~100 songs/hour → need 3-4 concurrent GPUs
- Off-peak: 0-1 GPU
- Monthly GPU cost: **$900-1,800/mo**

### CDN for Stem Delivery

R2 with a Cloudflare Worker in front is the optimal setup:

```
User → Cloudflare CDN edge → R2 (if not cached) → stem file
```

- R2 egress is free (unlike S3)
- Cloudflare caches at 300+ edge locations
- Set `Cache-Control: public, max-age=86400` on stem files (they don't change)
- Use presigned URLs with 24hr expiry for access control

**Implementation:** Create a Cloudflare Worker that validates the user's JWT, then proxies to R2:

```javascript
// workers/stem-proxy.js
export default {
  async fetch(request, env) {
    // Validate JWT from cookie or Authorization header
    // If valid, fetch from R2 and return with cache headers
    // If invalid, return 401
  }
}
```

Cost at 5,000 users: Cloudflare Workers free tier covers 100K requests/day. Beyond that, $5/mo for 10M requests.

### Multi-Region Deployment

**Not needed until 5,000+ users** unless you have a specific international audience. Reasons:

- Cloudflare's CDN already serves static assets globally
- R2 replicates automatically within Cloudflare's network
- The bottleneck is GPU processing time (2-5 min), not network latency (50-200ms)
- API latency for non-GPU calls (library, auth, billing) is <100ms via Cloudflare Tunnel

**When you do need it (5,000+ users with significant EU/Asia traffic):**
- Deploy GPU workers in eu-west (Modal/RunPod support region selection)
- Database: Neon supports read replicas in multiple regions
- API: Deploy Flask app on Railway or Fly.io (multi-region)

### Rate Limiting and Abuse Prevention

Your `middleware/rate_limit.py` with Flask-Limiter already exists. Extend it:

| Endpoint | Free | Pro | Premium |
|----------|------|-----|---------|
| POST /api/process | 3/day | 25/day | Unlimited |
| GET /api/stems/* | 100/hour | 500/hour | 2000/hour |
| POST /auth/* | 5/minute | 5/minute | 5/minute |
| File upload size | 50MB | 200MB | 500MB |

**Abuse prevention checklist:**
- [ ] Rate limit by IP AND by user ID (prevents multi-account abuse)
- [ ] Block disposable email domains on signup (use `disposable-email-domains` npm package or equivalent)
- [ ] Require email verification before first song (prevents bot abuse)
- [ ] Monitor for bulk download patterns (scraping stems)
- [ ] Add CAPTCHA on signup (Cloudflare Turnstile — free)
- [ ] Implement song fingerprinting to detect duplicate uploads (save GPU costs)
- [ ] Block uploads of silence, noise, or very short files (<30 seconds)

### Mobile App: PWA vs Native

See dedicated section below.

---

## Phase 4: 5,000+ Users (6-12 Months)

### Dedicated Infrastructure

At 5,000+ users, you're processing 2,000-5,000 songs/day. Infrastructure becomes a real line item:

| Component | Setup | Monthly Cost |
|-----------|-------|--------------|
| GPU cluster | 5-15 A10G instances (auto-scaling) | $3,000-8,000 |
| Database | Neon Pro or dedicated PostgreSQL | $50-200 |
| Redis | Managed (Upstash Pro or Redis Cloud) | $25-50 |
| R2 storage | 10-25TB | $150-375 |
| Monitoring | Datadog or Grafana Cloud | $0-50 |
| CDN (Cloudflare Pro) | Enhanced security, analytics | $20/mo |
| **Total** | | **$3,250-8,700/mo** |

**Consider reserved GPU instances** at this scale:
- RunPod reserved A10G: ~$0.22/hr (40% savings vs on-demand)
- AWS Reserved g5.xlarge (1-year): ~$0.64/hr (36% savings)
- Modal volume discounts: negotiate directly at >$1K/mo spend

### Team Hiring Needs

| Role | When | Why | Cost (contractor) |
|------|------|-----|-------------------|
| Part-time DevOps | 2,000+ users | Monitoring, scaling, incident response | $3-5K/mo |
| Customer support (part-time) | 1,000+ users | Respond to tickets, manage refunds | $1-2K/mo |
| Content marketer | 500+ users | SEO, social media, tutorial content | $2-4K/mo |
| iOS developer (contract) | If native app decision | Build and maintain iOS app | $8-15K one-time |

**Delay hiring as long as possible.** Use:
- Sentry for automated error reporting (replaces some DevOps)
- Intercom or Crisp for support (free tiers available)
- AI tools for content creation (you already have this capability)
- n8n workflows for automated support responses (you already have this)

### Revenue Projections

**Assumptions:**
- Free-to-paid conversion: 5-10% (industry average for music tools)
- Pro/Premium split among paid: 75% Pro / 25% Premium
- Monthly churn: 5-8% (music SaaS average)
- Annual subscribers: 30% of paid (lower churn)

| Total Users | Paid Users (7.5%) | Pro (75%) | Premium (25%) | Monthly Revenue | Annual Revenue |
|-------------|-------------------|-----------|---------------|-----------------|----------------|
| 1,000 | 75 | 56 × $10 | 19 × $20 | $940 | $11,280 |
| 2,500 | 188 | 141 × $10 | 47 × $20 | $2,350 | $28,200 |
| 5,000 | 375 | 281 × $10 | 94 × $20 | $4,690 | $56,280 |
| 10,000 | 750 | 563 × $10 | 187 × $20 | $9,370 | $112,440 |
| 25,000 | 1,875 | 1,406 × $10 | 469 × $20 | $23,440 | $281,280 |

**Break-even analysis:**

| Scale | Monthly Revenue | Monthly Cost | Net | Break-even? |
|-------|-----------------|--------------|-----|-------------|
| 100 users | $500 | $100-190 | +$310-400 | Yes |
| 500 users | $2,500 | $520-970 | +$1,530-1,980 | Yes |
| 2,500 users | $2,350 | $1,500-3,500 | -$1,150 to +$850 | Marginal |
| 5,000 users | $4,690 | $3,250-8,700 | -$4,010 to +$1,440 | Depends on GPU costs |

**Key insight:** StemScribe is profitable at small scale (100-500 users) because infrastructure costs are low. At 2,500-5,000 users, GPU costs grow faster than revenue unless conversion rates are above 10%. **The critical lever is conversion rate, not user count.**

Ways to improve conversion:
- Generous free tier hooks users (3 songs is good)
- "Aha moment" must happen on first song — quality must be excellent
- Prompt upgrade when free user hits limit ("Your 4th song this month — upgrade for $10/mo")
- Annual pricing with 40% discount reduces effective churn

### Competitive Moat

What keeps users on StemScribe vs Moises, LALAL.AI, etc:

1. **Unique feature combination:** Stem separation + chord detection + Guitar Pro export. No competitor does all three. This is the moat.

2. **Quality:** RoFormer separation quality is competitive with Moises. If you maintain model parity (or beat them with ensemble methods), quality isn't a reason to leave.

3. **Practice-focused UX:** The mixer + practice tab viewer + karaoke mode is purpose-built for learning songs. Moises has practice tools but no transcription. LALAL.AI has no practice tools at all.

4. **Price:** At $10/mo for Pro, you're at market rate. Don't compete on price — compete on the integrated workflow.

5. **Switching cost:** Once a user has a library of processed songs with chord charts and Guitar Pro files, switching to a competitor means re-processing everything and losing their custom annotations.

**Threats to watch:**
- Moises adding transcription (they have the resources, 70M users, $50M funding)
- Meta/Google open-sourcing a better separation model (possible, happened with Demucs)
- Apple adding stem separation to GarageBand/Logic (would crush the consumer market)

**Defensive moves:**
- Build features competitors can't easily copy: custom chord editing, practice tracking, setlist management for gigging musicians
- Own the "learn guitar solos" niche — hyper-specific beats generalist
- Build community features (share arrangements, collaborative practice)

---

## Cost Analysis Summary

| Phase | Users | Monthly Infra | Monthly Revenue | Net |
|-------|-------|---------------|-----------------|-----|
| 1 | 10-50 | $0 (M3 Max) | $0-250 | -$0 to +$250 |
| 2 (early) | 50-100 | $100-190 | $500 | +$310-400 |
| 2 (mid) | 100-250 | $270-495 | $1,250 | +$755-980 |
| 2 (late) | 250-500 | $520-970 | $2,500 | +$1,530-1,980 |
| 3 | 500-2,500 | $1,500-3,500 | $2,500-4,700 | +$1,000-3,200 |
| 4 | 5,000+ | $3,250-8,700 | $4,700-23,400 | +$1,440-14,700 |

**The business is profitable from day one at Phase 2.** The M3 Max is free infrastructure. The moment you start paying for cloud GPUs, you need ~50 paid users to cover costs. At current pricing (Pro $10, Premium $20), that's achievable with 100-200 total users at a 7.5% conversion rate.

---

## Mobile App Decision

### Option 1: PWA (Progressive Web App)

StemScribe already works in mobile Safari and Chrome. A PWA adds:

| Pro | Con |
|-----|-----|
| Zero additional development cost | No App Store discovery |
| No Apple 30% cut on subscriptions | No push notifications on iOS (until iOS 16.4+, now supported) |
| Instant updates (no app review) | Users don't think to "install" web apps |
| Single codebase | Can't access some native APIs (background audio, files) |
| Works on all platforms | Perceived as "less legitimate" than native apps |

**PWA implementation (2-4 hours):**
- Add `manifest.json` with app name, icons, theme color
- Add service worker for offline caching of UI (not stems)
- Add "Add to Home Screen" prompt
- Test on iOS Safari and Android Chrome

### Option 2: Native iOS/Android

| Pro | Con |
|-----|-----|
| App Store discovery (millions browse) | $99/yr Apple Developer Program |
| Push notifications (reliable) | Apple takes 30% of subscriptions (15% after year 1) |
| Better audio playback performance | 2-4 month development time |
| Background audio support | App review process (1-7 days per update) |
| Users trust native apps more | Need to maintain 3 codebases (web, iOS, Android) |

**Apple's 30% cut impact:**
- Pro $10/mo → Apple takes $3 → you get $7
- Premium $20/mo → Apple takes $6 → you get $14
- After year 1 with same subscriber: 15% cut instead
- This effectively cuts your revenue by 15-30%

**App review risks:**
- Apple may reject if app "merely wraps a website"
- Need native-feeling UI components
- Audio processing apps generally pass review without issues
- Subscription apps require "Restore Purchases" button

### Option 3: Hybrid (Capacitor/Ionic) — Recommended

Wrap the existing web app in a native shell using Capacitor:

| Pro | Con |
|-----|-----|
| 90% code reuse from web app | Still pays Apple's 30% cut |
| App Store presence | Some performance overhead vs true native |
| Push notifications via native APIs | Need to handle native quirks per platform |
| 1-2 weeks to ship (not months) | Capacitor bugs can be hard to debug |
| Access to native audio APIs | |

**Implementation:**
```bash
npm install @capacitor/core @capacitor/cli
npx cap init StemScribe io.stemscribe.app --web-dir frontend
npx cap add ios
npx cap add android
npx cap sync
npx cap open ios  # Opens in Xcode
```

**Cost:** $99/yr (Apple) + $25 one-time (Google Play) + 1-2 weeks of dev time.

### Recommendation

**Phase 2 (now-3 months): PWA only**
- Add manifest.json and service worker to the existing web app
- Add "Install App" banner for mobile users
- Zero cost, zero maintenance overhead
- Focus engineering time on core product (separation quality, UI, user accounts)

**Phase 3 (3-6 months): Capacitor hybrid if mobile traffic exceeds 30%**
- Only build native if analytics show significant mobile usage
- Use Capacitor to wrap the web app — 1-2 weeks, not months
- Submit to App Store and Google Play
- Route App Store subscribers through Apple IAP (required), web subscribers through Stripe (keep more revenue)

**Phase 4 (6-12 months): Consider true native only if**
- Mobile is >50% of usage
- Users report performance issues with hybrid
- You need native audio features (background separation, offline mode)

**Anti-pattern to avoid:** Don't build a native app before you have product-market fit. The web app is your testing ground. Building native too early burns months of dev time on a platform that may not matter.

---

## Marketing Strategy

### Content Marketing (Start Now)

**YouTube — Primary Channel**

Musicians learn from video. Create:

1. **"How to learn guitar solos with StemScribe"** (tutorial, 5-10 min)
   - Screen recording: upload song → isolate guitar → slow down → practice
   - Target keywords: "learn guitar solos," "isolate guitar from song," "slow down guitar solo"
   - This is your #1 discovery video

2. **"I isolated every instrument from [popular song]"** (reaction/demo, 3-5 min)
   - Pick songs trending on TikTok/Instagram
   - Show the separation quality
   - End with "try it yourself at stemscribe.io"

3. **Weekly "Stem of the Week"** (short, 60-90 sec)
   - Isolate an interesting guitar riff, bass line, or drum fill
   - Post as YouTube Shorts, Instagram Reels, TikTok simultaneously
   - Format: "You've heard this song 1000 times, but have you heard JUST the bass?"

**Blog / SEO (Start Month 2)**

Target these high-intent keywords:

| Keyword | Monthly Search Volume (est.) | Difficulty | Content |
|---------|------------------------------|------------|---------|
| stem separation | 5,000-10,000 | Medium | "Best Stem Separation Tools in 2026" (own the comparison) |
| isolate vocals from song | 10,000-20,000 | Medium | "How to Isolate Vocals: Complete Guide" |
| guitar tab from audio | 2,000-5,000 | Low | "AI Guitar Tab Transcription: How It Works" |
| slow down guitar solo | 3,000-8,000 | Low | "How to Slow Down Any Guitar Solo Without Changing Pitch" |
| remove vocals from song | 20,000-50,000 | High | "Remove Vocals from Any Song (Free Tool)" |
| learn guitar by ear | 5,000-10,000 | Medium | "Why Learning by Ear is Overrated (Use This Instead)" |

Publish on stemscribe.io/blog. Each post ends with a CTA to try StemScribe.

### Social Proof (Start Now)

1. **Beta tester testimonials:** Ask your 10 testers for a 2-sentence quote + permission to use. Put on landing page.

2. **Before/after demos:** Record a song playing normally → then with just the guitar isolated → then with the chord chart. This visual is compelling.

3. **"Built by a musician for musicians"** — Jeff's story. Indie dev building the tool he wanted. This resonates with the music community.

### Platform Strategy

| Platform | Content Type | Posting Frequency | Why |
|----------|-------------|-------------------|-----|
| YouTube | Tutorials, demos, comparisons | 2/week | Musicians search here first |
| Instagram Reels | 30-60 sec stem reveals | 3-5/week | Visual, shareable |
| TikTok | Same as Reels (cross-post) | 3-5/week | Younger musicians, viral potential |
| Reddit | r/guitar, r/musicproduction, r/WeAreTheMusicMakers | 2-3/week | Answer questions, soft promote |
| Facebook Groups | Guitar learning groups | 1-2/week | Older demographic, high engagement |
| Twitter/X | Updates, behind-the-scenes | Daily | Build dev/indie following |

**Reddit strategy (highest ROI for early stage):**
- Don't spam links. Answer questions genuinely.
- When someone asks "how do I learn this solo?" — explain the process, mention StemScribe naturally
- Post your own demos in r/guitar: "I isolated the guitar from [song] and made a practice track"
- r/guitarlessons, r/Luthier, r/Bass are also relevant

### Partnerships

| Partner Type | Examples | Approach |
|-------------|----------|----------|
| Guitar YouTube channels | JustinGuitar (7M subs), Paul Davids (4M), Marty Music (3M) | Offer free Premium account + affiliate deal ($5/signup) |
| Guitar teachers (local/online) | TakeLessons, Lessonface, local teachers | "Give your students a tool to practice between lessons" — free teacher account |
| Music schools | Berklee Online, School of Rock | Bulk licensing ($5/student/mo) |
| Podcast hosts | Guitar podcasts, music production podcasts | Guest appearance: "How AI is changing guitar practice" |
| Gear review sites | Sweetwater, Guitar World, Premier Guitar | Product review / sponsored content |

**Priority:** YouTube creators. A single mention from a channel with 100K+ subscribers = 500-2,000 signups. Offer them:
- Free lifetime Premium
- 30% affiliate commission (pay $3 per Pro signup, $6 per Premium)
- Early access to new features

### Launch Timeline

**Week 1-2 (Now):**
- [ ] Polish landing page for cold traffic (clear value prop, demo video, pricing)
- [ ] Set up basic analytics (Plausible or PostHog — privacy-friendly)
- [ ] Create stemscribe.io/blog with first SEO article
- [ ] Record first YouTube tutorial video
- [ ] Ask beta testers for testimonials

**Week 3-4:**
- [ ] Launch on Product Hunt (prep: screenshots, demo GIF, 3-sentence description)
- [ ] Post in r/guitar, r/musicproduction (genuine, not spammy)
- [ ] Start Instagram/TikTok posting (3x/week stem reveals)
- [ ] Reach out to 5 guitar YouTube channels for review

**Month 2:**
- [ ] Remove beta code gate — open to public with free tier
- [ ] Implement Google OAuth (user accounts)
- [ ] Publish 4 SEO blog posts
- [ ] Launch referral program ("Give a friend 3 free songs, get 3 bonus songs")
- [ ] Run small Reddit ads ($5/day, r/guitar targeting)

**Month 3:**
- [ ] First YouTube creator partnership live
- [ ] 8+ blog posts indexed in Google
- [ ] PWA manifest live (mobile install prompt)
- [ ] Implement annual pricing with 40% discount
- [ ] Target: 200+ registered users, 20+ paid

**Month 4-6:**
- [ ] Second wave of creator partnerships
- [ ] Add "Share" feature (shareable practice tracks with embedded player)
- [ ] Consider TikTok/Instagram ads ($10-20/day) if organic is working
- [ ] Target: 500-1,000 registered users, 50-100 paid
- [ ] Evaluate native app decision based on mobile traffic data

---

## Decision Checklist

Use these decision criteria at each fork in the road:

### When to move off the M3 Max
- [ ] Queue regularly has 3+ jobs during peak hours (2+ times per week)
- [ ] Users complain about wait times in feedback
- [ ] You need to step away from the Mac (travel, hardware failure)
- → **Action:** Deploy to RunPod Serverless, keep M3 as backup

### When to add a database
- [ ] You have more than 50 registered users
- [ ] You need to query job history (search, filter, analytics)
- [ ] Stripe webhook needs to update user plan status
- → **Action:** Set up Neon PostgreSQL, migrate JSON files

### When to hire help
- [ ] You're spending >10 hours/week on support instead of building
- [ ] Infrastructure incidents happen >1x/month
- [ ] Content creation is bottlenecked on your time
- → **Action:** Hire part-time support first ($1-2K/mo), then content ($2-4K/mo)

### When to build a native app
- [ ] Mobile traffic exceeds 30% of total (check analytics)
- [ ] Users request it (track in feedback)
- [ ] A competitor launches a strong native app
- → **Action:** Start with Capacitor hybrid, not full native

### When to raise prices
- [ ] Conversion rate is above 10% (indicates price is too low)
- [ ] Churn is below 3% monthly (users find it indispensable)
- [ ] You add a significant new feature (batch processing, API access)
- → **Action:** Grandfather existing users, raise for new signups only

---

## Appendix: Quick Reference Commands

```bash
# Start server
cd ~/stemscribe/backend && ../venv311/bin/python app.py

# Run tests
cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/ -v

# Check disk usage
du -sh ~/stemscribe/backend/outputs/

# Check R2 bucket size (requires awscli configured for R2)
aws s3 ls s3://stemscribe-uploads --recursive --summarize --endpoint-url https://$CF_ACCOUNT_ID.r2.cloudflarestorage.com

# Monitor GPU memory (macOS)
sudo powermetrics --samplers gpu_power -i 5000

# Check Cloudflare Tunnel status
cloudflared tunnel info stemscribe
```
