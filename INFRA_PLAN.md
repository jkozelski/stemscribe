# StemScribe Infrastructure Plan
## Auth, Billing, Rate Limiting & Deployment

**Author:** Dev agent
**Date:** 2026-02-28
**Status:** PLAN (not implemented yet)

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Authentication (JWT)](#2-authentication-jwt)
3. [Stripe Billing](#3-stripe-billing)
4. [Rate Limiting by Tier](#4-rate-limiting-by-tier)
5. [Database (Supabase/Postgres)](#5-database-supabasepostgres)
6. [File Storage (Cloudflare R2)](#6-file-storage-cloudflare-r2)
7. [GPU Processing (Replicate)](#7-gpu-processing-replicate)
8. [Deployment (Railway)](#8-deployment-railway)
9. [New Dependencies](#9-new-dependencies)
10. [File-by-File Implementation Order](#10-file-by-file-implementation-order)
11. [Environment Variables](#11-environment-variables)
12. [Cost Projections](#12-cost-projections)
13. [Security Checklist](#13-security-checklist)
14. [Open Questions](#14-open-questions)

---

## 1. Architecture Overview

```
                         ┌──────────────────┐
                         │  Cloudflare Pages │
                         │  (frontend)       │
                         │  stemscribe.app   │
                         └────────┬─────────┘
                                  │ HTTPS
                                  ▼
                         ┌──────────────────┐
                         │  Railway          │
                         │  (Flask API)      │
                         │  Gunicorn + 2w    │
                         │  api.stemscribe   │
                         └──┬──────┬──────┬─┘
                            │      │      │
                   ┌────────┘      │      └────────┐
                   ▼               ▼               ▼
          ┌──────────────┐ ┌─────────────┐ ┌─────────────┐
          │  Supabase    │ │ Cloudflare  │ │ Replicate   │
          │  (Postgres)  │ │ R2 (storage)│ │ (GPU jobs)  │
          │  Auth state  │ │ Stems, MIDI │ │ Demucs      │
          │  Jobs, usage │ │ Uploads     │ │ ~$0.017/run │
          └──────────────┘ └─────────────┘ └─────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Stripe          │
                         │  (billing)       │
                         │  Webhooks → API  │
                         └──────────────────┘
```

**Key decisions:**
- JWT auth managed by our Flask API (not Supabase Auth) for full control
- Supabase used as a Postgres database only (via psycopg2 / SQLAlchemy)
- Cloudflare R2 for all file storage (S3-compatible, free egress)
- Replicate for GPU-heavy work (Demucs separation) — no GPU needed on Railway
- Basic Pitch / CRNN transcription runs on Railway CPU (lightweight enough)

---

## 2. Authentication (JWT)

### Library: `Flask-JWT-Extended`

### Token Strategy

| Token | Lifetime | Storage | Purpose |
|-------|----------|---------|---------|
| Access token | 15 minutes | Memory / `Authorization` header | API authentication |
| Refresh token | 30 days | HTTP-only cookie | Silent token renewal |

### Endpoints

```
POST /auth/register        — email + password → create user, return tokens
POST /auth/login           — email + password → return tokens
POST /auth/refresh         — refresh cookie → new access token
POST /auth/logout          — revoke refresh token (blacklist)
POST /auth/forgot-password — send reset email (via Resend)
POST /auth/reset-password  — token + new password → update
GET  /auth/me              — return current user profile + plan info
```

### Password Handling
- Hash with `bcrypt` (via `passlib`)
- Minimum 8 characters, no other complexity rules (NIST 800-63B)
- Rate limit login attempts: 5/minute per IP

### Token Blacklisting
- Store revoked JTIs in a `token_blacklist` table (Supabase)
- Check on every request via `@jwt.token_in_blocklist_loader`
- Prune expired entries daily (cron or Railway scheduled task)

### New Files
```
backend/auth/
├── __init__.py
├── routes.py          — Blueprint with /auth/* endpoints
├── models.py          — User model, password hashing
├── decorators.py      — @require_plan('premium') decorator
└── email.py           — Password reset emails via Resend API
```

### Protecting Existing Routes
```python
from flask_jwt_extended import jwt_required, get_jwt_identity

@api_bp.route('/api/process', methods=['POST'])
@jwt_required()
def process_audio():
    user_id = get_jwt_identity()
    user = get_user(user_id)
    check_rate_limit(user)  # raises 429 if exceeded
    ...
```

### Free Tier (No Auth Required)
These routes remain open (no JWT needed):
- `GET /` — frontend static files
- `GET /api/status/<job_id>` — job polling (job_id is unguessable UUID)
- `POST /api/process` — allowed without auth, counts as anonymous (3 songs/month by IP)

---

## 3. Stripe Billing

### Products & Prices

| Plan | Monthly | Annual | Stripe Product Name |
|------|---------|--------|---------------------|
| Free | $0 | — | (no product) |
| Premium | $4.99/mo | $39.99/yr (save 33%) | stemscribe_premium |
| Pro | $14.99/mo | $119.99/yr (save 33%) | stemscribe_pro |

### Setup Steps (Stripe Dashboard)
1. Create 2 Products: `StemScribe Premium`, `StemScribe Pro`
2. Each product gets 2 Prices: monthly recurring + annual recurring
3. Store the 4 Price IDs as env vars
4. Register webhook endpoint: `https://api.stemscribe.app/webhooks/stripe`

### Checkout Flow
```
Frontend "Upgrade" button
  → POST /billing/create-checkout-session {plan: "premium", interval: "monthly"}
  → Backend creates Stripe Checkout Session with success_url + cancel_url
  → Returns session URL → frontend redirects to Stripe-hosted checkout
  → User pays → Stripe fires webhook → backend updates user plan
```

### Webhook Events to Handle

| Event | Action |
|-------|--------|
| `checkout.session.completed` | Set user plan, store stripe_customer_id + subscription_id |
| `customer.subscription.updated` | Update plan if changed (upgrade/downgrade) |
| `customer.subscription.deleted` | Revert to free plan |
| `invoice.payment_failed` | Flag user, send email, 3-day grace period |
| `invoice.paid` | Clear any payment failure flags |

### Customer Portal
- Use Stripe's hosted Customer Portal for plan management
- `POST /billing/portal-session` → returns portal URL
- Users can change plan, update payment, cancel — all via Stripe UI

### New Files
```
backend/billing/
├── __init__.py
├── routes.py          — Blueprint: /billing/create-checkout-session, /billing/portal-session
├── webhooks.py        — POST /webhooks/stripe (signature verified)
└── plans.py           — Plan definitions, feature flags per tier
```

### Webhook Signature Verification
```python
import stripe

@billing_bp.route('/webhooks/stripe', methods=['POST'])
def stripe_webhook():
    payload = request.get_data()
    sig_header = request.headers.get('Stripe-Signature')
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, STRIPE_WEBHOOK_SECRET
        )
    except stripe.error.SignatureVerificationError:
        return jsonify({'error': 'Invalid signature'}), 400

    handle_event(event)
    return jsonify({'status': 'ok'}), 200
```

---

## 4. Rate Limiting by Tier

### Tier Feature Matrix

| Feature | Free | Premium ($4.99) | Pro ($14.99) |
|---------|------|-----------------|--------------|
| Songs/month | 3 | 50 | Unlimited |
| Max song length | 5 min | 15 min | 30 min |
| Stem types | 4 (vocal, drum, bass, other) | 6 (+ guitar, piano) | 6 + stereo split |
| Audio output quality | 128kbps MP3 | 320kbps MP3 | WAV lossless |
| Chord recognition | No | Yes | Yes |
| MIDI export | No | Yes | Yes |
| Tab/GP5 export | No | No | Yes |
| Priority queue | No | No | Yes |
| API access | No | No | Yes (future) |

### Implementation

**Song count tracking:**
- `usage` table in Supabase tracks each job per user
- Query: `SELECT COUNT(*) FROM usage WHERE user_id = ? AND created_at >= date_trunc('month', NOW())`
- Anonymous (no auth) users tracked by IP hash in same table

**Rate limit middleware:**
```python
# backend/auth/decorators.py

PLAN_LIMITS = {
    'free':    {'songs_per_month': 3,  'max_duration_sec': 300,  'stems': 4},
    'premium': {'songs_per_month': 50, 'max_duration_sec': 900,  'stems': 6},
    'pro':     {'songs_per_month': -1, 'max_duration_sec': 1800, 'stems': 6},  # -1 = unlimited
}

def check_rate_limit(user):
    """Raises 429 if user has exceeded their plan's monthly limit."""
    limits = PLAN_LIMITS[user.plan]
    if limits['songs_per_month'] == -1:
        return  # unlimited
    count = get_monthly_usage(user.id)
    if count >= limits['songs_per_month']:
        raise RateLimitExceeded(
            f"You've used {count}/{limits['songs_per_month']} songs this month. "
            f"Upgrade to process more."
        )
```

**Song duration check:**
- After upload/download, probe audio duration with `librosa.get_duration()` or `ffprobe`
- Reject before processing if exceeds plan limit
- Return 413 with upgrade prompt

**General API rate limiting:**
- Use `Flask-Limiter` with Redis (Railway addon) or in-memory for MVP
- Global: 60 requests/minute per IP
- Auth endpoints: 5 requests/minute per IP (brute force protection)

---

## 5. Database (Supabase/Postgres)

### Why Supabase (not raw Postgres)
- Free tier: 500MB database, 2 projects
- Managed Postgres with connection pooling (PgBouncer)
- Dashboard for quick queries during development
- Row Level Security available if needed later
- Direct Postgres connection string works with SQLAlchemy

### Connection
```python
# Use SQLAlchemy with the direct Postgres connection string
from sqlalchemy import create_engine
engine = create_engine(os.environ['DATABASE_URL'])
```

For the MVP, use raw SQL via `psycopg2` to keep it simple (no ORM overhead). Migrate to SQLAlchemy if schema grows complex.

### Schema

```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    plan TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'premium', 'pro')),
    stripe_customer_id TEXT UNIQUE,
    stripe_subscription_id TEXT,
    payment_failed_at TIMESTAMPTZ,  -- set on invoice.payment_failed, cleared on invoice.paid
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_stripe_customer ON users(stripe_customer_id);

-- Processing jobs
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    anonymous_ip_hash TEXT,  -- for free-tier anonymous tracking
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'uploading', 'separating', 'transcribing', 'complete', 'failed')),
    source_type TEXT CHECK (source_type IN ('upload', 'url', 'youtube')),
    source_filename TEXT,
    source_url TEXT,
    r2_upload_key TEXT,       -- R2 object key for uploaded audio
    r2_stems_prefix TEXT,     -- R2 prefix for stem files
    r2_midi_key TEXT,         -- R2 key for MIDI output
    r2_gp5_key TEXT,          -- R2 key for Guitar Pro output
    separation_mode TEXT DEFAULT 'roformer',
    stems_json JSONB,         -- {stem_name: r2_key, ...}
    chords_json JSONB,        -- chord analysis results
    duration_seconds REAL,
    processing_time_seconds REAL,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_jobs_user ON jobs(user_id);
CREATE INDEX idx_jobs_status ON jobs(status);
CREATE INDEX idx_jobs_created ON jobs(created_at);

-- Usage tracking (for rate limiting)
CREATE TABLE usage (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    anonymous_ip_hash TEXT,
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    action TEXT NOT NULL CHECK (action IN ('separation', 'transcription', 'chord_analysis', 'export')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_usage_user_month ON usage(user_id, created_at);
CREATE INDEX idx_usage_anon_month ON usage(anonymous_ip_hash, created_at);

-- Token blacklist (for JWT logout)
CREATE TABLE token_blacklist (
    jti TEXT PRIMARY KEY,
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_blacklist_expires ON token_blacklist(expires_at);

-- Migration tracking
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### Migrations
- Plain SQL files in `backend/migrations/` numbered sequentially
- `001_initial_schema.sql`, `002_add_feature_x.sql`, etc.
- Simple Python runner script: `backend/migrate.py`
- No heavy ORM migration framework for MVP

---

## 6. File Storage (Cloudflare R2)

### Why R2
- S3-compatible API (use boto3)
- Zero egress fees (huge savings when users download stems)
- 10GB/month free tier (Class A: 1M ops, Class B: 10M ops)
- Jeff already has a Cloudflare account with R2

### Bucket Structure
```
stemscribe-uploads/
  └── {job_id}/
      ├── source.mp3              — original upload
      ├── stems/
      │   ├── vocals.wav
      │   ├── drums.wav
      │   ├── bass.wav
      │   ├── guitar.wav
      │   ├── piano.wav
      │   └── other.wav
      ├── midi/
      │   ├── drums.mid
      │   ├── bass.mid
      │   ├── guitar.mid
      │   └── piano.mid
      ├── chords.json
      └── output.gp5
```

### Upload Flow (Presigned URLs)
```
1. Frontend → POST /api/upload/start {filename, content_type}
2. Backend generates presigned PUT URL (5 min expiry)
3. Backend returns {upload_url, job_id}
4. Frontend uploads directly to R2 (no backend bandwidth used)
5. Frontend → POST /api/upload/complete {job_id}
6. Backend verifies object exists in R2, starts processing
```

### Download Flow (Presigned URLs)
```
1. Frontend → GET /api/jobs/{job_id}/download/{stem_name}
2. Backend verifies user owns job (or job is anonymous + has job_id)
3. Backend generates presigned GET URL (1 hour expiry)
4. Returns URL → frontend redirects or streams
```

### R2 Client Setup
```python
import boto3

r2 = boto3.client(
    's3',
    endpoint_url=f'https://{CF_ACCOUNT_ID}.r2.cloudflarestorage.com',
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    region_name='auto'
)
```

### Lifecycle / TTL Policy
- Uploads older than 24 hours: delete source audio (save storage)
- Stems older than 30 days: delete (free tier) or keep (paid users)
- Set via R2 lifecycle rules in Cloudflare dashboard

### New Files
```
backend/storage/
├── __init__.py
├── r2.py              — R2 client, upload/download/presign helpers
└── cleanup.py         — TTL enforcement (scheduled task)
```

---

## 7. GPU Processing (Replicate)

### Why Replicate (not self-hosted GPU)
- Pay-per-use: ~$0.017 per Demucs run
- No idle GPU costs
- No Docker/CUDA maintenance
- Cold starts ~15-30s (acceptable for audio processing)
- Scale to zero when no traffic

### Replicate Model
- `cjwbw/demucs` — htdemucs model on Replicate
- Input: audio file URL (R2 presigned GET URL)
- Output: separated stem URLs

### Processing Flow
```
1. Job created, source audio uploaded to R2
2. Backend generates presigned GET URL for source audio
3. Backend calls Replicate API:
   replicate.run("cjwbw/demucs:...", input={"audio": presigned_url})
4. Replicate processes (1-3 min for typical song)
5. Replicate returns stem URLs
6. Backend downloads stems from Replicate → uploads to R2
7. Backend runs transcription locally (CPU, no GPU needed):
   - Basic Pitch for guitar/bass
   - CRNN models for drums/piano
8. Backend uploads MIDI/GP5 to R2
9. Job status → 'complete'
```

### Fallback: Local Processing on Railway
- For transcription (Basic Pitch, CRNNs) — runs on CPU, ~30-60s per stem
- Railway's 8GB RAM is sufficient for inference
- Only Demucs separation needs GPU (sent to Replicate)

### Cost Management
- Track Replicate spend per user in `usage` table
- If Replicate costs exceed budget: queue jobs, batch process
- Future: migrate to Modal or self-hosted RunPod if volume justifies it

### New Files
```
backend/processing/
├── replicate_runner.py    — Replicate API calls for stem separation
└── (existing files for transcription stay the same)
```

---

## 8. Deployment (Railway)

### Railway Project Setup
```
Railway Project: stemscribe-api
├── Service: api (Flask + Gunicorn)
├── Plugin: Redis (for rate limiting / job queue, optional for MVP)
└── Env vars: see Section 11
```

### Dockerfile (Railway auto-detects)
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

EXPOSE 5555

CMD ["gunicorn", "--bind", "0.0.0.0:5555", "--workers", "2", "--timeout", "300", "backend.app:app"]
```

### Why Gunicorn with 2 Workers
- Railway Hobby: 8GB RAM, 8 vCPU
- Each worker ~500MB (PyTorch loaded for CRNN inference)
- 2 workers × 500MB = 1GB, leaving 7GB for processing
- `--timeout 300` — transcription can take up to 5 minutes

### Custom Domain
- `api.stemscribe.app` → Railway custom domain
- SSL handled automatically by Railway

### Health Check
```python
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'version': '2.0.0'})
```

### Procfile (alternative to Dockerfile)
```
web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 300 backend.app:app
```

### Frontend Deployment
- Cloudflare Pages (same as tidepool-artists setup)
- Connect GitHub repo, build from `frontend/` directory
- Custom domain: `stemscribe.app`

---

## 9. New Dependencies

Add to `requirements.txt`:
```
# Authentication
flask-jwt-extended        # JWT token management
passlib[bcrypt]           # Password hashing
email-validator           # Email format validation

# Billing
stripe                    # Stripe API client

# Database
psycopg2-binary           # Postgres driver (Supabase)

# Storage
boto3                     # S3/R2 client

# GPU Processing
replicate                 # Replicate API client

# Rate Limiting
flask-limiter             # Request rate limiting

# Production Server
gunicorn                  # WSGI server for Railway

# Email (password resets)
resend                    # Resend API (already used for tidepool)
```

---

## 10. File-by-File Implementation Order

### Phase A: Database + Auth (2-3 days)
```
1. backend/db.py                    — Supabase/Postgres connection pool
2. backend/migrations/001_initial.sql — Schema from Section 5
3. backend/migrate.py               — Migration runner
4. backend/auth/__init__.py
5. backend/auth/models.py           — User CRUD, password hashing
6. backend/auth/routes.py           — /auth/* endpoints
7. backend/auth/decorators.py       — @jwt_required, @require_plan
8. backend/auth/email.py            — Reset email via Resend
```

### Phase B: Billing (1-2 days)
```
9.  backend/billing/__init__.py
10. backend/billing/plans.py        — Tier definitions, feature flags
11. backend/billing/routes.py       — Checkout + portal session endpoints
12. backend/billing/webhooks.py     — Stripe webhook handler
```

### Phase C: Storage (1 day)
```
13. backend/storage/__init__.py
14. backend/storage/r2.py           — R2 client, presigned URL helpers
15. backend/storage/cleanup.py      — TTL enforcement
```

### Phase D: GPU Processing (1 day)
```
16. backend/processing/replicate_runner.py — Replicate Demucs integration
```

### Phase E: Integration + Deploy (2-3 days)
```
17. Wire auth + billing + storage into existing routes
18. Update frontend: add login/register/upgrade UI
19. Dockerfile + Railway setup
20. Cloudflare Pages for frontend
21. DNS: stemscribe.app → CF Pages, api.stemscribe.app → Railway
22. Stripe webhook registration
23. End-to-end testing
```

**Total estimate: 7-10 working days** (but we're not predicting, just sequencing).

---

## 11. Environment Variables

### Railway (API server)
```bash
# Database
DATABASE_URL=postgresql://user:pass@db.xxx.supabase.co:5432/postgres

# JWT
JWT_SECRET_KEY=<generate 64 char random>
JWT_ACCESS_TOKEN_EXPIRES=900         # 15 minutes
JWT_REFRESH_TOKEN_EXPIRES=2592000    # 30 days

# Stripe
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_PREMIUM_MONTHLY=price_xxx
STRIPE_PRICE_PREMIUM_ANNUAL=price_xxx
STRIPE_PRICE_PRO_MONTHLY=price_xxx
STRIPE_PRICE_PRO_ANNUAL=price_xxx

# Cloudflare R2
CF_ACCOUNT_ID=dcd390562dd56ea565ae735f112bf60b
R2_ACCESS_KEY_ID=<create in CF dashboard>
R2_SECRET_ACCESS_KEY=<create in CF dashboard>
R2_BUCKET_NAME=stemscribe-uploads

# Replicate (GPU)
REPLICATE_API_TOKEN=r8_...

# Email (password resets)
RESEND_API_KEY=re_BFpDdj9b...

# App
FLASK_ENV=production
APP_URL=https://stemscribe.app
API_URL=https://api.stemscribe.app
```

### Cloudflare Pages (Frontend)
```bash
API_URL=https://api.stemscribe.app
STRIPE_PUBLISHABLE_KEY=pk_live_...
```

---

## 12. Cost Projections

### At Launch (0-100 users)
| Service | Monthly Cost |
|---------|-------------|
| Railway (Hobby) | $5 |
| Supabase (Free) | $0 |
| Cloudflare R2 (Free tier) | $0 |
| Cloudflare Pages | $0 |
| Replicate (~500 songs) | $8.50 |
| Stripe fees | ~$3 (on ~$50 MRR) |
| Domain (stemscribe.app) | ~$1 |
| **Total** | **~$18/mo** |

### At 500 Users (~100 paid)
| Service | Monthly Cost |
|---------|-------------|
| Railway (Pro) | $20 |
| Supabase (Free, approaching limit) | $0 |
| Cloudflare R2 | ~$3 |
| Replicate (~5,000 songs) | $85 |
| Stripe fees | ~$30 (on ~$600 MRR) |
| **Total** | **~$138/mo** |
| **Revenue** | **~$600 MRR** |

### At 2,000 Users (~400 paid)
| Service | Monthly Cost |
|---------|-------------|
| Railway (Pro) | $20 |
| Supabase (Pro) | $25 |
| Cloudflare R2 | ~$10 |
| Replicate (~20,000 songs) | $340 |
| Stripe fees | ~$120 (on ~$2,400 MRR) |
| **Total** | **~$515/mo** |
| **Revenue** | **~$2,400 MRR** |

GPU costs dominate. At ~2,000 users, consider:
- Moving to Modal (reserved GPU pricing)
- Self-hosted RunPod with reserved instances
- Caching: if same song is processed twice, serve cached stems

---

## 13. Security Checklist

- [ ] JWT secret key: 64+ random characters, never in code
- [ ] Stripe webhook signature verification on every event
- [ ] bcrypt password hashing (cost factor 12)
- [ ] HTTPS only (Railway and Cloudflare handle this)
- [ ] CORS: restrict to `stemscribe.app` and `localhost` (dev)
- [ ] R2 presigned URLs: short expiry (5 min upload, 1 hour download)
- [ ] SQL injection prevention: parameterized queries only
- [ ] Rate limiting on auth endpoints (5/min per IP)
- [ ] No credentials in code or Git (env vars only)
- [ ] Stripe keys: use test keys during development (sk_test_*)
- [ ] Password reset tokens: single-use, 1-hour expiry
- [ ] Input validation on all user-supplied data
- [ ] File type validation before processing (audio MIME types only)

---

## 14. Open Questions

1. **Domain:** Is `stemscribe.app` available and purchased? Need to register before deploy.

2. **Supabase Auth vs. Custom JWT:** This plan uses custom JWT for full control. Supabase Auth could handle login/register/OAuth for free, but locks us into their token format. Recommendation: custom JWT is better for a product where we need fine-grained plan-based access control.

3. **OAuth Providers:** Should we support Google/GitHub login at launch? Adds ~1 day of work. Recommend: defer to post-launch; email/password is sufficient for MVP.

4. **Free Tier Anonymous Processing:** Currently the app works without login. Should we keep that? Recommendation: yes, allow 3 free songs/month tracked by IP hash. This lowers friction for first-time users and drives conversions.

5. **Replicate vs. Modal vs. RunPod:** Replicate is simplest for MVP. Modal offers better pricing at scale. RunPod is cheapest but requires Docker maintenance. Recommend: start Replicate, migrate to Modal if GPU costs exceed $200/month.

6. **Redis:** Railway offers a Redis plugin ($5/mo). Useful for Flask-Limiter and job queuing. For MVP, in-memory rate limiting + threading is sufficient. Add Redis when we need Celery for job queues.

7. **Email Provider:** Resend is already set up for Tidepool. Reuse it for StemScribe? Or set up a separate Resend domain for `stemscribe.app`? Recommend: separate domain, same Resend account.

8. **Monitoring:** Sentry (free tier) for error tracking, UptimeRobot for health checks. Add during Phase E deployment.

---

## Appendix: Quick-Start Commands

```bash
# Generate JWT secret
python -c "import secrets; print(secrets.token_urlsafe(64))"

# Create R2 bucket
npx wrangler r2 bucket create stemscribe-uploads

# Create R2 API token (in CF dashboard)
# Dashboard → R2 → Manage R2 API Tokens → Create API Token
# Permissions: Object Read & Write
# Specify bucket: stemscribe-uploads

# Test Replicate
pip install replicate
REPLICATE_API_TOKEN=r8_... python -c "
import replicate
output = replicate.run('cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81571f6c6c65f9f4',
    input={'audio': open('test.mp3', 'rb')})
print(output)
"

# Railway deploy
npm install -g @railway/cli
railway login
railway init
railway up
```
