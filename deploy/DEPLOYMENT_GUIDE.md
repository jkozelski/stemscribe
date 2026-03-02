# StemScribe Deployment Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     USERS                                │
│              (musicians, students, producers)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FRONTEND (Vercel / Cloudflare Pages)         │
│              - index.html (Neve console mixer)            │
│              - practice.html (tab viewer)                 │
│              - Landing page + pricing                     │
└────────────────────┬────────────────────────────────────┘
                     │ API calls
                     ▼
┌─────────────────────────────────────────────────────────┐
│              API SERVER (Railway / Render / Fly.io)       │
│              - Flask app (app.py)                         │
│              - Job queue (Redis + Celery)                 │
│              - Auth (JWT tokens)                          │
│              - Stripe webhook handler                     │
│              - Usage tracking / rate limiting              │
└────────────────────┬────────────────────────────────────┘
                     │ GPU processing jobs
                     ▼
┌─────────────────────────────────────────────────────────┐
│              GPU WORKERS (Replicate / Modal / RunPod)     │
│              - Demucs stem separation                     │
│              - Basic Pitch MIDI transcription              │
│              - Chord model inference                      │
└────────────────────┬────────────────────────────────────┘
                     │ results
                     ▼
┌─────────────────────────────────────────────────────────┐
│              STORAGE (Cloudflare R2 / S3)                │
│              - Temporary upload storage (24hr TTL)        │
│              - Processed stems (30-day TTL)                │
│              - MIDI/MusicXML exports                      │
└─────────────────────────────────────────────────────────┘
```

## Option A: Budget Deploy (~$30-50/month)

Best for: MVP launch, first 100-500 users

| Component | Service | Cost |
|-----------|---------|------|
| Frontend | Cloudflare Pages | Free |
| API Server | Railway (Hobby) | $5/mo |
| GPU Processing | Replicate (pay-per-use) | ~$0.016/song |
| Database | Supabase (free tier) | Free |
| File Storage | Cloudflare R2 (10GB free) | Free |
| Payments | Stripe (2.9% + $0.30/txn) | Per transaction |
| Domain | stemscribe.app | ~$12/year |
| Email | Brevo (free tier) | Free |
| **TOTAL** | | **~$20 + GPU costs** |

**GPU cost estimate at scale:**
- 100 users/month × 5 songs each = 500 songs → $8/mo
- 1,000 users/month × 10 songs each = 10,000 songs → $160/mo
- 5,000 users/month × 15 songs each = 75,000 songs → $1,200/mo

## Option B: Scale Deploy (~$100-300/month)

Best for: 500-5,000 users, production-ready

| Component | Service | Cost |
|-----------|---------|------|
| Frontend | Vercel Pro | $20/mo |
| API Server | Railway Pro | $20/mo |
| GPU Processing | Modal (reserved GPU) | ~$50-150/mo |
| Database | Supabase Pro | $25/mo |
| File Storage | Cloudflare R2 | ~$5/mo |
| CDN | Cloudflare | Free |
| Monitoring | Sentry | Free tier |
| **TOTAL** | | **~$120-220/mo** |

## Option C: Self-Hosted GPU (~$80-150/month)

Best for: Maximum control, lowest per-song cost

| Component | Service | Cost |
|-----------|---------|------|
| GPU Server | RunPod (RTX 4090 spot) | ~$0.39/hr on-demand |
| API Server | Same machine or Fly.io | $5-20/mo |
| Everything else | Same as Option A | ~$20/mo |

## Step-by-Step Setup

### 1. Domain & DNS
```bash
# Register stemscribe.app (or .com, .io)
# Point DNS to Cloudflare (free plan)
# Set up SSL (automatic with Cloudflare)
```

### 2. Frontend Deployment (Cloudflare Pages)
```bash
# Option: Push frontend to GitHub, connect to Cloudflare Pages
cd ~/stemscribe/frontend
# Create a simple build script or deploy as static files
```

### 3. API Server (Railway)
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and create project
railway login
railway init

# Set environment variables
railway variables set STRIPE_SECRET_KEY=sk_live_...
railway variables set STRIPE_WEBHOOK_SECRET=whsec_...
railway variables set JWT_SECRET=your-secret-here
railway variables set REPLICATE_API_TOKEN=r8_...
railway variables set DATABASE_URL=postgres://...

# Deploy
railway up
```

### 4. GPU Backend (Replicate)
```bash
# Install Replicate CLI
pip install replicate

# Test Demucs on Replicate
python3 -c "
import replicate
output = replicate.run(
    'cjwbw/demucs:25a173108cff36ef9f80f854c162d01df9e6528be175794b81571f6c6c65f9f4',
    input={'audio': open('test.mp3', 'rb')}
)
print(output)
"
```

### 5. Stripe Setup
```
1. Create Stripe account at stripe.com
2. Create Products:
   - StemScribe Premium ($4.99/mo)
   - StemScribe Pro ($14.99/mo)
   - StemScribe Premium Annual ($39.99/yr) -- save 33%
   - StemScribe Pro Annual ($119.99/yr) -- save 33%
3. Set up webhook endpoint: https://api.stemscribe.app/webhooks/stripe
4. Listen for events:
   - checkout.session.completed
   - customer.subscription.updated
   - customer.subscription.deleted
   - invoice.payment_failed
```

### 6. Database Schema (Supabase/Postgres)
```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    plan TEXT DEFAULT 'free' CHECK (plan IN ('free', 'premium', 'pro')),
    stripe_customer_id TEXT,
    stripe_subscription_id TEXT,
    songs_this_month INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Processing Jobs
CREATE TABLE jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'complete', 'failed')),
    source_url TEXT,
    source_filename TEXT,
    stems_url TEXT,  -- R2 storage URL
    midi_url TEXT,
    chords_json JSONB,
    processing_time_seconds REAL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Usage tracking for rate limiting
CREATE TABLE usage (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    job_id UUID REFERENCES jobs(id),
    action TEXT NOT NULL,  -- 'separation', 'transcription', 'chord_analysis'
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## Rate Limiting by Tier

| Feature | Free | Premium | Pro |
|---------|------|---------|-----|
| Songs/month | 3 | Unlimited | Unlimited |
| Stem types | 4 (vocal, drum, bass, other) | 6 (+ guitar, keys) | 6 + stereo split |
| Max song length | 5 min | 15 min | 30 min |
| Audio quality | 128kbps MP3 | 320kbps MP3 | WAV lossless |
| Chord recognition | No | Yes | Yes + all keys |
| MIDI export | No | Yes | Yes |
| Tab/notation | No | No | Yes |
| Priority queue | No | No | Yes |

## Monitoring Checklist

- [ ] Sentry for error tracking (free tier)
- [ ] UptimeRobot for uptime monitoring (free tier)
- [ ] Stripe Dashboard for revenue tracking
- [ ] Plausible or PostHog for analytics (privacy-friendly)
- [ ] GPU cost monitoring (set budget alerts)
