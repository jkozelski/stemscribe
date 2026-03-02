-- StemScribe Initial Schema
-- Migration 001: Users, jobs, usage tracking, token blacklist

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT,
    plan TEXT NOT NULL DEFAULT 'free' CHECK (plan IN ('free', 'premium', 'pro')),
    stripe_customer_id TEXT UNIQUE,
    stripe_subscription_id TEXT,
    payment_failed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_stripe_customer ON users(stripe_customer_id);

-- Processing jobs
CREATE TABLE IF NOT EXISTS jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    anonymous_ip_hash TEXT,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'uploading', 'separating', 'transcribing', 'complete', 'failed')),
    source_type TEXT CHECK (source_type IN ('upload', 'url', 'youtube')),
    source_filename TEXT,
    source_url TEXT,
    r2_upload_key TEXT,
    r2_stems_prefix TEXT,
    r2_midi_key TEXT,
    r2_gp5_key TEXT,
    separation_mode TEXT DEFAULT 'roformer',
    stems_json JSONB,
    chords_json JSONB,
    duration_seconds REAL,
    processing_time_seconds REAL,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at);

-- Usage tracking for rate limiting
CREATE TABLE IF NOT EXISTS usage (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    anonymous_ip_hash TEXT,
    job_id UUID REFERENCES jobs(id) ON DELETE SET NULL,
    action TEXT NOT NULL CHECK (action IN ('separation', 'transcription', 'chord_analysis', 'export')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_usage_user_month ON usage(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_usage_anon_month ON usage(anonymous_ip_hash, created_at);

-- JWT token blacklist (for logout / revocation)
CREATE TABLE IF NOT EXISTS token_blacklist (
    jti TEXT PRIMARY KEY,
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_blacklist_expires ON token_blacklist(expires_at);

-- Schema migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Record this migration
INSERT INTO schema_migrations (version, name) VALUES (1, '001_initial_schema')
ON CONFLICT (version) DO NOTHING;
