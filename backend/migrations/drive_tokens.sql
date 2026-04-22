-- Google Drive per-user OAuth tokens
-- Run this in Supabase SQL editor before deploying the Drive export feature.

CREATE TABLE IF NOT EXISTS user_drive_tokens (
    user_id        UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    access_token   TEXT,
    refresh_token  TEXT,
    email          TEXT,
    expires_at     TIMESTAMP WITH TIME ZONE,
    created_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- CSRF nonce for in-flight OAuth round-trips (short-lived, single-use)
CREATE TABLE IF NOT EXISTS user_drive_oauth_state (
    user_id     UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    nonce       TEXT NOT NULL,
    created_at  TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Row-Level Security: only the user themselves (or service role) can access
ALTER TABLE user_drive_tokens ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_drive_oauth_state ENABLE ROW LEVEL SECURITY;

-- Backend uses service_role key, so policies are mostly for completeness
CREATE POLICY "Users can read their own drive tokens"
    ON user_drive_tokens FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own drive tokens"
    ON user_drive_tokens FOR DELETE
    USING (auth.uid() = user_id);

-- Index for cleanup of abandoned OAuth state (rows older than 1 hour should be pruned)
CREATE INDEX IF NOT EXISTS idx_drive_oauth_state_created_at ON user_drive_oauth_state (created_at);
