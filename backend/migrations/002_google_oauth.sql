-- Migration 002: Google OAuth support
-- Adds google_id and avatar_url columns, makes password_hash nullable for OAuth-only users.

ALTER TABLE users ADD COLUMN IF NOT EXISTS google_id VARCHAR(255) UNIQUE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;
ALTER TABLE users ALTER COLUMN password_hash DROP NOT NULL;
