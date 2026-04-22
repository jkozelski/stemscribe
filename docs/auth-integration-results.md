# Auth Integration Results

**Date:** 2026-03-19

## Overview

Full authentication, per-user library, and download system built and integrated into StemScriber. All changes are backward compatible with existing users and data. 318 tests passing with no regressions.

---

## Google OAuth (Backend)

- **POST /auth/google** endpoint — verifies Google ID tokens, creates or links users, returns JWT
- User model extended with `google_id` and `avatar_url` columns
- Migration SQL: `backend/migrations/002_google_oauth.sql`
- Backward compatible — existing email/password auth unchanged
- Setup guide: `docs/google-oauth-setup.md`

## Per-User Libraries

- `ProcessingJob` now has `user_id` and `session_id` fields
- Jobs tagged with user on creation (upload and URL processing)
- **GET /api/library** filters by user (signed-in users see their own, anonymous users see session-based)
- `?all=true` admin override for Jeff to see all songs
- **POST /api/library/claim** — signed-in users can claim unclaimed jobs
- **DELETE** authorization — only owners can delete
- Session cookie (24h) for anonymous user tracking
- Backward compatible — existing jobs load without `user_id`

## Download System

- **GET /api/download/{job_id}/zip** — creates in-memory ZIP with all stems, MIDI, and GP files
- **GET /api/download/{job_id}/stem/{name}/mp3** — WAV-to-MP3 conversion via ffmpeg, cached
- 320kbps for Pro/Premium, 128kbps for free
- Frontend: "DL MP3" button per stem in mixer
- Frontend: "Download All (ZIP)" button after stems grid
- New CSS classes: `.stem-btn.mp3-dl`, `.download-toolbar`, `.download-all-btn`

## Frontend Auth UI

- `js/auth.js` — Google Sign-In, token management, profile UI
- `css/auth.css` — sign-in button, profile dropdown, save-prompt modal, beta code entry
- `landing.html` — sign-in button in nav, profile dropdown
- `index.html` — sign-in in header, save-prompt modal after processing
- `practice.html` — sign-in in header
- **GET /api/config** — exposes `GOOGLE_CLIENT_ID` to frontend
- Auto-checks auth on page load, auto-select for returning users

## Stripe + Beta Integration

- Beta plan added to `PLAN_HIERARCHY` and `PLAN_LIMITS` (same as Pro features, $0)
- Beta plan added to billing `PLANS` dict
- Beta code redemption now updates user's plan in database when authenticated
- Billing routes work with Google OAuth users (same JWT tokens)
- Stripe checkout, portal, webhooks — all pre-existing and compatible

## Tests

- **318 tests passing** (316 existing + 2 fixed billing tests)
- No regressions
- Google OAuth, library filtering, and downloads all integrate with existing test suite

---

## Files Changed

### Backend
| File | Change |
|------|--------|
| `auth/models.py` | `google_id`, `avatar_url`, new Google user functions |
| `auth/routes.py` | POST /auth/google endpoint |
| `auth/decorators.py` | Beta plan in hierarchy and limits |
| `models/job.py` | `user_id`, `session_id` on ProcessingJob |
| `routes/api.py` | ZIP download, MP3 download, /api/config, session tagging |
| `routes/library.py` | Per-user filtering, claim endpoint, delete auth |
| `routes/beta.py` | User account linking on redemption |
| `billing/plans.py` | Beta plan definition |
| `app.py` | CSP headers for Google |
| `tests/test_billing.py` | Fixed stale pricing assertions |

### Frontend
| File | Change |
|------|--------|
| `js/auth.js` | New — auth module |
| `css/auth.css` | New — auth styles |
| `landing.html` | GSI script, auth UI in nav |
| `index.html` | GSI script, auth UI in header, save modal |
| `practice.html` | GSI script, auth UI in header |

### Docs / Migrations
| File | Change |
|------|--------|
| `docs/google-oauth-setup.md` | Google OAuth configuration guide |
| `docs/auth-integration-results.md` | This file |
| `backend/migrations/002_google_oauth.sql` | google_id, avatar_url columns |

---

## What Jeff Needs To Do

1. Go to **console.cloud.google.com** and create project "StemScriber"
2. **APIs & Services > Credentials** > Create OAuth client ID (Web application)
3. Authorized JS origins: `https://stemscribe.io`, `http://localhost:5555`
4. Authorized redirect URIs: `https://stemscribe.io/auth/google/callback`, `http://localhost:5555/auth/google/callback`
5. Copy Client ID and add `GOOGLE_CLIENT_ID` to `~/stemscribe/.env`
6. Run migration: `psql "$DATABASE_URL" -f backend/migrations/002_google_oauth.sql`
7. Deploy to VPS

## What Claude Code Can Do

- Deploy all changes to VPS via SSH
- Run the migration
- Test endpoints
- Configure environment variables
