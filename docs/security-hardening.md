# StemScribe Security Hardening Summary

**Date:** 2026-03-19
**Context:** Beta hardening sprint — changes made by Agents 1-3, audited by Agent 4

---

## 1. Changes Made

### 1.1 Rate Limiting (`backend/middleware/rate_limit.py`)

**What was added:**
- Flask-Limiter integration with in-memory storage
- Global default: 60 requests/minute per IP
- Endpoint-specific limits:
  - Auth (login/register/forgot-password): 5/min
  - Upload/URL processing: 5/min
  - Songsterr: 30/min
  - Library: 60/min
  - Beta: 10/min
  - SMS: 10/min
  - Webhooks: 30/min
- Custom 429 error handler with JSON response
- IP extraction priority: CF-Connecting-IP > X-Forwarded-For > remote_addr (correct for Cloudflare Tunnel)

**Plan-based enforcement:**
- `enforce_plan_limits` decorator checks monthly song quota
- `enforce_duration_limit` checks audio duration against plan tier
- Free: 3 songs/month, 5 min max
- Premium: 50 songs/month, 15 min max
- Pro: unlimited, 30 min max
- Anonymous users tracked by IP hash

**Status:** Working. Verified via live testing — 429 responses triggered after ~57 requests in rapid succession.

### 1.2 Input Validation (`backend/middleware/validation.py`)

**What was added:**
- `sanitize_text()` — strips HTML tags, trims whitespace, enforces max length
- `strip_html_tags()` — regex-based HTML tag removal
- `validate_job_id()` — UUID format validation (hex + dashes only, max 36 chars)
- `validate_job_id_strict()` — full UUID v4 format
- `validate_email_format()` — basic email regex, 254 char max (RFC 5321)
- `validate_phone_number()` — digits/+/-/spaces/parens only, 20 char max
- `validate_url()` — scheme check, hostname validation, DNS resolution, private IP blocking
- `validate_file_upload()` — extension whitelist, MIME type check, 50MB size limit
- `validate_beta_code()` — alphanumeric + dashes, 50 char max
- `validate_ticket_id()` — delegates to strict UUID validation

**Status:** Module exists and is imported by `beta.py`. Not yet imported by all routes — some routes (api.py, support.py) have their own inline validators.

### 1.3 Security Headers (`backend/app.py`)

**What was added (after_request handler):**
- `X-Content-Type-Options: nosniff` — prevents MIME type sniffing
- `X-Frame-Options: DENY` — prevents clickjacking
- `X-XSS-Protection: 1; mode=block` — legacy XSS filter
- `Referrer-Policy: strict-origin-when-cross-origin` — limits referrer leakage
- `Permissions-Policy: camera=(), microphone=(self), geolocation=()` — restricts browser APIs
- `Content-Security-Policy` — restricts resource loading sources

**CSP Details:**
- `default-src 'self'`
- Scripts: self + unsafe-inline + unsafe-eval + CDNs (jsdelivr, cloudflare, unpkg)
- Styles: self + unsafe-inline + CDNs + Google Fonts
- Fonts: self + Google Fonts + CDNs
- Images: self + data: + blob:
- Media: self + blob:
- Connect: self only
- Workers: self + blob: + jsdelivr

**Status:** Code is correct and verified via Flask test client. Requires server restart to take effect on the running instance.

### 1.4 SSRF Prevention (`backend/services/url_resolver.py`)

**What was added:**
- `validate_url_no_ssrf()` function
- Blocks non-http/https schemes (file://, javascript:, ftp://, etc.)
- DNS resolution check — resolves hostname and validates all IPs
- Blocks private, loopback, link-local, and reserved IP ranges
- Blocks known cloud metadata hostnames (metadata.google.internal, 169.254.169.254)
- Applied to `/api/url` endpoint before any URL processing

**Status:** Working. All SSRF vectors tested and blocked.

### 1.5 Path Traversal Prevention (`backend/routes/api.py`)

**What was added:**
- `_validate_job_id()` — hex-only regex, max 36 chars
- `_safe_path()` — resolves path and validates it stays within base_dir
- `..` and `/` checks on filenames in download endpoints
- `secure_filename()` from werkzeug on uploaded files
- File type whitelist on downloads (stem, enhanced, midi, musicxml, gp, guitarpro)

**Status:** Working. Path traversal attempts return 400/404.

### 1.6 XSS Prevention (`backend/routes/support.py`)

**What was added:**
- `_sanitize()` function using `html.escape()` on all user inputs
- Applied to name, email, subject, and message fields in ticket creation
- Response text also sanitized

**Status:** Working. HTML entities properly escaped in stored data.

### 1.7 CORS Configuration (`backend/app.py`)

**What was changed:**
- Replaced wildcard CORS with configurable `CORS_ORIGINS` env var
- Default: `http://localhost:5555,http://localhost:3000`
- Production: set `CORS_ORIGINS=https://stemscribe.io`

**Status:** Configured. Default is localhost-only (safe for dev). Production env var should be set.

### 1.8 Upload Security

**What was added:**
- File extension whitelist: `.mp3, .wav, .flac, .m4a, .ogg, .aiff, .webm, .opus`
- Flask `MAX_CONTENT_LENGTH`: 500 MB
- Validation middleware additional check: 50 MB
- `secure_filename()` on all uploaded files

---

## 2. Configuration Reference

### Rate Limits

| Endpoint Group | Limit | Applied In |
|---------------|-------|------------|
| Global default | 60/min | Flask-Limiter |
| Auth (login/register) | 5/min | app.py |
| Upload/URL processing | 5/min | app.py |
| Songsterr | 30/min | app.py |
| Library | 60/min | app.py |
| Beta | 10/min | app.py |
| SMS | 10/min | app.py |
| Health | Exempt | app.py |

### Plan Limits

| Plan | Songs/Month | Max Duration |
|------|------------|-------------|
| Free | 3 | 5 min |
| Premium | 50 | 15 min |
| Pro | Unlimited | 30 min |

### Security Headers

| Header | Value |
|--------|-------|
| X-Content-Type-Options | nosniff |
| X-Frame-Options | DENY |
| X-XSS-Protection | 1; mode=block |
| Referrer-Policy | strict-origin-when-cross-origin |
| Permissions-Policy | camera=(), microphone=(self), geolocation=() |
| Content-Security-Policy | (see app.py for full value) |

---

## 3. What's Covered and What's Not

### Covered
- [x] Rate limiting (request-level + plan-level)
- [x] SSRF prevention (DNS resolution + IP blocking)
- [x] XSS prevention (HTML escaping on support tickets)
- [x] Path traversal prevention (safe_path + job_id validation)
- [x] File upload validation (extension + size)
- [x] Security response headers
- [x] CORS restriction (configurable origins)
- [x] URL validation (scheme + length + host)
- [x] Password security (bcrypt, 12 rounds, min 8 chars)
- [x] Debug mode off
- [x] .env gitignored

### Not Yet Covered
- [ ] Authentication on SMS endpoints
- [ ] Authentication on support ticket admin endpoints (list/read/respond/resolve)
- [ ] CSRF protection
- [ ] Twilio webhook signature validation
- [ ] Strict-Transport-Security (HSTS) header
- [ ] CSP without unsafe-inline/unsafe-eval
- [ ] Redis-backed rate limiter (currently in-memory)
- [ ] Request audit logging
- [ ] Dependency vulnerability scanning
- [ ] Automated security tests

---

## 4. Ongoing Security Checklist

### Every Deploy
- [ ] Server restarted after security changes (headers, rate limits)
- [ ] `CORS_ORIGINS` env var set to production domain
- [ ] `BETA_ADMIN_KEY` env var set (not using default)
- [ ] `debug=False` in app.py
- [ ] All tests passing (`pytest backend/tests/ -v`)

### Monthly
- [ ] Review rate limit thresholds (are they appropriate for current traffic?)
- [ ] Check for new endpoints without auth
- [ ] Review support ticket data for suspicious patterns
- [ ] Check Twilio usage for unauthorized sends
- [ ] Run `pip-audit` for dependency vulnerabilities

### Before Scaling Beyond Beta
- [ ] Add auth to SMS and support admin endpoints
- [ ] Add CSRF tokens
- [ ] Switch rate limiter to Redis
- [ ] Add Twilio webhook signature validation
- [ ] Tighten CSP (remove unsafe-inline/unsafe-eval)
- [ ] Add HSTS header
- [ ] Set up automated security scanning in CI
- [ ] Review all endpoints for auth requirements
