# StemScriber Security Audit Report

**Date:** 2026-03-19
**Auditor:** Agent 4 (Security Audit & Test)
**Scope:** Beta hardening — security changes made by Agents 1-3
**Environment:** localhost:5555, Python 3.11, Flask, Cloudflare Tunnel to stemscribe.io

---

## 1. Test Suite Results

**Command:** `cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/ -v`

| Result | Count |
|--------|-------|
| Passed | 315 |
| Failed | 2 |
| Total | 317 |

**Failures (pre-existing, NOT security-related):**
- `test_billing.py::TestPlans::test_premium_plan_pricing` — test expects $4.99, actual is $20
- `test_billing.py::TestPlans::test_pro_plan_unlimited` — test expects unlimited (-1), actual is 25 songs/month

These are stale test expectations from a pricing change. No security impact.

---

## 2. Attack Vectors Tested (Live Server)

### 2.1 SSRF Attacks — ALL BLOCKED

| Attack | Payload | Result |
|--------|---------|--------|
| Cloud metadata | `http://169.254.169.254/latest/meta-data/` | Blocked: "local/private network addresses are blocked" |
| javascript: scheme | `javascript:alert(1)` | Blocked: "Invalid URL format" |
| file:// scheme | `file:///etc/passwd` | Blocked: "Invalid URL format" |
| localhost | `http://localhost:5555/api/health` | Blocked: "local/private network addresses are blocked" |
| 127.0.0.1 | `http://127.0.0.1:5555/api/health` | Blocked: "local/private network addresses are blocked" |
| 0.0.0.0 | `http://0.0.0.0:5555/api/health` | Blocked: "local/private network addresses are blocked" |

SSRF protection is solid. DNS resolution check prevents rebinding attacks. Both `url_resolver.validate_url_no_ssrf()` and `middleware/validation.validate_url()` cover private IPs, loopback, link-local, and reserved ranges.

### 2.2 XSS Attacks — SANITIZED

| Attack | Payload | Stored Value |
|--------|---------|-------------|
| Script tag in name | `<script>alert(1)</script>` | `&lt;script&gt;alert(1)&lt;/script&gt;` |
| Event handler in message | `<img src=x onerror=alert(1)>` | `&lt;img src=x onerror=alert(1)&gt;` |

HTML entities are properly escaped via `html.escape()` in support ticket creation. The `middleware/validation.py` also provides `strip_html_tags()` and `sanitize_text()` for routes that use it.

### 2.3 Path Traversal — BLOCKED

| Attack | Payload | Result |
|--------|---------|--------|
| Directory traversal via chord-chart | `/api/chord-chart/../../etc/passwd` | 404 Not found |
| Filename traversal in downloads | `..` and `/` checks in `download_file()` | Blocked with validation |

`_safe_path()` in api.py resolves paths and validates they stay within base_dir. `_validate_job_id()` enforces hex-only characters.

### 2.4 Rate Limiting — WORKING

| Test | Result |
|------|--------|
| 57 rapid requests to `/api/health` | All returned 200 |
| Requests 58+ | Returned 429 "Rate limit exceeded" |
| Global limit | 60 requests/min per IP (confirmed) |

Health endpoint is supposed to be exempt from rate limiting, but still got rate-limited. This means the exemption in `app.py` line 181 may not be working correctly for the health blueprint endpoint vs the api blueprint health endpoint.

Endpoint-specific limits configured but not independently verified:
- Upload: 5/min
- Auth: 5/min
- Songsterr: 30/min
- Library: 60/min
- Beta: 10/min
- SMS: 10/min

---

## 3. CRITICAL FINDINGS

### 3.1 [HIGH] SMS Endpoints Have No Authentication

**Endpoints affected:**
- `POST /api/sms/send` — Anyone can send SMS through Jeff's Twilio account
- `GET /api/sms/inbox` — Anyone can read SMS inbox
- `POST /api/sms/mark-read` — Anyone can mark messages as read

**Impact:** An attacker could send SMS messages via Twilio (costing money) or read private communications.

**Recommendation:** Add `@auth_required` decorator to all SMS endpoints, or at minimum restrict to localhost/Cloudflare Tunnel internal IPs.

### 3.2 [HIGH] Support Ticket Endpoints Have No Authentication

**Endpoints affected:**
- `GET /api/support/tickets` — Anyone can list all support tickets (PII exposure: names, emails)
- `GET /api/support/ticket/<id>` — Anyone can read ticket details
- `POST /api/support/ticket/<id>/respond` — Anyone can add responses
- `POST /api/support/ticket/<id>/resolve` — Anyone can resolve tickets

**Impact:** Customer PII (names, emails, messages) exposed to unauthenticated users. Ticket manipulation possible.

**Recommendation:** Add auth to list/read/respond/resolve. Only ticket creation should be public.

### 3.3 [MEDIUM] Hardcoded Default Beta Admin Key

**File:** `backend/routes/beta.py` line 26
```python
BETA_ADMIN_KEY = os.environ.get('BETA_ADMIN_KEY', 'stemscribe-beta-admin-2026')
```

**Impact:** If `BETA_ADMIN_KEY` env var is not set, anyone who guesses the default key can generate beta invite codes.

**Recommendation:** Remove the default fallback. Require the env var to be set.

### 3.4 [MEDIUM] Hardcoded Twilio Account SID

**File:** `backend/routes/sms.py` line 119
```python
account_sid = os.environ.get('TWILIO_ACCOUNT_SID', 'AC61b4ba568a01c65bf90d98655261161b')
```

**Impact:** Account SID is not a secret (it's like a username), but hardcoding it in source makes credential rotation harder and leaks account identity in version control.

**Recommendation:** Move to env var only, no default.

### 3.5 [MEDIUM] Security Headers Not Active on Running Server

**Finding:** Security headers (X-Content-Type-Options, X-Frame-Options, CSP, etc.) are defined in `app.py` but are NOT present in responses from the running server.

**Root cause:** The server was started before the security headers code was added by Agents 1-3. A server restart is needed to activate them.

**Verification:** Headers DO appear when tested via Flask test client, confirming the code is correct but the running process is stale.

**Recommendation:** Restart the server to activate security headers.

### 3.6 [LOW] No Strict-Transport-Security Header

HSTS is not set. Since the app runs behind Cloudflare Tunnel, Cloudflare handles TLS termination, but adding HSTS would ensure browsers always use HTTPS.

### 3.7 [LOW] CSP Allows 'unsafe-inline' and 'unsafe-eval'

The Content-Security-Policy includes `'unsafe-inline'` and `'unsafe-eval'` for scripts. This weakens XSS protection via CSP. Acceptable for beta but should be tightened for production.

### 3.8 [LOW] CORS Default Includes localhost Only

CORS origins default to `http://localhost:5555,http://localhost:3000`. For production via Cloudflare Tunnel, `CORS_ORIGINS` env var should be set to `https://stemscribe.io`.

---

## 4. Endpoints Without Authentication

| Endpoint | Auth? | Risk |
|----------|-------|------|
| `POST /api/sms/send` | None | HIGH — can send SMS |
| `GET /api/sms/inbox` | None | HIGH — can read messages |
| `POST /api/sms/mark-read` | None | MEDIUM — can alter state |
| `GET /api/support/tickets` | None | HIGH — PII exposure |
| `GET /api/support/ticket/<id>` | None | MEDIUM — PII exposure |
| `POST /api/support/ticket/<id>/respond` | None | MEDIUM — ticket manipulation |
| `POST /api/support/ticket/<id>/resolve` | None | LOW — ticket manipulation |
| `POST /api/support/ticket` | None | LOW — intentionally public |
| `POST /api/beta/generate` | Admin key only | LOW — key-gated |
| `GET /api/health` | None | None — intentionally public |
| `GET /api/skills` | None | None — read-only |
| `GET /api/available-models` | None | None — read-only |

---

## 5. What's Covered vs. Not Covered

### Covered (Good)
- SSRF prevention with DNS resolution check
- XSS sanitization on support tickets (html.escape)
- Path traversal prevention (_safe_path, _validate_job_id)
- Rate limiting (global 60/min + endpoint-specific)
- File upload extension validation
- File upload size limit (500MB Flask, 50MB validation middleware)
- URL scheme validation (http/https only)
- URL length validation (2048 max)
- Password hashing (bcrypt, 12 rounds, NIST-compliant)
- Security headers (defined, need server restart)
- CORS configuration (configurable, not wildcard)
- Graceful shutdown for active jobs
- Debug mode disabled (`debug=False`)
- .env in .gitignore

### NOT Covered
- Authentication on SMS/support admin endpoints
- CSRF protection (no tokens)
- Request logging/audit trail
- Webhook signature validation (Twilio incoming SMS)
- Content-Type enforcement on POST endpoints
- API versioning
- Secrets rotation mechanism
- Automated security testing in CI
- Dependency vulnerability scanning
- Session management hardening (JWT is used, but no refresh token rotation)

---

## 6. Recommendations for Future Hardening

### Before Scaling Beyond Beta (Priority Order)

1. **Add auth to SMS and support admin endpoints** — This is the most critical gap
2. **Restart the server** to activate security headers
3. **Set CORS_ORIGINS env var** to `https://stemscribe.io` for production
4. **Remove hardcoded defaults** for BETA_ADMIN_KEY and TWILIO_ACCOUNT_SID
5. **Add Twilio webhook signature validation** on `/api/sms/incoming`
6. **Add CSRF tokens** for state-changing endpoints

### For Production Scale

7. Switch rate limiter storage from `memory://` to Redis (memory:// doesn't survive restarts and doesn't work across multiple processes)
8. Add request audit logging
9. Add dependency vulnerability scanning (pip-audit, safety)
10. Implement API key authentication for service-to-service calls
11. Add HSTS header
12. Tighten CSP to remove 'unsafe-inline' and 'unsafe-eval'
13. Add automated security tests to CI pipeline
14. Consider WAF rules via Cloudflare
