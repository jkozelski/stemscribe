# QA Report — Post Bug-Fix Audit
**Date:** 2026-03-18
**Tester:** Claude Code (automated E2E)
**Server:** http://localhost:5555

---

## 1. Test Suite Results

**Command:** `./venv311/bin/python -m pytest backend/tests/ -v --tb=short`
**Result:** 315 passed, 2 failed, 1 warning

### Failing Tests

| Test | File | Expected | Actual | Root Cause |
|------|------|----------|--------|------------|
| `test_premium_plan_pricing` | `test_billing.py:33` | `monthly_price == 4.99` | `20` | `billing/plans.py` was updated (Premium $20/mo) but test still expects old $4.99 pricing |
| `test_pro_plan_unlimited` | `test_billing.py:38` | `songs_per_month == -1` (unlimited) | `25` | `billing/plans.py` Pro plan has 25 songs/mo; test expects unlimited |

**Analysis:** The plan tiers in `backend/billing/plans.py` were restructured (Premium is now the top tier at $20/mo unlimited, Pro is mid-tier at $10/mo with 25 songs), but the test expectations in `backend/tests/test_billing.py` lines 32-39 still reference the old pricing. The landing page HTML also shows the old pricing ($4.99 Premium, $14.99 Pro). All three need to be aligned.

### Warning
- `RequestsDependencyWarning: urllib3 (2.6.3) or chardet/charset_normalizer version mismatch` — cosmetic, from requests library

---

## 2. Backend Health Check

**Endpoint:** `GET /health`
**Status:** 200 OK
**Response:** `{"service":"stemscribe-api","status":"ok"}`
**Verdict:** PASS

---

## 3. Practice Page (`practice.html`) — CRITICAL ISSUES

### 3a. JavaScript SyntaxError — Duplicate `NOTE_NAMES` Declaration (BLOCKER)

**Lines:** 1543 and 4404 of `frontend/practice.html`
- Line 1543: `const NOTE_NAMES = ['C', 'C#', 'D', ...]` (top-level scope)
- Line 4404: `var NOTE_NAMES = ['C','C#','D','D#', ...]` (same top-level scope, Piano Keyboard Chord Visualizer section)

**Impact:** The browser throws `SyntaxError: Identifier 'NOTE_NAMES' has already been declared` which kills the ENTIRE inline script block. This means:
- Job data never loads from `?job=` parameter
- Chord view never renders
- Stem channels never appear
- Audio player never initializes
- Practice mode is completely non-functional

**Fix:** Remove the duplicate declaration on line 4404 and reference the existing `NOTE_NAMES` const, or rename the second one (e.g., `PIANO_NOTE_NAMES`).

### 3b. CDN 404 Errors — jspdf and svg2pdf

| Library | Broken URL (cdnjs, 404) | Working Alternative |
|---------|------------------------|---------------------|
| jspdf 2.5.2 | `https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.2/jspdf.umd.min.js` | Use version 2.5.1: `https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js` |
| svg2pdf.js 2.2.4 | `https://cdnjs.cloudflare.com/ajax/libs/svg2pdf.js/2.2.4/svg2pdf.umd.min.js` | Use jsdelivr: `https://cdn.jsdelivr.net/npm/svg2pdf.js@2.2.4/dist/svg2pdf.umd.min.js` |

**Location:** `frontend/practice.html` lines 21-22
**Impact:** PDF export of notation will fail. Non-blocking for core practice functionality (once the NOTE_NAMES fix is applied).

### 3c. Playwright Verification — Practice Mode Load Test

**URL:** `http://localhost:5555/practice.html?job=4afcef17-c1ed-4bd1-8eb8-e43afe027ba5`
**Job:** "The Time Comes" by Kozelski (complete job with 6 stems, chords, MIDI, GP5)

| Check | Result |
|-------|--------|
| Page loads without crash | PASS (HTML renders) |
| No JS errors | FAIL — 3 errors (NOTE_NAMES duplicate + 2 CDN 404s) |
| Play button exists | PASS (button ref=e18 visible) |
| Chord view loads | FAIL — script block killed before init |
| Stem channels appear | FAIL — script block killed before init |
| Speed controls visible | PASS (slider + preset buttons render from static HTML) |
| Job data loaded | FAIL — "Load a track to begin" still shown |

---

## 4. Main App Page (`/app`)

**URL:** `http://localhost:5555/app`
**Result:** PASS — 0 errors, 1 warning (apple-mobile-web-app-capable deprecation)

| Check | Result |
|-------|--------|
| Page loads | PASS |
| Upload UI visible | PASS |
| Settings panel | PASS |
| Skills loaded | PASS (console log confirms) |

---

## 5. Library Page

**Trigger:** Click "Library" button on `/app`
**Result:** PASS — 21 songs loaded successfully

| Check | Result |
|-------|--------|
| Library opens | PASS |
| Songs listed with metadata | PASS |
| Thumbnails/artist info | PASS |
| Stem/MIDI/Tab badges | PASS |
| Delete buttons present | PASS |

---

## 6. Landing Page (`/`)

**URL:** `http://localhost:5555/`
**Result:** PASS — 0 errors, 1 warning

**Note:** Pricing on landing page shows Premium at $4.99/mo and Pro at $14.99/mo, but `backend/billing/plans.py` has Premium at $20/mo and Pro at $10/mo. These need to be aligned.

---

## 7. Frontend JS Module Syntax Checks

All standalone JS files pass `node --check`:

| File | Status |
|------|--------|
| mixer.js | OK |
| upload.js | OK |
| results.js | OK |
| panels.js | OK |
| progress.js | OK |
| utils.js | OK |
| archive.js | OK |
| trackinfo.js | OK |
| waveform.js | OK |

---

## 8. Backend Python Syntax

`backend/routes/api.py` — compiles cleanly, no syntax errors.

---

## Summary of Issues by Priority

### BLOCKER (must fix before any user testing)
1. **Duplicate `NOTE_NAMES` in `practice.html`** — Kills entire practice mode. Lines 1543 vs 4404. Remove the `var` redeclaration on line 4404.

### HIGH
2. **CDN 404s for jspdf 2.5.2 and svg2pdf.js 2.2.4** — PDF export broken. Fix URLs in `practice.html` lines 21-22.
3. **Pricing mismatch** — `billing/plans.py`, `test_billing.py`, and landing page HTML all show different prices. Align all three.

### LOW
4. **`apple-mobile-web-app-capable` deprecation warning** — Cosmetic, affects both app and landing page.
5. **urllib3/chardet version mismatch warning** — Test environment only, no user impact.

---

## Files Referenced
- `/Users/jeffkozelski/stemscribe/frontend/practice.html` (lines 21-22, 1543, 4404)
- `/Users/jeffkozelski/stemscribe/backend/billing/plans.py`
- `/Users/jeffkozelski/stemscribe/backend/tests/test_billing.py` (lines 32-39)
- `/Users/jeffkozelski/stemscribe/frontend/js/*.js` (all clean)
- `/Users/jeffkozelski/stemscribe/backend/routes/api.py` (clean)
