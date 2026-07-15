# Build Brief — OCR Support for UG PDF Chord Imports

**Date:** 2026-05-25
**For:** A fresh Claude Code session in another terminal
**Estimated effort:** 3–4 hours
**Priority:** PRE-LAUNCH (blocks launch June 20 — import feature unusable without this)

---

## Context — who you're working for

StemScriber (https://stemscriber.com) — Flask/Python backend on Hetzner VPS, vanilla JS frontend. Audio-to-chord-chart practice app. Public soft launch **June 20, 2026** at Refinery (Charleston).

The user is **Jeff Kozelski** (jkozelski@gmail.com). He doesn't write code; he tests features and reports what's broken in plain language. Don't dump jargon at him.

Project root (local Mac): `/Users/jeffkozelski/stemscribe/`
Production: `/opt/stemscribe/` on Hetzner VPS `5.161.203.112` — SSH: `ssh -i ~/.ssh/stemscribe_hetzner root@5.161.203.112`
Python venv on prod: `/opt/stemscribe/venv311`
Service: `systemctl restart stemscribe` (gunicorn behind Cloudflare Tunnel)

## The bug

The "Import Chart" feature on practice.html lets users upload a chord-chart file (PDF, TXT, ChordPro) → it parses chord/lyric structure → saves as the song's chord chart. Built 2026-05-23, modal redesigned 2026-05-25.

**Confirmed 2026-05-25 by direct test:** Ultimate Guitar's "Save as PDF" feature outputs **scanned image PDFs with zero extractable text**. Probably an intentional anti-scraping measure on UG's end. Tested 3 of Jeff's real UG exports (Hotel California, Aja, Fire and Rain from `/Users/jeffkozelski/Desktop/Ultimatescribe/`) — all returned 0 chars from pdfplumber.

The current import parser uses pdfplumber only. When zero text is extracted, the endpoint bails with `"This PDF looks scanned (no extractable text). v1 only supports text-based PDFs — try exporting from Ultimate Guitar directly."` — which is honest but useless, because **what Jeff just exported FROM Ultimate Guitar IS the scanned PDF**.

Without OCR, the import-chart feature is effectively non-functional for its primary user workflow.

## The build

Add Tesseract OCR as a fallback path in the existing PDF extractor. **Do not** rewrite the parser or change the API contract — only extend the extraction stage.

### Architecture

Current flow in `backend/processing/chord_chart_import.py`:
```
PDF → extract_text_from_pdf(path) [pdfplumber only] → parse_chord_chart_text → chord chart JSON
```

New flow:
```
PDF → extract_text_from_pdf(path)
        ├─ try pdfplumber (fast path, free for text PDFs)
        └─ if zero text → render pages to images → Tesseract OCR → combined text
      → parse_chord_chart_text → chord chart JSON
```

### Dependencies

On **prod VPS**:
```bash
apt update && apt install -y tesseract-ocr poppler-utils
/opt/stemscribe/venv311/bin/pip install pytesseract pdf2image
```

- **Tesseract** — Apache 2.0, OK for commercial use ([[project_training_data_legal]] doesn't apply, this is a runtime tool not training data)
- **pdf2image** — relies on `poppler-utils` (apt package, GPL but called as a subprocess, not linked — distribution is fine)
- **pytesseract** — Apache 2.0 wrapper

Verify install on prod after apt:
```bash
ssh -i ~/.ssh/stemscribe_hetzner root@5.161.203.112 'tesseract --version && /opt/stemscribe/venv311/bin/python -c "import pytesseract, pdf2image; print(\"ok\")"'
```

### Code change

File: `/opt/stemscribe/backend/processing/chord_chart_import.py`

Locate the existing `extract_text_from_pdf(path)` function. Add a fallback:

```python
def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF. Tries pdfplumber first (fast, free for text PDFs);
    falls back to Tesseract OCR for scanned PDFs (UG exports, tab book scans)."""
    import pdfplumber
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ''
            text_parts.append(t)
    combined = '\n'.join(text_parts)

    # If pdfplumber returned no usable text, fall back to OCR.
    # Threshold: <40 chars across the whole doc = effectively empty.
    if len(combined.strip()) < 40:
        try:
            from pdf2image import convert_from_path
            import pytesseract
            images = convert_from_path(path, dpi=200)  # 200 dpi is a good speed/accuracy tradeoff for tab text
            ocr_parts = []
            for img in images:
                page_text = pytesseract.image_to_string(img, lang='eng')
                ocr_parts.append(page_text)
            combined = '\n'.join(ocr_parts)
        except Exception as e:
            # Don't crash the request — return whatever pdfplumber got (likely empty).
            # The endpoint will then return the existing "looks scanned" message.
            import logging
            logging.getLogger(__name__).warning(f"OCR fallback failed for {path}: {e}")

    return combined
```

### Error handling on the endpoint side

File: `backend/routes/chord_sheet.py` — the `/api/import-chart/<job_id>` endpoint.

The existing "looks scanned" 422 response should now only fire if OCR ALSO returned nothing. Wording suggestion when OCR yields too little:
> "Couldn't read enough text from this PDF — even with OCR. Try a clearer scan or paste the chart text directly."

### Tests

Add to `backend/tests/test_chord_chart_import.py`:

1. **Smoke test that OCR is wired** — call `extract_text_from_pdf` against a tiny image-only PDF fixture (generate one with reportlab + a rendered PNG, OR ship a checked-in small fixture). Assert non-empty result.
2. **OCR doesn't kick in for text PDFs** — existing text-PDF fixture should NOT invoke pytesseract. Mock pytesseract and assert it was never called when pdfplumber returned text.
3. **OCR failure is non-fatal** — mock pytesseract to raise; assert function returns whatever pdfplumber had (empty string OK) without raising.

Run: `cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/test_chord_chart_import.py -v` — all should pass (35 existing + 3 new).

### Real-PDF acceptance test (the one that matters)

After deploying, test against Jeff's actual UG exports at `/Users/jeffkozelski/Desktop/Ultimatescribe/`. At minimum pick 3:
- `hotel california official.pdf`
- `aja official.pdf`
- `fire and rain official.pdf`

For each, upload via the live UI on https://stemscriber.com (open a song first, then click "Import Chart" in the practice header). Expected result: chord chart populates with verses/choruses/chord changes that match the source PDF.

If OCR accuracy is poor, knobs to tune:
- `dpi=200` → try `dpi=300` (slower, more accurate)
- Tesseract config: try `pytesseract.image_to_string(img, lang='eng', config='--psm 6')` (assume single block of text)
- Pre-process with PIL: grayscale + threshold before OCR

## Deploy discipline (READ CAREFULLY — these files are drift-managed)

**Prod is AHEAD of local for these files. Always pull from prod, patch /tmp, push back.** Do not assume local matches prod.

For each prod file you change:
1. `scp -i ~/.ssh/stemscribe_hetzner root@5.161.203.112:/opt/stemscribe/backend/<path> /tmp/<file>.prod.py`
2. Edit `/tmp/<file>.prod.py`
3. Syntax-check: `python3 -c "import ast; ast.parse(open('/tmp/<file>.prod.py').read())"`
4. Backup on prod: `ssh ... 'cp /opt/stemscribe/backend/<path> /opt/stemscribe/backend/<path>.PREDEPLOY-ocr-20260525'`
5. Push: `scp -i ~/.ssh/stemscribe_hetzner /tmp/<file>.prod.py root@5.161.203.112:/opt/stemscribe/backend/<path>`
6. Checksum verify both ends match
7. `ssh ... 'systemctl restart stemscribe && sleep 3 && systemctl is-active stemscribe'`

**WARNING — do NOT restart stemscribe while a song is processing.** Restart kills in-flight worker threads and orphans the job (Jeff hit this exact bug at 01:47 UTC tonight — see `/opt/stemscribe/outputs/442213d7-ddd8-4795-b5d6-7cb5688c3e02` for evidence). Before deploying, check there are no `status=processing` jobs:
```bash
ssh -i ~/.ssh/stemscribe_hetzner root@5.161.203.112 'cd /opt/stemscribe/backend && source /opt/stemscribe/.env && /opt/stemscribe/venv311/bin/python -c "
import json, glob
for p in glob.glob(\"/opt/stemscribe/outputs/*/job_metadata.json\"):
    j = json.load(open(p))
    if j.get(\"status\") == \"processing\":
        print(\"ACTIVE:\", p)
"'
```
If anything prints, wait a few minutes and retry.

## Acceptance criteria

1. ✅ Tesseract + poppler installed on prod, importable from venv311
2. ✅ `extract_text_from_pdf` falls back to OCR when pdfplumber returns <40 chars
3. ✅ All existing 35 chord_chart_import tests still pass
4. ✅ 3 new tests added covering: OCR wiring, OCR-not-invoked-for-text-PDFs, OCR-failure-non-fatal
5. ✅ Live test against ≥3 of Jeff's `/Users/jeffkozelski/Desktop/Ultimatescribe/` PDFs produces a populated chord chart in the UI
6. ✅ Pre-deploy backups exist on prod (`.PREDEPLOY-ocr-20260525` suffix)
7. ✅ Service restarts cleanly, healthcheck returns 200 after deploy

## Hand-back

When done, post a summary back to Jeff including:
- Which PDFs you tested and how the chord chart looked (subjective: did it match the source?)
- Any OCR quality issues he should know about (mis-recognized chord symbols are the common one)
- How much time it added per PDF (e.g. "5-page PDF took +8 seconds vs the previous instant-fail")
- Memory updates: add a note to `stemscriber_full_state.md` under a new session header

If anything's unclear before you start, ASK Jeff rather than guessing. Specifically:
- Whether to bump the page DPI if OCR quality is mediocre on first pass
- Whether to default to landscape orientation handling (some UG PDFs are sideways)

Plain language only when reporting back. He's a musician, not an ML engineer.
