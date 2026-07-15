# Import Chart from PDF — Build Brief

**For:** a fresh Claude Code agent. Read top to bottom. Everything below was investigated 2026-05-23 — don't re-investigate, just build. Length is intentional: the more you take from this brief, the less time you spend reverse-engineering a 9,800-line `practice.html`.

---

## The goal in one paragraph

Let a signed-in StemScriber user click the existing **"My Chart"** button on the practice page, **upload a PDF** (Ultimate-Guitar style chord-over-lyric), and have it parsed into a chord chart that replaces the auto-detected one for the currently-loaded song. The same button continues to accept pasted text as it does today; file upload is an added pane in the same modal, feeding the **same** backend parser. v1 also accepts `.txt` and `.chordpro` files (free wins because they share the parser). Scanned PDFs (OCR) and sheet-music notation PDFs (OMR) are explicitly **out of scope** for v1.

Jeff has real UG-exported PDFs on his Mac, ready to test the moment v1 is live. The success criterion is "Jeff drops a PDF onto a song and the chord chart updates to match what's in the PDF."

---

## Project context (skim, then move on)

- **What StemScriber is:** a web app that takes a song, separates its stems, detects chords, and shows a practice-mode page with player + chord chart + tabs. Production at `https://stemscriber.com`, served from a Hetzner VPS (`5.161.203.112`), code at `/opt/stemscribe/`.
- **Stack:** Python/Flask backend (modularized into blueprints under `backend/`), vanilla-JS frontend (`frontend/practice.html` is the main practice page, `frontend/index.html` is the upload landing). Python 3.11 in `venv311/`. Postgres in Supabase for users + jobs metadata. Per-job artifacts on disk at `outputs/<job_id>/`.
- **Launch:** June 20, 2026 (Refinery, Charleston). ~4 weeks out. This feature is a v1 quality-of-life add for the launch, not a blocker.
- **Who uses it:** musicians (Jeff's audience is hobbyists + cover bands + teachers). Plain language matters. Jeff himself is a working musician — when you write user-facing strings, no jargon.

### Prod access

- SSH: `ssh -i ~/.ssh/stemscribe_hetzner root@5.161.203.112`
- App root: `/opt/stemscribe/`
- Backend code: `/opt/stemscribe/backend/`
- Frontend files (served statically): `/opt/stemscribe/frontend/`
- venv: `/opt/stemscribe/venv311/`
- Restart: `systemctl restart stemscribe`
- Logs: `journalctl -u stemscribe -f`

---

## ⚠️ DRIFT WARNING — read before deploying anything

Three files have **diverged between local checkout and prod** — prod is AHEAD of local because of in-place patches over the last weeks. **You cannot edit local and `scp` it up.** You will silently wipe prod-only code. The drift-managed files this build will touch:

- `frontend/practice.html` (9,800+ lines, drift-managed)
- `backend/routes/api.py` (~1,000 lines, drift-managed) — **AVOID touching this file in this build.** Put your new route in `routes/chord_sheet.py` instead.

### Deploy discipline (use this for every prod change, no exceptions)

1. `scp` the file DOWN from prod to `/tmp/`:
   `scp root@5.161.203.112:/opt/stemscribe/frontend/practice.html /tmp/practice.prod.html`
2. Patch the `/tmp/` copy surgically. **Exact-anchor string replace.** Before each edit, grep for the anchor and confirm it's unique. If not unique, expand the anchor with more surrounding context.
3. Syntax-check: for Python, `python3 -c "import ast; ast.parse(open('/tmp/file.py').read())"`. For HTML, use Python's `html.parser` with a depth counter — `final depth=0` means tags balance.
4. Back up prod with a timestamped name: `ssh root@... cp <file> <file>.PREDEPLOY-importchart-20260523`. Don't skip this.
5. `scp` patched copy UP.
6. Verify `shasum -a 256` matches local↔prod **exactly**. If they don't match, stop and figure out why before going further.
7. `systemctl restart stemscribe`, wait for the app to come back (it loads ~260 MB of ML models on startup; first 200 from `curl http://localhost:5555/` usually takes 15–25s).
8. Confirm `systemctl is-active stemscribe` returns `active` and `curl -s -o /dev/null -w "%{http_code}\n" https://stemscriber.com/practice.html` returns 200.

This is the discipline; every successful deploy this week followed it. Skip a step and you'll either wipe prod-only code or leave a broken page live.

---

## What's already there (verified 2026-05-23 — don't re-verify)

### Frontend

- **The "My Chart" button** in `frontend/practice.html` at ~line 2334:
  ```html
  <button class="header-btn" id="useMyChartBtn" title="Paste your own chord chart from Ultimate Guitar, Songsterr, or anywhere else">...</button>
  ```
- **Its click handler** at ~line 9359:
  ```js
  document.getElementById('useMyChartBtn')?.addEventListener('click', () => {
      showImportChartModal();
  });
  ```
- **`showImportChartModal`** is the modal you'll be modifying. Grep for the function definition in `practice.html` to find it. It currently shows the paste-text flow. You'll add a file-picker pane next to (or above) the textarea.
- **`_parseChordsFromString(chordStr)`** at ~line 7755. This is a JS-side chord regex parser:
  ```js
  function _parseChordsFromString(chordStr) {
      if (!chordStr) return [];
      var results = [];
      var re = /([A-G][#b]?(?:m(?:aj)?|maj|dim|aug|sus|add)?[0-9]*(?:[#b][0-9]+)?(?:sus[24])?(?:add[0-9]+)?(?:\/[A-G][#b]?)?)/g;
      var m;
      while ((m = re.exec(chordStr)) !== null) {
          results.push({ name: m[1], charPos: m.index });
      }
      return results;
  }
  ```
  **Mirror this regex on the Python side** — same chord vocabulary, same edge cases. Don't reinvent.
- **`_buildSlashNotation(...)`** at ~line 7800. Builds chord/slash/lyric rows. Reference only — your output JSON feeds the same downstream code that already handles rendering.
- **Auth helper** in `frontend/js/auth.js`:
  ```js
  SS.authHeaders = function() {
      var headers = {};
      if (SS.accessToken) {
          headers['Authorization'] = 'Bearer ' + SS.accessToken;
      }
      return headers;
  };
  ```
  Use as `window.StemScriber.authHeaders()`. **Every** fetch to a user-data endpoint must include these.

### Backend

- **Existing manual-chart endpoint** in `backend/routes/api.py` at ~line 848: `GET/PUT /api/chord-chart/<job_id>`. PUT accepts JSON and writes `outputs/<job_id>/chord_chart.json`. Your new endpoint does effectively the same write — produce JSON in the right shape, write it to the same path.
- **Job auth pattern** (use this for the new endpoint):
  ```python
  @auth_required
  def import_chart(job_id):
      if not validate_job_id(job_id):
          return jsonify({'error': 'Invalid job ID'}), 400
      job = get_job(job_id)
      if not job:
          return jsonify({'error': 'Job not found'}), 404
      uid = str(g.current_user.id)
      if job.user_id != uid:
          return jsonify({'error': 'Forbidden'}), 403
      ...
  ```
- **`pdfplumber`** — check if already in `backend/requirements.txt` and installed in `venv311`:
  ```bash
  /opt/stemscribe/venv311/bin/python -c "import pdfplumber; print(pdfplumber.__version__)"
  ```
  If missing, add `pdfplumber>=0.10` to requirements and `pip install` into venv311 on prod.

### Internal chord chart JSON format

Pull a real one to mirror. On prod:
```bash
cat /opt/stemscribe/backend/outputs/d9e3368e-b8f8-45fe-a7e9-1b0df9a66285/chord_chart.json | python3 -m json.tool | head -80
```
That's "The Time Comes" (the demo). Structure (paraphrased — confirm by reading the actual file):
```json
{
  "title": "...",
  "artist": "...",
  "key": "C",
  "bpm": 120,
  "sections": [
    {
      "name": "Verse 1",
      "lines": [
        {
          "chords": "C       Am      F       G",
          "chord_beats": [{"name":"C","beats":4}, {"name":"Am","beats":4}, ...],
          "lyrics": "Yesterday all my troubles seemed so far away"
        },
        ...
      ]
    },
    ...
  ]
}
```
The exact field names matter — match the file's. **Your parser's output must be a drop-in replacement for the auto-generated chord_chart.json.** Don't invent new fields the renderer doesn't know about.

---

## Build

### Phase 1 — Backend parser (start here; testable in isolation)

**File:** `backend/processing/chord_chart_import.py` (new)

```python
"""
Import chord charts from user-uploaded files (PDF, TXT, ChordPro).

Output: dict matching the shape of outputs/<job_id>/chord_chart.json so the
practice-page renderer treats it identically to auto-detected charts.
"""

import re
from pathlib import Path

# Mirror the JS chord regex from practice.html:7755
CHORD_RE = re.compile(
    r"\b([A-G][#b]?"
    r"(?:m(?:aj)?|maj|dim|aug|sus|add)?"
    r"[0-9]*"
    r"(?:[#b][0-9]+)?"
    r"(?:sus[24])?"
    r"(?:add[0-9]+)?"
    r"(?:/[A-G][#b]?)?"
    r")\b"
)

SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")

def extract_text_from_pdf(path: str) -> str:
    """Extract text from a text-based PDF, preserving line structure.
    Returns empty string if the PDF has no extractable text (scanned/image PDF)."""
    import pdfplumber
    out = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            out.append(t)
    return "\n".join(out)

def is_chord_line(line: str) -> bool:
    """A chord line is mostly chord tokens with whitespace. Heuristic:
    after stripping whitespace, what fraction of the tokens match CHORD_RE?"""
    tokens = line.split()
    if not tokens or len(tokens) > 16:  # very long lines are lyrics
        return False
    matches = sum(1 for t in tokens if CHORD_RE.fullmatch(t))
    return matches / len(tokens) >= 0.6 and len(line.strip()) <= 80

def parse_chord_chart_text(text: str, source: str = 'paste') -> dict:
    """Parse UG-style chord-over-lyric text into the internal chart JSON shape.

    source: 'paste' | 'pdf' | 'chordpro' — used only for logging/metadata.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    sections = []
    current_section = {"name": "Intro", "lines": []}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        sec = SECTION_RE.match(line)
        if sec:
            if current_section["lines"]:
                sections.append(current_section)
            current_section = {"name": sec.group(1).strip(), "lines": []}
            i += 1
            continue
        if is_chord_line(line):
            chord_line = line
            # Look ahead one line for the lyric pair (skip blank line if present)
            lyric_line = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and not is_chord_line(lines[j]) and not SECTION_RE.match(lines[j]):
                lyric_line = lines[j]
                i = j + 1
            else:
                i += 1
            chords = [{"name": m.group(1), "charPos": m.start()} for m in CHORD_RE.finditer(chord_line)]
            current_section["lines"].append({
                "chords": chord_line,
                "chord_positions": chords,   # for the renderer's char-aligned mode
                "lyrics": lyric_line,
            })
        else:
            # Standalone lyric line with no chord line above it.
            current_section["lines"].append({
                "chords": "",
                "chord_positions": [],
                "lyrics": line,
            })
            i += 1
    if current_section["lines"]:
        sections.append(current_section)
    return {
        "title": "",      # caller fills from job metadata
        "artist": "",
        "key": "",
        "bpm": None,
        "source": f"imported:{source}",
        "sections": sections,
    }
```

**Tests:** `backend/tests/test_chord_chart_import.py`. Cover:
- A synthetic UG-style text fixture (string literal in the test).
- A real-ish UG export sample committed under `backend/tests/fixtures/sample_ug_chart.txt`.
- A PDF fixture under `backend/tests/fixtures/sample_ug_chart.pdf` (Jeff can drop one in or you generate one with reportlab).
- Edge cases: empty input, all-lyrics-no-chords, all-chords-no-lyrics, section markers only, weird capo line like `Capo: 3rd fret`.

Run the existing suite to confirm no regressions:
```bash
cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/ -v
```

### Phase 2 — Backend route

**File:** `backend/routes/chord_sheet.py` (existing, ~35 KB). Add at the bottom, in the same Blueprint as the existing routes:

```python
import os
import tempfile
from werkzeug.utils import secure_filename
from auth.middleware import auth_required
from middleware.validation import validate_job_id
from models.job import get_job, OUTPUT_DIR
from processing.chord_chart_import import (
    parse_chord_chart_text,
    extract_text_from_pdf,
)

ALLOWED_IMPORT_EXTS = {'.pdf', '.txt', '.cho', '.chordpro'}
MAX_IMPORT_BYTES = 5 * 1024 * 1024  # 5 MB plenty for any chord chart

@chord_sheet_bp.route('/api/import-chart/<job_id>', methods=['POST'])
@auth_required
def import_chart(job_id):
    if not validate_job_id(job_id):
        return jsonify({'error': 'Invalid job ID'}), 400
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    uid = str(g.current_user.id)
    if job.user_id != uid:
        return jsonify({'error': 'Forbidden'}), 403

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'Empty filename'}), 400
    name = secure_filename(f.filename)
    ext = os.path.splitext(name)[1].lower()
    if ext not in ALLOWED_IMPORT_EXTS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 415

    # Read into memory (cap size) then write to a temp file for PDF parsing
    blob = f.read(MAX_IMPORT_BYTES + 1)
    if len(blob) > MAX_IMPORT_BYTES:
        return jsonify({'error': 'File too large (max 5 MB)'}), 413

    if ext == '.pdf':
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(blob); tmp.flush()
            try:
                text = extract_text_from_pdf(tmp.name)
            finally:
                os.unlink(tmp.name)
    else:
        text = blob.decode('utf-8', errors='replace')

    if not text.strip():
        return jsonify({'error': 'No text could be extracted from this file'}), 422

    chart = parse_chord_chart_text(text, source=ext.lstrip('.'))
    # Fill in identity from the job metadata if missing
    meta = job.metadata or {}
    chart['title'] = chart.get('title') or meta.get('title') or job.filename
    chart['artist'] = chart.get('artist') or meta.get('artist') or ''

    out_path = OUTPUT_DIR / job_id / 'chord_chart.json'
    import json
    out_path.write_text(json.dumps(chart, indent=2))

    return jsonify({'status': 'ok', 'chart': chart})
```

Don't forget to register the blueprint if `chord_sheet_bp` isn't already in `app.py`'s factory — check first; it almost certainly is (chord_sheet.py is 35 KB of routes, all live).

### Phase 3 — Frontend modal expansion

In `frontend/practice.html`, find `showImportChartModal`. The simplest layout: keep the existing paste textarea, add an upload pane **above** it (file pickers feel more "primary" than text boxes in 2026).

Sketch of the added HTML inside the modal body:

```html
<!-- new upload pane -->
<div class="import-chart-upload" style="border:2px dashed var(--border); border-radius:10px; padding:1.25rem; text-align:center; margin-bottom:1rem;">
  <div style="font-family:Righteous,cursive; font-size:1.05rem; margin-bottom:0.5rem;">Upload a chord chart</div>
  <div style="color:var(--text-dim); font-size:0.85rem; margin-bottom:0.85rem;">PDF (Ultimate Guitar style), .txt, or .chordpro &mdash; 5 MB max</div>
  <input type="file" id="importChartFile" accept=".pdf,.txt,.cho,.chordpro" style="display:none;">
  <button type="button" class="header-btn" onclick="document.getElementById('importChartFile').click();">Choose file…</button>
  <div id="importChartStatus" style="margin-top:0.75rem; min-height:1.2em; color:var(--text-dim); font-size:0.85rem;"></div>
</div>
<!-- existing paste textarea + Save button stays below -->
```

Wire the file input:

```js
document.getElementById('importChartFile').addEventListener('change', async function() {
    var f = this.files && this.files[0];
    if (!f) return;
    var status = document.getElementById('importChartStatus');
    status.textContent = 'Reading ' + f.name + '…';
    var jobId = window.currentJobId;
    if (!jobId) { status.textContent = 'No song loaded.'; return; }
    var fd = new FormData();
    fd.append('file', f);
    try {
        var ah = (window.StemScriber && window.StemScriber.authHeaders) ? window.StemScriber.authHeaders() : {};
        var resp = await fetch(API_BASE + '/import-chart/' + encodeURIComponent(jobId), {
            method: 'POST', body: fd, headers: ah,
        });
        var data = await resp.json();
        if (!resp.ok) { status.textContent = data.error || ('Upload failed (' + resp.status + ')'); return; }
        status.textContent = 'Imported. Refreshing chart…';
        // Reuse whatever the existing paste-flow does on success — call the same
        // function so both flows feel identical. Find it inside showImportChartModal.
        if (typeof reloadChordChart === 'function') reloadChordChart();
        else window.location.reload();   // last-resort fallback
        // Close modal — same close call the paste flow uses.
    } catch (e) {
        status.textContent = 'Upload error: ' + e.message;
    }
});
```

(The exact "reload chord chart" function is in the paste-flow handler — find it by reading the existing `showImportChartModal` body; reuse it.)

---

## Scope — what's in v1, what's NOT

**In v1:**
- Text-based PDF (UG export, browser print-to-PDF, anything where you can select+copy text)
- `.txt` plain text chord-over-lyric
- `.chordpro` / `.cho`
- Section markers `[Verse]`, `[Chorus]`, `[Bridge]`, `[Intro]`, `[Outro]`, etc.
- Capo / key annotation lines passed through into a metadata field (don't have to transpose for v1; just preserve)

**Out of v1 (defer to v2 when there's a real customer asking):**
- **Scanned PDFs** (photo of a paper chart) — needs OCR. Real work. Tesseract or cloud OCR API. Days of work and accuracy is fiddly.
- **Sheet music notation PDFs** (notes on staves) — needs OMR (optical music recognition). Months of work. Not the audience.
- **Guitar Pro `.gp/.gp5`** — different binary format. The app already *exports* GP via `midi_to_gp.py`; importing would be a separate parser pass using `PyGuitarPro` or similar. Treat as its own feature later.
- **MusicXML** — possible later but the audience for "I have a MusicXML file" is small.
- **Images** (PNG/JPG of a chart) — same OCR problem as scanned PDFs.

If the parser encounters a scanned PDF (text extraction returns empty string), return a clear 422 with the message: `"This PDF looks scanned (no extractable text). v1 only supports text-based PDFs — try exporting from Ultimate Guitar directly."` Don't try OCR silently.

---

## Legal — important context

User-uploaded chord charts attached to that user's own account/job are **fine** per Alexandra Mayo (Morris Music Law), confirmed by Jeff on 2026-05-22. Storage scope to enforce:

- ✅ OK: `outputs/<job_id>/chord_chart.json` — per-job, owned by the uploading user, served back only to them or to admin.
- ❌ NOT OK: a global `charts/` library, a shared "community" pool, or any flow where one user's imported chart becomes visible/usable to another user. That was the previous mode that got the 15,417-chart library deleted in April 2026 after the Apr 10 lawyer call.

If you find yourself building anything that aggregates imported charts across users — stop and re-read this section.

---

## Acceptance — Jeff will run these against real UG PDFs

1. Sign in to stemscriber.com on desktop. Open any of the 8 songs in practice mode (e.g. "The Time Comes" demo).
2. Click **My Chart** → modal opens with two panes: upload (new) on top, paste (existing) below.
3. Click **Choose file…** → file dialog opens, filters to `.pdf/.txt/.cho/.chordpro`.
4. Select a real UG-exported PDF from Jeff's Mac → status shows "Reading…", then "Imported. Refreshing chart…", modal closes, the chord chart on the practice page **updates to match the PDF**.
5. **Reload the page** → the imported chord chart persists (it's saved to `outputs/<job_id>/chord_chart.json`).
6. Repeat with a `.txt` file and a `.chordpro` file — same outcome.
7. Try a scanned/image PDF → clear "this looks scanned, not supported in v1" message in the modal. No silent failure.
8. Anonymous `curl -X POST https://stemscriber.com/api/import-chart/<any-job-id> -F file=@chart.pdf` returns 401.
9. A signed-in `curl` with a wrong-owner job ID returns 403.
10. File >5 MB returns 413.
11. Existing functionality unchanged: paste-text flow still works; auto-detected chord charts on other songs still render normally.

When all 11 pass, deploy is complete. Update the in-memory state file (`stemscriber_full_state.md`) noting the import-chart feature shipped, and mark the brief's source task as completed.

---

## Don't-do-again reminders

- **Never edit local + scp up on the drift-managed files.** Always scp DOWN, patch /tmp, scp UP, checksum-verify.
- **Auth header on every user-data fetch.** The library-empty-on-phone bug (2026-05-22), the library-only-shows-demo bug (2026-05-23 morning), and the auto-detected-charts-not-tied-to-users bug (2026-05-22) were all this same missing `Authorization: Bearer` header. The fix has had to land in three different fetch sites; this is the fourth — get it right.
- **No new chord-library aggregation.** Per-job storage only. Re-read the Legal section above.
- **Test against real Jeff PDFs before declaring done.** Synthetic fixtures aren't enough — UG's actual export has quirks.

---

## Pointers (read only if stuck)

- StemScriber memory index: `~/.claude/projects/-Users-jeffkozelski/memory/MEMORY.md`
- Master project state: `~/.claude/projects/-Users-jeffkozelski/memory/stemscriber_full_state.md`
- Project CLAUDE.md: `~/stemscribe/.claude/CLAUDE.md`
- Today's library/practice fixes (recent context): backup files on prod under `*.PREDEPLOY-*-2026052[23]` show what changed.
- Plain-language style with Jeff: working musician, not an engineer. Analogies, not jargon. Don't use "honest" as a filler word. Big visuals over tiny text. CSS = "Check Screen Shot" (he'll send screenshots from `~/Desktop/screenshots/`).

Good luck. Ship it.
