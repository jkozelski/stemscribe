# Import Your Own Tabs — Build Brief

**For:** a fresh Claude Code agent. Read top to bottom, then build. Everything below was investigated 2026-05-24 — don't re-investigate, just build. Length is intentional: the more you take from this brief, the less time you spend reverse-engineering a 9,800-line `practice.html`.

This is a sibling to the chord-chart import that shipped 2026-05-23 (`docs/import-chart-build-brief-2026-05-23.md`). Same patterns, different cargo. Read that brief first if you want shorthand context, then come back here for the tab-specific spec.

---

## The goal in one paragraph

Let a signed-in StemScriber user upload their own **Guitar Pro tab file** (`.gp`, `.gp5`, `.gpx`) for a song they've already processed. The uploaded file replaces the auto-generated `.gp5` so the practice page renders the user's hand-edited or curated tab via alphaTab instead of the auto-transcription. Crucial distinction from the chord-chart import: **tabs are notation/fret data, not chord-over-lyric text.** Different file types, different parser, different storage path, separate button. Both buttons live in the practice-page header.

Jeff has Guitar Pro files on his Mac ready to test (mySongBook subscriber, plus his own .gp5 exports from previous StemScriber processing he edited by hand).

---

## Project context (skim, then move on)

- **What StemScriber is:** web app, stem separation + chord detection + practice player. Production at `https://stemscriber.com`. Hetzner VPS at `5.161.203.112`, code at `/opt/stemscribe/`.
- **Stack:** Python/Flask backend, vanilla-JS frontend. `frontend/practice.html` (9,800 lines, drift-managed) is the practice page that renders Guitar Pro via the **`alphaTab.min.js`** library loaded from `frontend/js/alphaTab.min.js` (712 KB).
- **How tabs are generated today:** the audio processing pipeline runs Basic Pitch for guitar/bass, custom CRNN models for drums/piano, then `midi_to_gp.py` stitches MIDI → `.gp5`. Output lands at `outputs/<job_id>/guitarpro/<stem>.gp5` (per-stem files).
- **Launch:** June 20, 2026 (Refinery, Charleston). v1 quality-of-life add — not a blocker.
- **Test material:** Jeff has real `.gp/.gp5` files from his mySongBook subscription + hand-edited exports from previous StemScriber runs.

### Prod access

- SSH: `ssh -i ~/.ssh/stemscribe_hetzner root@5.161.203.112`
- App root: `/opt/stemscribe/`
- Restart: `systemctl restart stemscribe`
- Healthcheck timer (auto-restart if hung): `systemctl status stemscribe-healthcheck.timer`

---

## ⚠️ DRIFT WARNING — read before deploying anything

Three files have prod-vs-local DRIFT — prod is AHEAD of local. **You cannot edit local and `scp` it up.** You will wipe prod-only code.

- `frontend/practice.html` (drift-managed — you WILL need to touch this)
- `backend/routes/api.py` (drift-managed — **AVOID** for this build; put your new route in `routes/chord_sheet.py` or a new `routes/imports.py`)

**Deploy discipline for every prod change:**
1. `scp` file DOWN from prod to `/tmp/`
2. Patch the `/tmp/` copy surgically — exact-anchor string replace, grep first to confirm anchor uniqueness
3. Syntax-check (Python: `python3 -c "import ast; ast.parse(...)"`; HTML: Python's `html.parser` depth counter)
4. Back up prod with timestamped name: `cp <file> <file>.PREDEPLOY-importtabs-20260524`
5. `scp` UP
6. Verify `shasum -a 256` matches local↔prod **exactly**
7. `systemctl restart stemscribe`, wait for healthcheck to confirm `200`

---

## What's already there (verified 2026-05-24 — don't re-verify)

### Frontend

- **alphaTab library** at `frontend/js/alphaTab.min.js` loaded by `practice.html` line 26: `<script src="https://cdn.jsdelivr.net/npm/@coderline/alphatab@1.3.1/dist/alphaTab.min.js"></script>`. This is what renders the `.gp5` file as readable tab notation.
- **The chord-chart import button "My Chart"** already exists in the header (`id="useMyChartBtn"`, ~line 2336). **Do NOT reuse it for tab import** — Jeff explicitly said "import your own tabs button" as a separate thing. Add a new button next to it.
- **Header order after the May 2026 reskin:** Back, Tuner, Scales, My Chart, [your new button here], (hidden GP/PDF/Print/XML buttons), Sign-In/Profile.
- **GP file already loaded by practice page:** look for `downloadGP` button (line ~2327) — that's the existing download flow. The render flow grabs `/outputs/<job_id>/guitarpro/<stem>.gp5` and feeds it to alphaTab.

### Backend

- **Existing chord-import endpoint** in `routes/chord_sheet.py`: `POST /api/import-chart/<job_id>` (shipped 2026-05-23). Mirror its structure for tab import — auth, owner check, file size cap, write to `outputs/<job_id>/`, return parsed result. Your endpoint will be different in body but identical in shape.
- **Guitar Pro Python library:** `PyGuitarPro` is available — check `/opt/stemscribe/venv311/bin/pip show pyguitarpro`. If installed, use it to validate uploaded files (parse them, confirm well-formed, optionally extract metadata). If not installed, add `pyguitarpro>=0.9` to `backend/requirements.txt` and install in venv311 on prod.

### Auth + file-upload pattern

- **Auth helper** in `frontend/js/auth.js`: `window.StemScriber.authHeaders()` returns `{Authorization: 'Bearer <jwt>'}` when signed in. **Every** fetch to a user-data endpoint MUST include these. (Same lesson from the May 2026 library-empty bugs.)
- **File upload pattern** to mirror: see the chord-import modal's file-picker (`importChartFile` element in `practice.html`, added 2026-05-23) — same JS pattern (FormData, `headers: SS.authHeaders()`, post to your endpoint).

---

## Build

### Phase 1 — Backend route + parser

**File:** `backend/processing/tab_import.py` (new)

```python
"""
Import user-uploaded Guitar Pro tab files (.gp, .gp5, .gpx) and validate
they are well-formed. The file replaces the auto-generated .gp5 for that
song so alphaTab renders the user's hand-edited tab.
"""

import shutil
from pathlib import Path

ALLOWED_TAB_EXTS = {'.gp', '.gp5', '.gpx'}
MAX_TAB_BYTES = 10 * 1024 * 1024  # 10 MB — Guitar Pro files are typically tiny but allow headroom

def validate_tab_file(path: str) -> dict:
    """Open the file with PyGuitarPro to confirm it's well-formed.
    Returns {ok: bool, error: str|None, meta: {title, artist, tempo, tracks_count}|None}.
    """
    try:
        import guitarpro  # PyGuitarPro
    except ImportError:
        # Defensive: if the lib isn't installed yet, accept the file without
        # deep validation (extension check still enforced by the route).
        return {"ok": True, "meta": None, "error": None,
                "warning": "PyGuitarPro not installed — file accepted without structural validation"}
    try:
        song = guitarpro.parse(path)
    except Exception as e:
        return {"ok": False, "error": f"Could not parse Guitar Pro file: {e}", "meta": None}
    meta = {
        "title": getattr(song, "title", "") or "",
        "artist": getattr(song, "artist", "") or "",
        "tempo": getattr(song, "tempo", 0) or 0,
        "tracks_count": len(getattr(song, "tracks", []) or []),
    }
    return {"ok": True, "error": None, "meta": meta}

def install_tab_file(src_path: str, job_dir: Path, original_filename: str) -> str:
    """Place the validated tab file at outputs/<job_id>/guitarpro/user-imported.gp5
    (renamed to .gp5 if not already, since alphaTab reads .gp5 by convention).
    Returns the final on-disk path."""
    gp_dir = job_dir / "guitarpro"
    gp_dir.mkdir(parents=True, exist_ok=True)
    dest = gp_dir / "user-imported.gp5"
    # Keep the original alongside under its original extension for archival
    archival = gp_dir / f"user-original{Path(original_filename).suffix.lower()}"
    shutil.copy2(src_path, archival)
    shutil.copy2(src_path, dest)
    return str(dest)
```

**Tests:** `backend/tests/test_tab_import.py`. Cover:
- A valid `.gp5` fixture committed under `backend/tests/fixtures/sample.gp5` — assert `validate_tab_file` returns `ok: True` with meta.
- A bogus binary blob — assert `ok: False`.
- A wrong-extension file (e.g., `.midi`) — should never reach `validate_tab_file` (route rejects on extension).

Run existing suite — confirm no regressions: `cd ~/stemscribe && ./venv311/bin/python -m pytest backend/tests/ -v`.

### Phase 2 — Backend endpoint

**File:** `backend/routes/chord_sheet.py` (existing, where `/api/import-chart` lives). Add a sibling endpoint:

```python
import os
import tempfile
from pathlib import Path
from werkzeug.utils import secure_filename
from auth.middleware import auth_required
from middleware.validation import validate_job_id
from models.job import get_job, OUTPUT_DIR
from processing.tab_import import (
    validate_tab_file,
    install_tab_file,
    ALLOWED_TAB_EXTS,
    MAX_TAB_BYTES,
)

@chord_sheet_bp.route('/api/import-tab/<job_id>', methods=['POST'])
@auth_required
def import_tab(job_id):
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
    if ext not in ALLOWED_TAB_EXTS:
        return jsonify({'error': f'Unsupported tab file type: {ext}. Allowed: .gp, .gp5, .gpx'}), 415

    blob = f.read(MAX_TAB_BYTES + 1)
    if len(blob) > MAX_TAB_BYTES:
        return jsonify({'error': 'File too large (max 10 MB)'}), 413

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(blob); tmp.flush()
        try:
            result = validate_tab_file(tmp.name)
            if not result.get('ok'):
                return jsonify({'error': result.get('error', 'Invalid Guitar Pro file')}), 422
            job_dir = OUTPUT_DIR / job_id
            dest = install_tab_file(tmp.name, job_dir, name)
        finally:
            os.unlink(tmp.name)

    # Update job metadata so frontend knows the GP file is user-imported (vs auto-generated)
    meta = job.metadata or {}
    meta['gp_source'] = 'user-imported'
    meta['gp_source_filename'] = name
    if result.get('meta'):
        meta['gp_meta'] = result['meta']
    job.metadata = meta
    from models.job import save_job_to_disk
    save_job_to_disk(job)

    return jsonify({
        'status': 'ok',
        'path': dest,
        'meta': result.get('meta'),
    })
```

### Phase 3 — Frontend UI

#### 3a. New header button — beside "My Chart"

Edit `frontend/practice.html` header, add **after** the My Chart button at ~line 2336. Use the same Lucide-style icon family as the May 2026 reskin (consistent stroke, sized via `.header-icon` class which already exists):

```html
<button class="header-btn" id="importTabBtn" title="Upload your own Guitar Pro tab (.gp / .gp5 / .gpx) for this song"><svg class="header-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 6h18M3 10h18M3 14h18M3 18h18"/><path d="M7 4v16M17 4v16"/></svg>My Tab</button>
```

(Six horizontal lines + two vertical lines = abstract guitar tab staff. Lucide-grade but readable. Kevin's brand icon for "My Tab" eventually replaces this — see brief `docs/kevin-header-icons-brief-2026-05-23.md`.)

#### 3b. Modal markup

Reuse the chord-import modal pattern. Add a small new modal (or extend the chord modal with a tab tab — your call, but a separate modal is cleaner because tab files are fundamentally different from chord text).

```html
<div id="importTabModal" class="practice-modal" hidden>
  <div class="modal-backdrop" onclick="hideImportTabModal()"></div>
  <div class="modal-body" style="max-width:540px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
      <h2 style="font-family:Righteous,cursive; margin:0;">Upload your tab</h2>
      <button onclick="hideImportTabModal()" style="background:none; border:none; color:var(--text-dim); font-size:1.5rem; cursor:pointer;">×</button>
    </div>
    <div class="import-tab-upload" style="border:2px dashed var(--border); border-radius:10px; padding:1.5rem; text-align:center;">
      <div style="font-family:Righteous,cursive; font-size:1.05rem; margin-bottom:0.5rem;">Drop a Guitar Pro file</div>
      <div style="color:var(--text-dim); font-size:0.85rem; margin-bottom:1rem;">.gp, .gp5, or .gpx — 10 MB max</div>
      <input type="file" id="importTabFile" accept=".gp,.gp5,.gpx" style="display:none;">
      <button type="button" class="header-btn" onclick="document.getElementById('importTabFile').click();">Choose file…</button>
      <div id="importTabStatus" style="margin-top:0.75rem; min-height:1.2em; color:var(--text-dim); font-size:0.85rem;"></div>
    </div>
    <p style="color:var(--text-dim); font-size:0.8rem; margin-top:1rem;">
      Your uploaded tab replaces the auto-generated one for this song. You can re-upload anytime.
    </p>
  </div>
</div>
```

#### 3c. JS wiring

```js
function showImportTabModal() {
  const m = document.getElementById('importTabModal');
  if (m) { m.hidden = false; document.getElementById('importTabStatus').textContent = ''; }
}
function hideImportTabModal() {
  const m = document.getElementById('importTabModal');
  if (m) m.hidden = true;
}

document.getElementById('importTabBtn')?.addEventListener('click', showImportTabModal);

document.getElementById('importTabFile').addEventListener('change', async function() {
  const f = this.files && this.files[0];
  if (!f) return;
  const status = document.getElementById('importTabStatus');
  status.textContent = 'Uploading ' + f.name + '…';
  const jobId = window.currentJobId;
  if (!jobId) { status.textContent = 'No song loaded.'; return; }
  const fd = new FormData();
  fd.append('file', f);
  try {
    const ah = (window.StemScriber && window.StemScriber.authHeaders) ? window.StemScriber.authHeaders() : {};
    const resp = await fetch(API_BASE + '/import-tab/' + encodeURIComponent(jobId), {
      method: 'POST', body: fd, headers: ah,
    });
    const data = await resp.json();
    if (!resp.ok) { status.textContent = data.error || ('Upload failed (' + resp.status + ')'); return; }
    status.textContent = 'Uploaded. Reloading tab…';
    // Find the function that loads the GP into alphaTab in this file and call it.
    // Likely candidates: reloadAlphaTab(), renderGuitarPro(), loadTabForCurrentJob().
    // Grep for `alphaTab` / `at.load` / `api.load` in practice.html to identify.
    if (typeof reloadAlphaTab === 'function') reloadAlphaTab();
    else window.location.reload();  // last-resort fallback
    setTimeout(hideImportTabModal, 800);
  } catch (e) {
    status.textContent = 'Upload error: ' + e.message;
  }
});
```

(The `reloadAlphaTab()` function isn't guaranteed to exist by that name — grep for `alphaTab` in `practice.html` to find the actual reload entrypoint, then call it. If none exists cleanly, fall back to `window.location.reload()` — acceptable for v1.)

---

## Scope — v1

**Included:**
- Guitar Pro 5 binary format (`.gp5`)
- Guitar Pro 6 ZIP format (`.gpx`)
- Guitar Pro 3/4 (`.gp`) — PyGuitarPro handles these too
- Per-song upload, replaces auto-generated `.gp5` for that song
- Owner-only access (job must belong to caller)
- Validation via PyGuitarPro (rejects malformed files)
- Original file archived alongside for safety

**Out of v1 (defer):**
- **MuseScore `.mscz`** — different library (`musescore`/`music21`). Add later if customers ask.
- **MIDI `.mid`** — would need conversion to GP5. Already have `midi_to_gp.py` so technically doable, but cleaner as a v2.
- **Tab editing in-browser** — alphaTab can render but not edit. Editing would need a separate full editor pane. Out of scope for launch.
- **Per-stem tab upload** (e.g., upload only your bass tab) — for v1, one user-imported `.gp5` replaces all per-stem auto-tabs.
- **Round-trip download** — let users download their imported tab. Trivial follow-up.

---

## ⚠️ WATCH OUT — patterns this project keeps tripping on

- **The auth header.** Every fetch to a user-data endpoint must include `window.StemScriber.authHeaders()`. Three bugs in this codebase came from missing it: the upload not stamping `user_id` (2026-05-22), the library returning empty on practice.html (2026-05-23), and the chord-import flow that almost shipped without it. This is the fourth — don't make it bug #4.
- **Drift-managed practice.html.** Always `scp` DOWN, patch `/tmp`, `scp` UP, checksum-verify. Never edit local and push.
- **Owner check on every endpoint that touches `outputs/<job_id>/`.** Same pattern as `/api/import-chart/<job_id>`: load job, compare `job.user_id == str(g.current_user.id)`, 403 if not. Otherwise any signed-in user could overwrite anyone's tab.
- **Healthcheck timer** (`stemscribe-healthcheck.timer`) auto-restarts on 2 failed probes. If your deploy takes >75s to come back up, the timer might restart it mid-startup. Use `systemctl stop stemscribe-healthcheck.timer` during long deploys, re-enable after.

---

## Legal note

User-uploaded Guitar Pro files attached to that user's own song are **fine** per Alexandra Mayo (confirmed by Jeff 2026-05-22). What's NOT OK:
- ❌ Storing a shared/scraped library of GP files served to other users
- ❌ Using uploaded tabs to train any model
- ❌ Pulling tabs from Ultimate Guitar / Songsterr / mySongBook on the server side (those are licensed databases — even if Jeff has personal subs, the *app* can't redistribute)

Per-job per-user storage at `outputs/<job_id>/guitarpro/user-imported.gp5` is the correct scope. **Don't build any cross-user aggregation.**

---

## Acceptance — Jeff will run these

1. Sign in to stemscriber.com on desktop. Open any of his completed songs in practice mode.
2. Click **My Tab** → modal opens with a "Choose file…" button.
3. Click **Choose file…** → file dialog opens, filters to `.gp/.gp5/.gpx`.
4. Select a real Guitar Pro file from Jeff's Mac (he has hand-edited exports + mySongBook downloads) → status shows "Uploading…", then "Uploaded. Reloading tab…", modal closes, the practice-page tab pane re-renders with **the user's tab content** (visually different from the auto-generated one).
5. **Reload the page** → the imported tab persists (it's now the default at `outputs/<job_id>/guitarpro/user-imported.gp5`).
6. Repeat with a `.gp5` and a `.gpx` file — same outcome.
7. Try a malformed file (rename a `.txt` to `.gp5`) → clear "Could not parse Guitar Pro file" message in the modal. No silent failure, no crash.
8. Anonymous `curl -X POST https://stemscriber.com/api/import-tab/<any-job-id> -F file=@tab.gp5` returns 401.
9. Signed-in `curl` with a wrong-owner job ID returns 403.
10. File >10 MB returns 413.
11. Existing functionality unchanged: My Chart paste/upload still works; auto-generated GP for other songs still renders normally.

When 11 of 11 pass, deploy is complete. Update `stemscriber_full_state.md` memory noting the tab-import feature shipped.

---

## Don't-do-again reminders

- **Never edit local + scp up on the drift-managed files.** Always scp DOWN, patch /tmp, scp UP, checksum-verify.
- **Auth header on every user-data fetch.** Fourth time this codebase has hit this — get it right.
- **Owner check on every endpoint that writes to `outputs/<job_id>/`.** A missing owner check = any user can overwrite another user's tab.
- **No shared tab library.** Per-job, per-user only. Cross-user pooling = the legal trap from April 2026.
- **Test against real Jeff files before declaring done.** A `.gp5` Jeff hand-edited from a previous StemScriber export is the primary acceptance fixture.

---

## Pointers (read only if stuck)

- Chord-import brief (sibling pattern): `docs/import-chart-build-brief-2026-05-23.md`
- StemScriber memory index: `~/.claude/projects/-Users-jeffkozelski/memory/MEMORY.md`
- Master project state: `~/.claude/projects/-Users-jeffkozelski/memory/stemscriber_full_state.md`
- Kevin icon brief (for eventual brand-style `my-tab.png`): `docs/kevin-header-icons-brief-2026-05-23.md`
- Plain-language style with Jeff: working musician, not an engineer. Analogies, not jargon. Don't use "honest" as a filler. Big visuals over tiny text.

Ship it.
