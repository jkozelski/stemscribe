# Library Feature — Cold-Start Build Brief

**For:** a fresh agent. Read this top to bottom, then build. Everything below was investigated 2026-05-22 — do NOT re-investigate, just build.

## The goal

StemScriber needs an account **Library** — a page that lists every song the logged-in user has processed. Hard requirement from Jeff: **it must sync across devices.** Process a song on desktop, it appears on phone, and vice versa — the same list everywhere the user signs in. This works automatically if the list is keyed on `user_id` (not the per-device session cookie).

## What's already there (verified 2026-05-22 — don't re-check)

- **Jobs persist to disk.** `backend/models/job.py`: `save_job_to_disk(job)` writes `outputs/<job_id>/job_metadata.json`; `load_all_jobs_from_disk()` repopulates the in-memory `jobs` dict on startup (only jobs with valid stems).
- **Jobs carry `user_id`.** `routes/api.py` upload handler sets `job.user_id = str(g.current_user.id)` when the uploader is logged in (else `None`). Anonymous uploads have no owner.
- **Job fields available** (`job.to_dict()` / attributes): `job_id`, `filename`, `status`, `created_at` (float epoch, set in `__init__`), `metadata` dict (has `title`, `artist`, `album`, `duration` from ID3), `stems`, `chord_progression`, `gp_files`, `user_id`.
- **`job.created_at`** exists — use it to sort newest-first.
- **No `library.html` exists** — it's a clean new page, zero drift risk.
- **CSS theme** (from `practice.html` `:root`): `--bg-deep:#0d0d12 --bg-dark:#13131a --bg-card:#1a1a24 --orange:#ff7b54 --pink:#ff6b9d --yellow:#e5c07b --text:#e8e4df --text-dim:#7a7a85 --border:#2a2a35`. Headings use the `Righteous` font.

## ⚠️ DRIFT WARNING — read before deploying anything

`routes/api.py`, `chart_formatter.py`, and `practice.html` have all **diverged between local and prod** — prod is AHEAD of local (prod `api.py` = 1000 lines, local = 922). **You CANNOT edit local and `scp` it up — you'd wipe prod-only code.** The discipline used for every deploy this week:
1. `scp` the file DOWN from prod to `/tmp/`
2. Patch the `/tmp` copy surgically (Python string-replace on exact anchors, assert anchor unique)
3. `python3 -c "import ast; ast.parse(...)"` syntax check
4. Back up prod: `ssh ... cp <file> <file>.PREDEPLOY-library-20260522`
5. `scp` patched copy UP
6. Verify `shasum -a 256` matches local↔prod
7. `systemctl restart stemscribe`, confirm `systemctl is-active` + import check
This is task #47 (reconcile the divergence) — not your job, just don't get bitten by it.

Prod: `ssh -i ~/.ssh/stemscribe_hetzner root@5.161.203.112`, app at `/opt/stemscribe/`, restart `systemctl restart stemscribe`.

## Build part 1 — backend: rewrite `/api/jobs`

`routes/api.py` ~line 825, `list_jobs()` is currently a STUB that returns **every job from every user** (`jobs.values()` unfiltered) — that's both wrong for a library and a privacy leak. Replace it with an account-filtered version:

```python
@api_bp.route('/api/jobs', methods=['GET'])
@auth_required(optional=True)
def list_jobs():
    """The logged-in user's processed songs, newest first — powers the account
    Library. Keyed on user_id so the same list appears on every device the user
    signs into. Anonymous callers get an empty list (a library needs an account)."""
    user = getattr(g, 'current_user', None)
    if not user:
        return jsonify({'jobs': [], 'authenticated': False})
    uid = str(user.id)
    mine = [j for j in jobs.values() if j.user_id == uid]
    mine.sort(key=lambda j: getattr(j, 'created_at', 0), reverse=True)
    library = []
    for j in mine:
        meta = j.metadata or {}
        library.append({
            'job_id': j.job_id,
            'title': meta.get('title') or (j.filename or 'Untitled').rsplit('.', 1)[0],
            'artist': meta.get('artist') or '',
            'created_at': getattr(j, 'created_at', 0),
            'status': j.status,
            'has_chords': bool(j.chord_progression),
            'has_stems': bool(j.stems),
            'duration': meta.get('duration'),
        })
    return jsonify({'jobs': library, 'authenticated': True, 'count': len(library)})
```
Verify `g`, `jobs`, `jsonify`, `auth_required` are already imported in `routes/api.py` (they are — other routes use them).

## Build part 2 — frontend: `frontend/library.html`

A new page. Requirements:
- On load, fetch `/api/jobs` with the user's auth token in the `Authorization: Bearer <token>` header. **FIND the auth pattern first** — grep `practice.html` / `index.html` for how the JWT is stored (likely `localStorage`) and sent. Match whatever the rest of the site does. (This was NOT fully nailed in the 2026-05-22 investigation — the prior grep only confirmed `API_BASE` exists; you must find the token storage key.)
- If `authenticated:false` → show a "Sign in to see your library" state with a link to sign-in.
- If authenticated → render each job as a card: title (Righteous font), artist, processed-date (format `created_at`), and an "Open" action linking to `practice.html?job=<job_id>`.
- Responsive: CSS grid that is multi-column on desktop, single column on phone. This is the cross-device requirement made visible — same page, both form factors.
- Match site theme (CSS vars above). Big visuals over tiny text per `[[feedback_visual_language]]` memory.
- Empty state if `count:0`: "No songs yet — upload one to get started."
- Keep v1 simple: list + open. NO delete button in v1 (Jeff didn't ask for it; deletion adds risk).

## Build part 3 — make it reachable

The Library needs a nav entry point in the app so users find it. The app shell is `index.html`. Adding a nav link means editing `index.html` — which has drift; drift-check it the same way. If wiring nav cleanly is risky, ship `library.html` standalone first, get Jeff's eyes on it, then wire nav as a second small deploy.

## Done = 

- `/api/jobs` returns only the caller's jobs, newest first — deployed, verified with a real logged-in request
- `library.html` live, lists songs, opens them, works on desktop AND phone (resize-test or real phone)
- Jeff can sign in on his phone and see the same songs as desktop

## Context you may want (don't need to read unless stuck)

- `docs/TOMORROW-2026-05-22.md` — yesterday/today's handoff
- `stemscriber_full_state.md` memory — full project state
- Launch is June 20 (Refinery soft launch). Library is a real v1.0 gap, not a nice-to-have — Jeff hit the exact use case (away from desktop, couldn't find his own processed song).
