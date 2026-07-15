# PWA (Progressive Web App) — Build Brief

**For:** a fresh Claude Code agent. Read top to bottom, then build. Everything below was investigated 2026-05-24 — don't re-investigate, just build. Length is intentional.

---

## The goal in one paragraph

Make stemscriber.com installable as an "app" on iOS and Android home screens. When a user adds it to their home screen, it gets a real icon, opens fullscreen (no browser chrome — no URL bar, no bottom toolbar), runs offline-safe for already-loaded pages, and feels like a native app. **No App Store, no rewrite, no review process.** This is the cheapest, fastest "app" path and ships before the June 20 soft launch at Refinery. Real native iOS via Capacitor is a separate, post-launch piece (see Task #13 follow-up).

Jeff's audience uses StemScriber on phones while practicing (instrument in one hand, phone on the music stand). The PWA experience is materially better than a Safari tab — the URL bar / bottom controls eat ~25% of phone screen real estate; fullscreen reclaims it.

---

## Project context (skim then move on)

- **What StemScriber is:** web app, stem separation + chord detection + practice player. Production at `https://stemscriber.com`. Hetzner VPS at `5.161.203.112`, code at `/opt/stemscribe/`.
- **Stack:** Python/Flask backend, vanilla-JS frontend. `frontend/index.html` is the app shell (signed-in upload + library), `frontend/landing.html` is the marketing site, `frontend/practice.html` is the practice page.
- **Launch:** June 20, 2026 (Refinery, Charleston). This PWA work should land at least a week before so it's stable through launch traffic.
- **SSH:** `ssh -i ~/.ssh/stemscribe_hetzner root@5.161.203.112`
- **Restart:** `systemctl restart stemscribe`

---

## ⚠️ DRIFT WARNING — read before deploying anything

Three files have prod-vs-local DRIFT — prod is AHEAD of local. **You cannot edit local and `scp` it up.** You will wipe prod-only code.

- `frontend/index.html` (drift-managed — you WILL need to touch this)
- `frontend/practice.html` (drift-managed — touch only if you add SW registration here too)
- `backend/routes/api.py` (drift-managed — don't touch for this build)

**Deploy discipline for every prod change:**
1. `scp` file DOWN from prod to `/tmp/`
2. Patch the `/tmp/` copy surgically — exact-anchor string replace, grep first to confirm anchor uniqueness
3. Syntax check (Python: `python3 -c "import ast; ast.parse(...)"`; HTML: Python's `html.parser` with depth counter; JS: `node --check`)
4. Back up prod with timestamped name: `cp <file> <file>.PREDEPLOY-pwa-20260524`
5. `scp` UP
6. Verify `shasum -a 256` matches local↔prod
7. `systemctl restart stemscribe` (or just confirm the static-file change served correctly; the backend has `Cache-Control: no-store` on HTML/JS so changes are picked up immediately)
8. Verify with `curl https://stemscriber.com/<file>` and check the new bytes are live

This is the discipline; every successful deploy this week followed it.

---

## What's already there (verified 2026-05-24 — don't re-verify)

### PWA icons already exist on prod

In `/opt/stemscribe/frontend/images/`:
- `icon-192.png` (192×192 PNG)
- `icon-192.svg` (vector)
- `icon-512.png` (512×512 PNG)
- `icon-512.svg` (vector)
- `apple-touch-icon.png` (180×180 — for iOS home screen)
- `favicon.png`, `favicon.svg`, `favicon.ico`
- `logomark.png` (full brand mark)

You should NOT need new icons from Kevin. The existing 192/512 PNGs satisfy the PWA manifest icon requirements.

### Apple touch icon already linked

`frontend/index.html` line ~31:
```html
<link rel="apple-touch-icon" sizes="180x180" href="/images/apple-touch-icon.png">
```

That alone makes iOS "Add to Home Screen" use the right icon. But it doesn't make the launched app fullscreen — that needs the apple-mobile-web-app meta tags (see below).

### ⚠️ Existing "service worker nuke" code

Several pages (including `practice.html`, `index.html`) have this near the top of `<body>`:

```html
<script>
    if (navigator.serviceWorker) { navigator.serviceWorker.getRegistrations().then(function(regs) { regs.forEach(function(r) { r.unregister(); }); }); }
</script>
```

This was added in May 2026 to nuke stale service workers that were caching old code and breaking deploys. **If you register a new service worker, this code will immediately unregister it.** You must update the nuke code to skip the new PWA service worker by name. Strategy:

```html
<script>
    if (navigator.serviceWorker) {
      navigator.serviceWorker.getRegistrations().then(function(regs) {
        regs.forEach(function(r) {
          // Skip the PWA service worker — keep the cache nuke for everything else
          if (r.active && r.active.scriptURL && r.active.scriptURL.indexOf('/pwa-sw.js') !== -1) return;
          r.unregister();
        });
      });
    }
</script>
```

Apply this same replacement everywhere the old nuke appears (grep `r.unregister()` across `frontend/`).

### Cache-Control header is `no-store`

App pages and JS files are served with `Cache-Control: no-store, no-cache, must-revalidate`. Your PWA service worker must respect this — DON'T aggressively cache HTML/JS, or you'll re-create the exact problem the nuke code was added to fix. **Cache strategy: stale-while-revalidate for static assets (images, fonts), network-first for HTML and API responses.**

### CSP allows what you'll need

`backend/app.py` CSP currently includes:
- `script-src 'self' 'unsafe-inline'` — your service worker registration JS will run inline
- `worker-src 'self' blob: cdn.jsdelivr.net` — service workers can be served from same origin ✓
- `connect-src 'self' accounts.google.com plausible.io` — your SW can fetch from same origin ✓

No CSP changes needed for the PWA itself.

---

## Build

### Phase 1 — Manifest

**File:** `frontend/manifest.webmanifest` (new)

```json
{
  "name": "StemScriber",
  "short_name": "StemScriber",
  "description": "Tear the sound apart — stem separation and practice tools for musicians.",
  "start_url": "/?source=pwa",
  "scope": "/",
  "display": "standalone",
  "orientation": "any",
  "background_color": "#0d0d12",
  "theme_color": "#ff7b54",
  "icons": [
    {
      "src": "/images/icon-192.png",
      "sizes": "192x192",
      "type": "image/png",
      "purpose": "any maskable"
    },
    {
      "src": "/images/icon-512.png",
      "sizes": "512x512",
      "type": "image/png",
      "purpose": "any maskable"
    }
  ],
  "categories": ["music", "education", "productivity"],
  "shortcuts": [
    {
      "name": "Practice Mode",
      "short_name": "Practice",
      "url": "/practice.html",
      "icons": [{"src": "/images/icons/practice.png", "sizes": "any"}]
    },
    {
      "name": "Library",
      "short_name": "Library",
      "url": "/?openLibrary=1",
      "icons": [{"src": "/images/icons/library.png", "sizes": "any"}]
    }
  ]
}
```

Serve it with `Content-Type: application/manifest+json`. Flask will guess from the `.webmanifest` extension; if it doesn't, force it in the static handler or add a small route.

### Phase 2 — Service Worker

**File:** `frontend/pwa-sw.js` (new, served from `/pwa-sw.js`)

Critical constraints:
- **Same-origin only**. Scope must be `/`.
- **Network-first for HTML and API** (the existing `no-store` headers must NOT be defeated).
- **Stale-while-revalidate for `/images/*`, `/css/*`, fonts** — these change rarely.
- **NEVER cache `/api/*`** — auth-bearing responses must not leak across users.
- **Skip caching anything with `Authorization` header** as a safety net.

```js
const CACHE = 'stemscriber-static-v1';
const STATIC = ['/images/icon-192.png', '/images/icon-512.png', '/images/logomark.png'];

self.addEventListener('install', (e) => {
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(STATIC).catch(() => null)));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;          // never proxy cross-origin
  if (e.request.method !== 'GET') return;                   // GET only
  if (e.request.headers.get('Authorization')) return;       // never cache auth-bearing
  if (url.pathname.startsWith('/api/')) return;             // never cache API
  if (url.pathname.endsWith('.html')) return;               // network-first for HTML (default)
  // Stale-while-revalidate for static assets
  if (/^\/(images|css|js|favicon|fonts)\//.test(url.pathname) ||
      url.pathname === '/manifest.webmanifest') {
    e.respondWith((async () => {
      const cache = await caches.open(CACHE);
      const cached = await cache.match(e.request);
      const fetchPromise = fetch(e.request).then((resp) => {
        if (resp.ok) cache.put(e.request, resp.clone());
        return resp;
      }).catch(() => cached);
      return cached || fetchPromise;
    })());
  }
});
```

Linter-clean it as you like; the logic is the requirement.

### Phase 3 — HTML hookup

In **`frontend/index.html`** (and `practice.html` if you want PWA install on that page too), inside `<head>`:

```html
<link rel="manifest" href="/manifest.webmanifest">
<meta name="theme-color" content="#ff7b54">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="StemScriber">
```

(The `apple-touch-icon` link is already there — leave it.)

Then near the bottom of `<body>`, after the existing service-worker nuke (which you've updated per the warning above):

```html
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
      navigator.serviceWorker.register('/pwa-sw.js', { scope: '/' }).catch(function(e){
        console.warn('PWA SW registration failed:', e);
      });
    });
  }
</script>
```

### Phase 4 — Install prompt (optional polish)

Android Chrome fires a `beforeinstallprompt` event when the app is installable. Capture it and show a soft "Install StemScriber as an app" banner the user can tap. iOS doesn't expose this — instead, the iOS path is a small one-time hint: *"Tap Share → Add to Home Screen to install."* Show this only on iOS Safari (sniff via `/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.matchMedia('(display-mode: standalone)').matches`), dismissible.

Don't be aggressive — one prompt, dismissible, never again unless they tap. Store dismissal in `localStorage.setItem('pwa-install-dismissed', '1')`.

---

## Scope — v1

**In v1:**
- Manifest + icons (already exist)
- Service worker (network-first for HTML/API, stale-while-revalidate for static)
- iOS/Android meta tags
- Service-worker-nuke compatibility (skip the PWA SW from being unregistered)
- Optional: dismissible install hint on landing + index

**Out of v1 (don't do these now):**
- **App Store distribution** — that's Capacitor work, separate effort post-launch
- **Offline song-processing** — processing requires the backend GPU pipeline; "offline mode" can only show already-loaded library cards, NOT new uploads or playback of un-downloaded stems
- **Background audio API tricks** — leave practice-page audio as-is; PWA doesn't change Web Audio behavior on iOS
- **Push notifications** — iOS PWA push is iOS 16.4+ AND only after add-to-home-screen, and requires Apple Push Notification setup. Defer to Capacitor pass.

---

## Acceptance — Jeff will run these

1. Open `https://stemscriber.com` in iOS Safari → tap Share button → "Add to Home Screen" → confirm. Result: icon on home screen labeled "StemScriber", uses the real PNG icon (not a screenshot of the page).
2. Tap the home-screen icon → app launches **fullscreen** (no Safari URL bar, no bottom toolbar). Status bar is dark/translucent.
3. Sign in works the same as in Safari (Google OAuth popup).
4. Library shows your songs (cross-device sync intact).
5. Practice page plays audio normally.
6. Force-close + reopen with airplane mode on: home page loads from cache (at least the shell), shows a clear "you're offline" or empty state — does NOT show a broken white screen.
7. Same flow on Android Chrome: PWA install prompt appears, install works, fullscreen launch.
8. **Critical regression check**: the existing service-worker-nuke still nukes any OTHER stale SWs, but does NOT nuke `/pwa-sw.js`. Verify by visiting after install and checking `navigator.serviceWorker.getRegistrations()` in DevTools — exactly one registration (the PWA one) should remain.
9. **API hygiene**: no `/api/*` response is ever served from cache. Verify by signing in, then signing out, then refreshing — old user's data must NOT appear.
10. Lighthouse PWA audit on stemscriber.com scores ≥ 90 on the PWA criteria.

---

## Don't-do-again reminders

- **Never edit local + scp up on the drift-managed files.** Always scp DOWN, patch /tmp, scp UP, checksum-verify.
- **Service workers can outlive deploys.** If a user installed the PWA, then you deploy a broken SW, their phone will keep running the broken one. Always bump the cache version (`CACHE = 'stemscriber-static-v2'` etc.) when you change the SW behavior — this triggers the `activate` cleanup.
- **Never cache `/api/*`** — auth-bearing responses are user-specific. Cross-user leak is a privacy bug.
- **iOS Safari is the strictest target.** Test there before declaring done. Android Chrome is permissive; iOS is not.
- **Don't aggressively prompt** — one dismissible hint, never spam. The "Add to Home Screen" prompt fatigue is real.

---

## Legal note

PWAs don't change any of the existing legal posture (chord lookup, lyrics, scraped content). The only legal-adjacent piece is iOS push notifications IF you add them later — those require Apple Developer Program enrollment and an APNs cert. Not in scope for v1.

---

## Pointers (read only if stuck)

- StemScriber memory index: `~/.claude/projects/-Users-jeffkozelski/memory/MEMORY.md`
- Master project state: `~/.claude/projects/-Users-jeffkozelski/memory/stemscriber_full_state.md`
- Brand colors: `--orange: #ff7b54`, `--pink: #ff6b9d`, `--bg-deep: #0d0d12`, `--bg-card: #1a1a24`
- Plain-language style with Jeff: working musician, not an engineer. Analogies, not jargon. Big visuals over tiny text.
- The mobile menu icon-padding fix (responsive.css) just shipped 2026-05-24 — your changes should not conflict.

Ship it.
