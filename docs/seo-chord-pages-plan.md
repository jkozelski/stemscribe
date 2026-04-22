# SEO Chord Pages — Implementation Plan

## Status: BUILT (not deployed)

Built and tested locally. All routes verified. Ready for review before deploying.

---

## Numbers

| Metric | Count |
|--------|-------|
| Artists | 663 |
| Songs | 15,418 |
| Sitemap URLs | 16,082 |
| New Flask blueprint | 1 file (routes/seo_chords.py) |
| Lines changed in app.py | 3 (import + register + sitemap index) |

---

## URL Scheme

```
/chords/                         → A-Z artist index (all 663 artists)
/chords/{artist-slug}/           → Song list for one artist
/chords/{artist-slug}/{song-slug} → Full chord chart page (the money page)
/sitemap-chords.xml              → XML sitemap of all chord pages
/sitemap.xml                     → Sitemap index (links to sitemap-static.xml + sitemap-chords.xml)
/sitemap-static.xml              → Original static pages sitemap (was /sitemap.xml)
```

Slugs reuse the existing chord_library directory names directly — no remapping needed.

Examples:
- `stemscriber.com/chords/radiohead/creep`
- `stemscriber.com/chords/led-zeppelin/stairway-to-heaven`
- `stemscriber.com/chords/nirvana/smells-like-teen-spirit`

---

## What Each Page Includes

### Song page (`/chords/{artist}/{song}`)
- **SEO meta tags:** `<title>`, `<meta name="description">`, `<link rel="canonical">`, Open Graph tags
- **Title:** e.g. "Creep Chords — Radiohead | StemScriber"
- **Description:** "Chords and lyrics for Creep by Radiohead. Uses: G, B, C, Cm. Free on StemScriber."
- **Breadcrumbs:** Chords / Radiohead / Creep
- **Chart metadata:** Key, capo, chord count
- **Chord badges:** Visual badges for each chord used
- **Full chord chart:** Server-rendered HTML — every section, chords above lyrics, monospace formatting
- **CTA box:** "Practice this song with stem separation" → links to /app
- **Internal links:** Up to 12 more songs by the same artist
- **Dark theme:** Matches existing site (same CSS variables, fonts, colors)

### Artist page (`/chords/{artist}/`)
- All songs listed in 2-column layout
- CTA for practicing that artist with stem separation

### Index page (`/chords/`)
- Grouped A-Z with song counts per artist
- Total count in page title for long-tail: "Guitar Chords & Lyrics — 663 Artists, 15,418 Songs"

---

## Files Changed

### New file
- **`backend/routes/seo_chords.py`** — Self-contained Flask blueprint with all routes, HTML templates inline (no Jinja2 template files needed), sitemap generator

### Modified files
- **`backend/app.py`** — Added import + `app.register_blueprint(seo_chords_bp)`, replaced static sitemap.xml route with sitemap index
- **`frontend/robots.txt`** — Fixed domain from stemscribe.io to stemscriber.com

---

## Architecture Decisions

1. **Server-rendered HTML, not JS** — Google crawls these directly. No JavaScript dependency. Each page is a complete HTML document.

2. **No template engine / no separate template files** — HTML is built inline in Python using f-strings + `markupsafe.escape()`. Keeps the entire feature in one file. No new dependencies.

3. **URL scheme: `/chords/` not `/c/`** — Longer but SEO-friendly. "/chords/radiohead/creep" is self-documenting and contains the keyword "chords" that people search for.

4. **Sitemap index pattern** — `/sitemap.xml` is now a sitemap index pointing to `/sitemap-static.xml` (the original) and `/sitemap-chords.xml` (15K+ URLs). This is the correct approach for large sitemaps per Google's documentation.

5. **No caching yet** — Pages are generated on each request. Each page takes ~1ms (just file read + string concatenation). Can add `@lru_cache` or pregeneration later if needed.

---

## SEO Strategy

Each song page targets: `"{song name} chords"`, `"{song name} {artist} chords"`, `"{song name} lyrics and chords"`

With 15,418 pages, we're targeting 15K+ long-tail keywords. The growth playbook identified this as the "biggest single lever" — every song is a potential organic search entry point that funnels into the app.

---

## Testing (completed)

```
/chords/               → 200, 73KB (663 artists rendered)
/chords/radiohead/     → 200, 18KB (159 songs)
/chords/radiohead/creep → 200, 12KB (full chart with meta tags)
/sitemap-chords.xml    → 200, 16,082 <url> entries
/sitemap.xml           → 200, sitemap index
/chords/nonexistent/   → 404
```

---

## Deploy Checklist

When ready to deploy:

1. `scp backend/routes/seo_chords.py root@5.161.203.112:/opt/stemscribe/backend/routes/`
2. `scp backend/app.py root@5.161.203.112:/opt/stemscribe/backend/`
3. `scp frontend/robots.txt root@5.161.203.112:/opt/stemscribe/frontend/`
4. `ssh root@5.161.203.112 "systemctl restart stemscribe"`
5. Verify: `curl -s https://stemscriber.com/chords/radiohead/creep | head -20`
6. Submit sitemap to Google Search Console: `https://stemscriber.com/sitemap.xml`
7. Optional: submit to Bing Webmaster Tools

---

## Future Enhancements (not built yet)

- **Chord diagram images** — SVG chord fingering diagrams next to chord badges
- **JSON-LD structured data** — MusicComposition schema for rich snippets
- **Page caching** — LRU cache or pre-render to static HTML for CDN
- **Search/filter** — Client-side search on the index page
- **Artist images** — Pull from Spotify/MusicBrainz for richer pages
- **Related artists** — Cross-link between similar artists
- **Print-friendly CSS** — @media print stylesheet for chord charts
