# Kevin Brief — 4 Header Icons for Practice Page

**For:** Kevin Hanley (social/brand lead). **Date raised:** 2026-05-23.
**Why now:** practice-page header was reskinned 2026-05-23 with utility stroke SVGs as placeholders. They're orderly but generic. The Library panel + home page already use Kevin's bold flat-vector icon style (`library.png`, `practice.png`, `settings.png`); the practice header should match.

## What to deliver

Four icons in the **same visual family** as the existing brand set:
- `library.png`
- `practice.png`  
- `settings.png`

(Look at any of those on stemscriber.com — they're the small colored chip icons in the top-right menu.)

### The four icons needed

| File name (suggested) | Use | What it should depict |
|---|---|---|
| `tuner.png` | Header "Tuner" button → links to tuner.html | A guitar tuner. Needle-style meter, or a simple tuning-fork shape, or a chromatic-tuner display. Read at 18×18 px on a dark background. |
| `scales.png` | Header "Scales" button → links to fretboard.html | A piano keyboard (cluster of white+black keys) **or** a guitar fretboard slice. Either reads as "scales/notes." Pick whichever sits nicer at 18 px. |
| `my-chart.png` | Header "My Chart" button → opens chord-import modal | A page/document with chord-name marks above lyric lines. The intent is "your own chord chart, imported or pasted." A simple doc icon with two or three chord-name dots above lines reads instantly. |
| `back.png` | Header "Back" link → returns to upload page | Left-pointing arrow in the same brand chip style. Simple, but having it match the family removes the only remaining stroke-icon outlier. |

(If the Back arrow feels redundant — i.e., a flat brand-style left-arrow doesn't add anything over the current stroke SVG — drop it from the order. Tuner / Scales / My Chart are the three that matter visually.)

### Specs to match the existing set

- **Format:** PNG with transparent background. Vector source (Figma / Illustrator / Affinity) kept too — we may need recolored variants for hover/active states later.
- **Source resolution:** 96×96 or 128×128 (we'll render at 18×18 and 22×22; need headroom for retina + future bigger placements).
- **Style:** bold, flat, slightly chunky vector. Single-color or two-color (orange + pink gradient is the brand palette: `--orange: #ff7b54`, `--pink: #ff6b9d`). Match what `library.png` does — it's the reference.
- **Background:** ideally a soft rounded-square chip behind the glyph (matches how `library.png` reads with `border-radius:4px` rendering). If you ship just the glyph on transparent, that works too — we'll apply `border-radius` in CSS.
- **Padding:** ~8% inset so glyphs don't kiss the edge.
- **Mood:** musician-friendly, warm, not corporate. Same energy as the 7-stem mixer icons.

### Where they'll live

Filenames go into `/opt/stemscribe/frontend/images/icons/` (next to the existing `library.png` / `practice.png` / `settings.png`).

Once delivered, the swap is one line per icon in `practice.html` — replace each `<svg class="header-icon">...</svg>` with `<img class="header-icon" src="/images/icons/<name>.png" alt="">`. Trivial integration on our end.

### Timeline

No fixed deadline, but launch is **June 20**. Anything that lands by **June 10** gets in for the soft launch; later than that, the placeholder stroke icons stay live and we swap post-launch.

### Reference materials

- Existing brand icons to match (live on prod): `https://stemscriber.com/images/icons/library.png`, `.../practice.png`, `.../settings.png`.
- The 7-stem mixer icons (also the same family): `https://stemscriber.com/practice.html?job=d9e3368e-b8f8-45fe-a7e9-1b0df9a66285` (the "STEM MIXER" section on the left).
- The placeholder stroke SVGs currently shown (so you can see what's being replaced): same URL as above, in the top header row.
- Brand colors: `--orange: #ff7b54`, `--pink: #ff6b9d`, on dark backgrounds `--bg-deep: #0d0d12` / `--bg-card: #1a1a24`.

Send drafts to `jkozelski@gmail.com` for sign-off before final.
