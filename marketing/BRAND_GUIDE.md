# StemScribe Brand Guide

## Brand Personality & Voice

StemScribe talks like a seasoned musician who happens to be a tech wizard -- the sound engineer at your favorite venue who knows every pedal on your board and can explain FFT transforms over a beer. The voice is:

- **Knowledgeable but never pretentious.** We say "stems" not "audio source separation outputs." We explain complex ML as "the AI does its thing."
- **Enthusiastic but not performative.** Genuine excitement about music, not marketing-speak excitement about features.
- **Casual with earned authority.** We can drop technical terms because we've earned the right -- but we always prioritize clarity over showing off.
- **Community-first.** StemScribe exists because musicians deserve better tools. We're builders who play, not suits who ship.

**Voice principles:**
1. Write like you're explaining something to a bandmate, not a customer
2. Use music terminology naturally -- "stems," "tabs," "the mix," "chops"
3. Humor is welcome when it's genuine, never forced
4. Avoid corporate language: no "leverage," "empower," "revolutionary," "seamless"
5. The tagline "Tear The Sound Apart" sets the tone -- visceral, active, physical

---

## Visual Direction

### The Aesthetic: Record Store Meets Control Room

StemScribe lives at the intersection of two spaces musicians know by heart:

1. **The record store** -- gig poster textures, hand-lettered type, ink grain, paper warmth, the smell of vinyl
2. **The control room** -- the Neve console, LED meters, warm tube glow, the precision of professional gear

The current UI already nails the control room half (the Neve console mixer is genuinely excellent). The brand guide focuses on bringing the record store warmth into the spaces around it -- the hero, the upload zone, the empty states, the marketing pages.

### Poster-Inspired Textures for Web UI

**Grain overlay (global):**
Apply a subtle noise texture to the background to break the digital flatness. This mimics offset print grain from gig posters.

```css
/* Add to .psych-bg or body */
.psych-bg::after {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
    opacity: 0.4;
    mix-blend-mode: overlay;
}
```

**Letterpress emboss effect (for headers):**
```css
.hero h1, .results-header h2 {
    text-shadow:
        0 1px 0 rgba(255,255,255,0.05),
        0 -1px 0 rgba(0,0,0,0.3);
}
```

**Torn-paper edge dividers:**
Instead of clean `border-top` dividers, use a rough/wavy SVG mask on section breaks to mimic torn poster paper. A 4px high SVG with irregular waves, colored `rgba(255,255,255,0.03)`.

**Halftone dot pattern** (for backgrounds on hover states or accents):
```css
.halftone-accent {
    background-image: radial-gradient(circle, currentColor 1px, transparent 1px);
    background-size: 4px 4px;
    opacity: 0.08;
}
```

---

## Color Palette

### Current Palette (Keep)
The existing dark theme palette is strong. These are the anchors:

| Token | Hex | Usage |
|-------|-----|-------|
| `--bg-deep` | `#0d0d12` | Page background |
| `--bg-dark` | `#13131a` | Card/section background |
| `--bg-card` | `#1a1a24` | Elevated surfaces |
| `--psych-orange` | `#ff7b54` | Primary accent, CTAs |
| `--psych-pink` | `#ff6b9d` | Secondary accent, gradients |
| `--psych-purple` | `#c678dd` | Tertiary, info panels |
| `--psych-blue` | `#61afef` | Links, interactive |
| `--psych-teal` | `#56b6c2` | Archive/live, secondary CTA |
| `--psych-yellow` | `#e5c07b` | Solo buttons, warnings |
| `--psych-green` | `#98c379` | Success, tips |
| `--cream` | `#f5f0e8` | VU meter faces, warm text |
| `--text` | `#e8e4df` | Body text |
| `--text-dim` | `#7a7a85` | Secondary text |

### New Additions: Warmth Layer

Add these to bring the record store warmth:

| Token | Hex | Usage |
|-------|-----|-------|
| `--amber-glow` | `#ffb347` | Warm highlights, tube amp feel |
| `--vinyl-black` | `#0a0a0f` | Deep blacks, vinyl record areas |
| `--poster-red` | `#e63946` | Urgency, live indicators |
| `--worn-paper` | `#e8dcc8` | Poster texture accents, blockquotes |
| `--ink-brown` | `#3d2b1f` | Earthy accents, wood tones |
| `--stage-smoke` | `rgba(255,255,255,0.04)` | Atmospheric layering |

### Instrument-Specific Accent Colors

These map to per-instrument themes in the mixer and future practice mode:

| Instrument | Primary | Secondary | Feeling |
|-----------|---------|-----------|---------|
| Drums | `#ff5252` | `#ff8a80` | Bold red -- punchy, explosive |
| Guitar | `#ffb347` | `#ffd180` | Amber/gold -- warm, analog, tube amp |
| Bass | `#7c4dff` | `#b388ff` | Deep purple -- heavy, groovy, subterranean |
| Piano | `#4dd0e1` | `#80deea` | Cool cyan -- elegant, clear, crystalline |
| Vocals | `#f48fb1` | `#f8bbd0` | Soft pink -- ethereal, floating, human |

---

## Typography

### Current Stack (Keep)
- **Righteous** -- Display/headlines. Perfect for the brand. Rounded, bold, has record-label energy without being kitschy.
- **Space Grotesk** -- Body/UI. Clean, modern, slightly techy. Good contrast with Righteous.
- **Outfit** -- Secondary body. Softer weight, used in subtitles and descriptions.

### Refinements

**Add one accent font for poster-mode contexts:**
- **Bebas Neue** (Google Fonts, free) -- All-caps condensed display font. Classic gig poster typography. Use ONLY for:
  - Landing page section headers
  - Feature callout labels
  - Poster-style promotional graphics
  - NOT in the app UI itself (keep Righteous there)

```html
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&display=swap" rel="stylesheet">
```

```css
.poster-heading {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 3px;
    text-transform: uppercase;
}
```

**Type scale tightening:**
- Hero H1: Keep at 4rem / Righteous but increase letter-spacing to 2px (currently 1px)
- Body: Keep at 1rem / Space Grotesk but increase line-height to 1.8 (currently uses browser default)
- UI labels: Standardize at 0.75rem / Space Grotesk / weight 600 / letter-spacing 1px / uppercase

---

## Per-Instrument Visual Themes

Each instrument in StemScribe has a distinct visual personality. These apply to the mixer channel strips, practice mode, and future instrument-specific pages.

### Drums: Bold, Punchy, High-Energy
- **Color:** Red/orange spectrum (`#ff5252` to `#ff7b54`)
- **Texture:** High-contrast halftone dots, bold geometric shapes
- **Motion:** Snappy, percussive micro-animations (sharp ease-out, no lingering)
- **Poster style:** Bold-edgy -- think punk flyers, high contrast black + red
- **CSS accent:**
```css
.stem-card[data-stem="drums"] { border-top: 3px solid #ff5252; }
```

### Guitar: Warm, Gritty, Analog
- **Color:** Amber/gold spectrum (`#ffb347` to `#ffd180`)
- **Texture:** Noise grain, aged paper, subtle warmth filter
- **Motion:** Smooth, slightly overdrive-like (ease-in-out with a slight overshoot)
- **Poster style:** Vintage-classic -- retro serif type, warm sepia tones, 60s/70s energy
- **CSS accent:**
```css
.stem-card[data-stem="guitar"] { border-top: 3px solid #ffb347; }
```

### Bass: Deep, Heavy, Groovy
- **Color:** Deep purple spectrum (`#7c4dff` to `#b388ff`)
- **Texture:** Low-frequency wave patterns, dense gradients, sub-bass visualization
- **Motion:** Slow, heavy, gravitational -- longer durations, heavier easing
- **Poster style:** Bold-edgy meets artistic -- dark backgrounds, neon accents, underground club aesthetic
- **CSS accent:**
```css
.stem-card[data-stem="bass"] { border-top: 3px solid #7c4dff; }
```

### Piano: Elegant, Clean, Sophisticated
- **Color:** Cool cyan/ivory spectrum (`#4dd0e1` + `#f5f0e8`)
- **Texture:** Clean lines, minimal grain, ivory key patterns, sheet music motifs
- **Motion:** Graceful, measured -- like a sustain pedal releasing
- **Poster style:** Jazz-elegant -- Art Deco lines, muted gold, sophisticated restraint
- **CSS accent:**
```css
.stem-card[data-stem="piano"] { border-top: 3px solid #4dd0e1; }
```

### Vocals: Ethereal, Floating, Expressive
- **Color:** Soft pink/rose spectrum (`#f48fb1` to `#f8bbd0`)
- **Texture:** Soft gradients, breath-like fades, aurora effects
- **Motion:** Flowing, organic -- sine-wave easing, floating particles
- **Poster style:** Artistic -- watercolor washes, dreamlike overlays, expressive brushstrokes
- **CSS accent:**
```css
.stem-card[data-stem="vocals"] { border-top: 3px solid #f48fb1; }
```

---

## Landing Page Hero Section Mockup

### Layout Description

The hero section should feel like you just walked into a legendary venue -- dark, moody, with one spotlight on the stage.

**Structure (top to bottom):**

1. **Background layer:** The existing psychedelic orbs stay, but add a subtle noise grain overlay (poster texture). The orbs should feel like stage lighting hitting fog.

2. **Logo:** `StemScribe` in Righteous, gradient stays (orange > pink > purple). The rotating four-pointed star symbol stays. Below the logo, the tagline "Tear The Sound Apart" in Outfit, 0.85rem, `--text-dim`, letter-spacing 2px.

3. **Hero headline:** Full width, centered. Two lines:
   - Line 1: "Tear The" in Righteous, white, 4rem
   - Line 2: "Sound" in Righteous, animated gradient (current behavior), 5rem -- this word POPS
   - Line 3: "Apart" in Righteous, white, 4rem
   - Total effect: "Sound" is huge, glowing, the center of gravity

4. **Subtitle:** "Drop any track. Get stems, tabs, sheet music, and practice tools. Learn from the masters." in Outfit, 1.2rem, `--text-dim`, max-width 550px centered.

5. **Visual element (new):** Between subtitle and upload zone, add a subtle animated visualization -- three horizontal lines (representing audio waveforms) that gently pulse in orange/pink/purple. These represent the three stems being "torn apart." CSS-only, no JS needed:
```css
.hero-waves {
    display: flex;
    gap: 6px;
    justify-content: center;
    margin: 2rem 0;
}
.hero-wave {
    width: 60px;
    height: 3px;
    border-radius: 2px;
    animation: wave-pulse 2s ease-in-out infinite;
}
.hero-wave:nth-child(1) {
    background: var(--psych-orange);
    animation-delay: 0s;
}
.hero-wave:nth-child(2) {
    background: var(--psych-pink);
    animation-delay: 0.3s;
}
.hero-wave:nth-child(3) {
    background: var(--psych-purple);
    animation-delay: 0.6s;
}
@keyframes wave-pulse {
    0%, 100% { transform: scaleX(1); opacity: 0.6; }
    50% { transform: scaleX(1.5); opacity: 1; }
}
```

6. **Upload zone:** Immediately below. The existing drop zone with dashed border is good but should add the poster grain texture inside and a subtle warm glow on hover (amber, not blue).

---

## Tidepool Artists Cross-Pollination

### StemScribe x Tidepool: Shared DNA

Both projects serve musicians. StemScribe helps them learn and practice; Tidepool helps them get discovered and booked. The brand connection should feel like two products from the same family -- not identical twins, but siblings who share DNA.

**Shared visual elements:**
- Both use dark themes with warm accents
- Both use poster-inspired textures (grain, halftone, torn edges)
- Both use Righteous as a display font
- The psychedelic orb background could appear on Tidepool as well (in each band's accent color)

**StemScribe integration on Tidepool:**
- Each band page on tidepoolartist.com could have a "Learn This Song" CTA that links to StemScribe with a pre-loaded track
- The visual language should make this transition feel natural -- same grain texture, same card style, same type hierarchy

**Tidepool visibility on StemScribe:**
- "Powered by Tidepool Artists" in the footer, small but present
- When a user processes a track by a Tidepool-represented band (KODA, Kozelski, Spare Kings, King Hippo, The Outervention), the track info panel could show a "This artist is on Tidepool" badge with a link

### Making Each Tidepool Band Page Visually Unique

The current problem: all 5 band pages use the same `ArtistPage` component with color swaps. They look like palette-swapped trading cards. Here are concrete ideas to differentiate:

**KODA:**
- **Style:** Bold-edgy. High contrast black and neon green/electric blue.
- **Texture:** Halftone dots, distorted grid patterns, glitch effects
- **Layout:** Asymmetric -- hero image bleeds off the edge, text overlaps image with a noise-masked edge
- **Unique element:** Animated EQ bars behind the band photo, pulsing to imply energy
- **Background:** Dark with subtle CRT scanline overlay

**Kozelski:**
- **Style:** Vintage-classic meets artistic. Warm sepia tones, analog photography feel.
- **Texture:** Film grain, light leaks, aged paper edges on photos
- **Layout:** Magazine editorial -- large type callouts, pull quotes, generous whitespace
- **Unique element:** A vinyl record illustration that rotates slowly behind the hero content (reuse StemScribe's vinyl CSS)
- **Background:** Dark brown-black (`#1a1410`) with warm amber orb

**Spare Kings:**
- **Style:** Jazz-elegant. Deep blues, warm golds, sophisticated restraint.
- **Texture:** Smooth gradients, brushed metal, subtle Art Deco line patterns
- **Layout:** Classic album-cover centered composition. Band photo in a gold-bordered frame.
- **Unique element:** Piano key pattern as a subtle decorative border. A "Live This Week" card with a different visual treatment for Saturday night vs. Sunday morning gigs.
- **Background:** Deep navy (`#0a1628`) with cool blue orbs

**King Hippo:**
- **Style:** Bold-edgy + punk energy. Bright, loud, unapologetic.
- **Texture:** Spray paint splatter, torn tape, xerox/zine aesthetic
- **Layout:** Collage style -- overlapping elements, rotated photos, handwritten-style annotations
- **Unique element:** A graffiti-style band name treatment (can be CSS text-stroke + transform rotate)
- **Background:** Near-black with neon pink/yellow splashes

**The Outervention:**
- **Style:** Artistic/experimental. Watercolor washes, cosmic textures.
- **Texture:** Soft blurred gradients, paint-drip effects, cosmic dust
- **Layout:** Full-bleed hero with the band photo emerging from an abstract color wash
- **Unique element:** Parallax scrolling on the hero -- the abstract wash moves at a different speed than the band content
- **Background:** Deep space black with aurora-like color bands (purple/teal/pink)

---

## Logo Refinement

### Current Logo
`<span class="logo-symbol">✦</span> StemScribe` -- The four-pointed star symbol rotates continuously. "StemScribe" is in Righteous with an orange-to-pink-to-purple gradient.

### Refinements

1. **Symbol:** The Unicode `✦` works but consider a custom SVG four-pointed star that has slightly rounded points and a subtle inner glow. This allows control over stroke weight and animation without depending on font rendering across devices.

2. **Rotation speed:** Currently 10s full rotation. Slow to 20s -- the current speed is slightly distracting when reading content. The rotation should feel ambient, not attention-grabbing.

3. **Gradient direction:** Change from `135deg` (diagonal) to `90deg` (horizontal left-to-right). This reads more naturally with the horizontal text flow.

4. **Favicon:** Use the `✦` symbol in a filled circle with the orange-pink gradient background. Simple and recognizable at small sizes.

5. **Wordmark lockup:** For marketing materials, create a stacked version:
   - `✦` symbol large and centered
   - `STEMSCRIBE` below in Righteous, letter-spacing 4px
   - `TEAR THE SOUND APART` below that in Space Grotesk, letter-spacing 3px, `--text-dim`

6. **Color usage:** The gradient wordmark is for dark backgrounds only. On light backgrounds, use solid `#1a1a24` (near-black). Never place the gradient text on a busy or light background.

---

## Tone of Voice Examples

### Social Media Posts

**Post 1 -- Product launch / feature announcement:**
> Just shipped A-B looping in Practice Mode. Set your loop points, slow it down to 25%, and woodshed that solo until your fingers remember it. No more rewinding. No more guessing. Just you and the riff.

**Post 2 -- Community / enthusiasm:**
> Someone just ran Cornell '77 Scarlet > Fire through StemScribe and isolated Jerry's guitar for the entire 24-minute jam. We didn't build this for that. But we're glad we did.

**Post 3 -- Educational / helpful:**
> Pro tip: Export the bass stem as MIDI, drop it into your DAW, and re-amp it through your favorite plugin. You now have the exact bass line from the record, played through YOUR tone. That's what stem separation is actually for.

### UI Microcopy

**Processing wait state (replaces "This might take a few minutes. The AI is doing its thing."):**
> Sit tight -- the AI is pulling this track apart note by note. Grab a coffee or tune your guitar.

**Empty library state:**
> Nothing here yet. Drop a track to start building your library. Every song you process lives here forever.

**Successful processing completion:**
> Done. 6 stems, MIDI for every instrument, and Guitar Pro tabs ready to download. Go learn something.

**Error state (file too large):**
> That file is too big for us right now (500MB max). Try a shorter clip or a compressed format like MP3.

**Upload hover state:**
> Drop it like it's hot.

**Speed control tooltip:**
> Slow it down. Speed it up. Your tempo, your rules.
