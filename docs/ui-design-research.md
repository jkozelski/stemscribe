# StemScribe UI Design Research
**Date:** 2026-03-16
**Palette:** Dark background, orange (#FF8800), yellow accents
**Vibe:** Aged, weathered, vintage gear — realistic and premium

---

## Part 1: Icon Style Research

### Style 1: Weathered Vintage Gear (SELECTED - Primary)
Instruments rendered with visible age, patina, and wear — road-worn guitar finishes, oxidized brass, cracked tolex, scratched chrome hardware. Sits between photorealism and stylized illustration. Musicians connect emotionally to gear that looks played and loved.

**Why it works for StemScribe:** Warm-toned patina (aged brass, amber lacquer, worn sunburst finishes) naturally complements the orange (#FF8800) palette. Vintage gear has warm amber/honey tones that feel native to the app.

**Scalability:** Best at 48px+. Below 32px, texture detail fades. Use simplified variants for sidebar.

**References:**
- [iStock — Vintage Amplifier Photos (10,500+)](https://www.istockphoto.com/photos/vintage-amplifier)
- [Dreamstime — Guitar Amp Illustrations](https://www.dreamstime.com/illustration/guitar-amp.html)
- [GraphicRiver — Guitar Icon 3D Renders](https://graphicriver.net/guitar+icon-and-music-graphics-in-graphics/3d-renders)

### Style 2: Aged Chrome / Metallic Emblem (SELECTED - Secondary)
Instruments as tarnished chrome emblems — like vintage car hood ornaments or old amp badges. Oxidized silver with warm amber backlighting. Inherently premium, dark-theme-native.

**Why it works:** Chrome highlights create high-contrast that remains readable at small sizes. Orange accent reflections on chrome look stunning — like stage lighting bouncing off hardware.

**Scalability:** Good across all sizes. Chrome highlights + silhouette remain recognizable even at 16px.

**References:**
- [3D Dark Chrome Icon Collection (Icoon.co)](https://www.icoon.co/collection/3d-dark-chrome-icon-collection)
- [Chrome Instruments Pack (Packsia)](https://www.packsia.com/motion-pack/chrome-instruments-pack)
- [Diverse Concepts 3D Metallic Finish Icons (Figma)](https://www.figma.com/community/file/1454151981625191116)

### Style 3: Worn Backstage / Roadie Aesthetic (SELECTED - Accent)
Beat-up gear on dark road cases, stage-lit with amber lighting. Gritty, authentic — cigarette burns, belt buckle rash, gaffer tape, coiled XLR cables. Tells a story.

**Why it works:** Maximum authenticity for the working musician vibe. The context (road case, amp cabinet, backstage floor) adds narrative depth.

**Scalability:** Poor below 32px — too much scene detail. Best for hero images, feature sections, onboarding.

**References:**
- Shot concepts based on touring photography aesthetics
- Best for marketing/hero contexts, not inline UI icons

### Other Styles Researched (Not Selected)

**Soft 3D Rendered** — Clean, Apple-emoji-like 3D. Good scalability but too polished/new-looking for the aged aesthetic Jeff wants.

**Studio-Lit Product Shot** — Like Apple product photography. Premium but too pristine.

**Neo-Skeuomorphic** — Best scalability across all sizes (16px to 48px+). Good fallback for simplified sidebar variants if needed.

**Detailed Illustration (Digital Painting)** — Concept art quality. Gorgeous at large sizes but doesn't scale and requires hand-crafted work.

### Scalability Summary

| Style | 16px | 32px | 48px+ |
|-------|------|------|-------|
| Weathered Vintage | Poor | OK | Excellent |
| Aged Chrome | Good | Excellent | Excellent |
| Backstage/Roadie | Poor | Poor | Excellent |
| Neo-Skeuomorphic (backup) | Good | Good | Excellent |

---

## Part 2: Button & Control Styles

### 1. Neumorphic Stomp Switch (Guitar Pedal Style)
Large circular button that protrudes from surface. Depresses inward on press via `inset box-shadow`. LED glow dot for on-state. Perfect for solo/mute/bypass toggles.
- **Tech:** Pure CSS, layered box-shadow, `:active` inset swap
- **Ref:** [Pedalboard.js](https://dashersw.github.io/pedalboard.js/) | [Guitar Pedal UI - Dribbble](https://dribbble.com/tags/guitar_pedal_user_interface)

### 2. Backlit Transport Controls (Play/Pause/Stop)
Geometric icons in softly glowing containers. Active button gets warm orange backlight. SVG morph animation between states.
- **Tech:** SVG path morphing, CSS clip-path transitions, orange glow box-shadow
- **Ref:** [Pure CSS Play/Pause - CSS-Tricks](https://css-tricks.com/making-pure-css-playpause-button/)

### 3. Rotary Knob (Amp/Plugin Style)
Skeuomorphic rotary knob with position indicator and circular value arc in orange. Controlled via vertical drag.
- **Tech:** SVG + JS drag, `stroke-dasharray` for value arc
- **Libraries:** [webaudio-controls](https://github.com/g200kg/webaudio-controls), [AudioKnobs](https://github.com/Megaemce/AudioKnobs)
- **Ref:** Neural DSP, Guitar Rig, Amplitube, Moog interfaces

### 4. Console Fader / Vertical Slider
Mixing console fader look. Wide flat rectangular thumb, groove track, orange fill. Optional dB scale markings.
- **Tech:** `<input type="range">` with custom thumb/track via vendor pseudo-elements
- **Ref:** [CSS Range Sliders](https://freefrontend.com/css-range-sliders/)

### 5. LED Toggle Indicator
Small circular toggle mimicking hardware LEDs. Off = dim dark circle. On = vibrant orange with 3-layer glow bloom.
- **Tech:** Pure CSS, stacked `box-shadow` for realistic light falloff
- **CSS:** `box-shadow: 0 0 4px #FF8800, 0 0 12px rgba(255,136,0,0.5), 0 0 20px rgba(255,136,0,0.2)`

### 6. Pad Grid Button (Ableton/Maschine Style)
Square buttons in grid layout. Dark matte resting state, orange color flood on active. Fast scale-down press animation.
- **Tech:** CSS Grid, `transition: background 50ms`, `transform: scale(0.95)` on press
- **Ref:** [Ableton Push](https://www.ableton.com/en/push/manual/)

### 7. Textured Metal Toggle Switch (Amp Channel Select)
Chrome/brushed metal cap that slides between positions. Dark recessed slot housing. Active side label glows orange.
- **Tech:** CSS checkbox hack, linear-gradient brushed metal, translateX slide
- **Ref:** [Skeuomorphic Audio Controller UI (Figma)](https://www.figma.com/community/file/859424303965872933)

### 8. Neumorphic Soft Button (Modern DAW Style)
Softly extruded from dark background using dual-shadow technique. On press, shadows invert to inset.
- **Tech:** Pure CSS, dual box-shadow (light + dark), 150ms transition
- **Tool:** [neumorphism.io](https://neumorphism.io/)
- **Ref:** [Neumorphism and CSS - CSS-Tricks](https://css-tricks.com/neumorphism-and-css/)

### 9. VU Meter / Level Indicator Bar
Horizontal/vertical bar with green-to-red gradient. Segmented variant uses discrete LED blocks. Peak hold indicator.
- **Tech:** Canvas for real-time, CSS gradient for static
- **Ref:** [LANDR - Skeuomorphism in Plugins](https://blog.landr.com/skeuomorphism-plugins/)

### 10. Hybrid Glass Button (Premium CTA)
Solid orange fill with gradient, inner glass highlight, soft glow halo. For primary actions (Process, Export, Start).
- **Tech:** `linear-gradient`, `::after` pseudo-element highlight, box-shadow glow
- **CSS:** `background: linear-gradient(180deg, #ffa033 0%, #FF8800 50%, #e67700 100%)`

### Implementation Priority for StemScribe

| Control | Best Approach | Priority |
|---------|--------------|----------|
| Stomp switch, toggle, pad | Pure CSS | High (mixer controls) |
| Rotary knob | SVG + JS / webaudio-controls | Medium |
| Fader/slider | Styled `<input type="range">` | High (volume/pan) |
| Transport play/pause | SVG + CSS clip-path | High |
| VU meter (real-time) | Canvas | Medium |
| LED indicators | Pure CSS box-shadow | High |

### Key Libraries
- [webaudio-controls](https://github.com/g200kg/webaudio-controls) — Web Component knobs, sliders, switches
- [AudioKnobs](https://github.com/Megaemce/AudioKnobs) — SVG knobs with Moog-style shadows
- [neumorphism.io](https://neumorphism.io/) — CSS shadow generator

---

## Part 3: Font Recommendations

### Current Font: Righteous (Google Fonts, free)
Retro, rounded, bold — vintage signage feel. Already in use.

### Recommended Pairings

#### Pairing 1: "The Gear Head" (TOP PICK)
| Role | Font | Link |
|------|------|------|
| Header | **Instrument Serif** | [Google Fonts](https://fonts.google.com/specimen/Instrument+Serif) |
| Body | **Instrument Sans** | [Google Fonts](https://fonts.google.com/specimen/Instrument+Sans) |
| Mono | **JetBrains Mono** | [Google Fonts](https://fonts.google.com/specimen/JetBrains+Mono) |
*Cohesive family system. Serif has old-style warmth that feels like vintage gear. All free.*

#### Pairing 2: "The Concert Poster"
| Role | Font | Link |
|------|------|------|
| Header | **Bebas Neue** | [Google Fonts](https://fonts.google.com/specimen/Bebas+Neue) |
| Body | **Inter** | [Google Fonts](https://fonts.google.com/specimen/Inter) |
| Mono | **IBM Plex Mono** | [Google Fonts](https://fonts.google.com/specimen/IBM+Plex+Mono) |
*Maximum contrast. Bebas screams concert poster, Inter stays invisible.*

#### Pairing 3: "The Amp Stack" (Keeps Righteous)
| Role | Font | Link |
|------|------|------|
| Header | **Righteous** (keep) | [Google Fonts](https://fonts.google.com/specimen/Righteous) |
| Body | **DM Sans** | [Google Fonts](https://fonts.google.com/specimen/DM+Sans) |
| Mono | **Space Mono** | [Google Fonts](https://fonts.google.com/specimen/Space+Mono) |
*Preserves existing brand. DM Sans pairs with Righteous's rounded geometry.*

#### Pairing 4: "The Vintage Marshall"
| Role | Font | Link |
|------|------|------|
| Header | **Jost** (free Futura alt — Marshall DNA) | [Google Fonts](https://fonts.google.com/specimen/Jost) |
| Body | **Inter** | [Google Fonts](https://fonts.google.com/specimen/Inter) |
| Mono | **Fira Code** | [GitHub](https://github.com/tonsky/FiraCode) |
*Geometric precision of Futura that Marshall amps were built on.*

#### Pairing 5: "The Premium Hybrid" (RECOMMENDED)
| Role | Font | Link |
|------|------|------|
| Primary Header | **Righteous** (brand continuity) | [Google Fonts](https://fonts.google.com/specimen/Righteous) |
| Secondary Header | **Instrument Serif** (elegant contrast) | [Google Fonts](https://fonts.google.com/specimen/Instrument+Serif) |
| Body | **Inter** | [Google Fonts](https://fonts.google.com/specimen/Inter) |
| Mono | **JetBrains Mono** | [Google Fonts](https://fonts.google.com/specimen/JetBrains+Mono) |
*Best of both worlds. Righteous for brand, Instrument Serif for editorial elegance.*

### Monospace Fonts for Tabs/Chords (Ranked)

1. **JetBrains Mono** — Max x-height, 8 weights, best readability. [Google Fonts](https://fonts.google.com/specimen/JetBrains+Mono)
2. **Fira Code** — Programming ligatures, more character. [GitHub](https://github.com/tonsky/FiraCode)
3. **IBM Plex Mono** — Industrial, clean. [Google Fonts](https://fonts.google.com/specimen/IBM+Plex+Mono)
4. **Space Mono** — Quirky, geometric. [Google Fonts](https://fonts.google.com/specimen/Space+Mono)
5. **DM Mono** — Light, airy, pairs with DM Sans. [Google Fonts](https://fonts.google.com/specimen/DM+Mono)

### Vintage / Specialty Fonts

| Font | Vibe | Source |
|------|------|--------|
| **Eurostile** | Hiwatt/Fender amp faceplate | Paid (Adobe Fonts) |
| **Jost** | Free Futura (Marshall DNA) | [Google Fonts](https://fonts.google.com/specimen/Jost) |
| **Rock Salt** | Gritty hand-drawn grunge | [Google Fonts](https://fonts.google.com/specimen/Rock+Salt) |
| **Audiowide** | Futuristic tech-music crossover | [Google Fonts](https://fonts.google.com/specimen/Audiowide) |

### Dark Theme Typography Tips
- Use **Medium (500)** weight instead of Regular (400) for body text — dark backgrounds cause halation
- Variable fonts (Inter, JetBrains Mono) allow fine-tuning weight
- Avoid ultra-thin weights on dark backgrounds
- High-contrast serifs (Abril Fatface) look stunning as headers but not body text

### Google Fonts Import (All Free)
```
https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Instrument+Sans:wght@400;500;600;700&family=Bebas+Neue&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Righteous&family=DM+Sans:wght@400;500;700&family=Jost:wght@400;500;600;700&display=swap
```

---

## Part 4: Final Icon Inventory

All icons saved to `~/stemscribe/frontend/icons/custom/`

### Style 1: Weathered Vintage Gear (`*-vintage.png`)
Photorealistic aged instruments with warm orange side lighting on black backgrounds.

| Instrument | 16px | 32px | 48px | 128px |
|-----------|------|------|------|-------|
| Guitar (Stratocaster) | `16/guitar-vintage.png` | `32/guitar-vintage.png` | `48/guitar-vintage.png` | `128/guitar-vintage.png` |
| Bass (P-Bass) | `16/bass-vintage.png` | `32/bass-vintage.png` | `48/bass-vintage.png` | `128/bass-vintage.png` |
| Drums (Snare + Sticks) | `16/drums-vintage.png` | `32/drums-vintage.png` | `48/drums-vintage.png` | `128/drums-vintage.png` |
| Piano (Aged Keys) | `16/piano-vintage.png` | `32/piano-vintage.png` | `48/piano-vintage.png` | `128/piano-vintage.png` |
| Vocals (SM58 Mic) | `16/vocals-vintage.png` | `32/vocals-vintage.png` | `48/vocals-vintage.png` | `128/vocals-vintage.png` |
| Other (Music Note + Headphones) | `16/other-vintage.png` | `32/other-vintage.png` | `48/other-vintage.png` | `128/other-vintage.png` |

### Style 2: Aged Chrome Emblem (`*-chrome.png`)
Tarnished chrome metal emblems with amber backlighting, vintage amp badge aesthetic.

| Instrument | 16px | 32px | 48px | 128px |
|-----------|------|------|------|-------|
| Guitar | `16/guitar-chrome.png` | `32/guitar-chrome.png` | `48/guitar-chrome.png` | `128/guitar-chrome.png` |
| Bass | `16/bass-chrome.png` | `32/bass-chrome.png` | `48/bass-chrome.png` | `128/bass-chrome.png` |
| Drums | `16/drums-chrome.png` | `32/drums-chrome.png` | `48/drums-chrome.png` | `128/drums-chrome.png` |
| Piano | `16/piano-chrome.png` | `32/piano-chrome.png` | `48/piano-chrome.png` | `128/piano-chrome.png` |
| Vocals | `16/vocals-chrome.png` | `32/vocals-chrome.png` | `48/vocals-chrome.png` | `128/vocals-chrome.png` |
| Other | `16/other-chrome.png` | `32/other-chrome.png` | `48/other-chrome.png` | `128/other-chrome.png` |

### Style 3: Worn Backstage / Roadie (`*-backstage.png`)
Beat-up gear on dark road cases, stage amber lighting, gritty touring aesthetic.

| Instrument | 16px | 32px | 48px | 128px |
|-----------|------|------|------|-------|
| Guitar (Road-worn) | `16/guitar-backstage.png` | `32/guitar-backstage.png` | `48/guitar-backstage.png` | `128/guitar-backstage.png` |
| Bass (Leaning on Amp) | `16/bass-backstage.png` | `32/bass-backstage.png` | `48/bass-backstage.png` | `128/bass-backstage.png` |
| Drums (Battered Snare) | `16/drums-backstage.png` | `32/drums-backstage.png` | `48/drums-backstage.png` | `128/drums-backstage.png` |
| Piano (70s Wurlitzer) | `16/piano-backstage.png` | `32/piano-backstage.png` | `48/piano-backstage.png` | `128/piano-backstage.png` |
| Vocals (Taped SM58) | `16/vocals-backstage.png` | `32/vocals-backstage.png` | `48/vocals-backstage.png` | `128/vocals-backstage.png` |
| Other (Worn Mixer Fader) | `16/other-backstage.png` | `32/other-backstage.png` | `48/other-backstage.png` | `128/other-backstage.png` |

### Total: 72 icon files (18 icons x 4 sizes)

---

## Trend Context (2025-2026)

The design industry is moving toward **"Multi-Material & Luxe Finishes"** — layering glass, metallic, and tactile textures for premium depth (Envato 2026 trend report). Apple's Liquid Glass update normalized translucent/refractive treatments. The overall direction is away from flat design toward "refined realism."

For music apps specifically, the industry has settled on **neo-skeuomorphic** — not the full photorealistic textures of 2010-era plugins, but enough 3D depth and hardware metaphor to feel tactile and familiar to musicians. Neural DSP and Native Instruments (Kontakt 7) lead this hybrid approach.

StemScribe's aged/weathered direction is a strong differentiator — most music apps go pristine/modern. The vintage gear aesthetic speaks directly to working musicians who value instruments with history.
