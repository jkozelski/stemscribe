# StemScriber Mobile Layout Audit

**Date:** 2026-03-18
**Tested by:** Automated Playwright Agent
**Server:** http://localhost:5555

## Test Matrix

| Page | 375px (iPhone SE) | 390px (iPhone 14) | 414px (iPhone Plus) | 768px (iPad) |
|------|:-:|:-:|:-:|:-:|
| `/` (index) | Tested | Tested | Tested | Tested |
| `/landing.html` | Tested | Tested | Tested | Tested |
| `/practice.html` | Tested | Tested | Tested | Tested |

Screenshots saved to: `~/stemscribe/docs/screenshots/`

---

## 1. Index Page (`/`)

### Horizontal Scroll
| Width | Result |
|-------|--------|
| 375px | PASS - scrollWidth matches viewport (375) |
| 390px | PASS - scrollWidth matches viewport (390) |
| 414px | PASS - scrollWidth matches viewport (414) |
| 768px | PASS - scrollWidth matches viewport (768) |

### Overflowing Elements
| Issue | Severity | All Widths |
|-------|----------|------------|
| `div.orb` elements overflow right (decorative bg orbs, 400-600px wide) | LOW | These are background decoration elements with `overflow: hidden` on parent - no visible scroll caused |

### Buttons Too Small to Tap (< 44px)
| Element | Size | Severity | Notes |
|---------|------|----------|-------|
| Hamburger menu `"☰"` | 21x29 | **HIGH** | Way too small for mobile tap target |
| `"StemScriber"` nav link | 161x36 | MEDIUM | Width fine, height 36px is borderline |
| Footer links (Contact, Facebook, Instagram, etc.) | ~53-108 x 18 | **HIGH** | All footer links only 18px tall |
| `"Request access"` mailto link | 99x16 | **HIGH** | Only 16px tall |
| `"DMCA"` footer link | 38x18 | **HIGH** | Both too narrow and too short |
| `"Powered by Tidepool Artists"` | 174x16 | MEDIUM | Only 16px tall |

### Unreadable Text (< 12px)
| Element | Font Size | Severity |
|---------|-----------|----------|
| Feature badge spans ("Free", "Premium", "Pro") | 11.2px | LOW | Barely below threshold, still readable |

### Visual Assessment
- Layout stacks well at all phone widths
- Pricing cards stack vertically on phones, readable
- Hero section text is well-sized
- "How It Works" steps stack nicely
- CTA buttons ("Try It Free", "See How It Works") are properly sized
- At 768px (iPad), the layout still uses single-column mobile layout (hamburger menu still showing instead of desktop nav)

---

## 2. Landing Page (`/landing.html`)

### Horizontal Scroll
| Width | Result |
|-------|--------|
| 375px | PASS |
| 390px | PASS |
| 414px | PASS |
| 768px | PASS |

### Overflowing Elements
| Issue | Severity | All Widths |
|-------|----------|------------|
| `div.orb` decorative elements | LOW | Same as index - no user-visible overflow |

### Buttons Too Small to Tap (< 44px)
| Element | Size | Severity | Notes |
|---------|------|----------|-------|
| Hamburger menu `"☰"` | 21x29 | **HIGH** | Same as index |
| `"StemScriber"` nav link | 161x36 | MEDIUM | Height borderline |
| All footer links | ~9-108 x 18 | **HIGH** | 18px tall across the board |
| `"X"` (Twitter/X link) | 9x18 | **CRITICAL** | Only 9px wide - virtually untappable |
| `"TikTok"` | 41x18 | **HIGH** | Below 44px on both dimensions |
| `"DMCA"` | 38x18 | **HIGH** | Below 44px on both dimensions |
| `"Request access"` | 99x16 | **HIGH** | Only 16px tall |

### Unreadable Text (< 12px)
| Element | Font Size | Severity |
|---------|-----------|----------|
| Feature badge spans ("Free", "Premium", "Pro") | 11.2px | LOW |

### Visual Assessment
- Very similar layout to index page, stacks well
- Landing page uses SVG icons for features instead of emoji (index uses emoji)
- Pricing structure differs from index ($0 / $10 / $20 vs $0 / $4.99 / $14.99) - this may be intentional for A/B testing
- At 768px, still single-column with hamburger menu

---

## 3. Practice Page (`/practice.html`)

### Horizontal Scroll
| Width | Result |
|-------|--------|
| 375px | PASS - scrollWidth matches viewport |
| 390px | PASS |
| 414px | PASS |
| 768px | PASS |

### Overflowing Elements
| Issue | Severity | All Widths |
|-------|----------|------------|
| `button.stem-tab` elements overflow far right (679-1107px) | **HIGH** | Instrument tabs at bottom extend way past viewport. These are the Guitar/Bass/Piano/Drums/Vocals tab buttons. They appear to be in a horizontally scrolling container but individual buttons extend to 1107px right. |
| SVG chord diagram elements (circle, rect, line) | MEDIUM | Chord diagram SVGs overflow slightly at smaller widths |
| Fretboard/chord diagram `div` elements | MEDIUM | Some diagram containers extend 30-60px past viewport edge |

### Buttons Too Small to Tap (< 44px)
| Element | Size | Severity | Notes |
|---------|------|----------|-------|
| Speed preset buttons ("50%", "75%", etc.) | 67-73 x 40 | MEDIUM | Width fine, height 40px is close but below 44 |
| "Tab" view button | 53x36 | **HIGH** | 36px tall |
| "Chords" view button | 73x36 | **HIGH** | 36px tall |
| Transpose "-" button | 28x28 | **HIGH** | Way too small |
| Transpose "+" button | 28x28 | **HIGH** | Way too small |
| "Use #" button | 58x36 | **HIGH** | 36px tall |
| "Keys" button | 56x36 | **HIGH** | 36px tall |
| Close "x" button | 13x23 | **CRITICAL** | Extremely small, nearly impossible to tap |
| M (Mute) buttons (768px only) | 36x36 | MEDIUM | Below 44px threshold |
| S (Solo) buttons (768px only) | 36x36 | MEDIUM | Below 44px threshold |

### Unreadable Text (< 12px)
| Element | Font Size | Severity |
|---------|-----------|----------|
| "Back" link (375px) | 10.4px | **HIGH** | Below readable threshold on smallest phones |
| "Speed" label | 11.2px | MEDIUM | Borderline |
| "100%" speed value | 11.2px | MEDIUM | Borderline |
| "Stem Mixer" label | 11.2px | MEDIUM | Borderline |
| "View" label | 11.2px | MEDIUM | Borderline |
| "Transpose" label | 11.2px | MEDIUM | Borderline |
| Chord diagram "x" text | 9px | LOW | Part of chord SVG, acceptable |
| Chord diagram "4" (fret number) | 8px | LOW | Part of chord SVG, acceptable |
| Time display "0:03" | 11.2px | MEDIUM | Borderline |
| M/S button text (768px) | 11.2px | MEDIUM | Borderline |

### Console Errors (all widths)
- AlphaTab ScoreLoader import failure (RangeError: Invalid typed array)
- Failed to load chord chart API endpoint (404)
- AlphaTab rendering container warning

### Visual Assessment
- Practice page is significantly more complex than marketing pages
- Stem mixer renders well at all widths - sliders and M/S buttons are accessible
- Chord view loads and displays correctly with chord diagrams
- The instrument tab bar at the bottom (Guitar/Bass/Piano/Drums/Vocals) overflows but appears to be intentionally horizontally scrollable
- At 768px, the layout opens up nicely - more breathing room for controls
- Song title and artist display well at all sizes
- Playback controls (play/pause/skip) are well-sized

---

## Summary of Issues by Priority

### CRITICAL (fix before launch)
1. **"X" social link is 9x18px** on landing.html footer - virtually untappable
2. **Close "x" button is 13x23px** on practice.html - nearly impossible to tap on mobile
3. **Transpose +/- buttons are 28x28px** on practice.html - well below tap target

### HIGH (should fix)
4. **Hamburger menu button is 21x29px** - too small for reliable mobile tapping (all pages)
5. **All footer links are only 18px tall** - need more padding/line-height (both index and landing)
6. **Practice page View/Tab/Chords/Keys buttons are 36px tall** - need padding to reach 44px
7. **"Back" link text is 10.4px** at 375px on practice page - bump to 12px minimum
8. **stem-tab buttons overflow** past viewport on practice page (up to 1107px right)

### MEDIUM (nice to have)
9. Feature badge text ("Free", "Premium", "Pro") at 11.2px is borderline
10. Speed/mixer labels at 11.2px are borderline on practice page
11. M/S buttons at 36x36 on 768px could be larger
12. At 768px, pages still show mobile hamburger menu - could switch to desktop nav

### LOW (cosmetic)
13. Decorative `.orb` background elements extend past viewport (no visible effect due to overflow:hidden)
14. Chord diagram SVG text at 8-9px is acceptable for diagram context

---

## Screenshots Index

| File | Page | Width |
|------|------|-------|
| `index-375px.png` | Index | 375px |
| `index-390px.png` | Index | 390px |
| `index-414px.png` | Index | 414px |
| `index-768px.png` | Index | 768px |
| `landing-375px.png` | Landing | 375px |
| `landing-390px.png` | Landing | 390px |
| `landing-414px.png` | Landing | 414px |
| `landing-768px.png` | Landing | 768px |
| `practice-375px.png` | Practice | 375px |
| `practice-390px.png` | Practice | 390px |
| `practice-414px.png` | Practice | 414px |
| `practice-768px.png` | Practice | 768px |

---

## Recommended CSS Fixes

### 1. Footer links tap targets
```css
footer a {
  display: inline-block;
  padding: 12px 8px;
  min-height: 44px;
  min-width: 44px;
  line-height: 20px;
}
```

### 2. Hamburger menu button
```css
.nav-toggle {
  min-width: 44px;
  min-height: 44px;
  display: flex;
  align-items: center;
  justify-content: center;
}
```

### 3. Practice page small buttons
```css
.practice-toolbar button {
  min-height: 44px;
  min-width: 44px;
  padding: 8px 12px;
}

.transpose-btn {
  min-width: 44px;
  min-height: 44px;
}
```

### 4. "Back" link minimum font size
```css
@media (max-width: 390px) {
  .back-link {
    font-size: 12px;
  }
}
```
