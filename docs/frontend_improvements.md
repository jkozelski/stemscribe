# StemScribe Frontend Improvements -- Research & Proposals

## Research Date: 2026-02-27

---

## Part 1: Competitive Research

### 1. Songsterr -- Interactive Guitar Tab Player
- **Tab Display:** Multi-instrument tablature rendered inline with standard notation; active note highlighted with a colored cursor that scrolls horizontally in sync with playback.
- **Playback UX:** Speed control (slow down to 15%), loop selection by dragging across measures, solo/mute per instrument track. Backing tracks with the user's part muted.
- **Visual Features:** Clean, minimal dark-on-light layout. Active beat highlighted with a vertical cursor line. Smooth auto-scroll during playback.
- **Mobile:** Responsive tab rendering that adapts to screen width; swipe to navigate between sections.
- **Color Scheme:** White background, blue accent for active notes, gray for inactive instruments.

### 2. Ultimate Guitar -- Tab Community & Viewer
- **Tab Display:** Text-based chord/tab format with chord diagrams on tap. "Tab Pro" mode adds synchronized playback with interactive notation.
- **Playback UX:** Auto-scroll at adjustable speed, pitch-preserving speed change, loop sections, multi-track instrument isolation.
- **Visual Features:** Clean reading experience with customizable font size/style. "Fit to Screen" mode. Left-handed mode.
- **Mobile:** Tab playlists and collections. Large tap targets for chord diagrams. Offline access for saved tabs.
- **Color Scheme:** Dark theme with orange/amber accents. High-contrast text.

### 3. Moises.ai -- AI Stem Separation
- **Stem Display:** DAW-like interface with stacked waveforms per stem. Each stem has its own color-coded track.
- **Playback UX:** Per-stem volume faders, mute/solo buttons, pan controls. Real-time stem manipulation during playback.
- **Visual Features:** Color-coded waveforms (vocals = purple, drums = orange, bass = blue, etc.). Clean, uncluttered GUI. Drag-and-drop upload.
- **Advanced Features:** AI chord detection with editable timeline. Key detection. Pitch/tempo adjustment. Click track generation. Lyric transcription.
- **Mobile:** Multi-stem separation from single upload. Simplified mobile editing workflow.
- **Color Scheme:** Dark theme with vibrant stem colors. Professional DAW aesthetic.

### 4. Soundslice -- Interactive Sheet Music
- **Notation Display:** SVG-rendered standard notation and tablature. Notes light up during playback. Click any note to jump to that position.
- **Playback UX:** Pitch-preserving speed control. Drag across notes to create loops (snaps to nearest note/rest/barline). Hide/mute individual parts in ensembles.
- **Visual Features:** Hover states that change note color and cursor to indicate editability. Chord chart view. Visual fingering diagrams (guitar, trombone, trumpet).
- **Editor UI:** Top toolbar for commands. Bottom panel for general functionality. Keyboard shortcut support. Zoom-friendly design for accessibility.
- **Color Scheme:** Light theme with blue active-note highlights. Clean, academic aesthetic.

### 5. Guitar Pro -- Desktop Tab Software
- **Notation Display:** Full standard notation + tablature rendering. Multi-track score view. Slash notation for drums.
- **Playback UX:** Built-in MIDI synthesizer with realistic instrument sounds. Track-level volume/pan/mute.
- **Visual Features:** Improved contrast and colors in v8. Export to SVG and transparent PNG. Cautionary accidentals. Numbered notation (jianpu) support.
- **Mobile (iOS):** Dark mode that dims background and highlights the score. Reduces eye fatigue. Commands and menus recede into background.
- **Color Scheme:** Desktop is light theme only. Mobile supports dark/light toggle.

### 6. BandLab -- Online Music Creation
- **Interface Design:** Drag-and-drop multi-track editor. "Track type" and "tools" headers at top. Segmented creation workflow.
- **Features:** Multi-track Mix Editor. Tap tempo. Magnetic timeline. Lyric editor. Loop packs sorted by genre/instrument.
- **Collaboration:** Real-time collaborative workspaces. In-app chat. Group projects.
- **Mobile:** Full creation suite on mobile. Voice recording with AutoPitch. Drum machine. Sampler.
- **Color Scheme:** Dark theme with colorful track/instrument indicators.

### 7. Yousician -- Music Learning App
- **Learning UI:** "Guitar Hero" style -- notes scroll vertically toward a virtual fretboard. Bouncing ball guides timing.
- **Gamification:** Points for accuracy. Streak counters. Leaderboards. Weekly challenges. Progress tracking.
- **Visual Features:** Vibrant, inviting color scheme. Clean graphic elements. Visual cues guiding the user through tasks.
- **Feedback:** Real-time audio analysis and accuracy scoring. Immediate visual feedback on correct/incorrect notes.
- **Color Scheme:** Dark backgrounds with bright, saturated accent colors. Game-like energy.

### 8. Spotify / Apple Music -- Modern Music Players
- **Player UI:** Full-screen now-playing view with album art. Progress bar/waveform scrubber. Minimal controls.
- **Visual Features:** Apple Music -- "Liquid Glass" aesthetic, animated album art on lock screen. Spotify -- color extraction from album art for dynamic backgrounds.
- **Animations:** Smooth page transitions. Spring-based micro-animations. Parallax scrolling. Blur/glass morphism.
- **Mobile Design:** Bottom sheet mini-player. Swipe gestures for skip/dismiss. Large touch targets. Edge-to-edge content.
- **Color Scheme:** Apple -- light/dark adaptive with glass morphism. Spotify -- dark theme (#121212) with green (#1DB954) accent.

### 9. LALAL.AI -- Stem Separation
- **Interface:** Simple drag-and-drop upload. One separation type at a time (vocal/instrumental, drums/drumless, bass/bassless).
- **Playback:** Preview separated stems before download. Quality rating (thumbs up/down) inline.
- **Visual Features:** Modernized UI with dark/light mode toggle. Processing progress indicators. Stem-splitting history.
- **Output Options:** MP3, WAV, FLAC format selection. De-echo and neural network selection settings.
- **Color Scheme:** Clean dark theme. Professional, minimal aesthetic.

### 10. Additional Notable Tools
- **alphaTab:** Open-source JS library for rendering Guitar Pro files in browser with built-in MIDI synthesizer, playback cursor, and interactive selection. Supports GP3-7, MusicXML, and custom markup.
- **Wavesurfer.js:** Open-source audio waveform library. Clickable regions, fade effects, recording, minimap, timeline, spectrogram.
- **Waveform Playlist:** Open-source multi-track web audio editor with canvas waveforms, drag-and-drop editing, Tone.js effects, WAV export.
- **Midiano:** Browser-based MIDI visualization with falling-note piano roll and GPU-accelerated particle effects.
- **WaveRoll:** JS library for comparative MIDI piano-roll visualization with synchronized playback.

---

## Part 2: Key Design Patterns Identified

### Pattern A: Stem Mixing Console
Every stem separation tool (Moises, LALAL.AI) uses the same proven pattern:
- Color-coded waveform per stem (stacked vertically)
- Per-stem volume slider + mute (M) + solo (S) buttons
- Master transport controls (play/pause, seek, loop)
- Visual feedback on which stems are active

### Pattern B: Interactive Tab/Notation Display
Best-in-class tab viewers (Songsterr, Soundslice, alphaTab) share:
- Synchronized playback cursor highlighting the active beat
- Click-to-jump on any note
- Speed control (pitch-preserving)
- Loop selection by drag or measure selection
- Instrument track switching

### Pattern C: Modern Music Player Chrome
Spotify and Apple Music established:
- Dark theme as default
- Album-art-driven color extraction for dynamic theming
- Bottom mini-player with swipe-to-expand
- Smooth spring animations and transitions
- Glass morphism / blur effects

### Pattern D: Gamification & Engagement
Yousician proves engagement through:
- Real-time visual feedback (correct/incorrect)
- Progress tracking and streaks
- Competitive elements (leaderboards)
- Animated, vibrant visual language

---

## Part 3: Proposed Improvements for StemScribe

### Priority 1: Critical / High Impact

#### 1.1 Stem Mixing Console (Complexity: MEDIUM)
**What:** Replace basic audio playback with a full stem mixing interface.
- Color-coded waveform visualization for each stem (vocals, guitar, bass, drums, piano) using Wavesurfer.js
- Per-stem vertical volume sliders with mute/solo toggle buttons
- Color assignments: vocals = purple (#A855F7), guitar = amber (#F59E0B), bass = blue (#3B82F6), drums = orange (#F97316), piano = emerald (#10B981)
- Master playback transport: play/pause, seek bar, current time / total time
- Loop selection: click-and-drag on waveform to create a loop region

**Why:** This is the core UX of any stem separation tool. Moises, LALAL.AI, and every competitor does this. Without it, StemScribe feels incomplete.

**Libraries:** Wavesurfer.js (waveform rendering), Web Audio API (stem mixing and routing)

#### 1.2 Interactive Tab Viewer (Complexity: HIGH)
**What:** Render Guitar Pro (.gp) and MIDI transcription results as interactive, playable tablature.
- Use alphaTab.js to render tablature + standard notation from Guitar Pro / MusicXML output
- Synchronized playback cursor that highlights the active beat
- Click any note to jump playback to that position
- Speed control slider (50% to 150%, pitch-preserved)
- Instrument track selector (switch between guitar, bass, drums, piano views)

**Why:** This is the "wow factor" that differentiates StemScribe from plain stem separators. Seeing your transcribed music as playable, interactive tabs is the core value proposition.

**Libraries:** alphaTab.js (notation rendering + synthesis), alphaSynth (built-in MIDI playback)

#### 1.3 Dark Mode (Complexity: LOW)
**What:** Implement a dark theme as the default, with a light mode toggle.
- Dark background (#0F172A or #1E1B2E for a music-app feel)
- High-contrast text (#F8FAFC)
- Vibrant accent colors for stems and interactive elements
- Tailwind CSS `dark:` variant classes
- System preference detection with manual override toggle
- Persist preference in localStorage

**Why:** Every single music app researched uses a dark theme. Musicians practice at night, in studios, and in dim environments. Dark mode is table stakes.

---

### Priority 2: High Value / Medium Effort

#### 2.1 Processing Progress & Loading States (Complexity: LOW)
**What:** Replace generic loading spinners with music-themed progress visualization.
- Multi-stage progress indicator: "Downloading audio" -> "Separating stems" -> "Transcribing to MIDI" -> "Generating tabs" -> "Done"
- Animated waveform or equalizer bars as loading animation (CSS-only)
- Estimated time remaining based on historical processing times
- Skeleton screens for tab viewer and stem mixer while loading

**Why:** Stem separation and transcription take time. Users need clear feedback on what is happening and how long it will take. This reduces perceived wait time and abandonment.

#### 2.2 Mobile-Optimized Stem Player (Complexity: MEDIUM)
**What:** Responsive stem mixing that works well on phones and tablets.
- Horizontal waveform view with touch-friendly controls
- Swipe gestures: swipe up on a stem to solo, swipe down to mute
- Bottom sheet player (a la Spotify) that expands to full-screen mixing console
- Simplified controls at small breakpoints: toggle buttons instead of sliders
- Touch-friendly loop selection with haptic feedback

**Why:** Many musicians practice with their phone propped up. A great mobile experience is a differentiator since most transcription tools have poor mobile UX.

#### 2.3 Playback Speed & Loop Controls (Complexity: LOW)
**What:** Practice-oriented playback tools.
- Speed slider: 25% to 200% with pitch preservation (Web Audio API playbackRate)
- Preset buttons: 50%, 75%, 100%
- A-B loop: tap "A" to set loop start, "B" to set loop end, tap again to clear
- Visual loop region highlighted on waveform and tab display
- Keyboard shortcuts: space = play/pause, [ = set A, ] = set B, +/- = speed

**Why:** This is the number one practice feature in Songsterr, Soundslice, and Ultimate Guitar. Musicians need to slow down and repeat difficult passages.

---

### Priority 3: Polish & Delight

#### 3.1 Dynamic Color Theming (Complexity: LOW)
**What:** Extract dominant colors from YouTube video thumbnails to create per-song color accents.
- Use a canvas-based color extraction algorithm on the YouTube thumbnail
- Apply extracted colors as CSS custom properties for gradients, glows, and accents
- Subtle gradient backgrounds behind the player area
- Animated color transitions when switching between songs

**Why:** Spotify does this brilliantly -- each song feels unique. It adds visual richness with minimal performance cost.

#### 3.2 Animated Waveform Visualizer (Complexity: MEDIUM)
**What:** Real-time audio visualization during playback.
- Frequency bar visualizer (Web Audio API AnalyserNode) in the header/background
- Per-stem mini waveforms that pulse with audio amplitude
- Subtle glow effects on active stems
- Canvas or WebGL-based rendering for smooth 60fps animation

**Why:** Visual feedback during playback makes the app feel alive. It is a "wow factor" that users remember and share.

#### 3.3 Piano Roll MIDI View (Complexity: MEDIUM)
**What:** Alternative visualization for MIDI transcription results.
- Horizontal piano roll with falling/scrolling notes (like Midiano)
- Color-coded by instrument/stem
- Synchronized with audio playback
- Click to select notes, hover to see note name and velocity
- Toggle between tab view and piano roll view

**Why:** Piano roll is the standard MIDI visualization. Musicians familiar with DAWs expect it. It also works for instruments where tablature does not apply (piano, drums).

#### 3.4 Smooth Page Transitions & Micro-Animations (Complexity: LOW)
**What:** Polish the overall feel of the app with purposeful animations.
- Page transitions using Astro View Transitions API
- Spring-based animations for expanding/collapsing panels (CSS `transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1)`)
- Staggered fade-in for stem tracks loading one by one
- Button press feedback (scale down on click, spring back)
- Skeleton loading states with subtle shimmer animation

**Why:** Micro-interactions communicate quality. The difference between a "tool" and a "product" is often in the animation polish.

#### 3.5 Keyboard Shortcuts & Accessibility (Complexity: LOW)
**What:** Power-user shortcuts and screen reader support.
- Space: play/pause
- Left/Right arrows: seek 5 seconds
- Up/Down arrows: volume
- 1-5: toggle stem mute (1=vocals, 2=guitar, 3=bass, 4=drums, 5=piano)
- S: solo focused stem
- [ / ]: set loop points
- +/-: adjust speed
- ARIA labels on all controls
- Focus indicators for keyboard navigation
- Toast/tooltip showing shortcut hints

**Why:** Power users (musicians) will use the app repeatedly. Keyboard shortcuts dramatically improve the practice workflow.

---

### Priority 4: Future "Wow Factor" Features

#### 4.1 Collaborative Session Sharing (Complexity: HIGH)
**What:** Share a StemScribe session via URL with custom stem mix settings.
- Encode stem volumes, loop points, speed, and active view in URL parameters
- "Share mix" button generates a shareable link
- Recipients see the same song with the sender's mix settings applied
- Optional: real-time collaborative mixing (WebSocket-based)

**Why:** BandLab proves that collaboration drives engagement. A band could share a StemScribe link and discuss parts.

#### 4.2 Fretboard / Keyboard Visualization (Complexity: HIGH)
**What:** Visual instrument display synced with tab playback.
- Animated guitar fretboard showing finger positions in real-time
- Piano keyboard showing active keys for piano stems
- Drum pad layout showing active hits for drum stems
- Follows playback cursor in the tab viewer

**Why:** Yousician and Soundslice both use this for learning. It bridges the gap between reading notation and physical instrument playing.

#### 4.3 Export & Download Options (Complexity: MEDIUM)
**What:** Let users download their separated stems and transcriptions.
- Download individual stems as WAV/MP3
- Download MIDI file
- Download Guitar Pro (.gp) file
- Download PDF of rendered tablature
- Export to MusicXML for use in other notation software

**Why:** Users want to take their transcriptions into their DAW, print tabs for practice, or share with bandmates.

#### 4.4 Song Section Detection (Complexity: HIGH)
**What:** Automatically detect and label song sections (intro, verse, chorus, bridge, solo, outro).
- Visual markers on the waveform timeline
- Section navigation buttons for quick jumping
- Section-based looping ("loop the chorus")
- Color-coded section regions

**Why:** Moises added this recently and it is a major UX improvement. Musicians think in sections, not timestamps.

---

## Part 4: Recommended Technology Stack

| Component | Library | Purpose |
|---|---|---|
| Waveform rendering | Wavesurfer.js v7 | Per-stem waveform display with regions/markers |
| Tab/notation rendering | alphaTab.js | Guitar Pro / MusicXML rendering with playback |
| Audio routing | Web Audio API | Stem mixing, volume control, speed change |
| Piano roll | Custom Canvas or WaveRoll | MIDI visualization |
| Animations | CSS transitions + Astro View Transitions | Page and micro-animations |
| Color extraction | Canvas getImageData + quantization | Dynamic theming from thumbnails |
| Icons | Lucide or Heroicons | Consistent icon system |

---

## Part 5: Implementation Roadmap

### Phase 1 -- Foundation (2-3 weeks)
- [ ] Dark mode theme system
- [ ] Stem mixing console with Wavesurfer.js
- [ ] Basic playback transport controls
- [ ] Loading/progress states

### Phase 2 -- Core Features (3-4 weeks)
- [ ] alphaTab.js tab viewer integration
- [ ] Speed control and A-B looping
- [ ] Mobile-responsive stem player
- [ ] Keyboard shortcuts

### Phase 3 -- Polish (2-3 weeks)
- [ ] Dynamic color theming
- [ ] Page transitions and micro-animations
- [ ] Piano roll MIDI view
- [ ] Accessibility audit

### Phase 4 -- Advanced (4+ weeks)
- [ ] Session sharing
- [ ] Fretboard/keyboard visualization
- [ ] Export/download options
- [ ] Song section detection

---

## Summary: Top 5 Impact Features

1. **Stem Mixing Console** -- The fundamental UX every competitor has. Use Wavesurfer.js with color-coded per-stem waveforms, volume/mute/solo controls.
2. **Interactive Tab Viewer** -- The "killer feature." alphaTab.js renders Guitar Pro files with synchronized playback cursor, click-to-jump, and speed control.
3. **Dark Mode** -- Every music app uses dark theme. Implement with Tailwind dark: variants. Ship it as the default.
4. **Practice Tools (Speed + Looping)** -- Musicians need to slow down and repeat. A-B looping and pitch-preserved speed control are essential.
5. **Animated Waveform Visualizer** -- The "wow factor." Real-time frequency visualization makes the app feel alive and professional.
