# Independent visual audit — questionnaire for stemscriber.com

**Purpose:** Adversarial visual / design / brand audit of the StemScriber website. The benchmark is **"does this look like a $1M website built by a team with real craft"** — NOT "is it OK." Most consumer-product websites with serious investment behind them clear that bar; most AI-template-built sites in 2025-2026 don't. The goal is to identify exactly what's keeping us in the second category and how to move into the first.

**How to run:**
1. Open ChatGPT-4o (vision required) or Claude with vision enabled in a fresh chat — independent context, not the model that did the original work.
2. Provide the URLs (or screenshots) below. The model should navigate / view the site directly, mobile and desktop.
3. Paste the prompt below + this questionnaire.
4. Apply findings to a v2 visual refresh sprint.

**URLs to audit:**
- `https://stemscriber.com` — public landing page (primary surface)
- `https://stemscriber.com/app` — the actual upload UI (secondary surface, only post-signup-or-direct-link)
- `https://stemscriber.com/youtube-fallback.html` — secondary help page (newly added today)

**Reference benchmarks** (sites that *do* clear the $1M-craft bar — for comparison, not imitation):
- `https://moises.ai` — direct music-tech competitor; full-bleed video hero, real musicians, editorial layout
- `https://klang.io` — direct competitor; product-screenshots-in-device-frames hero, layered depth
- `https://linear.app` — gold-standard SaaS craft, custom illustration, motion, density
- `https://resend.com` — clean editorial typography, single-column with intent
- `https://plain.com` — voice-laden copy, deliberate-craft vibes

---

## Prompt to use

> I'm submitting a consumer music-tech web app called StemScriber for a brutal visual audit. The product is genuinely good — the engineering and ML are real. The visual presentation is not yet at the level the product deserves. I need a sharp outside eye on what's keeping it from feeling like a "$1M website built by a team with real craft" and what specific changes would close that gap.
>
> Please review the URLs and reference benchmarks attached, then answer the questions below numbered and directly. Be honest about what looks AI-template-stock, what looks original, and what feels deliberately crafted. Where you find issues, be specific (which section, which element, which CSS pattern). Where competitors do something better, name them. The founder is the only developer — recommendations need to be implementable by one person in 1-2 weeks.

---

## Questions to address

Answer each numbered item directly. Don't reorder, don't merge.

### First-impression and overall craft level

1. **Five-second test.** What's your honest first impression of `stemscriber.com` as a visitor with no context? Be brutal: does it read as a real product, an indie hobby project, a polished SaaS, an AI-generated template, or something in between?

2. **Craft level.** On a 1-10 scale where 1 is "Wix template" and 10 is "Linear/Stripe/Apple," where does StemScriber land currently? Where do the reference benchmarks land? What is the gap made of?

3. **AI-template tells.** What specific elements scream "this was made fast with AI tools in 2025"? Examples might include: gradient orbs in the background, centered-stack layouts, identical-width feature cards in 3-column grids, default Inter body font, etc. Be specific to what you see.

### Hero / above-the-fold

4. **Hero comparison.** Compare StemScriber's hero to Moises's, Klangio's, and Linear's heroes. What do they have that StemScriber doesn't? What does StemScriber have that they don't (legitimate strengths to keep)?

5. **The waveform-stack visual.** StemScriber's hero contains an animated waveform-stack visual showing a full mix splitting into 6 stems. Is this an asset or a liability? Does it earn its place in the hero, or does it feel like a placeholder that should be replaced with a real photo / product screenshot / video?

6. **Real photography vs. abstract.** Moises uses a full-bleed video of a musician playing guitar. Klangio uses a real product screenshot in a tilted MacBook frame. StemScriber currently uses no photography at all — only abstract gradients and waveforms. Is this the single highest-leverage gap, or is something else higher priority?

### Layout, rhythm, and information design

7. **Section rhythm.** The landing page has approximately 5-6 sections in this pattern: tiny uppercase tag → big H2 → small subhead → grid-of-cards. Do they feel like distinct moments with intent, or templated repetition? Where does the rhythm break or feel mechanical?

8. **Layout variety.** Most sections are centered text + a centered grid. Is this a problem? Should some sections be asymmetric (text left, image right), full-bleed, or break the container width? Which specific sections would benefit?

9. **Density and depth.** Do the sections feel sparse and floaty, or anchored and confident? What's the visible page weight per scroll-screen? How does it compare to the reference benchmarks?

### Typography

10. **Typography stack.** Currently uses Righteous (chunky display H1/H2), Space Grotesk (UI), Outfit (body), Fraunces italic (one accent line). Is this stack working? Are there too many faces? Is any face out of brand register? Specifically critique the H1 ("TEAR THE SOUND APART") and the italic serif voice line under it ("We don't guess. We listen.").

11. **Headline scale.** Is the H1 the right size? Do H2s have enough hierarchy below it? Does any section's title compete with the H1 inappropriately?

### Color, light, and motion

12. **Color palette.** Coral/orange accent on near-black background. Is this distinctive or generic-dark-mode? What would deepen it? Does any section misuse the accent color?

13. **Background gradients / orbs.** Decorative coral and pink "orb" blobs are blurred behind sections. Are these helping or hurting? Are they an AI-template tell or a deliberate craft choice?

14. **Motion.** What animates on the page currently? What should animate that doesn't? Where would motion add craft signal vs. where would it feel gimmicky?

### Iconography and imagery

15. **Custom icons.** The Features section uses 6 custom dark/coral 3D-rendered icons (mixer, chord, practice clock, play button, guitar, download arrow) instead of emoji or stock icon-fonts. Are these working as a coherent set? Do they feel custom or do they read as stock-photo-y?

16. **Imagery gap.** What's the highest-impact imagery to add? Specifically rank: (a) hero photo of a musician learning, (b) product screenshots of the practice page in real use, (c) testimonial photos with named people, (d) custom illustration of the audio-to-chord-chart concept, (e) a hero video of the product in action.

### Specific sections critique

17. **"Three Steps. Any Song." section.** 3 numbered cards with brief copy. Does this section earn its space, or could it be cut/redesigned? Compare to how Linear or Resend would handle the equivalent "how it works."

18. **"Stop Paying for Two Tools" comparison section.** Two pricing cards side by side ($15.99 vs $10) with checkmark/cross lists. Strong selling point — but does the visual treatment do it justice?

19. **"Pick Your Plan" pricing section.** Standard 3-column SaaS pricing widget (Free / Pro / Premium). How template-y does it feel? What would a craft-forward version look like?

20. **Footer.** 4-column footer with tiny uppercase headers (Product / Help / Legal / Connect). How template-y? What would make it feel like a deliberate end-of-page beat instead of generic SaaS chrome?

### Mobile vs. desktop

21. **Mobile experience.** The vast majority of consumer traffic is mobile. How does the site read on a phone? Where does the desktop layout break down or feel cramped?

22. **Touch targets and CTA hierarchy.** Are the primary CTAs ("Try It Free", "Get Chords") doing the right work? Are there too many secondary buttons competing?

### Brand consistency

23. **Brand voice match.** The recently-added voice line is "We don't guess. We listen." (italic serif, Fraunces). Does the rest of the site match that voice — confident, direct, slightly heretical? Or is most copy more generic SaaS-marketing tone?

24. **Coherence across pages.** stemscriber.com (landing) vs. /app (upload UI) vs. /youtube-fallback.html — do they feel like the same brand? Where does coherence break?

### What's working

25. **Real strengths.** What does StemScriber currently do BETTER than the reference benchmarks? What should NOT be touched in any redesign because it's working?

### Path to $1M-craft level

26. **Top 3 changes for biggest visual leap.** If the founder could only make 3 changes in the next 2 weeks, which 3 would close the most distance between current state and "$1M website"? Be concrete: which file, which section, which element, modeled after which reference.

27. **The single hardest thing.** What's the single biggest obstacle to reaching that level? (e.g., "you need a hero photo and AI image generators are bad at hands; you'll need a real photographer or stock photography from Unsplash") — i.e., the constraint that's not just a design decision but a resource/capability constraint.

28. **Realistic ceiling for a solo founder in 2 weeks.** With no design hire and ~10 dev hours per week, what's the best achievable level the founder can hit? "$1M website" or "$200K website that punches above its weight"? Be calibrated — over-promising helps no one.

### Open question

29. **What's not asked here that should be.** What question should this audit include that I haven't asked?

---

## Response format requested

Numbered answers, one per question. For each:
- Direct answer (1-3 sentences for quick items; more for complex ones)
- Specific element/section/file referenced where applicable
- Reference-benchmark comparison if relevant ("Moises does X better because Y")
- Severity tag at the end:
  - 🔴 RED — major issue, must fix before launch to hit "$1M" benchmark
  - 🟡 YELLOW — meaningful issue, should fix
  - 🟢 GREEN — working well / no issue

End with an overall summary: where the site lands on the 1-10 craft scale, where it could realistically land in 2 weeks, the 3-5 highest-leverage actions ranked by impact.

---

*Where you're uncertain, say so. Where you'd need more information than the URLs provide, ask. The goal is sharp honest feedback, not encouragement.*
