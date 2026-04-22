# StemScriber Launch Marketing — Drafts

**Drafted:** 2026-04-16. Launch target: 2026-05-12.
**Status:** First-pass drafts. Jeff to edit/approve before sending.

---

## 1. YouTube Shorts (~45 sec)

**Hook (0-3s)** — text on screen, no voiceover yet:
> "Every song. Every instrument. Isolated."

Visual: A familiar song playing (use one of your own — e.g. Kozelski "The Time Comes") over a waveform. After 1 second, the waveform splits vertically into 6 colored tracks (vocals, guitar, bass, drums, piano, other) — mimic the landing page hero animation but shot on the actual app.

**Second (3-15s)** — voiceover starts, screen recording:
> "Drop any song into StemScriber — MP3, WAV, whatever you've got."

Visual: drag-and-drop a file onto the app. Cursor releases it. Progress bar starts moving.

**Third (15-25s)** — fast cuts:
> "You get the stems, the chord chart, and the guitar tabs. All from one upload."

Visual: cuts between — the 6 stem tracks with mute/solo buttons, the chord chart rendering, guitar tab notation populating.

**Fourth (25-38s)** — practice mode:
> "Practice mode lets you slow down tricky parts without pitch shift, loop sections, and mute the instrument you're learning."

Visual: slow the speed slider down to 0.75x, click A-B loop on a chord change, mute the guitar track.

**Close (38-45s)** — CTA:
> "Three free songs a month. No card required. StemScriber.com."

Visual: cursor hovers over "Try It Free" button, clicks. Text overlay: **stemscriber.com** + logo.

**Caption / description:**
> Stem separation + chord detection + tab export + practice mode. The whole pipeline, one upload. Free tier available at stemscriber.com.
> #musician #guitarist #stems #learnsongs #musicproduction

**Hashtag strategy** — don't spam, pick 5: `#musician #guitarist #learnsongs #musicpractice #stemseparation`

---

## 2. Reddit Post — r/WeAreTheMusicMakers

**Title options** (pick one):
- *"I built a tool that separates stems AND generates chord charts from any song — free tier for this sub"*
- *"Launching a stem separation + chord detection app next month. Looking for honest feedback before the May 12 launch."*
- *"Made something for guys like us who learn songs by ear — would love rough feedback"*

Recommended: option 2. Most honest, least marketing-y.

**Body:**

> Hey r/WeAreTheMusicMakers —
>
> I've been building StemScriber on and off for about a year. Shipping publicly on May 12. Before that I want to get honest feedback from people who actually learn and play, not Twitter randos.
>
> **What it does:** upload a song → get the 6 stems separated (vocals, guitar, bass, drums, piano, other) + a chord chart + guitar tabs + a practice mode with speed/pitch/looping.
>
> **What it's not:** not trying to replace Moises or Ultimate Guitar. The difference is one upload gets you everything — stems AND chords AND tabs — without bouncing between 3 tools.
>
> **Tech honesty:**
> - Stem separation uses BS-RoFormer (same model as most commercial tools — the outputs are good, not magic)
> - Chord detection is bass-root-first with stem-aware multi-pitch analysis — around 95% on the songs I've tested, but harder songs (two guitars in the same pan, heavy synth pads) still trip it up
> - Guitar tab is string+fret output, not a rewrite of the part
>
> **Free tier:** 3 songs/month, no card. Pro is $10/mo for heavier use.
>
> If you want to kick the tires before launch: **stemscriber.com**. Pick any song you know well and tell me where it falls apart — that's more useful than compliments. Especially interested in:
> - Songs where the chord detection misses
> - Stems that came out muddy on mixes that should've been clean
> - Anything that feels slow or confusing in the UI
>
> Happy to answer any technical questions. Not here to spam — I'll probably not post again for a while.

**Posting tips:**
- Post on a Tuesday or Wednesday around 10 AM EST for max engagement
- Do NOT cross-post to 5 subs on the same day — that's rule-breaking on most music subs
- Reply to every comment for the first 24 hrs — the sub rewards engagement
- Don't link the stemscriber.com in the body as a clickable link — just the text so it doesn't trigger auto-mod
- Read the sub's self-promotion rules first (usually requires you be an active member, not just post-and-leave)

**Alternative subs to post in sequence (one per week):**
- r/guitar (massive — use a tab-focused angle)
- r/WeAreTheMusicMakers (producer angle — what we drafted above)
- r/edmproduction (stem angle — remixing samples)
- r/Songwriting (learning-songs angle)

---

## 3. Friends worth asking for pre-launch feedback

Two personal asks to musicians who'd give honest reactions. Not a "launch channel" — just people whose opinion matters.

### Tim Davis — KODA (kodamusic21@gmail.com / text)

> Hey Tim — gonna finally launch StemScriber on May 12. Wondering if you'd be down to try it out this week on some of the KODA tunes we've recorded — I want to see how well the stem separation + chord detection handles soul/R&B harmonies before other people bang on it. You'd get free unlimited access (obviously) and anything you find weird I can probably still fix. Let me know and I'll text you a link.

### Stephen Jenkins — Spare Kings (stephenejenkins@hotmail.com)

> Hey Stephen — the audio tool I've been building is close to launch (May 12). It separates songs into individual instruments and writes out the chord chart automatically. Figured the piano jazz stuff you play would be a real test for it — lot of extended chords, close voicings. Can I send you an early access link? Completely free, and I can actually use your honest reactions before I let it loose on the internet.

### KODA and Spare Kings — follow-up script if they say yes

> Perfect. Go to stemscriber.com, sign in with Google, and drop any audio file (MP3, WAV, whatever). It'll take a minute or two to process. After that you can download each stem, play with the mixer, slow stuff down, look at the chord chart.
>
> What I'd really love to know:
> 1. Did the chord chart match what you actually played?
> 2. Did any stem come out bad (muddy, bleed from other instruments, missing a part)?
> 3. Anything that felt confusing in the UI?
>
> Text me or email jkozelski@gmail.com — all feedback welcome, especially the rough stuff.

### Generic "friends & musicians you know" text (for anyone else)

> Heads up — I'm launching a thing called StemScriber on May 12. Upload a song, get every instrument isolated + a chord chart + guitar tabs. Free tier available. Would love it if you'd kick the tires before launch day and tell me what's wrong. stemscriber.com. Three free songs per month.

---

## 4. Launch day checklist (reminder for May 12)

Not drafts — just the punch list:

- [ ] Post the Reddit draft (r/WeAreTheMusicMakers first, then stagger others over 2-3 weeks)
- [ ] Publish YouTube Shorts (cross-post to Instagram Reels + TikTok same day)
- [ ] Send Tidepool texts morning of launch
- [ ] Post on personal X, Facebook, Instagram with a quick screen recording
- [ ] Update Tidepool Artists website with a "Tools" link pointing to StemScriber
- [ ] Email Alexandra Mayo with launch-day confirmation (short, respectful of her time)
- [ ] Watch Plausible for traffic and Cloudflare for errors in the first 24 hrs
- [ ] Check Supabase for signups hourly

---

## Notes for Jeff

- **Tone across all these:** musician talking to musician. No "AI-powered" or "revolutionary." Matches the voice guide in customer-service-playbook.md.
- **Don't mention Alex (Vapi voice agent) publicly until it's Phase 2 ready** — unadvertised works better right now.
- **Don't mention the chord library or Phi-3 anywhere** — one is dead, one is deferred.
- **The Reddit post's self-criticism ("not magic", "still trips it up") is intentional** — it reads as honest, which the sub respects. Don't edit that out.
