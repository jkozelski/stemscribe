# StemScriber marketing drafts — refresh (2026-05-12)
**Drafted:** 2026-05-12
**Supersedes:** marketing-drafts-2026-04-26.md
**For:** Jeff to redline before publishing. Nothing ships without sign-off.

Major shifts vs Apr 26 draft:
- Stopped attacking Chordify. Additive framing only ("plus extensions"), not
  "they get it wrong." Jeff's note: "They're gonna get kinda close just with
  no extensions" — be honest about that.
- Klangio treated as complementary (sheet music / per-instrument), not a
  competitor.
- Launch arc rewritten for the June 20 Refinery soft launch + public ~1-2
  weeks later. Soft launch is backstage / invite-only.
- Real F1 numbers replace the old "we cracked jazz" hype.
- No "AI" anywhere customer-facing.

---

## Tagline + positioning

**Tagline:** *Hear it. Chart it. Play it.*

**Positioning statement (one paragraph, reusable everywhere):**

> StemScriber takes any song you can upload and gives you a working chord
> chart in seconds — with the 7ths, 9ths, and extensions other tools skip.
> Built by a gigging musician for the songs you actually want to learn.
> Most come out right. The hard ones come out mostly right. You can fix any
> chord in one tap.

Short one-liner for socials / bios:

> *Fast first-draft chord charts you finish in seconds. Built by a working
> musician, not a SaaS team.*

---

## Landing page hero copy (~50 words)

**Headline:** Hear it. Chart it. Play it.

**Sub:** Drop in any song. Get a chord chart with the 7ths, 9ths and slash
chords other tools leave out. Loop a section, isolate the bass, drop the
key. Fix any chord in one tap.

**CTA primary:** Try a song free
**CTA secondary:** See a sample chart (Steely Dan — Aja)

---

## YouTube Short (60s vertical) — script

**[0:00–0:04]** Hook on screen, dark background, big text:
> *"What's the first chord of Aja?"*

Cut to a guitar in Jeff's hands, sitting on the porch. He shrugs.

**[0:04–0:10]** Voiceover, conversational:
> "Most chord apps will tell you Bm. That's fine. It's close. But Aja
> isn't a Bm song — it's a Bm9 song. That's why it sounds like Steely Dan."

**[0:10–0:18]** Screen recording: drag-drop `Aja.mp3` onto stemscriber.com.
Speed up the processing pass.

**[0:18–0:32]** Slow pan across the chord chart that appears:
`Bmin7  Cmin9  Gmin9  Amin9  F#min9  Emin9`. Voiceover:
> "Same chord names you'd write on a napkin at rehearsal. Real 7ths.
> Real 9ths. Right key."

**[0:32–0:45]** Practice mode: stem mixer pulled up, bass solo'd, eight
bars play with chords highlighted in sync.

**[0:45–0:55]** Phone shot: same chart on phone, finger taps a chord and
holds — popup lets the user change it. Voiceover:
> "Fast first draft. You fix anything in one tap."

**[0:55–0:60]** Logo + URL: stemscriber.com.

### Notes
- No "AI" language anywhere
- No competitor logos on screen
- "Close. It's a Bm9 song" is the additive framing — not an attack
- Use the real Aja job ID `f0a2363f...` chart, don't fabricate
- Aim for 45-50s in practice

---

## Reddit — r/WeAreTheMusicMakers

**Title:** First-draft chord charts for the songs you actually want to learn
— with the 7ths and 9ths included

**Body:**

Hey all — long-time lurker. I'm a gigging musician (20+ years, mostly
Charleston) and I built a tool for a problem I kept running into: the
audio-to-chord tools out there give you the bones of a song but stop at
triads. For most pop/rock that's fine. For the stuff I actually wanted
to chart — Steely Dan, Jamiroquai, Stevie — you lose the harmony that
makes the song sound like itself.

So I built **StemScriber** (stemscriber.com). Upload a song, get a chord
chart with the 7ths, 9ths, sus and slash chords filled in. Stem mixer
built in. Loop, transpose, key change. Fix any chord in one tap.

Honest numbers, because I know this sub: I ran an 18-song audit against
a hand-graded reference. Mean F1 0.804. Roughly 70% of songs come out
ready-to-play, ~17% need a couple tweaks, ~13% need real cleanup. Aja
came back 226 for 226 bars on the jazz extensions. Heavy metal, dense
classical, ambient, and solo-instrument tracks are still hard.

It's beta. Free tier, no card. The biggest thing I want from this sub
is: tell me where it falls apart. If a song you care about comes back
wrong, I'd love to dig in.

(Built solo. Happy to nerd out on detection logic in the comments.)

### Notes
- No "we beat Chordify" framing
- Honest F1 stated up front
- Builder-musician voice, not founder voice
- Disclose self-promotion per sub rules (it does)

---

## Reddit — r/guitar

**Title:** Built a chord-chart tool that includes 7ths and extensions —
free to try

**Body:**

I'm a gigging guitarist and I kept charting songs by ear because the
chord-chart sites would give me triads when the song had real harmony
in it. So I built **StemScriber** (stemscriber.com) to do a better
first pass.

What it does:
- Chord chart from any upload, with 7ths / 9ths / sus / slash chords
- Stem mixer (solo the bass, drop the vocals, loop 8 bars)
- Practice mode: transpose, capo hints, slow without pitch shift
- One-tap chord edits if it gets something wrong
- Works on phone

What I'd tell you honestly: in an 18-song audit it averaged 0.804 F1
against a hand-graded reference. Most songs are ready to play. Some
need a couple tweaks. A few need real cleanup. Heavy metal, dense
classical, ambient and solo-instrument tracks are still hard.

Free tier, no card. If you break it, tell me what song — that's how
this gets better.

### Notes
- "Free tier, no card" stated explicitly (sub culture)
- Honest caveats kept

---

## Twitter / X thread (7 tweets)

**1/** Built a thing for working musicians.

You drop in a song. You get a chord chart with the 7ths, 9ths, sus
and slash chords filled in — not just triads.

stemscriber.com

[screenshot: Aja chord chart, Bmin7 / Cmin9 / Gmin9 / Amin9 ...]

**2/** Honest numbers, because anyone can show a cherry-pick:

18-song audit, mean F1 0.804 against a hand-graded reference. ~70%
come back ready to play. ~17% need a couple tweaks. ~13% need real
cleanup.

I'll publish the audit.

**3/** Aja came back 226-for-226 on the jazz extensions. That one I'm
proud of — jazz harmony is where most chord tools quit.

[screenshot: 226 bars of Bm9 / Cm9 / Gm9 ...]

**4/** What's in the box on one upload:

- Chord chart with extensions
- 6 separated stems (vocals/bass/drums/guitar/piano/other)
- Practice mode (loop, transpose, slow, capo)
- One-tap chord edits

Free tier. No card.

**5/** What I'd tell you to skip it for:

Heavy metal, dense classical, ambient, solo-instrument recordings.
The detector was trained where I live as a musician — pop, rock, R&B,
funk, country, jazz with clear changes.

**6/** Built solo over six months while playing four gigs a week.

Stack is Python + Flask, GPU stem separation, bass-anchored chord
detection, and a 337-class chord vocabulary so the extensions actually
have names.

Soft launch in Charleston June 20 at the Refinery.

**7/** If you try it and it breaks on a song you care about, tell me.
That's the bar.

stemscriber.com / @stemscriber

### Notes
- No "AI" anywhere
- No URL in tweet 1 (Twitter penalizes)
- "Soft launch" mention is fine — invites curiosity without committing
- Schedule Tue/Thu morning Pacific

---

## Hacker News — Show HN

**Title:** Show HN: StemScriber – chord charts from audio, with the
extensions included

**Body post (top comment from OP):**

I'm the (solo) builder. StemScriber takes a song upload and gives back
a chord chart with the 7ths, 9ths, sus and slash chords filled in,
plus six separated stems and a practice mode.

The technical bit HN might care about: most chord detectors collapse
to triads because stem-separation bleed makes minor/dominant/extension
disambiguation noisy. We do a few things differently:

1. **Bass-anchored root detection.** Run pyin on an isolated bass stem,
   then template-match a 337-class chord vocabulary against the full
   mix conditioned on that root. The root is almost always right;
   the quality is what's hard.

2. **Family-aware consistency.** min7 / min9 / min11 / madd9 are the
   same musical idea. Treating them as separate qualities and voting
   blew up on jazz tracks. Grouping by family before voting fixed it.

3. **Asymmetric m3 priority.** Hearing the minor 3rd on a root at
   least 3 times is positive evidence. Not hearing it is weak evidence.
   That asymmetry matters: it flipped 8 of 10 audit songs to grade B+.

Numbers, since HN will ask: 18-song audit, mean F1 0.804 against a
hand-graded reference. Aja came back 226-for-226 on extensions.
Heavy metal, dense classical, ambient and solo-instrument tracks
still struggle.

Stack: Python/Flask, Modal for GPU stem-sep, Postgres, Cloudflare.
Closed source for now, happy to discuss internals in the thread.

Free tier, no card. stemscriber.com.

### Notes
- HN audience: technical detail wins over story
- Don't post until weekday morning Pacific
- Stay in the comments — HN punishes drive-by posters
- No competitor name-dropping

---

## Product Hunt — short description

> **StemScriber** — Drop in any song. Get a chord chart with the 7ths,
> 9ths and slash chords most tools skip. Six separated stems and a
> practice mode in the same upload. Built by a gigging musician for
> the songs you actually want to learn.

**Tags:** music, audio, chord-chart, practice-tool, music-theory,
musicians, stems

**Maker comment for launch day:**

> Hi PH — solo builder here. I'm a gigging guitarist and I kept charting
> songs by ear because chord-chart tools stopped at triads. StemScriber
> is the first-draft I wish I'd had: chord chart with real extensions,
> stems separated, practice mode, all from one upload. Honest beta —
> 18-song audit at 0.804 F1, jazz works, heavy metal still hard. Would
> love your feedback on what songs to fix next.

### Notes
- 240-char summary length respected
- Coordinate with Twitter thread + Reddit on launch day

---

## Refinery cohort email (invite-only soft launch)

**Subject:** A weird ask before the Refinery show

**Body:**

Hey [name],

So you know I've been heads-down on this side project for a while —
StemScriber, the chord-chart tool. It's ready enough that I'm doing a
soft launch the weekend of the Refinery show (June 20, Watkins Glen
50th — me + Chester on The Band, The Reckoning, Idlewild South).

Here's the ask: I want the people playing the show to be the first
real users. If you upload one of our tribute-set tunes — or anything
in your own catalog — and tell me where the chord chart matches what
you actually play and where it doesn't, you'd be doing me a real favor.
Takes five minutes per song.

In exchange you get:
- A beta code (REFINERY-XXXX) before the public launch
- StemScriber for life
- A name credit when we go public, if you want it

Bring your code to the show — I'll have a few extras to hand around
backstage. Public reveal is a week or two after, once we've used it on
real tribute-set work and have something honest to point to.

Link's at stemscriber.com. Reply with thoughts, push-back, "you're
crazy," whatever.

See you at the Refinery.

Jeff

### Notes
- Personal, not a blast — send one-by-one to Chester, Reckoning,
  Idlewild South, Tim Davis (KODA), Stephen Jenkins (Spare Kings), etc.
- No public-mic launch implied
- Beta code format `REFINERY-XXXX` per launch-date memory
- Public launch deliberately framed as "after we have real users"

---

## The differentiation cheat sheet (5 bullets, honest + additive)

Use these as the source-of-truth bullets for any new copy. All
additive, all honest, none attack-framed.

1. **Extensions included.** The chord chart names the 7ths, 9ths, sus
   and slash chords most other tools omit. You get the chords that
   make the song sound like itself, not just the shapes.

2. **Any song you can upload.** Not a catalog. Indie recordings, live
   bootlegs, covers, your own demos — anything you have an audio file
   for. Catalog tools cover the hits; we cover what you actually play.

3. **Stems + chords from one upload.** Vocals, bass, drums, guitar,
   piano, other — separated and in a mixer next to the chord chart.
   Loop the bass, solo the vocal, fix a chord, keep going.

4. **One-tap fixes.** When a chord is wrong (and some will be), you
   tap it and change it on the spot. Don't re-upload. Don't fight the
   tool.

5. **Built by a working musician.** 20+ years of gigging informs every
   decision — vocabulary, practice mode, mixer behavior, the songs we
   trained on. It feels like a tool a musician would build because it
   is one.

---

## Coordination playbook for launch arc

**Soft launch — Saturday June 20, 2026 (Refinery, Charleston):**
- Site is up, BETA-gated.
- Beta codes (REFINERY-XXXX) handed out backstage to the three bands
  and invited friends.
- No public announcement. No mic-launch.
- Plausible analytics watch: signups by REFINERY-XXXX code prefix.

**Bridge week (June 21 – ~July 1):**
- Refinery cohort uses the tool on tribute-set work.
- Collect chord charts they generated + quotes they're willing to give.
- Fix the worst bugs they surface.
- Pre-write social with their permission-granted content.

**Public launch (~late June / early July):**
- Twitter thread first thing Tuesday morning.
- Reddit (r/WeAreTheMusicMakers + r/guitar) ~30 min later.
- HN Show post by 9am Pacific.
- Product Hunt same morning.
- Real user quotes from the Refinery cohort in all of them.

**Throughout public-launch day:**
- Admin dashboard for queue depth (cap at 4).
- If sustained queue >8, scale CPX41 → CPX51 (one click).
- If HN hits the front page, manually pause signups 30 min if needed.

---

## What this is NOT yet

- Drafts. Edit ruthlessly.
- No graphics. Short needs a video editor pass; tweets need real
  screenshots from the Aja and Free Fallin' job IDs.
- No pricing. Free tier mentioned; Pro tier left vague on purpose.
- No date precision beyond "Refinery weekend" + "~1-2 weeks later".
