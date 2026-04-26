# StemScriber marketing drafts — first pass
**Drafted:** 2026-04-26
**For:** Jeff to review, edit, and use when ready

These are first drafts to react to, not finished copy. Goal: give you
material to redline rather than starting from blank pages. None of this
ships without your sign-off.

Background context driving the angle: the Apr 25 audit cracked Steely
Dan and Jamiroquai chord detection — most music transcription tools
fall apart on jazz. That's the unique selling moment. Every draft below
leans into "we can do the songs Chordify can't."

---

## 1. YouTube Short (60 seconds, vertical)

### Script

**[0:00–0:03]** Pull-quote screen, dark background, big text:
> "Most chord apps fail on jazz."

**[0:03–0:08]** Chordify screenshot of a Steely Dan song showing wrong
or missing 7ths. Voiceover (or text-overlay if no voice):
> "Steely Dan, Jamiroquai, Stevie Wonder — the songs musicians actually
> want to learn are the ones AI keeps getting wrong."

**[0:08–0:15]** Switch to StemScriber. Drag-drop "Aja.mp3" onto the
upload box. Speed up the processing montage.

**[0:15–0:30]** Show the chord chart appearing. Slow zoom on the actual
extensions: `Bmin7  Cmin9  Gmin9  Amin9  F#min9  Emin9`. Voiceover:
> "Real 7ths. Real 9ths. Right key."

**[0:30–0:45]** Practice mode: stem mixer pulled up, vocals isolated,
play 8 seconds of the song with chords highlighted in sync.

**[0:45–0:55]** Cut to chord chart on phone. Pan across `Cmin7 Gmin7 
Dmin7 Amin9` (Alright by Jamiroquai). Voiceover:
> "From your phone. From any song. From the artists you actually want
> to learn."

**[0:55–0:60]** Logo + URL: stemscriber.com. Tagline:
> "Tear the sound apart."

### Notes
- 60s upper limit; aim for 45-50s in practice
- No "AI" branding per your prefs
- Use the audit chord charts directly — they're already on the VPS as
  job IDs `5b731e47...` (Alright) and `f0a2363f...` (Aja)
- Vocals in the practice-mode segment should be isolated to flex the
  stem-separation feature subtly without naming it

---

## 2. Reddit — r/WeAreTheMusicMakers

**Title:** I built a tool that finally gets jazz chord charts right (Steely
Dan, Jamiroquai included)

**Body:**

Hi everyone — long-time lurker, first-time poster. I've been a working
musician for 20+ years and I've been frustrated for forever that the
audio-to-chord-chart tools out there bail on anything more complex
than 4 chords. Try Chordify or Moises on Aja or Peg and you get plain
triads where the song is full of m7s and m9s.

So I built **StemScriber** (stemscriber.com). Stem separation + chord
detection + a practice mode in one upload. Free to try.

I'm posting because we just shipped a fix that explicitly handles
extension-heavy progressions. Here's the same Steely Dan track before
and after:

```
Yesterday on Aja:   "all triads, zero 7ths in 226 bars"
Today on Aja:       Bmin7 / Cmin9 / Gmin9 / Amin9 / F#min9 / Emin9 ...
                    (every bar shows real harmony)
```

What it works well on: pop, rock, R&B, funk, soul, jazz with clear
chord changes. What it's still tricky on: heavy metal, dense classical,
ambient, solo-instrument tracks. Honest about what's not there yet.

It's beta. Free tier: a few songs/month. No card needed. Would love
brutal feedback from this sub specifically — if it falls apart on a
song you care about, tell me and I'll dig in.

(Built solo. Happy to answer engineering questions in comments.)

### Notes
- Tone: honest, not promotional, leans into the "we lurkers are also
  builders" vibe of WAYTMM
- Has to disclose self-promotion per sub rules (it does)
- The "before/after" framing makes the post about the *problem* people
  recognize, not about your product
- Don't post this until #54 (concurrency cap) is verified holding under
  load — a Reddit hit could spike traffic

---

## 3. Reddit — r/guitar

**Title:** New chord-chart tool that handles 7ths and extensions (free
to try)

**Body:**

Built a tool called StemScriber for getting accurate chord charts from
audio. The thing that makes it different from Chordify / Moises:

- Detects 7ths, 9ths, slash chords, sus chords — actual jazz harmony
- Stem mixer built-in (loop the bass, isolate the vocals, etc.)
- Practice mode with auto-scroll, transpose, capo hints
- Works on phone

Tested it on Steely Dan's Aja yesterday — 226 bars of dense Bm jazz,
all extensions detected correctly. Posted here because if anyone's
chord-chart bar is "I can read what the actual recording is doing,"
this might clear it.

stemscriber.com — free tier, no card.

Honest caveats: built solo, in beta, breaks on heavy metal / ambient /
solo instruments. Pop/rock/R&B/funk/jazz is where it shines.

### Notes
- Shorter, more product-forward than the WAYTMM post (different sub culture)
- Mention "free tier, no card" upfront — guitar subs hate paywall surprises

---

## 4. Tidepool Artists outreach (email)

**Subject:** Want to test StemScriber on your masters?

Hi [name],

Quick ask — I've been heads-down on StemScriber (the audio-to-chord-chart
tool I've been building) and we just hit a quality milestone where it
handles complex jazz and funk progressions correctly. Steely Dan, Jamiroquai
etc. all transcribe accurately now.

Would you be willing to upload one of your tracks (one I have rights to,
e.g. anything from [recent album]) and tell me where the chord chart
matches what you actually played and where it doesn't? Free, takes 5
minutes, you'd literally be helping shape the product.

In return: feature spot in the launch announcement (with your permission),
StemScriber Pro for life, and the satisfaction of knowing your bandmates
can finally read the chart from your isolated vocals stem.

Link: stemscriber.com — sign in, upload a track, let me know what you see.

Reply with thoughts, push-back, "you're crazy," whatever.

Jeff

### Notes
- Personal-but-direct; you've mentioned Tidepool relationships are "fragile"
  so the ask is small (one upload) and has clear value (Pro for life,
  feature in launch)
- Send to Tim Davis, Stephen Jenkins, the Spare Kings folks separately,
  not as a mailing-list blast
- Track who responds — those who DO are your launch-day endorsers

---

## 5. Twitter/X launch thread (5 tweets)

**1/** Most audio-to-chord apps choke on anything more complex than
4 power chords. We just shipped a chord detector that handles real
jazz harmony.

Steely Dan's Aja, transcribed in real time. 226 bars, all extensions:

[screenshot of chord chart sample: Bmin7 Cmin9 Gmin9 Amin9 ...]

**2/** Yesterday's same detector on the same song: 226 bars of plain
triads. Zero 7ths. Zero 9ths.

Today: 100% of bars carry the right extensions.

What changed: family-aware consistency in the simplification gate, plus
m3-detection priority for minor songs.

**3/** It's part of a bigger thing called StemScriber. One upload gets
you:
- 6 separated stems (vocals/bass/drums/guitar/piano/other)
- Professional chord chart with diagrams
- Practice mode (loop, transpose, slow without pitch shift)

stemscriber.com. Free tier, no card.

**4/** Built solo over six months. Wrote and threw away custom CRNN
models, fought through stem-bleed simplification bugs, ate three real
fixes for breakfast yesterday. The honest report: pop/rock/R&B/funk/jazz
work great. Heavy metal and ambient need more time.

**5/** It's beta. The Reddit and Hacker News crowd is welcome to break
it. If it falls apart on your favorite song, I want to know — that's
the bar I'm aiming for.

stemscriber.com / @stemscriber

### Notes
- Tweet 1 is the hook — make sure the screenshot is sharp
- Don't link to your launch in the first tweet (Twitter penalizes early
  outbound links). Save the URL for tweets 3 and 5.
- Schedule for a Tuesday or Thursday morning, not weekends

---

## 6. Hacker News submission

**Title:** StemScriber – Audio to chord chart, handles jazz and extensions

**Comment to post immediately as the OP:**

Hi HN — I'm the (solo) builder. I've spent the last six months on
this and the audit results from yesterday are what made me feel ready
to post.

The technical hook: most audio-to-chord tools simplify complex
progressions to triads because their detector can't disambiguate jazz
extensions from stem-separation bleed. We solved that yesterday with
two changes:

1. *Family-aware consistency*: when min7 / min9 / min11 / madd9 all
   appear on a track, they're the same musical idea. Old logic counted
   them as different qualities and assumed bleed. New logic groups
   them by family and trusts the song-level signal.

2. *m3-detection priority*: when the detector hears the minor 3rd on
   a root at least 3 times, that's asymmetric positive evidence — the
   chord IS minor even if dominant detections outnumber. Detecting
   the m3 is reliable; not detecting it is weak signal.

Result: Jamiroquai, Steely Dan, dense jazz now transcribe correctly.

Stack: Python/Flask, Modal (cloud GPU for stem separation), Postgres,
Cloudflare. Detector is bass-anchored (pyin on isolated bass) +
337-class template matcher with the new simplification logic. Code
is closed for now but I'm happy to discuss internals.

Free tier. No card. stemscriber.com.

Caveats: beta, built solo, breaks on metal/classical/solo instruments.
Pop/rock/R&B/funk/jazz is where it shines.

### Notes
- HN audience cares about the technical details, not the product story
- Mention solo-built and the technical specifics — that's HN catnip
- Don't post until weekday morning Pacific time (the upvote curve
  works best then)
- Be in the comments to answer questions; HN punishes drive-by posters

---

## 7. Product Hunt — short description

> StemScriber — audio to chord chart for the songs Chordify gets wrong.
> Upload any track, get separated stems and a chord chart that handles
> jazz extensions, slash chords, and section labels. Practice mode
> built in. Built for working musicians, not casual listeners.

**Tags:** music, audio-to-text, chord-chart, music-theory, practice-tool

### Notes
- 240 chars / Product Hunt summary length
- Product Hunt audience is product builders who'd appreciate "Chordify
  gets wrong" — punchy, specific
- Coordinate launch day with the email list + Twitter thread

---

## Coordination playbook for launch day

When you're ready to flip the switch:

1. **24h before:** flip `ENABLE_JOB_EMAILS=true` on VPS. Verify queue
   handling works (concurrency cap holds at 4).
2. **Morning of:** Tweet the thread. ~30 min later, post Reddit. ~1h
   after that, submit to Hacker News.
3. **Throughout day:** monitor admin dashboard for queue depth. If
   it hits 8+ sustained, scale to CPX51 (one click, $25 more/mo).
4. **Evening of:** Tidepool Artists email (more personal, lower volume).

Stagger so no single channel spikes traffic past CPX41 capacity. If
HN hits front page, manually disable signups for 30 min while you
breathe.

---

## What this is NOT yet

- Drafts. Edit ruthlessly.
- No graphics yet — Shorts script needs a video editor pass; tweet
  thread needs the screenshot grabbed.
- No date in any of them. Add when you decide.
- No pricing called out (you haven't decided either).
