# StemScriber External Audit Prompts
**For:** parallel critique by OpenAI (ChatGPT, $200/yr plan) + Manus ($20/mo, 4000 credits/mo)
**Date:** 2026-04-17 (22 days to launch)
**Goal:** independent red-team of our May-5 launch plan — find what we're missing, what's impractical, what will embarrass us after launch.

## Credit / cost strategy

**Manus (limited budget, 4000 credits/mo):** autonomous, execution-heavy tasks. One well-scoped build task will burn ~100-200 credits. Don't waste credits asking Manus for opinions — use ChatGPT for that. Use Manus ONLY where you need actual running code that proves or disproves something. One instrument (piano), one working proof, done.

**ChatGPT Plus (unlimited within usage limits):** red-team critique, plan-level analysis, marketing copy review. Free to iterate until you get what you need.

**Split of labor for this audit:**
- Manus → build a working piano transcription pipeline, run it on a real track, tell us if the output is garbage. Burns ~150-250 credits total. Deliverable: code + 1 test song + honest report.
- ChatGPT → red-team the launch plan (Prompt A below). Free. Deliverable: numbered findings + concrete actions.

Do both in parallel. Don't use Manus for the red-team — ChatGPT is cheaper for that and just as good.

---

## Prompt A — OpenAI (ChatGPT / GPT-5 / o1) — "Red-Team Critique"

Copy/paste this into a new conversation. Attach `transcription-failure-inventory.md` if the model supports file upload, or paste its contents after the prompt.

```
You are a senior ML engineer hired to red-team a musician's software product 22 days before launch. The founder is a musician, not an ML specialist. You will be critical but constructive. Do not hedge. Point out what will embarrass us, break in production, or fail silently.

PRODUCT CONTEXT
StemScriber (stemscriber.com) takes an uploaded song and produces:
1. 6 separated audio stems (vocals, guitar, bass, drums, piano, other) via BS-RoFormer-SW running on Modal GPU. This works well — 95% quality on real-world audio.
2. A professional chord chart: detected chords with diagrams, lyrics placed under the chord they change on, section labels (verse/chorus/solo), transpose controls, playback-synced measure highlighting, auto-scroll, print-to-PDF. 95% chord quality after a bleed-simplification pass that strips over-extended chord qualities when systematic stem bleed is detected.
3. A practice mode: mute/solo/volume per stem, speed 50-200% without pitch shift, A-B looping, seek, rate-matched loop.

What we had, tried, and gave up on:
- Custom CRNN guitar tab model (Trimplexx, trained on GuitarSet) — 85% test F1 on paper, complete nonsense on BS-RoFormer stems due to domain gap.
- Custom CRNN drums/bass/piano models — same story, nonsense output.
- Phi-3 QLoRA chart formatter — repetition hallucination, contaminated training data, abandoned. Rule-based formatter ships instead.

For May 12, we are SHIPPING without any tab or sheet-music view. The whole Tab / Sheet Music button is being cut. Only the chord chart + practice mode ships. Post-launch roadmap: Basic Pitch for guitar+bass, Sony hFT-Transformer for piano, Omnizart for drums.

OUR PLAN (what I want you to attack)
Front end:
- Remove the Tab / Sheet Music button from the notation area
- Remove the top stem-picker row (Guitar / Bass / Piano / Drums icons) because it only drove the now-cut notation view
- Keep the chord chart, transpose, sharps/flats, auto-scroll, print, piano visualizer
- Keep the stem mixer sidebar — that's the practice feature
- Rewrite marketing copy from "audio to tab" to "stems + chord chart + practice mode"

Backend:
- Keep the chord detection pipeline (BTC + V8 + stem-aware bass-root-first + bleed-simplification). Shipping at 95% quality on small test set.
- Stop generating MIDI / MusicXML / Guitar Pro files at runtime (save Modal compute) but keep the generation code commented-in-place.
- Chart formatter fixes still needed: (a) detect instrumental subsections inside verse/chorus when lyric gaps exceed a threshold, (b) label solo sections with the loudest-stem name (Piano Solo, Guitar Solo, Sax Solo) via RMS analysis of stems during the instrumental window.

Post-launch ML roadmap:
| Instrument | Model | License | Weeks out |
|------------|-------|---------|-----------|
| Guitar     | Spotify Basic Pitch + our existing A* fret optimizer | Apache-2.0 | 1-2 |
| Bass       | Basic Pitch + octave-clamp + kick-bleed gate | Apache-2.0 | 3-4 |
| Piano      | Sony hFT-Transformer + Piano-Hands splitter + partitura quantization | MIT | 4-6 |
| Drums      | Omnizart drum + music21 percussion staff | MIT | 4-6 |

WHAT I WANT FROM YOU
1. Attack the chord-detection quality claim. Our 95% figure comes from a small test set of songs the founder knows well. What's the likely real-world number on a pop/rock sample of 100 random songs? What songs or genres will it fail on and make us look bad?
2. Attack the launch-without-notation decision. Is there a minimum-viable notation feature we could ship in 18 days that's better than cutting it entirely? (Be specific — say what and how, or say "no, cutting is right.")
3. Attack the ML roadmap. Is the Basic-Pitch-for-everything pattern naive? Is hFT-Transformer going to work on pop piano stems that bled through BS-RoFormer? Will Omnizart's training-convergence bug (documented in its repo) bite us?
4. Attack the product positioning. "Stems + chord chart + practice mode" — is this actually differentiated against Moises + Chordify, or are we just a UI wrapper on things people already pay for separately?
5. Attack the timeline. 18 days with one developer. What's the biggest thing we will cut the night before launch?
6. Anything the founder missed BECAUSE he's a musician and not an engineer. What should the engineer catch that a musician wouldn't?

Format: numbered findings. Each finding = one problem + evidence + concrete action. No marketing language. No praise. Under 2000 words.
```

---

## Prompt B — Manus (autonomous agent) — "Build a Proof, Prove We're Wrong"

Manus executes — don't just ask for an opinion, ask it to build. Give it one instrument (piano, the hardest) and see if it actually works.

Copy/paste:

```
I am 22 days from launching stemscriber.com, a web app that takes a song, separates it into 6 stems, and produces a chord chart + practice mode. Stem separation works. Chord detection works. What does NOT work: automatic music transcription — I trained custom CRNN models for guitar/bass/piano/drums on GuitarSet/Slakh2100/MAESTRO and the output on my real BS-RoFormer stems is nonsense due to domain gap.

I need you to build a working proof of concept for ONE instrument so I know my post-launch ML roadmap is real, not theoretical.

TASK: Build a Python script that takes a piano stem WAV file (~4 minutes, 44.1kHz) and produces a MIDI file with separate left-hand and right-hand tracks, suitable for grand-staff music notation.

CONSTRAINTS
- Must use pretrained models only — I will not train anything.
- License must be commercial-safe: MIT, Apache-2.0, BSD. GPL/CC-NC/CC-SA are REJECTED.
- Must run on a single A10G GPU (Modal serverless) OR CPU in under 2 minutes per song.
- Must handle real-world BS-RoFormer piano stems, not clean MAESTRO recordings. Stems have compression artifacts, slight bleed, and non-classical timing.

MY PLANNED STACK (which I want you to VALIDATE or IMPROVE)
1. Sony hFT-Transformer (https://github.com/sony/hFT-Transformer, MIT) for audio → note MIDI
2. asigalov61/Piano-Hands (Apache-2.0) for per-note LH/RH classification
3. CPJKU/partitura (BSD-3) for performance-aware quantization at 1/16 with triplet tolerance
4. music21 (BSD) for MIDI → MusicXML export with two PartStaff objects

WHAT I WANT YOU TO DELIVER
1. A working `piano_transcribe.py` script that runs end-to-end on an input WAV and writes two output files: a `.mid` file and a `.musicxml` file.
2. Test it on a real pop/rock piano stem (not MAESTRO). Use an open-source track if you need one — suggest a CC-licensed song on Free Music Archive or similar and download it yourself.
3. Write a short report (under 500 words) on:
   - Did it actually produce readable sheet music or just a pile of 32nd-note garbage?
   - Which step was the weakest — transcription, hand-splitting, quantization, or export?
   - Would you pick a different stack? What and why?
   - What does it cost in Modal compute per song (time × $0.60/hr on A10G)?
4. Push the code to a public GitHub gist or a downloadable zip.

Do NOT just summarize the repos and the plan. I need actual executable code and actual output listened to / viewed. If the output is garbage, say "it's garbage" and show me the MusicXML rendered via OSMD so I can see.

If you hit a blocker (dependency, license, audio I/O), solve it and keep going. Do not stop and ask me clarifying questions unless you genuinely cannot proceed.
```

---

## Notes for Jeff

**Both prompts reference `transcription-failure-inventory.md`** — attach or paste that file to both conversations so the audit starts from our actual context, not their prior beliefs about AMT.

**OpenAI takes 5-10 minutes of reading + back-and-forth.** Manus will run autonomously for 30-90 minutes. Hit both in the next hour and you'll have two independent signals by tomorrow morning.

**What you're buying:**
- OpenAI red-team → whether our LAUNCH plan is sound
- Manus build → whether our POST-LAUNCH roadmap actually ships working code

**What you're NOT buying:**
- Either to tell you what to launch. That's your call as the founder.
- Either to replace the 4-agent research we already did. They're validators, not primary planners.
