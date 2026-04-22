# StemScriber Customer Service Playbook
**Last updated: 2026-04-16**

## Support Channels
- **Contact form:** stemscriber.com website
- **Voice agent (Alex):** +1 (844) 791-5323
- **Email:** support@stemscriber.com
- **Handler:** Jeff (all human-touch support)

---

## 1. Response Time Targets

| Issue Type | Acknowledge | Resolve/Update |
|---|---|---|
| Critical (can't login, data loss) | **2 hours** | Same day |
| Billing / refunds | **4 hours** | 5-7 business days |
| Bug reports | **12 hours** | 48 hours (fix or workaround) |
| General questions | **24 hours** | 24 hours |
| Feature requests | **48 hours** | Log and acknowledge |

---

## 2. Template Responses

### "My stems sound bad / there's bleed"

> Hey [name],
>
> Thanks for letting me know — some songs are definitely trickier to separate than others, especially when instruments share similar frequency ranges (like acoustic guitar and piano, or two electric guitars panned together).
>
> A couple things to try:
> - Switch the separation model — go to Settings and try **RoFormer** if you were on Demucs, or vice versa. They handle different material differently.
> - If the song has a lot going on in the mix, the vocal stem usually comes out cleanest, and instruments that sit in their own frequency space (like bass) tend to separate well.
>
> What song were you working on? I'd like to test it on my end and see if I can get better results for you.
>
> — Jeff

### "The chord detection is wrong"

> Hey [name],
>
> Appreciate you flagging that — chord detection is something we're actively improving, and your feedback directly helps.
>
> Could you let me know which chords look wrong and what the correct ones should be? Our detection is stem-aware (we find the bass root first, then the quality on top) so knowing exactly where it went sideways helps me tune it.
>
> — Jeff

### "I can't upload my song"

> Hey [name],
>
> Sorry about that — let's figure out what's going on.
>
> Quick checklist:
> 1. **File format** — we support MP3, WAV, FLAC, and OGG. Other formats need to be converted first.
> 2. **File size** — make sure the file isn't over the upload limit.
> 3. **Error message** — did you see any error text on screen? A screenshot would be super helpful.
> 4. **Browser** — try a different browser (Chrome tends to work best). Also try clearing your cache.
>
> If none of that works, send me the file directly and I'll process it on my end and get you the stems.
>
> — Jeff

### "How do I get a refund?"

> Hey [name],
>
> No worries at all. We have a **30-day money-back guarantee** — if it's not working for you, you get your money back.
>
> If a song failed to process, your credits are restored automatically. If that didn't happen, let me know and I'll fix it right away.
>
> For a full refund, just reply here confirming and I'll get it processed within 5-7 business days.
>
> — Jeff

### "How do I use practice mode?"

> Hey [name],
>
> Practice mode is one of my favorite features — here's the quick rundown:
>
> 1. **Open any separated song** and hit the Practice Mode button.
> 2. **Isolate stems** — mute the instrument you want to play along with so you can fill in that part yourself. Or solo just the instrument you're trying to learn.
> 3. **Slow it down** — use the speed control to drop the tempo without changing pitch. Great for nailing tricky parts.
> 4. **Loop sections** — click and drag on the waveform to set a loop. Hit that chorus 50 times until it's locked in.
> 5. **Chords display** — the detected chords show above the waveform so you can follow along.
> 6. **Guitar tabs** — for the guitar stem, we show string and fret positions so you can see exactly how to play it.
>
> Let me know if you have any questions — happy to walk you through it!
>
> — Jeff

### "Feature request: [X]"

> Hey [name],
>
> That's a great idea — I've added it to our feature list. We're a small team so I can't promise a timeline, but I want you to know it's logged and we review requests regularly.
>
> [If already planned:] Actually, this is already on our roadmap! Can't give an exact date yet but it's coming.
>
> [If not feasible right now:] This one would take some significant work to build out, but I like the thinking. Going to keep it on the list for when we can get to it.
>
> Thanks for taking the time to share this — feedback from musicians like you is literally how we decide what to build next.
>
> — Jeff

### "Bug report: [X]"

> Hey [name],
>
> Thanks for reporting this — really helpful.
>
> So I can track it down, could you send me:
> - **What browser and OS** you're using (e.g., Chrome on Mac)
> - **Which song** triggered it (if applicable)
> - **Steps to reproduce** — what were you doing right before it happened?
> - **Screenshot or screen recording** if you have one
>
> I'll dig into it and get back to you with an update within 48 hours.
>
> — Jeff

---

## 3. Voice Agent (Alex) Integration

Alex is the StemScriber voice agent, live at **+1 (844) 791-5323**. Alex is the first line of contact for phone inquiries.

### What Alex handles directly
- FAQ: how the product works, supported formats, upload flow
- Pricing questions (Free / Pro / Premium)
- Feature explanations: stem separation, chord detection, practice mode, guitar tabs
- Basic troubleshooting: upload errors, "why does my stem sound muddy", chord accuracy
- Creating a support ticket for anything requiring human follow-up

### What Alex CANNOT do (these route to email)
- Account lookups
- Refunds or payment changes
- Subscription cancellations
- Password resets
- Anything requiring access to a specific user's data or songs

Alex's stock answer in those cases: "Shoot an email to support at stemscriber dot com and we'll get on it within 24 hours."

### Escalation flow (from Alex to Jeff)
When Alex creates a ticket, the handoff is **async only**:
1. Alex creates the support ticket
2. Email summary goes to **support@stemscriber.com**
3. SMS summary goes to Jeff via Twilio (+1 844-791-5323 → +1 803-414-9454)

**Alex NEVER transfers a live call to Jeff's personal phone.** Per Jeff's preference, escalations reach him async — email or SMS summary only, never a ringing handset. If a caller demands a human on the call, Alex says: "I'm creating a ticket — someone will reach out within 24 hours via the email on file" and ends gracefully.

### Hard-escalation triggers inside Alex
Alex auto-creates a ticket and ends the call politely when:
- Caller says "human" / "agent" / "real person" twice
- Caller says "cancel" twice
- Caller mentions lawsuit, DMCA, copyright claim, or legal action
- Caller becomes angry or uses profanity
- Caller asks about anything illegal (DRM bypass, stream ripping)

### Owner mode
An owner mode exists so Jeff can bypass the customer service register when calling in. It requires a verification phrase stored only in the Vapi prompt itself — **do not document the phrase in plaintext anywhere**, including this playbook. If someone claims to be Jeff without the correct phrase, Alex stays in normal customer mode and does not leak owner behavior.

### AI disclosure
Alex does not lead with "I'm an AI." The greeting is warm and human-style: "Hi, I'm Alex from StemScriber — how can I help you today?" If a caller directly asks "are you a real person?" or "is this AI?", Alex is honest: "I'm an AI assistant — but I'm here to help you find what you need." Legal requires honesty on direct question; no unsolicited disclosure.

---

## 4. Escalation Process

**Jeff handles all human-touch support.** Alex handles phone triage.

Escalate to Jeff via **email (support@stemscriber.com) + SMS summary** (never a phone call to 803-414-9454) for:
- Security issues (unauthorized access, data exposure)
- Data loss (user's uploaded songs or saved work gone)
- Legal concerns (DMCA, copyright claims)
- Payment system errors (double charges, failed refunds)

### Logging
- Log every support interaction in the Google Sheet tracker
- Fields: date, user name, channel (voice/text/email/form), issue type, summary, status, resolution
- Tag with priority: P0/P1/P2/P3

---

## 5. Feature Request Handling

1. **Thank them** — always. They took time to share an idea.
2. **Log it** — add to the StemScriber ideas list with:
   - Who requested it
   - Date
   - Description (in their words)
   - Your assessment of priority/effort
3. **Be honest** — "We're a small team" is fine. Don't promise timelines you can't keep.
4. **Follow up** — when you ship something a user requested, text/email them personally. "Hey, remember that feature you asked for? It's live." This builds loyalty like nothing else.
5. **Monthly review** — review the full request list, look for patterns, prioritize.

---

## 6. Bug Report Handling

### Priority Levels

| Priority | Description | Target Fix Time | Example |
|---|---|---|---|
| **P0** | App is unusable | 24 hours | Can't login, upload crashes server, all stems silent |
| **P1** | Major feature broken | 48 hours | Practice mode won't play, chord detection returns nothing |
| **P2** | Minor issue | Next release | Waveform display glitch, wrong icon showing |
| **P3** | Cosmetic | Backlog | Alignment off by a pixel, typo in UI |

### Bug Report Workflow
1. **Reproduce it** — try to hit the same issue yourself
2. **Log it** with: steps to reproduce, expected vs actual behavior, browser/OS, priority level
3. **Acknowledge to user** — within 12 hours, let them know you've seen it
4. **Fix or workaround** — within 48 hours for P0/P1
5. **Close the loop** — tell the user when it's fixed. "That bug you found? Squashed."

---

## 7. Current Product Features (what we support)

- **Stem separation** into vocals, guitar, bass, drums, piano, and other/backing vocals
- **Chord detection** — stem-aware, bass-root-first, with bleed simplification (~95% quality accuracy)
- **Practice mode** — speed and pitch control, section looping, stem isolation
- **Guitar tab detection** — Trimplexx model showing string and fret positions
- **Stem downloads** — individual files or full ZIP

Karaoke / lyrics display is **not offered** right now (licensing is being sorted). Don't promise it or tease it in support responses.

## 8. Current Pricing

| Tier | Price | Allowance |
|---|---|---|
| Free | $0 | 3 songs/month |
| Pro | $10/mo or $100/yr | Extended uploads, longer songs |
| Premium | $20/mo or $200/yr | Unlimited |

---

## 9. Tone Guide

### Do
- **Be a musician talking to a musician.** "What song were you working on?" not "Please provide the audio file identifier."
- **Use "we" and "I"** — "I'll look into this" / "We're working on improving that"
- **Be honest about limitations.** "Chord detection isn't perfect yet — we're improving it" beats "Our state-of-the-art AI provides industry-leading accuracy."
- **Show genuine curiosity** about what they're playing/learning.
- **Start casual.** "Hey [name]" not "Dear valued customer."
- **Sign off with your name.** "— Jeff"
- **Say sorry when something doesn't work.** No corporate deflection.

### Don't
- Use jargon they don't need: "the RoFormer transformer architecture" — just say "a different separation model."
- Over-promise: "We'll have that fixed by tomorrow" when you're not sure.
- Be defensive: "That's actually working as intended" when something clearly isn't great.
- Ghost them. Even if you don't have an answer yet, say so.
- **Lead with "AI."** Describe what the product does ("separates stems", "detects chords"), not how ("AI-powered"). Only confirm AI if directly asked.

### Good vs Bad Examples

**Bad:**
> Thank you for reaching out to StemScriber support. We have received your inquiry regarding stem separation quality. Our engineering team will review your case and provide an update at their earliest convenience. Ticket #4829 has been assigned.

**Good:**
> Hey Sarah, sorry the guitar stem came out muddy on that track. Some songs with heavy distortion are tough — the frequencies overlap a lot. Try switching to RoFormer in settings, it sometimes handles that better. What song is it? I'll run it through on my end too. — Jeff

---

## 10. Quick Reference Card

**Someone can't do something →** Help them do it. Offer to do it for them if needed.

**Something sounds bad →** Acknowledge, suggest alternatives, test it yourself.

**Something is broken →** Thank them, get repro steps, fix fast, close the loop.

**Someone wants a feature →** Thank them, log it, be honest about timeline.

**Someone wants money back →** Make it easy. 30-day guarantee, no friction.

**Someone is frustrated →** Empathize first, solve second. "I get it, that's annoying" before jumping to solutions.

**Someone calls the 844 line →** Alex handles it. If they need a human, ticket goes to email + SMS to Jeff — never a live transfer to Jeff's phone.

**When in doubt →** Be the kind of support you'd want to get as a musician trying to learn a song.
