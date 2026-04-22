# StemScribe — Lawyer Call Prep Sheet

**Jeff Kozelski | Kozelski Enterprises LLC (SC) | stemscriber.com**
**jeff@tidepoolartist.com | 803-414-9454**

---

## What StemScribe Does (30-second pitch)

Music practice app. Users upload a song → AI separates it into individual instruments (vocals, guitar, bass, drums, piano) → they can mute/solo instruments, see chord charts, practice along at any speed. Think "karaoke machine meets guitar teacher."

---

## Current Business State

- Live at stemscriber.com
- Solo founder, self-funded
- LLC registered in South Carolina
- Trademark filed: USPTO Class 9 + Class 42
- DMCA agent registered: DMCA-1070849
- Pricing: Free / Pro $10/mo / Premium $20/mo
- Stripe payments live
- 11 beta testers, 3 active
- ~24 songs processed in library
- Infrastructure: $8/mo VPS + ~$0.06/song cloud GPU

---

## 4 Things I Need Legal Guidance On

### 1. YouTube URL Downloading
**What we do:** Users can paste a YouTube URL → we use yt-dlp (open source tool) to download the audio → process it into stems.

**My concern:** YouTube ToS prohibits downloading. Moises (competitor with 70M users) refuses to do this. Is this a real liability at my scale? When would I need to remove it? Can I keep it in beta and remove before public launch?

**Questions for lawyer:**
- Is this a ToS violation or a legal violation?
- Does scale matter? (11 users vs 10,000)
- Could I accept URLs but have the USER's browser download it client-side instead?
- What's the actual enforcement risk from Google/YouTube?

---

### 2. Chord Chart Library (2,665 songs)
**What we did:** Scraped free (not Pro) chord charts from Ultimate Guitar. These are user-submitted chord interpretations — someone typed "E7#9, G, A" and uploaded it. We transformed everything into our own JSON format, removed all attribution and UG references. Stored as static files, served as "StemScribe Library."

**My concern:** The chords themselves aren't copyrightable (Spirit v. Page/Plant confirms this). But scraping past UG's ToS could be a contract issue. We also stripped attribution.

**Questions for lawyer:**
- Are chord progressions (E7#9 → G → A) copyrightable? (I believe no)
- Is the specific text arrangement copyrightable? (chords above lyrics)
- Is scraping public web pages a legal issue or just a ToS issue?
- If I delete the files and recreate the same chords by hand/AI, is that clean?
- What about using AI to detect chords from audio instead? (we already can)

---

### 3. Server-Side Stem Caching
**What we do:** When someone processes a song, we store the separated stems on our server. If another user submits the same URL, they get cached results instantly (no reprocessing needed). Saves ~$0.06/song in GPU costs.

**My concern:** UMG v. MP3.com — they paid $53M for server-side copies of copyrighted music. Our stems are derivative works from copyrighted songs stored on our server and served to multiple users.

**Questions for lawyer:**
- Is per-user caching okay? (only that user can access their stems)
- What about cross-user caching? (user A's stems served to user B)
- What's the right auto-deletion timeframe? (24 hours? 7 days? 30 days?)
- Does it matter that stems are separated instruments, not the full song?
- Would client-side-only storage (browser/device) solve this?

---

### 4. Songsterr API Usage
**What we do:** We pull guitar tablature and chord charts from Songsterr's public API (no API key needed). We also discovered their ChordPro CDN which serves chord-over-lyrics charts. We display this data in our practice mode.

**My concern:** Songsterr's ToS says commercial use requires written permission. We charge $10/mo.

**Questions for lawyer:**
- Should I contact Songsterr directly for a commercial license?
- Is displaying their data (with attribution) fair use for educational purposes?
- If they say no, what are the alternatives?
- Could I partner with them instead of licensing?

---

## Additional Legal Questions

### Stem Separation Itself
- Is AI stem separation of copyrighted audio legal?
- Is it transformative use (education/practice) or derivative work?
- Moises does this with 70M users and hasn't been sued — is that precedent?
- What about the pending RIAA v. Suno/Udio case?

### DMCA Compliance
- I have DMCA agent registered and 6 legal pages. Am I fully compliant?
- Do I need a repeat infringer policy? (I don't have one)
- Does the YouTube feature disqualify me from safe harbor?

### Business Structure
- Is my SC LLC sufficient or do I need a different structure?
- Should I have separate entities for IP vs operations?
- When should I get business insurance (E&O/cyber)?

### Trademark
- I filed USPTO Class 9 + Class 42. Is that sufficient?
- Should I be monitoring for infringement?
- Do I need international trademark protection?

### Future Plans
- I want to add a "what instrument do you play?" feature to personalize the experience
- I want to expand the chord library to 10,000+ songs
- I want to offer AI-generated drum and piano notation
- At what user count do I need to worry about licensing?

---

## What I Want From This Call

1. **Priority list** — what do I need to fix NOW vs what can wait?
2. **The Moises model** — how do I structure my app like Moises to minimize risk?
3. **Cost estimate** — what will it cost to get fully legal?
4. **Timeline** — what needs to happen before I do a public launch?
5. **Ongoing relationship** — can you be my ongoing counsel as I grow?

---

## Competitors for Reference

| App | Users | Model | Sued? |
|-----|-------|-------|-------|
| Moises | 70M | Upload only, ephemeral, no URLs | No |
| Chordify | Millions | Analyzes URLs, no audio storage | No |
| Yousician | Millions | Fully licensed catalog | No |
| AudioShake | B2B | Licensed by Warner/Universal/Sony | No |
| Jammit | ~100K | Licensed multitracks | Shut down (couldn't afford licenses) |

---

## Documents I Can Send Before the Call

- Link to the app: stemscriber.com
- Current Terms of Service: stemscriber.com/terms.html
- Current Privacy Policy: stemscriber.com/privacy.html
- Current DMCA Policy: stemscriber.com/dmca.html
- Full legal brief: ~/stemscribe/docs/legal-brief.md (can email)
