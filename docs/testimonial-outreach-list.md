# Testimonial Outreach List

**Goal:** Get the original 10 to actually use the app + add 10-15 pro musicians for free lifetime in exchange for a quote. **Even minimal use is fine** — bar is "tried it once, gave us a quote."

**Status:** in progress, contacts being pulled from Contacts.app

---

## The Original 10 (Mar 17 batch — codes already valid in beta_codes.json)

**Source of truth:** Jeff's "Charleston Musicians" Google Sheet (164 rows) — better data than Apple Contacts for every name on this list.

| # | Name | Code | Status | Phone | Email | Bands |
|---|------|------|--------|-------|-------|-------|
| 1 | Tom Eden | STEM-0A1C-5D56 | not redeemed | 424-406-0848 (cell) / 860-377-1690 | tomjeden@gmail.com | King Hippo (Bass, Drums) |
| 2 | Stefan Custodi | STEM-EDDA-D7E7 | not redeemed | 803-586-1515 | stefancustodi@gmail.com | (not in CM sheet — own Musician tag) |
| 3 | Will Evans | STEM-53B2-2B70 | not redeemed | 803-960-9455 | funkguitarwill@gmail.com | Kozelski (Guitar) — also Stereo Reform |
| 4 | Tim Davis | STEM-9A06-487A | not redeemed | 803-237-2716 | kodamusic21@gmail.com | KoDa (Vocals) |
| 5 | **Fuller Conden** | STEM-6188-8CA6 | ✅ redeemed 3/18 | 347-624-1948 | fullercondon@gmail.com | Spare Kings (Upright Bass) |
| 6 | Bobby Hogg | STEM-2B24-8B51 | not redeemed | 843-579-2869 | robertghogg@gmail.com | KoDa, The Reckoning, TAB (Bass, Guitar) |
| 7 | **Elliott Genther** | STEM-9ACF-F9D1 | ✅ redeemed 3/18 | 843-513-2528 | _none on file_ | Green Levels |
| 8 | Stephen Jenkins | STEM-5676-0FCD | not redeemed | 919-801-5636 | stephenejenkins@hotmail.com | Spare Kings (Keys, Guitar, Vocals) |
| 9 | Wes Powers | STEM-4B36-0EBE | not redeemed | 828-443-7512 | wpowers70@bellsouth.net | Kozelski, The Reckoning, Sailin' Shoes (Drums) |
| 10 | Frank Lewis | STEM-CFC1-09EA | not redeemed | 843-345-5044 | _none_ | "Possible Student" — verify if pro |

> **Notes:**
> - beta_codes.json has #9 as "West Powers" — typo, real name is "Wes Powers". Should fix.
> - Bobby Hogg goes by "robert hogg" lowercase in Apple Contacts now — but the sheet's "Bobby Hogg" with `robertghogg@gmail.com` IS him. Use that.
> - #10 Frank Lewis labeled "Possible Student" in Contacts — likely not a serious target, verify before counting.
> - Stefan Custodi is the only one not in the Charleston Musicians sheet.

---

## New Pro Musicians — Lifetime Comp (10–15 to add)

| # | Name | Phone | Email | Band/Instrument | Status |
|---|------|-------|-------|-----------------|--------|
| _candidates from contacts scan, awaiting Jeff approval_ |

**Mechanism:** new code label `lifetime-comp`. When they sign up, manually mark their Stripe account as comp.

---

## Outreach Email Templates (drafts)

### Nudge — for the 8 unredeemed from original 10

```
Subject: Hey — your StemScriber code is still good

[name],

Sent you this back in March, never heard back so figured I'd try again. No pressure — even messing around with it for 5 minutes would help me a ton at this stage.

Beta code: [CODE]
Site: https://stemscriber.com

Upload any song, it splits the stems + auto-generates a chord chart. Honest impressions are worth more than polite ones.

— Jeff
```

### Pro lifetime ask — for new musicians

```
Subject: Want a free lifetime account in exchange for an honest quote?

[name],

Building a tool that splits any song into stems + auto-generates a chord chart. Soft launch June 20 at The Refinery (Watkins Glen 50 tribute show).

Want to give you a free lifetime account — no strings, just an honest one-line quote I can use on the site if you end up liking it. Even if you only try it once, that's fine.

Code: [CODE-LIFETIME]
https://stemscriber.com

— Jeff
```

---

## Action items

- [ ] Finish populating contact info from Contacts.app
- [ ] Generate 10–15 lifetime-comp codes
- [ ] Fix `West Powers` → `Wes Powers` in beta_codes.json
- [ ] Send nudge emails/texts to the 8 unredeemed
- [ ] Ask Fuller + Elliott directly for a testimonial quote
- [ ] Surface new musician candidates from Contacts.app to Jeff for approval
