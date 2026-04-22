---
name: stemscribe-knowledge
description: Core knowledge about StemScriber product, pricing, features, and positioning. Shared by all agents.
---

# StemScriber Product Knowledge

## What Is StemScriber?

StemScriber is an AI-powered music tool that helps musicians learn songs by:
1. **Separating** any recording into individual instrument stems
2. **Transcribing** those stems to MIDI and tablature
3. **Recognizing** chord progressions with AI
4. **Transposing** to any key

## Core Technology

| Component | Technology | License |
|-----------|-----------|---------|
| Stem separation | Demucs v4 (htdemucs_6s) | MIT (Meta) |
| MIDI transcription | Basic Pitch | Apache 2.0 (Spotify) |
| Chord recognition | Custom TheoryAwareChordModel | Proprietary (ours) |
| Notation | Music21 + AlphaTab | BSD / proprietary |

## Stem Types (6-stem separation)
1. Vocals
2. Guitar
3. Bass
4. Drums
5. Piano/Keys
6. Other (synths, strings, etc.)

Plus **stereo splitting** for dual-instrument bands:
- Jerry Garcia (right) / Bob Weir (left) for Grateful Dead
- Dual drummers (Kreutzmann/Hart)

## Pricing

| Tier | Price | Key Features |
|------|-------|-------------|
| Free | $0 | 3 songs/month, 4-stem, standard quality |
| Premium | $4.99/mo | Unlimited songs, 6-stem, chords, MIDI export |
| Pro | $14.99/mo | All Premium + transposition, stereo split, tabs, priority |
| Premium Annual | $39.99/yr | Save 33% |
| Pro Annual | $119.99/yr | Save 33% |

## Key Differentiators (vs competitors)

1. **Price**: $4.99/mo vs Moises $18/mo - 73% cheaper
2. **Jazz expertise**: 99.9% chord accuracy on jazz standards (competitors are weak on jazz)
3. **Transposition**: All 12 keys instantly (worship teams love this)
4. **Jam band support**: Only tool that separates dual guitars (Jerry/Bob)
5. **Multi-genre**: Expanding to Bluegrass, Rock, Country (underserved by competitors)
6. **Chord + stems together**: Most tools do one or the other, not both

## Target Audiences

1. **Jazz musicians** - Learning standards, need chord charts
2. **Worship teams** - Weekly charts in different keys
3. **Music students** - Ear training, transcription homework
4. **Guitar/bass players** - Learn specific parts from recordings
5. **Producers/remixers** - Clean stems for sampling
6. **Jam band fans** - Grateful Dead, Phish, Allman Brothers

## Brand Voice Guidelines

- Speak as a fellow musician, not a tech company
- Enthusiastic but authentic - never hype
- Educational and helpful first, promotional second
- Use music terminology naturally
- Reference specific genres and artists (jazz standards, Dead tunes, etc.)
- Never say: "revolutionary", "game-changing", "leverage AI", "unlock the power"
- Do say: "learn songs faster", "hear every part", "practice with isolated stems"

## Competitor Quick Reference

| Competitor | Price | Weakness StemScriber Exploits |
|-----------|-------|------------------------------|
| Moises | $3.99-$18/mo | Expensive pro tier, weak on jazz chords |
| Chordify | $3.49-$6.99/mo | No stem separation, chords only |
| LALAL.AI | $18-$300 packs | No subscription, no chords, no transcription |
| RipX | $99-$199 one-time | Desktop only, no chord recognition |

## Legal Status

- All open-source dependencies properly licensed (MIT, BSD, Apache 2.0)
- Users responsible for copyright of uploaded audio
- DMCA safe harbor protections in place
- Chord progressions are not copyrightable (legal precedent)
- Same legal model as Moises and Chordify (both profitable)
