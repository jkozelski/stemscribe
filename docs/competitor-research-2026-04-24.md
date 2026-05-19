# Competitor Research — Audio → Chord Chart / Notation Services
**Date:** 2026-04-24
**Scope:** Klangio, Chordify, Moises, Ultimate Guitar, AnthemScore, Songsterr, Mixed In Key
**Method:** Public web research only (blogs, pricing pages, arXiv, ISMIR archives, Reddit/TalkBass/Trustpilot, LinkedIn, company docs)

---

## Comparison Matrix

| Company | Pipeline (inferred/published) | Output formats | Chord vocabulary | Price (USD) | Quality claim / reality |
|---|---|---|---|---|---|
| **Klangio (klang.io)** | Polyphonic note detection via instrument-specific deep models; published work on hierarchical frequency-time transformers, procedural data gen, dual-task monophonic singing transcription. No public stem separator. | MIDI (quant + unquant), MusicXML, PDF sheet, Guitar Pro | Lead-sheet "chord symbols" + note-level; extent of extensions not published | $24.99/yr per app (promo), $49.99/yr Universe bundle; 20-sec free demo | No public accuracy numbers; marketing claim "high-accuracy"; recent arXiv papers show active R&D |
| **Chordify** | Deep NN trained on spectrograms → chords; internal beat tracker, tuning estimator, polyphonic F0. No stem separation disclosed. Large-vocabulary ACE academically contributed (CASD dataset). | On-screen chord chart, MIDI (Premium), PDF (Premium). No MusicXML, no tabs. | Primarily triads + some 7ths; users repeatedly report extensions/jazz fail | Basic free (ad-supported); Premium ~$3.49/mo (yearly); Premium+Toolkit from $2.25/mo (yearly) | No published accuracy; Trustpilot/TalkBass/Quora users report key errors, jazz failure, "hilariously bad" funk |
| **Moises** | Proprietary stem separator (started on Deezer Spleeter, now in-house models); chord detector runs on the mix. Three difficulty tiers (Easy/Medium/Advanced). | On-screen chords + lyrics, guitar shapes, capo-adjusted. No MusicXML/PDF sheet export in chord product. | Advanced mode claims jazz/bossa extensions; users report Cmaj9#11 mis-detected as Cmaj7, slash chords "almost always wrong" | Free (5 tracks, 1 min chord); Premium $3.99/mo ($2.99 annual); Pro $9.99/mo | Third-party review cites "85–90% on pop/rock, 40–50% on jazz" (StemSplit 2026) |
| **Ultimate Guitar** | NOT an audio→chord-detection product. Crowdsourced tab library + Practice Mode. No public evidence of a shipped audio-upload AI chord detector. Muse Group also owns MuseScore/Audacity. | User-contributed tabs, Official Tabs, chord sheets | Whatever humans transcribe | Pro $3–4/mo, Plus ~$10/mo | Quality depends on crowd; "Official Tabs" are licensed/curated |
| **AnthemScore** (Lunaverus) | CNN trained on "millions of samples"; outputs note grid → MIDI/MusicXML/PDF. Desktop-only (Win/Mac/Linux). | MIDI, MusicXML, PDF | Note-level, not chord-symbol | $45 one-time (no subscription); 30-day free trial | Marketing claims "high accuracy"; community (MuseScore forums) notes quality drops on complex / distorted audio |
| **Songsterr** | Crowdsourced tablature + newly released AI tab generator (YouTube link → tab). Underlying AI not disclosed. | Interactive tab, notation | Note/tab, not chord-chart centric | Free w/ 20-sec AI limit; Plus $10/mo for full length | Heise review: "fairly accurate"; TalkBass users: ignores palm muting, misses position shifts |
| **Mixed In Key** | Key + BPM + cue points; developed with data scientists in UK/CH/US. Camelot system. | ID3 metadata; key/BPM tags | Key only (not chords) | MIK 11: one-time ~$58–$97 depending on tier | Dubspot test: outperformed Beatport; "~10% more accurate than next best" per marketing |

---

## Klangio (klang.io)

- **Founders / team:** Sebastian Murgul (CEO, also PhD candidate at KIT) and Alexander Lüngen (CTO), founded 2018 in Karlsruhe. Team of ~10 including 3 dedicated AI researchers. ([About page](https://klang.io/about-us/))
- **Pipeline:** Instrument-specific deep models. Murgul has published on arXiv in 2025: *"Joint Transcription of Acoustic Guitar Strumming Directions and Chords"* ([arXiv:2508.07973](https://arxiv.org/abs/2508.07973)) and *"Exploring Procedural Data Generation for Automatic Acoustic Guitar Fingerpicking Transcription"* ([arXiv:2508.07987](https://www.arxiv.org/abs/2508.07987)) — the latter synthesizes training data via MIDI + extended Karplus-Strong physical modelling + audio augmentation. Klangio also open-sourced [DTMST (Dual-Task Monophonic Singing Transcription)](https://github.com/klangio/dtmst).
- **Stem separation?** Not disclosed. Transcription Studio's "Rock Mode" does claim separate tracks per instrument — likely an internal separator — but no model is named. ([MusicRadar](https://www.musicradar.com/music-tech/klang-io-says-transcription-studio-is-the-worlds-first-ai-music-tool-that-can-transcribe-multiple-instruments-simultaneously))
- **Outputs:** MIDI, MusicXML, PDF notation, Guitar Pro. Integrates with MuseScore/Sibelius/Finale/Logic/Ableton.
- **Pricing:** During "Klangio Days" promo: $24.99/yr per single app; $49.99/yr Universe bundle. ([Klangio Days](https://klang.io/klangio-days/))
- **Positioning:** "Audio → sheet music / tabs / MIDI in seconds." Heavy emphasis on notation quality and DAW/notation-editor integration.
- **Gap vs chord-chart product:** Klangio primarily does **note-by-note transcription**. Chord symbols appear on lead sheets, but vocabulary/extension support not documented.
- **Uncertainty flagged:** No public accuracy benchmarks on chord extensions; I could not find what their chord-symbol post-processor looks like.

## Chordify

- **Founder:** Bas de Haas, PhD Utrecht University (music informatics, 2011), co-founded Chordify with Tijmen Ruizendaal. Beta debuted at ISMIR 2013. ~40 employees, "millions of users monthly." ([ISMIR 2020 virtual booth](https://github.com/chordify/ISMIR2020-industrybooth/blob/master/chordify_virtual_booth.md))
- **Pipeline (published):** "Deep neural networks trained on spectrograms" → chord labels. They maintain in-house beat tracking, tempo, tuning, audio fingerprinting, and polyphonic F0. ([Algorithm explainer](https://chordify.net/pages/technology-algorithm-explained/))
- **Stem separation?** No public evidence. Detection appears to run on full mix.
- **Published research:** "Annotator Subjectivity in Harmony Annotations of Popular Music" (JNMR 2019); beat annotation paper at ISMIR 2019; released the [CASD dataset](https://github.com/chordify/CASD) (multiple expert references per song).
- **Chord vocabulary:** Not officially published. User reports on [TalkBass](https://www.talkbass.com/threads/regarding-chordify.1439139/) and [Basschat](https://www.basschat.co.uk/topic/204574-chordifyis-it-legit/) say major/minor triads dominate, 7ths occasional, jazz extensions and slash chords frequently fail.
- **Section labeling:** **Not automatic.** A user request thread on Chordify Support has been open for 6 years without implementation. ([Chordify Support feature request](https://support.chordify.net/hc/en-us/community/posts/360005503397-Allow-users-to-define-sections-Intro-Verse-Chorus-etc))
- **Outputs:** On-screen chord chart; Premium adds MIDI download + PDF print. No MusicXML, no tabs.
- **Pricing:** Basic free (ads); Premium ~$3.49/mo yearly; Premium + Toolkit from $2.25/mo. ([Chordify Premium](https://chordify.net/premium))
- **Reality check:** Trustpilot scoring is mixed — users love breadth of catalog, complain about accuracy on anything non-pop. Quora example: wrong key transposition on Jason Mraz "I'm Yours." ([Quora](https://www.quora.com/Is-the-Chordify-app-accurate-Im-yours-by-Jason-Mraz-is-in-key-of-C-major-but-the-app-shows-chords-in-key-of-B-major))
- **Uncertainty flagged:** Exact chord vocabulary and current model architecture undisclosed post-2020; inference based on user reports and their academic output.

## Moises

- **Founders:** Geraldo Ramos (CEO), Eddie Hsu, Jardson Almeida, founded 2019 in Brazil. Parent brand is now "Music.AI" (B2B API side). ([BBH profile](https://www.bbh.com/us/en/insights/capital-partners-insights/standing-out-from-the-noise-a-conversation-with-geraldo-ramos-and-eddie-hsu-co-founders-of-musicai.html))
- **Stem separator origin:** Started on Deezer's Spleeter (2019), migrated to in-house models. ([Music Ally 2019](https://musically.com/2019/11/22/moises-makes-deezers-spleeter-audio-separation-tool-user-friendly/))
- **Chord detector:** Three modes — Easy / Medium / Advanced. Advanced mode unlocks jazz/bossa extensions. Model architecture not published.
- **Stem-aware chord detection?** Help docs and marketing suggest chord detection runs on the original mix, not on isolated stems, though stems are available in the same product. Not definitively confirmed.
- **Outputs:** On-screen chords synced to timeline + lyrics; capo mode; guitar diagrams. No MusicXML / PDF sheet-music export through the chord feature.
- **Pricing:** Free (5 tracks, 1-min chord detection), Premium $3.99/mo (~$2.99 annual), Pro $9.99/mo (Hi-Fi, VST plugins, 180-min uploads, API).
- **Quality claim vs reality:** Third-party [StemSplit review](https://stemsplit.io/blog/moises-ai-review) cites 85–90% on pop/rock, 40–50% on jazz/extensions. Reddit/forum threads corroborate: Cmaj9#11 → Cmaj7, slash chords frequently wrong.
- **Engineering hints:** [Music.AI research portal](https://music.ai) exists for B2B/API. 2025 "AI Studio" release introduces generative instrument synthesis — pivot toward creation, not just analysis. ([Oct 2025 release notes](https://moises.ai/blog/moises-news/improvements-latest-releases/))
- **Uncertainty flagged:** Whether stem separation is piped into chord detection is not publicly stated.

## Ultimate Guitar (Muse Group)

- **Context:** Owned by Muse Group (also MuseScore, Audacity, Hal Leonard). ([Scoring Notes](https://www.scoringnotes.com/news/muse-group-formed-to-support-musescore-ultimate-guitar-acquires-audacity/))
- **Audio → chord feature?** I could not confirm UG has shipped an audio-upload-to-AI-chord-chart feature comparable to Chordify/Moises. Their newer "Practice Mode" focuses on listening to the user play along with existing tabs, not transcribing from audio. ([Muse Group post](https://www.mu.se/posts/ultimate-guitar-launches-practice-mode))
- **Core model:** Massive crowdsourced tab library + licensed "Official Tabs."
- **Pricing:** Pro ~$3–4/mo, Plus ~$9.99/mo (varies by region/promo).
- **Uncertainty flagged (important):** The user asked about a "recent AI chord detection" feature. My research surfaced no public product page / announcement for an AI audio-to-chord feature on Ultimate Guitar itself. If this exists, it is either very new or only exposed in mobile beta — I cannot confirm it from public sources.

## AnthemScore (Lunaverus)

- **Pipeline:** Desktop CNN trained on "millions of samples," converts audio into a note grid the user can edit and then export. ([Lunaverus](https://www.lunaverus.com/))
- **Outputs:** MIDI, MusicXML, PDF.
- **Pricing:** **$45 one-time**, 30-day free trial. No subscription.
- **Chord product?** Note-level transcription, chord symbols inferred from stacked notes in their editor — not a chord-chart product per se.
- **Reality check:** [MuseScore forum thread](https://musescore.org/en/node/279528) praises it for clean solo piano; users note degradation on distorted/complex audio.

## Songsterr

- **Product:** Primarily crowdsourced/licensed tab, paying royalties on content. ([Songsterr Plus](https://www.songsterr.com/plus))
- **Recent AI feature:** Audio-URL → guitar/bass/drum tabs via AI. Pipeline not disclosed. ([TalkBass thread](https://www.talkbass.com/threads/songsterr-recently-released-an-automated-ai-tab-writing-feature.1667303/), [Heise review](https://www.heise.de/en/news/Songsterr-uses-AI-to-create-fairly-accurate-guitar-tabs-from-your-own-recordings-10356152.html))
- **Pricing:** Free (20-sec AI limit); Plus $9.99/mo unlocks full-length AI generation + library.
- **Quality:** TalkBass users say palm muting + position shifts are missed; heise says "fairly accurate."

## Mixed In Key

- Key + BPM + cue points for DJs, not chord charts. Included only because user listed it. Algorithm undisclosed; marketing claims "~10% more accurate than next best" — corroborated by [Dubspot 2016 test](https://blog.dubspot.com/dubspot-lab-report-mixed-in-key-vs-beatport) against Beatport (68% vs MIK winner). Pricing: MIK 11 is one-time purchase. Not directly relevant to StemScriber's competitive frame.

---

## StemScriber Differentiation (against the set above)

**What StemScriber appears to do that none of these competitors publicly advertise:**

1. **Stem-aware chord detection as a pipeline primitive** — running chord detection on bass (for root) + separated harmony stem, then fusing with a full-mix chord classifier. Moises has stems AND chord detection, but public docs treat them as separate features on the timeline, not as a fused pipeline. Chordify has no stem separation at all. Klangio has multi-instrument transcription but outputs note-level per instrument, not a fused chord chart.
2. **Published internal chord-detection benchmarks and per-song truth-tracking** (your Alright / Peg ground-truth files) — no competitor posts head-to-head chord-level evaluations publicly.
3. **Dedicated guitar-tab model (Trimplexx-family CRNN) trained alongside chord detection** and wired into the same chart. Klangio has separate Guitar2Tabs; Moises has no tab output; Chordify has no tabs.
4. **DMCA-registered, consent-popup, session-gated UX** — none of the competitors researched publish a consent-on-every-upload flow.

**What all (or most) competitors do that StemScriber does not publicly claim:**

1. **MusicXML / Guitar Pro / PDF notation export** — Klangio, AnthemScore, Songsterr all ship this. Chordify has PDF print. Moises has PDF-less chord sheets. (Per your docs, `sheet-music-phase0-2026-04-24.md` exists, so this is in flight.)
2. **Native MIDI export of the full transcription** — Klangio, AnthemScore, Moises, Chordify Premium all offer.
3. **Massive catalog / pre-transcribed library** — Chordify and Songsterr pre-compute on well-known songs so users don't wait. StemScriber is on-demand only.
4. **Mobile app** — Chordify, Moises, Songsterr, UG all have native iOS/Android. StemScriber is web.
5. **DAW plugin / integration with notation editors** — Klangio ships a plugin.

**Failure modes compared to competitors:**

- **7ths preservation:** Chordify and Moises (basic tier) both lose 7ths in practice per user reviews. Moises Advanced mode claims to support extensions but community reports 40–50% accuracy there. Klangio's chord-symbol behaviour on 7ths is undocumented. StemScriber's recent 7ths fix puts it at parity or better than Chordify's typical output; head-to-head test needed.
- **Section labeling on uniform progressions:** Chordify has no automatic section labeling at all (6-year-old open feature request). Moises does not publicly label sections either. This is a lateral move for StemScriber — no competitor shipped a strong solution, so it's a green-field opportunity, but it's also why Chordify hasn't prioritized it.
- **Dense jazz bailing to root-only (Peg):** Moises publicly admits jazz accuracy drops to 40–50%; Chordify users report total failure. This is an industry-wide weakness, not a StemScriber-specific one. Peg at "97% root-only" is probably *worse* than Moises Advanced on the same song, but *better* than Chordify default.

---

## What StemScriber Could Borrow (1-paragraph, per user request)

Klangio's notation-export surface (MusicXML + Guitar Pro + DAW-editor compatibility) is the single highest-leverage capability the competitors share that StemScriber lacks shipped-to-user — and the `sheet-music-phase0` doc indicates you're already heading there. Chordify's CASD-style subjectivity dataset approach is worth mirroring internally: multiple reference annotations per song let you evaluate ambiguity rather than pretending there's a single correct answer, especially on the jazz cases. Moises' Easy/Medium/Advanced tiering is a clever UX hack for the 7ths-vs-triads trade-off — users pick their complexity tolerance rather than you silently choosing for them. Otherwise, no competitor has publicly solved the stem-aware + fused chord pipeline you're building, so the differentiation is real and defensible.

---

## Sources (consolidated)

- [Chordify algorithm explainer](https://chordify.net/pages/technology-algorithm-explained/)
- [Chordify ISMIR 2020 virtual booth](https://github.com/chordify/ISMIR2020-industrybooth/blob/master/chordify_virtual_booth.md)
- [Chordify CASD dataset](https://github.com/chordify/CASD)
- [Chordify Premium pricing](https://chordify.net/premium)
- [Chordify section-labeling feature request (6 yrs open)](https://support.chordify.net/hc/en-us/community/posts/360005503397-Allow-users-to-define-sections-Intro-Verse-Chorus-etc)
- [TalkBass Chordify accuracy thread](https://www.talkbass.com/threads/regarding-chordify.1439139/)
- [Basschat Chordify legitimacy thread](https://www.basschat.co.uk/topic/204574-chordifyis-it-legit/)
- [Klangio About/team](https://klang.io/about-us/)
- [Klangio Days pricing](https://klang.io/klangio-days/)
- [Klangio Transcription Studio MusicRadar](https://www.musicradar.com/music-tech/klang-io-says-transcription-studio-is-the-worlds-first-ai-music-tool-that-can-transcribe-multiple-instruments-simultaneously)
- [Murgul — Joint Transcription of Guitar Strumming (arXiv 2508.07973)](https://arxiv.org/abs/2508.07973)
- [Murgul — Procedural Data for Fingerpicking Transcription (arXiv 2508.07987)](https://www.arxiv.org/abs/2508.07987)
- [Klangio DTMST repo](https://github.com/klangio/dtmst)
- [Moises chord finder feature page](https://moises.ai/features/chord-finder/)
- [Moises advanced chord detection blog](https://moises.ai/blog/latest/advanced-chord-detection/)
- [Moises AI Review (StemSplit 2026)](https://stemsplit.io/blog/moises-ai-review)
- [Moises origin / Spleeter — Music Ally 2019](https://musically.com/2019/11/22/moises-makes-deezers-spleeter-audio-separation-tool-user-friendly/)
- [BBH interview — Ramos/Hsu Music.AI](https://www.bbh.com/us/en/insights/capital-partners-insights/standing-out-from-the-noise-a-conversation-with-geraldo-ramos-and-eddie-hsu-co-founders-of-musicai.html)
- [Muse Group / Ultimate Guitar — Scoring Notes](https://www.scoringnotes.com/news/muse-group-formed-to-support-musescore-ultimate-guitar-acquires-audacity/)
- [Ultimate Guitar Practice Mode announcement](https://www.mu.se/posts/ultimate-guitar-launches-practice-mode)
- [AnthemScore / Lunaverus](https://www.lunaverus.com/)
- [Songsterr AI tabs (heise review)](https://www.heise.de/en/news/Songsterr-uses-AI-to-create-fairly-accurate-guitar-tabs-from-your-own-recordings-10356152.html)
- [Songsterr Plus pricing](https://www.songsterr.com/plus)
- [Mixed In Key accuracy vs Beatport — Dubspot](https://blog.dubspot.com/dubspot-lab-report-mixed-in-key-vs-beatport)
- [Quora — Chordify key error "I'm Yours"](https://www.quora.com/Is-the-Chordify-app-accurate-Im-yours-by-Jason-Mraz-is-in-key-of-C-major-but-the-app-shows-chords-in-key-of-B-major)
- [Bas de Haas profile — Chordify CEO / Utrecht](https://www.linkedin.com/in/wbdehaas/)
