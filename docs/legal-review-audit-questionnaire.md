# Independent AI audit — questionnaire for the StemScriber Legal Review Package

**Purpose:** Adversarial second-look on the legal-review-package before it goes to retained counsel. Goal is to catch errors, gaps, weak positions, or missing authorities that the original drafting (Jeff + Claude) may have missed.

**How to run:**
1. Open ChatGPT (or another LLM that isn't Claude — independent perspective is the point).
2. Upload or paste both files: `legal-review-package-2026-04-30.md` AND this questionnaire.
3. Use the prompt below.
4. Capture the response and apply edits to v1.1 of the package before it goes to Alexandra.

---

## Prompt to use

> I'm submitting a pre-launch legal review package for a consumer music-transcription web app called StemScriber. The package will be reviewed by retained music-industry counsel. Before it goes to her, I want an independent adversarial audit. The package is attached. Below are specific questions I need you to address — please answer each one numbered and directly, not as flowing narrative. Be skeptical. Where you find errors or weak positions, be concrete and specific. Cite your reasoning. If you don't know something, say so explicitly rather than guessing.

---

## Questions to address

Answer each numbered item in the response. Don't reorder, don't merge.

### Citation and authority verification

1. **Citation accuracy.** Walk through every named case, statute, and book citation in the package. For each, does the cited authority actually support the position attributed to it? Flag any where the citation is wrong, the authority doesn't say what it's claimed to say, or the case is being misrepresented.

2. **Missing primary authority.** Are there cases, statutes, or regulations directly relevant to a US consumer music-transcription tool that the package omits? Particularly look for: AI training-data cases, music-IP cases, platform-liability cases, DMCA §512 case law, §230 case law, derivative-work cases, transformative-use cases.

3. **Stale citations.** Any cited case or position that has been superseded, distinguished, or overruled by a more recent decision (2024-2026)? Flag and provide the more current authority.

### Position strength

4. **Weakest position.** Of the 8 decisions in §3 of the package, which is most vulnerable to plaintiff challenge? Walk through the strongest argument against it.

5. **Strongest position.** Which is best-supported and least likely to be successfully challenged?

6. **Plaintiff's-lawyer attack surface.** If you were drafting a complaint against StemScriber on behalf of a major label, what theory of liability would you pursue first? Where in the package's architecture or posture would you find purchase?

### Architectural gaps

7. **Claim vs. implementation.** The package claims certain architectural guardrails (per-URL attestation, no public caching, pass-through processing, audit-trail persistence, retention limits, etc.). Are any of these claims internally contradicted or unsupported by the rest of the package? Are any operationally fragile in ways the doc doesn't address?

8. **Single points of failure.** What single technical or operational failure would most damage the legal posture (e.g., DMCA-agent email goes down, retention policy fails to run, attestation log isn't actually persisted)? Are any of these adequately addressed in the package?

### Regulatory blind spots

9. **What's not covered that should be.** The package explicitly puts privacy/GDPR/CCPA, payments, employment, tax, and insurance out of scope (§7). For a consumer SaaS launching publicly in the US, are any of those *actually* out of scope, or are there issues we should pull back in? Specifically:
   - State-by-state consumer privacy laws (CCPA, NY SHIELD, Colorado, Virginia, etc.)
   - FTC click-to-cancel rule for subscription billing
   - ADA/WCAG accessibility for the web app
   - State-by-state automatic renewal disclosure requirements
   - Children's online privacy (COPPA) — is StemScriber under-13 aware?

10. **Cross-border issues.** If StemScriber accepts users from outside the US (likely by default — there's no geofence), what additional exposure exists that the package doesn't address?

### Industry comparison accuracy

11. **Peer competitor claims.** The package asserts that Klangio, Moises, Chordify, and AudioShake all accept YouTube URLs and operate under similar user-warranty postures. Verify this. If any claim is wrong or stale, flag it.

12. **Klangio EU TDM defense.** The package suggests Klangio's Germany jurisdiction gives them additional cover under EU DSM Directive Article 4. Is this characterization accurate? Are there reasons it's an apples-to-oranges comparison?

### Re-evaluation triggers

13. **Watch list completeness.** §5 names *Bartz v. Anthropic*, *Concord v. Anthropic*, *NYT v. OpenAI*, *Getty v. Stability* as the active AI cases whose outcomes would shift posture. Are there cases that aren't named that should be? Any named that are noise?

14. **Trigger calibration.** Are the re-evaluation triggers well-calibrated, or are they too vague to operationalize ("major shift in scale" — at what threshold? "publicly changes enforcement posture" — measured how?)?

### Internal consistency

15. **Self-contradictions.** Does the package contradict itself anywhere? Particularly check that the architectural diagram (§4) matches what's described in §3, that the ToS clauses in Appendix A actually say what the prose attributes to them, and that the attestation modal copy in Appendix B matches operational claims.

### Tone and effectiveness for external counsel

16. **Professional tone.** Is the package well-pitched for a music-industry attorney? Too defensive? Too presumptuous? Disrespectful in any way? Where could the framing be improved without changing the substance?

17. **What would the attorney flag first.** If you were the attorney receiving this package, what's the first concern you'd raise back to the founder? (We want to anticipate this and address it preemptively if possible.)

### Open question

18. **What's not asked here that should be.** What question should I be asking that this questionnaire doesn't include? What would a sophisticated reviewer want me to interrogate that I haven't?

---

## Response format requested

Numbered answers, one per question, in order. For each:
- A concise direct answer (1-3 sentences for simple items; longer for complex ones)
- Specific citation or evidence where applicable
- A clear "no issue found" if that's genuinely the answer (don't pad)
- A red/yellow/green tag at the end of each answer indicating severity:
  - 🔴 RED — must address before sending to counsel
  - 🟡 YELLOW — should address, not blocking
  - 🟢 GREEN — no issue found / package handled this well

End with an overall summary: how many reds, how many yellows, top 3 priorities for v1.1 revision.

---

*If anything in your answer is uncertain, please say so explicitly. The goal is calibrated honesty, not encouragement.*
