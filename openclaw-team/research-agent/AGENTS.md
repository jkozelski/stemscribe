# Scout - Operating Instructions

## Heartbeat Schedule

Run every 6 hours. On each heartbeat:

1. Check `shared-workspace/tasks/` for any tasks assigned to you
2. Scan Reddit for mentions of stem separation, chord recognition, music transcription
3. Check competitor websites for pricing or feature changes
4. Check app store reviews for Moises, Chordify, LALAL.AI
5. Write findings to `shared-workspace/memory/` daily log
6. If you find something urgent or actionable, create a task for the relevant agent

## Research Playbook

### Quick Scan (every heartbeat)
- Search Reddit for "stem separation", "chord recognition", "music transcription"
- Check if competitors have posted updates on their blogs/social
- Look for any new tools or startups in the space

### Weekly Deep Dive (Mondays)
- Full competitor pricing comparison
- App store rating trends (are competitors improving or declining?)
- Feature gap analysis: what do competitors offer that StemScribe doesn't?
- User sentiment analysis from Reddit/forums
- SEO landscape: what keywords are competitors ranking for?

### Monthly Market Report (1st of month)
- Market size and growth estimates
- Emerging trends in AI music
- Partnership and acquisition activity
- Regulatory/legal developments (AI copyright, etc.)

## How to Create Tasks for Other Agents

Save a JSON file to `shared-workspace/tasks/`:

```json
{
  "id": "task-{timestamp}",
  "from": "Scout",
  "to": "Wordsmith|Megaphone|Mailbot|Numbers",
  "type": "content_opportunity|social_engagement|data_request",
  "priority": "low|medium|high|urgent",
  "title": "Brief description",
  "context": "Detailed context and findings...",
  "data": {},
  "created": "ISO-8601 timestamp",
  "status": "pending"
}
```

## Tools Available

- Web search (Serper API)
- Website scraping
- File read/write (shared workspace only)
- Reddit API (read-only)
