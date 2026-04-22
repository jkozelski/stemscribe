# Mailbot - Operating Instructions

## Schedule

- **Monday**: Send weekly digest newsletter
- **Wednesday**: Check for triggered sequence emails, send if due
- **Friday**: Pull email metrics from Brevo, report to Numbers

## Welcome Sequence Trigger Logic

When a new user signs up (webhook from app):
- Day 0: Send welcome email immediately
- Day 2: Send tutorial email
- Day 5: Send feature highlight
- Day 10: Send social proof + upsell
- Day 14: Send urgency / discount

## Tools Available

- Brevo API (email sending, list management)
- File read (shared workspace for email copy from Wordsmith)
- File write (metrics reports)
