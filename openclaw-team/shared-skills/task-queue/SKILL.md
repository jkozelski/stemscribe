---
name: stemscribe-task-queue
description: Inter-agent task queue for the StemScribe team. Agents create, read, and complete tasks for each other.
requires:
  - filesystem
---

# Task Queue Skill

This skill manages the inter-agent task queue in `shared-workspace/tasks/`.

## Creating a Task

Write a JSON file to `shared-workspace/tasks/` with this format:

```json
{
  "id": "task-{unix-timestamp}",
  "from": "AgentName",
  "to": "TargetAgent",
  "type": "content_opportunity|social_engagement|data_request|email_copy|report_request",
  "priority": "low|medium|high|urgent",
  "title": "Brief description of what needs to be done",
  "context": "Detailed context, data, links, and reasoning",
  "data": {},
  "created": "ISO-8601 timestamp",
  "status": "pending|in_progress|completed|cancelled",
  "completed_at": null,
  "result": null
}
```

Filename format: `{to}-{type}-{unix-timestamp}.json`
Example: `Wordsmith-content_opportunity-1707580800.json`

## Checking Your Tasks

On each heartbeat, read all `.json` files in `shared-workspace/tasks/` where
the `"to"` field matches your agent name and `"status"` is `"pending"`.

## Completing a Task

Update the JSON file:
- Set `"status"` to `"completed"`
- Set `"completed_at"` to current ISO-8601 timestamp
- Set `"result"` to a brief summary of what you did

## Priority Levels

- **low**: Do it when you have time (within 48 hours)
- **medium**: Do it on your next scheduled work session (within 24 hours)
- **high**: Do it on your next heartbeat (within 6 hours)
- **urgent**: Do it immediately, interrupt current work
