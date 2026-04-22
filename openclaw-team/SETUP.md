# StemScriber OpenClaw Agent Team - Setup Guide

## Overview

This is a **5-agent AI team** running on OpenClaw that handles all research
and marketing for StemScriber. Each agent runs as its own OpenClaw instance
with dedicated skills and a shared workspace for inter-agent communication.

## The Team

| Agent | Role | Always-On? | Skills |
|-------|------|-----------|--------|
| **Scout** | Research & Intelligence | Yes (heartbeat: 6hr) | Web search, competitor monitoring, Reddit scanning |
| **Wordsmith** | Content & SEO | On-demand + weekly cron | Blog writing, SEO optimization, keyword research |
| **Megaphone** | Social Media & Community | Yes (heartbeat: 4hr) | Reddit posting, social scheduling, engagement |
| **Mailbot** | Email Marketing | Weekly cron | Email sequences, newsletters, drip campaigns |
| **Numbers** | Analytics & Reporting | Daily cron | Metrics tracking, revenue reports, KPI dashboards |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 YOUR MACHINE / VPS                    │
│                                                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │  Scout   │  │ Wordsmith│  │Megaphone │           │
│  │ (research)│  │ (content)│  │ (social) │           │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │                 │
│       ▼              ▼              ▼                 │
│  ┌─────────────────────────────────────────┐         │
│  │         SHARED WORKSPACE                 │         │
│  │  tasks/    - Task queue (JSON files)     │         │
│  │  reports/  - Research reports            │         │
│  │  content/  - Blog posts, social copy     │         │
│  │  memory/   - Shared context & history    │         │
│  └─────────────────────────────────────────┘         │
│       ▲              ▲                               │
│       │              │                               │
│  ┌────┴─────┐  ┌────┴─────┐                         │
│  │ Mailbot  │  │ Numbers  │                         │
│  │ (email)  │  │(analytics)│                         │
│  └──────────┘  └──────────┘                         │
│                                                       │
│  Messaging: Signal / Telegram / Discord               │
│  LLM: Claude Sonnet 4.5 (via Anthropic API)          │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

1. Install OpenClaw: https://docs.openclaw.ai/getting-started
   ```bash
   # macOS
   brew install openclaw

   # or from source
   git clone https://github.com/openclaw/openclaw.git
   cd openclaw && npm install && npm link
   ```

2. Get API keys:
   - Anthropic API key (Claude) - required
   - Serper API key (web search) - free 2,500 searches at serper.dev
   - Brevo API key (email) - free 300 emails/day at brevo.com

3. Set up messaging (pick one):
   - Telegram bot (easiest)
   - Signal
   - Discord

## Quick Start

```bash
# 1. Copy this team to your OpenClaw workspace
cp -r ~/stemscribe/openclaw-team/* ~/.openclaw/

# 2. Set up credentials
echo "YOUR_ANTHROPIC_KEY" > ~/.openclaw/credentials/anthropic
echo "YOUR_SERPER_KEY" > ~/.openclaw/credentials/serper
chmod 600 ~/.openclaw/credentials/*

# 3. Register all agents
openclaw agents add research-agent --name "Scout"
openclaw agents add marketing-agent --name "Wordsmith"
openclaw agents add community-agent --name "Megaphone"
openclaw agents add email-agent --name "Mailbot"
openclaw agents add analytics-agent --name "Numbers"

# 4. Verify setup
openclaw doctor --fix

# 5. Start the team
openclaw agents start --all

# Or start individually:
openclaw agents start research-agent
openclaw agents start marketing-agent
```

## Inter-Agent Communication

Agents communicate through the shared workspace:

1. **Task Queue** (`shared-workspace/tasks/`):
   - Agents create JSON task files for other agents
   - Example: Scout finds a trending topic → creates task for Wordsmith to write about it

2. **Shared Memory** (`shared-workspace/memory/`):
   - Daily logs that all agents can read
   - Competitor updates, user feedback, content calendar

3. **@mentions**:
   - In the messaging interface, use @Scout, @Wordsmith, etc. to direct requests

## Monthly Cost Estimate

| Item | Cost |
|------|------|
| Claude API (5 agents, moderate use) | $15-40/mo |
| Serper web search | Free tier (2,500/mo) |
| Brevo email | Free tier (300/day) |
| Buffer social scheduling | $5/mo |
| **Total** | **$20-45/mo** |

vs. hiring humans: $15,000-30,000/month for equivalent team
