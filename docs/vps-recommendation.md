# StemScriber VPS Hosting Recommendation

**Date:** March 19, 2026
**Status:** Final recommendation
**Budget:** $20/mo

---

## TL;DR

**Go with Hetzner Cloud CX32 in Ashburn, VA for $12/mo.** Best value by a wide margin. Full SSH access, 20TB bandwidth, standard Ubuntu. Jeff signs up, Claude Code does the rest.

---

## Platform Comparison (8 Evaluated)

| Rank | Platform | Cost/mo | vCPU | RAM | Storage | Verdict |
|------|----------|---------|------|-----|---------|---------|
| 1 | **Hetzner Cloud** | **$4-12** | 2-4 | 4-8 GB | 40-80 GB | Best value. Full SSH. 20TB bandwidth. US datacenter. |
| 2 | Oracle Free Tier | $0 | 4 ARM | 24 GB | 200 GB | Free but unreliable (idle reclaim, ARM lottery). Good backup. |
| 3 | AWS Lightsail | $20 | 2 | 4 GB | 80 GB | Solid but burstable CPU. On budget ceiling. |
| 4 | DigitalOcean | $24 | 2 | 4 GB | 80 GB | Great but $4 over budget. |
| 5 | Fly.io | ~$15 | 2 shared | 4 GB | 3 GB | Non-standard SSH (flyctl only). Shared CPU. |
| 6 | Railway | ~$60 | varies | varies | varies | No SSH. 3x budget. |
| 7 | Render | $85 | 2 | 4 GB | varies | 4x budget for 4GB tier. |
| 8 | Cloud Run | ~$130 | varies | varies | none | Wrong architecture entirely (stateless, no persistent disk). |

---

## Primary Recommendation: Hetzner Cloud

### Recommended Plan: CX32

| Spec | Value |
|------|-------|
| **vCPU** | 4 (Intel, shared) |
| **RAM** | 8 GB |
| **SSD** | 80 GB |
| **Bandwidth** | 20 TB/mo included |
| **Location** | Ashburn, VA (US East) |
| **Price** | ~$12/mo |

### Why 8 GB RAM

StemScriber's memory footprint under load:

| Component | RAM Usage |
|-----------|-----------|
| Whisper large-v3 model | ~3 GB |
| PyTorch chord detection models | ~500 MB |
| Flask + gunicorn workers | ~200 MB |
| ffmpeg / yt-dlp processing | ~300 MB |
| OS + system overhead | ~500 MB |
| **Total under load** | **~4.5 GB** |
| **Headroom for concurrent jobs** | **~3.5 GB** |

With 4 GB RAM, a single Whisper transcription would leave almost nothing for concurrent requests. 8 GB gives real breathing room.

### Why Ashburn, VA

- US East coast: low latency to most US users
- Hetzner's US datacenter (launched 2023)
- Closest to Jeff's user base during beta

### Budget Alternative: CPX22

If RAM proves excessive or Jeff switches Whisper to the "medium" model (~1.5 GB):

| Spec | Value |
|------|-------|
| **vCPU** | 2 (AMD EPYC, shared) |
| **RAM** | 4 GB |
| **SSD** | 40 GB |
| **Price** | ~$7/mo |

Can always upgrade later with no downtime via Hetzner's resize feature.

### Why Hetzner Wins

- **Full root SSH access** -- Claude Code can manage everything remotely
- **20 TB bandwidth** -- stems at 6-7 MB each means ~3 million downloads before hitting the cap
- **Standard Ubuntu** -- install anything via apt, no proprietary runtime
- **cloudflared runs natively** -- same Cloudflare Tunnel setup as the Mac
- **No vendor lock-in** -- standard Linux server, move anywhere anytime
- **Hetzner API** -- can automate provisioning, snapshots, monitoring

---

## Cost Breakdown

| Item | Monthly Cost |
|------|-------------|
| Hetzner CX32 (4 vCPU, 8 GB) | $12.00 |
| Automated snapshots (80 GB @ $0.06/GB) | ~$4.80 |
| Additional volume storage (if needed) | $0.05/GB |
| **Total estimate** | **$12-17/mo** |

### Comparison

| Option | Monthly Cost |
|--------|-------------|
| Hetzner CX32 | $12 |
| Mac running 24/7 (electricity alone) | ~$5-10 |
| Mac running 24/7 (wear + electricity) | ~$15-25 |
| AWS Lightsail equivalent | $20 |
| DigitalOcean equivalent | $24 |

The Hetzner setup is **40% under the $20/mo budget** and frees the Mac from running around the clock.

---

## What Jeff Needs To Do

1. **Sign up at [hetzner.com](https://www.hetzner.com/cloud/)** -- credit card required, takes ~5 minutes
2. **That's it.**

Claude Code handles everything else.

---

## What Claude Code Will Do

| Step | Task | Time Estimate |
|------|------|---------------|
| 1 | Provision the VPS (via Hetzner dashboard or API) | 5 min |
| 2 | SSH in, install system packages (Python 3.11, ffmpeg, yt-dlp, etc.) | 15 min |
| 3 | Deploy the Flask backend + all dependencies | 20 min |
| 4 | Install and configure cloudflared, migrate the Tunnel from Mac | 10 min |
| 5 | Copy song data, models, and database from Mac to VPS | 15-30 min |
| 6 | Set up systemd services (gunicorn, cloudflared auto-start on boot) | 10 min |
| 7 | Configure firewall (ufw), fail2ban, SSH hardening | 10 min |
| 8 | End-to-end testing (upload song, verify stems, chords, lyrics) | 15 min |
| 9 | Monitor for a week, tune gunicorn workers and memory | ongoing |

**Total deployment time: ~2 hours**, most of it automated.

---

## Risks and Mitigations

### Shared CPU Performance

**Risk:** Heavy Whisper transcription may be slower than M3 Apple Silicon.

**Mitigations:**
- CX32 has 4 vCPUs -- enough to handle transcription without starving the web server
- Whisper "medium" model is 2x faster with minimal quality loss (fallback option)
- GPU-heavy work already runs on Modal, not the VPS
- Can upgrade to dedicated CPU instance ($30/mo) if shared CPU becomes a bottleneck

### Newer US Datacenter

**Risk:** Hetzner's Ashburn datacenter launched in 2023. Less track record than their EU locations.

**Mitigations:**
- Hetzner's EU infrastructure has been rock-solid for 25+ years
- Ashburn is a major interconnection hub (same area as AWS us-east-1)
- SLA is the same across all locations

### No Managed Backups by Default

**Risk:** Data loss if disk fails or accidental deletion.

**Mitigations:**
- Automated snapshots: $0.06/GB/mo (~$4.80 for 80 GB)
- rsync critical data to Cloudflare R2 (songs, database) on a cron
- Song data can be re-processed; models can be re-downloaded
- Database is small (SQLite, <100 MB) -- easy to back up frequently

### Hetzner Account Verification

**Risk:** Hetzner sometimes requires ID verification for new accounts (anti-fraud).

**Mitigation:** Usually resolved within 24 hours. Have a government ID ready just in case.

---

## Optional: Oracle Cloud Free Tier as Backup

| Spec | Value |
|------|-------|
| **OCPU** | 4 (ARM Ampere) |
| **RAM** | 24 GB |
| **Storage** | 200 GB |
| **Price** | Free forever (in theory) |

### Setup Plan

- Provision as a hot standby with the same deployment
- Keep cloudflared installed but not running (activate on failover)
- Sync data from Hetzner nightly via rsync

### Oracle Risks

| Risk | Severity | Notes |
|------|----------|-------|
| Idle instance reclamation | High | Oracle may terminate instances with low utilization. Needs a keepalive cron job. |
| ARM compatibility | Medium | All Python packages must work on aarch64. Most do, but needs testing. |
| Account termination | Low | Oracle has terminated free-tier accounts without warning. Rare but documented. |
| Instance availability | Medium | ARM instances in popular regions often unavailable ("out of capacity"). May need to retry for days. |

**Verdict:** Worth having as a zero-cost failover, but not reliable enough to be the primary server.

---

## Future Optimization Path

| Timeline | Action | Impact |
|----------|--------|--------|
| **Now** | Deploy to Hetzner CX32 ($12/mo), Whisper large-v3 | Stable always-on hosting, Mac freed up |
| **Month 2** | If RAM is tight, switch Whisper to "medium" model | Could downgrade to CPX22 ($7/mo), save $5/mo |
| **Month 3** | Remove tensorflow if basic-pitch works with ONNX | Saves ~1 GB disk, ~500 MB RAM |
| **Month 6** | If user base grows past 50+, evaluate scaling | Dedicated CPU instance or horizontal scaling |
| **Long term** | Move all file storage to Cloudflare R2, keep VPS stateless | Easier scaling, cheaper storage, simpler backups |

---

## Decision Matrix

| Factor | Hetzner CX32 | Oracle Free | AWS Lightsail |
|--------|--------------|-------------|---------------|
| Monthly cost | $12 | $0 | $20 |
| RAM | 8 GB | 24 GB | 4 GB |
| CPU | 4 vCPU (x86) | 4 OCPU (ARM) | 2 vCPU (burstable) |
| SSH access | Full root | Full root | Full root |
| Reliability | High | Uncertain | High |
| Claude Code manageable | Yes | Yes | Yes |
| Bandwidth | 20 TB | 10 TB | 4 TB |
| Upgrade path | Resize in minutes | None (free tier cap) | Resize available |
| Vendor lock-in | None | None | Low |
| **Overall** | **Best choice** | **Free backup** | **Safe but pricey** |

---

## Bottom Line

Hetzner CX32 at $12/mo is the clear winner. It has enough RAM for Whisper large-v3, enough CPU for concurrent processing, enough bandwidth for thousands of users, and full SSH access so Claude Code can manage everything. It comes in 40% under budget with room to scale up or down as needed.

Jeff signs up. Claude Code deploys. StemScriber runs 24/7.
