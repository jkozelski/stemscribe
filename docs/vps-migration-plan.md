# StemScriber VPS Migration Plan

**From:** M3 Mac (localhost) via Cloudflare Tunnel
**To:** Hetzner Cloud VPS (US/Ashburn)
**Date:** 2026-03-19

---

## 1. Pre-Migration Checklist

- [ ] Sign up for Hetzner Cloud account at https://console.hetzner.cloud
- [ ] Create VPS instance (see recommended spec below)
- [ ] Generate SSH key pair on Mac if not already done: `ssh-keygen -t ed25519`
- [ ] Add public key during VPS creation
- [ ] Verify SSH access: `ssh root@<VPS_IP>`
- [ ] Confirm no migration needed for cloud services (already external):
  - **Modal** -- GPU stem separation, already cloud-based
  - **Supabase** -- Postgres database, already cloud-based
  - **Cloudflare R2** -- Object storage, already cloud-based
  - **Stripe** -- Payments, API-based
  - **Twilio** -- SMS, API-based
  - **Resend** -- Email, API-based

### Recommended VPS Spec

| Option | vCPU | RAM | Disk | Cost | Notes |
|--------|------|-----|------|------|-------|
| CPX22 | 2 AMD | 4 GB | 80 GB | ~$7/mo | Tight for Whisper large-v3 |
| **CX32** | **4 shared** | **8 GB** | **80 GB** | **~$12/mo** | **Recommended -- comfortable for Whisper large-v3** |

> **Recommendation:** Go with CX32 ($12/mo). Whisper large-v3 loads ~3GB into RAM, and with Flask + Python overhead you want 8GB. The extra 2 vCPUs help with concurrent requests.

---

## 2. Server Setup (Day 1)

### 2.1 System Packages

```bash
ssh root@<VPS_IP>

# Update system
apt-get update && apt-get upgrade -y

# Install dependencies
apt-get install -y \
  ffmpeg \
  libsndfile1 \
  python3.11 \
  python3.11-dev \
  python3.11-venv \
  build-essential \
  libpq-dev \
  git \
  curl \
  htop \
  unzip
```

> **Note:** Ubuntu 22.04 may not have Python 3.11 in default repos. If not available:
> ```bash
> apt-get install -y software-properties-common
> add-apt-repository -y ppa:deadsnakes/ppa
> apt-get update
> apt-get install -y python3.11 python3.11-dev python3.11-venv
> ```

### 2.2 Create App Directory Structure

```bash
mkdir -p /opt/stemscribe
mkdir -p /opt/stemscribe/uploads
mkdir -p /opt/stemscribe/outputs
mkdir -p /opt/stemscribe/logs
```

### 2.3 Transfer Backend Code

**Option A: rsync from Mac (recommended)**
```bash
# Run from Mac
rsync -avz --exclude='venv311' --exclude='__pycache__' --exclude='.git' \
  --exclude='uploads/' --exclude='outputs/' \
  ~/stemscribe/backend/ root@<VPS_IP>:/opt/stemscribe/backend/
```

**Option B: Git clone (if repo exists)**
```bash
cd /opt/stemscribe
git clone <repo_url> backend
```

### 2.4 Set Up Python 3.11 Virtual Environment

```bash
cd /opt/stemscribe
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r backend/requirements.txt
```

> **Note:** Some packages (torch, tensorflow) are large. This step may take 10-15 minutes on a fresh VPS. Expect ~4-5GB of disk usage for the venv.

### 2.5 Whisper Model

The Whisper large-v3 model (~3GB) will auto-download to `~/.cache/huggingface/` on the first transcription request. No manual action needed, but be aware:

- First transcription will be slow (download time)
- To pre-download, run:
  ```bash
  source /opt/stemscribe/venv/bin/activate
  python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
  ```

---

## 3. Cloudflare Tunnel Migration

### 3.1 Install cloudflared on VPS

```bash
curl -L --output cloudflared.deb https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
dpkg -i cloudflared.deb
rm cloudflared.deb
```

### 3.2 Migrate the Tunnel

**Option A: Move existing tunnel token (simplest)**

1. Get the tunnel token from Mac:
   ```bash
   # On Mac -- find your tunnel token
   cat ~/.cloudflared/*.json
   # Or check: cloudflared tunnel list
   ```

2. On VPS, authenticate and configure:
   ```bash
   # If using a tunnel token (connector install method):
   cloudflared service install <TUNNEL_TOKEN>
   ```

**Option B: Create a new tunnel**

```bash
# On VPS
cloudflared tunnel login
cloudflared tunnel create stemscribe-vps

# Create config
mkdir -p /etc/cloudflared
cat > /etc/cloudflared/config.yml << 'EOF'
tunnel: <TUNNEL_ID>
credentials-file: /root/.cloudflared/<TUNNEL_ID>.json

ingress:
  - hostname: stemscribe.io
    service: http://localhost:5555
  - service: http_status:404
EOF
```

### 3.3 Update DNS (if creating new tunnel)

```bash
cloudflared tunnel route dns stemscribe-vps stemscribe.io
```

> If reusing the existing tunnel token, DNS does not need to change -- Cloudflare routes to whichever connector is active.

### 3.4 Test

```bash
# Start Flask on VPS first
cd /opt/stemscribe
source venv/bin/activate
python backend/app.py &

# Start tunnel
cloudflared tunnel run stemscribe-vps

# From any browser, verify:
# https://stemscribe.io loads correctly
```

---

## 4. Environment & Secrets

### 4.1 Transfer .env File

```bash
# From Mac
scp ~/stemscribe/.env root@<VPS_IP>:/opt/stemscribe/.env
```

### 4.2 Verify API Keys

After copying, test each service from VPS:

- [ ] Supabase: Can Flask connect to the database?
- [ ] Cloudflare R2: Can it read/write objects?
- [ ] Stripe: Test webhook signature verification
- [ ] Twilio: Send a test SMS
- [ ] Resend: Send a test email
- [ ] Modal: Trigger a test stem separation

### 4.3 IP Whitelisting

Check if any services restrict by IP:

- [ ] Supabase -- if network restrictions are set, add VPS IP
- [ ] Stripe webhooks -- no IP restriction, uses signature verification
- [ ] Modal -- no IP restriction, uses API token

---

## 5. Data Migration

### Current Data Footprint

| Directory | Size | Contents |
|-----------|------|----------|
| `uploads/` | ~7.7 GB | Original uploaded song files |
| `outputs/` | ~2.9 GB | Separated stems, charts, etc. |
| **Total** | **~10.6 GB** | Fits on 80 GB SSD with room to spare |

### Option A: rsync to VPS (recommended for now)

```bash
# From Mac -- run these in parallel with screen/tmux
rsync -avz --progress ~/stemscribe/uploads/ root@<VPS_IP>:/opt/stemscribe/uploads/
rsync -avz --progress ~/stemscribe/outputs/ root@<VPS_IP>:/opt/stemscribe/outputs/
```

> Estimated transfer time: 30-60 minutes depending on upload speed.

### Option B: Move Everything to Cloudflare R2 (longer term)

- Modify backend to read/write all files from R2 instead of local disk
- Eliminates VPS disk as a bottleneck
- Allows VPS to stay small and ephemeral
- **Not blocking for migration -- do this as a follow-up project**

---

## 6. Service Management

### 6.1 Flask/Gunicorn systemd Service

```bash
# Install gunicorn in the venv
source /opt/stemscribe/venv/bin/activate
pip install gunicorn

# Create service file
cat > /etc/systemd/system/stemscribe.service << 'EOF'
[Unit]
Description=StemScriber Flask Backend
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/opt/stemscribe/backend
Environment="PATH=/opt/stemscribe/venv/bin"
EnvironmentFile=/opt/stemscribe/.env
ExecStart=/opt/stemscribe/venv/bin/gunicorn \
  --bind 127.0.0.1:5555 \
  --workers 2 \
  --timeout 300 \
  --access-logfile /opt/stemscribe/logs/access.log \
  --error-logfile /opt/stemscribe/logs/error.log \
  app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable stemscribe
systemctl start stemscribe
```

> **Note:** `--timeout 300` gives long-running requests (Whisper, chord detection) 5 minutes. Adjust if needed. `--workers 2` is safe for 4-8GB RAM with Whisper loaded.

### 6.2 Cloudflared systemd Service

If installed via `cloudflared service install`, this is already handled. Otherwise:

```bash
cat > /etc/systemd/system/cloudflared.service << 'EOF'
[Unit]
Description=Cloudflare Tunnel
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/cloudflared tunnel --config /etc/cloudflared/config.yml run
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable cloudflared
systemctl start cloudflared
```

### 6.3 Log Rotation

```bash
cat > /etc/logrotate.d/stemscribe << 'EOF'
/opt/stemscribe/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    postrotate
        systemctl reload stemscribe 2>/dev/null || true
    endscript
}
EOF
```

### 6.4 Verify Auto-Start

```bash
# Reboot and confirm both services come back
reboot

# After reconnecting:
systemctl status stemscribe
systemctl status cloudflared
curl -s http://localhost:5555/api/health
```

---

## 7. SSH Access for Claude Code

### 7.1 Add SSH Key

```bash
# On VPS, add Claude Code's public key
mkdir -p /root/.ssh
echo "<CLAUDE_CODE_PUBLIC_KEY>" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
```

### 7.2 Connection Details

```
Host: <VPS_IP>
Port: 22
User: root
Key: (Claude Code SSH key)
App directory: /opt/stemscribe/
Backend code: /opt/stemscribe/backend/
Logs: /opt/stemscribe/logs/
Venv: /opt/stemscribe/venv/
```

### 7.3 Test Remote Workflow

```bash
# Verify Claude Code can:
ssh root@<VPS_IP> "systemctl status stemscribe"
ssh root@<VPS_IP> "tail -20 /opt/stemscribe/logs/error.log"
ssh root@<VPS_IP> "source /opt/stemscribe/venv/bin/activate && cd /opt/stemscribe/backend && python -m pytest tests/ -v"
```

---

## 8. Testing & Cutover

### 8.1 Pre-Cutover Testing (VPS running alongside Mac)

Run Flask on VPS at port 5555 but do NOT switch the tunnel yet. Test manually:

```bash
# On VPS
curl http://localhost:5555/api/health
```

- [ ] Health endpoint responds
- [ ] Song upload works (file saves to /opt/stemscribe/uploads/)
- [ ] Stem separation triggers Modal and completes
- [ ] Chord detection runs and returns results
- [ ] Whisper transcription completes (test with a short song)
- [ ] Stripe payment webhook fires correctly
- [ ] Email sending works (Resend)
- [ ] SMS sending works (Twilio)
- [ ] Existing song library loads and plays in frontend

### 8.2 Cutover

1. **Stop tunnel on Mac:**
   ```bash
   # On Mac
   sudo launchctl stop com.cloudflare.cloudflared  # or however it's running
   # OR: cloudflared tunnel stop
   ```

2. **Start/verify tunnel on VPS:**
   ```bash
   systemctl status cloudflared  # should already be running
   ```

3. **Verify stemscribe.io resolves to VPS:**
   ```bash
   curl -I https://stemscribe.io
   ```

### 8.3 Post-Cutover Monitoring

- [ ] Monitor for 24 hours
- [ ] Watch VPS resource usage: `htop`, `df -h`, `free -h`
- [ ] Check logs: `tail -f /opt/stemscribe/logs/error.log`
- [ ] Test from mobile device
- [ ] Test from multiple browsers
- [ ] Verify no Cloudflare caching issues (cache bypass rule should still be active)

---

## 9. Rollback Plan

### If VPS Fails

1. SSH into Mac
2. Restart cloudflared on Mac:
   ```bash
   # On Mac
   cloudflared tunnel run stemscribe
   ```
3. Traffic immediately routes back to Mac -- zero DNS propagation delay since Cloudflare Tunnel handles routing

### Retention Policy

- [ ] Keep Mac Flask server ready to run for **1 week** after cutover
- [ ] Keep local copy of all data (uploads/, outputs/) on Mac for **30 days**
- [ ] After 30 days of stable VPS operation, archive Mac data to external drive

---

## 10. Timeline

| Day | Task | Estimated Time |
|-----|------|----------------|
| **Day 1** | Hetzner signup, VPS creation, SSH setup | 15 min |
| **Day 1** | System packages, Python venv, pip install | 30 min |
| **Day 1** | Transfer code + .env, configure gunicorn | 30 min |
| **Day 1** | Cloudflared install + tunnel config | 30 min |
| **Day 1** | Test all endpoints on VPS (no cutover) | 30 min |
| **Day 2** | rsync uploads/ and outputs/ to VPS | 30-60 min |
| **Day 2** | Switch Cloudflare Tunnel from Mac to VPS | 15 min |
| **Day 2** | Smoke test stemscribe.io from browser + mobile | 15 min |
| **Days 3-7** | Monitor, fix issues as they arise | as needed |
| **Day 7** | Decommission Mac tunnel, keep data as backup | 15 min |

**Total hands-on time: ~3-4 hours spread over 2 days**

---

## 11. Cost Optimization Tips

### Quick Wins

- **Whisper model size:** Switching from `large-v3` to `medium` reduces RAM from ~3GB to ~1.5GB, allowing the $7/mo CPX22 (4GB RAM). Accuracy trade-off is modest for English lyrics.

- **Remove tensorflow:** If `basic-pitch` can use the ONNX backend instead of TensorFlow, you save ~1GB disk and ~500MB RAM.
  ```bash
  # Check if basic-pitch supports ONNX
  pip install basic-pitch[onnx]
  pip uninstall tensorflow
  ```

- **Move uploads/outputs to R2:** Keeps VPS disk usage minimal. Could even drop to a smaller SSD tier in the future.

### Longer Term

- Use `gunicorn --preload` to share Whisper model memory across workers (copy-on-write)
- Add swap file as safety net:
  ```bash
  fallocate -l 2G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo '/swapfile none swap sw 0 0' >> /etc/fstab
  ```

---

## 12. Optional: Oracle Cloud Free Tier as Hot Standby

Oracle Cloud offers an always-free ARM instance (4 OCPU, 24GB RAM) that could serve as a $0/mo hot standby.

### Setup

- [ ] Create Oracle Cloud account (free tier)
- [ ] Provision ARM Ampere A1 instance (up to 4 OCPU / 24 GB RAM free)
- [ ] Mirror the Hetzner setup (same systemd services, same code)
- [ ] Keep code in sync via git pull or rsync cron
- [ ] Do NOT run cloudflared -- only activate if Hetzner goes down

### Preventing Idle Reclamation

Oracle reclaims idle free-tier instances. Add a cron job:

```bash
# On Oracle VPS
crontab -e
# Add:
*/10 * * * * curl -s http://localhost:5555/api/health > /dev/null 2>&1
```

### Failover Process

1. Hetzner goes down
2. SSH into Oracle instance
3. Start cloudflared tunnel
4. Traffic routes to Oracle
5. Fix Hetzner, switch back

---

## Disk Budget (CX32 -- 80 GB SSD)

| Item | Size | Notes |
|------|------|-------|
| Ubuntu 22.04 base | ~3 GB | |
| System packages + ffmpeg | ~1 GB | |
| Python venv + packages | ~5 GB | torch is the big one |
| Whisper large-v3 model | ~3 GB | in ~/.cache |
| Backend code | < 100 MB | |
| uploads/ | ~7.7 GB | grows with new songs |
| outputs/ | ~2.9 GB | grows with new songs |
| Logs + swap | ~3 GB | |
| **Total** | **~26 GB** | **54 GB free** |

> Plenty of room. At current growth rate (~1GB/month), disk won't be a concern for years. Moving to R2 later eliminates this entirely.

---

## Quick Reference Commands

```bash
# Check service status
systemctl status stemscribe
systemctl status cloudflared

# View logs
journalctl -u stemscribe -f
tail -f /opt/stemscribe/logs/error.log

# Restart services
systemctl restart stemscribe
systemctl restart cloudflared

# Check resources
htop
df -h
free -h

# Deploy code update
cd /opt/stemscribe/backend
git pull  # or rsync from Mac
systemctl restart stemscribe
```
