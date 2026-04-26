"""Queue-depth alerting cron.

Runs every 5 minutes via systemd timer. Tracks the count of jobs in
'processing' state. If sustained above threshold for 15 min (3 samples),
SMS Jeff via the existing Twilio config.

Cooldown: don't re-alert for 30 min after firing once. Avoid spamming
during a sustained spike.

State persistence: a tiny JSON at /var/lib/stemscribe/queue_monitor.json
holds the last 5 samples and last-alert timestamp.

Usage:
  /opt/stemscribe/venv311/bin/python /opt/stemscribe/backend/scripts/monitor_queue.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('queue_monitor')

# Tunables
THRESHOLD = 4              # alert if active processing jobs > this
SUSTAINED_SAMPLES = 3      # for this many consecutive samples (5 min × 3 = 15 min)
COOLDOWN_SEC = 30 * 60     # don't re-alert within 30 min of last alert
STATE_PATH = Path('/var/lib/stemscribe/queue_monitor.json')
API_BASE = os.environ.get('STEMSCRIBE_API_BASE', 'http://localhost:5555')
JEFF_PHONE = '+18034149454'
TWILIO_FROM = '+18447915323'


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {'samples': [], 'last_alert_ts': 0}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def _current_queue_depth() -> int:
    """Count jobs currently in 'processing' state by scanning job_metadata.json
    files in outputs/. Faster and more reliable than hitting /api/jobs which
    returns the full chord-progression payload for every job."""
    outputs_dir = Path(os.environ.get('STEMSCRIBE_OUTPUTS_DIR', '/opt/stemscribe/outputs'))
    if not outputs_dir.exists():
        logger.error(f'Outputs dir not found: {outputs_dir}')
        return -1
    count = 0
    skipped = 0
    for meta_path in outputs_dir.glob('*/job_metadata.json'):
        try:
            with meta_path.open() as f:
                meta = json.load(f)
            if meta.get('status') == 'processing':
                count += 1
        except Exception:
            skipped += 1
    if skipped:
        logger.debug(f'skipped {skipped} unreadable metadata files')
    return count


def _send_sms(body: str) -> None:
    """SMS via Twilio. Reads creds from env (loaded by systemd unit)."""
    try:
        from twilio.rest import Client
        sid = os.environ.get('TWILIO_ACCOUNT_SID')
        token = os.environ.get('TWILIO_AUTH_TOKEN')
        if not sid or not token:
            logger.error('Twilio creds missing in env')
            return
        client = Client(sid, token)
        m = client.messages.create(body=body, from_=TWILIO_FROM, to=JEFF_PHONE)
        logger.info(f'SMS sent: {m.sid} status={m.status}')
    except Exception as e:
        logger.error(f'SMS failed: {e}')


def main() -> int:
    state = _load_state()
    depth = _current_queue_depth()
    now = int(time.time())

    if depth < 0:
        logger.warning('Skipping cycle — could not read queue depth')
        return 1

    # Append to ring buffer of last SUSTAINED_SAMPLES samples
    samples = state.get('samples', [])
    samples.append({'ts': now, 'depth': depth})
    samples = samples[-SUSTAINED_SAMPLES:]
    state['samples'] = samples

    logger.info(
        f'queue_depth={depth} threshold={THRESHOLD} '
        f'samples_above_threshold='
        f'{sum(1 for s in samples if s["depth"] > THRESHOLD)}/{len(samples)}'
    )

    # Trigger condition: all stored samples above threshold AND we have a
    # full window AND cooldown has expired.
    all_above = (
        len(samples) >= SUSTAINED_SAMPLES
        and all(s['depth'] > THRESHOLD for s in samples)
    )
    cooldown_expired = (now - state.get('last_alert_ts', 0)) >= COOLDOWN_SEC

    if all_above and cooldown_expired:
        peak = max(s['depth'] for s in samples)
        body = (
            f'StemScriber queue depth alert: '
            f'{peak} jobs processing, sustained for ~15 min. '
            f'Threshold is {THRESHOLD}. Concurrency cap should be holding '
            f'but you may want to scale to CPX51 if this continues.'
        )
        _send_sms(body)
        state['last_alert_ts'] = now
        logger.warning(f'ALERT FIRED: peak={peak} samples_window={SUSTAINED_SAMPLES}')

    _save_state(state)
    return 0


if __name__ == '__main__':
    sys.exit(main())
