"""
SMS Routes — Two-way SMS via Twilio for Claude Code <-> Jeff texting.

POST /api/sms/incoming  — Twilio webhook (receives SMS, stores to JSON, forwards to KOZMO)
GET  /api/sms/inbox     — Returns unread messages for Claude Code to poll
POST /api/sms/mark-read — Marks messages as read
POST /api/sms/send      — Send an SMS reply back to Jeff
"""

import json
import os
import logging
import threading
import base64
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, request, jsonify

from middleware.validation import validate_phone_number, sanitize_text

logger = logging.getLogger(__name__)

sms_bp = Blueprint('sms', __name__, url_prefix='/api/sms')

INBOX_FILE = Path(__file__).parent.parent / 'sms_inbox.json'
JEFF_PHONE = '+18034149454'
TWILIO_FROM = '+18447915323'

# KOZMO n8n webhook for processing SMS replies
KOZMO_WEBHOOK_URL = 'https://n8n.kozbotix.com/webhook/sms-incoming'
# Timeout for KOZMO processing (2 minutes — AI agent can be slow)
KOZMO_TIMEOUT = 120


def _load_inbox():
    if INBOX_FILE.exists():
        with open(INBOX_FILE, 'r') as f:
            return json.load(f)
    return []


def _save_inbox(messages):
    with open(INBOX_FILE, 'w') as f:
        json.dump(messages, f, indent=2)


def _forward_to_kozmo(body, sender, media=None):
    """Background thread: Forward incoming SMS to n8n for processing.
    
    n8n handles the AI reply and sends the SMS response via /api/sms/send.
    This function only forwards — it does NOT send a reply itself (to avoid duplicates).
    """
    try:
        import requests

        logger.info(f"Forwarding SMS from {sender} to KOZMO: {body[:80]}... ({len(media or [])} media)")

        # Fetch and base64-encode any media attachments from Twilio
        encoded_media = []
        if media:
            account_sid = os.environ.get('TWILIO_ACCOUNT_SID', 'AC61b4ba568a01c65bf90d98655261161b')
            auth_token = os.environ.get('TWILIO_AUTH_TOKEN', '')
            if auth_token:
                for m in media:
                    try:
                        media_resp = requests.get(
                            m['url'],
                            auth=(account_sid, auth_token),
                            timeout=30,
                        )
                        if media_resp.status_code == 200:
                            b64 = base64.b64encode(media_resp.content).decode('utf-8')
                            encoded_media.append({
                                'content_type': m.get('content_type', 'image/jpeg'),
                                'data': b64,
                            })
                            logger.info(f"Fetched media: {m.get('content_type')} ({len(media_resp.content)} bytes)")
                        else:
                            logger.warning(f"Failed to fetch media {m['url']}: HTTP {media_resp.status_code}")
                    except Exception as me:
                        logger.warning(f"Failed to fetch media {m['url']}: {me}")
            else:
                logger.warning("TWILIO_AUTH_TOKEN not set — cannot fetch MMS media")

        chat_text = body if body.strip() else '[Image message with no text]'

        # Forward to n8n webhook — n8n handles AI reply + SMS send
        payload = {
            'chatInput': f"[SMS from {sender}]: {chat_text}",
            'message': body,
            'sessionId': f'sms_{sender.replace("+", "")}',
            'source': 'sms',
            'from': sender,
        }
        if encoded_media:
            payload['media'] = encoded_media

        resp = requests.post(
            KOZMO_WEBHOOK_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=KOZMO_TIMEOUT,
        )

        if resp.status_code == 200:
            logger.info(f"KOZMO processed SMS from {sender} successfully")
        else:
            logger.error(f"KOZMO returned status {resp.status_code}: {resp.text[:200]}")

    except Exception as e:
        logger.error(f"Failed to forward SMS to KOZMO: {e}", exc_info=True)


@sms_bp.route('/incoming', methods=['POST'])
def incoming_sms():
    """Twilio webhook — receives incoming SMS, stores it, and forwards to KOZMO for reply."""
    sender = request.form.get('From', '')
    body = request.form.get('Body', '')
    message_sid = request.form.get('MessageSid', '')
    num_media = int(request.form.get('NumMedia', 0))

    # Collect media URLs (photos, images, etc.)
    media = []
    for i in range(num_media):
        media_url = request.form.get(f'MediaUrl{i}', '')
        media_type = request.form.get(f'MediaContentType{i}', '')
        if media_url:
            media.append({'url': media_url, 'content_type': media_type})

    logger.info(f"SMS received from {sender}: {body[:50]}... ({num_media} media)")

    messages = _load_inbox()
    msg = {
        'id': message_sid or f"msg_{len(messages)}",
        'from': sender,
        'body': body,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'read': False
    }
    if media:
        msg['media'] = media
    messages.append(msg)
    _save_inbox(messages)

    # Forward to KOZMO in background thread (non-blocking)
    # Trigger on text OR media (MMS images)
    if body.strip() or media:
        thread = threading.Thread(
            target=_forward_to_kozmo,
            args=(body, sender, media),
            daemon=True,
        )
        thread.start()
        logger.info(f"KOZMO forwarding thread started for message from {sender}")

    # Return TwiML response (empty — KOZMO reply comes via separate SMS)
    twiml = '<?xml version="1.0" encoding="UTF-8"?><Response></Response>'
    return twiml, 200, {'Content-Type': 'text/xml'}


@sms_bp.route('/inbox', methods=['GET'])
def get_inbox():
    """Returns unread messages for Claude Code to poll."""
    messages = _load_inbox()
    unread = [m for m in messages if not m.get('read')]
    return jsonify({'unread_count': len(unread), 'messages': unread})


@sms_bp.route('/mark-read', methods=['POST'])
def mark_read():
    """Marks messages as read. Send {"ids": [...]} or {"all": true}."""
    data = request.get_json(silent=True) or {}
    messages = _load_inbox()

    if data.get('all'):
        for m in messages:
            m['read'] = True
    else:
        ids_to_mark = set(data.get('ids', []))
        for m in messages:
            if m['id'] in ids_to_mark:
                m['read'] = True

    _save_inbox(messages)
    return jsonify({'status': 'ok'})


@sms_bp.route('/send', methods=['POST'])
def send_sms():
    """Send an SMS reply back. Body: {"to": "+1...", "body": "text"}. Requires admin key."""
    # Auth check — only allow from localhost or with admin key
    admin_key = os.environ.get('BETA_ADMIN_KEY', 'stemscribe-beta-admin-2026')
    req_key = request.headers.get('X-Admin-Key', '')
    is_localhost = request.remote_addr in ('127.0.0.1', '::1', 'localhost')
    if not is_localhost and req_key != admin_key:
        return jsonify({'error': 'Unauthorized'}), 401
    data = request.get_json(force=True, silent=True) or {}
    body = sanitize_text(data.get('body', ''), max_length=1600)
    to = data.get('to', JEFF_PHONE)

    if not body:
        return jsonify({'error': 'body is required'}), 400

    # Validate phone number format
    if to != JEFF_PHONE and not validate_phone_number(to):
        return jsonify({'error': 'Invalid phone number format'}), 400

    try:
        from twilio.rest import Client
        account_sid = os.environ.get('TWILIO_ACCOUNT_SID', 'AC61b4ba568a01c65bf90d98655261161b')
        auth_token = os.environ.get('TWILIO_AUTH_TOKEN', '')

        if not auth_token:
            return jsonify({'error': 'TWILIO_AUTH_TOKEN not set in environment'}), 500

        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=body,
            from_=TWILIO_FROM,
            to=to
        )
        logger.info(f"SMS sent to {to}: {body[:50]}... (SID: {message.sid})")
        return jsonify({'status': 'sent', 'sid': message.sid})

    except ImportError:
        return jsonify({'error': 'twilio package not installed — pip install twilio'}), 500
    except Exception as e:
        logger.error(f"Failed to send SMS: {e}")
        return jsonify({'error': str(e)}), 500
