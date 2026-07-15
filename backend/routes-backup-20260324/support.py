"""
Support ticket system — simple JSON-backed ticket management for beta testers.

No database required. Tickets are stored in a JSON file on disk.
"""

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import requests
from flask import Blueprint, request, jsonify

from middleware.validation import validate_email_format as _validate_email, validate_job_id

logger = logging.getLogger(__name__)

support_bp = Blueprint("support", __name__)

# Store tickets in data/ directory
TICKETS_FILE = Path(__file__).parent.parent / 'data' / 'support_tickets.json'

# Thread-safe file lock
_file_lock = threading.Lock()

# Simple email regex
_EMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')

# Subjects that auto-set high priority
_HIGH_PRIORITY_SUBJECTS = {'billing/refund', 'billing', 'refund'}

# n8n webhook integration (optional — set N8N_WEBHOOK_URL in .env to enable)
_N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', '')
_N8N_SUPPORT_API_KEY = os.environ.get('N8N_SUPPORT_API_KEY', '')


def _sanitize(text):
    """Strip HTML tags and escape special characters."""
    if not isinstance(text, str):
        return ''
    return escape(text.strip())


def _load_tickets():
    """Load tickets from disk (caller must hold _file_lock)."""
    if TICKETS_FILE.exists():
        try:
            return json.loads(TICKETS_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt tickets file, returning empty list")
            return []
    return []


def _save_tickets(tickets):
    """Save tickets to disk (caller must hold _file_lock)."""
    TICKETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TICKETS_FILE.write_text(json.dumps(tickets, indent=2))


def _find_ticket(tickets, ticket_id):
    """Find a ticket by ID, return (index, ticket) or (None, None)."""
    for i, t in enumerate(tickets):
        if t['id'] == ticket_id:
            return i, t
    return None, None


def _forward_to_n8n(ticket):
    """Fire-and-forget ticket data to n8n webhook. Non-blocking via thread."""
    if not _N8N_WEBHOOK_URL:
        return

    def _send():
        try:
            headers = {'Content-Type': 'application/json'}
            if _N8N_SUPPORT_API_KEY:
                headers['x-api-key'] = _N8N_SUPPORT_API_KEY
            requests.post(
                _N8N_WEBHOOK_URL,
                json=ticket,
                headers=headers,
                timeout=5,
            )
            logger.info(f"Forwarded ticket {ticket['id']} to n8n")
        except Exception as e:
            logger.warning(f"Failed to forward ticket to n8n: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ---- Endpoints ----

@support_bp.route('/api/support/ticket', methods=['POST'])
def create_ticket():
    """
    Create a new support ticket.

    Body (JSON):
        name: Customer name (required)
        email: Customer email (required)
        subject: Ticket subject (optional, defaults to "General")
        message: Ticket message (required)

    Returns: {success: true, ticket_id: "..."}
    """
    data = request.get_json(silent=True) or {}

    name = _sanitize(data.get('name', ''))
    email = _sanitize(data.get('email', ''))
    subject = _sanitize(data.get('subject', '')) or 'General'
    message = _sanitize(data.get('message', ''))

    # Validation
    errors = []
    if not name:
        errors.append('name is required')
    elif len(name) > 200:
        errors.append('name is too long (max 200 characters)')
    if not email:
        errors.append('email is required')
    elif not _validate_email(email):
        errors.append('invalid email format')
    if not message:
        errors.append('message is required')
    elif len(message) > 5000:
        errors.append('message is too long (max 5000 characters)')
    if len(subject) > 200:
        errors.append('subject is too long (max 200 characters)')

    if errors:
        return jsonify({'error': 'Validation failed', 'details': errors}), 400

    # Auto-set priority
    priority = 'high' if subject.lower() in _HIGH_PRIORITY_SUBJECTS else 'normal'

    ticket = {
        'id': str(uuid.uuid4()),
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'name': name,
        'email': email,
        'subject': subject,
        'message': message,
        'status': 'open',
        'priority': priority,
        'responses': [],
        'resolved_date': None,
    }

    with _file_lock:
        tickets = _load_tickets()
        tickets.append(ticket)
        _save_tickets(tickets)

    logger.info(f"Support ticket created: {ticket['id']} ({subject}) priority={priority}")

    # Forward to n8n for SMS/email/sheet logging (non-blocking)
    _forward_to_n8n(ticket)

    return jsonify({'success': True, 'ticket_id': ticket['id']}), 201


@support_bp.route('/api/support/tickets', methods=['GET'])
def list_tickets():
    """
    List all support tickets. Supports optional filtering.

    Query params:
        status: Filter by status (open, in-progress, resolved)
        subject: Filter by subject (partial match, case-insensitive)

    Returns: {tickets: [...], count: N}
    """
    status_filter = request.args.get('status', '').strip().lower()
    subject_filter = request.args.get('subject', '').strip().lower()

    with _file_lock:
        tickets = _load_tickets()

    if status_filter:
        tickets = [t for t in tickets if t['status'] == status_filter]
    if subject_filter:
        tickets = [t for t in tickets if subject_filter in t.get('subject', '').lower()]

    return jsonify({'tickets': tickets, 'count': len(tickets)})


@support_bp.route('/api/support/ticket/<ticket_id>', methods=['GET'])
def get_ticket(ticket_id):
    """
    Get a single ticket by ID.

    Returns: ticket object or 404.
    """
    if not validate_job_id(ticket_id):
        return jsonify({'error': 'Invalid ticket ID'}), 400
    with _file_lock:
        tickets = _load_tickets()

    _, ticket = _find_ticket(tickets, ticket_id)
    if ticket is None:
        return jsonify({'error': 'Ticket not found'}), 404

    return jsonify(ticket)


@support_bp.route('/api/support/ticket/<ticket_id>/respond', methods=['POST'])
def respond_to_ticket(ticket_id):
    """
    Add a response to a ticket and update status.

    Body (JSON):
        response_text: The response message (required)
        status: Optional new status ("in-progress" or "resolved", defaults to "in-progress")

    Returns: {success: true}
    """
    if not validate_job_id(ticket_id):
        return jsonify({'error': 'Invalid ticket ID'}), 400
    data = request.get_json(silent=True) or {}
    response_text = _sanitize(data.get('response_text', ''))

    if not response_text:
        return jsonify({'error': 'response_text is required'}), 400

    new_status = data.get('status', 'in-progress')
    if new_status not in ('in-progress', 'resolved'):
        new_status = 'in-progress'

    with _file_lock:
        tickets = _load_tickets()
        idx, ticket = _find_ticket(tickets, ticket_id)

        if ticket is None:
            return jsonify({'error': 'Ticket not found'}), 404

        ticket['responses'].append({
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'text': response_text,
        })
        ticket['status'] = new_status

        if new_status == 'resolved':
            ticket['resolved_date'] = datetime.now(timezone.utc).isoformat()

        tickets[idx] = ticket
        _save_tickets(tickets)

    logger.info(f"Support ticket {ticket_id}: response added, status={new_status}")

    return jsonify({'success': True})


@support_bp.route('/api/support/ticket/<ticket_id>/resolve', methods=['POST'])
def resolve_ticket(ticket_id):
    """
    Mark a ticket as resolved.

    Returns: {success: true}
    """
    if not validate_job_id(ticket_id):
        return jsonify({'error': 'Invalid ticket ID'}), 400
    with _file_lock:
        tickets = _load_tickets()
        idx, ticket = _find_ticket(tickets, ticket_id)

        if ticket is None:
            return jsonify({'error': 'Ticket not found'}), 404

        ticket['status'] = 'resolved'
        ticket['resolved_date'] = datetime.now(timezone.utc).isoformat()
        tickets[idx] = ticket
        _save_tickets(tickets)

    logger.info(f"Support ticket {ticket_id}: resolved")

    return jsonify({'success': True})
