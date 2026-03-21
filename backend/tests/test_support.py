"""
Tests for the support ticket system.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from routes.support import TICKETS_FILE, _file_lock


@pytest.fixture
def app(tmp_path):
    """Create a test app with a temporary tickets file."""
    test_tickets_file = tmp_path / 'support_tickets.json'
    with patch('routes.support.TICKETS_FILE', test_tickets_file):
        app = create_app()
        app.config['TESTING'] = True
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_ticket():
    return {
        'name': 'Test User',
        'email': 'test@example.com',
        'subject': 'Bug Report',
        'message': 'The chord detection shows wrong chords for song X.',
    }


class TestCreateTicket:
    """Test POST /api/support/ticket."""

    def test_create_ticket_success(self, client, sample_ticket):
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data['success'] is True
        assert 'ticket_id' in data
        assert len(data['ticket_id']) == 36  # UUID format

    def test_create_ticket_missing_name(self, client, sample_ticket):
        del sample_ticket['name']
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 400
        data = resp.get_json()
        assert 'name is required' in data['details']

    def test_create_ticket_missing_email(self, client, sample_ticket):
        del sample_ticket['email']
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 400
        assert 'email is required' in resp.get_json()['details']

    def test_create_ticket_missing_message(self, client, sample_ticket):
        del sample_ticket['message']
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 400
        assert 'message is required' in resp.get_json()['details']

    def test_create_ticket_invalid_email(self, client, sample_ticket):
        sample_ticket['email'] = 'not-an-email'
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 400
        assert 'invalid email format' in resp.get_json()['details']

    def test_create_ticket_multiple_errors(self, client):
        resp = client.post('/api/support/ticket', json={})
        assert resp.status_code == 400
        details = resp.get_json()['details']
        assert len(details) >= 2  # at least name and message

    def test_create_ticket_html_sanitization(self, client, sample_ticket):
        sample_ticket['name'] = '<script>alert("xss")</script>'
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 201
        # Retrieve and verify sanitized
        ticket_id = resp.get_json()['ticket_id']
        resp2 = client.get(f'/api/support/ticket/{ticket_id}')
        data = resp2.get_json()
        assert '<script>' not in data['name']
        assert '&lt;script&gt;' in data['name']

    def test_create_ticket_default_subject(self, client, sample_ticket):
        del sample_ticket['subject']
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 201
        ticket_id = resp.get_json()['ticket_id']
        resp2 = client.get(f'/api/support/ticket/{ticket_id}')
        assert resp2.get_json()['subject'] == 'General'

    def test_billing_ticket_gets_high_priority(self, client, sample_ticket):
        sample_ticket['subject'] = 'Billing/Refund'
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 201
        ticket_id = resp.get_json()['ticket_id']
        resp2 = client.get(f'/api/support/ticket/{ticket_id}')
        assert resp2.get_json()['priority'] == 'high'

    def test_normal_ticket_gets_normal_priority(self, client, sample_ticket):
        resp = client.post('/api/support/ticket', json=sample_ticket)
        assert resp.status_code == 201
        ticket_id = resp.get_json()['ticket_id']
        resp2 = client.get(f'/api/support/ticket/{ticket_id}')
        assert resp2.get_json()['priority'] == 'normal'

    def test_ticket_initial_status_is_open(self, client, sample_ticket):
        resp = client.post('/api/support/ticket', json=sample_ticket)
        ticket_id = resp.get_json()['ticket_id']
        resp2 = client.get(f'/api/support/ticket/{ticket_id}')
        ticket = resp2.get_json()
        assert ticket['status'] == 'open'
        assert ticket['responses'] == []
        assert ticket['resolved_date'] is None


class TestListTickets:
    """Test GET /api/support/tickets."""

    def test_list_empty(self, client):
        resp = client.get('/api/support/tickets')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['tickets'] == []
        assert data['count'] == 0

    def test_list_all_tickets(self, client, sample_ticket):
        client.post('/api/support/ticket', json=sample_ticket)
        client.post('/api/support/ticket', json={**sample_ticket, 'subject': 'Feature Request'})
        resp = client.get('/api/support/tickets')
        assert resp.get_json()['count'] == 2

    def test_filter_by_status(self, client, sample_ticket):
        # Create two tickets, resolve one
        r1 = client.post('/api/support/ticket', json=sample_ticket)
        client.post('/api/support/ticket', json=sample_ticket)
        tid = r1.get_json()['ticket_id']
        client.post(f'/api/support/ticket/{tid}/resolve')

        resp = client.get('/api/support/tickets?status=open')
        assert resp.get_json()['count'] == 1

        resp = client.get('/api/support/tickets?status=resolved')
        assert resp.get_json()['count'] == 1

    def test_filter_by_subject(self, client, sample_ticket):
        client.post('/api/support/ticket', json=sample_ticket)
        client.post('/api/support/ticket', json={**sample_ticket, 'subject': 'Feature Request'})

        resp = client.get('/api/support/tickets?subject=Bug')
        assert resp.get_json()['count'] == 1

        resp = client.get('/api/support/tickets?subject=feature')
        assert resp.get_json()['count'] == 1


class TestGetTicket:
    """Test GET /api/support/ticket/<id>."""

    def test_get_existing_ticket(self, client, sample_ticket):
        r = client.post('/api/support/ticket', json=sample_ticket)
        tid = r.get_json()['ticket_id']
        resp = client.get(f'/api/support/ticket/{tid}')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['id'] == tid
        assert data['name'] == 'Test User'
        assert data['email'] == 'test@example.com'

    def test_get_nonexistent_ticket(self, client):
        resp = client.get('/api/support/ticket/00000000-0000-0000-0000-000000000099')
        assert resp.status_code == 404


class TestRespondToTicket:
    """Test POST /api/support/ticket/<id>/respond."""

    def test_respond_success(self, client, sample_ticket):
        r = client.post('/api/support/ticket', json=sample_ticket)
        tid = r.get_json()['ticket_id']

        resp = client.post(f'/api/support/ticket/{tid}/respond', json={
            'response_text': 'Thanks for reporting! We are looking into it.',
        })
        assert resp.status_code == 200
        assert resp.get_json()['success'] is True

        # Verify
        ticket = client.get(f'/api/support/ticket/{tid}').get_json()
        assert ticket['status'] == 'in-progress'
        assert len(ticket['responses']) == 1
        assert 'looking into it' in ticket['responses'][0]['text']

    def test_respond_missing_text(self, client, sample_ticket):
        r = client.post('/api/support/ticket', json=sample_ticket)
        tid = r.get_json()['ticket_id']
        resp = client.post(f'/api/support/ticket/{tid}/respond', json={})
        assert resp.status_code == 400

    def test_respond_nonexistent_ticket(self, client):
        resp = client.post('/api/support/ticket/00000000-0000-0000-0000-000000000099/respond', json={
            'response_text': 'Hello',
        })
        assert resp.status_code == 404

    def test_respond_with_resolved_status(self, client, sample_ticket):
        r = client.post('/api/support/ticket', json=sample_ticket)
        tid = r.get_json()['ticket_id']
        client.post(f'/api/support/ticket/{tid}/respond', json={
            'response_text': 'Fixed!',
            'status': 'resolved',
        })
        ticket = client.get(f'/api/support/ticket/{tid}').get_json()
        assert ticket['status'] == 'resolved'
        assert ticket['resolved_date'] is not None

    def test_multiple_responses(self, client, sample_ticket):
        r = client.post('/api/support/ticket', json=sample_ticket)
        tid = r.get_json()['ticket_id']
        client.post(f'/api/support/ticket/{tid}/respond', json={'response_text': 'Response 1'})
        client.post(f'/api/support/ticket/{tid}/respond', json={'response_text': 'Response 2'})
        ticket = client.get(f'/api/support/ticket/{tid}').get_json()
        assert len(ticket['responses']) == 2


class TestResolveTicket:
    """Test POST /api/support/ticket/<id>/resolve."""

    def test_resolve_success(self, client, sample_ticket):
        r = client.post('/api/support/ticket', json=sample_ticket)
        tid = r.get_json()['ticket_id']

        resp = client.post(f'/api/support/ticket/{tid}/resolve')
        assert resp.status_code == 200
        assert resp.get_json()['success'] is True

        ticket = client.get(f'/api/support/ticket/{tid}').get_json()
        assert ticket['status'] == 'resolved'
        assert ticket['resolved_date'] is not None

    def test_resolve_nonexistent(self, client):
        resp = client.post('/api/support/ticket/00000000-0000-0000-0000-000000000099/resolve')
        assert resp.status_code == 404
