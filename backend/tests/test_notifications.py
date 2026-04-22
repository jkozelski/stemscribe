"""
Tests for job-completion email notifications (backend/notifications.py).

The Resend SDK is mocked in conftest.py, so these tests assert on the mock's
call args rather than hitting the real API.
"""

import os
import sys
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def _make_job(**overrides):
    """Build a minimal duck-typed job object for notifications."""
    job = MagicMock()
    job.job_id = overrides.get('job_id', 'test-job-123')
    job.filename = overrides.get('filename', 'song.mp3')
    job.metadata = overrides.get('metadata', {'title': 'Test Song'})
    job.user_id = overrides.get('user_id', None)
    return job


class TestJobEmailsEnabled:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv('ENABLE_JOB_EMAILS', raising=False)
        from notifications import job_emails_enabled
        assert job_emails_enabled() is False

    def test_enabled_when_true(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'true')
        from notifications import job_emails_enabled
        assert job_emails_enabled() is True

    def test_enabled_accepts_various_truthy(self, monkeypatch):
        from notifications import job_emails_enabled
        for val in ('1', 'TRUE', 'yes', 'on', 'True'):
            monkeypatch.setenv('ENABLE_JOB_EMAILS', val)
            assert job_emails_enabled() is True, f"value {val!r} should be truthy"

    def test_disabled_when_false(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'false')
        from notifications import job_emails_enabled
        assert job_emails_enabled() is False


class TestSendJobCompleteEmail:
    def test_skips_when_disabled(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'false')
        from notifications import send_job_complete_email
        job = _make_job()
        assert send_job_complete_email(job) is False

    def test_skips_when_no_recipient(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'true')
        monkeypatch.setenv('RESEND_API_KEY', 're_test')
        from notifications import send_job_complete_email
        job = _make_job(user_id=None, metadata={'title': 'No User Song'})
        # No notify_email, no user_id → nothing to send to → False
        assert send_job_complete_email(job) is False

    def test_skips_when_no_api_key(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'true')
        monkeypatch.delenv('RESEND_API_KEY', raising=False)
        from notifications import send_job_complete_email
        job = _make_job(metadata={'title': 'Song', 'notify_email': 'jeff@example.com'})
        assert send_job_complete_email(job) is False

    def test_sends_with_explicit_notify_email(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'true')
        monkeypatch.setenv('RESEND_API_KEY', 're_test')
        monkeypatch.setenv('APP_URL', 'https://stemscriber.com')

        import resend  # conftest installs a MagicMock
        resend.Emails = MagicMock()
        resend.Emails.send = MagicMock(return_value={'id': 'sent-123'})

        from notifications import send_job_complete_email
        job = _make_job(metadata={
            'title': 'Alright',
            'notify_email': 'jeff@example.com',
        })
        assert send_job_complete_email(job) is True

        assert resend.Emails.send.called
        payload = resend.Emails.send.call_args.args[0]
        assert payload['to'] == ['jeff@example.com']
        assert 'Alright' in payload['subject']
        assert 'Alright' in payload['html']
        assert 'Alright' in payload['text']
        assert f"/practice.html?job={job.job_id}" in payload['html']

    def test_never_raises_on_send_error(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'true')
        monkeypatch.setenv('RESEND_API_KEY', 're_test')

        import resend
        resend.Emails = MagicMock()
        resend.Emails.send = MagicMock(side_effect=RuntimeError('boom'))

        from notifications import send_job_complete_email
        job = _make_job(metadata={'title': 'Song', 'notify_email': 'jeff@example.com'})
        # Must return False instead of raising — the pipeline must never break
        # on a notification failure.
        assert send_job_complete_email(job) is False

    def test_falls_back_to_user_email_when_no_notify(self, monkeypatch):
        monkeypatch.setenv('ENABLE_JOB_EMAILS', 'true')
        monkeypatch.setenv('RESEND_API_KEY', 're_test')

        import resend
        resend.Emails = MagicMock()
        resend.Emails.send = MagicMock(return_value={'id': 'sent-456'})

        fake_user = MagicMock()
        fake_user.email = 'user@example.com'
        fake_user.display_name = 'Jeff Kozelski'

        with patch('auth.models.get_user_by_id', return_value=fake_user):
            from notifications import send_job_complete_email
            job = _make_job(user_id='user-uuid', metadata={'title': 'Peg'})
            assert send_job_complete_email(job) is True

        payload = resend.Emails.send.call_args.args[0]
        assert payload['to'] == ['user@example.com']
        # Display-name-aware greeting: includes the first name
        assert ' Jeff,' in payload['text']
