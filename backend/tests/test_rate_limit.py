"""
Tests for the rate limiting middleware.

Tests cover:
    - Flask-Limiter initialization and 429 handler
    - Plan-based song quota enforcement (enforce_plan_limits)
    - Duration limit enforcement (enforce_duration_limit)
    - Usage recording (record_usage_event)
    - IP extraction for rate limiting keys
"""

import os
from unittest.mock import patch, MagicMock

import pytest
from flask import Flask, g

# Set test env vars before imports
os.environ.setdefault('STRIPE_PRICE_PREMIUM_MONTHLY', 'price_test_pm')
os.environ.setdefault('STRIPE_PRICE_PREMIUM_ANNUAL', 'price_test_pa')
os.environ.setdefault('STRIPE_PRICE_PRO_MONTHLY', 'price_test_prm')
os.environ.setdefault('STRIPE_PRICE_PRO_ANNUAL', 'price_test_pra')


@pytest.fixture
def app():
    """Create a minimal Flask app for testing middleware."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestInitLimiter:
    """Test Flask-Limiter initialization."""

    def test_init_limiter_registers_429_handler(self, app):
        from middleware.rate_limit import init_limiter
        init_limiter(app)
        assert 429 in app.error_handler_spec[None]

    def test_limiter_attaches_to_app(self, app):
        from middleware.rate_limit import init_limiter, limiter
        init_limiter(app)
        assert limiter is not None


class TestGetKey:
    """Test IP extraction for rate limiting."""

    def test_get_key_uses_forwarded_for(self, app):
        from middleware.rate_limit import _get_key
        with app.test_request_context(
            '/', headers={'X-Forwarded-For': '203.0.113.50, 70.41.3.18'}
        ):
            assert _get_key() == '203.0.113.50'

    def test_get_key_single_forwarded_ip(self, app):
        from middleware.rate_limit import _get_key
        with app.test_request_context(
            '/', headers={'X-Forwarded-For': '10.0.0.1'}
        ):
            assert _get_key() == '10.0.0.1'

    def test_get_key_falls_back_to_remote_addr(self, app):
        from middleware.rate_limit import _get_key
        with app.test_request_context('/'):
            key = _get_key()
            assert key is not None


class TestEnforcePlanLimits:
    """Test the enforce_plan_limits decorator."""

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_rate_limit')
    @patch('auth.decorators.get_client_ip_hash')
    def test_authenticated_user_under_limit_passes(
        self, mock_ip_hash, mock_check, mock_get_user, app
    ):
        from middleware.rate_limit import enforce_plan_limits

        mock_user = MagicMock()
        mock_user.plan = 'premium'
        mock_user.id = 'user-123'
        mock_get_user.return_value = mock_user
        mock_check.return_value = 5

        @app.route('/test-auth')
        @enforce_plan_limits
        def test_route():
            return 'ok'

        with app.test_client() as client:
            resp = client.get('/test-auth')
            assert resp.status_code == 200
            assert resp.data == b'ok'

        mock_check.assert_called_once_with(user=mock_user, ip_hash=None)

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_rate_limit')
    @patch('auth.decorators.get_client_ip_hash')
    def test_anonymous_user_under_limit_passes(
        self, mock_ip_hash, mock_check, mock_get_user, app
    ):
        from middleware.rate_limit import enforce_plan_limits

        mock_get_user.return_value = None
        mock_ip_hash.return_value = 'abc123hash'
        mock_check.return_value = 1

        @app.route('/test-anon')
        @enforce_plan_limits
        def test_route():
            return 'ok'

        with app.test_client() as client:
            resp = client.get('/test-anon')
            assert resp.status_code == 200

        mock_check.assert_called_once_with(user=None, ip_hash='abc123hash')

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_rate_limit')
    @patch('auth.decorators.get_client_ip_hash')
    def test_user_over_limit_returns_429(
        self, mock_ip_hash, mock_check, mock_get_user, app
    ):
        from middleware.rate_limit import enforce_plan_limits
        from auth.decorators import RateLimitExceeded

        mock_get_user.return_value = None
        mock_ip_hash.return_value = 'abc123hash'
        mock_check.side_effect = RateLimitExceeded(
            "You've used 3/3 songs this month.",
            plan='free',
            usage_count=3,
            limit=3,
        )

        @app.route('/test-over')
        @enforce_plan_limits
        def test_route():
            return 'should not reach'

        with app.test_client() as client:
            resp = client.get('/test-over')
            assert resp.status_code == 429
            data = resp.get_json()
            assert data['usage'] == 3
            assert data['limit'] == 3
            assert data['plan'] == 'free'
            assert 'upgrade_url' in data

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_rate_limit')
    @patch('auth.decorators.get_client_ip_hash')
    def test_pro_user_unlimited_passes(
        self, mock_ip_hash, mock_check, mock_get_user, app
    ):
        from middleware.rate_limit import enforce_plan_limits

        mock_user = MagicMock()
        mock_user.plan = 'pro'
        mock_user.id = 'user-pro'
        mock_get_user.return_value = mock_user
        mock_check.return_value = 500

        @app.route('/test-pro')
        @enforce_plan_limits
        def test_route():
            from flask import g
            assert g.plan_limits['songs_per_month'] == -1
            return 'ok'

        with app.test_client() as client:
            resp = client.get('/test-pro')
            assert resp.status_code == 200

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_rate_limit')
    @patch('auth.decorators.get_client_ip_hash')
    def test_sets_g_context(
        self, mock_ip_hash, mock_check, mock_get_user, app
    ):
        from middleware.rate_limit import enforce_plan_limits

        mock_user = MagicMock()
        mock_user.plan = 'premium'
        mock_user.id = 'user-ctx'
        mock_get_user.return_value = mock_user
        mock_check.return_value = 10

        @app.route('/test-ctx')
        @enforce_plan_limits
        def test_route():
            from flask import g
            assert g.current_user == mock_user
            assert g.ip_hash is None
            assert g.usage_count == 10
            assert g.plan_limits['songs_per_month'] == 50
            return 'ok'

        with app.test_client() as client:
            resp = client.get('/test-ctx')
            assert resp.status_code == 200


class TestEnforceDurationLimit:
    """Test duration limit enforcement."""

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_duration_limit')
    def test_under_limit_returns_none(self, mock_check_dur, mock_get_user, app):
        from middleware.rate_limit import enforce_duration_limit

        mock_get_user.return_value = None
        mock_check_dur.return_value = (True, 300)

        with app.test_request_context('/'):
            result = enforce_duration_limit(120)
            assert result is None

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_duration_limit')
    def test_over_limit_returns_413(self, mock_check_dur, mock_get_user, app):
        from middleware.rate_limit import enforce_duration_limit

        mock_get_user.return_value = None
        mock_check_dur.return_value = (False, 300)

        with app.test_request_context('/'):
            result = enforce_duration_limit(600)
            assert result is not None
            response, status = result
            assert status == 413
            data = response.get_json()
            assert data['duration'] == 600
            assert data['max_duration'] == 300
            assert data['plan'] == 'free'
            assert 'upgrade_url' in data

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_duration_limit')
    def test_premium_user_longer_limit(self, mock_check_dur, mock_get_user, app):
        from middleware.rate_limit import enforce_duration_limit

        mock_user = MagicMock()
        mock_user.plan = 'premium'
        mock_get_user.return_value = mock_user
        mock_check_dur.return_value = (True, 900)

        with app.test_request_context('/'):
            g.current_user = mock_user
            result = enforce_duration_limit(600)
            assert result is None

    @patch('auth.decorators.get_current_user')
    @patch('auth.decorators.check_duration_limit')
    def test_pro_user_over_30min_returns_413(self, mock_check_dur, mock_get_user, app):
        from middleware.rate_limit import enforce_duration_limit

        mock_user = MagicMock()
        mock_user.plan = 'pro'
        mock_get_user.return_value = mock_user
        mock_check_dur.return_value = (False, 1800)

        with app.test_request_context('/'):
            g.current_user = mock_user
            result = enforce_duration_limit(2000)
            assert result is not None
            _, status = result
            assert status == 413


class TestRecordUsageEvent:
    """Test usage event recording."""

    @patch('auth.models.record_usage')
    def test_records_for_authenticated_user(self, mock_record, app):
        from middleware.rate_limit import record_usage_event

        mock_user = MagicMock()
        mock_user.id = 'user-rec'

        with app.app_context():
            record_usage_event(
                user=mock_user,
                job_id='job-123',
                action='separation',
            )

        mock_record.assert_called_once_with(
            user_id='user-rec',
            anonymous_ip_hash=None,
            job_id='job-123',
            action='separation',
        )

    @patch('auth.models.record_usage')
    def test_records_for_anonymous_user(self, mock_record, app):
        from middleware.rate_limit import record_usage_event

        with app.app_context():
            record_usage_event(
                ip_hash='anon-hash-abc',
                job_id='job-456',
                action='separation',
            )

        mock_record.assert_called_once_with(
            user_id=None,
            anonymous_ip_hash='anon-hash-abc',
            job_id='job-456',
            action='separation',
        )

    @patch('auth.models.record_usage')
    def test_records_default_action(self, mock_record, app):
        from middleware.rate_limit import record_usage_event

        mock_user = MagicMock()
        mock_user.id = 'user-def'

        with app.app_context():
            record_usage_event(user=mock_user, job_id='job-789')

        _, kwargs = mock_record.call_args
        assert kwargs['action'] == 'separation'

    @patch('auth.models.record_usage')
    def test_records_custom_action(self, mock_record, app):
        from middleware.rate_limit import record_usage_event

        mock_user = MagicMock()
        mock_user.id = 'user-cust'

        with app.app_context():
            record_usage_event(
                user=mock_user,
                job_id='job-999',
                action='transcription',
            )

        _, kwargs = mock_record.call_args
        assert kwargs['action'] == 'transcription'


class TestModuleConstants:
    """Test that module-level constants are properly defined."""

    def test_auth_limit_string(self):
        from middleware.rate_limit import AUTH_LIMIT
        assert AUTH_LIMIT == "5 per minute"

    def test_webhook_limit_string(self):
        from middleware.rate_limit import WEBHOOK_LIMIT
        assert WEBHOOK_LIMIT == "30 per minute"

    def test_processing_limit_string(self):
        from middleware.rate_limit import PROCESSING_LIMIT
        assert PROCESSING_LIMIT == "10 per minute"
