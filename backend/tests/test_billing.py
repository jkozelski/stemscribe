"""
Tests for the billing module — plans, routes, and webhook handling.

These tests mock Stripe API calls and database access so they can
run without external services.
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

# Set test env vars before any billing imports
os.environ.setdefault('STRIPE_PRICE_PREMIUM_MONTHLY', 'price_test_pm')
os.environ.setdefault('STRIPE_PRICE_PREMIUM_ANNUAL', 'price_test_pa')
os.environ.setdefault('STRIPE_PRICE_PRO_MONTHLY', 'price_test_prm')
os.environ.setdefault('STRIPE_PRICE_PRO_ANNUAL', 'price_test_pra')

from billing.plans import PLANS, get_price_id, plan_from_price_id, get_stripe_prices


class TestPlans:
    """Test plan definitions and price ID lookups."""

    def test_three_plans_defined(self):
        assert set(PLANS.keys()) == {'free', 'premium', 'pro'}

    def test_free_plan_is_free(self):
        assert PLANS['free']['monthly_price'] == 0
        assert PLANS['free']['songs_per_month'] == 3
        assert PLANS['free']['max_duration_sec'] == 300

    def test_premium_plan_pricing(self):
        assert PLANS['premium']['monthly_price'] == 4.99
        assert PLANS['premium']['annual_price'] == 39.99
        assert PLANS['premium']['songs_per_month'] == 50

    def test_pro_plan_unlimited(self):
        assert PLANS['pro']['songs_per_month'] == -1  # unlimited
        assert PLANS['pro']['max_duration_sec'] == 1800

    def test_get_price_id_valid(self):
        assert get_price_id('premium', 'monthly') == 'price_test_pm'
        assert get_price_id('premium', 'annual') == 'price_test_pa'
        assert get_price_id('pro', 'monthly') == 'price_test_prm'
        assert get_price_id('pro', 'annual') == 'price_test_pra'

    def test_get_price_id_free_returns_none(self):
        assert get_price_id('free', 'monthly') is None

    def test_get_price_id_invalid_plan(self):
        assert get_price_id('enterprise', 'monthly') is None

    def test_plan_from_price_id_valid(self):
        assert plan_from_price_id('price_test_pm') == 'premium'
        assert plan_from_price_id('price_test_pa') == 'premium'
        assert plan_from_price_id('price_test_prm') == 'pro'
        assert plan_from_price_id('price_test_pra') == 'pro'

    def test_plan_from_price_id_unknown(self):
        assert plan_from_price_id('price_unknown_123') is None

    def test_plan_from_price_id_empty(self):
        assert plan_from_price_id('') is None

    def test_get_stripe_prices_returns_dict(self):
        prices = get_stripe_prices()
        assert isinstance(prices, dict)
        assert 'premium_monthly' in prices
        assert 'pro_annual' in prices


class TestWebhookEventHandlers:
    """Test the individual webhook event handler functions."""

    @patch('billing.webhooks.update_user_plan')
    @patch('billing.webhooks.get_user_by_id')
    def test_checkout_completed_activates_plan(self, mock_get_user, mock_update):
        from billing.webhooks import _handle_checkout_completed

        mock_user = MagicMock()
        mock_user.id = 'user-123'
        mock_get_user.return_value = mock_user

        session = {
            'metadata': {'user_id': 'user-123', 'plan': 'premium'},
            'customer': 'cus_abc123',
            'subscription': 'sub_xyz789',
        }

        _handle_checkout_completed(session)

        mock_update.assert_called_once_with(
            user_id='user-123',
            plan='premium',
            stripe_customer_id='cus_abc123',
            stripe_subscription_id='sub_xyz789',
        )

    @patch('billing.webhooks.update_user_plan')
    @patch('billing.webhooks.get_user_by_id')
    def test_checkout_completed_missing_metadata_no_crash(self, mock_get_user, mock_update):
        from billing.webhooks import _handle_checkout_completed

        session = {'metadata': {}, 'customer': 'cus_abc'}
        _handle_checkout_completed(session)
        mock_update.assert_not_called()

    @patch('billing.webhooks.update_user_plan')
    @patch('billing.webhooks._find_user_by_stripe_customer')
    def test_subscription_deleted_reverts_to_free(self, mock_find, mock_update):
        from billing.webhooks import _handle_subscription_deleted

        mock_user = MagicMock()
        mock_user.id = 'user-456'
        mock_find.return_value = mock_user

        subscription = {'customer': 'cus_del123'}
        _handle_subscription_deleted(subscription)

        mock_update.assert_called_once_with(
            user_id='user-456',
            plan='free',
            stripe_customer_id='cus_del123',
            stripe_subscription_id=None,
        )

    @patch('billing.webhooks.set_payment_failed')
    @patch('billing.webhooks._find_user_by_stripe_customer')
    def test_payment_failed_flags_user(self, mock_find, mock_set_failed):
        from billing.webhooks import _handle_payment_failed

        mock_user = MagicMock()
        mock_user.id = 'user-789'
        mock_find.return_value = mock_user

        invoice = {'customer': 'cus_fail123'}
        _handle_payment_failed(invoice)

        mock_set_failed.assert_called_once_with('user-789')

    @patch('billing.webhooks.clear_payment_failed')
    @patch('billing.webhooks._find_user_by_stripe_customer')
    def test_payment_succeeded_clears_flag(self, mock_find, mock_clear):
        from billing.webhooks import _handle_payment_succeeded

        mock_user = MagicMock()
        mock_user.id = 'user-789'
        mock_user.payment_failed_at = '2026-02-28T12:00:00Z'  # truthy
        mock_find.return_value = mock_user

        invoice = {'customer': 'cus_ok123'}
        _handle_payment_succeeded(invoice)

        mock_clear.assert_called_once_with('user-789')

    @patch('billing.webhooks.clear_payment_failed')
    @patch('billing.webhooks._find_user_by_stripe_customer')
    def test_payment_succeeded_no_flag_skips_clear(self, mock_find, mock_clear):
        from billing.webhooks import _handle_payment_succeeded

        mock_user = MagicMock()
        mock_user.id = 'user-789'
        mock_user.payment_failed_at = None  # no failure flag
        mock_find.return_value = mock_user

        invoice = {'customer': 'cus_ok123'}
        _handle_payment_succeeded(invoice)

        mock_clear.assert_not_called()

    @patch('billing.webhooks._find_user_by_stripe_customer')
    def test_subscription_deleted_unknown_customer_no_crash(self, mock_find):
        from billing.webhooks import _handle_subscription_deleted

        mock_find.return_value = None
        _handle_subscription_deleted({'customer': 'cus_unknown'})
        # Should not raise

    @patch('billing.webhooks.update_user_plan')
    @patch('billing.webhooks._find_user_by_stripe_customer')
    def test_subscription_updated_changes_plan(self, mock_find, mock_update):
        from billing.webhooks import _handle_subscription_updated

        mock_user = MagicMock()
        mock_user.id = 'user-111'
        mock_user.plan = 'premium'
        mock_find.return_value = mock_user

        subscription = {
            'customer': 'cus_upd',
            'id': 'sub_upd',
            'items': {
                'data': [{'price': {'id': 'price_test_prm'}}]
            },
            'metadata': {},
        }
        _handle_subscription_updated(subscription)

        mock_update.assert_called_once_with(
            user_id='user-111',
            plan='pro',
            stripe_customer_id='cus_upd',
            stripe_subscription_id='sub_upd',
        )

    @patch('billing.webhooks.update_user_plan')
    @patch('billing.webhooks._find_user_by_stripe_customer')
    def test_subscription_updated_same_plan_no_update(self, mock_find, mock_update):
        from billing.webhooks import _handle_subscription_updated

        mock_user = MagicMock()
        mock_user.id = 'user-111'
        mock_user.plan = 'pro'
        mock_find.return_value = mock_user

        subscription = {
            'customer': 'cus_same',
            'id': 'sub_same',
            'items': {
                'data': [{'price': {'id': 'price_test_prm'}}]
            },
            'metadata': {},
        }
        _handle_subscription_updated(subscription)

        mock_update.assert_not_called()
