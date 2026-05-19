"""
Tests for billing.refund_linkage — the deterministic charge<->customer<->
job-count linkage used to judge refunds on real data.

All Stripe + DB access is mocked, mirroring tests/test_billing.py.
`_ensure_stripe_key` is patched explicitly in every Stripe-touching test so
the suite never depends on an ambient STRIPE_SECRET_KEY env var. These tests
also assert the HARD invariant that the module never calls a Stripe write
method.
"""

import os
from unittest.mock import patch, MagicMock

os.environ.setdefault('STRIPE_SECRET_KEY', 'sk_test_dummy')

from billing.refund_linkage import (
    build_refund_evaluation_packet,
    RefundEvaluationPacket,
)


def _stripe_charge(**over):
    base = {
        'id': 'ch_1', 'customer': 'cus_1', 'amount': 1000,
        'currency': 'usd', 'livemode': True, 'created': 1700000000,
        'refunded': False, 'amount_refunded': 0,
    }
    base.update(over)
    return base


def _stripe_customer(**over):
    base = {'id': 'cus_1', 'email': 'player@example.com', 'deleted': False}
    base.update(over)
    return base


class TestInputGuards:
    def test_no_args_returns_not_ok_with_warning(self):
        pkt = build_refund_evaluation_packet()
        assert isinstance(pkt, RefundEvaluationPacket)
        assert pkt.ok is False
        assert any('No charge_id or customer_id' in w for w in pkt.warnings)

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    def test_both_args_prefers_charge_and_warns(self, mock_find, mock_disp,
                                                mock_cust, mock_charge, _k):
        mock_charge.retrieve.return_value = _stripe_charge()
        mock_charge.list.return_value = MagicMock(data=[])
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_disp.list.return_value = MagicMock(data=[])
        mock_find.return_value = {'id': 'u1', 'email': 'a@b.co',
                                  'plan': 'pro', 'created_at': 1690000000}
        with patch('billing.refund_linkage.get_monthly_usage', return_value=2):
            pkt = build_refund_evaluation_packet(charge_id='ch_1',
                                                 customer_id='cus_OTHER')
        assert any('Both charge_id and customer_id' in w for w in pkt.warnings)
        assert pkt.charge_id == 'ch_1'


class TestChargePath:
    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    def test_full_clean_linkage(self, mock_find, mock_disp, mock_cust,
                                mock_charge, _k):
        mock_charge.retrieve.return_value = _stripe_charge()
        mock_charge.list.return_value = MagicMock(data=[])  # no prior refunds
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_disp.list.return_value = MagicMock(data=[])
        mock_find.return_value = {
            'id': 'user-99', 'email': 'player@example.com',
            'plan': 'pro', 'created_at': 1690000000,
        }
        with patch('billing.refund_linkage.get_monthly_usage', return_value=7) as mu:
            pkt = build_refund_evaluation_packet(charge_id='ch_1')

        mu.assert_called_once_with('user-99')  # job count from DB helper
        assert pkt.ok is True
        assert pkt.stripe_customer_id == 'cus_1'
        assert pkt.user_id == 'user-99'
        assert pkt.email == 'player@example.com'
        assert pkt.plan == 'pro'
        assert pkt.charge_amount == 1000
        assert pkt.currency == 'usd'
        assert pkt.livemode is True
        assert pkt.job_count_current_period == 7
        assert pkt.prior_refund is False
        assert pkt.open_dispute is False
        # purchase date (charge.created) used, not account creation
        assert pkt.signup_or_purchase_date.startswith('2023-11-14')

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    def test_charge_not_found(self, mock_charge, _k):
        # In the test env `stripe` is a MagicMock, so the module resolves the
        # Stripe error classes to plain `Exception` (see refund_linkage
        # _resolve_exc). Raise a real Exception so the not-found path is
        # exercised; in production this is stripe.error.InvalidRequestError.
        from billing.refund_linkage import _InvalidRequestError
        mock_charge.retrieve.side_effect = _InvalidRequestError('No such charge')
        pkt = build_refund_evaluation_packet(charge_id='ch_missing')
        assert pkt.ok is False
        assert any('not found in Stripe' in w for w in pkt.warnings)

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    def test_charge_without_customer(self, mock_charge, _k):
        mock_charge.retrieve.return_value = _stripe_charge(customer=None)
        pkt = build_refund_evaluation_packet(charge_id='ch_1')
        assert pkt.ok is False
        assert pkt.charge_amount == 1000  # charge facts still returned
        assert any('no associated Stripe customer' in w for w in pkt.warnings)

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    @patch('billing.refund_linkage._find_user_row_by_email')
    def test_email_fallback_match_warns(self, mock_by_email, mock_by_cust,
                                        mock_disp, mock_cust, mock_charge, _k):
        mock_charge.retrieve.return_value = _stripe_charge()
        mock_charge.list.return_value = MagicMock(data=[])
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_disp.list.return_value = MagicMock(data=[])
        mock_by_cust.return_value = None  # not matched by customer id
        mock_by_email.return_value = {
            'id': 'u-email', 'email': 'player@example.com',
            'plan': 'pro', 'created_at': 1690000000,
        }
        with patch('billing.refund_linkage.get_monthly_usage', return_value=0):
            pkt = build_refund_evaluation_packet(charge_id='ch_1')
        assert pkt.user_id == 'u-email'
        assert any('matched by EMAIL' in w for w in pkt.warnings)

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=False)
    def test_no_stripe_key_charge_path_blocked(self, _k):
        pkt = build_refund_evaluation_packet(charge_id='ch_1')
        assert pkt.ok is False
        assert any('cannot resolve charge facts' in w for w in pkt.warnings)


class TestCustomerPath:
    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    def test_customer_id_uses_account_signup_date(self, mock_find, mock_disp,
                                                  mock_cust, mock_charge, _k):
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_charge.list.return_value = MagicMock(data=[])
        mock_disp.list.return_value = MagicMock(data=[])
        mock_find.return_value = {
            'id': 'user-c', 'email': 'player@example.com',
            'plan': 'lifetime', 'created_at': 1690000000,
        }
        with patch('billing.refund_linkage.get_monthly_usage', return_value=12):
            pkt = build_refund_evaluation_packet(customer_id='cus_1')
        assert pkt.ok is True
        assert pkt.charge_id is None  # no specific charge resolved
        assert pkt.charge_amount is None
        assert pkt.signup_or_purchase_date.startswith('2023-07-22')
        assert pkt.job_count_current_period == 12

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    @patch('billing.refund_linkage._find_user_row_by_email')
    def test_no_local_user_not_ok_and_warns(self, mock_by_email, mock_find,
                                            mock_disp, mock_cust, mock_charge, _k):
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_charge.list.return_value = MagicMock(data=[])
        mock_disp.list.return_value = MagicMock(data=[])
        mock_find.return_value = None
        mock_by_email.return_value = None
        pkt = build_refund_evaluation_packet(customer_id='cus_unknown')
        assert pkt.ok is False
        assert pkt.job_count_current_period is None
        assert any('No local user linked' in w for w in pkt.warnings)


class TestPriorRefundAndDispute:
    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    def test_prior_refund_detected_excludes_current_charge(self, mock_find,
                                                           mock_disp, mock_cust,
                                                           mock_charge, _k):
        mock_charge.retrieve.return_value = _stripe_charge(id='ch_current')
        # one OTHER charge fully refunded => prior_refund True
        mock_charge.list.return_value = MagicMock(data=[
            {'id': 'ch_current', 'refunded': True, 'amount_refunded': 1000},
            {'id': 'ch_old', 'refunded': True, 'amount_refunded': 500},
        ])
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_disp.list.return_value = MagicMock(data=[])
        mock_find.return_value = {'id': 'u1', 'email': 'a@b.co',
                                  'plan': 'pro', 'created_at': 1690000000}
        with patch('billing.refund_linkage.get_monthly_usage', return_value=1):
            pkt = build_refund_evaluation_packet(charge_id='ch_current')
        assert pkt.prior_refund is True

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    def test_only_current_charge_refunded_is_not_prior(self, mock_find,
                                                       mock_disp, mock_cust,
                                                       mock_charge, _k):
        mock_charge.retrieve.return_value = _stripe_charge(id='ch_current')
        mock_charge.list.return_value = MagicMock(data=[
            {'id': 'ch_current', 'refunded': True, 'amount_refunded': 1000},
        ])
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_disp.list.return_value = MagicMock(data=[])
        mock_find.return_value = {'id': 'u1', 'email': 'a@b.co',
                                  'plan': 'pro', 'created_at': 1690000000}
        with patch('billing.refund_linkage.get_monthly_usage', return_value=1):
            pkt = build_refund_evaluation_packet(charge_id='ch_current')
        assert pkt.prior_refund is False

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    @patch('billing.refund_linkage.stripe.Charge')
    @patch('billing.refund_linkage.stripe.Customer')
    @patch('billing.refund_linkage.stripe.Dispute')
    @patch('billing.refund_linkage._find_user_row_by_customer')
    def test_open_dispute_detected(self, mock_find, mock_disp, mock_cust,
                                   mock_charge, _k):
        mock_charge.list.return_value = MagicMock(data=[])
        mock_cust.retrieve.return_value = _stripe_customer()
        mock_disp.list.return_value = MagicMock(data=[
            {'status': 'needs_response', 'charge': 'ch_disp'},
        ])
        # charge.retrieve is called twice: once for the evaluated charge, once
        # to resolve the disputed charge's customer.
        def _retrieve(cid):
            if cid == 'ch_disp':
                return {'customer': 'cus_1'}
            return _stripe_charge()
        mock_charge.retrieve.side_effect = _retrieve
        mock_find.return_value = {'id': 'u1', 'email': 'a@b.co',
                                  'plan': 'pro', 'created_at': 1690000000}
        with patch('billing.refund_linkage.get_monthly_usage', return_value=1):
            pkt = build_refund_evaluation_packet(charge_id='ch_1')
        assert pkt.open_dispute is True


class TestReadOnlyInvariant:
    """The module must NEVER expose or call a Stripe write surface."""

    @patch('billing.refund_linkage._ensure_stripe_key', return_value=True)
    def test_no_write_methods_called(self, _k):
        with patch('billing.refund_linkage.stripe.Charge') as mc, \
             patch('billing.refund_linkage.stripe.Customer') as mcu, \
             patch('billing.refund_linkage.stripe.Dispute') as md, \
             patch('billing.refund_linkage._find_user_row_by_customer') as mf:
            mc.retrieve.return_value = _stripe_charge()
            mc.list.return_value = MagicMock(data=[])
            mcu.retrieve.return_value = _stripe_customer()
            md.list.return_value = MagicMock(data=[])
            mf.return_value = {'id': 'u1', 'email': 'a@b.co',
                               'plan': 'pro', 'created_at': 1690000000}
            with patch('billing.refund_linkage.get_monthly_usage', return_value=3):
                build_refund_evaluation_packet(charge_id='ch_1')

            # Only read methods may be touched.
            for mock_obj, allowed in (
                (mc, {'retrieve', 'list'}),
                (mcu, {'retrieve'}),
                (md, {'list'}),
            ):
                called = {c[0] for c in mock_obj.method_calls}
                forbidden = called - allowed
                assert not forbidden, (
                    f"refund_linkage called non-read Stripe methods: {forbidden}")
            # Explicitly assert no money-moving / mutating calls exist.
            for verb in ('create', 'modify', 'delete', 'cancel', 'update'):
                assert not getattr(mc, verb).called
                assert not getattr(mcu, verb).called

    def test_source_has_no_write_tokens(self):
        import billing.refund_linkage as mod
        src = open(mod.__file__).read()
        for bad in ('stripe.Refund.create', 'stripe.Charge.create',
                    'stripe.Subscription.cancel', 'stripe.Subscription.delete',
                    'stripe.Customer.modify', '.refund(', '.capture('):
            assert bad not in src, f"write-surface token present: {bad}"
