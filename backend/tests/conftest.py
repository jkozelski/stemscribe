"""
Shared pytest fixtures for backend tests.

Handles mocking of heavy ML dependencies and optional infrastructure libs
so that app.py can be imported without requiring scipy, torch, demucs,
basic_pitch, audio_separator, librosa, psycopg2, boto3, stripe, etc.

IMPORTANT: This file runs before any test module is collected. It installs
mock modules into sys.modules at import time so that `import app` succeeds
even without the real libraries installed.
"""

import sys
import os
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# 0. Set default env vars needed by auth/billing modules during import
# ---------------------------------------------------------------------------
os.environ.setdefault('JWT_SECRET_KEY', 'test-secret-key-for-pytest')
os.environ.setdefault('STRIPE_PRICE_PREMIUM_MONTHLY', 'price_test_pm')
os.environ.setdefault('STRIPE_PRICE_PREMIUM_ANNUAL', 'price_test_pa')
os.environ.setdefault('STRIPE_PRICE_PRO_MONTHLY', 'price_test_prm')
os.environ.setdefault('STRIPE_PRICE_PRO_ANNUAL', 'price_test_pra')


# ---------------------------------------------------------------------------
# 1. Build a minimal mock torch with a real nn.Module class
#    (MagicMock can't be used as a base class for real class definitions)
# ---------------------------------------------------------------------------

class _FakeModule:
    """Minimal stand-in for torch.nn.Module so that class Foo(nn.Module) works."""
    def __init__(self, *args, **kwargs):
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


# torch mock
_mock_torch = MagicMock()

# torch.nn must be a module-like object with a real Module class
_mock_nn = types.ModuleType('torch.nn')
_mock_nn.Module = _FakeModule
_mock_nn.Linear = MagicMock()
_mock_nn.LayerNorm = MagicMock()
_mock_nn.Dropout = MagicMock()
_mock_nn.TransformerEncoder = MagicMock()
_mock_nn.TransformerEncoderLayer = MagicMock()
_mock_nn.ReLU = MagicMock()
_mock_nn.GELU = MagicMock()
_mock_nn.BatchNorm1d = MagicMock()
_mock_nn.Sequential = MagicMock()
_mock_nn.Conv1d = MagicMock()
_mock_nn.Conv2d = MagicMock()
_mock_nn.MaxPool1d = MagicMock()
_mock_nn.MaxPool2d = MagicMock()
_mock_nn.GRU = MagicMock()
_mock_nn.LSTM = MagicMock()
_mock_nn.Embedding = MagicMock()
_mock_nn.Sigmoid = MagicMock()
_mock_nn.Softmax = MagicMock()
_mock_nn.functional = MagicMock()

_mock_torch.nn = _mock_nn

# torch.nn.functional
_mock_nn_functional = MagicMock()
_mock_nn.functional = _mock_nn_functional

# ---------------------------------------------------------------------------
# 2. Build scipy mock with proper submodule structure
# ---------------------------------------------------------------------------
_mock_scipy = MagicMock()
_mock_scipy_signal = MagicMock()
_mock_scipy_io = MagicMock()
_mock_scipy_io_wavfile = MagicMock()
_mock_scipy.signal = _mock_scipy_signal
_mock_scipy.io = _mock_scipy_io
_mock_scipy.io.wavfile = _mock_scipy_io_wavfile

# ---------------------------------------------------------------------------
# 3. Build psycopg2 mock with pool and extras submodules
# ---------------------------------------------------------------------------
_mock_psycopg2 = MagicMock()
_mock_psycopg2_pool = MagicMock()
_mock_psycopg2_extras = MagicMock()
_mock_psycopg2_extras.RealDictCursor = MagicMock()
_mock_psycopg2.pool = _mock_psycopg2_pool
_mock_psycopg2.extras = _mock_psycopg2_extras

# ---------------------------------------------------------------------------
# 3b. Build passlib mock with a working bcrypt hasher
# ---------------------------------------------------------------------------
_mock_passlib = MagicMock()
_mock_passlib_hash = MagicMock()
# Make bcrypt.verify return True and bcrypt.using().hash return a fake hash
_mock_bcrypt = MagicMock()
_mock_bcrypt.verify.return_value = True
_mock_bcrypt.using.return_value.hash.return_value = '$2b$12$fakehashfortest'
_mock_passlib_hash.bcrypt = _mock_bcrypt
_mock_passlib.hash = _mock_passlib_hash

# ---------------------------------------------------------------------------
# 3c. Build botocore mock for R2 client Config
# ---------------------------------------------------------------------------
_mock_botocore = MagicMock()
_mock_botocore_config = MagicMock()
_mock_botocore_exceptions = MagicMock()
# ClientError needs to be an actual exception class for except clauses
class _FakeClientError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__("mock client error")
        self.response = {'Error': {'Code': '404', 'Message': 'Not Found'}}
_mock_botocore_exceptions.ClientError = _FakeClientError
_mock_botocore.config = _mock_botocore_config
_mock_botocore.exceptions = _mock_botocore_exceptions

# ---------------------------------------------------------------------------
# 3d. Build stripe mock with proper error classes
# ---------------------------------------------------------------------------
_mock_stripe = MagicMock()
class _FakeStripeSignatureError(Exception):
    pass
_mock_stripe_error = MagicMock()
_mock_stripe_error.SignatureVerificationError = _FakeStripeSignatureError
_mock_stripe.error = _mock_stripe_error

# ---------------------------------------------------------------------------
# 4. List of all heavy modules to mock
# ---------------------------------------------------------------------------
_MODULE_MOCKS = {
    # scipy
    'scipy': _mock_scipy,
    'scipy.signal': _mock_scipy_signal,
    'scipy.io': _mock_scipy_io,
    'scipy.io.wavfile': _mock_scipy_io_wavfile,
    # torch
    'torch': _mock_torch,
    'torch.nn': _mock_nn,
    'torch.nn.functional': _mock_nn_functional,
    'torchaudio': MagicMock(),
    # ML/audio libraries
    'librosa': MagicMock(),
    'librosa.display': MagicMock(),
    'demucs': MagicMock(),
    'basic_pitch': MagicMock(),
    'basic_pitch.inference': MagicMock(),
    'audio_separator': MagicMock(),
    'audio_separator.separator': MagicMock(),
    # Optional service modules that may not exist
    'drive_service': MagicMock(),
    # Database (psycopg2) — only mock if not installed
    'psycopg2': _mock_psycopg2,
    'psycopg2.pool': _mock_psycopg2_pool,
    'psycopg2.extras': _mock_psycopg2_extras,
    # Password hashing
    'passlib': _mock_passlib,
    'passlib.hash': _mock_passlib_hash,
    # Email validation
    'email_validator': MagicMock(),
    # Cloud storage (boto3 / botocore)
    'boto3': MagicMock(),
    'botocore': _mock_botocore,
    'botocore.config': _mock_botocore_config,
    'botocore.exceptions': _mock_botocore_exceptions,
    # Billing (Stripe)
    'stripe': _mock_stripe,
    'stripe.error': _mock_stripe_error,
    # GPU processing
    'replicate': MagicMock(),
    # Email sending
    'resend': MagicMock(),
}

# ---------------------------------------------------------------------------
# 5. Install mocks into sys.modules (only if not already loaded for real)
# ---------------------------------------------------------------------------
for _mod_name, _mock_obj in _MODULE_MOCKS.items():
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _mock_obj

# ---------------------------------------------------------------------------
# 6. Ensure backend directory is on sys.path
# ---------------------------------------------------------------------------
_backend_dir = os.path.join(os.path.dirname(__file__), '..')
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)
