"""
User model — CRUD operations against Supabase/Postgres.

No ORM; plain SQL via db.py helpers.
"""

import logging

from passlib.hash import bcrypt

from db import query_one, execute, execute_returning

logger = logging.getLogger(__name__)

# Minimum password length per NIST 800-63B
MIN_PASSWORD_LENGTH = 8


class User:
    """Lightweight user object hydrated from a database row dict."""

    __slots__ = (
        'id', 'email', 'password_hash', 'display_name', 'plan',
        'stripe_customer_id', 'stripe_subscription_id',
        'payment_failed_at', 'created_at', 'updated_at',
        'google_id', 'apple_id', 'avatar_url',
        'extras_balance', 'extras_last_purchase_at',
    )

    def __init__(self, row: dict):
        for key in self.__slots__:
            setattr(self, key, row.get(key))

    def to_dict(self):
        """Public-safe representation (no password hash)."""
        return {
            'id': str(self.id),
            'email': self.email,
            'display_name': self.display_name,
            'plan': self.plan,
            'stripe_customer_id': self.stripe_customer_id,
            'avatar_url': self.avatar_url,
            'extras_balance': int(self.extras_balance or 0),
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return bcrypt.verify(password, self.password_hash)


def hash_password(password: str) -> str:
    return bcrypt.using(rounds=12).hash(password)


def validate_password(password: str) -> str | None:
    """Return an error message if password is invalid, or None if OK."""
    if not password or len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters"
    return None


def create_user(email: str, password: str, display_name: str = None) -> User:
    """Create a new user. Raises ValueError if email is taken."""
    existing = get_user_by_email(email)
    if existing:
        raise ValueError("An account with this email already exists")

    pw_error = validate_password(password)
    if pw_error:
        raise ValueError(pw_error)

    row = execute_returning(
        """
        INSERT INTO users (email, password_hash, display_name)
        VALUES (%s, %s, %s)
        RETURNING *
        """,
        (email.lower().strip(), hash_password(password), display_name),
    )
    logger.info(f"Created user {row['id']} ({email})")
    return User(row)


def get_user_by_id(user_id: str) -> User | None:
    row = query_one("SELECT * FROM users WHERE id = %s", (user_id,))
    return User(row) if row else None


def get_user_by_email(email: str) -> User | None:
    row = query_one("SELECT * FROM users WHERE email = %s", (email.lower().strip(),))
    return User(row) if row else None


def update_user_plan(user_id: str, plan: str, stripe_customer_id: str = None,
                     stripe_subscription_id: str = None):
    """Update a user's plan and Stripe IDs."""
    execute(
        """
        UPDATE users
        SET plan = %s, stripe_customer_id = %s, stripe_subscription_id = %s,
            updated_at = NOW()
        WHERE id = %s
        """,
        (plan, stripe_customer_id, stripe_subscription_id, user_id),
    )
    logger.info(f"Updated user {user_id} to plan={plan}")


def set_user_plan(user_id: str, plan: str):
    """Set ONLY the plan column, leaving stripe_* IDs untouched (RevenueCat/Apple IAP)."""
    execute(
        "UPDATE users SET plan = %s, updated_at = NOW() WHERE id = %s",
        (plan, user_id),
    )
    logger.info(f"Set user {user_id} plan={plan} (plan-only, RevenueCat)")


def update_user_extras_balance(user_id: str, new_balance: int):
    """Set the user's extras_balance to an absolute value (used by the
    rate-limit decrement path when a song consumes one extra credit)."""
    execute(
        "UPDATE users SET extras_balance = %s, updated_at = NOW() WHERE id = %s",
        (max(0, int(new_balance)), user_id),
    )


def increment_user_extras_balance(user_id: str, delta: int) -> int:
    """Atomically add `delta` to extras_balance (called by the Stripe
    overage-pack webhook). Returns the new balance.

    Using SQL +%s rather than read-modify-write so concurrent webhook
    retries can't double-credit OR drop credits."""
    row = execute_returning(
        """
        UPDATE users
        SET extras_balance = COALESCE(extras_balance, 0) + %s,
            extras_last_purchase_at = NOW(),
            updated_at = NOW()
        WHERE id = %s
        RETURNING extras_balance
        """,
        (int(delta), user_id),
    )
    new_balance = int(row['extras_balance']) if row else 0
    logger.info(f"User {user_id} extras_balance += {delta} → {new_balance}")
    return new_balance


def update_user_password(user_id: str, new_password: str):
    """Set a new password for a user."""
    pw_error = validate_password(new_password)
    if pw_error:
        raise ValueError(pw_error)
    execute(
        "UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s",
        (hash_password(new_password), user_id),
    )
    logger.info(f"Password updated for user {user_id}")


def set_payment_failed(user_id: str):
    execute(
        "UPDATE users SET payment_failed_at = NOW(), updated_at = NOW() WHERE id = %s",
        (user_id,),
    )


def clear_payment_failed(user_id: str):
    execute(
        "UPDATE users SET payment_failed_at = NULL, updated_at = NOW() WHERE id = %s",
        (user_id,),
    )


def get_monthly_usage(user_id: str) -> int:
    """Count how many separation jobs this user has run this calendar month."""
    row = query_one(
        """
        SELECT COUNT(*) as cnt FROM usage
        WHERE user_id = %s
          AND action = 'separation'
          AND created_at >= date_trunc('month', NOW())
        """,
        (user_id,),
    )
    return row['cnt'] if row else 0


def get_anonymous_monthly_usage(ip_hash: str) -> int:
    """Count anonymous usage by IP hash this month."""
    row = query_one(
        """
        SELECT COUNT(*) as cnt FROM usage
        WHERE anonymous_ip_hash = %s
          AND action = 'separation'
          AND created_at >= date_trunc('month', NOW())
        """,
        (ip_hash,),
    )
    return row['cnt'] if row else 0


def record_usage(user_id: str = None, anonymous_ip_hash: str = None,
                 job_id: str = None, action: str = 'separation'):
    """Record a usage event for rate limiting."""
    execute(
        """
        INSERT INTO usage (user_id, anonymous_ip_hash, job_id, action)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, anonymous_ip_hash, job_id, action),
    )


# ---- Google OAuth helpers ----

def get_user_by_google_id(google_id: str) -> User | None:
    """Look up a user by their Google sub ID."""
    row = query_one("SELECT * FROM users WHERE google_id = %s", (google_id,))
    return User(row) if row else None


def create_google_user(email: str, display_name: str, google_id: str,
                       avatar_url: str = None) -> User:
    """Create a new user via Google OAuth (no password)."""
    existing = get_user_by_email(email)
    if existing:
        raise ValueError("An account with this email already exists")

    row = execute_returning(
        """
        INSERT INTO users (email, display_name, google_id, avatar_url)
        VALUES (%s, %s, %s, %s)
        RETURNING *
        """,
        (email.lower().strip(), display_name, google_id, avatar_url),
    )
    logger.info(f"Created Google user {row['id']} ({email})")
    return User(row)


def link_google_account(user_id: str, google_id: str, avatar_url: str = None):
    """Link a Google account to an existing user."""
    execute(
        """
        UPDATE users
        SET google_id = %s, avatar_url = COALESCE(%s, avatar_url), updated_at = NOW()
        WHERE id = %s
        """,
        (google_id, avatar_url, user_id),
    )
    logger.info(f"Linked Google account to user {user_id}")


def create_email_user(email: str, display_name: str = None) -> User:
    """Create a new passwordless user (magic-link sign-in). No password_hash,
    no google_id — the only auth handle is the email + magic-link token in
    their inbox. password_hash is nullable since migration 002."""
    existing = get_user_by_email(email)
    if existing:
        raise ValueError("An account with this email already exists")

    row = execute_returning(
        """
        INSERT INTO users (email, display_name)
        VALUES (%s, %s)
        RETURNING *
        """,
        (email.lower().strip(), display_name),
    )
    logger.info(f"Created passwordless user {row['id']} ({email})")
    return User(row)


# ---- Sign in with Apple helpers (added 2026-06-06 pre-launch sprint) -------

def get_user_by_apple_id(apple_id: str) -> User | None:
    """Look up a user by their Apple `sub` identifier (stable across sign-ins)."""
    row = query_one("SELECT * FROM users WHERE apple_id = %s", (apple_id,))
    return User(row) if row else None


def link_apple_account(user_id: str, apple_id: str):
    """Attach an Apple identity to an existing user account."""
    execute(
        "UPDATE users SET apple_id = %s, updated_at = NOW() WHERE id = %s",
        (apple_id, user_id),
    )
    logger.info(f"Linked Apple {apple_id[:8]}... to user {user_id}")


def create_apple_user(email: str, apple_id: str, display_name: str = None) -> User:
    """Create a new passwordless user from Apple. `password_hash` stays null."""
    existing = get_user_by_email(email)
    if existing:
        link_apple_account(str(existing.id), apple_id)
        return get_user_by_id(str(existing.id))
    row = execute_returning(
        """
        INSERT INTO users (email, display_name, apple_id)
        VALUES (%s, %s, %s)
        RETURNING *
        """,
        (email.lower().strip(), display_name, apple_id),
    )
    logger.info(f"Created Apple user {row['id']} ({email})")
    return User(row)
