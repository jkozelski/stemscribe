"""
User model — CRUD operations against Supabase/Postgres.

No ORM; plain SQL via db.py helpers.
"""

import uuid
import logging
from datetime import datetime, timezone

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
            'created_at': self.created_at.isoformat() if self.created_at else None,
        }

    def check_password(self, password: str) -> bool:
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
