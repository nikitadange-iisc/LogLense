"""
Auth helpers: user whitelist from env + JWT creation/verification.

Configure via .env:
    ALLOWED_USERS=admin:secret,alice:pass456
    JWT_SECRET=change-this-in-production
"""
import os
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

SECRET_KEY       = os.getenv("JWT_SECRET", "logsense-dev-secret-change-in-prod")
ALGORITHM        = "HS256"
TOKEN_EXPIRE_HRS = 24 * 7   # 7-day tokens so users don't get logged out constantly


def _load_users() -> dict:
    """Parse ALLOWED_USERS=user1:pass1,user2:pass2 from env."""
    raw = os.getenv("ALLOWED_USERS", "admin:admin")
    users = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if ":" in entry:
            username, password = entry.split(":", 1)
            users[username.strip()] = password.strip()
    return users


USERS: dict = _load_users()


def verify_user(username: str, password: str) -> bool:
    expected = USERS.get(username)
    return expected is not None and password == expected


def create_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HRS)
    return jwt.encode({"sub": username, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> str:
    """Return username. Raises JWTError on invalid/expired token."""
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload["sub"]
