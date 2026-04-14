"""
Authentication for EnterpriseRAG.

Two auth modes:
  1. API-key auth (RAG_API_KEY env var) — used by the existing admin document endpoints.
  2. JWT auth — used by user management and the query endpoint when JWT_SECRET_KEY is set.

Password hashing uses bcrypt via passlib.
"""

import secrets
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

from .utils.config import settings

# ── Password hashing ──────────────────────────────────────────
_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)

def generate_password(length: int = 12) -> str:
    """Generate a random secure password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
    # Guarantee at least one of each character class
    pwd = [
        secrets.choice(string.ascii_uppercase),
        secrets.choice(string.ascii_lowercase),
        secrets.choice(string.digits),
        secrets.choice("!@#$%^&*"),
    ]
    pwd += [secrets.choice(alphabet) for _ in range(length - 4)]
    secrets.SystemRandom().shuffle(pwd)
    return "".join(pwd)


# ── JWT ───────────────────────────────────────────────────────
def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expire_minutes)
    payload["exp"] = expire
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)

def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Bearer extraction (shared for both API-key and JWT) ───────
_bearer = HTTPBearer(auto_error=False)
_oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """
    FastAPI dependency that validates the Bearer token against RAG_API_KEY.
    If RAG_API_KEY is not set, authentication is disabled (useful for local dev).
    """
    if not settings.rag_api_key:
        return  # Auth disabled

    if credentials is None or credentials.credentials != settings.rag_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """
    FastAPI dependency — validates JWT and returns the decoded payload.
    Raises 401 if the token is missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_access_token(credentials.credentials)


async def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    """FastAPI dependency — requires the caller to have role='admin'."""
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
