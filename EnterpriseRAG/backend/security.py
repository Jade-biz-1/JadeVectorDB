"""
API key authentication for EnterpriseRAG endpoints.
Set RAG_API_KEY env var to enable. Leave unset to disable auth (dev mode).
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from .utils.config import settings

_bearer = HTTPBearer(auto_error=False)


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
