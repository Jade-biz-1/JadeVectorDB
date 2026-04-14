"""
User management endpoints + auth endpoints.

Auth endpoints (no JWT required):
  POST /api/auth/login
  POST /api/auth/change-password   (requires valid JWT of any role)

Admin-only endpoints (require JWT with role=admin):
  GET    /api/users
  POST   /api/users
  DELETE /api/users/{id}
  POST   /api/users/{id}/reset-password
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status

from ..models.schemas import (
    ChangePasswordRequest,
    CreateUserResponse,
    LoginRequest,
    LoginResponse,
    ResetPasswordResponse,
    UserCreate,
    UserListResponse,
    UserResponse,
)
from ..security import (
    create_access_token,
    generate_password,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)
from ..services.metadata_db import MetadataDB
from ..utils.config import settings
from ..logging_config import get_logger

log = get_logger(__name__)

# Shared DB instance — module-level so it matches the one in rag_service / mock_service
_db = MetadataDB(settings.metadata_db_path)

router = APIRouter(prefix="/api", tags=["auth & users"])


# ── Helpers ───────────────────────────────────────────────────

def _row_to_response(row: dict) -> UserResponse:
    return UserResponse(
        id=row["id"],
        username=row["username"],
        email=row["email"],
        role=row["role"],
        must_change_password=bool(row["must_change_password"]),
        created_at=row["created_at"],
        last_login=row.get("last_login"),
        is_active=bool(row["is_active"]),
    )


# ── Auth endpoints ────────────────────────────────────────────

@router.post("/auth/login", response_model=LoginResponse)
async def login(body: LoginRequest):
    """Authenticate and receive a JWT."""
    user = _db.user_get_by_username(body.username)
    if not user or not bool(user["is_active"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    _db.user_update(user["id"], {"last_login": datetime.now(timezone.utc).isoformat()})

    token = create_access_token({
        "sub": user["id"],
        "username": user["username"],
        "role": user["role"],
        "must_change_password": bool(user["must_change_password"]),
    })
    log.info("user_login", username=user["username"])
    return LoginResponse(
        access_token=token,
        must_change_password=bool(user["must_change_password"]),
        user=_row_to_response(user),
    )


@router.post("/auth/change-password")
async def change_password(
    body: ChangePasswordRequest,
    current: dict = Depends(get_current_user),
):
    """
    Self-service password change. Validates current password, then updates.
    Clears must_change_password flag on success.
    """
    if body.new_password != body.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")

    user = _db.user_get_by_id(current["sub"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if not verify_password(body.current_password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    _db.user_update(user["id"], {
        "hashed_password": hash_password(body.new_password),
        "must_change_password": 0,
    })
    log.info("password_changed", username=user["username"])
    return {"success": True, "message": "Password updated successfully"}


# ── Admin: user management ────────────────────────────────────

@router.get("/users", response_model=UserListResponse)
async def list_users(
    offset: int = 0,
    limit: int = 100,
    _admin: dict = Depends(require_admin),
):
    """List all users (admin only)."""
    rows = _db.user_list(offset=offset, limit=limit)
    total = _db.user_count()
    return UserListResponse(
        users=[_row_to_response(r) for r in rows],
        total=total,
        offset=offset,
        limit=limit,
    )


@router.post("/users", response_model=CreateUserResponse, status_code=201)
async def create_user(
    body: UserCreate,
    admin: dict = Depends(require_admin),
):
    """
    Create a new user (admin only).
    A secure password is generated and returned once — it is not stored in plaintext.
    The user must change it on first login.
    """
    if _db.user_get_by_username(body.username):
        raise HTTPException(status_code=409, detail=f"Username '{body.username}' already exists")
    if _db.user_get_by_email(body.email):
        raise HTTPException(status_code=409, detail=f"Email '{body.email}' already in use")

    plain_password = generate_password()
    user_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    _db.user_insert({
        "id": user_id,
        "username": body.username,
        "email": body.email,
        "hashed_password": hash_password(plain_password),
        "role": body.role,
        "created_by": admin["sub"],
        "created_at": now,
    })

    log.info("user_created", username=body.username, created_by=admin["username"])
    row = _db.user_get_by_id(user_id)
    return CreateUserResponse(user=_row_to_response(row), generated_password=plain_password)


@router.delete("/users/{user_id}", status_code=200)
async def deactivate_user(
    user_id: str,
    admin: dict = Depends(require_admin),
):
    """Deactivate a user account (soft delete). Admins cannot deactivate themselves."""
    if user_id == admin["sub"]:
        raise HTTPException(status_code=400, detail="Cannot deactivate your own account")
    user = _db.user_get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    _db.user_delete(user_id)
    log.info("user_deactivated", user_id=user_id, by=admin["username"])
    return {"success": True, "message": f"User '{user['username']}' deactivated"}


@router.post("/users/{user_id}/reset-password", response_model=ResetPasswordResponse)
async def reset_user_password(
    user_id: str,
    _admin: dict = Depends(require_admin),
):
    """
    Admin resets a user's password. A new secure password is generated and
    returned once. The user must change it on next login.
    """
    user = _db.user_get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    plain_password = generate_password()
    _db.user_update(user_id, {
        "hashed_password": hash_password(plain_password),
        "must_change_password": 1,
    })
    log.info("password_reset", user_id=user_id, by=_admin["username"])
    return ResetPasswordResponse(
        user_id=user_id,
        username=user["username"],
        generated_password=plain_password,
    )
