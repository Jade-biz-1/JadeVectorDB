"""
EnterpriseRAG - FastAPI Application Entry Point
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler
from prometheus_fastapi_instrumentator import Instrumentator

from .api import query, admin
from .api import users as users_api
from .utils.config import settings, is_mock_mode
from .rate_limiter import limiter
from .logging_config import configure_logging, get_logger
from .services.rag_service import production_service
from .services.metadata_db import MetadataDB
from .security import hash_password
from . import metrics as _metrics  # noqa: F401 — registers all metric objects

# Initialise structured logging before anything else
configure_logging()
log = get_logger(__name__)


def _bootstrap_admin():
    """Create default admin account if no users exist."""
    db = MetadataDB(settings.metadata_db_path)
    if db.user_exists():
        return

    import uuid
    from datetime import datetime, timezone

    admin_id = str(uuid.uuid4())
    db.user_insert({
        "id": admin_id,
        "username": settings.admin_username,
        "email": f"{settings.admin_username}@localhost",
        "hashed_password": hash_password(settings.admin_default_password),
        "role": "admin",
        "created_by": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    log.info(
        "bootstrap_admin_created",
        username=settings.admin_username,
        note="Change password on first login",
    )
    print(f"\n{'='*60}")
    print("  Bootstrap admin account created:")
    print(f"  Username : {settings.admin_username}")
    print(f"  Password : {settings.admin_default_password}")
    print("  ⚠️  Change this password immediately after first login!")
    print(f"{'='*60}\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle."""
    _bootstrap_admin()
    log.info(
        "startup",
        app=settings.app_name,
        version=settings.app_version,
        mode=settings.mode,
        auth_enabled=bool(settings.rag_api_key),
    )
    if not is_mock_mode():
        await production_service.ensure_ready()
    yield
    # Close persistent HTTP clients on shutdown
    await production_service.aclose()
    log.info("shutdown", app=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG system for organizational document Q&A",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Rate limiter ──────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Prometheus metrics (/metrics) ─────────────────────────────
Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    excluded_handlers=["/metrics", "/api/health"],
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request logging middleware ────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000)
    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=duration_ms,
        client=request.client.host if request.client else "unknown",
    )
    return response

# ── Routers ───────────────────────────────────────────────────
app.include_router(query.router)
app.include_router(admin.router)
app.include_router(users_api.router)


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "mode": "mock" if is_mock_mode() else "production",
        "docs": "/docs",
        "health": "/api/health",
    }
