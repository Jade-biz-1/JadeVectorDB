"""
EnterpriseRAG - FastAPI Application Entry Point
"""

import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

from .api import query, admin
from .utils.config import settings, is_mock_mode
from .rate_limiter import limiter
from .logging_config import configure_logging, get_logger
from .services.rag_service import production_service

# Initialise structured logging before anything else
configure_logging()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle."""
    log.info(
        "startup",
        app=settings.app_name,
        version=settings.app_version,
        mode=settings.mode,
        auth_enabled=bool(settings.rag_api_key),
    )
    yield
    # Close persistent HTTP clients on shutdown
    await production_service.aclose()
    log.info("shutdown", app=settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG system for maintenance documentation Q&A",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── Rate limiter ──────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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


@app.get("/")
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "mode": "mock" if is_mock_mode() else "production",
        "docs": "/docs",
        "health": "/api/health",
    }
