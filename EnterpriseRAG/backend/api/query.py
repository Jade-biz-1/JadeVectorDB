"""
Query API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from ..models.schemas import QueryRequest, QueryResponse, SystemStats, HealthResponse
from ..utils.config import is_mock_mode
from ..services.mock_service import mock_service
from ..services.rag_service import production_service
from ..security import verify_api_key
from ..rate_limiter import limiter
from ..logging_config import get_logger
from datetime import datetime

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["query"])
_protected = [Depends(verify_api_key)]


@router.post("/query", response_model=QueryResponse, dependencies=_protected)
@limiter.limit("20/minute")
async def query_documents(request: Request, body: QueryRequest):
    """
    Query maintenance documentation using RAG.
    Limited to 20 requests per minute per IP.
    """
    try:
        svc = mock_service if is_mock_mode() else production_service
        return await svc.query(
            question=body.question,
            device_type=body.device_type,
            top_k=body.top_k,
        )
    except Exception as e:
        log.error("query_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/stats", response_model=SystemStats, dependencies=_protected)
async def get_stats():
    """Get system statistics."""
    try:
        svc = mock_service if is_mock_mode() else production_service
        return await svc.get_stats()
    except Exception as e:
        log.error("stats_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check — public endpoint, no auth required."""
    mode = "mock" if is_mock_mode() else "production"

    if is_mock_mode():
        components = {"database": "simulated", "llm": "simulated", "embeddings": "simulated"}
    else:
        try:
            stats = await production_service.get_stats()
            components = {
                "database": stats.db_status,
                "llm": stats.llm_status,
                "embeddings": stats.llm_status,
            }
        except Exception:
            components = {"database": "unknown", "llm": "unknown", "embeddings": "unknown"}

    status = "healthy" if all(v in ("healthy", "simulated") for v in components.values()) else "degraded"
    return HealthResponse(
        status=status,
        timestamp=datetime.utcnow().isoformat() + "Z",
        mode=mode,
        components=components,
    )
