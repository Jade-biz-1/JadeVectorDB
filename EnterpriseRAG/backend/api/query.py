"""
Query API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from ..models.schemas import AnalyticsResponse, QueryRecord, QueryRequest, QueryResponse, SystemStats, HealthResponse
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
    Query documents using RAG.
    Limited to 20 requests per minute per IP.
    """
    try:
        svc = mock_service if is_mock_mode() else production_service
        return await svc.query(
            question=body.question,
            category=body.category,
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


@router.get("/analytics", response_model=AnalyticsResponse, dependencies=_protected)
async def get_analytics(
    recent: int = Query(20, ge=1, le=100, description="Number of recent queries to return"),
):
    """Query analytics: totals, averages, category breakdown, and recent query log."""
    try:
        svc = mock_service if is_mock_mode() else production_service
        data = svc.get_analytics(recent_limit=recent)
        # Coerce recent_queries rows to QueryRecord (success is stored as int)
        records = [
            QueryRecord(
                id=r["id"],
                question=r["question"],
                category=r.get("category") or r.get("device_type") or "all",
                mode=r["mode"],
                confidence=r["confidence"] or 0,
                processing_time_ms=r["processing_time_ms"] or 0,
                sources_count=r["sources_count"] or 0,
                success=bool(r["success"]),
                timestamp=r["timestamp"],
            )
            for r in data["recent_queries"]
        ]
        return AnalyticsResponse(
            total_queries=data["total_queries"],
            avg_confidence=data["avg_confidence"],
            avg_processing_time_ms=data["avg_processing_time_ms"],
            success_rate=data["success_rate"],
            category_breakdown=data.get("category_breakdown") or data.get("device_type_breakdown", {}),
            recent_queries=records,
        )
    except Exception as e:
        log.error("analytics_endpoint_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


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
