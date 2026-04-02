"""
Query API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException
from ..models.schemas import QueryRequest, QueryResponse, SystemStats, HealthResponse
from ..utils.config import is_mock_mode
from ..services.mock_service import mock_service
from ..services.rag_service import production_service
from ..security import verify_api_key
from datetime import datetime

router = APIRouter(prefix="/api", tags=["query"])

# /api/health is public; /api/query and /api/stats require auth
_protected = [Depends(verify_api_key)]


@router.post("/query", response_model=QueryResponse, dependencies=_protected)
async def query_documents(request: QueryRequest):
    """
    Query maintenance documentation

    **Modes:**
    - Mock: Returns simulated responses (no dependencies)
    - Production: Full RAG with JadeVectorDB + Ollama
    """
    try:
        if is_mock_mode():
            return await mock_service.query(
                question=request.question,
                device_type=request.device_type,
                top_k=request.top_k,
            )
        else:
            return await production_service.query(
                question=request.question,
                device_type=request.device_type,
                top_k=request.top_k,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/stats", response_model=SystemStats, dependencies=_protected)
async def get_stats():
    """
    Get system statistics
    """
    try:
        if is_mock_mode():
            return await mock_service.get_stats()
        else:
            return await production_service.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    """
    mode = "mock" if is_mock_mode() else "production"

    if is_mock_mode():
        components = {
            "database": "simulated",
            "llm": "simulated",
            "embeddings": "simulated",
        }
    else:
        # Check actual component health
        try:
            stats = await production_service.get_stats()
            components = {
                "database": stats.db_status,
                "llm": stats.llm_status,
                "embeddings": stats.llm_status,  # Ollama handles both
            }
        except Exception:
            components = {
                "database": "unknown",
                "llm": "unknown",
                "embeddings": "unknown",
            }

    return HealthResponse(
        status="healthy" if all(v in ["healthy", "simulated"] for v in components.values()) else "degraded",
        timestamp=datetime.utcnow().isoformat() + "Z",
        mode=mode,
        components=components,
    )
