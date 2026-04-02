"""
EnterpriseRAG - FastAPI Application Entry Point
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import query, admin
from .utils.config import settings, is_mock_mode

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="RAG system for maintenance documentation Q&A",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(query.router)
app.include_router(admin.router)


@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "mode": "mock" if is_mock_mode() else "production",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.on_event("startup")
async def startup_event():
    """
    Startup event handler
    """
    print("\n✅ EnterpriseRAG backend started successfully")
    print(f"   Mode: {settings.mode.upper()}")
    print(f"   API Documentation: http://{settings.api_host}:{settings.api_port}/docs")
    print(f"   Health Check: http://{settings.api_host}:{settings.api_port}/api/health")

    if not is_mock_mode():
        print(f"   JadeVectorDB: {settings.jadevectordb_url}")
        print(f"   Ollama: {settings.ollama_url}")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler
    """
    print("\n🛑 EnterpriseRAG backend shutting down")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
    )
