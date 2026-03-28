#!/usr/bin/env python3
"""
RAG UI Alternative #3: FastAPI + React (Backend)
Modern REST API with proper async support

Usage:
    pip install fastapi uvicorn python-multipart
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API Docs: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
import time

# Initialize FastAPI app
app = FastAPI(
    title="Maintenance Documentation RAG API",
    description="REST API for querying maintenance documentation using RAG",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite/React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class QueryRequest(BaseModel):
    """RAG query request"""
    question: str = Field(..., min_length=1, description="User's question")
    device_type: Optional[str] = Field("all", description="Device type filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

    class Config:
        schema_extra = {
            "example": {
                "question": "How do I replace the hydraulic fluid?",
                "device_type": "hydraulic_pump",
                "top_k": 5
            }
        }


class SourceDocument(BaseModel):
    """Source document citation"""
    doc_name: str
    page_numbers: str
    section: str
    relevance: float = Field(ge=0, le=1)
    excerpt: str


class QueryResponse(BaseModel):
    """RAG query response"""
    success: bool = True
    answer: str
    sources: List[SourceDocument]
    query: str
    device_type: str
    timestamp: str
    confidence: float
    processing_time_ms: int


class SystemStats(BaseModel):
    """System statistics"""
    status: str
    total_documents: int
    total_chunks: int
    total_queries: int
    uptime_seconds: int
    db_status: str
    llm_status: str


# Mock RAG Service
class MockRAGService:
    """Simulates RAG service for demo"""

    def __init__(self):
        self.query_count = 0
        self.start_time = time.time()

    async def query(self, question: str, device_type: str, top_k: int) -> Dict:
        """Process RAG query"""
        start = time.time()

        # Simulate async processing
        await asyncio.sleep(0.3)

        self.query_count += 1

        device_name = device_type.replace("_", " ").title() if device_type != "all" else "the equipment"

        answer = f"""**Maintenance Procedure for {device_name}:**

1. **Safety Lockout**:
   - Power down equipment completely
   - Apply lockout/tagout (LOTO) procedures
   - Verify zero energy state with appropriate testing equipment

2. **Access Preparation**:
   - Gather required tools: 10mm socket set, torque wrench, inspection mirror
   - Review technical specifications for proper torque values
   - Prepare cleaning materials and replacement parts

3. **Inspection Process**:
   - Remove access panels (torque: 15-20 Nm)
   - Conduct visual inspection for wear, leaks, corrosion
   - Document any anomalies with photos/notes

4. **Maintenance Actions**:
   - Clean components with approved solvent
   - Apply specified lubricant (refer to tech manual for type)
   - Replace worn gaskets/seals as needed
   - Verify all connections are secure

5. **Reassembly & Testing**:
   - Reinstall panels with proper torque (refer to manual)
   - Remove LOTO devices following procedure
   - Conduct operational test under no-load conditions
   - Monitor for abnormal vibration or noise

⚠️ **Critical Safety Note**: Never bypass safety interlocks. Always follow LOTO procedures.

**Estimated Duration**: 30-45 minutes
**Skill Level Required**: Qualified maintenance technician"""

        sources = [
            SourceDocument(
                doc_name=f"{device_type.upper()}_Maintenance_Manual_v3.2.pdf",
                page_numbers="23-25",
                section="Chapter 4: Routine Maintenance Procedures",
                relevance=0.89,
                excerpt="Standard preventive maintenance procedure for routine inspections, including disassembly, inspection, and reassembly protocols..."
            ),
            SourceDocument(
                doc_name="Safety_Guidelines_2025.pdf",
                page_numbers="12-14",
                section="Section 2: Lockout/Tagout Procedures",
                relevance=0.82,
                excerpt="All maintenance work must follow LOTO procedures. Ensure equipment is de-energized and verified before performing any maintenance..."
            ),
            SourceDocument(
                doc_name=f"{device_type.upper()}_Technical_Specifications.pdf",
                page_numbers="34-35",
                section="Torque Specifications",
                relevance=0.76,
                excerpt="Access panel fasteners: 15-20 Nm. Main housing bolts: 40-45 Nm. Always use calibrated torque wrench..."
            ),
            SourceDocument(
                doc_name="Lubrication_Guide.pdf",
                page_numbers="8-9",
                section="Recommended Lubricants",
                relevance=0.68,
                excerpt=f"For {device_name}: Use synthetic oil ISO VG 68. Apply at 3-month intervals or after 500 operating hours..."
            )
        ]

        processing_time = int((time.time() - start) * 1000)

        return {
            "answer": answer,
            "sources": sources[:top_k],
            "query": question,
            "device_type": device_type,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85,
            "processing_time_ms": processing_time
        }

    def get_stats(self) -> SystemStats:
        """Get system statistics"""
        return SystemStats(
            status="operational",
            total_documents=127,
            total_chunks=3842,
            total_queries=self.query_count,
            uptime_seconds=int(time.time() - self.start_time),
            db_status="connected",
            llm_status="ready"
        )


# Initialize service
import asyncio
rag_service = MockRAGService()


# API Endpoints
@app.get("/", tags=["root"])
async def root():
    """API root endpoint"""
    return {
        "name": "Maintenance Documentation RAG API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "query": "/api/query",
            "stats": "/api/stats",
            "health": "/health"
        }
    }


@app.post("/api/query", response_model=QueryResponse, tags=["rag"])
async def query_documentation(request: QueryRequest):
    """
    Query maintenance documentation using RAG

    This endpoint:
    1. Embeds the user's question
    2. Searches JadeVectorDB for relevant chunks
    3. Sends context + question to LLM
    4. Returns answer with source citations
    """
    try:
        result = await rag_service.query(
            question=request.question,
            device_type=request.device_type,
            top_k=request.top_k
        )
        return QueryResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats", response_model=SystemStats, tags=["system"])
async def get_stats():
    """Get system statistics and status"""
    return rag_service.get_stats()


@app.get("/health", tags=["system"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "operational",
            "database": "connected",
            "llm": "ready"
        }
    }


@app.get("/api/devices", tags=["metadata"])
async def get_device_types():
    """Get list of available device types"""
    return {
        "devices": [
            {"value": "all", "label": "All Devices"},
            {"value": "hydraulic_pump", "label": "Hydraulic Pump"},
            {"value": "air_compressor", "label": "Air Compressor"},
            {"value": "generator", "label": "Generator"},
            {"value": "cnc_machine", "label": "CNC Machine"},
            {"value": "conveyor_belt", "label": "Conveyor Belt"}
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("🚀 Starting FastAPI RAG Backend")
    print("="*60)
    print("\n📍 API: http://localhost:8000")
    print("📚 Docs: http://localhost:8000/docs")
    print("🔌 Running in offline mode")
    print("\n💡 Tip: Press Ctrl+C to stop\n")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
