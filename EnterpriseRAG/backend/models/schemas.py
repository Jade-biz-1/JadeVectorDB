"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime


# Query Models
class QueryRequest(BaseModel):
    """RAG query request"""
    question: str = Field(..., min_length=1, description="User's question")
    device_type: Optional[str] = Field("all", description="Device type filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

    class Config:
        json_schema_extra = {
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
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: int
    mode: Literal["mock", "production"]


# Document Management Models
class DocumentUploadResponse(BaseModel):
    """Response after document upload"""
    success: bool
    doc_id: str
    filename: str
    status: Literal["pending", "processing", "complete", "failed"]
    message: str


class DocumentInfo(BaseModel):
    """Document information"""
    id: str
    filename: str
    device_type: str
    status: Literal["pending", "processing", "complete", "failed"]
    uploaded_at: str
    processed_at: Optional[str] = None
    chunk_count: Optional[int] = None
    error: Optional[str] = None


class DocumentListResponse(BaseModel):
    """List of documents"""
    documents: List[DocumentInfo]
    total: int
    offset: int = 0
    limit: int = 100


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion"""
    success: bool
    doc_id: str
    doc_name: str
    chunks_found: int
    chunks_deleted: int
    chunks_failed: int
    message: str
    note: Optional[str] = None


# System Models
class SystemStats(BaseModel):
    """System statistics"""
    status: str
    total_documents: int
    total_chunks: int
    total_queries: int
    uptime_seconds: int
    mode: Literal["mock", "production"]
    db_status: str
    llm_status: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    mode: Literal["mock", "production"]
    components: dict


# Analytics Models
class QueryRecord(BaseModel):
    """Single query log entry"""
    id: int
    question: str
    device_type: str
    mode: str
    confidence: float
    processing_time_ms: int
    sources_count: int
    success: bool
    timestamp: str


class AnalyticsResponse(BaseModel):
    """Query analytics summary"""
    total_queries: int
    avg_confidence: float
    avg_processing_time_ms: float
    success_rate: float
    device_type_breakdown: Dict[str, int]
    recent_queries: List[QueryRecord]


# Processing Models
class ProcessingStatus(BaseModel):
    """Document processing status"""
    doc_id: str
    status: Literal["pending", "processing", "complete", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str
    chunks_processed: int = 0
    total_chunks: Optional[int] = None
