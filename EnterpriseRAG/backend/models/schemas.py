"""
Pydantic models for request/response validation
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Any, Dict, List, Optional, Literal
from datetime import datetime


# ── Query Models ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    """RAG query request"""
    question: str = Field(..., min_length=1, description="User's question")
    category: Optional[str] = Field("all", description="Document category filter")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the process for submitting an expense report?",
                "category": "hr",
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
    category: str
    timestamp: str
    confidence: float = Field(ge=0, le=1)
    processing_time_ms: int
    mode: Literal["mock", "production"]


# ── Document Management Models ────────────────────────────────

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
    category: str
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


# ── System Models ─────────────────────────────────────────────

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


# ── Analytics Models ──────────────────────────────────────────

class QueryRecord(BaseModel):
    """Single query log entry"""
    id: int
    question: str
    category: str
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
    category_breakdown: Dict[str, int]
    recent_queries: List[QueryRecord]


# ── Processing Models ─────────────────────────────────────────

class ProcessingStatus(BaseModel):
    """Document processing status"""
    doc_id: str
    status: Literal["pending", "processing", "complete", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str
    chunks_processed: int = 0
    total_chunks: Optional[int] = None


# ── User Management Models ────────────────────────────────────

class UserCreate(BaseModel):
    """Admin request to create a new user"""
    username: str = Field(..., min_length=3, max_length=64)
    email: str = Field(..., description="User email address")
    role: Literal["admin", "user"] = "user"

    class Config:
        json_schema_extra = {
            "example": {
                "username": "alice",
                "email": "alice@example.com",
                "role": "user"
            }
        }


class UserResponse(BaseModel):
    """Public user information (no password fields)"""
    id: str
    username: str
    email: str
    role: Literal["admin", "user"]
    must_change_password: bool
    created_at: str
    last_login: Optional[str] = None
    is_active: bool


class CreateUserResponse(BaseModel):
    """Response when admin creates a user — includes one-time generated password"""
    user: UserResponse
    generated_password: str = Field(
        ..., description="Initial password — shown ONCE, not stored in plaintext"
    )


class ResetPasswordResponse(BaseModel):
    """Response when admin resets a user's password — includes one-time generated password"""
    user_id: str
    username: str
    generated_password: str = Field(
        ..., description="New generated password — shown ONCE, not stored in plaintext"
    )


class UserListResponse(BaseModel):
    """Paginated list of users"""
    users: List[UserResponse]
    total: int
    offset: int = 0
    limit: int = 100


class LoginRequest(BaseModel):
    """Login credentials"""
    username: str
    password: str

    class Config:
        json_schema_extra = {
            "example": {"username": "admin", "password": "Admin@1234"}
        }


class LoginResponse(BaseModel):
    """JWT login response"""
    access_token: str
    token_type: str = "bearer"
    must_change_password: bool
    user: UserResponse


class ChangePasswordRequest(BaseModel):
    """Self-service password change"""
    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str

    class Config:
        json_schema_extra = {
            "example": {
                "current_password": "Admin@1234",
                "new_password": "MyNewPass@99",
                "confirm_password": "MyNewPass@99"
            }
        }
