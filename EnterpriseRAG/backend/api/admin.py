"""
Admin API endpoints for document management
"""

from fastapi import APIRouter, Depends, Query, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from ..models.schemas import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDeleteResponse,
    ProcessingStatus,
)
from ..utils.config import is_mock_mode, settings
from ..services.mock_service import mock_service
from ..services.rag_service import production_service
from ..security import verify_api_key
from ..rate_limiter import limiter
from ..logging_config import get_logger

log = get_logger(__name__)

router = APIRouter(
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(verify_api_key)],
)


def _svc():
    return mock_service if is_mock_mode() else production_service


@router.post("/documents/upload", response_model=DocumentUploadResponse)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    device_type: str = Form(...),
):
    """
    Upload a maintenance document (PDF or DOCX).
    Limited to 10 uploads per minute per IP.
    Production mode saves the file to disk so it can be reprocessed later.
    """
    if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(settings.allowed_extensions)}",
        )

    file_content = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    try:
        result = await _svc().upload_document(
            filename=file.filename,
            device_type=device_type,
            file_content=file_content,
        )
        return DocumentUploadResponse(**result)
    except Exception as e:
        log.error("upload_endpoint_error", filename=file.filename, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    offset: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(100, ge=1, le=500, description="Maximum documents to return"),
):
    """List uploaded documents with pagination."""
    try:
        documents, total = await _svc().list_documents(offset=offset, limit=limit)
        return DocumentListResponse(
            documents=documents,
            total=total,
            offset=offset,
            limit=limit,
        )
    except Exception as e:
        log.error("list_documents_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{doc_id}/status", response_model=ProcessingStatus)
async def get_document_status(doc_id: str):
    """Get document processing status with real chunk-level progress."""
    try:
        return await _svc().get_processing_status(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error("status_endpoint_error", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str, background_tasks: BackgroundTasks):
    """
    Delete a document and all its vectors.
    Schedules background compaction if >10% of the index is deleted.
    """
    try:
        return await _svc().delete_document(doc_id)
    except Exception as e:
        log.error("delete_endpoint_error", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(doc_id: str):
    """
    Reprocess an existing document — deletes current vectors and re-embeds
    from the saved file. Returns 409 if the original file is missing.
    """
    try:
        return await _svc().reprocess_document(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        log.error("reprocess_endpoint_error", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Reprocess failed: {str(e)}")
