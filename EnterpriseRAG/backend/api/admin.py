"""
Admin API endpoints for document management
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from typing import List
from ..models.schemas import (
    DocumentUploadResponse,
    DocumentInfo,
    DocumentListResponse,
    DocumentDeleteResponse,
    ProcessingStatus,
)
from ..utils.config import is_mock_mode, settings
from ..services.mock_service import mock_service
from ..services.rag_service import production_service

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    device_type: str = Form(...),
):
    """
    Upload a maintenance document (PDF or DOCX)

    **Parameters:**
    - file: PDF or DOCX file
    - device_type: Equipment type (e.g., "hydraulic_pump", "air_compressor")

    **Processing:**
    - Mock mode: Simulates upload
    - Production mode: Extracts text, chunks, generates embeddings, stores in JadeVectorDB
    """
    # Validate file type
    allowed_extensions = settings.allowed_extensions
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}",
        )

    # Check file size
    file_content = await file.read()
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    try:
        if is_mock_mode():
            result = await mock_service.upload_document(
                filename=file.filename,
                device_type=device_type,
                file_content=file_content,
            )
        else:
            result = await production_service.upload_document(
                filename=file.filename,
                device_type=device_type,
                file_content=file_content,
            )

        return DocumentUploadResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    """
    List all uploaded documents
    """
    try:
        if is_mock_mode():
            documents = await mock_service.list_documents()
        else:
            documents = await production_service.list_documents()

        return DocumentListResponse(
            documents=documents,
            total=len(documents),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@router.get("/documents/{doc_id}/status", response_model=ProcessingStatus)
async def get_document_status(doc_id: str):
    """
    Get document processing status
    """
    try:
        if is_mock_mode():
            return await mock_service.get_processing_status(doc_id)
        else:
            return await production_service.get_processing_status(doc_id)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str, background_tasks: BackgroundTasks):
    """
    Delete document and all associated vectors

    **Process:**
    1. Find all chunks for the document
    2. Delete vectors from JadeVectorDB (index updates incrementally)
    3. Remove metadata
    4. Schedule background compaction if threshold exceeded (>10% deleted)

    **Note:** Compaction runs in background without interrupting queries
    """
    try:
        if is_mock_mode():
            return await mock_service.delete_document(doc_id)
        else:
            # Production mode handles compaction internally
            return await production_service.delete_document(doc_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")


@router.post("/documents/{doc_id}/reprocess")
async def reprocess_document(doc_id: str):
    """
    Reprocess an existing document

    **Use cases:**
    - Document was updated
    - Processing failed previously
    - Different chunking strategy needed
    """
    try:
        # Get document metadata
        if is_mock_mode():
            documents = await mock_service.list_documents()
        else:
            documents = await production_service.list_documents()

        doc = next((d for d in documents if d.id == doc_id), None)
        if not doc:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")

        # For now, return not implemented
        # In production, would:
        # 1. Delete existing vectors
        # 2. Re-upload and process file
        return {
            "success": False,
            "message": "Reprocessing not yet implemented. Please delete and re-upload.",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reprocess failed: {str(e)}")
