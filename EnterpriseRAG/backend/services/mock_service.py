"""
Mock RAG service for demo/testing without external dependencies.
Returns realistic but simulated responses.
Uses the same MetadataDB so uploaded documents persist across restarts.
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional
from ..logging_config import get_logger
from ..models.schemas import (
    QueryResponse,
    SourceDocument,
    DocumentInfo,
    SystemStats,
    DocumentDeleteResponse,
    ProcessingStatus,
)
from ..utils.config import settings
from .metadata_db import MetadataDB

log = get_logger(__name__)

# Pre-loaded demo documents seeded on first run
_SEED_DOCUMENTS = [
    {
        "id": "doc_001",
        "filename": "employee_handbook.pdf",
        "category": "hr",
        "status": "complete",
        "uploaded_at": "2026-03-20T10:30:00Z",
        "processed_at": "2026-03-20T10:31:45Z",
        "chunk_count": 156,
        "chunks_done": 156,
        "error": None,
    },
    {
        "id": "doc_002",
        "filename": "it_security_policy.pdf",
        "category": "it",
        "status": "complete",
        "uploaded_at": "2026-03-21T14:20:00Z",
        "processed_at": "2026-03-21T14:22:10Z",
        "chunk_count": 203,
        "chunks_done": 203,
        "error": None,
    },
    {
        "id": "doc_003",
        "filename": "finance_procedures.pdf",
        "category": "finance",
        "status": "complete",
        "uploaded_at": "2026-03-22T09:15:00Z",
        "processed_at": "2026-03-22T09:17:30Z",
        "chunk_count": 178,
        "chunks_done": 178,
        "error": None,
    },
]


class MockRAGService:
    """Mock RAG service with simulated responses and persistent metadata."""

    def __init__(self):
        self.start_time = datetime.utcnow()
        self.db = MetadataDB(settings.metadata_db_path)
        self._seed_demo_documents()

    def _seed_demo_documents(self):
        """Insert demo documents on first run if the DB is empty."""
        if not self.db.list_all():
            for doc in _SEED_DOCUMENTS:
                # MetadataDB.insert() only uses id/filename/category/status/uploaded_at,
                # so we call update() afterwards for the remaining fields.
                self.db.insert({
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "category": doc["category"],
                    "status": doc["status"],
                    "uploaded_at": doc["uploaded_at"],
                })
                self.db.update(doc["id"], {
                    "processed_at": doc["processed_at"],
                    "chunk_count": doc["chunk_count"],
                    "chunks_done": doc["chunks_done"],
                    "error": doc["error"],
                })

    async def query(
        self,
        question: str,
        category: Optional[str] = "all",
        top_k: int = 5,
    ) -> QueryResponse:
        """Simulate RAG query with realistic responses."""
        self.db.increment_query_count()
        log.info("mock_query", question=question[:120], category=category)
        await asyncio.sleep(0.3)

        answer = self._generate_mock_answer(question, category)
        sources = self._generate_mock_sources(question, category, top_k)

        self.db.log_query(
            question=question,
            category=category or "all",
            mode="mock",
            confidence=0.87,
            processing_time_ms=320,
            sources_count=len(sources),
            success=True,
        )

        return QueryResponse(
            success=True,
            answer=answer,
            sources=sources,
            query=question,
            category=category or "all",
            timestamp=datetime.utcnow().isoformat() + "Z",
            confidence=0.87,
            processing_time_ms=320,
            mode="mock",
        )

    async def upload_document(
        self, filename: str, category: str = "general", file_content: bytes = b""
    ) -> dict:
        """Simulate document upload and schedule simulated processing."""
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        uploaded_at = datetime.utcnow().isoformat() + "Z"
        log.info("mock_upload", doc_id=doc_id, filename=filename, category=category)

        self.db.insert({
            "id": doc_id,
            "filename": filename,
            "category": category,
            "status": "processing",
            "uploaded_at": uploaded_at,
        })

        # Simulate async processing that completes after a short delay
        asyncio.create_task(self._simulate_processing(doc_id))

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "status": "processing",
            "message": f"Document '{filename}' uploaded successfully (mock mode)",
        }

    async def _simulate_processing(self, doc_id: str):
        """Simulate chunking and embedding over ~2 seconds."""
        simulated_chunks = 120
        self.db.update(doc_id, {"chunk_count": simulated_chunks, "chunks_done": 0})
        step = simulated_chunks // 4
        for done in [step, step * 2, step * 3, simulated_chunks]:
            await asyncio.sleep(0.5)
            self.db.update(doc_id, {"chunks_done": done})
        self.db.update(doc_id, {
            "status": "complete",
            "processed_at": datetime.utcnow().isoformat() + "Z",
            "chunk_count": simulated_chunks,
            "chunks_done": simulated_chunks,
        })

    async def get_processing_status(self, doc_id: str) -> ProcessingStatus:
        """Get document processing status with real progress."""
        doc = self.db.get(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        status = doc["status"]
        chunks_done = doc.get("chunks_done") or 0
        total_chunks = doc.get("chunk_count")

        if status == "complete":
            progress = 100
        elif status == "failed":
            progress = 0
        elif total_chunks:
            progress = min(int(chunks_done / total_chunks * 100), 99)
        else:
            progress = 5

        message = (
            "Processing complete" if status == "complete"
            else f"Processing failed: {doc.get('error')}" if status == "failed"
            else "Extracting and chunking document"
        )

        return ProcessingStatus(
            doc_id=doc_id,
            status=status,
            progress=progress,
            message=message,
            chunks_processed=chunks_done,
            total_chunks=total_chunks,
        )

    async def list_documents(self, offset: int = 0, limit: int = 100) -> tuple[List[DocumentInfo], int]:
        """List documents with pagination."""
        total = self.db.count_all()
        rows = self.db.list_all(offset=offset, limit=limit)
        docs = [
            DocumentInfo(
                id=doc["doc_id"],
                filename=doc["filename"],
                category=doc.get("category", "general"),
                status=doc["status"],
                uploaded_at=doc["uploaded_at"],
                processed_at=doc.get("processed_at"),
                chunk_count=doc.get("chunk_count"),
                error=doc.get("error"),
            )
            for doc in rows
        ]
        return docs, total

    async def reprocess_document(self, doc_id: str) -> dict:
        """Simulate reprocessing (mock mode)."""
        doc = self.db.get(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        self.db.update(doc_id, {
            "status": "processing",
            "processed_at": None,
            "chunk_count": None,
            "chunks_done": 0,
            "error": None,
        })
        asyncio.create_task(self._simulate_processing(doc_id))

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": doc["filename"],
            "status": "processing",
            "message": f"Document '{doc['filename']}' is being reprocessed (mock mode)",
        }

    async def delete_document(self, doc_id: str) -> DocumentDeleteResponse:
        """Delete document from persistent store."""
        log.info("mock_delete", doc_id=doc_id)
        doc = self.db.get(doc_id)
        if not doc:
            return DocumentDeleteResponse(
                success=False,
                doc_id=doc_id,
                doc_name="unknown",
                chunks_found=0,
                chunks_deleted=0,
                chunks_failed=0,
                message=f"Document {doc_id} not found",
            )

        chunk_count = doc.get("chunk_count") or 0
        doc_name = doc["filename"]

        self.db.delete(doc_id)
        await asyncio.sleep(0.1)

        return DocumentDeleteResponse(
            success=True,
            doc_id=doc_id,
            doc_name=doc_name,
            chunks_found=chunk_count,
            chunks_deleted=chunk_count,
            chunks_failed=0,
            message=f"Document '{doc_name}' deleted successfully (mock mode)",
        )

    def get_analytics(self, recent_limit: int = 20) -> dict:
        return self.db.get_analytics(recent_limit=recent_limit)

    async def get_stats(self) -> SystemStats:
        """Get system statistics."""
        all_docs = self.db.list_all()
        total_docs = len(all_docs)
        total_chunks = sum(
            doc.get("chunk_count") or 0
            for doc in all_docs
            if doc.get("status") == "complete"
        )
        uptime = int((datetime.utcnow() - self.start_time).total_seconds())

        return SystemStats(
            status="healthy",
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_queries=self.db.get_query_count(),
            uptime_seconds=uptime,
            mode="mock",
            db_status="simulated",
            llm_status="simulated",
        )

    # ── Mock answer/source generation ─────────────────────────

    def _generate_mock_answer(self, question: str, category: Optional[str]) -> str:
        q = question.lower()

        if any(w in q for w in ["expense", "reimburs", "receipt"]):
            return """To submit an expense report:

1. **Log In**: Access the employee portal at portal.company.com
2. **Navigate**: Go to Finance → Expense Reports → New Report
3. **Fill Details**: Enter expense date, amount, category, and business purpose
4. **Attach Receipts**: Upload digital copies of all receipts
5. **Submit**: Click Submit for manager approval

**Important**: Expenses must be submitted within 30 days. Expenses over $500 require VP approval.
Refer to the Finance Procedures document, Section 4.2 for full policy."""

        if any(w in q for w in ["leave", "vacation", "time off", "pto"]):
            return """To request time off:

1. **Log In**: Access the HR portal
2. **Navigate**: Go to My Team → Time Off → New Request
3. **Select Dates**: Choose start and end dates
4. **Select Type**: Annual leave, sick leave, or other
5. **Submit**: Your manager will approve or decline within 2 business days

**Accrual**: Full-time employees accrue 1.5 days per month (18 days/year).
Refer to the Employee Handbook, Section 6 (Leave Policies) for details."""

        if any(w in q for w in ["password", "access", "account", "it", "system"]):
            return """To request IT access:

1. **Submit Ticket**: Go to it-support.company.com → New Request → Access Management
2. **Specify System**: Name the application or system you need access to
3. **Provide Justification**: State your business reason
4. **Manager Approval**: Your manager must approve before IT provisions access
5. **Processing Time**: Standard requests are completed within 2 business days

Refer to the IT Security Policy, Section 3 (Access Management) for full details."""

        if any(w in q for w in ["onboard", "new hire", "first day", "checklist"]):
            return """New employee onboarding checklist:

**Before Day 1:**
- Complete digital paperwork via HR portal
- Receive equipment and credentials from IT

**Day 1:**
- Meet with manager and team
- Complete mandatory compliance training
- Set up email, Slack, and required systems

**Week 1:**
- Complete all onboarding modules in the LMS
- Review team processes and tools
- Schedule 1:1 meetings with key stakeholders

Refer to the Employee Handbook, Chapter 2 (Onboarding) for the full guide."""

        return f"""Based on the {'[' + category + '] ' if category and category != 'all' else ''}documentation:

The relevant information includes:
1. Following the documented procedures in the source materials
2. Consulting the applicable policy section for specifics
3. Reaching out to the responsible team if clarification is needed

Please review the sources cited below for the detailed guidance."""

    def _generate_mock_sources(
        self, question: str, category: Optional[str], top_k: int
    ) -> List[SourceDocument]:
        all_docs = self.db.list_all()
        relevant = [
            d for d in all_docs
            if d.get("status") == "complete"
            and (not category or category == "all" or d.get("category") == category)
        ] or [d for d in all_docs if d.get("status") == "complete"]

        sources = []
        for i, doc in enumerate(relevant[:top_k]):
            sources.append(
                SourceDocument(
                    doc_name=doc["filename"],
                    page_numbers=f"{12 + i}-{13 + i}",
                    section=self._get_relevant_section(question),
                    relevance=round(0.95 - (i * 0.08), 2),
                    excerpt=(
                        "...the procedure for this operation requires following standard "
                        "safety protocols. Refer to the detailed instructions and diagrams "
                        "provided in this section. Always use manufacturer-approved parts..."
                    ),
                )
            )
        return sources

    def _get_relevant_section(self, question: str) -> str:
        q = question.lower()
        if any(w in q for w in ["expense", "reimburs", "receipt"]):
            return "Section 4: Expense Reimbursement"
        if any(w in q for w in ["leave", "vacation", "pto"]):
            return "Section 6: Leave Policies"
        if any(w in q for w in ["password", "access", "it"]):
            return "Section 3: Access Management"
        if any(w in q for w in ["onboard", "new hire"]):
            return "Chapter 2: Onboarding"
        return "Section 1: General Procedures"


# Global instance
mock_service = MockRAGService()
