"""
Mock RAG service for demo/testing without external dependencies.
Returns realistic but simulated responses.
Uses the same MetadataDB so uploaded documents persist across restarts.
"""

import asyncio
import uuid
from datetime import datetime
from typing import List, Optional
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

# Pre-loaded demo documents seeded on first run
_SEED_DOCUMENTS = [
    {
        "id": "doc_001",
        "filename": "hydraulic_pump_manual.pdf",
        "device_type": "hydraulic_pump",
        "status": "complete",
        "uploaded_at": "2026-03-20T10:30:00Z",
        "processed_at": "2026-03-20T10:31:45Z",
        "chunk_count": 156,
        "chunks_done": 156,
        "error": None,
    },
    {
        "id": "doc_002",
        "filename": "air_compressor_guide.pdf",
        "device_type": "air_compressor",
        "status": "complete",
        "uploaded_at": "2026-03-21T14:20:00Z",
        "processed_at": "2026-03-21T14:22:10Z",
        "chunk_count": 203,
        "chunks_done": 203,
        "error": None,
    },
    {
        "id": "doc_003",
        "filename": "conveyor_system_manual.pdf",
        "device_type": "conveyor",
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
                # MetadataDB.insert() only uses id/filename/device_type/status/uploaded_at,
                # so we call update() afterwards for the remaining fields.
                self.db.insert({
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "device_type": doc["device_type"],
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
        device_type: Optional[str] = "all",
        top_k: int = 5,
    ) -> QueryResponse:
        """Simulate RAG query with realistic responses."""
        self.db.increment_query_count()
        await asyncio.sleep(0.3)

        answer = self._generate_mock_answer(question, device_type)
        sources = self._generate_mock_sources(question, device_type, top_k)

        return QueryResponse(
            success=True,
            answer=answer,
            sources=sources,
            query=question,
            device_type=device_type or "all",
            timestamp=datetime.utcnow().isoformat() + "Z",
            confidence=0.87,
            processing_time_ms=320,
            mode="mock",
        )

    async def upload_document(
        self, filename: str, device_type: str, file_content: bytes
    ) -> dict:
        """Simulate document upload and schedule simulated processing."""
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"
        uploaded_at = datetime.utcnow().isoformat() + "Z"

        self.db.insert({
            "id": doc_id,
            "filename": filename,
            "device_type": device_type,
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

    async def list_documents(self) -> List[DocumentInfo]:
        """List all documents."""
        return [
            DocumentInfo(
                id=doc["doc_id"],
                filename=doc["filename"],
                device_type=doc["device_type"],
                status=doc["status"],
                uploaded_at=doc["uploaded_at"],
                processed_at=doc.get("processed_at"),
                chunk_count=doc.get("chunk_count"),
                error=doc.get("error"),
            )
            for doc in self.db.list_all()
        ]

    async def delete_document(self, doc_id: str) -> DocumentDeleteResponse:
        """Delete document from persistent store."""
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

    def _generate_mock_answer(self, question: str, device_type: Optional[str]) -> str:
        q = question.lower()

        if any(w in q for w in ["replace", "change", "swap"]):
            if "filter" in q:
                return """To replace the air filter:

1. **Safety First**: Shut down the equipment and disconnect power
2. **Locate Filter**: Remove the access panel on the right side
3. **Remove Old Filter**: Unscrew the retaining clips and slide out the old filter
4. **Install New Filter**: Insert the new filter ensuring the airflow direction arrow points inward
5. **Secure**: Replace retaining clips and access panel
6. **Test**: Restart equipment and verify normal operation

**Important**: Always use OEM-approved filters. Replacement interval: Every 500 operating hours or 3 months."""

            if any(w in q for w in ["oil", "fluid"]):
                return """To replace the hydraulic fluid:

1. **Preparation**: Ensure system is cool and depressurized
2. **Drain**: Open drain valve at the bottom of the reservoir
3. **Clean**: Wipe reservoir interior with lint-free cloth
4. **Refill**: Add manufacturer-specified hydraulic oil (ISO VG 46)
5. **Bleed**: Run system at low pressure to remove air bubbles
6. **Check Level**: Ensure fluid is at the "FULL" mark

**Capacity**: 15 liters | **Fluid**: Mobil DTE 25 or equivalent | **Interval**: Every 2000 hours or annually"""

        if any(w in q for w in ["troubleshoot", "problem", "issue", "not working"]):
            return """Common troubleshooting steps:

**If equipment won't start:**
1. Check power supply and circuit breakers
2. Verify emergency stop button is not engaged
3. Check control panel for error codes
4. Inspect safety interlocks

**If performance is degraded:**
1. Check fluid levels (oil, coolant, hydraulic)
2. Inspect filters for clogging
3. Verify pressure gauges show normal ranges
4. Listen for unusual noises indicating wear

**If overheating:**
1. Check cooling fan operation
2. Clean air vents and heat exchangers
3. Verify coolant circulation

Refer to Section 7 (Troubleshooting) for error code meanings."""

        if any(w in q for w in ["maintain", "maintenance", "service"]):
            return """Regular maintenance schedule:

**Daily:** Visual inspection, check fluid levels, verify normal operation
**Weekly:** Inspect filters, check belt tension, lubricate moving parts
**Monthly:** Replace air filters, check electrical connections
**Quarterly:** Change hydraulic fluid, inspect safety systems, calibrate sensors
**Annually:** Complete system overhaul, replace wear parts, professional inspection

Detailed procedures are in Section 6 (Preventive Maintenance)."""

        return f"""Based on the maintenance documentation for {device_type or 'your equipment'}:

The recommended procedure involves:
1. Following standard safety protocols
2. Consulting the technical specifications in Chapter 3
3. Using manufacturer-approved parts and tools
4. Documenting all maintenance activities

**Safety Note**: Always wear appropriate PPE and follow lockout/tagout procedures."""

    def _generate_mock_sources(
        self, question: str, device_type: Optional[str], top_k: int
    ) -> List[SourceDocument]:
        all_docs = self.db.list_all()
        relevant = [
            d for d in all_docs
            if d.get("status") == "complete"
            and (device_type == "all" or d["device_type"] == device_type)
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
        if any(w in q for w in ["replace", "install"]):
            return "Section 5: Component Replacement"
        if any(w in q for w in ["troubleshoot", "problem"]):
            return "Section 7: Troubleshooting Guide"
        if any(w in q for w in ["maintain", "service"]):
            return "Section 6: Preventive Maintenance"
        if any(w in q for w in ["safety", "warning"]):
            return "Section 2: Safety Guidelines"
        return "Section 4: Operating Procedures"


# Global instance
mock_service = MockRAGService()
