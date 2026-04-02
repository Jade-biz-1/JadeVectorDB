"""
Mock RAG service for demo/testing without external dependencies
Returns realistic but simulated responses
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


class MockRAGService:
    """Mock RAG service with simulated responses"""

    def __init__(self):
        self.mock_documents = {
            "doc_001": {
                "id": "doc_001",
                "filename": "hydraulic_pump_manual.pdf",
                "device_type": "hydraulic_pump",
                "status": "complete",
                "uploaded_at": "2026-03-20T10:30:00Z",
                "processed_at": "2026-03-20T10:31:45Z",
                "chunk_count": 156,
            },
            "doc_002": {
                "id": "doc_002",
                "filename": "air_compressor_guide.pdf",
                "device_type": "air_compressor",
                "status": "complete",
                "uploaded_at": "2026-03-21T14:20:00Z",
                "processed_at": "2026-03-21T14:22:10Z",
                "chunk_count": 203,
            },
            "doc_003": {
                "id": "doc_003",
                "filename": "conveyor_system_manual.pdf",
                "device_type": "conveyor",
                "status": "complete",
                "uploaded_at": "2026-03-22T09:15:00Z",
                "processed_at": "2026-03-22T09:17:30Z",
                "chunk_count": 178,
            },
        }
        self.query_count = 0
        self.start_time = datetime.utcnow()

    async def query(
        self,
        question: str,
        device_type: Optional[str] = "all",
        top_k: int = 5,
    ) -> QueryResponse:
        """
        Simulate RAG query with realistic responses
        """
        self.query_count += 1

        # Simulate processing delay
        await asyncio.sleep(0.3)

        # Generate mock answer based on question keywords
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
        """
        Simulate document upload
        """
        doc_id = f"doc_{str(uuid.uuid4())[:8]}"

        # Simulate upload processing
        await asyncio.sleep(0.2)

        self.mock_documents[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "device_type": device_type,
            "status": "processing",
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "processed_at": None,
            "chunk_count": None,
        }

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "status": "processing",
            "message": f"Document '{filename}' uploaded successfully (mock mode)",
        }

    async def get_processing_status(self, doc_id: str) -> ProcessingStatus:
        """
        Get document processing status
        """
        doc = self.mock_documents.get(doc_id)
        if not doc:
            raise ValueError(f"Document {doc_id} not found")

        # Simulate processing progress
        if doc["status"] == "processing":
            # Randomly complete processing in mock mode
            import random

            if random.random() > 0.5:
                doc["status"] = "complete"
                doc["processed_at"] = datetime.utcnow().isoformat() + "Z"
                doc["chunk_count"] = random.randint(100, 250)

        return ProcessingStatus(
            doc_id=doc_id,
            status=doc["status"],
            progress=100 if doc["status"] == "complete" else 65,
            message="Processing complete" if doc["status"] == "complete" else "Extracting and chunking document",
            chunks_processed=doc.get("chunk_count", 0) if doc["status"] == "complete" else 65,
            total_chunks=doc.get("chunk_count", 100),
        )

    async def list_documents(self) -> List[DocumentInfo]:
        """
        List all documents
        """
        return [
            DocumentInfo(
                id=doc["id"],
                filename=doc["filename"],
                device_type=doc["device_type"],
                status=doc["status"],
                uploaded_at=doc["uploaded_at"],
                processed_at=doc.get("processed_at"),
                chunk_count=doc.get("chunk_count"),
            )
            for doc in self.mock_documents.values()
        ]

    async def delete_document(self, doc_id: str) -> DocumentDeleteResponse:
        """
        Delete document (mock)
        """
        doc = self.mock_documents.get(doc_id)
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

        chunk_count = doc.get("chunk_count", 0)
        doc_name = doc["filename"]

        # Remove from mock storage
        del self.mock_documents[doc_id]

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
        """
        Get system statistics
        """
        total_docs = len(self.mock_documents)
        total_chunks = sum(
            doc.get("chunk_count", 0)
            for doc in self.mock_documents.values()
            if doc.get("chunk_count")
        )
        uptime = int((datetime.utcnow() - self.start_time).total_seconds())

        return SystemStats(
            status="healthy",
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_queries=self.query_count,
            uptime_seconds=uptime,
            mode="mock",
            db_status="simulated",
            llm_status="simulated",
        )

    def _generate_mock_answer(self, question: str, device_type: Optional[str]) -> str:
        """
        Generate realistic mock answer based on question
        """
        question_lower = question.lower()

        # Pattern matching for common maintenance questions
        if any(word in question_lower for word in ["replace", "change", "swap"]):
            if "filter" in question_lower:
                return """To replace the air filter:

1. **Safety First**: Shut down the equipment and disconnect power
2. **Locate Filter**: Remove the access panel on the right side
3. **Remove Old Filter**: Unscrew the retaining clips and slide out the old filter
4. **Install New Filter**: Insert the new filter ensuring the airflow direction arrow points inward
5. **Secure**: Replace retaining clips and access panel
6. **Test**: Restart equipment and verify normal operation

**Important**: Always use OEM-approved filters. Replacement interval: Every 500 operating hours or 3 months."""

            elif "oil" in question_lower or "fluid" in question_lower:
                return """To replace the hydraulic fluid:

1. **Preparation**: Ensure system is cool and depressurized
2. **Drain**: Open drain valve at the bottom of the reservoir
3. **Clean**: Wipe reservoir interior with lint-free cloth
4. **Refill**: Add manufacturer-specified hydraulic oil (ISO VG 46)
5. **Bleed**: Run system at low pressure to remove air bubbles
6. **Check Level**: Ensure fluid is at the "FULL" mark

**Capacity**: 15 liters
**Recommended Fluid**: Mobil DTE 25 or equivalent
**Change Interval**: Every 2000 hours or annually"""

        elif any(word in question_lower for word in ["troubleshoot", "problem", "issue", "not working"]):
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
4. Reduce operating load temporarily

Refer to Section 7 (Troubleshooting) for error code meanings."""

        elif any(word in question_lower for word in ["maintain", "maintenance", "service"]):
            return """Regular maintenance schedule:

**Daily:**
- Visual inspection for leaks
- Check fluid levels
- Verify proper operation

**Weekly:**
- Inspect filters
- Check belt tension
- Lubricate moving parts

**Monthly:**
- Replace air filters
- Check electrical connections
- Inspect hoses for wear

**Quarterly:**
- Change hydraulic fluid
- Inspect safety systems
- Calibrate sensors

**Annually:**
- Complete system overhaul
- Replace wear parts
- Professional inspection

Detailed procedures are in Section 6 (Preventive Maintenance)."""

        else:
            # Generic answer
            return f"""Based on the maintenance documentation for {device_type or 'your equipment'}:

The recommended procedure involves:
1. Following standard safety protocols
2. Consulting the technical specifications in Chapter 3
3. Using manufacturer-approved parts and tools
4. Documenting all maintenance activities

For detailed step-by-step instructions, please refer to the relevant section of the manual. If this is a critical operation, contact a certified technician.

**Safety Note**: Always wear appropriate PPE and follow lockout/tagout procedures."""

        return answer

    def _generate_mock_sources(
        self, question: str, device_type: Optional[str], top_k: int
    ) -> List[SourceDocument]:
        """
        Generate mock source citations
        """
        # Select relevant mock documents
        relevant_docs = []
        for doc in self.mock_documents.values():
            if device_type == "all" or doc["device_type"] == device_type:
                relevant_docs.append(doc)

        # If no documents match, use all
        if not relevant_docs:
            relevant_docs = list(self.mock_documents.values())

        # Generate sources
        sources = []
        for i, doc in enumerate(relevant_docs[:top_k]):
            relevance = 0.95 - (i * 0.08)  # Decreasing relevance
            sources.append(
                SourceDocument(
                    doc_name=doc["filename"],
                    page_numbers=f"{12 + i}-{13 + i}",
                    section=self._get_relevant_section(question),
                    relevance=round(relevance, 2),
                    excerpt=self._get_excerpt(question, doc["filename"]),
                )
            )

        return sources

    def _get_relevant_section(self, question: str) -> str:
        """
        Get relevant section based on question
        """
        question_lower = question.lower()
        if any(word in question_lower for word in ["replace", "install"]):
            return "Section 5: Component Replacement"
        elif any(word in question_lower for word in ["troubleshoot", "problem"]):
            return "Section 7: Troubleshooting Guide"
        elif any(word in question_lower for word in ["maintain", "service"]):
            return "Section 6: Preventive Maintenance"
        elif any(word in question_lower for word in ["safety", "warning"]):
            return "Section 2: Safety Guidelines"
        else:
            return "Section 4: Operating Procedures"

    def _get_excerpt(self, question: str, doc_name: str) -> str:
        """
        Generate relevant excerpt
        """
        return f"...the procedure for this operation requires following standard safety protocols. Refer to the detailed instructions and diagrams provided in this section. Always use manufacturer-approved parts..."


# Global instance
mock_service = MockRAGService()
