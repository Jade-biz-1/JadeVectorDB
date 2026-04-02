"""
Production RAG service using JadeVectorDB + Ollama
"""

import asyncio
import httpx
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from ..models.schemas import (
    QueryResponse,
    SourceDocument,
    DocumentInfo,
    SystemStats,
    DocumentDeleteResponse,
    ProcessingStatus,
)
from ..utils.config import settings
from .document_processor import DocumentProcessor
from .metadata_db import MetadataDB


class ProductionRAGService:
    """Production RAG service with JadeVectorDB and Ollama"""

    def __init__(self):
        self.jadevectordb_url = settings.jadevectordb_url
        self.ollama_url = settings.ollama_url
        self.database_id = settings.jadevectordb_database_id
        self.doc_processor = DocumentProcessor()
        self.start_time = datetime.utcnow()
        self.db = MetadataDB(settings.metadata_db_path)

    async def query(
        self,
        question: str,
        device_type: Optional[str] = "all",
        top_k: int = 5,
    ) -> QueryResponse:
        """Execute RAG query pipeline"""
        start_time = datetime.utcnow()
        query_count = self.db.increment_query_count()

        try:
            question_embedding = await self._generate_embedding(question)
            search_results = await self._search_vectors(question_embedding, device_type, top_k)
            context = self._build_context(search_results)
            answer = await self._generate_answer(question, context)
            sources = self._extract_sources(search_results)
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

            return QueryResponse(
                success=True,
                answer=answer,
                sources=sources,
                query=question,
                device_type=device_type or "all",
                timestamp=datetime.utcnow().isoformat() + "Z",
                confidence=self._calculate_confidence(search_results),
                processing_time_ms=processing_time,
                mode="production",
            )

        except Exception as e:
            return QueryResponse(
                success=False,
                answer=f"Error processing query: {str(e)}",
                sources=[],
                query=question,
                device_type=device_type or "all",
                timestamp=datetime.utcnow().isoformat() + "Z",
                confidence=0.0,
                processing_time_ms=0,
                mode="production",
            )

    async def upload_document(
        self, filename: str, device_type: str, file_content: bytes
    ) -> dict:
        """Upload and begin processing a document"""
        doc_id = f"doc_{str(uuid.uuid4())[:12]}"
        uploaded_at = datetime.utcnow().isoformat() + "Z"

        self.db.insert({
            "id": doc_id,
            "filename": filename,
            "device_type": device_type,
            "status": "processing",
            "uploaded_at": uploaded_at,
        })

        asyncio.create_task(self._process_document(doc_id, filename, device_type, file_content))

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "status": "processing",
            "message": f"Document '{filename}' is being processed",
        }

    async def _process_document(
        self, doc_id: str, filename: str, device_type: str, file_content: bytes
    ):
        """
        Process document: extract text, chunk, embed, and store.
        Updates progress in the DB after each chunk so the status endpoint
        reflects real progress rather than a stuck 50%.
        """
        try:
            # Step 1: Extract text
            text = await self.doc_processor.extract_text(filename, file_content)

            # Step 2: Chunk
            chunks = await self.doc_processor.chunk_text(text)
            total_chunks = len(chunks)

            # Record total so the status endpoint can show X/Y progress
            self.db.update(doc_id, {"chunk_count": total_chunks, "chunks_done": 0})

            # Step 3: Embed and store each chunk, updating progress as we go
            embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self._generate_embedding(chunk["text"])
                except Exception as embed_err:
                    raise RuntimeError(
                        f"Embedding failed on chunk {i + 1}/{total_chunks}: {embed_err}"
                    ) from embed_err

                embeddings.append(embedding)
                self.db.update(doc_id, {"chunks_done": i + 1})

            # Step 4: Store all vectors in JadeVectorDB
            await self._store_vectors(doc_id, chunks, embeddings, device_type)

            # Step 5: Mark complete
            self.db.update(doc_id, {
                "status": "complete",
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "chunk_count": total_chunks,
                "chunks_done": total_chunks,
                "error": None,
            })

        except Exception as e:
            self.db.update(doc_id, {
                "status": "failed",
                "error": str(e),
            })

    async def get_processing_status(self, doc_id: str) -> ProcessingStatus:
        """Get document processing status with real progress"""
        meta = self.db.get(doc_id)
        if not meta:
            raise ValueError(f"Document {doc_id} not found")

        status = meta["status"]
        chunks_done = meta.get("chunks_done") or 0
        total_chunks = meta.get("chunk_count")

        if status == "complete":
            progress = 100
        elif status == "failed":
            progress = 0
        elif total_chunks:
            progress = min(int(chunks_done / total_chunks * 100), 99)
        else:
            progress = 5  # Just started

        message = self._get_status_message(status, meta.get("error"))

        return ProcessingStatus(
            doc_id=doc_id,
            status=status,
            progress=progress,
            message=message,
            chunks_processed=chunks_done,
            total_chunks=total_chunks,
        )

    async def list_documents(self) -> List[DocumentInfo]:
        """List all documents"""
        return [
            DocumentInfo(
                id=meta["doc_id"],
                filename=meta["filename"],
                device_type=meta["device_type"],
                status=meta["status"],
                uploaded_at=meta["uploaded_at"],
                processed_at=meta.get("processed_at"),
                chunk_count=meta.get("chunk_count"),
                error=meta.get("error"),
            )
            for meta in self.db.list_all()
        ]

    async def delete_document(self, doc_id: str) -> DocumentDeleteResponse:
        """Delete document and all associated vectors"""
        meta = self.db.get(doc_id)
        if not meta:
            return DocumentDeleteResponse(
                success=False,
                doc_id=doc_id,
                doc_name="unknown",
                chunks_found=0,
                chunks_deleted=0,
                chunks_failed=0,
                message=f"Document {doc_id} not found",
            )

        doc_name = meta["filename"]
        chunk_count = meta.get("chunk_count") or 0

        try:
            deleted_count = await self._delete_vectors(doc_id)
            self.db.delete(doc_id)
            asyncio.create_task(self._check_and_compact())

            return DocumentDeleteResponse(
                success=True,
                doc_id=doc_id,
                doc_name=doc_name,
                chunks_found=chunk_count,
                chunks_deleted=deleted_count,
                chunks_failed=0,
                message=f"Document '{doc_name}' deleted successfully",
                note="Index compaction scheduled if threshold exceeded",
            )

        except Exception as e:
            return DocumentDeleteResponse(
                success=False,
                doc_id=doc_id,
                doc_name=doc_name,
                chunks_found=chunk_count,
                chunks_deleted=0,
                chunks_failed=chunk_count,
                message=f"Error deleting document: {str(e)}",
            )

    async def get_stats(self) -> SystemStats:
        """Get system statistics"""
        all_docs = self.db.list_all()
        total_docs = len(all_docs)
        total_chunks = sum(
            doc.get("chunk_count") or 0
            for doc in all_docs
            if doc.get("status") == "complete"
        )
        uptime = int((datetime.utcnow() - self.start_time).total_seconds())
        db_status = await self._check_jadevectordb_health()
        llm_status = await self._check_ollama_health()

        return SystemStats(
            status="healthy" if db_status == "healthy" and llm_status == "healthy" else "degraded",
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_queries=self.db.get_query_count(),
            uptime_seconds=uptime,
            mode="production",
            db_status=db_status,
            llm_status=llm_status,
        )

    # ==================== Private Methods ====================

    async def _generate_embedding(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=settings.embedding_timeout) as client:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={"model": settings.ollama_embedding_model, "prompt": text},
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def _search_vectors(
        self, query_embedding: List[float], device_type: Optional[str], top_k: int
    ) -> List[Dict[str, Any]]:
        async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
            filter_query = None
            if device_type and device_type != "all":
                filter_query = {"device_type": {"$eq": device_type}}

            response = await client.post(
                f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/search",
                json={
                    "vector": query_embedding,
                    "top_k": top_k,
                    "filter": filter_query,
                    "include_metadata": True,
                },
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            return response.json()["results"]

    async def _generate_answer(self, question: str, context: str) -> str:
        prompt = f"""You are a maintenance documentation assistant. Answer the question based ONLY on the provided context.

Context from maintenance manuals:
{context}

Question: {question}

Instructions:
- Provide a clear, step-by-step answer
- Use bullet points or numbered lists for procedures
- Include safety warnings if relevant
- If the context doesn't contain the answer, say so
- Be concise but complete

Answer:"""

        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": settings.ollama_llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": settings.llm_temperature,
                        "num_predict": settings.llm_max_tokens,
                    },
                },
            )
            response.raise_for_status()
            return response.json()["response"]

    async def _store_vectors(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        device_type: str,
    ):
        vectors = [
            {
                "id": f"{doc_id}_chunk_{i}",
                "vector": embedding,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk["text"],
                    "page": chunk.get("page", "unknown"),
                    "section": chunk.get("section", ""),
                    "device_type": device_type,
                },
            }
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
        ]

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/vectors/batch",
                json={"vectors": vectors},
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()

    async def _delete_vectors(self, doc_id: str) -> int:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/query",
                json={"filter": {"doc_id": {"$eq": doc_id}}, "limit": 10000},
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            vector_ids = [result["id"] for result in response.json()["results"]]

            for i in range(0, len(vector_ids), 100):
                batch = vector_ids[i : i + 100]
                await client.post(
                    f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/vectors/delete-batch",
                    json={"ids": batch},
                    headers=self._get_auth_headers(),
                )

            return len(vector_ids)

    async def _check_and_compact(self):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/stats",
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                stats = response.json()
                total = stats.get("total_vectors", 0)
                deleted = stats.get("deleted_vectors", 0)
                if total > 0 and (deleted / total) > 0.10:
                    await client.post(
                        f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/compact",
                        headers=self._get_auth_headers(),
                    )
        except Exception:
            pass  # Compaction is best-effort

    async def _check_jadevectordb_health(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.jadevectordb_url}/health")
                return "healthy" if response.status_code == 200 else "degraded"
        except Exception:
            return "unavailable"

    async def _check_ollama_health(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                return "healthy" if response.status_code == 200 else "degraded"
        except Exception:
            return "unavailable"

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        parts = []
        for i, result in enumerate(search_results):
            meta = result.get("metadata", {})
            parts.append(
                f"[Source {i+1} - Page {meta.get('page', 'unknown')}, "
                f"{meta.get('section', '')}]\n{meta.get('text', '')}\n"
            )
        return "\n---\n".join(parts)

    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[SourceDocument]:
        sources = []
        seen_docs = set()
        for result in search_results:
            meta = result.get("metadata", {})
            doc_id = meta.get("doc_id")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            doc_meta = self.db.get(doc_id) or {}
            sources.append(
                SourceDocument(
                    doc_name=doc_meta.get("filename", "unknown"),
                    page_numbers=str(meta.get("page", "unknown")),
                    section=meta.get("section", ""),
                    relevance=result.get("score", 0.0),
                    excerpt=meta.get("text", "")[:200] + "...",
                )
            )
        return sources

    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        if not search_results:
            return 0.0
        top_scores = [r.get("score", 0.0) for r in search_results[:3]]
        return round(sum(top_scores) / len(top_scores), 2)

    def _get_auth_headers(self) -> Dict[str, str]:
        if settings.jadevectordb_api_key:
            return {"Authorization": f"Bearer {settings.jadevectordb_api_key}"}
        return {}

    def _get_status_message(self, status: str, error: Optional[str] = None) -> str:
        if status == "failed":
            return f"Processing failed: {error}" if error else "Processing failed"
        return {
            "pending": "Queued for processing",
            "processing": "Extracting text and generating embeddings",
            "complete": "Processing complete",
        }.get(status, "Unknown status")


# Global instance
production_service = ProductionRAGService()
