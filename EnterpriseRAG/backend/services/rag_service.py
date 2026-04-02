"""
Production RAG service using JadeVectorDB + Ollama
"""

import asyncio
import httpx
import uuid
from datetime import datetime
from pathlib import Path
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
        self.jadevectordb_url = settings.jadevectordb_url.rstrip("/")
        self.ollama_url = settings.ollama_url
        self.database_id = settings.jadevectordb_database_id
        self.doc_processor = DocumentProcessor()
        self.start_time = datetime.utcnow()
        self.db = MetadataDB(settings.metadata_db_path)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────

    async def query(
        self,
        question: str,
        device_type: Optional[str] = "all",
        top_k: int = 5,
    ) -> QueryResponse:
        """Execute RAG query pipeline"""
        start_time = datetime.utcnow()
        self.db.increment_query_count()

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

        # Persist the file so reprocessing is possible later
        suffix = Path(filename).suffix
        file_path = settings.upload_dir / f"{doc_id}{suffix}"
        file_path.write_bytes(file_content)

        self.db.insert({
            "id": doc_id,
            "filename": filename,
            "device_type": device_type,
            "status": "processing",
            "uploaded_at": uploaded_at,
        })
        self.db.update(doc_id, {"file_path": str(file_path)})

        asyncio.create_task(self._process_document(doc_id, filename, device_type, file_content))

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "status": "processing",
            "message": f"Document '{filename}' is being processed",
        }

    async def reprocess_document(self, doc_id: str) -> dict:
        """
        Delete existing vectors for a document and reprocess from the saved file.
        Only works if the file was persisted on upload.
        """
        meta = self.db.get(doc_id)
        if not meta:
            raise ValueError(f"Document {doc_id} not found")

        file_path = meta.get("file_path")
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(
                f"Original file not found for document '{meta['filename']}'. "
                "Delete and re-upload to reprocess."
            )

        # Delete existing vectors
        await self._delete_vectors(doc_id)

        # Reset status and kick off processing again
        self.db.update(doc_id, {
            "status": "processing",
            "processed_at": None,
            "chunk_count": None,
            "chunks_done": 0,
            "error": None,
        })

        file_content = Path(file_path).read_bytes()
        asyncio.create_task(
            self._process_document(doc_id, meta["filename"], meta["device_type"], file_content)
        )

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": meta["filename"],
            "status": "processing",
            "message": f"Document '{meta['filename']}' is being reprocessed",
        }

    async def _process_document(
        self, doc_id: str, filename: str, device_type: str, file_content: bytes
    ):
        """
        Process document: extract → chunk → embed → store.
        Updates chunk progress in DB after each embedding so the status
        endpoint reflects real progress.
        """
        try:
            text = await self.doc_processor.extract_text(filename, file_content)
            chunks = await self.doc_processor.chunk_text(text)
            total_chunks = len(chunks)

            self.db.update(doc_id, {"chunk_count": total_chunks, "chunks_done": 0})

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

            await self._store_vectors(doc_id, chunks, embeddings, device_type)

            self.db.update(doc_id, {
                "status": "complete",
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "chunk_count": total_chunks,
                "chunks_done": total_chunks,
                "error": None,
            })

        except Exception as e:
            self.db.update(doc_id, {"status": "failed", "error": str(e)})

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
            progress = 5

        return ProcessingStatus(
            doc_id=doc_id,
            status=status,
            progress=progress,
            message=self._get_status_message(status, meta.get("error")),
            chunks_processed=chunks_done,
            total_chunks=total_chunks,
        )

    async def list_documents(self) -> List[DocumentInfo]:
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

            # Remove uploaded file from disk
            file_path = meta.get("file_path")
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass

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

    # ── Private: JadeVectorDB calls ───────────────────────────

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
        """
        Use /search for unfiltered queries and /search/advanced when a
        device_type filter is requested.
        JadeVectorDB filter format: {"combination":"AND","conditions":[{"field":"metadata.<key>","op":"EQUALS","value":"..."}]}
        Response shape: results[].vectorId / similarityScore / vector.metadata
        """
        async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
            if device_type and device_type != "all":
                endpoint = f"{self.jadevectordb_url}/v1/databases/{self.database_id}/search/advanced"
                payload = {
                    "queryVector": query_embedding,
                    "topK": top_k,
                    "includeMetadata": True,
                    "filters": {
                        "combination": "AND",
                        "conditions": [
                            {
                                "field": "metadata.device_type",
                                "op": "EQUALS",
                                "value": device_type,
                            }
                        ],
                    },
                }
            else:
                endpoint = f"{self.jadevectordb_url}/v1/databases/{self.database_id}/search"
                payload = {
                    "queryVector": query_embedding,
                    "topK": top_k,
                    "includeMetadata": True,
                }

            response = await client.post(endpoint, json=payload, headers=self._get_auth_headers())
            response.raise_for_status()
            return response.json().get("results", [])

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
        """
        Batch insert vectors. JadeVectorDB batch body:
        {"vectors": [{"id": "...", "values": [...], "metadata": {...}}]}
        """
        vectors = [
            {
                "id": f"{doc_id}_chunk_{i}",
                "values": embedding,          # ← "values", not "vector"
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
                f"{self.jadevectordb_url}/v1/databases/{self.database_id}/vectors/batch",
                json={"vectors": vectors},
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()

    async def _delete_vectors(self, doc_id: str) -> int:
        """
        Delete all chunk vectors for a document using known IDs
        (format: {doc_id}_chunk_{i}).  We know chunk_count from metadata.
        Falls back to 0 deleted if the document was never fully processed.
        """
        meta = self.db.get(doc_id)
        chunk_count = (meta.get("chunk_count") or 0) if meta else 0
        if chunk_count == 0:
            return 0

        deleted = 0
        async with httpx.AsyncClient(timeout=30) as client:
            for i in range(chunk_count):
                vector_id = f"{doc_id}_chunk_{i}"
                resp = await client.delete(
                    f"{self.jadevectordb_url}/v1/databases/{self.database_id}/vectors/{vector_id}",
                    headers=self._get_auth_headers(),
                )
                if resp.status_code in (200, 204):
                    deleted += 1

        return deleted

    async def _check_and_compact(self):
        """Best-effort: trigger compaction if >10% of vectors are deleted."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.jadevectordb_url}/v1/databases/{self.database_id}/stats",
                    headers=self._get_auth_headers(),
                )
                resp.raise_for_status()
                stats = resp.json()
                total = stats.get("total_vectors", 0)
                deleted = stats.get("deleted_vectors", 0)
                if total > 0 and (deleted / total) > 0.10:
                    await client.post(
                        f"{self.jadevectordb_url}/v1/databases/{self.database_id}/compact",
                        headers=self._get_auth_headers(),
                    )
        except Exception:
            pass

    async def _check_jadevectordb_health(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self.jadevectordb_url}/health",
                    headers=self._get_auth_headers(),
                )
                return "healthy" if resp.status_code == 200 else "degraded"
        except Exception:
            return "unavailable"

    async def _check_ollama_health(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.ollama_url}/api/tags")
                return "healthy" if resp.status_code == 200 else "degraded"
        except Exception:
            return "unavailable"

    # ── Private: response helpers ─────────────────────────────

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        parts = []
        for i, result in enumerate(search_results):
            # JadeVectorDB response: result.vector.metadata
            meta = result.get("vector", {}).get("metadata", {})
            parts.append(
                f"[Source {i+1} - Page {meta.get('page', 'unknown')}, "
                f"{meta.get('section', '')}]\n{meta.get('text', '')}\n"
            )
        return "\n---\n".join(parts)

    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[SourceDocument]:
        sources = []
        seen_docs: set = set()
        for result in search_results:
            meta = result.get("vector", {}).get("metadata", {})
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
                    relevance=result.get("similarityScore", 0.0),   # ← correct field
                    excerpt=meta.get("text", "")[:200] + "...",
                )
            )
        return sources

    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        if not search_results:
            return 0.0
        top_scores = [r.get("similarityScore", 0.0) for r in search_results[:3]]
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
