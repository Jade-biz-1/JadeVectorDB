"""
Production RAG service using JadeVectorDB + Ollama
"""

import asyncio
import httpx
import time
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
from ..logging_config import get_logger
from ..metrics import (
    rag_queries_total,
    rag_query_duration_seconds,
    rag_query_confidence,
    rag_documents_processed_total,
    rag_document_processing_duration_seconds,
    rag_document_chunks,
    rag_embedding_batch_duration_seconds,
    rag_active_processing_tasks,
    rag_stored_documents,
    rag_stored_chunks,
)

log = get_logger(__name__)


class ProductionRAGService:
    """Production RAG service with JadeVectorDB and Ollama"""

    def __init__(self):
        self.jadevectordb_url = settings.jadevectordb_url.rstrip("/")
        self.ollama_url = settings.ollama_url.rstrip("/")
        self.database_id = settings.jadevectordb_database_id
        self.doc_processor = DocumentProcessor()
        self.start_time = datetime.utcnow()
        self.db = MetadataDB(settings.metadata_db_path)
        settings.upload_dir.mkdir(parents=True, exist_ok=True)

        # Persistent clients — reused across requests (connection pooling)
        _conn_limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
        self._jade_client = httpx.AsyncClient(
            base_url=self.jadevectordb_url,
            timeout=30,
            limits=_conn_limits,
        )
        self._ollama_client = httpx.AsyncClient(
            base_url=self.ollama_url,
            timeout=settings.llm_timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def aclose(self) -> None:
        """Close persistent HTTP clients — call from FastAPI shutdown handler."""
        await self._jade_client.aclose()
        await self._ollama_client.aclose()
        log.info("http_clients_closed")

    async def ensure_ready(self) -> None:
        """Resolve/create the JadeVectorDB database at startup.

        JadeVectorDB generates an internal ID (e.g. db_<timestamp>) that is
        different from the human-readable name we configure. This method lists
        existing databases, finds the one matching our configured name, and
        stores its actual ID. If it doesn't exist yet, it creates it.
        """
        try:
            await self._ensure_database_id()
        except Exception as e:
            # Non-fatal at startup — the first real query will fail cleanly and
            # the health endpoint will report degraded status.
            log.error("ensure_ready_failed", error=str(e))

    async def _ensure_database_id(self) -> None:
        configured_name = settings.jadevectordb_database_id
        resp = await self._jade_client.get(
            "/v1/databases", headers=self._get_auth_headers(), timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        for db in data.get("databases", []):
            if db.get("name") == configured_name:
                self.database_id = db["databaseId"]
                log.info(
                    "jadevectordb_database_found",
                    name=configured_name,
                    database_id=self.database_id,
                )
                return

        # Database doesn't exist yet — create it
        payload = {
            "name": configured_name,
            "description": "RAG document store for EnterpriseRAG",
            "vectorDimension": settings.embedding_dimension,
            "indexType": "HNSW",
        }
        create_resp = await self._jade_client.post(
            "/v1/databases", json=payload, headers=self._get_auth_headers(), timeout=10
        )
        create_resp.raise_for_status()
        self.database_id = create_resp.json()["databaseId"]
        log.info(
            "jadevectordb_database_created",
            name=configured_name,
            database_id=self.database_id,
            vector_dimension=settings.embedding_dimension,
        )

    # ── Public API ────────────────────────────────────────────

    async def query(
        self,
        question: str,
        category: Optional[str] = "all",
        top_k: int = 5,
    ) -> QueryResponse:
        start_time = datetime.utcnow()
        _t0 = time.perf_counter()
        self.db.increment_query_count()
        log.info("query_started", question=question[:120], category=category, top_k=top_k)
        _cat = category or "all"

        try:
            question_embedding = await self._generate_embedding(question, is_query=True)
            search_results = await self._search_vectors(question_embedding, category, top_k)
            search_results = self._enrich_results(search_results)
            context = self._build_context(search_results)
            answer = await self._generate_answer(question, context)
            sources = self._extract_sources(search_results)
            processing_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            confidence = self._calculate_confidence(search_results)

            rag_queries_total.labels(status="success", category=_cat).inc()
            rag_query_confidence.observe(confidence)

            log.info(
                "query_completed",
                processing_time_ms=processing_time,
                confidence=confidence,
                sources=len(sources),
            )
            self.db.log_query(
                question=question,
                category=_cat,
                mode="production",
                confidence=confidence,
                processing_time_ms=processing_time,
                sources_count=len(sources),
                success=True,
            )
            return QueryResponse(
                success=True,
                answer=answer,
                sources=sources,
                query=question,
                category=_cat,
                timestamp=datetime.utcnow().isoformat() + "Z",
                confidence=confidence,
                processing_time_ms=processing_time,
                mode="production",
            )

        except Exception as e:
            rag_queries_total.labels(status="error", category=_cat).inc()
            log.error("query_failed", error=str(e), question=question[:120])
            self.db.log_query(
                question=question,
                category=_cat,
                mode="production",
                confidence=0.0,
                processing_time_ms=0,
                sources_count=0,
                success=False,
            )
            return QueryResponse(
                success=False,
                answer=f"Error processing query: {str(e)}",
                sources=[],
                query=question,
                category=_cat,
                timestamp=datetime.utcnow().isoformat() + "Z",
                confidence=0.0,
                processing_time_ms=0,
                mode="production",
            )
        finally:
            rag_query_duration_seconds.observe(time.perf_counter() - _t0)

    async def upload_document(
        self, filename: str, category: str = "general", file_content: bytes = b""
    ) -> dict:
        doc_id = f"doc_{str(uuid.uuid4())[:12]}"
        uploaded_at = datetime.utcnow().isoformat() + "Z"

        suffix = Path(filename).suffix
        file_path = settings.upload_dir / f"{doc_id}{suffix}"
        file_path.write_bytes(file_content)

        self.db.insert({
            "id": doc_id,
            "filename": filename,
            "category": category,
            "status": "processing",
            "uploaded_at": uploaded_at,
        })
        self.db.update(doc_id, {"file_path": str(file_path)})

        log.info(
            "document_upload_received",
            doc_id=doc_id,
            filename=filename,
            category=category,
            size_bytes=len(file_content),
        )
        asyncio.create_task(self._process_document(doc_id, filename, category, file_content))

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "status": "processing",
            "message": f"Document '{filename}' is being processed",
        }

    async def reprocess_document(self, doc_id: str) -> dict:
        meta = self.db.get(doc_id)
        if not meta:
            raise ValueError(f"Document {doc_id} not found")

        file_path = meta.get("file_path")
        if not file_path or not Path(file_path).exists():
            raise FileNotFoundError(
                f"Original file not found for document '{meta['filename']}'. "
                "Delete and re-upload to reprocess."
            )

        log.info("document_reprocess_started", doc_id=doc_id, filename=meta["filename"])
        await self._delete_vectors(doc_id)
        self.db.update(doc_id, {
            "status": "processing",
            "processed_at": None,
            "chunk_count": None,
            "chunks_done": 0,
            "error": None,
        })

        file_content = Path(file_path).read_bytes()
        asyncio.create_task(
            self._process_document(doc_id, meta["filename"], meta.get("category", "general"), file_content)
        )

        return {
            "success": True,
            "doc_id": doc_id,
            "filename": meta["filename"],
            "status": "processing",
            "message": f"Document '{meta['filename']}' is being reprocessed",
        }

    def _refresh_doc_gauges(self) -> None:
        """Update rag_stored_* gauges from the SQLite metadata store."""
        try:
            docs = self.db.list_all(limit=10_000)
            counts: Dict[str, int] = {}
            for d in docs:
                counts[d.get("status", "unknown")] = counts.get(d.get("status", "unknown"), 0) + 1
            for status in ("complete", "processing", "failed", "pending"):
                rag_stored_documents.labels(status=status).set(counts.get(status, 0))
            rag_stored_chunks.set(
                sum(d.get("chunk_count", 0) for d in docs if d.get("status") == "complete")
            )
        except Exception:
            pass

    async def _process_document(
        self, doc_id: str, filename: str, category: str, file_content: bytes
    ):
        log.info("document_processing_started", doc_id=doc_id, filename=filename)
        rag_active_processing_tasks.inc()
        _t0 = time.perf_counter()
        _status = "failed"
        try:
            text = await self.doc_processor.extract_text(filename, file_content)
            chunks = await self.doc_processor.chunk_text(text)
            total_chunks = len(chunks)
            self.db.update(doc_id, {"chunk_count": total_chunks, "chunks_done": 0})
            log.info("document_chunked", doc_id=doc_id, total_chunks=total_chunks)

            embeddings = []
            _embed_t0 = time.perf_counter()
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self._generate_embedding(chunk["text"])
                except Exception as embed_err:
                    raise RuntimeError(
                        f"Embedding failed on chunk {i + 1}/{total_chunks}: "
                        f"{type(embed_err).__name__}: {embed_err!r}"
                    ) from embed_err
                embeddings.append(embedding)
                self.db.update(doc_id, {"chunks_done": i + 1})
            rag_embedding_batch_duration_seconds.observe(time.perf_counter() - _embed_t0)

            # Persist chunk texts in SQLite so search results can be enriched
            self.db.chunks_insert_batch(doc_id, chunks)

            await self._store_vectors(doc_id, chunks, embeddings, category)

            self.db.update(doc_id, {
                "status": "complete",
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "chunk_count": total_chunks,
                "chunks_done": total_chunks,
                "error": None,
            })
            log.info("document_processing_complete", doc_id=doc_id, chunks=total_chunks)
            _status = "success"
            rag_document_chunks.observe(total_chunks)

        except Exception as e:
            log.error("document_processing_failed", doc_id=doc_id, error=str(e))
            self.db.update(doc_id, {"status": "failed", "error": str(e)})
        finally:
            rag_documents_processed_total.labels(status=_status).inc()
            rag_document_processing_duration_seconds.observe(time.perf_counter() - _t0)
            rag_active_processing_tasks.dec()
            self._refresh_doc_gauges()

    async def get_processing_status(self, doc_id: str) -> ProcessingStatus:
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

    async def list_documents(self, offset: int = 0, limit: int = 100) -> tuple[List[DocumentInfo], int]:
        total = self.db.count_all()
        rows = self.db.list_all(offset=offset, limit=limit)
        docs = [
            DocumentInfo(
                id=r["doc_id"],
                filename=r["filename"],
                category=r.get("category", "general"),
                status=r["status"],
                uploaded_at=r["uploaded_at"],
                processed_at=r.get("processed_at"),
                chunk_count=r.get("chunk_count"),
                error=r.get("error"),
            )
            for r in rows
        ]
        return docs, total

    async def delete_document(self, doc_id: str) -> DocumentDeleteResponse:
        meta = self.db.get(doc_id)
        if not meta:
            log.warning("document_delete_not_found", doc_id=doc_id)
            return DocumentDeleteResponse(
                success=False, doc_id=doc_id, doc_name="unknown",
                chunks_found=0, chunks_deleted=0, chunks_failed=0,
                message=f"Document {doc_id} not found",
            )

        doc_name = meta["filename"]
        chunk_count = meta.get("chunk_count") or 0
        log.info("document_delete_started", doc_id=doc_id, filename=doc_name)

        try:
            deleted_count = await self._delete_vectors(doc_id)

            file_path = meta.get("file_path")
            if file_path:
                try:
                    Path(file_path).unlink(missing_ok=True)
                except Exception:
                    pass

            self.db.chunks_delete_for_doc(doc_id)
            self.db.delete(doc_id)
            asyncio.create_task(self._check_and_compact())

            log.info("document_deleted", doc_id=doc_id, chunks_deleted=deleted_count)
            return DocumentDeleteResponse(
                success=True, doc_id=doc_id, doc_name=doc_name,
                chunks_found=chunk_count, chunks_deleted=deleted_count, chunks_failed=0,
                message=f"Document '{doc_name}' deleted successfully",
                note="Index compaction scheduled if threshold exceeded",
            )

        except Exception as e:
            log.error("document_delete_failed", doc_id=doc_id, error=str(e))
            return DocumentDeleteResponse(
                success=False, doc_id=doc_id, doc_name=doc_name,
                chunks_found=chunk_count, chunks_deleted=0, chunks_failed=chunk_count,
                message=f"Error deleting document: {str(e)}",
            )

    def get_analytics(self, recent_limit: int = 20) -> dict:
        return self.db.get_analytics(recent_limit=recent_limit)

    async def get_stats(self) -> SystemStats:
        all_docs = self.db.list_all(limit=10_000)
        total_docs = self.db.count_all()
        total_chunks = sum(
            d.get("chunk_count") or 0 for d in all_docs if d.get("status") == "complete"
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

    # ── Private: JadeVectorDB / Ollama calls ─────────────────

    async def _generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        # mxbai-embed-large: query prefix for retrieval, no prefix for documents.
        # nomic-embed-text: "search_query:" / "search_document:" prefixes.
        model = settings.ollama_embedding_model
        if is_query and "mxbai" in model:
            prompt = "Represent this sentence for searching relevant passages: " + text
        elif "nomic" in model:
            prompt = ("search_query: " if is_query else "search_document: ") + text
        else:
            prompt = text
        response = await self._ollama_client.post(
            "/api/embeddings",
            json={"model": model, "prompt": prompt},
            timeout=settings.embedding_timeout,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    async def _search_vectors(
        self, query_embedding: List[float], category: Optional[str], top_k: int
    ) -> List[Dict[str, Any]]:
        if category and category != "all":
            endpoint = f"/v1/databases/{self.database_id}/search/advanced"
            payload = {
                "queryVector": query_embedding,
                "topK": top_k,
                "includeMetadata": True,
                "filters": {
                    "combination": "AND",
                    "conditions": [
                        {"field": "metadata.category", "op": "EQUALS", "value": category}
                    ],
                },
            }
        else:
            endpoint = f"/v1/databases/{self.database_id}/search"
            payload = {"queryVector": query_embedding, "topK": top_k, "includeMetadata": True}

        response = await self._jade_client.post(
            endpoint, json=payload, headers=self._get_auth_headers(),
            timeout=settings.search_timeout,
        )
        response.raise_for_status()
        return response.json().get("results", [])

    async def _generate_answer(self, question: str, context: str) -> str:
        prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.

Context from documentation:
{context}

Question: {question}

Instructions:
- Provide a clear, step-by-step answer
- Use bullet points or numbered lists for procedures
- Include safety warnings if relevant
- If the context doesn't contain the answer, say so
- Be concise but complete

Answer:"""

        # Use streaming so bytes arrive incrementally — avoids read timeout
        # on slow CPU inference where stream=False sends nothing until done.
        import json as _json
        tokens: List[str] = []
        async with self._ollama_client.stream(
            "POST",
            "/api/generate",
            json={
                "model": settings.ollama_llm_model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": settings.llm_temperature,
                    "num_predict": settings.llm_max_tokens,
                },
            },
            timeout=settings.llm_timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    chunk = _json.loads(line)
                    tokens.append(chunk.get("response", ""))
                    if chunk.get("done"):
                        break
        return "".join(tokens)

    async def _store_vectors(
        self,
        doc_id: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        category: str,
    ):
        total = len(chunks)
        log.info("vectors_store_started", doc_id=doc_id, total=total)
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector_id = f"{doc_id}_chunk_{i}"
            payload = {
                "id": vector_id,
                "values": embedding,
                "metadata": {
                    "doc_id": str(doc_id),
                    "chunk_index": str(i),
                    "page": str(chunk.get("page", "unknown")),
                    "section": str(chunk.get("section", "")),
                    "category": str(category),
                },
            }
            # Retry up to 3 times on transient errors
            for attempt in range(3):
                try:
                    response = await self._jade_client.post(
                        f"/v1/databases/{self.database_id}/vectors",
                        json=payload,
                        headers=self._get_auth_headers(),
                        timeout=30,
                    )
                    response.raise_for_status()
                    break
                except (httpx.RemoteProtocolError, httpx.ConnectError) as e:
                    if attempt == 2:
                        raise
                    wait = 2 ** attempt
                    log.warning("vector_store_retry", doc_id=doc_id, vector_id=vector_id, attempt=attempt + 1, error=str(e))
                    await asyncio.sleep(wait)

            if (i + 1) % 100 == 0 or (i + 1) == total:
                log.info("vectors_stored_progress", doc_id=doc_id, done=i + 1, total=total)
            # Small pace to avoid overwhelming JadeVectorDB
            if (i + 1) % 50 == 0:
                await asyncio.sleep(0.1)

    async def _delete_vectors(self, doc_id: str) -> int:
        meta = self.db.get(doc_id)
        chunk_count = (meta.get("chunk_count") or 0) if meta else 0
        if chunk_count == 0:
            return 0

        deleted = 0
        for i in range(chunk_count):
            resp = await self._jade_client.delete(
                f"/v1/databases/{self.database_id}/vectors/{doc_id}_chunk_{i}",
                headers=self._get_auth_headers(),
            )
            if resp.status_code in (200, 204):
                deleted += 1
        return deleted

    async def _check_and_compact(self):
        try:
            resp = await self._jade_client.get(
                f"/v1/databases/{self.database_id}/stats",
                headers=self._get_auth_headers(),
            )
            resp.raise_for_status()
            stats = resp.json()
            total = stats.get("total_vectors", 0)
            deleted = stats.get("deleted_vectors", 0)
            remaining = total - deleted
            # Skip compaction when no live vectors remain — compacting an
            # empty index corrupts the HNSW structure in JadeVectorDB.
            if total > 0 and remaining > 0 and (deleted / total) > 0.10:
                await self._jade_client.post(
                    f"/v1/databases/{self.database_id}/compact",
                    headers=self._get_auth_headers(),
                )
                log.info("compaction_triggered", database_id=self.database_id)
        except Exception:
            pass  # Best-effort

    async def _check_jadevectordb_health(self) -> str:
        try:
            resp = await self._jade_client.get("/health", headers=self._get_auth_headers())
            return "healthy" if resp.status_code == 200 else "degraded"
        except Exception:
            return "unavailable"

    async def _check_ollama_health(self) -> str:
        try:
            resp = await self._ollama_client.get("/api/tags")
            return "healthy" if resp.status_code == 200 else "degraded"
        except Exception:
            return "unavailable"

    # ── Private: response helpers ─────────────────────────────

    def _enrich_results(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Look up chunk text from SQLite and attach it to each search result."""
        for result in search_results:
            vector_id = result.get("vectorId", "")
            chunk = self.db.chunk_get(vector_id) if vector_id else None
            result["_chunk"] = chunk or {}
        return search_results

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        parts = []
        for i, result in enumerate(search_results):
            chunk = result.get("_chunk", {})
            parts.append(
                f"[Source {i+1} - Page {chunk.get('page', 'unknown')}, "
                f"{chunk.get('section', '')}]\n{chunk.get('text', '')}\n"
            )
        return "\n---\n".join(parts)

    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[SourceDocument]:
        sources = []
        seen_docs: set = set()
        for result in search_results:
            chunk = result.get("_chunk", {})
            doc_id = chunk.get("doc_id")
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            doc_meta = self.db.get(doc_id) or {}
            sources.append(
                SourceDocument(
                    doc_name=doc_meta.get("filename", "unknown"),
                    page_numbers=str(chunk.get("page", "unknown")),
                    section=chunk.get("section", ""),
                    relevance=result.get("score", 0.0),
                    excerpt=chunk.get("text", "")[:200] + "...",
                )
            )
        return sources

    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        if not search_results:
            return 0.0
        scores = [r.get("score", 0.0) for r in search_results[:3]]
        return round(sum(scores) / len(scores), 2)

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
