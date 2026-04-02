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


class ProductionRAGService:
    """Production RAG service with JadeVectorDB and Ollama"""

    def __init__(self):
        self.jadevectordb_url = settings.jadevectordb_url
        self.ollama_url = settings.ollama_url
        self.database_id = settings.jadevectordb_database_id
        self.doc_processor = DocumentProcessor()
        self.query_count = 0
        self.start_time = datetime.utcnow()

        # Document metadata store (in production, use persistent storage)
        self.document_metadata: Dict[str, Dict[str, Any]] = {}

    async def query(
        self,
        question: str,
        device_type: Optional[str] = "all",
        top_k: int = 5,
    ) -> QueryResponse:
        """
        Execute RAG query pipeline
        """
        start_time = datetime.utcnow()
        self.query_count += 1

        try:
            # 1. Generate embedding for question
            question_embedding = await self._generate_embedding(question)

            # 2. Search vector database
            search_results = await self._search_vectors(
                question_embedding, device_type, top_k
            )

            # 3. Build context from chunks
            context = self._build_context(search_results)

            # 4. Generate answer using LLM
            answer = await self._generate_answer(question, context)

            # 5. Extract source citations
            sources = self._extract_sources(search_results)

            # Calculate processing time
            processing_time = int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            )

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
            # Fallback error response
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
        """
        Upload and process document
        """
        doc_id = f"doc_{str(uuid.uuid4())[:12]}"

        # Store metadata
        self.document_metadata[doc_id] = {
            "id": doc_id,
            "filename": filename,
            "device_type": device_type,
            "status": "processing",
            "uploaded_at": datetime.utcnow().isoformat() + "Z",
            "processed_at": None,
            "chunk_count": None,
            "error": None,
        }

        # Process document in background
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
        Process document: extract text, chunk, embed, and store
        """
        try:
            # 1. Extract text from PDF/DOCX
            text = await self.doc_processor.extract_text(filename, file_content)

            # 2. Chunk text semantically
            chunks = await self.doc_processor.chunk_text(text)

            # 3. Generate embeddings for all chunks
            embeddings = []
            for chunk in chunks:
                embedding = await self._generate_embedding(chunk["text"])
                embeddings.append(embedding)

            # 4. Store vectors in JadeVectorDB
            await self._store_vectors(doc_id, chunks, embeddings, device_type)

            # 5. Update metadata
            self.document_metadata[doc_id].update({
                "status": "complete",
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "chunk_count": len(chunks),
            })

        except Exception as e:
            # Update metadata with error
            self.document_metadata[doc_id].update({
                "status": "failed",
                "error": str(e),
            })

    async def get_processing_status(self, doc_id: str) -> ProcessingStatus:
        """
        Get document processing status
        """
        metadata = self.document_metadata.get(doc_id)
        if not metadata:
            raise ValueError(f"Document {doc_id} not found")

        return ProcessingStatus(
            doc_id=doc_id,
            status=metadata["status"],
            progress=100 if metadata["status"] == "complete" else 50,
            message=self._get_status_message(metadata["status"]),
            chunks_processed=metadata.get("chunk_count", 0),
            total_chunks=metadata.get("chunk_count"),
        )

    async def list_documents(self) -> List[DocumentInfo]:
        """
        List all documents
        """
        return [
            DocumentInfo(
                id=meta["id"],
                filename=meta["filename"],
                device_type=meta["device_type"],
                status=meta["status"],
                uploaded_at=meta["uploaded_at"],
                processed_at=meta.get("processed_at"),
                chunk_count=meta.get("chunk_count"),
                error=meta.get("error"),
            )
            for meta in self.document_metadata.values()
        ]

    async def delete_document(self, doc_id: str) -> DocumentDeleteResponse:
        """
        Delete document and all associated vectors
        """
        metadata = self.document_metadata.get(doc_id)
        if not metadata:
            return DocumentDeleteResponse(
                success=False,
                doc_id=doc_id,
                doc_name="unknown",
                chunks_found=0,
                chunks_deleted=0,
                chunks_failed=0,
                message=f"Document {doc_id} not found",
            )

        doc_name = metadata["filename"]
        chunk_count = metadata.get("chunk_count", 0)

        try:
            # Delete vectors from JadeVectorDB
            deleted_count = await self._delete_vectors(doc_id)

            # Remove from metadata
            del self.document_metadata[doc_id]

            # Schedule compaction check
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
        """
        Get system statistics
        """
        total_docs = len(self.document_metadata)
        total_chunks = sum(
            meta.get("chunk_count", 0)
            for meta in self.document_metadata.values()
            if meta.get("chunk_count")
        )
        uptime = int((datetime.utcnow() - self.start_time).total_seconds())

        # Check component health
        db_status = await self._check_jadevectordb_health()
        llm_status = await self._check_ollama_health()

        return SystemStats(
            status="healthy" if db_status == "healthy" and llm_status == "healthy" else "degraded",
            total_documents=total_docs,
            total_chunks=total_chunks,
            total_queries=self.query_count,
            uptime_seconds=uptime,
            mode="production",
            db_status=db_status,
            llm_status=llm_status,
        )

    # ==================== Private Methods ====================

    async def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding using Ollama
        """
        async with httpx.AsyncClient(timeout=settings.embedding_timeout) as client:
            response = await client.post(
                f"{self.ollama_url}/api/embeddings",
                json={
                    "model": settings.ollama_embedding_model,
                    "prompt": text,
                },
            )
            response.raise_for_status()
            return response.json()["embedding"]

    async def _search_vectors(
        self, query_embedding: List[float], device_type: Optional[str], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search vectors in JadeVectorDB
        """
        async with httpx.AsyncClient(timeout=settings.search_timeout) as client:
            # Build filter
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
        """
        Generate answer using Ollama LLM
        """
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
        Store vectors in JadeVectorDB
        """
        # Prepare vectors with metadata
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vectors.append({
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
            })

        # Batch insert
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/vectors/batch",
                json={"vectors": vectors},
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()

    async def _delete_vectors(self, doc_id: str) -> int:
        """
        Delete all vectors for a document
        """
        # Query for all vector IDs matching doc_id
        async with httpx.AsyncClient(timeout=30) as client:
            # Search with filter to find all chunks
            response = await client.post(
                f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/query",
                json={
                    "filter": {"doc_id": {"$eq": doc_id}},
                    "limit": 10000,
                },
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            vector_ids = [result["id"] for result in response.json()["results"]]

            # Delete in batches
            for i in range(0, len(vector_ids), 100):
                batch = vector_ids[i:i+100]
                await client.post(
                    f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/vectors/delete-batch",
                    json={"ids": batch},
                    headers=self._get_auth_headers(),
                )

            return len(vector_ids)

    async def _check_and_compact(self):
        """
        Check if compaction is needed and trigger it
        """
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Get database stats
                response = await client.get(
                    f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/stats",
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                stats = response.json()

                total_vectors = stats.get("total_vectors", 0)
                deleted_vectors = stats.get("deleted_vectors", 0)

                # Trigger compaction if >10% deleted
                if total_vectors > 0 and (deleted_vectors / total_vectors) > 0.10:
                    await client.post(
                        f"{self.jadevectordb_url}/api/v1/databases/{self.database_id}/compact",
                        headers=self._get_auth_headers(),
                    )
        except Exception:
            # Compaction is best-effort, don't fail if it errors
            pass

    async def _check_jadevectordb_health(self) -> str:
        """
        Check JadeVectorDB health
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.jadevectordb_url}/health")
                if response.status_code == 200:
                    return "healthy"
                return "degraded"
        except Exception:
            return "unavailable"

    async def _check_ollama_health(self) -> str:
        """
        Check Ollama health
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    return "healthy"
                return "degraded"
        except Exception:
            return "unavailable"

    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Build context string from search results
        """
        context_parts = []
        for i, result in enumerate(search_results):
            metadata = result.get("metadata", {})
            text = metadata.get("text", "")
            page = metadata.get("page", "unknown")
            section = metadata.get("section", "")

            context_parts.append(
                f"[Source {i+1} - Page {page}, {section}]\n{text}\n"
            )

        return "\n---\n".join(context_parts)

    def _extract_sources(self, search_results: List[Dict[str, Any]]) -> List[SourceDocument]:
        """
        Extract source citations from search results
        """
        sources = []
        seen_docs = set()

        for result in search_results:
            metadata = result.get("metadata", {})
            doc_id = metadata.get("doc_id")

            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)

            doc_meta = self.document_metadata.get(doc_id, {})
            sources.append(
                SourceDocument(
                    doc_name=doc_meta.get("filename", "unknown"),
                    page_numbers=str(metadata.get("page", "unknown")),
                    section=metadata.get("section", ""),
                    relevance=result.get("score", 0.0),
                    excerpt=metadata.get("text", "")[:200] + "...",
                )
            )

        return sources

    def _calculate_confidence(self, search_results: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on search results
        """
        if not search_results:
            return 0.0

        # Average of top 3 scores
        top_scores = [result.get("score", 0.0) for result in search_results[:3]]
        return round(sum(top_scores) / len(top_scores), 2)

    def _get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for JadeVectorDB
        """
        headers = {}
        if settings.jadevectordb_api_key:
            headers["Authorization"] = f"Bearer {settings.jadevectordb_api_key}"
        return headers

    def _get_status_message(self, status: str) -> str:
        """
        Get human-readable status message
        """
        messages = {
            "pending": "Queued for processing",
            "processing": "Extracting text and generating embeddings",
            "complete": "Processing complete",
            "failed": "Processing failed",
        }
        return messages.get(status, "Unknown status")


# Global instance
production_service = ProductionRAGService()
