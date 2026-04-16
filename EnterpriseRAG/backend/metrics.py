"""
Prometheus metrics for EnterpriseRAG.
Import this module early (before any code that records observations)
so that all metric objects are singletons registered with the default registry.
"""

from prometheus_client import Counter, Gauge, Histogram

# ── Query metrics ─────────────────────────────────────────────
rag_queries_total = Counter(
    "rag_queries_total",
    "Total RAG queries processed",
    ["status", "category"],
)

rag_query_duration_seconds = Histogram(
    "rag_query_duration_seconds",
    "End-to-end RAG query duration in seconds",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

rag_query_confidence = Histogram(
    "rag_query_confidence_score",
    "Distribution of RAG query confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Document processing metrics ───────────────────────────────
rag_documents_processed_total = Counter(
    "rag_documents_processed_total",
    "Total documents processed",
    ["status"],
)

rag_document_processing_duration_seconds = Histogram(
    "rag_document_processing_duration_seconds",
    "Total document processing duration (chunking + embedding + vector storage)",
    buckets=[30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 2400.0, 3600.0],
)

rag_document_chunks = Histogram(
    "rag_document_chunks_total",
    "Number of chunks produced per processed document",
    buckets=[50.0, 100.0, 250.0, 500.0, 1000.0, 2000.0, 5000.0],
)

rag_embedding_batch_duration_seconds = Histogram(
    "rag_embedding_batch_duration_seconds",
    "Total time to generate all embeddings for one document",
    buckets=[10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0],
)

# ── Live-state gauges ─────────────────────────────────────────
rag_active_processing_tasks = Gauge(
    "rag_active_processing_tasks",
    "Number of documents currently being processed",
)

rag_stored_documents = Gauge(
    "rag_stored_documents_total",
    "Total documents in the metadata store by status",
    ["status"],
)

rag_stored_chunks = Gauge(
    "rag_stored_chunks_total",
    "Total chunks whose vectors are stored in JadeVectorDB",
)
