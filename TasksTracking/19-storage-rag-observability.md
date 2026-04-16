# Phase 19: Storage Fixes, RAG Embedding Quality & Observability

**Phase**: Bug-fix & Enhancement  
**Branch**: `runAndFix`  
**Started**: 2026-04-15  
**Completed**: 2026-04-16

---

## Background

Three interrelated problems were discovered during real-world testing of the EnterpriseRAG application against JadeVectorDB:

1. The batch vector endpoint silently drops all vectors (returns 201 but never persists).
2. JadeVectorDB had a hard 1,000-vector capacity limit baked into `database_layer.cpp`, causing 400 errors after the first ~1,000 inserts into any database.
3. Query retrieval with `nomic-embed-text` returned wrong documents ("What is Delta Lake?" → Deep Learning content) due to inverted similarity scores for out-of-vocabulary proper nouns.

In addition, the observability stack (Prometheus + Grafana) was wired up but unused — no EnterpriseRAG metrics and no pre-built dashboards.

---

## Tasks

### T19.01: Document and work around broken batch vector endpoint
**Status**: ✅ COMPLETED  
**Description**: `POST /v1/databases/{id}/vectors/batch` returns HTTP 201 but silently discards all vectors.  
- Rewrote `_store_vectors()` in `EnterpriseRAG/backend/services/rag_service.py` to use single-vector `POST /v1/databases/{id}/vectors` per chunk
- Removed `text` field from vector metadata (control characters in PDF chunks caused 400 errors on single-vector endpoint too)
- Added `⚠️` note to `README.md` API endpoint table
- **Root cause not fixed in JadeVectorDB C++ layer** — tracked as a known issue

---

### T19.02: Fix hard 1,000-vector capacity limit in JadeVectorDB
**Status**: ✅ COMPLETED  
**Description**: `database_layer.cpp:947` hardcoded `initial_capacity=1000`. The data section of the mmap'd vector file was sized at creation time as `1000 × dim × 4` bytes and never grew. `allocate_vector_space()` returned 0 after the 1,000th insert, causing every subsequent `store_vector` call to return HTTP 400.

**Fix Part 1 — Raise initial capacity**:
- Changed `initial_capacity` from `1000` → `50000` in `backend/src/services/database_layer.cpp:947`

**Fix Part 2 — Auto-growing data section**:
- Added `resize_data_section(DatabaseFile*, size_t new_capacity)` to `memory_mapped_vector_store.cpp`:
  - Unmaps file, truncates to new size, remaps
  - Moves STRINGS section after enlarged DATA section using `memmove`
  - Updates all `string_offset` pointers in every index entry
  - Updates `data_capacity` and `vector_ids_offset` in header
  - Flushes to disk
- Added `ensure_capacity()` implementing the previously-declared stub
- Wired auto-growth into `allocate_vector_space()`: doubles capacity until space fits, then continues
- Added declaration to `memory_mapped_vector_store.h`

**Files changed**:
- `backend/src/services/database_layer.cpp`
- `backend/src/storage/memory_mapped_vector_store.cpp`
- `backend/src/storage/memory_mapped_vector_store.h`

---

### T19.03: Fix EnterpriseRAG retrieval quality — switch to mxbai-embed-large
**Status**: ✅ COMPLETED  
**Description**: `nomic-embed-text` gave inverted similarity scores for proper nouns not in its training vocabulary (e.g. "Delta Lake" scored 0.55 against Databricks content, 0.64 against irrelevant floating-point text). With 2,600 Deep Learning chunks vs 138 Databricks chunks, statistically wrong documents dominated top-5 results.

**Fix**:
- Switched `OLLAMA_EMBEDDING_MODEL` from `nomic-embed-text` → `mxbai-embed-large`
- Updated `EMBEDDING_DIMENSION` from `768` → `1024`
- Added model-specific query prefix logic in `_generate_embedding()`:
  - mxbai query prefix: `"Represent this sentence for searching relevant passages: "`
  - nomic prefixes retained as fallback: `"search_query:"` / `"search_document:"`
- Updated `EnterpriseRAG/backend/utils/config.py` defaults
- Updated `EnterpriseRAG/.env.docker`

**Verification**: "What is Delta Lake?" → `data-lakehouse-for-dummies-databricks-special-edition.pdf`, confidence 0.57, relevance 0.64. "What is positional encoding?" → `DeepLearningNew.pdf`, confidence 0.77, relevance 0.78.

**Files changed**:
- `EnterpriseRAG/backend/services/rag_service.py`
- `EnterpriseRAG/backend/utils/config.py`
- `EnterpriseRAG/.env.docker`

---

### T19.04: Improve embedding error logging
**Status**: ✅ COMPLETED  
**Description**: `httpx` timeout/connection exceptions have empty `str()` representations. The existing error log `f"Embedding failed on chunk {i+1}/{total}: {embed_err}"` produced blank error messages, making failures impossible to diagnose.

**Fix**: Changed to `f"... {type(embed_err).__name__}: {embed_err!r}"` to always include the exception class and full repr.

**Files changed**:
- `EnterpriseRAG/backend/services/rag_service.py`

---

### T19.05: Wire up Prometheus + Grafana observability
**Status**: ✅ COMPLETED  
**Description**: Prometheus was scraping only itself and JadeVectorDB. Grafana provisioning directory was empty — no datasources, no dashboards.

**EnterpriseRAG backend instrumentation**:
- Added `prometheus-fastapi-instrumentator>=6.3.0` to `requirements.txt`
- Created `EnterpriseRAG/backend/metrics.py` — 15 custom metrics:
  - `rag_queries_total` (counter, labels: status, category)
  - `rag_query_duration_seconds` (histogram)
  - `rag_query_confidence_score` (histogram)
  - `rag_documents_processed_total` (counter, label: status)
  - `rag_document_processing_duration_seconds` (histogram)
  - `rag_document_chunks_total` (histogram)
  - `rag_embedding_batch_duration_seconds` (histogram)
  - `rag_active_processing_tasks` (gauge)
  - `rag_stored_documents_total` (gauge, label: status)
  - `rag_stored_chunks_total` (gauge)
- Wired `Instrumentator` into `main.py` — exposes `/metrics` endpoint
- Instrumented `query()`, `_process_document()` in `rag_service.py`
- Added `_refresh_doc_gauges()` to keep stored docs/chunks gauges current after each document

**Prometheus**:
- Added `rag-backend:8000` scrape target to `prometheus.yml`

**Grafana provisioning** (auto-loads on container start):
- `grafana/provisioning/datasources/prometheus.yaml` — Prometheus datasource
- `grafana/provisioning/dashboards/dashboard.yaml` — dashboard provider
- `grafana/provisioning/dashboards/enterpriserag.json` — 12-panel EnterpriseRAG dashboard
- `grafana/provisioning/dashboards/jadevectordb.json` — 12-panel JadeVectorDB dashboard

**Files changed**:
- `EnterpriseRAG/backend/metrics.py` (new)
- `EnterpriseRAG/backend/main.py`
- `EnterpriseRAG/backend/requirements.txt`
- `EnterpriseRAG/backend/services/rag_service.py`
- `prometheus.yml`
- `grafana/provisioning/datasources/prometheus.yaml` (new)
- `grafana/provisioning/dashboards/dashboard.yaml` (new)
- `grafana/provisioning/dashboards/enterpriserag.json` (new)
- `grafana/provisioning/dashboards/jadevectordb.json` (new)

---

## Completion Criteria — ALL MET

- Single-vector endpoint used for all vector storage; no silent data loss ✅
- JadeVectorDB data section grows automatically beyond initial capacity ✅
- "What is Delta Lake?" returns Databricks PDF with confidence > 0.5 ✅
- "What is positional encoding?" returns DeepLearning PDF with confidence > 0.7 ✅
- `/metrics` endpoint live on `rag-backend:8000`, all 4 Prometheus targets `up` ✅
- Grafana loads with two pre-built dashboards on container start ✅
- README.md updated with batch endpoint caveat and EnterpriseRAG section ✅
