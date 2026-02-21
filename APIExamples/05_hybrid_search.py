#!/usr/bin/env python3
"""
JadeVectorDB API Example — Hybrid Search
==========================================

Hybrid search combines vector similarity (dense retrieval) with BM25 keyword
matching (sparse retrieval) for superior relevance. This is the recommended
approach for text-heavy workloads where both semantic meaning and exact
keyword matches matter.

This example demonstrates:
  1. Build a BM25 index on a metadata text field
  2. Check BM25 index build status
  3. Perform a hybrid search (vector + keyword)
  4. Use different fusion methods (RRF vs weighted linear)
  5. Add documents to the BM25 index incrementally
  6. Rebuild the BM25 index from scratch

APIs covered:
  - client.build_bm25_index(database_id, text_field, incremental)
  - client.get_bm25_index_status(database_id)
  - client.rebuild_bm25_index(database_id, text_field)
  - client.add_bm25_documents(database_id, documents)
  - client.hybrid_search(database_id, query_text, query_vector, top_k,
                         fusion_method, alpha, filters)
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, Vector, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# --- Setup ---
DIMENSION = 128
db_id = client.create_database(
    name="hybrid-search-example",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created database: {db_id}\n")


def make_vector(seed: float) -> list:
    return [round(seed + i * 0.001, 6) for i in range(DIMENSION)]


# Store documents with text in metadata
docs = [
    ("doc-1", 0.10, {"text": "Introduction to machine learning algorithms and neural networks",
                      "topic": "ml"}),
    ("doc-2", 0.20, {"text": "Deep learning with convolutional neural networks for image classification",
                      "topic": "ml"}),
    ("doc-3", 0.30, {"text": "Natural language processing with transformer models and attention",
                      "topic": "nlp"}),
    ("doc-4", 0.40, {"text": "Database indexing strategies for high-performance query execution",
                      "topic": "databases"}),
    ("doc-5", 0.50, {"text": "Vector similarity search using approximate nearest neighbor algorithms",
                      "topic": "databases"}),
]

for vid, seed, meta in docs:
    client.store_vector(db_id, vid, make_vector(seed), meta)
print(f"Stored {len(docs)} documents\n")

# ---------------------------------------------------------------------------
# 1. Build BM25 index
# ---------------------------------------------------------------------------
# Before hybrid search, you need to build a BM25 index on the text field.
# Specify which metadata field contains the searchable text.

print("=== 1. Build BM25 Index ===")
try:
    build_result = client.build_bm25_index(
        database_id=db_id,
        text_field="text",       # Index the "text" metadata field
        incremental=False,       # Full build (not incremental)
    )
    print(f"Build initiated: {json.dumps(build_result, indent=2)}")
except JadeVectorDBError as e:
    print(f"Build BM25 index: {e}")

# Give the index a moment to build
time.sleep(1)

# ---------------------------------------------------------------------------
# 2. Check BM25 index status
# ---------------------------------------------------------------------------

print("\n=== 2. BM25 Index Status ===")
try:
    status = client.get_bm25_index_status(database_id=db_id)
    print(f"Status: {json.dumps(status, indent=2)}")
except JadeVectorDBError as e:
    print(f"Status check: {e}")

# ---------------------------------------------------------------------------
# 3. Hybrid search with RRF fusion
# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion (RRF) merges results from vector search and BM25
# by combining their rank positions. This is robust and parameter-free.

print("\n=== 3. Hybrid Search (RRF) ===")
try:
    results = client.hybrid_search(
        database_id=db_id,
        query_text="neural networks machine learning",  # BM25 keyword query
        query_vector=make_vector(0.15),                  # Dense vector query
        top_k=5,
        fusion_method="rrf",     # Reciprocal Rank Fusion
    )
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  {r.get('id', '?'):10s}  score={r.get('score', r.get('similarity', '?'))}")
except JadeVectorDBError as e:
    print(f"Hybrid search: {e}")

# ---------------------------------------------------------------------------
# 4. Hybrid search with weighted linear fusion
# ---------------------------------------------------------------------------
# Weighted linear fusion lets you control the balance between vector
# similarity and BM25 keyword scores using the alpha parameter:
#   - alpha=1.0 → 100% vector, 0% BM25
#   - alpha=0.0 → 0% vector, 100% BM25
#   - alpha=0.7 → 70% vector, 30% BM25 (good default)

print("\n=== 4. Hybrid Search (Weighted Linear, alpha=0.3 — keyword-heavy) ===")
try:
    results = client.hybrid_search(
        database_id=db_id,
        query_text="database indexing query performance",
        query_vector=make_vector(0.42),
        top_k=3,
        fusion_method="linear",
        alpha=0.3,               # Favor BM25 keyword matching
    )
    print(f"Found {len(results)} results:")
    for r in results:
        print(f"  {r.get('id', '?'):10s}  score={r.get('score', r.get('similarity', '?'))}")
except JadeVectorDBError as e:
    print(f"Hybrid search: {e}")

# ---------------------------------------------------------------------------
# 5. Hybrid search with metadata filters
# ---------------------------------------------------------------------------

print("\n=== 5. Hybrid Search with Filter (topic=ml) ===")
try:
    results = client.hybrid_search(
        database_id=db_id,
        query_text="neural networks",
        query_vector=make_vector(0.15),
        top_k=5,
        fusion_method="rrf",
        filters={"topic": "ml"},  # Only search within ML documents
    )
    print(f"Found {len(results)} results (filtered to ml):")
    for r in results:
        print(f"  {r.get('id', '?'):10s}")
except JadeVectorDBError as e:
    print(f"Filtered hybrid search: {e}")

# ---------------------------------------------------------------------------
# 6. Add documents to BM25 index incrementally
# ---------------------------------------------------------------------------
# After the initial build, you can add new documents without rebuilding
# the entire index.

print("\n=== 6. Add Documents to BM25 Index ===")
try:
    result = client.add_bm25_documents(
        database_id=db_id,
        documents=[
            {"doc_id": "doc-6", "text": "Reinforcement learning for autonomous agents"},
            {"doc_id": "doc-7", "text": "Graph neural networks for social network analysis"},
        ],
    )
    print(f"Added documents: {json.dumps(result, indent=2)}")
except JadeVectorDBError as e:
    print(f"Add BM25 documents: {e}")

# ---------------------------------------------------------------------------
# 7. Rebuild BM25 index from scratch
# ---------------------------------------------------------------------------
# Use this after significant data changes or if the incremental index
# becomes fragmented.

print("\n=== 7. Rebuild BM25 Index ===")
try:
    result = client.rebuild_bm25_index(
        database_id=db_id,
        text_field="text",
    )
    print(f"Rebuild initiated: {json.dumps(result, indent=2)}")
except JadeVectorDBError as e:
    print(f"Rebuild: {e}")

# --- Cleanup ---
client.delete_database(database_id=db_id)
print(f"\nCleaned up database: {db_id}")
print("Hybrid search examples complete.")
