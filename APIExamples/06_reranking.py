#!/usr/bin/env python3
"""
JadeVectorDB API Example — Reranking
=======================================

Reranking is a two-stage retrieval pattern:
  Stage 1 (Retrieval): Fast approximate search returns candidate results
  Stage 2 (Reranking):  A more accurate model re-scores and reorders them

This is especially powerful when combining vector search with cross-encoder
models that understand query-document relevance at a deeper level.

This example demonstrates:
  1. Search with reranking (integrated two-stage pipeline)
  2. Standalone document reranking (bring your own candidates)
  3. Get reranking configuration
  4. Update reranking configuration

APIs covered:
  - client.rerank_search(database_id, query_text, query_vector, top_k)
  - client.rerank(query, documents, top_k)
  - client.get_reranking_config(database_id)
  - client.update_reranking_config(database_id, model_name, batch_size,
                                   score_threshold, combine_scores, rerank_weight)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# --- Setup ---
DIMENSION = 128
db_id = client.create_database(
    name="reranking-example",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created database: {db_id}\n")


def make_vector(seed: float) -> list:
    return [round(seed + i * 0.001, 6) for i in range(DIMENSION)]


# Seed with some documents
for i, (vid, seed, text) in enumerate([
    ("doc-1", 0.10, "Python is a popular programming language for data science"),
    ("doc-2", 0.20, "JavaScript frameworks like React and Vue dominate web development"),
    ("doc-3", 0.30, "Machine learning models require large datasets for training"),
    ("doc-4", 0.40, "SQL databases handle structured data and complex queries"),
    ("doc-5", 0.50, "Python data analysis with pandas and NumPy libraries"),
]):
    client.store_vector(db_id, vid, make_vector(seed), {"text": text, "index": i})
print("Stored 5 documents\n")

# ---------------------------------------------------------------------------
# 1. Search with reranking
# ---------------------------------------------------------------------------
# rerank_search() performs vector retrieval first, then reranks the top
# candidates using the query text for better relevance.

print("=== 1. Search with Reranking ===")
try:
    results = client.rerank_search(
        database_id=db_id,
        query_text="Python for data analysis",  # Text used by the reranker
        query_vector=make_vector(0.12),          # Vector for initial retrieval
        top_k=3,                                 # Final number of results
    )
    print(json.dumps(results, indent=2))
except JadeVectorDBError as e:
    print(f"Rerank search: {e}")

# ---------------------------------------------------------------------------
# 2. Standalone document reranking
# ---------------------------------------------------------------------------
# rerank() is useful when you already have a set of candidate documents
# (from any source) and want to reorder them by relevance to a query.
# This decouples retrieval from reranking.

print("\n=== 2. Standalone Reranking ===")
try:
    documents = [
        {"id": "a", "text": "Python is great for machine learning and AI applications"},
        {"id": "b", "text": "The weather forecast predicts rain for the weekend"},
        {"id": "c", "text": "Data science workflows often start with Python scripting"},
        {"id": "d", "text": "Deep learning frameworks like PyTorch are built on Python"},
    ]

    reranked = client.rerank(
        query="Python machine learning tools",
        documents=documents,
        top_k=3,                      # Only return the top 3 most relevant
    )
    print("Reranked results:")
    for r in reranked:
        print(f"  {r.get('id', '?'):5s}  score={r.get('score', '?')}")
except JadeVectorDBError as e:
    print(f"Standalone rerank: {e}")

# ---------------------------------------------------------------------------
# 3. Get reranking configuration
# ---------------------------------------------------------------------------
# Each database can have its own reranking model and parameters.

print("\n=== 3. Get Reranking Config ===")
try:
    config = client.get_reranking_config(database_id=db_id)
    print(json.dumps(config, indent=2))
except JadeVectorDBError as e:
    print(f"Get config: {e}")

# ---------------------------------------------------------------------------
# 4. Update reranking configuration
# ---------------------------------------------------------------------------
# Tune the reranking behavior per database:
#   - model_name:       Which reranking model to use
#   - batch_size:       How many documents to rerank at once
#   - score_threshold:  Minimum reranking score to include in results
#   - combine_scores:   Whether to combine vector + rerank scores
#   - rerank_weight:    Weight given to the reranking score (0.0-1.0)

print("\n=== 4. Update Reranking Config ===")
try:
    updated = client.update_reranking_config(
        database_id=db_id,
        batch_size=32,
        score_threshold=0.5,
        combine_scores=True,
        rerank_weight=0.6,         # 60% rerank score, 40% vector score
    )
    print(f"Updated config: {json.dumps(updated, indent=2)}")
except JadeVectorDBError as e:
    print(f"Update config: {e}")

# --- Cleanup ---
client.delete_database(database_id=db_id)
print(f"\nCleaned up database: {db_id}")
print("Reranking examples complete.")
