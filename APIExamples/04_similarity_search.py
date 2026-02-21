#!/usr/bin/env python3
"""
JadeVectorDB API Example — Similarity Search
===============================================

This example demonstrates:
  1. Basic similarity search with top-k and threshold
  2. Advanced search with metadata filters, inclusion options
  3. Real-world pattern: "find similar products"

APIs covered:
  - client.search(database_id, query_vector, top_k, threshold, filters)
  - client.advanced_search(database_id, query_vector, top_k, threshold,
                           filters, include_metadata, include_values)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, Vector, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# --- Setup: create database and seed with product vectors ---
DIMENSION = 128
db_id = client.create_database(
    name="search-example",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created database: {db_id}\n")


def make_vector(seed: float) -> list:
    return [round(seed + i * 0.001, 6) for i in range(DIMENSION)]


# Seed with sample product embeddings
products = [
    ("laptop-001",   0.10, {"name": "Gaming Laptop",      "category": "electronics", "price": 1299}),
    ("laptop-002",   0.11, {"name": "Business Laptop",     "category": "electronics", "price": 999}),
    ("phone-001",    0.20, {"name": "Smartphone Pro",      "category": "electronics", "price": 899}),
    ("shoe-001",     0.50, {"name": "Running Shoes",       "category": "sports",      "price": 129}),
    ("shoe-002",     0.51, {"name": "Trail Running Shoes", "category": "sports",      "price": 149}),
    ("headphone-001",0.15, {"name": "Noise-Cancel Buds",   "category": "electronics", "price": 199}),
    ("book-001",     0.80, {"name": "ML Textbook",         "category": "books",       "price": 59}),
    ("book-002",     0.81, {"name": "Deep Learning Guide", "category": "books",       "price": 45}),
]

for vid, seed, meta in products:
    client.store_vector(db_id, vid, make_vector(seed), meta)
print(f"Stored {len(products)} product vectors\n")

# ---------------------------------------------------------------------------
# 1. Basic similarity search
# ---------------------------------------------------------------------------
# Given a query vector (e.g., a user's search embedding), find the top-k
# most similar items. Results are ranked by cosine similarity (or the
# metric configured on the index).

print("=== 1. Basic Search — 'find laptops like this' ===")
query = make_vector(0.105)   # Close to laptop-001 (0.10) and laptop-002 (0.11)

results = client.search(
    database_id=db_id,
    query_vector=query,
    top_k=5,                 # Return at most 5 results
)

for r in results:
    print(f"  {r.get('id', '?'):20s}  similarity={r.get('similarity', r.get('score', '?'))}")
print()

# ---------------------------------------------------------------------------
# 2. Search with similarity threshold
# ---------------------------------------------------------------------------
# The threshold parameter filters out results below a minimum similarity
# score. Useful when you only want highly relevant matches.

print("=== 2. Search with Threshold (>= 0.95) ===")
results = client.search(
    database_id=db_id,
    query_vector=query,
    top_k=10,
    threshold=0.95,          # Only return very similar vectors
)
print(f"  {len(results)} results above threshold")
for r in results:
    print(f"  {r.get('id', '?'):20s}  similarity={r.get('similarity', r.get('score', '?'))}")
print()

# ---------------------------------------------------------------------------
# 3. Advanced search with metadata filters
# ---------------------------------------------------------------------------
# advanced_search() adds fine-grained control:
#   - filters:          Only return vectors whose metadata matches these criteria
#   - include_metadata:  Include metadata in results (default True)
#   - include_values:    Include raw vector values in results (default False)

print("=== 3. Advanced Search — electronics only, include values ===")
results = client.advanced_search(
    database_id=db_id,
    query_vector=query,
    top_k=5,
    filters={"category": "electronics"},   # Only match electronics
    include_metadata=True,                 # We want to see product names
    include_values=False,                  # Don't need raw vectors in response
)

for r in results:
    meta = r.get("metadata", {})
    print(f"  {r.get('id', '?'):20s}  {meta.get('name', '?'):25s}  ${meta.get('price', '?')}")
print()

# ---------------------------------------------------------------------------
# 4. Advanced search — books category, include vector values
# ---------------------------------------------------------------------------
# Setting include_values=True is useful when you need to inspect or
# re-process the stored embeddings (e.g., for debugging or re-ranking).

print("=== 4. Advanced Search — books, include vector values ===")
book_query = make_vector(0.805)  # Close to book embeddings

results = client.advanced_search(
    database_id=db_id,
    query_vector=book_query,
    top_k=3,
    filters={"category": "books"},
    include_metadata=True,
    include_values=True,
)

for r in results:
    meta = r.get("metadata", {})
    vals = r.get("values", [])
    print(f"  {r.get('id', '?'):20s}  {meta.get('name', '?'):25s}  vector[0:3]={vals[:3]}")
print()

# --- Cleanup ---
client.delete_database(database_id=db_id)
print(f"Cleaned up database: {db_id}")
print("Similarity search examples complete.")
