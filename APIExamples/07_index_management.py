#!/usr/bin/env python3
"""
JadeVectorDB API Example — Index Management
==============================================

Indexes control how vectors are organized for search. Choosing the right
index type and parameters dramatically affects search speed, accuracy, and
memory usage.

Index types:
  - HNSW   : Hierarchical Navigable Small World — best general-purpose.
              Great balance of speed and recall. Tunable via M and efConstruction.
  - IVF    : Inverted File Index — partitions vectors into clusters.
              Fast for large datasets. Tunable via nlist and nprobe.
  - LSH    : Locality-Sensitive Hashing — memory-efficient approximate search.
  - FLAT   : Brute-force exact search — perfect recall but O(n) per query.

This example demonstrates:
  1. Create an HNSW index with custom parameters
  2. List all indexes on a database
  3. Update index parameters
  4. Delete an index

APIs covered:
  - client.create_index(database_id, index_type, name, parameters)
  - client.list_indexes(database_id)
  - client.update_index(database_id, index_id, parameters)
  - client.delete_index(database_id, index_id)
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
db_id = client.create_database(
    name="index-mgmt-example",
    vector_dimension=128,
    index_type="FLAT",       # Start with FLAT, then add better indexes
)
print(f"Created database: {db_id}\n")

# ---------------------------------------------------------------------------
# 1. Create an HNSW index
# ---------------------------------------------------------------------------
# HNSW parameters:
#   M               : Number of bi-directional links per node (default 16).
#                     Higher M = better recall but more memory.
#   efConstruction  : Size of the dynamic candidate list during build
#                     (default 200). Higher = better index quality but
#                     slower build time.

print("=== 1. Create HNSW Index ===")
try:
    index = client.create_index(
        database_id=db_id,
        index_type="HNSW",
        name="product-hnsw-v1",
        parameters={
            "M": 16,
            "efConstruction": 200,
        },
    )
    index_id = index.get("index_id", index.get("indexId", ""))
    print(f"Created HNSW index: {index_id}")
    print(json.dumps(index, indent=2))
except JadeVectorDBError as e:
    print(f"Create index: {e}")
    index_id = None

# ---------------------------------------------------------------------------
# 2. List all indexes
# ---------------------------------------------------------------------------

print("\n=== 2. List Indexes ===")
try:
    indexes = client.list_indexes(database_id=db_id)
    print(f"Found {len(indexes)} index(es):")
    for idx in indexes:
        idx_id = idx.get("index_id", idx.get("indexId", "?"))
        idx_type = idx.get("index_type", idx.get("indexType", "?"))
        idx_name = idx.get("name", "unnamed")
        print(f"  - {idx_name} ({idx_type}) — ID: {idx_id}")
except JadeVectorDBError as e:
    print(f"List indexes: {e}")

# ---------------------------------------------------------------------------
# 3. Update index parameters
# ---------------------------------------------------------------------------
# Some parameters can be updated without rebuilding the index (e.g.,
# efSearch for HNSW, nprobe for IVF).

print("\n=== 3. Update Index Parameters ===")
if index_id:
    try:
        updated = client.update_index(
            database_id=db_id,
            index_id=index_id,
            parameters={
                "efSearch": 150,     # Increase search quality at query time
            },
        )
        print(f"Updated index: {json.dumps(updated, indent=2)}")
    except JadeVectorDBError as e:
        print(f"Update index: {e}")
else:
    print("Skipping (no index_id)")

# ---------------------------------------------------------------------------
# 4. Delete an index
# ---------------------------------------------------------------------------
# Removing an index frees memory but queries will fall back to a less
# efficient search method (or fail if no other index exists).

print("\n=== 4. Delete Index ===")
if index_id:
    try:
        deleted = client.delete_index(database_id=db_id, index_id=index_id)
        print(f"Deleted index {index_id}: {deleted}")
    except JadeVectorDBError as e:
        print(f"Delete index: {e}")
else:
    print("Skipping (no index_id)")

# --- Cleanup ---
client.delete_database(database_id=db_id)
print(f"\nCleaned up database: {db_id}")
print("Index management examples complete.")
