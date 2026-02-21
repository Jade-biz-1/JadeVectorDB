#!/usr/bin/env python3
"""
JadeVectorDB API Example — Vector Operations
===============================================

This example covers every vector CRUD operation:
  1. Store a single vector with metadata
  2. Batch store multiple vectors
  3. Retrieve a single vector by ID
  4. List vectors with pagination
  5. Update a vector's values and metadata
  6. Batch-get multiple vectors in one call
  7. Delete a vector

APIs covered:
  - client.store_vector(database_id, vector_id, values, metadata)
  - client.batch_store_vectors(database_id, vectors)
  - client.retrieve_vector(database_id, vector_id)
  - client.list_vectors(database_id, limit, offset)
  - client.update_vector(database_id, vector_id, values, metadata)
  - client.batch_get_vectors(database_id, vector_ids)
  - client.delete_vector(database_id, vector_id)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, Vector, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# --- Setup: create a temporary database ---
DIMENSION = 128
db_id = client.create_database(
    name="vector-ops-example",
    description="Temporary DB for vector operations demo",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created temp database: {db_id}\n")


def make_vector(seed: float, dim: int = DIMENSION) -> list:
    """Generate a deterministic vector for demo purposes."""
    return [round(seed + i * 0.001, 6) for i in range(dim)]


# ---------------------------------------------------------------------------
# 1. Store a single vector
# ---------------------------------------------------------------------------
# Each vector needs a unique ID, a list of float values matching the database
# dimension, and optional metadata (key-value pairs for filtering/display).

print("=== 1. Store Single Vector ===")
client.store_vector(
    database_id=db_id,
    vector_id="product-001",
    values=make_vector(0.1),
    metadata={
        "name": "Wireless Headphones",
        "category": "electronics",
        "price": 79.99,
        "in_stock": True,
    },
)
print("Stored: product-001")

# ---------------------------------------------------------------------------
# 2. Batch store multiple vectors
# ---------------------------------------------------------------------------
# For bulk ingestion, batch_store_vectors() is significantly faster than
# calling store_vector() in a loop — it sends all vectors in one request.

print("\n=== 2. Batch Store Vectors ===")
products = [
    Vector(
        id="product-002",
        values=make_vector(0.2),
        metadata={"name": "Running Shoes", "category": "sports", "price": 129.99},
    ),
    Vector(
        id="product-003",
        values=make_vector(0.3),
        metadata={"name": "Coffee Maker", "category": "kitchen", "price": 49.99},
    ),
    Vector(
        id="product-004",
        values=make_vector(0.4),
        metadata={"name": "Yoga Mat", "category": "sports", "price": 29.99},
    ),
    Vector(
        id="product-005",
        values=make_vector(0.5),
        metadata={"name": "Desk Lamp", "category": "office", "price": 39.99},
    ),
]
client.batch_store_vectors(database_id=db_id, vectors=products)
print(f"Batch stored {len(products)} vectors")

# ---------------------------------------------------------------------------
# 3. Retrieve a single vector
# ---------------------------------------------------------------------------
# Returns a Vector dataclass with .id, .values, and .metadata attributes.
# Returns None if the vector does not exist (404).

print("\n=== 3. Retrieve Vector ===")
vec = client.retrieve_vector(database_id=db_id, vector_id="product-001")
if vec:
    print(f"  ID       : {vec.id}")
    print(f"  Values   : [{vec.values[0]}, {vec.values[1]}, ... ] ({len(vec.values)} dims)")
    print(f"  Metadata : {vec.metadata}")
else:
    print("  Vector not found")

# ---------------------------------------------------------------------------
# 4. List vectors with pagination
# ---------------------------------------------------------------------------
# Useful for browsing or exporting. Supports limit/offset pagination.

print("\n=== 4. List Vectors (page 1, limit 3) ===")
page = client.list_vectors(database_id=db_id, limit=3, offset=0)
vectors_list = page.get("vectors", page) if isinstance(page, dict) else page
if isinstance(vectors_list, list):
    for v in vectors_list:
        vid = v.get("id", v.get("vector_id", "?"))
        print(f"  - {vid}")
else:
    print(f"  Response: {json.dumps(page, indent=2)[:200]}")

# ---------------------------------------------------------------------------
# 5. Update a vector
# ---------------------------------------------------------------------------
# Replaces the vector's values and optionally its metadata.
# The vector ID must already exist in the database.

print("\n=== 5. Update Vector ===")
client.update_vector(
    database_id=db_id,
    vector_id="product-001",
    values=make_vector(0.15),   # New embedding (e.g., after re-encoding)
    metadata={
        "name": "Wireless Headphones Pro",   # Updated product name
        "category": "electronics",
        "price": 99.99,                       # Price increase
        "in_stock": True,
    },
)
print("Updated: product-001 (new values + metadata)")

# Verify the update
updated_vec = client.retrieve_vector(database_id=db_id, vector_id="product-001")
if updated_vec:
    print(f"  New price: {updated_vec.metadata.get('price')}")

# ---------------------------------------------------------------------------
# 6. Batch-get multiple vectors
# ---------------------------------------------------------------------------
# Efficiently retrieve several vectors in one network round-trip.

print("\n=== 6. Batch Get Vectors ===")
batch_result = client.batch_get_vectors(
    database_id=db_id,
    vector_ids=["product-002", "product-003", "product-005"],
)
print(f"Retrieved {len(batch_result)} vectors:")
for v in batch_result:
    vid = v.get("id", v.get("vector_id", "?"))
    meta = v.get("metadata", {})
    print(f"  - {vid}: {meta.get('name', '?')}")

# ---------------------------------------------------------------------------
# 7. Delete a vector
# ---------------------------------------------------------------------------
# Permanently removes the vector. Returns True on success.

print("\n=== 7. Delete Vector ===")
client.delete_vector(database_id=db_id, vector_id="product-004")
print("Deleted: product-004")

# Verify deletion
missing = client.retrieve_vector(database_id=db_id, vector_id="product-004")
print(f"Retrieve after delete: {missing}")  # Should be None

# --- Cleanup ---
client.delete_database(database_id=db_id)
print(f"\nCleaned up database: {db_id}")
print("Vector operations complete.")
