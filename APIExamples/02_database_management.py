#!/usr/bin/env python3
"""
JadeVectorDB API Example — Database Management
=================================================

This example demonstrates the full database lifecycle:
  1. Create a database with custom dimension and index type
  2. List all databases
  3. Get detailed information about a single database
  4. Update a database (rename, change description)
  5. Delete a database

APIs covered:
  - client.create_database(name, description, vector_dimension, index_type)
  - client.list_databases()
  - client.get_database(database_id)
  - client.update_database(database_id, name, description, vector_dimension, index_type)
  - client.delete_database(database_id)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# 1. Create a database
# ---------------------------------------------------------------------------
# Every database has a fixed vector dimension and an index type.
#   - vector_dimension: Must match the size of vectors you store later.
#                       Common values: 128 (small), 384 (MiniLM), 768 (BERT),
#                       1536 (OpenAI ada-002), 3072 (OpenAI text-embedding-3-large)
#   - index_type:       Algorithm used for nearest-neighbor search.
#                       Options: "HNSW" (default, best general-purpose),
#                       "IVF" (fast for large datasets), "LSH" (memory-efficient),
#                       "FLAT" (exact, brute-force — best for <10k vectors)

print("=== 1. Create Database ===")
try:
    db_id = client.create_database(
        name="product-embeddings",
        description="Vector store for e-commerce product descriptions",
        vector_dimension=384,       # Matches sentence-transformers/all-MiniLM-L6-v2
        index_type="HNSW",
    )
    print(f"Created database: {db_id}")
except JadeVectorDBError as e:
    print(f"Error: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 2. List all databases
# ---------------------------------------------------------------------------
# Returns a list of database summary objects. Useful for admin UIs and
# discovering which databases are available on the server.

print("\n=== 2. List Databases ===")
databases = client.list_databases()
# The response may be a list or a dict with a 'databases' key
db_list = databases if isinstance(databases, list) else databases.get("databases", [])
for db in db_list:
    name = db.get("name", "unnamed")
    did = db.get("databaseId", db.get("database_id", "?"))
    print(f"  - {name}  (ID: {did})")

# ---------------------------------------------------------------------------
# 3. Get database details
# ---------------------------------------------------------------------------
# Retrieves the full configuration for a single database, including its
# current vector count, dimension, index type, and timestamps.

print(f"\n=== 3. Get Database: {db_id} ===")
db_info = client.get_database(database_id=db_id)
print(json.dumps(db_info, indent=2))

# ---------------------------------------------------------------------------
# 4. Update a database
# ---------------------------------------------------------------------------
# You can update the name, description, or even the index type of an existing
# database. Only the fields you pass are changed; others remain untouched.

print(f"\n=== 4. Update Database: {db_id} ===")
updated = client.update_database(
    database_id=db_id,
    description="Product embeddings (v2) — includes images and reviews",
)
print(f"Updated description: {updated.get('description', '(see response)')}")

# ---------------------------------------------------------------------------
# 5. Delete a database
# ---------------------------------------------------------------------------
# Permanently removes the database and all its vectors. This is irreversible.
# In production, consider requiring confirmation or soft-delete patterns.

print(f"\n=== 5. Delete Database: {db_id} ===")
deleted = client.delete_database(database_id=db_id)
print(f"Deleted: {deleted}")

print("\nDatabase lifecycle complete.")
