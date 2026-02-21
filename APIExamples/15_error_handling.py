#!/usr/bin/env python3
"""
JadeVectorDB API Example — Error Handling
============================================

Production applications need robust error handling. This example shows
best practices for handling JadeVectorDB errors, implementing retries,
and building resilient vector search pipelines.

Topics covered:
  1. Catching JadeVectorDBError with context
  2. Handling specific error scenarios (not found, dimension mismatch, auth)
  3. Connection error handling and retry pattern
  4. Graceful degradation pattern
  5. Bulk operation error handling

APIs covered:
  - JadeVectorDBError — base exception for all client errors
  - ImportExportError — exception for import/export operations
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError
from jadevectordb.import_export import ImportExportError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

# ---------------------------------------------------------------------------
# 1. Basic error handling
# ---------------------------------------------------------------------------
# JadeVectorDBError is raised for any API-level failure (4xx/5xx responses).
# The error message contains the server's response text.

print("=== 1. Basic Error Handling ===")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

try:
    # This should fail — the database doesn't exist
    client.get_database("nonexistent-database-id")
except JadeVectorDBError as e:
    print(f"Caught expected error: {e}")
print()

# ---------------------------------------------------------------------------
# 2. Handling specific scenarios
# ---------------------------------------------------------------------------
# Check the error message to handle different failure modes differently.

print("=== 2. Specific Error Scenarios ===")

# 2a. Vector not found
try:
    result = client.retrieve_vector("some-db", "nonexistent-vector")
    if result is None:
        print("  Vector not found (returned None — not an error)")
except JadeVectorDBError as e:
    error_msg = str(e).lower()
    if "not found" in error_msg or "404" in error_msg:
        print(f"  Resource not found: {e}")
    else:
        print(f"  Other error: {e}")

# 2b. Authentication failure
print()
try:
    bad_client = JadeVectorDB(base_url=SERVER_URL, api_key="invalid-token")
    bad_client.list_databases()
except JadeVectorDBError as e:
    error_msg = str(e).lower()
    if "401" in error_msg or "unauthorized" in error_msg or "auth" in error_msg:
        print(f"  Authentication failed (expected): {e}")
    else:
        print(f"  Error: {e}")

# ---------------------------------------------------------------------------
# 3. Retry pattern for transient failures
# ---------------------------------------------------------------------------
# Network issues, server restarts, and rate limits are transient — retrying
# after a brief delay often succeeds.

print("\n=== 3. Retry Pattern ===")


def with_retry(func, max_retries=3, base_delay=1.0):
    """
    Execute a function with exponential backoff retry.

    Retries on:
      - Connection errors (server unreachable)
      - 5xx server errors (transient failures)
      - Rate limit errors (429)

    Does NOT retry on:
      - 4xx client errors (bad request, not found, unauthorized)
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except JadeVectorDBError as e:
            last_error = e
            error_msg = str(e).lower()

            # Don't retry client errors (4xx) — they won't succeed
            is_client_error = any(code in error_msg for code in ["400", "401", "403", "404", "409", "422"])
            if is_client_error:
                raise

            # Retry on server errors and transient failures
            delay = base_delay * (2 ** attempt)
            print(f"  Attempt {attempt + 1} failed: {e}")
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)

        except ConnectionError as e:
            last_error = e
            delay = base_delay * (2 ** attempt)
            print(f"  Connection error: {e}")
            print(f"  Retrying in {delay}s...")
            time.sleep(delay)

    raise last_error


# Demo the retry pattern
try:
    result = with_retry(lambda: client.get_health())
    print(f"  Health check succeeded: {result.get('status', '?')}")
except (JadeVectorDBError, ConnectionError) as e:
    print(f"  All retries failed: {e}")

# ---------------------------------------------------------------------------
# 4. Graceful degradation
# ---------------------------------------------------------------------------
# When a vector database call fails, your application shouldn't crash.
# Fall back to a default response or alternative data source.

print("\n=== 4. Graceful Degradation ===")


def search_products(query_vector, top_k=5):
    """
    Search for products with graceful fallback.

    If the vector search fails, returns an empty result set instead
    of crashing the application.
    """
    try:
        results = client.search(
            database_id="products-db",
            query_vector=query_vector,
            top_k=top_k,
        )
        return {"source": "vector_search", "results": results}

    except JadeVectorDBError as e:
        # Log the error for debugging
        print(f"  Vector search unavailable: {e}")
        # Return empty results — the UI can show "no results found"
        return {"source": "fallback", "results": []}

    except Exception as e:
        # Catch unexpected errors too
        print(f"  Unexpected error: {e}")
        return {"source": "error", "results": []}


dummy_query = [0.1] * 128
result = search_products(dummy_query)
print(f"  Source: {result['source']}, Results: {len(result['results'])}")

# ---------------------------------------------------------------------------
# 5. Bulk operation error handling
# ---------------------------------------------------------------------------
# When processing many items, don't let one failure stop the entire batch.

print("\n=== 5. Bulk Operation Error Handling ===")


def bulk_store_with_error_tracking(client, database_id, items):
    """
    Store multiple vectors, tracking successes and failures individually.
    """
    results = {"success": 0, "failed": 0, "errors": []}

    for item in items:
        try:
            client.store_vector(
                database_id=database_id,
                vector_id=item["id"],
                values=item["values"],
                metadata=item.get("metadata"),
            )
            results["success"] += 1
        except JadeVectorDBError as e:
            results["failed"] += 1
            results["errors"].append({"id": item["id"], "error": str(e)})

    return results


# This will fail (no real database), but demonstrates the pattern
test_items = [
    {"id": "item-1", "values": [0.1] * 128, "metadata": {"name": "Widget A"}},
    {"id": "item-2", "values": [0.2] * 128, "metadata": {"name": "Widget B"}},
    {"id": "item-3", "values": [0.3] * 5, "metadata": {"name": "Bad dimension"}},  # Wrong dimension
]

try:
    db_id = client.create_database("error-handling-demo", vector_dimension=128)
    stats = bulk_store_with_error_tracking(client, db_id, test_items)
    print(f"  Successes: {stats['success']}")
    print(f"  Failures : {stats['failed']}")
    for err in stats["errors"]:
        print(f"    - {err['id']}: {err['error'][:80]}")
    client.delete_database(db_id)
except JadeVectorDBError as e:
    print(f"  Demo setup failed: {e}")

print("\nError handling examples complete.")
