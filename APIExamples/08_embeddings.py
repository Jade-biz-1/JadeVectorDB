#!/usr/bin/env python3
"""
JadeVectorDB API Example — Embeddings Generation
====================================================

JadeVectorDB can generate vector embeddings from text, eliminating the
need for a separate embedding service. This is useful for:
  - Quick prototyping without external API keys
  - Consistent embedding generation across your pipeline
  - Reducing infrastructure complexity

This example demonstrates:
  1. Generate an embedding from a single text string
  2. Use the generated embedding for search
  3. Generate embeddings with different model/provider options

APIs covered:
  - client.generate_embeddings(text, input_type, model, provider)
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
# 1. Generate a single text embedding
# ---------------------------------------------------------------------------
# Pass a string and receive a vector representation. The embedding dimension
# depends on the model selected.

print("=== 1. Generate Single Embedding ===")
try:
    result = client.generate_embeddings(
        text="What is a vector database and how does it work?",
        input_type="text",       # "text" for general text, "query" for search queries
        model="default",         # Use the server's default embedding model
        provider="default",      # Use the server's default provider
    )

    embedding = result.get("embedding", result.get("embeddings", []))
    if isinstance(embedding, list) and len(embedding) > 0:
        # If it's a list of lists (batch), take the first
        if isinstance(embedding[0], list):
            vec = embedding[0]
        else:
            vec = embedding
        print(f"  Dimension : {len(vec)}")
        print(f"  First 5   : {vec[:5]}")
        print(f"  Model     : {result.get('model', 'N/A')}")
    else:
        print(f"  Response: {json.dumps(result, indent=2)[:300]}")

except JadeVectorDBError as e:
    print(f"Generate embedding: {e}")
    print("  (Embedding generation may require a configured embedding provider)")

# ---------------------------------------------------------------------------
# 2. End-to-end: embed text, then search
# ---------------------------------------------------------------------------
# A common pattern: generate an embedding for a user's query, then use it
# to search the vector database.

print("\n=== 2. Embed + Search Pattern ===")
try:
    # Step 1: Generate embedding for the search query
    query_result = client.generate_embeddings(
        text="lightweight laptop for programming",
        input_type="query",      # Hint that this is a search query
    )

    query_vec = query_result.get("embedding", query_result.get("embeddings", []))
    if isinstance(query_vec, list) and len(query_vec) > 0:
        if isinstance(query_vec[0], list):
            query_vec = query_vec[0]
        print(f"  Generated query vector with {len(query_vec)} dimensions")
        print("  Ready to call client.search(database_id, query_vector=query_vec, top_k=5)")
    else:
        print(f"  Response: {json.dumps(query_result, indent=2)[:200]}")

except JadeVectorDBError as e:
    print(f"Embed + search: {e}")

# ---------------------------------------------------------------------------
# 3. Generate with explicit model and provider
# ---------------------------------------------------------------------------
# If multiple embedding models or providers are configured, you can
# specify which one to use.

print("\n=== 3. Explicit Model / Provider ===")
try:
    result = client.generate_embeddings(
        text="Graph neural networks for recommendation systems",
        input_type="text",
        model="default",          # Replace with your model name
        provider="default",       # Replace with your provider
    )
    print(f"  Response keys: {list(result.keys())}")
except JadeVectorDBError as e:
    print(f"Explicit model: {e}")

print("\nEmbedding examples complete.")
