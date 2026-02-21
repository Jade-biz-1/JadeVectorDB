#!/usr/bin/env python3
"""
JadeVectorDB API Example — Getting Started
============================================

This example demonstrates how to:
  1. Initialize the JadeVectorDB client
  2. Check server health
  3. Retrieve server status information

These are the first calls you should make to verify connectivity before
performing any data operations.

APIs covered:
  - JadeVectorDB()          — Client constructor
  - client.get_health()     — Lightweight liveness probe
  - client.get_status()     — Detailed server status (uptime, version, etc.)
"""

import os
import sys

# ---------------------------------------------------------------------------
# The jadevectordb package is not on PyPI — it ships with this repo under
# cli/python/. The line below makes it importable without a pip install.
# Alternatively, run: pip install -e cli/python/  (from the project root)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))

from jadevectordb import JadeVectorDB, JadeVectorDBError

# ---------------------------------------------------------------------------
# 1. Initialize the client
# ---------------------------------------------------------------------------
# The client needs the server URL and an optional API key for authentication.
# In production, always pass an API key. For local development, the key may
# be omitted if the server has authentication disabled.

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY")  # None is OK for health/status

client = JadeVectorDB(
    base_url=SERVER_URL,
    api_key=API_KEY,           # Pass None to connect without auth
)

print(f"Client initialized — pointing at {SERVER_URL}")
print()

# ---------------------------------------------------------------------------
# 2. Health check
# ---------------------------------------------------------------------------
# get_health() is a lightweight endpoint. Use it in readiness/liveness probes,
# monitoring dashboards, or as a pre-flight check before running a pipeline.

try:
    health = client.get_health()
    print("=== Health Check ===")
    print(f"  Status : {health.get('status', 'unknown')}")
    print(f"  Full   : {health}")
    print()
except JadeVectorDBError as e:
    print(f"Health check failed: {e}")
    print("Is the server running? Start it with: cd backend/build && ./jadevectordb")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 3. Server status
# ---------------------------------------------------------------------------
# get_status() returns richer information such as uptime, version, number of
# databases, and memory usage. Useful for admin dashboards and diagnostics.

try:
    status = client.get_status()
    print("=== Server Status ===")
    for key, value in status.items():
        print(f"  {key}: {value}")
    print()
except JadeVectorDBError as e:
    # Status may require authentication on some deployments
    print(f"Status check failed (may need auth): {e}")

print("Getting started complete — server is reachable and healthy.")
