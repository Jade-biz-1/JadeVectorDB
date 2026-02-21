#!/usr/bin/env python3
"""
JadeVectorDB API Example — API Key Management
================================================

API keys provide machine-to-machine authentication for programmatic access.
They are ideal for:
  - CI/CD pipelines that push embeddings to the database
  - Backend services that query the vector store
  - Third-party integrations with scoped permissions

This example demonstrates:
  1. Create an API key for a user
  2. List all API keys
  3. Use an API key for authentication
  4. Revoke an API key

APIs covered:
  - client.create_api_key(user_id, description, permissions, validity_days)
  - client.list_api_keys(user_id)
  - client.revoke_api_key(key_id)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# You'll need a valid user_id. In production, get this from your auth system.
# For this demo, we'll create a temporary user.
USER_ID = os.environ.get("JADEVECTORDB_USER_ID", "")

if not USER_ID:
    print("Note: No JADEVECTORDB_USER_ID set. Attempting to create a temp user...\n")
    try:
        user = client.create_user(
            username="api_key_demo_user",
            password="DemoPass123!",
            roles=["developer"],
        )
        USER_ID = user.get("user_id", user.get("id", ""))
        print(f"Created temp user: {USER_ID}\n")
    except JadeVectorDBError as e:
        print(f"Could not create temp user: {e}")
        print("Set JADEVECTORDB_USER_ID to an existing user ID and retry.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# 1. Create an API key
# ---------------------------------------------------------------------------
# API keys are tied to a user and inherit that user's role-based permissions.
# You can also specify explicit permissions and an expiration period.

print("=== 1. Create API Key ===")
try:
    key_info = client.create_api_key(
        user_id=USER_ID,
        description="CI/CD pipeline — nightly embedding updates",
        permissions=["read", "write"],   # Scope the key's access
        validity_days=90,                 # Expires in 90 days (0 = never)
    )
    api_key_value = key_info.get("api_key", key_info.get("key", ""))
    key_id = key_info.get("key_id", key_info.get("id", ""))
    print(f"Key ID    : {key_id}")
    print(f"API Key   : {api_key_value[:20]}...  (truncated for security)")
    print(f"Expires   : {key_info.get('expires_at', key_info.get('expiresAt', 'never'))}")
    print()
except JadeVectorDBError as e:
    print(f"Create API key: {e}")
    api_key_value = None
    key_id = None

# ---------------------------------------------------------------------------
# 2. List API keys
# ---------------------------------------------------------------------------
# List all keys, or filter by user_id.

print("=== 2. List API Keys ===")
try:
    keys = client.list_api_keys(user_id=USER_ID)
    print(f"Keys for user {USER_ID}: {len(keys)}")
    for k in keys:
        kid = k.get("key_id", k.get("id", "?"))
        desc = k.get("description", "")
        status = k.get("status", "active")
        print(f"  - {kid:20s}  {status:10s}  {desc}")
except JadeVectorDBError as e:
    print(f"List API keys: {e}")

# List all keys (no user filter)
print("\n--- List All API Keys ---")
try:
    all_keys = client.list_api_keys()
    print(f"Total keys on server: {len(all_keys)}")
except JadeVectorDBError as e:
    print(f"List all: {e}")

# ---------------------------------------------------------------------------
# 3. Use the API key for authentication
# ---------------------------------------------------------------------------
# Create a new client instance using the API key instead of a user token.

print("\n=== 3. Authenticate with API Key ===")
if api_key_value:
    try:
        key_client = JadeVectorDB(base_url=SERVER_URL, api_key=api_key_value)
        status = key_client.get_status()
        print(f"Authenticated successfully with API key")
        print(f"  Server status: {status.get('status', '?')}")
    except JadeVectorDBError as e:
        print(f"API key auth: {e}")
else:
    print("Skipping (no API key created)")

# ---------------------------------------------------------------------------
# 4. Revoke an API key
# ---------------------------------------------------------------------------
# Immediately invalidates the key. Any requests using the revoked key will
# be rejected. This is permanent — you cannot un-revoke a key.

print("\n=== 4. Revoke API Key ===")
if key_id:
    try:
        revoked = client.revoke_api_key(key_id=key_id)
        print(f"Revoked key: {key_id}")
    except JadeVectorDBError as e:
        print(f"Revoke: {e}")

    # Verify the revoked key no longer works
    if api_key_value:
        try:
            revoked_client = JadeVectorDB(base_url=SERVER_URL, api_key=api_key_value)
            revoked_client.get_status()
            print("  WARNING: Revoked key still works (unexpected)")
        except JadeVectorDBError:
            print("  Confirmed: Revoked key is rejected")
else:
    print("Skipping (no key_id)")

# --- Cleanup temp user ---
try:
    client.delete_user(user_id=USER_ID)
except JadeVectorDBError:
    pass

print("\nAPI key management examples complete.")
