#!/usr/bin/env python3
"""
JadeVectorDB API Example — User Management
=============================================

JadeVectorDB has a built-in user system with role-based access control.
Users can be assigned roles like "admin", "developer", or "user" which
determine what operations they can perform.

This example demonstrates the full user lifecycle:
  1. Create a new user with a role
  2. List all users (with optional filters)
  3. Get detailed info for a specific user
  4. Update a user's role
  5. Deactivate a user (temporary disable)
  6. Activate a user (re-enable)
  7. Delete a user

APIs covered:
  - client.create_user(username, password, roles, email)
  - client.list_users(role, status)
  - client.get_user(user_id)
  - client.update_user(user_id, is_active, roles)
  - client.activate_user(user_id)
  - client.deactivate_user(user_id)
  - client.delete_user(user_id)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

# You need admin-level credentials to manage users
client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# ---------------------------------------------------------------------------
# 1. Create a new user
# ---------------------------------------------------------------------------
# New users need a unique username and a password. You can assign one or
# more roles and optionally provide an email address.
#
# Common roles:
#   "admin"     — Full access to all resources and admin endpoints
#   "developer" — Can create databases, store/search vectors, manage indexes
#   "user"      — Read-only access (search, retrieve)

print("=== 1. Create User ===")
try:
    user = client.create_user(
        username="alice_engineer",
        password="SecurePass123!",
        roles=["developer"],
        email="alice@example.com",
    )
    user_id = user.get("user_id", user.get("id", ""))
    print(f"Created user: {user_id}")
    print(json.dumps(user, indent=2))
except JadeVectorDBError as e:
    print(f"Create user: {e}")
    user_id = None

# ---------------------------------------------------------------------------
# 2. List all users
# ---------------------------------------------------------------------------
# Optionally filter by role or status (active/inactive).

print("\n=== 2. List Users ===")
try:
    users = client.list_users()
    print(f"Total users: {len(users)}")
    for u in users[:5]:  # Show first 5
        uid = u.get("user_id", u.get("id", "?"))
        uname = u.get("username", "?")
        uroles = u.get("roles", [])
        print(f"  {uid:20s}  {uname:20s}  roles={uroles}")
except JadeVectorDBError as e:
    print(f"List users: {e}")

# Filter by role
print("\n--- List Users (role=developer) ---")
try:
    devs = client.list_users(role="developer")
    print(f"  Found {len(devs)} developer(s)")
except JadeVectorDBError as e:
    print(f"Filter by role: {e}")

# ---------------------------------------------------------------------------
# 3. Get user details
# ---------------------------------------------------------------------------

print("\n=== 3. Get User Details ===")
if user_id:
    try:
        details = client.get_user(user_id=user_id)
        print(json.dumps(details, indent=2))
    except JadeVectorDBError as e:
        print(f"Get user: {e}")
else:
    print("Skipping (no user_id)")

# ---------------------------------------------------------------------------
# 4. Update user role
# ---------------------------------------------------------------------------
# Promote the user from "developer" to "admin".

print("\n=== 4. Update User Role ===")
if user_id:
    try:
        updated = client.update_user(
            user_id=user_id,
            roles=["admin"],        # Promote to admin
        )
        print(f"Updated roles: {updated.get('roles', '(see response)')}")
    except JadeVectorDBError as e:
        print(f"Update user: {e}")
else:
    print("Skipping (no user_id)")

# ---------------------------------------------------------------------------
# 5. Deactivate a user
# ---------------------------------------------------------------------------
# Deactivation is a soft disable — the user account still exists but
# cannot authenticate. Use this for temporary suspensions.

print("\n=== 5. Deactivate User ===")
if user_id:
    try:
        result = client.deactivate_user(user_id=user_id)
        print(f"Deactivated user: {user_id}")
        print(f"  is_active: {result.get('is_active', result.get('isActive', '?'))}")
    except JadeVectorDBError as e:
        print(f"Deactivate: {e}")
else:
    print("Skipping (no user_id)")

# ---------------------------------------------------------------------------
# 6. Activate a user
# ---------------------------------------------------------------------------
# Re-enable a previously deactivated user.

print("\n=== 6. Activate User ===")
if user_id:
    try:
        result = client.activate_user(user_id=user_id)
        print(f"Activated user: {user_id}")
        print(f"  is_active: {result.get('is_active', result.get('isActive', '?'))}")
    except JadeVectorDBError as e:
        print(f"Activate: {e}")
else:
    print("Skipping (no user_id)")

# ---------------------------------------------------------------------------
# 7. Delete a user
# ---------------------------------------------------------------------------
# Permanently removes the user account. This is irreversible.

print("\n=== 7. Delete User ===")
if user_id:
    try:
        deleted = client.delete_user(user_id=user_id)
        print(f"Deleted user {user_id}: {deleted}")
    except JadeVectorDBError as e:
        print(f"Delete user: {e}")
else:
    print("Skipping (no user_id)")

print("\nUser management examples complete.")
