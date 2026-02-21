#!/usr/bin/env python3
"""
JadeVectorDB API Example — Password Management
==================================================

JadeVectorDB provides two password management flows:
  1. User-initiated password change (requires current password)
  2. Admin-initiated password reset (requires admin privileges)

APIs covered:
  - client.change_password(user_id, old_password, new_password)
  - client.admin_reset_password(user_id, new_password)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# --- Setup: create a temporary user for this demo ---
print("Setting up demo user...")
user_id = None
try:
    user = client.create_user(
        username="password_demo_user",
        password="OldPassword123!",
        roles=["user"],
    )
    user_id = user.get("user_id", user.get("id", ""))
    print(f"Created user: {user_id}\n")
except JadeVectorDBError as e:
    print(f"Setup: {e}")
    print("Some examples may not work without a valid user_id.\n")

# ---------------------------------------------------------------------------
# 1. User-initiated password change
# ---------------------------------------------------------------------------
# The user must know their current password. This is the standard flow
# for "Change Password" in user profile settings.
#
# Password requirements typically include:
#   - Minimum 8 characters
#   - At least one uppercase letter
#   - At least one number
#   - At least one special character

print("=== 1. Change Password (User-Initiated) ===")
if user_id:
    try:
        success = client.change_password(
            user_id=user_id,
            old_password="OldPassword123!",
            new_password="NewSecurePass456!",
        )
        print(f"Password changed successfully: {success}")
    except JadeVectorDBError as e:
        print(f"Change password: {e}")
else:
    print("Skipping (no user_id)")

# ---------------------------------------------------------------------------
# 2. Admin-initiated password reset
# ---------------------------------------------------------------------------
# An admin can reset any user's password without knowing the current one.
# This is used for:
#   - Users who forgot their password
#   - Security incidents requiring forced password rotation
#   - Initial account setup by IT administrators
#
# Requires admin-level API key or token.

print("\n=== 2. Admin Reset Password ===")
if user_id:
    try:
        result = client.admin_reset_password(
            user_id=user_id,
            new_password="AdminReset789!",
        )
        print(f"Admin password reset: {json.dumps(result, indent=2)}")
    except JadeVectorDBError as e:
        print(f"Admin reset: {e}")
        print("  (This may require admin-level credentials)")
else:
    print("Skipping (no user_id)")

# --- Cleanup ---
if user_id:
    try:
        client.delete_user(user_id=user_id)
        print(f"\nCleaned up user: {user_id}")
    except JadeVectorDBError:
        pass

print("\nPassword management examples complete.")
