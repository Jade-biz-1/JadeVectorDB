#!/usr/bin/env python3
"""
JadeVectorDB API Example — Security & Audit
===============================================

JadeVectorDB records every significant action in an audit log, providing
a complete trail for compliance, debugging, and security monitoring.

This example demonstrates:
  1. View recent audit log entries
  2. Filter audit log by user or event type
  3. View active sessions for a user
  4. Get audit statistics summary

APIs covered:
  - client.get_audit_log(user_id, event_type, limit)
  - client.get_sessions(user_id)
  - client.get_audit_stats()
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
# 1. View recent audit log entries
# ---------------------------------------------------------------------------
# The audit log captures events like user logins, database creation,
# vector operations, permission changes, and API key management.

print("=== 1. Recent Audit Log ===")
try:
    audit = client.get_audit_log(limit=10)

    # Response may be a dict with "events"/"entries" key or a list
    entries = audit if isinstance(audit, list) else audit.get("events", audit.get("entries", []))
    print(f"Showing {len(entries)} most recent events:")
    for entry in entries[:10]:
        event_type = entry.get("event_type", entry.get("eventType", "?"))
        user = entry.get("user_id", entry.get("userId", "system"))
        timestamp = entry.get("timestamp", entry.get("created_at", "?"))
        print(f"  [{timestamp}] {event_type:25s}  user={user}")
except JadeVectorDBError as e:
    print(f"Audit log: {e}")

# ---------------------------------------------------------------------------
# 2. Filter audit log by event type
# ---------------------------------------------------------------------------
# Common event types: "login", "database_create", "vector_store",
# "search", "user_create", "api_key_create", etc.

print("\n=== 2. Filter by Event Type ===")
try:
    login_events = client.get_audit_log(event_type="login", limit=5)
    entries = login_events if isinstance(login_events, list) else login_events.get("events", login_events.get("entries", []))
    print(f"Recent login events: {len(entries)}")
    for entry in entries:
        user = entry.get("user_id", entry.get("userId", "?"))
        ts = entry.get("timestamp", "?")
        print(f"  [{ts}] Login by {user}")
except JadeVectorDBError as e:
    print(f"Filter by type: {e}")

# ---------------------------------------------------------------------------
# 3. Filter audit log by user
# ---------------------------------------------------------------------------

print("\n=== 3. Filter by User ID ===")
USER_ID = os.environ.get("JADEVECTORDB_USER_ID", "")
if USER_ID:
    try:
        user_events = client.get_audit_log(user_id=USER_ID, limit=5)
        entries = user_events if isinstance(user_events, list) else user_events.get("events", user_events.get("entries", []))
        print(f"Events for user {USER_ID}: {len(entries)}")
        for entry in entries:
            event_type = entry.get("event_type", entry.get("eventType", "?"))
            print(f"  {event_type}")
    except JadeVectorDBError as e:
        print(f"Filter by user: {e}")
else:
    print("Set JADEVECTORDB_USER_ID to filter by a specific user")

# ---------------------------------------------------------------------------
# 4. View active sessions
# ---------------------------------------------------------------------------
# Shows currently active authentication sessions for a user.
# Useful for security monitoring and detecting unauthorized access.

print("\n=== 4. Active Sessions ===")
if USER_ID:
    try:
        sessions = client.get_sessions(user_id=USER_ID)
        session_list = sessions if isinstance(sessions, list) else sessions.get("sessions", [])
        print(f"Active sessions for {USER_ID}: {len(session_list)}")
        for s in session_list:
            sid = s.get("session_id", s.get("id", "?"))
            created = s.get("created_at", s.get("createdAt", "?"))
            print(f"  Session {sid} — created {created}")
    except JadeVectorDBError as e:
        print(f"Sessions: {e}")
else:
    print("Set JADEVECTORDB_USER_ID to view sessions")

# ---------------------------------------------------------------------------
# 5. Audit statistics summary
# ---------------------------------------------------------------------------
# High-level metrics: total events, events by type, events per day, etc.
# Useful for admin dashboards.

print("\n=== 5. Audit Statistics ===")
try:
    stats = client.get_audit_stats()
    print(json.dumps(stats, indent=2))
except JadeVectorDBError as e:
    print(f"Audit stats: {e}")

print("\nSecurity & audit examples complete.")
