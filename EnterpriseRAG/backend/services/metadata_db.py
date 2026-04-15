"""
SQLite-backed persistent metadata store for EnterpriseRAG documents and users.
Replaces the in-memory dict so state survives restarts.
"""

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetadataDB:
    """Thread-safe SQLite store for document metadata, query stats, and users."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # Each thread gets its own connection to avoid SQLite threading issues.
        self._local = threading.local()
        self._init_schema()

    # ── Connection management ─────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        if not getattr(self._local, "conn", None):
            conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _cursor(self):
        conn = self._conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()

    # ── Schema ────────────────────────────────────────────────

    def _init_schema(self):
        with self._cursor() as cur:
            # Documents table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id       TEXT PRIMARY KEY,
                    filename     TEXT NOT NULL,
                    category     TEXT NOT NULL DEFAULT 'general',
                    status       TEXT NOT NULL DEFAULT 'pending',
                    uploaded_at  TEXT NOT NULL,
                    processed_at TEXT,
                    chunk_count  INTEGER,
                    chunks_done  INTEGER NOT NULL DEFAULT 0,
                    file_path    TEXT,
                    error        TEXT
                )
            """)
            # Migrations from older schema
            # Rename device_type → category FIRST (before adding category column)
            try:
                cur.execute("ALTER TABLE documents RENAME COLUMN device_type TO category")
            except Exception:
                pass
            for col, definition in [
                ("file_path", "TEXT"),
                ("category", "TEXT NOT NULL DEFAULT 'general'"),
            ]:
                try:
                    cur.execute(f"ALTER TABLE documents ADD COLUMN {col} {definition}")
                except Exception:
                    pass  # Column already exists
            # Drop orphaned device_type if category already exists (partial migration cleanup)
            cols = {row[1] for row in cur.execute("PRAGMA table_info(documents)")}
            if "device_type" in cols and "category" in cols:
                cur.execute("ALTER TABLE documents DROP COLUMN device_type")

            # Stats table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    key   TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0
                )
            """)
            cur.execute(
                "INSERT OR IGNORE INTO stats (key, value) VALUES ('query_count', 0)"
            )

            # Queries table (category replaces device_type)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS queries (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    question           TEXT NOT NULL,
                    category           TEXT NOT NULL DEFAULT 'all',
                    mode               TEXT NOT NULL DEFAULT 'mock',
                    confidence         REAL DEFAULT 0,
                    processing_time_ms INTEGER DEFAULT 0,
                    sources_count      INTEGER DEFAULT 0,
                    success            INTEGER NOT NULL DEFAULT 1,
                    timestamp          TEXT NOT NULL
                )
            """)
            # Migration: rename device_type → category in queries table
            try:
                cur.execute("ALTER TABLE queries RENAME COLUMN device_type TO category")
            except Exception:
                pass

            # Users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id                   TEXT PRIMARY KEY,
                    username             TEXT NOT NULL UNIQUE,
                    email                TEXT NOT NULL UNIQUE,
                    hashed_password      TEXT NOT NULL,
                    role                 TEXT NOT NULL DEFAULT 'user',
                    must_change_password INTEGER NOT NULL DEFAULT 1,
                    created_by           TEXT,
                    created_at           TEXT NOT NULL,
                    last_login           TEXT,
                    is_active            INTEGER NOT NULL DEFAULT 1
                )
            """)

            # Any document stuck in 'processing' or 'pending' at startup means
            # the server was killed mid-flight — mark them failed so users know.
            cur.execute(
                "UPDATE documents SET status = 'failed', "
                "error = 'Interrupted by server restart' "
                "WHERE status IN ('processing', 'pending')"
            )

    # ── Document CRUD ─────────────────────────────────────────

    def insert(self, doc: Dict[str, Any]):
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents
                    (doc_id, filename, category, status, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc["id"], doc["filename"], doc.get("category", "general"),
                 doc["status"], doc["uploaded_at"]),
            )

    def update(self, doc_id: str, fields: Dict[str, Any]):
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [doc_id]
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE documents SET {set_clause} WHERE doc_id = ?", values
            )

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_all(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM documents ORDER BY uploaded_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [dict(row) for row in cur.fetchall()]

    def count_all(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]

    def delete(self, doc_id: str):
        with self._cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def exists(self, doc_id: str) -> bool:
        return self.get(doc_id) is not None

    # ── Stats ─────────────────────────────────────────────────

    def increment_query_count(self) -> int:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE stats SET value = value + 1 WHERE key = 'query_count'"
            )
            cur.execute("SELECT value FROM stats WHERE key = 'query_count'")
            return cur.fetchone()[0]

    def get_query_count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT value FROM stats WHERE key = 'query_count'")
            row = cur.fetchone()
            return row[0] if row else 0

    # ── Query analytics ───────────────────────────────────────

    def log_query(
        self,
        question: str,
        category: str,
        mode: str,
        confidence: float,
        processing_time_ms: int,
        sources_count: int,
        success: bool,
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries
                    (question, category, mode, confidence,
                     processing_time_ms, sources_count, success, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question[:500],  # cap length
                    category,
                    mode,
                    confidence,
                    processing_time_ms,
                    sources_count,
                    1 if success else 0,
                    datetime.utcnow().isoformat() + "Z",
                ),
            )

    def get_analytics(self, recent_limit: int = 20) -> Dict[str, Any]:
        with self._cursor() as cur:
            # Aggregate stats
            cur.execute("""
                SELECT
                    COUNT(*)                        AS total,
                    AVG(confidence)                 AS avg_confidence,
                    AVG(processing_time_ms)         AS avg_processing_time_ms,
                    SUM(CASE WHEN success=1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS success_rate
                FROM queries
            """)
            agg = dict(cur.fetchone())

            # Category breakdown
            cur.execute("""
                SELECT category, COUNT(*) AS cnt
                FROM queries
                GROUP BY category
                ORDER BY cnt DESC
            """)
            category_breakdown = {row["category"]: row["cnt"] for row in cur.fetchall()}

            # Recent queries
            cur.execute(
                "SELECT * FROM queries ORDER BY id DESC LIMIT ?",
                (recent_limit,),
            )
            recent = [dict(row) for row in cur.fetchall()]

        return {
            "total_queries": agg["total"] or 0,
            "avg_confidence": round(agg["avg_confidence"] or 0, 3),
            "avg_processing_time_ms": round(agg["avg_processing_time_ms"] or 0, 1),
            "success_rate": round(agg["success_rate"] or 0, 3),
            "category_breakdown": category_breakdown,
            "recent_queries": recent,
        }

    # ── User CRUD ─────────────────────────────────────────────

    def user_insert(self, user: Dict[str, Any]) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO users
                    (id, username, email, hashed_password, role,
                     must_change_password, created_by, created_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    user["id"],
                    user["username"],
                    user["email"],
                    user["hashed_password"],
                    user.get("role", "user"),
                    1,  # must_change_password = True
                    user.get("created_by"),
                    user.get("created_at", datetime.utcnow().isoformat() + "Z"),
                ),
            )

    def user_get_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def user_get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cur.fetchone()
            return dict(row) if row else None

    def user_get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM users WHERE email = ?", (email,))
            row = cur.fetchone()
            return dict(row) if row else None

    def user_list(self, offset: int = 0, limit: int = 100) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM users ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
            return [dict(row) for row in cur.fetchall()]

    def user_count(self) -> int:
        with self._cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            return cur.fetchone()[0]

    def user_update(self, user_id: str, fields: Dict[str, Any]) -> None:
        if not fields:
            return
        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [user_id]
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE users SET {set_clause} WHERE id = ?", values
            )

    def user_delete(self, user_id: str) -> None:
        """Soft-delete by deactivating the user."""
        with self._cursor() as cur:
            cur.execute(
                "UPDATE users SET is_active = 0 WHERE id = ?", (user_id,)
            )

    def user_exists(self) -> bool:
        """Return True if at least one user exists (for bootstrap check)."""
        return self.user_count() > 0
