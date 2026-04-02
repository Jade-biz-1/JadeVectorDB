"""
SQLite-backed persistent metadata store for EnterpriseRAG documents.
Replaces the in-memory dict so state survives restarts.
"""

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


class MetadataDB:
    """Thread-safe SQLite store for document metadata and query stats."""

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
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id       TEXT PRIMARY KEY,
                    filename     TEXT NOT NULL,
                    device_type  TEXT NOT NULL,
                    status       TEXT NOT NULL DEFAULT 'pending',
                    uploaded_at  TEXT NOT NULL,
                    processed_at TEXT,
                    chunk_count  INTEGER,
                    chunks_done  INTEGER NOT NULL DEFAULT 0,
                    file_path    TEXT,
                    error        TEXT
                )
            """)
            # Migration: add file_path if upgrading from an older schema
            try:
                cur.execute("ALTER TABLE documents ADD COLUMN file_path TEXT")
            except Exception:
                pass  # Column already exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    key   TEXT PRIMARY KEY,
                    value INTEGER NOT NULL DEFAULT 0
                )
            """)
            cur.execute(
                "INSERT OR IGNORE INTO stats (key, value) VALUES ('query_count', 0)"
            )
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
                    (doc_id, filename, device_type, status, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (doc["id"], doc["filename"], doc["device_type"],
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
