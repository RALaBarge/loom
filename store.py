from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT DEFAULT '',
    timestamp TEXT NOT NULL,
    token_count INTEGER DEFAULT 0,
    latency_ms REAL DEFAULT NULL,
    reasoning_content TEXT DEFAULT '',
    finish_reason TEXT DEFAULT '',
    tool_calls TEXT DEFAULT '[]',
    provider TEXT DEFAULT '',
    response_headers TEXT DEFAULT '{}',
    context_id TEXT DEFAULT '',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE TABLE IF NOT EXISTS context_blobs (
    id TEXT PRIMARY KEY,
    conv_id TEXT NOT NULL,
    messages_json BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_context_blobs_conv ON context_blobs(conv_id);
CREATE TABLE IF NOT EXISTS memory_entries (
    id TEXT PRIMARY KEY,
    project TEXT DEFAULT '',
    summary TEXT NOT NULL,
    tags TEXT DEFAULT '[]',
    source TEXT DEFAULT '',
    session_id TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_entries_project ON memory_entries(project);
CREATE INDEX IF NOT EXISTS idx_memory_entries_created ON memory_entries(created_at);
"""

VEC_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
    embedding FLOAT[768],
    +role TEXT,
    +conv_id TEXT,
    +content TEXT,
    +timestamp TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory USING vec0(
    embedding FLOAT[768],
    +entry_id TEXT,
    +project TEXT,
    +summary TEXT,
    +created_at TEXT
);
"""

MSG_MIGRATIONS = [
    "ALTER TABLE messages ADD COLUMN reasoning_content TEXT DEFAULT ''",
    "ALTER TABLE messages ADD COLUMN finish_reason TEXT DEFAULT ''",
    "ALTER TABLE messages ADD COLUMN tool_calls TEXT DEFAULT '[]'",
    "ALTER TABLE messages ADD COLUMN provider TEXT DEFAULT ''",
    "ALTER TABLE messages ADD COLUMN response_headers TEXT DEFAULT '{}'",
    "ALTER TABLE messages ADD COLUMN context_id TEXT DEFAULT ''",
]


class Store:
    def __init__(self, db_path: str | Path):
        """Initialize the Store, creating database and vector tables as needed."""
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
        self._migrate()
        self._init_vec()

    def _migrate(self):
        """Run schema migrations to upgrade database from older versions."""
        with self._conn() as conn:
            for sql in MSG_MIGRATIONS:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass

    @contextmanager
    def _conn(self):
        """Context manager for SQLite transactions with auto-commit/rollback."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _raw_conn(self):
        """Create a connection with sqlite-vec extension for vector operations."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return conn

    def _init_vec(self):
        """Initialize vector embedding tables with sqlite-vec schema."""
        conn = self._raw_conn()
        try:
            conn.executescript(VEC_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def ensure_conversation(self, conv_id: str) -> bool:
        """Returns True if this is a newly-created conversation (first turn)."""
        with self._conn() as conn:
            cur = conn.execute(
                "INSERT OR IGNORE INTO conversations (id, created_at) VALUES (?, ?)",
                (conv_id, _now()),
            )
            return cur.rowcount > 0

    def insert_message(self, *, msg_id: str, conv_id: str, role: str, content: str,
                       model: str = "", timestamp: str = "", token_count: int = 0,
                       latency_ms: float | None = None, reasoning_content: str = "",
                       finish_reason: str = "", tool_calls: str = "[]",
                       provider: str = "", response_headers: str = "{}",
                       context_id: str = ""):
        """Store a single message with optional metadata (tokens, latency, reasoning)."""
        with self._conn() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO messages
                   (id, conversation_id, role, content, model, timestamp,
                    token_count, latency_ms, reasoning_content, finish_reason,
                    tool_calls, provider, response_headers, context_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (msg_id, conv_id, role, content, model, timestamp,
                 token_count, latency_ms, reasoning_content, finish_reason,
                 tool_calls, provider, response_headers, context_id),
            )

    def insert_context_blob(self, blob_id: str, conv_id: str,
                            messages_json: bytes, created_at: str):
        """Store the full conversation context for a specific point in time."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO context_blobs (id, conv_id, messages_json, created_at) VALUES (?, ?, ?, ?)",
                (blob_id, conv_id, messages_json, created_at),
            )

    def insert_embedding(self, *, embedding: list[float], role: str, conv_id: str,
                         content: str, timestamp: str):
        """Store a message embedding for semantic search."""
        conn = self._raw_conn()
        try:
            conn.execute(
                """INSERT INTO vec_messages(embedding, role, conv_id, content, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (json.dumps(embedding), role, conv_id, content, timestamp),
            )
            conn.commit()
        finally:
            conn.close()

    def search(self, embedding: list[float], limit: int = 5,
               role: str | None = None) -> list[dict[str, Any]]:
        """Semantic search over stored embeddings, returning closest matches."""
        conn = self._raw_conn()
        try:
            if role:
                rows = conn.execute(
                    """SELECT * FROM (
                           SELECT rowid, distance, role, conv_id, content, timestamp
                           FROM vec_messages
                           WHERE embedding MATCH ? ORDER BY distance LIMIT 200
                       ) WHERE role = ? LIMIT ?""",
                    (json.dumps(embedding), role, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT rowid, distance, role, conv_id, content, timestamp
                       FROM vec_messages
                       WHERE embedding MATCH ? ORDER BY distance LIMIT ?""",
                    (json.dumps(embedding), limit),
                ).fetchall()
            return [
                {
                    "distance": round(r["distance"], 4),
                    "role": r["role"],
                    "conversation_id": r["conv_id"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_conversations(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent conversations with message count and preview."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT c.id, c.created_at,
                          (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as msg_count,
                          (SELECT m.model FROM messages m WHERE m.conversation_id = c.id ORDER BY m.timestamp LIMIT 1) as model,
                          (SELECT m.content FROM messages m WHERE m.conversation_id = c.id AND m.role = 'user' ORDER BY m.timestamp LIMIT 1) as first_message
                   FROM conversations c
                   ORDER BY c.created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "created_at": r["created_at"],
                    "message_count": r["msg_count"],
                    "model": r["model"] or "",
                    "first_message": (r["first_message"] or "")[:200],
                }
                for r in rows
            ]

    def get_messages(self, conv_id: str) -> list[dict[str, Any]]:
        """Get all messages in a conversation ordered by timestamp."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, role, content, model, timestamp, token_count, latency_ms,
                          reasoning_content, finish_reason, tool_calls, provider,
                          response_headers, context_id
                   FROM messages WHERE conversation_id = ? ORDER BY timestamp""",
                (conv_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_surrounding(self, conv_id: str, timestamp: str, before: int = 3,
                        after: int = 3) -> list[dict[str, Any]]:
        """Get messages before and after a specific timestamp for context."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, role, content, model, timestamp
                   FROM messages WHERE conversation_id = ? AND timestamp < ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (conv_id, timestamp, before),
            ).fetchall()
            before_msgs = [dict(r) for r in reversed(rows)]
            rows = conn.execute(
                """SELECT id, role, content, model, timestamp
                   FROM messages WHERE conversation_id = ? AND timestamp > ?
                   ORDER BY timestamp LIMIT ?""",
                (conv_id, timestamp, after),
            ).fetchall()
            after_msgs = [dict(r) for r in rows]
            return before_msgs + after_msgs

    def delete_conversation(self, conv_id: str):
        """Delete a conversation and all associated messages and embeddings."""
        with self._conn() as conn:
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
            conn.execute("DELETE FROM context_blobs WHERE conv_id = ?", (conv_id,))
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn = self._raw_conn()
        try:
            conn.execute("DELETE FROM vec_messages WHERE conv_id = ?", (conv_id,))
            conn.commit()
        finally:
            conn.close()

    # ── Distilled memory entries ────────────────────────────────────

    def insert_memory_entry(self, *, entry_id: str, project: str, summary: str,
                            tags: list[str], source: str, session_id: str,
                            created_at: str):
        """Store a distilled memory entry with metadata."""
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO memory_entries
                   (id, project, summary, tags, source, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, project, summary, json.dumps(tags), source,
                 session_id, created_at),
            )

    def insert_memory_embedding(self, *, embedding: list[float], entry_id: str,
                                project: str, summary: str, created_at: str):
        """Store a memory entry embedding for semantic search."""
        conn = self._raw_conn()
        try:
            conn.execute(
                """INSERT INTO vec_memory(embedding, entry_id, project, summary, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (json.dumps(embedding), entry_id, project, summary, created_at),
            )
            conn.commit()
        finally:
            conn.close()

    def search_memory(self, embedding: list[float], limit: int = 5,
                      project: str | None = None) -> list[dict[str, Any]]:
        """Semantic search over distilled memory entries."""
        conn = self._raw_conn()
        try:
            if project:
                rows = conn.execute(
                    """SELECT * FROM (
                           SELECT rowid, distance, entry_id, project, summary, created_at
                           FROM vec_memory
                           WHERE embedding MATCH ? ORDER BY distance LIMIT 200
                       ) WHERE project = ? LIMIT ?""",
                    (json.dumps(embedding), project, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT rowid, distance, entry_id, project, summary, created_at
                       FROM vec_memory
                       WHERE embedding MATCH ? ORDER BY distance LIMIT ?""",
                    (json.dumps(embedding), limit),
                ).fetchall()
            return [
                {
                    "distance": round(r["distance"], 4),
                    "entry_id": r["entry_id"],
                    "project": r["project"],
                    "summary": r["summary"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        finally:
            conn.close()

    def get_memory_entries(self, limit: int = 50,
                           project: str | None = None) -> list[dict[str, Any]]:
        """Retrieve distilled memory entries, optionally filtered by project."""
        with self._conn() as conn:
            if project:
                rows = conn.execute(
                    """SELECT id, project, summary, tags, source, session_id, created_at
                       FROM memory_entries WHERE project = ?
                       ORDER BY created_at DESC LIMIT ?""",
                    (project, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT id, project, summary, tags, source, session_id, created_at
                       FROM memory_entries ORDER BY created_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_projects(self) -> list[dict[str, Any]]:
        """Get all memory projects with entry counts and last update time."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT project, COUNT(*) as n, MAX(created_at) as last
                   FROM memory_entries WHERE project != ''
                   GROUP BY project ORDER BY last DESC"""
            ).fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> dict[str, Any]:
        """Get storage statistics (conversation, message, and token counts)."""
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) as n FROM conversations").fetchone()
            nconv = row["n"]
            row = conn.execute("SELECT COUNT(*) as n FROM messages").fetchone()
            nmsg = row["n"]
            row = conn.execute("SELECT COUNT(*) as n FROM context_blobs").fetchone()
            nctx = row["n"]
            row = conn.execute(
                "SELECT COALESCE(SUM(token_count), 0) as t FROM messages"
            ).fetchone()
            ntokens = row["t"]
            row = conn.execute("SELECT COUNT(*) as n FROM memory_entries").fetchone()
            nmem = row["n"]
        conn = self._raw_conn()
        try:
            row = conn.execute("SELECT COUNT(*) as n FROM vec_messages").fetchone()
            nemb = row["n"] if row else 0
        finally:
            conn.close()
        return {
            "conversations": nconv,
            "messages": nmsg,
            "context_blobs": nctx,
            "total_tokens": ntokens,
            "embeddings": nemb,
            "memory_entries": nmem,
        }


def _now() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
