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
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('active','superseded','retracted')),
    superseded_by TEXT,
    source TEXT,
    created_at TEXT,
    updated_at TEXT,
    last_seen_at TEXT
);
CREATE TABLE IF NOT EXISTS memory_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id TEXT,
    version INTEGER,
    text TEXT,
    reason TEXT,
    created_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_memory_versions_mem ON memory_versions(memory_id, version);
"""

VEC_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_messages USING vec0(
    embedding FLOAT[768],
    +role TEXT,
    +conv_id TEXT,
    +content TEXT,
    +timestamp TEXT,
    +message_id TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memory USING vec0(
    embedding FLOAT[768],
    +entry_id TEXT,
    +project TEXT,
    +summary TEXT,
    +created_at TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding FLOAT[768],
    +memory_id TEXT,
    +text TEXT
);
"""

MSG_MIGRATIONS = [
    "ALTER TABLE messages ADD COLUMN reasoning_content TEXT DEFAULT ''",
    "ALTER TABLE messages ADD COLUMN finish_reason TEXT DEFAULT ''",
    "ALTER TABLE messages ADD COLUMN tool_calls TEXT DEFAULT '[]'",
    "ALTER TABLE messages ADD COLUMN provider TEXT DEFAULT ''",
    "ALTER TABLE messages ADD COLUMN response_headers TEXT DEFAULT '{}'",
    "ALTER TABLE messages ADD COLUMN context_id TEXT DEFAULT ''",
    "ALTER TABLE messages ADD COLUMN seq INTEGER",
]


class MemoryNotFound(Exception):
    """No memory with that id."""


class MemoryConflict(Exception):
    """The memory exists but is not in a state that allows the operation."""


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
            self._backfill_seq(conn)
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_conv_seq "
                "ON messages(conversation_id, seq)"
            )

    @staticmethod
    def _backfill_seq(conn):
        """Give every message without a seq its per-conversation turn number.

        Rows are numbered in (timestamp, rowid) order after the highest seq the
        conversation already has, so the pass is safe to repeat.
        """
        convs = [r[0] for r in conn.execute(
            "SELECT DISTINCT conversation_id FROM messages WHERE seq IS NULL")]
        for conv in convs:
            nxt = conn.execute(
                "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE conversation_id = ?",
                (conv,),
            ).fetchone()[0]
            ids = [r[0] for r in conn.execute(
                "SELECT rowid FROM messages WHERE conversation_id = ? AND seq IS NULL "
                "ORDER BY timestamp, rowid", (conv,))]
            for rid in ids:
                conn.execute("UPDATE messages SET seq = ? WHERE rowid = ?", (nxt, rid))
                nxt += 1

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
        """Initialize vector embedding tables with sqlite-vec schema.

        ``CREATE VIRTUAL TABLE IF NOT EXISTS`` cannot add a column to an existing
        table, and SQLite refuses ``ALTER TABLE`` on a virtual table, so a database
        created before ``vec_messages.message_id`` existed would keep the old shape and
        every search would fail on the missing column. Rebuild the table instead: copy
        the rows out, drop it (which also removes sqlite-vec's shadow tables), recreate
        it with the extra column under the same name, and put the rows back with their
        original rowids so vectors and distances are unchanged. Existing rows keep
        ``message_id`` NULL and fall back to the ambiguous lookup in ``search``.
        """
        conn = self._raw_conn()
        try:
            conn.executescript(VEC_SCHEMA)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(vec_messages)")}
            if "message_id" not in cols:
                saved = conn.execute(
                    "SELECT rowid, embedding, role, conv_id, content, timestamp FROM vec_messages"
                ).fetchall()
                conn.execute("DROP TABLE vec_messages")
                conn.execute(
                    """CREATE VIRTUAL TABLE vec_messages USING vec0(
                           embedding FLOAT[768],
                           +role TEXT,
                           +conv_id TEXT,
                           +content TEXT,
                           +timestamp TEXT,
                           +message_id TEXT
                       )"""
                )
                for rowid, embedding, role, conv_id, content, timestamp in saved:
                    conn.execute(
                        """INSERT INTO vec_messages(rowid, embedding, role, conv_id,
                                                    content, timestamp, message_id)
                           VALUES (?, ?, ?, ?, ?, ?, NULL)""",
                        (rowid, embedding, role, conv_id, content, timestamp),
                    )
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
                       context_id: str = "") -> bool:
        """Append a message with optional metadata (tokens, latency, reasoning).

        Append-only: a message id that already exists is left untouched. Returns
        True if a row was inserted. The per-conversation seq is assigned inside
        the same write transaction, so concurrent writers never share a number.
        """
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.execute(
                """INSERT OR IGNORE INTO messages
                   (id, conversation_id, role, content, model, timestamp,
                    token_count, latency_ms, reasoning_content, finish_reason,
                    tool_calls, provider, response_headers, context_id, seq)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           (SELECT COALESCE(MAX(seq), -1) + 1 FROM messages
                            WHERE conversation_id = ?))""",
                (msg_id, conv_id, role, content, model, timestamp,
                 token_count, latency_ms, reasoning_content, finish_reason,
                 tool_calls, provider, response_headers, context_id, conv_id),
            )
            return cur.rowcount > 0

    def insert_context_blob(self, blob_id: str, conv_id: str,
                            messages_json: bytes, created_at: str):
        """Store the full conversation context for a specific point in time."""
        with self._conn() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO context_blobs (id, conv_id, messages_json, created_at) VALUES (?, ?, ?, ?)",
                (blob_id, conv_id, messages_json, created_at),
            )

    def insert_embedding(self, *, embedding: list[float], role: str, conv_id: str,
                         content: str, timestamp: str, message_id: str | None = None):
        """Store a message embedding for semantic search.

        Pass ``message_id`` (the ``messages.id`` this vector came from) so search
        results can report the exact row's ``seq``. Without it the seq lookup has to
        guess by (conversation, role, timestamp, content), which is ambiguous when two
        messages share all four — see ``search``.
        """
        conn = self._raw_conn()
        try:
            conn.execute(
                """INSERT INTO vec_messages(embedding, role, conv_id, content, timestamp, message_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (json.dumps(embedding), role, conv_id, content, timestamp, message_id),
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
                           SELECT rowid, distance, role, conv_id, content, timestamp, message_id
                           FROM vec_messages
                           WHERE embedding MATCH ? ORDER BY distance LIMIT 200
                       ) WHERE role = ? LIMIT ?""",
                    (json.dumps(embedding), role, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT rowid, distance, role, conv_id, content, timestamp, message_id
                       FROM vec_messages
                       WHERE embedding MATCH ? ORDER BY distance LIMIT ?""",
                    (json.dumps(embedding), limit),
                ).fetchall()
            out = []
            for r in rows:
                seq = None
                if r["message_id"]:
                    # Exact: the vector row names the message row it came from.
                    m = conn.execute(
                        "SELECT seq FROM messages WHERE id = ?", (r["message_id"],)
                    ).fetchone()
                    if m:
                        seq = m["seq"]
                if seq is None:
                    # Legacy row written before message_id existed, or the message is gone.
                    # Best effort, and ambiguous when two messages share all four fields.
                    m = conn.execute(
                        """SELECT seq FROM messages WHERE conversation_id = ? AND role = ?
                           AND timestamp = ? AND content = ? ORDER BY seq LIMIT 1""",
                        (r["conv_id"], r["role"], r["timestamp"], r["content"]),
                    ).fetchone()
                    if m:
                        seq = m["seq"]
                out.append({
                    "distance": round(r["distance"], 4),
                    "role": r["role"],
                    "conversation_id": r["conv_id"],
                    "content": r["content"],
                    "timestamp": r["timestamp"],
                    "message_id": r["message_id"],
                    "seq": seq,
                })
            return out
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
                """SELECT id, seq, role, content, model, timestamp, token_count, latency_ms,
                          reasoning_content, finish_reason, tool_calls, provider,
                          response_headers, context_id
                   FROM messages WHERE conversation_id = ? ORDER BY seq, timestamp""",
                (conv_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def conversation_exists(self, conv_id: str) -> bool:
        with self._conn() as conn:
            return conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone() is not None

    def get_message_range(self, conv_id: str, from_seq: int = 0,
                          to_seq: int | None = None,
                          limit: int = 500) -> list[dict[str, Any]]:
        """Messages with from_seq <= seq <= to_seq (inclusive), ordered by seq."""
        hi = to_seq if to_seq is not None else 2**62
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, seq, role, content, timestamp FROM messages
                   WHERE conversation_id = ? AND seq >= ? AND seq <= ?
                   ORDER BY seq LIMIT ?""",
                (conv_id, from_seq, hi, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_surrounding(self, conv_id: str, timestamp: str, before: int = 3,
                        after: int = 3) -> list[dict[str, Any]]:
        """Get messages before and after a specific timestamp for context."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT id, seq, role, content, model, timestamp
                   FROM messages WHERE conversation_id = ? AND timestamp < ?
                   ORDER BY timestamp DESC, seq DESC LIMIT ?""",
                (conv_id, timestamp, before),
            ).fetchall()
            before_msgs = [dict(r) for r in reversed(rows)]
            rows = conn.execute(
                """SELECT id, seq, role, content, model, timestamp
                   FROM messages WHERE conversation_id = ? AND timestamp > ?
                   ORDER BY timestamp, seq LIMIT ?""",
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

    # ── Curated memories with version history ───────────────────────
    # Only active memories have a vec row; relational rows and versions are
    # kept forever (retract is a soft delete). Embeddings are computed by the
    # caller before these methods run, so a failed embedding writes nothing.

    @staticmethod
    def _mem_dump(source: Any) -> str | None:
        if source is None:
            return None
        return source if isinstance(source, str) else json.dumps(source)

    @staticmethod
    def _vec_drop(conn, memory_id: str):
        for r in conn.execute(
                "SELECT rowid FROM vec_memories WHERE memory_id = ?", (memory_id,)).fetchall():
            conn.execute("DELETE FROM vec_memories WHERE rowid = ?", (r["rowid"],))

    @staticmethod
    def _vec_put(conn, memory_id: str, text: str, embedding: list[float]):
        conn.execute(
            "INSERT INTO vec_memories(embedding, memory_id, text) VALUES (?, ?, ?)",
            (json.dumps(embedding), memory_id, text),
        )

    @staticmethod
    def _mem_row(conn, memory_id: str):
        row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if row is None:
            raise MemoryNotFound(memory_id)
        return row

    @staticmethod
    def _mem_insert(conn, memory_id: str, text: str, source: Any, now: str):
        if conn.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone():
            raise MemoryConflict(f"memory {memory_id} already exists")
        conn.execute(
            """INSERT INTO memories (id, text, status, superseded_by, source,
                                     created_at, updated_at, last_seen_at)
               VALUES (?, ?, 'active', NULL, ?, ?, ?, ?)""",
            (memory_id, text, Store._mem_dump(source), now, now, now),
        )
        conn.execute(
            "INSERT INTO memory_versions (memory_id, version, text, reason, created_at) "
            "VALUES (?, 1, ?, 'created', ?)", (memory_id, text, now),
        )

    def create_memory(self, *, memory_id: str, text: str, source: Any,
                      embedding: list[float]) -> int:
        conn = self._raw_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._mem_insert(conn, memory_id, text, source, _now())
                self._vec_put(conn, memory_id, text, embedding)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return 1
        finally:
            conn.close()

    def revise_memory(self, memory_id: str, *, text: str, reason: str,
                      embedding: list[float]) -> int:
        conn = self._raw_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._mem_row(conn, memory_id)
                if row["status"] != "active":
                    raise MemoryConflict(f"memory {memory_id} is {row['status']}")
                now = _now()
                ver = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM memory_versions WHERE memory_id = ?",
                    (memory_id,)).fetchone()[0]
                conn.execute(
                    "INSERT INTO memory_versions (memory_id, version, text, reason, created_at) "
                    "VALUES (?, ?, ?, ?, ?)", (memory_id, ver, text, reason, now))
                conn.execute("UPDATE memories SET text = ?, updated_at = ? WHERE id = ?",
                             (text, now, memory_id))
                self._vec_drop(conn, memory_id)
                self._vec_put(conn, memory_id, text, embedding)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return ver
        finally:
            conn.close()

    def supersede_memory(self, memory_id: str, *, reason: str, new_id: str | None = None,
                         new_text: str | None = None, new_source: Any = None,
                         new_embedding: list[float] | None = None) -> str | None:
        """Mark a memory superseded; optionally create its replacement in the same transaction."""
        conn = self._raw_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._mem_row(conn, memory_id)
                if row["status"] != "active":
                    raise MemoryConflict(f"memory {memory_id} is {row['status']}")
                now = _now()
                if new_text is not None:
                    self._mem_insert(conn, new_id, new_text, new_source, now)
                    self._vec_put(conn, new_id, new_text, new_embedding)
                elif new_id is not None:
                    self._mem_row(conn, new_id)
                conn.execute(
                    "UPDATE memories SET status = 'superseded', superseded_by = ?, updated_at = ? "
                    "WHERE id = ?", (new_id, now, memory_id))
                ver = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM memory_versions WHERE memory_id = ?",
                    (memory_id,)).fetchone()[0]
                conn.execute(
                    "INSERT INTO memory_versions (memory_id, version, text, reason, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (memory_id, ver, row["text"], f"superseded: {reason}", now))
                self._vec_drop(conn, memory_id)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            return new_id
        finally:
            conn.close()

    def retract_memory(self, memory_id: str, *, reason: str):
        conn = self._raw_conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._mem_row(conn, memory_id)
                if row["status"] != "active":
                    raise MemoryConflict(f"memory {memory_id} is {row['status']}")
                now = _now()
                conn.execute("UPDATE memories SET status = 'retracted', updated_at = ? WHERE id = ?",
                             (now, memory_id))
                ver = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) + 1 FROM memory_versions WHERE memory_id = ?",
                    (memory_id,)).fetchone()[0]
                conn.execute(
                    "INSERT INTO memory_versions (memory_id, version, text, reason, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (memory_id, ver, row["text"], f"retracted: {reason}", now))
                self._vec_drop(conn, memory_id)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        finally:
            conn.close()

    def touch_memory(self, memory_id: str):
        with self._conn() as conn:
            cur = conn.execute("UPDATE memories SET last_seen_at = ? WHERE id = ?",
                               (_now(), memory_id))
            if cur.rowcount == 0:
                raise MemoryNotFound(memory_id)

    def _mem_public(self, conn, row) -> dict[str, Any]:
        out = dict(row)
        src = out.get("source")
        if src:
            try:
                out["source"] = json.loads(src)
            except ValueError:
                pass
        out["version"] = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM memory_versions WHERE memory_id = ?",
            (row["id"],)).fetchone()[0]
        return out

    def get_memory(self, memory_id: str) -> dict[str, Any]:
        with self._conn() as conn:
            row = self._mem_row(conn, memory_id)
            out = self._mem_public(conn, row)
            out["versions"] = [dict(r) for r in conn.execute(
                "SELECT version, text, reason, created_at FROM memory_versions "
                "WHERE memory_id = ? ORDER BY version", (memory_id,))]
            return out

    def list_memories(self, *, status: str | None = "active", limit: int = 50,
                      offset: int = 0) -> list[dict[str, Any]]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE status = ? ORDER BY created_at DESC, id "
                    "LIMIT ? OFFSET ?", (status, limit, offset)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories ORDER BY created_at DESC, id LIMIT ? OFFSET ?",
                    (limit, offset)).fetchall()
            return [self._mem_public(conn, r) for r in rows]

    def search_memories(self, embedding: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """Nearest active memories (only active ones have a vec row)."""
        conn = self._raw_conn()
        try:
            rows = conn.execute(
                """SELECT rowid, distance, memory_id, text FROM vec_memories
                   WHERE embedding MATCH ? ORDER BY distance LIMIT ?""",
                (json.dumps(embedding), limit),
            ).fetchall()
            out = []
            for r in rows:
                m = conn.execute("SELECT updated_at FROM memories WHERE id = ?",
                                 (r["memory_id"],)).fetchone()
                ver = conn.execute(
                    "SELECT COALESCE(MAX(version), 0) FROM memory_versions WHERE memory_id = ?",
                    (r["memory_id"],)).fetchone()[0]
                out.append({"id": r["memory_id"], "text": r["text"],
                            "distance": round(r["distance"], 4), "version": ver,
                            "updated_at": m["updated_at"] if m else None})
            return out
        finally:
            conn.close()

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
