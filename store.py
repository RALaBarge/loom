from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import sqlite_vec

SCHEMA = """
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
CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
    embedding FLOAT[768],
    +memory_id TEXT,
    +text TEXT
);
"""

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
        conn = self._raw_conn()
        try:
            conn.executescript(VEC_SCHEMA)
            conn.commit()
        finally:
            conn.close()

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
                    target = self._mem_row(conn, new_id)
                    if new_id == memory_id or target["status"] != "active":
                        raise MemoryConflict("new_id must be a different, active memory")
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
        """Memory counts by status."""
        with self._conn() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM memories GROUP BY status").fetchall()
        return {"memories": {r["status"]: r["n"] for r in rows}}


def _now() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
