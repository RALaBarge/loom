"""The event ledger and the memory review queue.

Events are facts: one row per entry a harness wrote (for Pi, one line of its session file), append-only, keyed by
(source, session, entry_id) so a retried batch changes nothing. The entry itself, already redacted by the sender, is a
content-addressed blob. Everything else (the embedding index, turn views, memories) is derived and can be rebuilt.

Trust is per event. `untrusted` marks content that came from outside the conversation (tool results): it is stored for
audit and replay but never embedded and never offered for recall, so its `text` is always empty.

Proposals are candidate memories waiting for a human. Only `review` decides them; `pop` turns accepted ones into
memories. The routes and who may call them are in main.py.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS blobs (
    hash TEXT PRIMARY KEY,
    data BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    session TEXT NOT NULL,
    entry_id TEXT NOT NULL,
    parent TEXT,
    type TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    ts TEXT NOT NULL DEFAULT '',
    received_at TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    trust TEXT NOT NULL CHECK (trust IN ('trusted', 'untrusted')),
    redactor TEXT NOT NULL,
    blob TEXT REFERENCES blobs(hash),
    text TEXT NOT NULL DEFAULT '',
    embedded_with TEXT,
    forgotten_at TEXT,
    UNIQUE (source, session, entry_id)
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(source, session, seq);
CREATE INDEX IF NOT EXISTS idx_events_blob ON events(blob);
CREATE TABLE IF NOT EXISTS proposals (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'accepted', 'rejected', 'applied')),
    by TEXT NOT NULL CHECK (by IN ('ryan', 'agent', 'proposer')),
    events TEXT NOT NULL DEFAULT '[]',
    note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT,
    applied_at TEXT,
    memory_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_proposals_status ON proposals(status, created_at);
"""
VEC_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_events USING vec0(
    embedding FLOAT[768],
    +seq INTEGER,
    +model TEXT
);
"""

ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
WORD_RE = re.compile(r"^[A-Za-z0-9_.-]{0,64}$")
MAX_RAW = 4 * 1024 * 1024     # one entry; the sender cuts anything larger
MAX_TEXT = 64 * 1024          # the embeddable part of one entry
MAX_BATCH = 500
MAX_PROPOSAL = 8000


class Invalid(ValueError):
    """A request field is missing or out of bounds; the message says which."""


class NotFound(LookupError):
    pass


class Conflict(Exception):
    pass


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _word(e: dict, key: str, required: bool = False, rx=WORD_RE) -> str:
    v = e.get(key)
    if v in (None, "") and not required:
        return ""
    if not isinstance(v, str) or not v or not rx.match(v):
        raise Invalid(f"{key} must match {rx.pattern}")
    return v


def check_event(e: Any) -> dict:
    """Validate one incoming event; returns the row to store (raw stays a str until written)."""
    if not isinstance(e, dict):
        raise Invalid("each event must be an object")
    out = {
        "source": _word(e, "source", True),
        "session": _word(e, "session", True, ID_RE),
        "entry_id": _word(e, "id", True, ID_RE),
        "parent": _word(e, "parent", False, ID_RE) or None,
        "type": _word(e, "type", True),
        "role": _word(e, "role"),
        "model": e.get("model") or "",
        "provider": _word(e, "provider"),
        "redactor": _word(e, "redactor", True),
    }
    for key in ("ts", "model"):
        v = e.get(key) or ""
        if not isinstance(v, str) or len(v) > 128:
            raise Invalid(f"{key} must be a string of at most 128 characters")
        out[key] = v
    trust = e.get("trust")
    if trust not in ("trusted", "untrusted"):
        raise Invalid("trust must be 'trusted' or 'untrusted'")
    out["trust"] = trust
    raw = e.get("raw")
    if not isinstance(raw, str) or not raw or len(raw.encode()) > MAX_RAW:
        raise Invalid(f"raw must be a non-empty string of at most {MAX_RAW} bytes")
    out["raw"] = raw
    text = e.get("text") or ""
    if not isinstance(text, str) or len(text) > MAX_TEXT:
        raise Invalid(f"text must be a string of at most {MAX_TEXT} characters")
    out["text"] = text.strip() if trust == "trusted" else ""   # untrusted content is never embedded
    return out


def _put_blob(conn, raw: str) -> str:
    data = raw.encode()
    h = hashlib.sha256(data).hexdigest()
    conn.execute("INSERT OR IGNORE INTO blobs (hash, data, created_at) VALUES (?, ?, ?)", (h, data, now()))
    return h


def _drop_blob_if_unused(conn, h: str | None):
    if h and not conn.execute("SELECT 1 FROM events WHERE blob = ? LIMIT 1", (h,)).fetchone():
        conn.execute("DELETE FROM blobs WHERE hash = ?", (h,))


def _drop_vec(conn, seq: int):
    for r in conn.execute("SELECT rowid FROM vec_events WHERE seq = ?", (seq,)).fetchall():
        conn.execute("DELETE FROM vec_events WHERE rowid = ?", (r[0],))


class Ledger:
    """Event and proposal storage on Loom's database. `store` supplies the connections (store.Store)."""

    def __init__(self, store):
        self.store = store
        with store._conn() as conn:
            conn.executescript(SCHEMA)
        conn = store._raw_conn()
        try:
            conn.executescript(VEC_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _tx(self):
        conn = self.store._raw_conn()
        conn.execute("BEGIN IMMEDIATE")
        return conn

    # ---- events ----

    def ingest(self, events: list[dict]) -> list[dict]:
        """Store a batch in one transaction. Returns one ack per event, in order, only after the commit."""
        rows = [check_event(e) for e in events]
        acks = []
        conn = self._tx()
        try:
            t = now()
            for r in rows:
                old = conn.execute(
                    "SELECT seq, blob FROM events WHERE source = ? AND session = ? AND entry_id = ?",
                    (r["source"], r["session"], r["entry_id"])).fetchone()
                if old:
                    acks.append({"session": r["session"], "id": r["entry_id"], "seq": old[0], "hash": old[1],
                                 "new": False})
                    continue
                h = _put_blob(conn, r["raw"])
                cur = conn.execute(
                    """INSERT INTO events (source, session, entry_id, parent, type, role, ts, received_at, model,
                                           provider, trust, redactor, blob, text)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (r["source"], r["session"], r["entry_id"], r["parent"], r["type"], r["role"], r["ts"], t,
                     r["model"], r["provider"], r["trust"], r["redactor"], h, r["text"]))
                acks.append({"session": r["session"], "id": r["entry_id"], "seq": cur.lastrowid, "hash": h,
                             "new": True})
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return acks

    @staticmethod
    def _public(row, raw: bytes | None = None) -> dict:
        d = {k: row[k] for k in ("seq", "source", "session", "entry_id", "parent", "type", "role", "ts",
                                 "received_at", "model", "provider", "trust", "redactor", "blob", "text",
                                 "forgotten_at")}
        if raw is not None:
            d["raw"] = raw.decode()
        return d

    def events(self, *, session: str | None = None, after: int = 0, limit: int = 200,
               raw: bool = False) -> list[dict]:
        with self.store._conn() as conn:
            q, args = "SELECT * FROM events WHERE seq > ?", [after]
            if session:
                q += " AND session = ?"
                args.append(session)
            rows = conn.execute(q + " ORDER BY seq LIMIT ?", (*args, limit)).fetchall()
            out = []
            for r in rows:
                data = None
                if raw and r["blob"]:
                    b = conn.execute("SELECT data FROM blobs WHERE hash = ?", (r["blob"],)).fetchone()
                    data = b[0] if b else None
                out.append(self._public(r, data))
            return out

    def sessions(self, limit: int = 100) -> list[dict]:
        with self.store._conn() as conn:
            return [dict(r) for r in conn.execute(
                """SELECT source, session, COUNT(*) AS events, MIN(ts) AS first_ts, MAX(ts) AS last_ts,
                          MAX(seq) AS last_seq, MAX(forgotten_at IS NOT NULL) AS forgotten
                   FROM events GROUP BY source, session ORDER BY last_seq DESC LIMIT ?""", (limit,))]

    def pending_embeddings(self, model: str, limit: int = 16) -> list[tuple[int, str]]:
        with self.store._conn() as conn:
            return [(r[0], r[1]) for r in conn.execute(
                """SELECT seq, text FROM events WHERE trust = 'trusted' AND text != '' AND forgotten_at IS NULL
                   AND (embedded_with IS NULL OR embedded_with != ?) ORDER BY seq LIMIT ?""", (model, limit))]

    def put_embedding(self, seq: int, model: str, embedding: list[float]):
        conn = self._tx()
        try:
            row = conn.execute("SELECT forgotten_at, trust FROM events WHERE seq = ?", (seq,)).fetchone()
            if row is None or row[0] is not None or row[1] != "trusted":
                conn.rollback()
                return
            _drop_vec(conn, seq)
            conn.execute("INSERT INTO vec_events (embedding, seq, model) VALUES (?, ?, ?)",
                         (json.dumps(embedding), seq, model))
            conn.execute("UPDATE events SET embedded_with = ? WHERE seq = ?", (model, seq))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def search(self, embedding: list[float], model: str, limit: int = 5) -> list[dict]:
        """Nearest trusted, unforgotten events embedded with `model` (vectors from another model don't compare)."""
        conn = self.store._raw_conn()
        try:
            hits = conn.execute(
                """SELECT seq, distance, model FROM vec_events WHERE embedding MATCH ? AND k = ?
                   ORDER BY distance""", (json.dumps(embedding), limit * 4)).fetchall()
            out = []
            for h in hits:
                if h["model"] != model:
                    continue
                r = conn.execute("SELECT * FROM events WHERE seq = ? AND forgotten_at IS NULL AND trust = 'trusted'",
                                 (h["seq"],)).fetchone()
                if r:
                    out.append(dict(self._public(r), distance=round(h["distance"], 4)))
                if len(out) == limit:
                    break
            return out
        finally:
            conn.close()

    def forget(self, session: str) -> int:
        """Tombstone a session: rows stay, text and vectors go, nothing recalls it. Blobs go at purge."""
        conn = self._tx()
        try:
            seqs = [r[0] for r in conn.execute(
                "SELECT seq FROM events WHERE session = ? AND forgotten_at IS NULL", (session,))]
            for s in seqs:
                _drop_vec(conn, s)
            conn.execute("UPDATE events SET forgotten_at = ?, text = '', embedded_with = NULL "
                         "WHERE session = ? AND forgotten_at IS NULL", (now(), session))
            conn.commit()
            return len(seqs)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def purge(self) -> int:
        """Drop the content of every forgotten event; returns the number of blobs deleted."""
        conn = self._tx()
        try:
            hashes = [r[0] for r in conn.execute(
                "SELECT DISTINCT blob FROM events WHERE forgotten_at IS NOT NULL AND blob IS NOT NULL")]
            conn.execute("UPDATE events SET blob = NULL WHERE forgotten_at IS NOT NULL")
            before = conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
            for h in hashes:
                _drop_blob_if_unused(conn, h)
            gone = before - conn.execute("SELECT COUNT(*) FROM blobs").fetchone()[0]
            conn.commit()
            return gone
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def scrub(self, seq: int, raw: str, text: str | None, redactor: str) -> str:
        """Replace one event's content after a better redactor found something. The only edit events allow."""
        if not isinstance(raw, str) or not raw or len(raw.encode()) > MAX_RAW:
            raise Invalid("raw must be a non-empty string")
        if not WORD_RE.match(redactor or "") or not redactor:
            raise Invalid("redactor required")
        conn = self._tx()
        try:
            row = conn.execute("SELECT blob, trust, text FROM events WHERE seq = ?", (seq,)).fetchone()
            if row is None:
                raise NotFound(seq)
            new_text = row["text"] if text is None else (text.strip() if row["trust"] == "trusted" else "")
            h = _put_blob(conn, raw)
            conn.execute("UPDATE events SET blob = ?, redactor = ?, text = ?, embedded_with = CASE WHEN ? = text "
                         "THEN embedded_with ELSE NULL END WHERE seq = ?", (h, redactor, new_text, new_text, seq))
            if new_text != row["text"]:
                _drop_vec(conn, seq)
            _drop_blob_if_unused(conn, row["blob"])
            conn.commit()
            return h
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- proposals ----

    def propose(self, *, text: Any, by: Any, events: Any = None, note: Any = "") -> str:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_PROPOSAL:
            raise Invalid(f"text must be a non-empty string of at most {MAX_PROPOSAL} characters")
        if by not in ("ryan", "agent", "proposer"):
            raise Invalid("by must be ryan, agent or proposer")
        events = events or []
        if not isinstance(events, list) or len(events) > 200 or not all(
                isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in events):
            raise Invalid("events must be a list of event seq numbers")
        if not isinstance(note, str) or len(note) > 2000:
            raise Invalid("note must be a string of at most 2000 characters")
        pid = hashlib.sha256(f"{time.time_ns()}:{text}".encode()).hexdigest()[:16]
        with self.store._conn() as conn:
            known = {r[0] for r in conn.execute(
                f"SELECT seq FROM events WHERE seq IN ({','.join('?' * len(events))})", events)} if events else set()
            if known != set(events):
                raise Invalid(f"unknown events: {sorted(set(events) - known)}")
            conn.execute("INSERT INTO proposals (id, text, status, by, events, note, created_at) "
                         "VALUES (?, ?, 'pending', ?, ?, ?, ?)",
                         (pid, text.strip(), by, json.dumps(sorted(set(events))), note.strip(), now()))
        return pid

    @staticmethod
    def _prop(row) -> dict:
        d = dict(row)
        d["events"] = json.loads(d["events"])
        return d

    def proposals(self, status: str | None = "pending", limit: int = 100) -> list[dict]:
        with self.store._conn() as conn:
            if status:
                rows = conn.execute("SELECT * FROM proposals WHERE status = ? ORDER BY created_at, id LIMIT ?",
                                    (status, limit))
            else:
                rows = conn.execute("SELECT * FROM proposals ORDER BY created_at DESC, id LIMIT ?", (limit,))
            return [self._prop(r) for r in rows]

    def proposal(self, pid: str) -> dict:
        with self.store._conn() as conn:
            r = conn.execute("SELECT * FROM proposals WHERE id = ?", (pid,)).fetchone()
            if r is None:
                raise NotFound(pid)
            return self._prop(r)

    def decide(self, pid: str, action: Any, text: Any = None) -> dict:
        if action not in ("accept", "reject"):
            raise Invalid("action must be accept or reject")
        if text is not None and (not isinstance(text, str) or not text.strip() or len(text) > MAX_PROPOSAL):
            raise Invalid(f"text must be a non-empty string of at most {MAX_PROPOSAL} characters")
        with self.store._conn() as conn:
            r = conn.execute("SELECT status FROM proposals WHERE id = ?", (pid,)).fetchone()
            if r is None:
                raise NotFound(pid)
            if r[0] != "pending":
                raise Conflict(f"proposal is {r[0]}")
            status = "accepted" if action == "accept" else "rejected"
            if text is not None:
                conn.execute("UPDATE proposals SET text = ? WHERE id = ?", (text.strip(), pid))
            conn.execute("UPDATE proposals SET status = ?, decided_at = ? WHERE id = ?", (status, now(), pid))
        return self.proposal(pid)

    def mark_applied(self, pid: str, memory_id: str):
        with self.store._conn() as conn:
            cur = conn.execute("UPDATE proposals SET status = 'applied', applied_at = ?, memory_id = ? "
                               "WHERE id = ? AND status = 'accepted'", (now(), memory_id, pid))
            if cur.rowcount != 1:
                raise Conflict(f"proposal {pid} is not accepted")

    def stats(self) -> dict:
        with self.store._conn() as conn:
            one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
            return {
                "events": one("SELECT COUNT(*) FROM events"),
                "untrusted": one("SELECT COUNT(*) FROM events WHERE trust = 'untrusted'"),
                "forgotten": one("SELECT COUNT(*) FROM events WHERE forgotten_at IS NOT NULL"),
                "unembedded": one("SELECT COUNT(*) FROM events WHERE trust = 'trusted' AND text != '' "
                                  "AND forgotten_at IS NULL AND embedded_with IS NULL"),
                "blobs": one("SELECT COUNT(*) FROM blobs"),
                "blob_bytes": one("SELECT COALESCE(SUM(LENGTH(data)), 0) FROM blobs"),
                "proposals_pending": one("SELECT COUNT(*) FROM proposals WHERE status = 'pending'"),
                "proposals_accepted": one("SELECT COUNT(*) FROM proposals WHERE status = 'accepted'"),
            }
