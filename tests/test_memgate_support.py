"""Tests for the seq / append-only / curated-memory additions.

Embeddings are stubbed with a deterministic hash vector, so no Ollama is needed.
Every test runs against a temp database; nothing touches ~/.loom.
"""
import hashlib
import sqlite3
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from store import Store  # noqa: E402

OLD_SCHEMA = """
CREATE TABLE conversations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL);
CREATE TABLE messages (
    id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL, role TEXT NOT NULL,
    content TEXT NOT NULL, model TEXT DEFAULT '', timestamp TEXT NOT NULL,
    token_count INTEGER DEFAULT 0, latency_ms REAL DEFAULT NULL,
    reasoning_content TEXT DEFAULT '', finish_reason TEXT DEFAULT '',
    tool_calls TEXT DEFAULT '[]', provider TEXT DEFAULT '',
    response_headers TEXT DEFAULT '{}', context_id TEXT DEFAULT '',
    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
);
CREATE INDEX idx_messages_conv ON messages(conversation_id, timestamp);
"""


def stub_vec(text: str) -> list[float]:
    out, i = [], 0
    while len(out) < 768:
        h = hashlib.sha256(f"{i}:{text}".encode()).digest()
        out.extend((b - 127.5) / 127.5 for b in h)
        i += 1
    return out[:768]


@pytest.fixture
def client(tmp_path, monkeypatch):
    async def fake_embed(text):
        return stub_vec(text) if text.strip() else []

    monkeypatch.setattr(main, "_embed", fake_embed)
    monkeypatch.setattr(main, "_store", Store(tmp_path / "loom.db"))
    yield TestClient(main._app)


@pytest.fixture
def embed_down(monkeypatch):
    async def boom(text):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(main, "_embed", boom)


def log(client, conv, user, assistant):
    return client.post("/log", json={
        "request": {"messages": [{"role": "user", "content": user}]},
        "response": {"choices": [{"message": {"content": assistant}}], "usage": {"total_tokens": 7}},
        "metadata": {"conversation_id": conv, "model": "m", "provider": "p"},
    })


# ── L1: seq and append-only ────────────────────────────────────────

def test_migration_backfills_seq_in_order(tmp_path):
    db = tmp_path / "old.db"
    c = sqlite3.connect(db)
    c.executescript(OLD_SCHEMA)
    c.execute("INSERT INTO conversations VALUES ('c1','t0'), ('c2','t0')")
    rows = [  # equal timestamps on purpose: rowid breaks the tie
        ("a", "c1", "user", "one", "2026-01-01T00:00:00Z"),
        ("b", "c1", "assistant", "two", "2026-01-01T00:00:00Z"),
        ("c", "c1", "user", "three", "2026-01-01T00:00:05Z"),
        ("d", "c2", "user", "other", "2026-01-01T00:00:01Z"),
    ]
    for r in rows:
        c.execute("INSERT INTO messages (id, conversation_id, role, content, timestamp) "
                  "VALUES (?,?,?,?,?)", r)
    c.commit()
    c.close()

    store = Store(db)
    msgs = store.get_messages("c1")
    assert [(m["id"], m["seq"]) for m in msgs] == [("a", 0), ("b", 1), ("c", 2)]
    assert [(m["id"], m["seq"]) for m in store.get_messages("c2")] == [("d", 0)]
    # idempotent: opening again changes nothing, and new inserts continue the sequence
    store2 = Store(db)
    assert store2.insert_message(msg_id="e", conv_id="c1", role="user", content="four",
                                 timestamp="2026-01-01T00:01:00Z")
    assert [m["seq"] for m in store2.get_messages("c1")] == [0, 1, 2, 3]


def test_resending_a_message_id_changes_nothing(tmp_path):
    store = Store(tmp_path / "l.db")
    store.ensure_conversation("c")
    assert store.insert_message(msg_id="x", conv_id="c", role="user", content="original",
                                timestamp="t1") is True
    assert store.insert_message(msg_id="x", conv_id="c", role="user", content="TAMPERED",
                                timestamp="t2") is False
    (m,) = store.get_messages("c")
    assert m["content"] == "original" and m["seq"] == 0 and m["timestamp"] == "t1"


def test_concurrent_inserts_give_unique_dense_seq(tmp_path):
    store = Store(tmp_path / "l.db")
    store.ensure_conversation("c")
    errors = []

    def work(t):
        try:
            for i in range(25):
                store.insert_message(msg_id=f"{t}-{i}", conv_id="c", role="user",
                                     content="x", timestamp="t")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert sorted(m["seq"] for m in store.get_messages("c")) == list(range(200))


def test_range_endpoint_is_inclusive_and_ordered(client):
    for i in range(3):
        log(client, "c", f"u{i}", f"a{i}")  # seq 0..5
    r = client.get("/conversations/c/messages", params={"from_seq": 1, "to_seq": 3})
    assert r.status_code == 200
    got = r.json()["messages"]
    assert [m["seq"] for m in got] == [1, 2, 3]
    assert set(got[0]) == {"id", "seq", "role", "content", "timestamp"}
    assert [m["seq"] for m in client.get("/conversations/c/messages",
                                         params={"from_seq": 4}).json()["messages"]] == [4, 5]
    assert [m["seq"] for m in client.get("/conversations/c/messages",
                                         params={"limit": 2}).json()["messages"]] == [0, 1]
    assert client.get("/conversations/c/messages",
                      params={"from_seq": 50}).json()["messages"] == []
    assert client.get("/conversations/nope/messages").status_code == 404


def test_seq_in_conversation_and_search_payloads(client):
    log(client, "c", "hello there", "general kenobi")
    msgs = client.get("/conversations/c").json()["messages"]
    assert [m["seq"] for m in msgs] == [0, 1]
    hits = client.post("/search", json={"query": "hello there", "limit": 5}).json()["results"]
    assert hits and hits[0]["seq"] == 0


def test_health_lists_features(client):
    body = client.get("/health").json()
    assert body["ok"] is True and set(body["features"]) == {"seq", "memories_v1"}


def test_no_route_deletes_or_rewrites_messages(client):
    for route in client.app.routes:
        methods = getattr(route, "methods", set()) or set()
        if "/memories" in route.path:
            continue
        assert not (methods & {"DELETE", "PUT", "PATCH"}), route.path


# ── Existing behaviour is unchanged ────────────────────────────────

def test_existing_routes_keep_their_shape(client):
    r = log(client, "c", "q", "a")
    assert r.json() == {"id": "c", "stored": True}
    convs = client.get("/conversations").json()["conversations"]
    assert set(convs[0]) == {"id", "created_at", "message_count", "model", "first_message"}
    conv = client.get("/conversations/c").json()
    assert conv["id"] == "c"
    assert {"id", "role", "content", "model", "timestamp", "token_count", "latency_ms",
            "reasoning_content", "finish_reason", "tool_calls", "provider",
            "response_headers", "context_id"} <= set(conv["messages"][0])
    assert set(client.get("/stats").json()) == {
        "conversations", "messages", "context_blobs", "total_tokens", "embeddings",
        "memory_entries"}
    hit = client.post("/search", json={"query": "q"}).json()["results"][0]
    assert {"distance", "role", "conversation_id", "content", "timestamp"} <= set(hit)
    assert client.get("/conversations/missing").status_code == 404


def test_search_still_degrades_to_200_on_embedding_failure(client, embed_down):
    r = client.post("/search", json={"query": "q"})
    assert r.status_code == 200 and r.json()["error"] == "embedding failed"


def test_mcp_tools_are_unchanged():
    src = (Path(__file__).resolve().parent.parent / "mcp_server.py").read_text(encoding="utf-8")
    for name in ("loom_search", "loom_conversations", "loom_conversation", "loom_stats"):
        assert f'"{name}": {{' in src


# ── L2: curated memories ───────────────────────────────────────────

def mem(client, text, **kw):
    r = client.post("/memories", json={"text": text, **kw})
    assert r.status_code == 200, r.text
    return r.json()


def test_memory_lifecycle(client):
    a = mem(client, "Keeps notes in plain markdown files.",
            source={"conversation_id": "c1", "seq_range": [12, 14]})
    assert a["version"] == 1
    got = client.get(f"/memories/{a['id']}").json()
    assert got["status"] == "active" and got["source"]["seq_range"] == [12, 14]
    assert [v["version"] for v in got["versions"]] == [1]

    r = client.post(f"/memories/{a['id']}/revise",
                    json={"text": "Keeps notes in markdown.", "reason": "shorter"})
    assert r.json()["version"] == 2
    got = client.get(f"/memories/{a['id']}").json()
    assert got["text"] == "Keeps notes in markdown." and len(got["versions"]) == 2

    hits = client.post("/memories/search", json={"query": "Keeps notes in markdown."}).json()["results"]
    assert hits[0]["id"] == a["id"] and hits[0]["version"] == 2 and hits[0]["distance"] < 0.01

    r = client.post(f"/memories/{a['id']}/supersede",
                    json={"new_text": "Uses Obsidian for notes.", "reason": "changed tools"})
    assert r.status_code == 200
    new_id = r.json()["superseded_by"]
    old = client.get(f"/memories/{a['id']}").json()
    assert old["status"] == "superseded" and old["superseded_by"] == new_id
    ids = [h["id"] for h in client.post("/memories/search",
                                        json={"query": "notes", "limit": 10}).json()["results"]]
    assert ids == [new_id]

    assert client.post(f"/memories/{new_id}/retract", json={"reason": "wrong"}).status_code == 200
    assert client.post("/memories/search", json={"query": "notes"}).json()["results"] == []
    assert client.get(f"/memories/{new_id}").json()["status"] == "retracted"
    # history is kept forever; there is no hard delete
    assert len(client.get(f"/memories/{new_id}").json()["versions"]) == 2
    assert client.delete(f"/memories/{new_id}").status_code in (404, 405)


def test_inactive_memories_cannot_be_changed(client):
    a = mem(client, "one")
    client.post(f"/memories/{a['id']}/retract", json={"reason": "x"})
    assert client.post(f"/memories/{a['id']}/revise", json={"text": "two"}).status_code == 409
    assert client.post(f"/memories/{a['id']}/retract", json={"reason": "x"}).status_code == 409
    assert client.post(f"/memories/{a['id']}/supersede", json={"new_text": "t"}).status_code == 409
    assert client.post("/memories", json={"id": a["id"], "text": "dup"}).status_code == 409
    assert client.post("/memories/nope/revise", json={"text": "t"}).status_code == 404


def test_list_and_touch(client):
    a, b = mem(client, "alpha"), mem(client, "beta")
    client.post(f"/memories/{b['id']}/retract", json={"reason": "x"})
    active = client.get("/memories").json()["memories"]
    assert [m["id"] for m in active] == [a["id"]]
    assert len(client.get("/memories", params={"status": "all"}).json()["memories"]) == 2
    before = client.get(f"/memories/{a['id']}").json()
    client.post(f"/memories/{a['id']}/touch")
    after = client.get(f"/memories/{a['id']}").json()
    assert after["text"] == before["text"] and after["updated_at"] == before["updated_at"]
    assert client.post("/memories/nope/touch").status_code == 404


def test_embedding_failure_is_a_503_and_writes_nothing(client, embed_down):
    assert client.post("/memories", json={"text": "x"}).status_code == 503
    r = client.post("/memories/search", json={"query": "x"})
    assert r.status_code == 503 and r.json() == {"error": "embedding unavailable"}
    assert main._store.list_memories(status=None) == []


def test_revise_and_supersede_write_nothing_when_embedding_fails(client, monkeypatch):
    a = mem(client, "original")

    async def boom(text):
        raise RuntimeError("down")

    monkeypatch.setattr(main, "_embed", boom)
    assert client.post(f"/memories/{a['id']}/revise", json={"text": "new"}).status_code == 503
    assert client.post(f"/memories/{a['id']}/supersede", json={"new_text": "new"}).status_code == 503
    got = main._store.get_memory(a["id"])
    assert got["text"] == "original" and got["status"] == "active" and len(got["versions"]) == 1


def test_supersede_without_new_text_just_closes_the_memory(client):
    a = mem(client, "gone soon")
    r = client.post(f"/memories/{a['id']}/supersede", json={"reason": "obsolete"})
    assert r.status_code == 200 and r.json()["superseded_by"] is None
    assert client.post("/memories/search", json={"query": "gone soon"}).json()["results"] == []


def test_search_limit_is_capped(client):
    for i in range(3):
        mem(client, f"m{i}")
    assert len(client.post("/memories/search", json={"query": "m", "limit": 1}).json()["results"]) == 1
    assert len(client.post("/memories/search", json={"query": "m", "limit": 999}).json()["results"]) == 3
