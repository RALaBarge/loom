"""Tests for curated memories.

Embeddings are stubbed with a deterministic hash vector, so no Ollama is needed.
Every test runs against a temp database; nothing touches ~/.loom.
"""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from store import Store  # noqa: E402

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


def test_health_lists_features(client):
    body = client.get("/health").json()
    assert body["ok"] is True and set(body["features"]) == {"memories_v1", "events_v1", "proposals_v1"}


def test_the_old_log_and_conversation_routes_are_gone(client):
    paths = {getattr(r, "path", "") for r in client.app.routes}
    assert not paths & {"/log", "/search", "/conversations", "/conversations/{conv_id}"}


def test_no_route_deletes_or_rewrites(client):
    for route in client.app.routes:
        assert not ((getattr(route, "methods", set()) or set()) & {"DELETE", "PUT", "PATCH"}), route.path


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


# ── Input validation (found in review) ─────────────────────────────

@pytest.mark.parametrize("limit", ["abc", None, {"a": 1}, 2.5, True])
def test_search_limit_must_be_an_integer(client, limit):
    r = client.post("/memories/search", json={"query": "x", "limit": limit})
    assert r.status_code == 400


@pytest.mark.parametrize("text", [{"a": 1}, [1, 2], True, 5, None, "", "   ", "x" * 8001])
def test_memory_text_must_be_a_bounded_string(client, text):
    assert client.post("/memories", json={"text": text}).status_code == 400
    a = mem(client, "fine")
    assert client.post(f"/memories/{a['id']}/revise", json={"text": text}).status_code == 400
    if text is not None:  # null new_text means "no replacement", which is valid
        assert client.post(f"/memories/{a['id']}/supersede", json={"new_text": text}).status_code == 400


@pytest.mark.parametrize("bad", [123, "x" * 65, "a/b", "has space", ""])
def test_memory_ids_are_restricted(client, bad):
    assert client.post("/memories", json={"text": "t", "id": bad}).status_code == 400


def test_bare_new_id_must_be_a_different_active_memory(client):
    d, e = mem(client, "dee"), mem(client, "eee")
    assert client.post(f"/memories/{e['id']}/supersede", json={"new_id": e["id"]}).status_code == 409
    client.post(f"/memories/{d['id']}/retract", json={"reason": "x"})
    assert client.post(f"/memories/{e['id']}/supersede", json={"new_id": d["id"]}).status_code == 409
    assert client.get(f"/memories/{e['id']}").json()["status"] == "active"
    f = mem(client, "eff")
    assert client.post(f"/memories/{e['id']}/supersede", json={"new_id": f["id"]}).status_code == 200
