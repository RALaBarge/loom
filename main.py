import json
import logging
import os
import re
import time
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from .store import MemoryConflict, MemoryNotFound, Store
except ImportError:
    from store import MemoryConflict, MemoryNotFound, Store

OLLAMA_URL = os.environ.get("LOOM_OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("LOOM_EMBED_MODEL", "nomic-embed-text")
PORT = int(os.environ.get("LOOM_PORT", "9741"))
DB_PATH = str(Path(os.environ.get("LOOM_DB_PATH", str(Path.home() / ".loom" / "loom.db"))).expanduser().resolve())

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [loom] %(message)s"
)
_log = logging.getLogger("loom")

_app = FastAPI()
_store: Store | None = None


def _now() -> str:
    """Get current UTC timestamp in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


async def _embed(text: str) -> list[float]:
    """Request embedding from Ollama."""
    if not text.strip():
        return []
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": [text]},
        )
        data = resp.json()
        return data.get("embeddings", [[]])[0]


@_app.post("/log")
async def log_interaction(request: Request):
    """Receive and store a logged interaction (request + response + metadata)."""
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)

    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)

    req = body.get("request", {})
    resp = body.get("response", {})
    metadata = body.get("metadata", {})

    conv_id = metadata.get("conversation_id", uuid4().hex)
    model = metadata.get("model", "")
    provider = metadata.get("provider", "")
    ts = _now()

    try:
        # Extract user message from request
        messages = req.get("messages", [])
        user_msg = None
        if messages:
            for m in reversed(messages):
                if m.get("role") == "user":
                    user_msg = m.get("content", "")
                    if isinstance(user_msg, list):
                        user_msg = json.dumps(user_msg)
                    break

        # Extract assistant message from response
        assistant_msg = ""
        if "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            msg = choice.get("message", {})
            assistant_msg = msg.get("content", "")
            if isinstance(assistant_msg, list):
                assistant_msg = json.dumps(assistant_msg)

        # Ensure conversation exists
        _store.ensure_conversation(conv_id)

        # Store user message
        if user_msg:
            _store.insert_message(
                msg_id=uuid4().hex,
                conv_id=conv_id,
                role="user",
                content=user_msg,
                model=model,
                timestamp=ts,
            )
            # Embed user message in background
            try:
                embedding = await _embed(user_msg)
                if embedding:
                    _store.insert_embedding(
                        embedding=embedding,
                        role="user",
                        conv_id=conv_id,
                        content=user_msg,
                        timestamp=ts,
                    )
            except Exception:
                _log.exception("embed failed for user message")

        # Store assistant message
        if assistant_msg:
            usage = resp.get("usage", {})
            _store.insert_message(
                msg_id=uuid4().hex,
                conv_id=conv_id,
                role="assistant",
                content=assistant_msg,
                model=model,
                timestamp=ts,
                token_count=usage.get("total_tokens", 0),
                provider=provider,
            )
            # Embed assistant message in background
            try:
                embedding = await _embed(assistant_msg)
                if embedding:
                    _store.insert_embedding(
                        embedding=embedding,
                        role="assistant",
                        conv_id=conv_id,
                        content=assistant_msg,
                        timestamp=ts,
                    )
            except Exception:
                _log.exception("embed failed for assistant message")

        return JSONResponse({
            "id": conv_id,
            "stored": bool(user_msg or assistant_msg),
        })

    except Exception as e:
        _log.exception("log handler failed")
        return JSONResponse({"error": str(e)}, status_code=500)


@_app.post("/search")
async def search(request: Request):
    """Semantic search over logged interactions."""
    try:
        body = await request.json()
    except Exception as e:
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)

    query = body.get("query", "").strip()
    if not query:
        return JSONResponse({"results": []})

    limit = min(body.get("limit", 5), 100)
    role = body.get("role")
    conv_id_filter = body.get("conversation_id")

    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)

    try:
        embedding = await _embed(query)
    except Exception:
        _log.exception("search embed failed")
        return JSONResponse({"results": [], "error": "embedding failed"})

    if not embedding:
        return JSONResponse({"results": []})

    try:
        results = _store.search(embedding, limit=limit, role=role)
        if conv_id_filter:
            results = [r for r in results if r["conversation_id"] == conv_id_filter]
        return JSONResponse({"results": results})
    except Exception:
        _log.exception("search failed")
        return JSONResponse({"results": [], "error": "search failed"})


@_app.get("/conversations")
async def list_conversations(limit: int = 50):
    """List conversations."""
    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)
    results = _store.get_conversations(limit=min(limit, 200))
    return JSONResponse({"conversations": results})


@_app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get messages in a conversation."""
    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)
    messages = _store.get_messages(conv_id)
    if not messages:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse({"id": conv_id, "messages": messages})


@_app.get("/conversations/{conv_id}/messages")
async def get_message_range(conv_id: str, from_seq: int = 0,
                            to_seq: int | None = None, limit: int = 500):
    """Messages with from_seq <= seq <= to_seq (inclusive), ordered by seq."""
    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)
    if not _store.conversation_exists(conv_id):
        return JSONResponse({"error": "not found"}, status_code=404)
    msgs = _store.get_message_range(conv_id, from_seq, to_seq, max(1, min(limit, 1000)))
    return JSONResponse({"id": conv_id, "messages": msgs})


# ── Curated memories ────────────────────────────────────────────────

_EMBED_DOWN = {"error": "embedding unavailable"}


async def _embed_strict(text: str) -> list[float] | None:
    """Embedding for the memory routes: None means unavailable (never an empty success)."""
    try:
        emb = await _embed(text)
    except Exception:
        _log.exception("memory embed failed")
        return None
    return emb or None


async def _json_body(request: Request):
    try:
        body = await request.json()
    except Exception as e:
        return None, JSONResponse({"error": f"invalid json: {e}"}, status_code=400)
    if not isinstance(body, dict):
        return None, JSONResponse({"error": "body must be an object"}, status_code=400)
    return body, None


_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
MAX_MEMORY_CHARS = 8000


def _text_field(body: dict, key: str):
    """A required, non-empty string of bounded length -> (value, None) or (None, 400 response)."""
    v = body.get(key)
    if not isinstance(v, str):
        return None, JSONResponse({"error": f"{key} must be a string"}, status_code=400)
    v = v.strip()
    if not v:
        return None, JSONResponse({"error": f"{key} required"}, status_code=400)
    if len(v) > MAX_MEMORY_CHARS:
        return None, JSONResponse({"error": f"{key} too long (max {MAX_MEMORY_CHARS})"}, status_code=400)
    return v, None


def _id_field(body: dict, key: str):
    """An optional id: absent/None -> (None, None); otherwise must match _ID_RE."""
    v = body.get(key)
    if v is None:
        return None, None
    if not isinstance(v, str) or not _ID_RE.match(v):
        return None, JSONResponse({"error": f"{key} must match [A-Za-z0-9._-]{{1,64}}"}, status_code=400)
    return v, None


def _reason_field(body: dict) -> str:
    r = body.get("reason", "")
    return r.strip()[:MAX_MEMORY_CHARS] if isinstance(r, str) else ""


def _mem_error(e: Exception):
    if isinstance(e, MemoryNotFound):
        return JSONResponse({"error": "not found"}, status_code=404)
    if isinstance(e, MemoryConflict):
        return JSONResponse({"error": str(e)}, status_code=409)
    raise e


@_app.post("/memories/search")
async def memories_search(request: Request):
    body, err = await _json_body(request)
    if err:
        return err
    query = body.get("query", "")
    if not isinstance(query, str):
        return JSONResponse({"error": "query must be a string"}, status_code=400)
    query = query.strip()
    if not query:
        return JSONResponse({"results": []})
    limit = body.get("limit", 5)
    if isinstance(limit, bool) or not isinstance(limit, int):
        return JSONResponse({"error": "limit must be an integer"}, status_code=400)
    limit = max(1, min(limit, 50))
    emb = await _embed_strict(query)
    if emb is None:
        return JSONResponse(_EMBED_DOWN, status_code=503)
    return JSONResponse({"results": _store.search_memories(emb, limit=limit)})


@_app.post("/memories")
async def memories_create(request: Request):
    body, err = await _json_body(request)
    if err:
        return err
    text, err = _text_field(body, "text")
    if err:
        return err
    given_id, err = _id_field(body, "id")
    if err:
        return err
    emb = await _embed_strict(text)
    if emb is None:
        return JSONResponse(_EMBED_DOWN, status_code=503)
    mem_id = given_id or uuid4().hex
    try:
        ver = _store.create_memory(memory_id=mem_id, text=text,
                                   source=body.get("source"), embedding=emb)
    except (MemoryConflict, MemoryNotFound) as e:
        return _mem_error(e)
    return JSONResponse({"id": mem_id, "version": ver})


@_app.get("/memories")
async def memories_list(status: str = "active", limit: int = 50, offset: int = 0):
    if status not in ("active", "superseded", "retracted", "all"):
        return JSONResponse({"error": "bad status"}, status_code=400)
    items = _store.list_memories(status=None if status == "all" else status,
                                 limit=max(1, min(limit, 200)), offset=max(0, offset))
    return JSONResponse({"memories": items})


@_app.get("/memories/{mem_id}")
async def memories_get(mem_id: str):
    try:
        return JSONResponse(_store.get_memory(mem_id))
    except MemoryNotFound as e:
        return _mem_error(e)


@_app.post("/memories/{mem_id}/revise")
async def memories_revise(mem_id: str, request: Request):
    body, err = await _json_body(request)
    if err:
        return err
    text, err = _text_field(body, "text")
    if err:
        return err
    try:
        _store.get_memory(mem_id)
    except MemoryNotFound as e:
        return _mem_error(e)
    emb = await _embed_strict(text)
    if emb is None:
        return JSONResponse(_EMBED_DOWN, status_code=503)
    try:
        ver = _store.revise_memory(mem_id, text=text, reason=_reason_field(body),
                                   embedding=emb)
    except (MemoryConflict, MemoryNotFound) as e:
        return _mem_error(e)
    return JSONResponse({"id": mem_id, "version": ver})


@_app.post("/memories/{mem_id}/supersede")
async def memories_supersede(mem_id: str, request: Request):
    body, err = await _json_body(request)
    if err:
        return err
    new_text = body.get("new_text")
    new_id, err = _id_field(body, "new_id")
    if err:
        return err
    emb = None
    if new_text is not None:
        new_text, err = _text_field(body, "new_text")
        if err:
            return err
        new_id = new_id or uuid4().hex
        try:
            _store.get_memory(mem_id)
        except MemoryNotFound as e:
            return _mem_error(e)
        emb = await _embed_strict(new_text)
        if emb is None:
            return JSONResponse(_EMBED_DOWN, status_code=503)
    try:
        out = _store.supersede_memory(
            mem_id, reason=_reason_field(body), new_id=new_id,
            new_text=new_text, new_source=body.get("source"), new_embedding=emb)
    except (MemoryConflict, MemoryNotFound) as e:
        return _mem_error(e)
    return JSONResponse({"id": mem_id, "superseded_by": out})


@_app.post("/memories/{mem_id}/retract")
async def memories_retract(mem_id: str, request: Request):
    body, err = await _json_body(request)
    if err:
        return err
    try:
        _store.retract_memory(mem_id, reason=_reason_field(body))
    except (MemoryConflict, MemoryNotFound) as e:
        return _mem_error(e)
    return JSONResponse({"id": mem_id, "status": "retracted"})


@_app.post("/memories/{mem_id}/touch")
async def memories_touch(mem_id: str):
    try:
        _store.touch_memory(mem_id)
    except MemoryNotFound as e:
        return _mem_error(e)
    return JSONResponse({"id": mem_id})


@_app.get("/stats")
async def stats():
    """Get storage statistics."""
    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)
    return JSONResponse(_store.get_stats())


@_app.get("/health")
async def health():
    """Health check."""
    return {"ok": True, "features": ["seq", "memories_v1"]}


def serve():
    """Start the Loom server."""
    import uvicorn

    global _store
    _store = Store(DB_PATH)
    _log.info("loom started: db=%s ollama=%s model=%s", DB_PATH, OLLAMA_URL, EMBED_MODEL)

    bind_host = os.environ.get("LOOM_BIND_HOST", "127.0.0.1")
    uvicorn.run(_app, host=bind_host, port=PORT)


if __name__ == "__main__":
    serve()
