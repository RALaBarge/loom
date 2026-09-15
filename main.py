import json
import logging
import os
import time
from pathlib import Path
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from .store import Store
except ImportError:
    from store import Store

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


@_app.get("/stats")
async def stats():
    """Get storage statistics."""
    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)
    return JSONResponse(_store.get_stats())


@_app.get("/health")
async def health():
    """Health check."""
    return {"ok": True}


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
