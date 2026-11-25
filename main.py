import logging
import os
import re
from pathlib import Path
from uuid import uuid4

import asyncio
import hmac
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

try:
    from . import ledger as L
    from .store import MemoryConflict, MemoryNotFound, Store
except ImportError:
    import ledger as L
    from store import MemoryConflict, MemoryNotFound, Store

OLLAMA_URL = os.environ.get("LOOM_OLLAMA_URL", "http://localhost:11434").rstrip("/")
EMBED_MODEL = os.environ.get("LOOM_EMBED_MODEL", "nomic-embed-text")
PORT = int(os.environ.get("LOOM_PORT", "9741"))
DB_PATH = str(Path(os.environ.get("LOOM_DB_PATH", str(Path.home() / ".loom" / "loom.db"))).expanduser().resolve())
# With LOOM_TOKENS_DIR set, every write needs a bearer token for its role, read from <dir>/<role>.token:
#   ingest  POST /events                                the shipper
#   memq    POST /proposals, /proposals/pop, touch     the memq tool (Ryan or the agent)
#   review  deciding proposals, editing memories, forget/purge/scrub   the operator (the console)
# A role whose file is missing is locked. Reads stay open: Loom listens on loopback only.
# Unset, Loom runs open as before (a standalone install with no other local users).
TOKENS_DIR = os.environ.get("LOOM_TOKENS_DIR", "")
MAX_EVENTS_BODY = 16 * 1024 * 1024
# A CPU without AVX embeds at about 0.5 s per token, so a request may take minutes; one cut off early keeps the
# embedding server busy anyway. Events are embedded from their first EMBED_CHARS characters (the text itself is kept
# whole); that is enough to find them.
EMBED_TIMEOUT = 300
EMBED_CHARS = 1500

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s [loom] %(message)s"
)
_log = logging.getLogger("loom")

_store: Store | None = None
_ledger: "L.Ledger | None" = None


@asynccontextmanager
async def _lifespan(app):
    task = asyncio.create_task(_embed_worker())
    try:
        yield
    finally:
        task.cancel()


_app = FastAPI(lifespan=_lifespan)


def _token(role: str) -> bytes | None:
    try:
        with open(os.path.join(TOKENS_DIR, role + ".token"), "rb") as f:
            tok = f.read().strip()
    except OSError:
        return None
    return tok or None


def _denied(request: Request, role: str):
    """None if the caller may act as `role`, else the 401/403 response."""
    if not TOKENS_DIR:
        return None
    want = _token(role)
    if want is None:
        return JSONResponse({"error": f"{role} is locked (no token configured)"}, status_code=403)
    got = request.headers.get("authorization", "")
    if not got.startswith("Bearer ") or not hmac.compare_digest(got[7:].strip().encode(), want):
        return JSONResponse({"error": f"{role} token required"}, status_code=401)
    return None


async def _embed(text: str) -> list[float]:
    """Request embedding from Ollama."""
    if not text.strip():
        return []
    async with httpx.AsyncClient(timeout=EMBED_TIMEOUT) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": [text]},
        )
        data = resp.json()
        return data.get("embeddings", [[]])[0]


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
    if (deny := _denied(request, "review")):
        return deny
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
    if (deny := _denied(request, "review")):
        return deny
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
    if (deny := _denied(request, "review")):
        return deny
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
    if (deny := _denied(request, "review")):
        return deny
    body, err = await _json_body(request)
    if err:
        return err
    try:
        _store.retract_memory(mem_id, reason=_reason_field(body))
    except (MemoryConflict, MemoryNotFound) as e:
        return _mem_error(e)
    return JSONResponse({"id": mem_id, "status": "retracted"})


@_app.post("/memories/{mem_id}/touch")
async def memories_touch(mem_id: str, request: Request):
    if (deny := _denied(request, "memq")):
        return deny
    try:
        _store.touch_memory(mem_id)
    except MemoryNotFound as e:
        return _mem_error(e)
    return JSONResponse({"id": mem_id})


# ── Events (the ledger) and the review queue ───────────────────────

def _ledger_error(e: Exception):
    if isinstance(e, L.Invalid):
        return JSONResponse({"error": str(e)}, status_code=400)
    if isinstance(e, L.NotFound):
        return JSONResponse({"error": "not found"}, status_code=404)
    if isinstance(e, L.Conflict):
        return JSONResponse({"error": str(e)}, status_code=409)
    raise e


@_app.post("/events")
async def events_ingest(request: Request):
    """A batch of events. The reply comes after the commit; an ack per event is the sender's receipt."""
    if (deny := _denied(request, "ingest")):
        return deny
    data = b""
    async for chunk in request.stream():   # counted as it arrives: a chunked body has no Content-Length to check
        data += chunk
        if len(data) > MAX_EVENTS_BODY:
            return JSONResponse({"error": "batch too large"}, status_code=413)
    try:
        body = json.loads(data)
    except ValueError as e:
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)
    if not isinstance(body, dict):
        return JSONResponse({"error": "body must be an object"}, status_code=400)
    events = body.get("events")
    if not isinstance(events, list) or not 1 <= len(events) <= L.MAX_BATCH:
        return JSONResponse({"error": f"events must be a list of 1 to {L.MAX_BATCH}"}, status_code=400)
    try:
        acks = _ledger.ingest(events)
    except L.Invalid as e:
        return _ledger_error(e)
    return JSONResponse({"acks": acks})


@_app.get("/events")
async def events_list(session: str | None = None, after: int = 0, limit: int = 200, raw: bool = False):
    return JSONResponse({"events": _ledger.events(session=session, after=max(0, after),
                                                  limit=max(1, min(limit, 1000)), raw=raw)})


@_app.get("/events/sessions")
async def events_sessions(limit: int = 100):
    return JSONResponse({"sessions": _ledger.sessions(limit=max(1, min(limit, 1000)))})


@_app.post("/events/search")
async def events_search(request: Request):
    body, err = await _json_body(request)
    if err:
        return err
    query = body.get("query")
    limit = body.get("limit", 5)
    if not isinstance(query, str) or not query.strip():
        return JSONResponse({"error": "query must be a non-empty string"}, status_code=400)
    if isinstance(limit, bool) or not isinstance(limit, int):
        return JSONResponse({"error": "limit must be an integer"}, status_code=400)
    emb = await _embed_strict(query.strip())
    if emb is None:
        return JSONResponse(_EMBED_DOWN, status_code=503)
    return JSONResponse({"results": _ledger.search(emb, EMBED_MODEL, limit=max(1, min(limit, 50)))})


@_app.post("/events/forget")
async def events_forget(request: Request):
    if (deny := _denied(request, "review")):
        return deny
    body, err = await _json_body(request)
    if err:
        return err
    session = body.get("session")
    if not isinstance(session, str) or not L.ID_RE.match(session):
        return JSONResponse({"error": "session required"}, status_code=400)
    return JSONResponse({"session": session, "forgotten": _ledger.forget(session)})


@_app.post("/events/purge")
async def events_purge(request: Request):
    if (deny := _denied(request, "review")):
        return deny
    return JSONResponse({"blobs_deleted": _ledger.purge()})


@_app.post("/events/{seq}/scrub")
async def events_scrub(seq: int, request: Request):
    if (deny := _denied(request, "review")):
        return deny
    body, err = await _json_body(request)
    if err:
        return err
    try:
        h = _ledger.scrub(seq, body.get("raw"), body.get("text"), body.get("redactor"))
    except (L.Invalid, L.NotFound) as e:
        return _ledger_error(e)
    return JSONResponse({"seq": seq, "hash": h})


@_app.post("/proposals")
async def proposals_create(request: Request):
    if (deny := _denied(request, "memq")):
        return deny
    body, err = await _json_body(request)
    if err:
        return err
    try:
        pid = _ledger.propose(text=body.get("text"), by=body.get("by"), events=body.get("events"),
                              note=body.get("note", ""))
    except L.Invalid as e:
        return _ledger_error(e)
    return JSONResponse({"id": pid, "status": "pending"})


@_app.get("/proposals")
async def proposals_list(status: str = "pending", limit: int = 100):
    if status not in ("pending", "accepted", "rejected", "applied", "all"):
        return JSONResponse({"error": "bad status"}, status_code=400)
    return JSONResponse({"proposals": _ledger.proposals(None if status == "all" else status,
                                                        limit=max(1, min(limit, 500)))})


@_app.post("/proposals/pop")
async def proposals_pop(request: Request):
    """Move every accepted proposal into memory. Stops at the first one that cannot be embedded."""
    if (deny := _denied(request, "memq")):
        return deny
    applied = []
    for p in _ledger.proposals("accepted", limit=500):
        emb = await _embed_strict(p["text"])
        if emb is None:
            return JSONResponse(dict(_EMBED_DOWN, applied=applied), status_code=503)
        mem_id = "p-" + p["id"]
        source = {"proposal": p["id"], "by": p["by"], "events": p["events"]}
        try:
            _store.create_memory(memory_id=mem_id, text=p["text"], source=source, embedding=emb)
        except MemoryConflict:
            pass                     # a pop that died after the memory but before the mark: finish it
        try:
            _ledger.mark_applied(p["id"], mem_id)
        except L.Conflict:
            continue                 # a concurrent pop applied it first
        applied.append({"proposal": p["id"], "memory": mem_id, "text": p["text"]})
    return JSONResponse({"applied": applied})


@_app.get("/proposals/{pid}")
async def proposals_get(pid: str):
    try:
        return JSONResponse(_ledger.proposal(pid))
    except L.NotFound as e:
        return _ledger_error(e)


@_app.post("/proposals/{pid}/decide")
async def proposals_decide(pid: str, request: Request):
    if (deny := _denied(request, "review")):
        return deny
    body, err = await _json_body(request)
    if err:
        return err
    try:
        return JSONResponse(_ledger.decide(pid, body.get("action"), body.get("text")))
    except (L.Invalid, L.NotFound, L.Conflict) as e:
        return _ledger_error(e)


async def _embed_worker():
    """Embed trusted event text in the background, a few at a time; never on the ingest path."""
    while True:
        try:
            work = _ledger.pending_embeddings(EMBED_MODEL) if _ledger else []
            for seq, text in work:
                emb = await _embed_strict(text[:EMBED_CHARS])
                if emb is None:
                    break
                _ledger.put_embedding(seq, EMBED_MODEL, emb)
            await asyncio.sleep(1 if work else 10)
        except asyncio.CancelledError:
            raise
        except Exception:
            _log.exception("embed worker")
            await asyncio.sleep(30)


@_app.get("/stats")
async def stats():
    """Get storage statistics."""
    if not _store:
        return JSONResponse({"error": "store not initialized"}, status_code=500)
    out = _store.get_stats()
    if _ledger:
        out["ledger"] = _ledger.stats()
    return JSONResponse(out)


@_app.get("/health")
async def health():
    """Health check."""
    return {"ok": True, "features": ["memories_v1", "events_v1", "proposals_v1"]}


def serve():
    """Start the Loom server."""
    import uvicorn

    global _store, _ledger
    _store = Store(DB_PATH)
    _ledger = L.Ledger(_store)
    _log.info("loom started: db=%s ollama=%s model=%s", DB_PATH, OLLAMA_URL, EMBED_MODEL)

    bind_host = os.environ.get("LOOM_BIND_HOST", "127.0.0.1")
    uvicorn.run(_app, host=bind_host, port=PORT)


if __name__ == "__main__":
    serve()
