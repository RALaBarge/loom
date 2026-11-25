# Loom

A local store for what an agent harness did, and for the short memories a person chose to keep from it.

- **Events:** every entry the harness wrote (for Pi, one line of a session file), append-only, acked after commit.
  Trusted text (the person's prompts, final replies) is embedded for search; tool output is kept but never embedded or
  returned by search.
- **Proposals:** candidate memories wait in a queue until a person accepts, edits or rejects them.
- **Memories:** accepted facts with full version history; only active ones are searchable.

SQLite with sqlite-vec; embeddings from Ollama (`nomic-embed-text`, 768 dimensions). Loom stores and searches; it
does not proxy model calls.

## Events and the review queue

`ledger.py` is an append-only event store and a queue of proposed memories that a person decides.

- `POST /events` `{"events": [...]}` stores a batch in one transaction and replies after the commit with one ack per
  event (`seq`, content `hash`, `new`). An event is one entry a harness wrote (for Pi, one session-file line): `source`,
  `session`, `id`, `parent`, `type`, `role`, `ts`, `model`, `provider`, `trust` (`trusted` or `untrusted`), `redactor`,
  `raw` (the redacted entry) and `text` (what may be embedded). (source, session, id) is unique, so a retry stores
  nothing twice. `raw` goes into a content-addressed blob.
- `untrusted` events (tool output) are kept for audit and replay but never embedded and never returned by search.
- A background task embeds trusted `text`; the index records the embedding model, so a new model re-embeds.
- `GET /events?session=&after=&raw=`, `GET /events/sessions`, `POST /events/search`.
- `POST /events/forget` `{"session"}` tombstones a session; `POST /events/purge` deletes forgotten content;
  `POST /events/{seq}/scrub` replaces one event's content (the only edit, for a secret a newer redactor found).
- `POST /proposals` `{"text", "by": "ryan"|"agent"|"proposer", "events": [seq], "note"}` queues a candidate memory;
  `POST /proposals/{id}/decide` `{"action": "accept"|"reject", "text"?}` decides it; `POST /proposals/pop` turns every
  accepted proposal into a memory (source: the proposal, who proposed it, its events).

With `LOOM_TOKENS_DIR` set, each write needs `Authorization: Bearer` with its role's token from `<dir>/<role>.token`:
`ingest` (events), `memq` (propose, pop, touch) and `review` (decide, memory edits, forget, purge, scrub). A role
without a token file is locked. Reads stay open. Unset, Loom is open as before.

## Memories

Memories are short facts you chose to keep, separate from the event log. Each keeps its full
version history; nothing is ever hard-deleted. Only `active` memories are searchable.

| Route | Body | Effect |
|---|---|---|
| `POST /memories` | `{"id"?, "text", "source"?}` | Create version 1 (`source` is any JSON) |
| `GET /memories?status=active&limit=&offset=` | | List (`status`: active, superseded, retracted, all) |
| `GET /memories/{id}` | | One memory with all versions |
| `POST /memories/{id}/revise` | `{"text", "reason"?}` | New version, re-embedded |
| `POST /memories/{id}/supersede` | `{"new_text"?, "new_id"?, "reason"?}` | Mark superseded; optionally create the replacement in the same transaction |
| `POST /memories/{id}/retract` | `{"reason"}` | Soft delete |
| `POST /memories/{id}/touch` | | Update `last_seen_at` only |
| `POST /memories/search` | `{"query", "limit"}` | Nearest active memories (default 5, max 50) |

If the embedding service is down, these routes return `503 {"error": "embedding unavailable"}`
and write nothing.
Changing a superseded or retracted memory returns `409`.

## Other routes

- `GET /stats`: memory counts by status, and `ledger` (events, untrusted, forgotten, unembedded, blobs, proposals by status).
- `GET /health`: `{"ok": true, "features": ["memories_v1", "events_v1", "proposals_v1"]}`.

## Run

```bash
pip install -r requirements.txt
ollama pull nomic-embed-text
python run.py serve          # 127.0.0.1:9741
python run.py search QUERY   # memories; `events QUERY` for event text; `stats`
```

Environment: `LOOM_DB_PATH` (default `~/.loom/loom.db`), `LOOM_OLLAMA_URL` (default `http://localhost:11434`),
`LOOM_EMBED_MODEL`, `LOOM_PORT`, `LOOM_BIND_HOST` (default `127.0.0.1`), `LOOM_TOKENS_DIR`.

A database from before the event store keeps its old `conversations`/`messages` tables; nothing reads them any more,
and they can be dropped by hand.

## Tests

```bash
python -m pytest -q tests    # embeddings stubbed, temp databases only
```

## License

[MIT](LICENSE).
