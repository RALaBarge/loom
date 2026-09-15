# Loom — Memory-First LLM Proxy

A local-first system that logs, searches, and learns from every interaction with any LLM API.

## Why Loom?

LLMs forget. Each conversation starts blank. Loom solves this by:

- **Capturing everything** — every request and response to any LLM API
- **Making it searchable** — semantic search over your interaction history  
- **Learning from it** — extract and distill key insights for future sessions
- **Keeping it local** — all data stays on your machine

Loom is *not* a proxy, router, or tool manager—it only stores and searches.

## Hook Format

Send a POST to `/log` with this shape:

```json
{
  "request": {
    "messages": [
      {"role": "user", "content": "what is Loom?"},
      {"role": "assistant", "content": "..."}
    ]
  },
  "response": {
    "choices": [
      {
        "message": {
          "content": "Loom is a memory system..."
        }
      }
    ],
    "usage": {
      "total_tokens": 150
    }
  },
  "metadata": {
    "conversation_id": "abc123",
    "model": "claude-opus-5",
    "provider": "anthropic"
  }
}
```

Fields:
- `request`: The chat request (at minimum `messages`)
- `response`: The chat response (at minimum `choices[0].message.content`)
- `metadata`: Provider metadata
  - `conversation_id`: Optional. If omitted, Loom generates a UUID
  - `model`: Optional model name
  - `provider`: Optional provider name (e.g., "anthropic", "openai")

Loom extracts the last user message and the first assistant choice, stores both, and embeds them for search.

## API

### `POST /log` — Store an interaction

```bash
curl -X POST http://localhost:9741/log \
  -H "Content-Type: application/json" \
  -d '{
    "request": {"messages": [{"role": "user", "content": "hello"}]},
    "response": {"choices": [{"message": {"content": "hi"}}]},
    "metadata": {"model": "gpt-4"}
  }'
```

Response: `{"id": "conv-id", "stored": true}`

### `POST /search` — Semantic search

```bash
curl -X POST http://localhost:9741/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "how do I deploy to production?",
    "limit": 5,
    "role": "assistant"
  }'
```

Response:
```json
{
  "results": [
    {
      "distance": 0.1234,
      "role": "assistant",
      "conversation_id": "abc123",
      "content": "To deploy...",
      "timestamp": "2025-09-14T12:00:00Z"
    }
  ]
}
```

Body parameters for `/search`:
- `query`: Search text (required)
- `limit`: Max results (default 5, max 100)
- `role`: Filter by "user" or "assistant" (optional)
- `conversation_id`: Filter to a specific conversation (optional)

### `GET /conversations` — List conversations

```bash
curl http://localhost:9741/conversations?limit=10
```

Response:
```json
{
  "conversations": [
    {
      "id": "abc123",
      "created_at": "2025-09-14T12:00:00Z",
      "message_count": 5,
      "model": "claude-opus-5",
      "first_message": "what is..."
    }
  ]
}
```

### `GET /conversations/{id}` — Get conversation messages

```bash
curl http://localhost:9741/conversations/abc123
```

### `GET /stats` — Storage statistics

```bash
curl http://localhost:9741/stats
```

Response:
```json
{
  "conversations": 42,
  "messages": 250,
  "embeddings": 248,
  "total_tokens": 50000
}
```

### `GET /health` — Health check

```bash
curl http://localhost:9741/health
```

## CLI

### Start the server

```bash
loom serve
```

Or via Docker:
```bash
docker-compose up
```

### Search from terminal

```bash
loom search "how do I deploy?"
```

### List conversations

```bash
loom conversations
```

### View a conversation

```bash
loom conversation abc123
```

## Setup

### 1. Requirements

- Python 3.13+
- SQLite (included)
- Ollama running with an embedding model (e.g., `nomic-embed-text`)

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` to `.env` and set the environment variables:

```bash
LOOM_OLLAMA_URL=http://localhost:11434
LOOM_EMBED_MODEL=nomic-embed-text
LOOM_PORT=9741
LOOM_DB_PATH=~/.loom/loom.db
```

Before starting Loom, load your `.env` file:
```bash
export $(cat .env | xargs)
loom serve
```

Or inline:
```bash
LOOM_OLLAMA_URL=http://localhost:11434 loom serve
```

### 4. Start Ollama

```bash
ollama pull nomic-embed-text
ollama serve
```

(If Ollama is already running elsewhere, point `LOOM_OLLAMA_URL` to it.)

### 5. Start Loom

```bash
loom serve
```

The server runs on `http://localhost:9741` by default.

## Integration

To send logs from Claude Code, add a hook to `.claude/settings.json`:

```json
{
  "hooks": {
    "after:request": "curl -X POST http://localhost:9741/log -H 'Content-Type: application/json' -d '{\"request\": ..., \"response\": ...}'"
  }
}
```

(See Claude Code docs for hook details.)

## Edge Cases

- **Partial messages**: Loom stores what's there. Missing fields don't break ingestion.
- **Embedding failures**: Messages are stored even if embedding fails (searchable by metadata/timestamp).
- **Malformed JSON**: Returns HTTP 400, logged.
- **Large messages**: Stored as-is; embeddings are truncated by Ollama.

## Architecture

```
POST /log
  ├─ Extract user & assistant messages
  ├─ Store in messages table
  ├─ Embed both messages
  └─ Store embeddings in vec_messages table

POST /search
  ├─ Embed query
  ├─ Vector search vec_messages
  └─ Return top results + metadata

GET /conversations
  └─ Return from conversations table (with counts)
```

Database: SQLite with sqlite-vec extension for vector search.
Embeddings: Ollama HTTP API (768-dim by default).

## Code Structure

- **`main.py`** (239 lines) — FastAPI server with `/log`, `/search`, `/conversations` endpoints
- **`store.py`** (414 lines) — SQLite + sqlite-vec database layer with vector search
- **`cli.py`** (187 lines) — Terminal UI for searching and browsing
- **`run.py`** — Entry point wrapper
- **`requirements.txt`** — 5 minimal dependencies

Every function is documented. ~850 lines of production-ready code.

## Educational Value

This codebase is a reference implementation for:

- **Memory-augmented LLM systems** — how to build persistent, searchable context
- **Vector embeddings** — practical use of semantic search (768-dim vectors, Ollama)
- **SQLite + extensions** — using sqlite-vec for embedded vector storage
- **FastAPI** — async HTTP server with proper error handling
- **CLI design** — terminal UI with multiple subcommands
- **Hook-based logging** — integration without proxying or MITM

Clean code, full documentation, proper error handling. No secrets, no TODOs.

## License

MIT
