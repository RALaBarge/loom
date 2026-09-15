"""Loom CLI — search, browse, chat from the terminal."""
import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

DB_PATH = os.environ.get("LOOM_DB_PATH", str(Path.home() / ".loom" / "loom.db"))
LOOM_URL = os.environ.get("LOOM_URL", f"http://localhost:{os.environ.get('LOOM_PORT', '9741')}")


def main():
    """CLI entry point for Loom — serve, search, browse, or chat."""
    args = sys.argv[1:] if len(sys.argv) > 1 else ["serve"]

    if args[0] == "serve":
        try:
            from .main import serve
        except ImportError:
            from main import serve
        serve()

    elif args[0] == "search":
        query = " ".join(args[1:]) if len(args) > 1 else ""
        if not query:
            print("usage: loom search <query>")
            sys.exit(1)
        _search(query)

    elif args[0] in ("conversations", "convs"):
        _list_conversations()

    elif args[0] in ("conversation", "conv"):
        conv_id = args[1] if len(args) > 1 else ""
        if not conv_id:
            print("usage: loom conversation <id>")
            sys.exit(1)
        _show_conversation(conv_id)

    elif args[0] == "chat":
        asyncio.run(_chat())

    elif args[0] in ("version", "--version", "-v"):
        print("loom 0.2.0")

    else:
        print("loom — memory-first LLM proxy")
        print()
        print("  serve         start the proxy server")
        print("  search        loom search <query>")
        print("  conversations loom conversations")
        print("  conversation  loom conversation <id>")
        print("  chat          loom chat")
        print("  version       loom version")


def _search(query: str):
    """Semantic search over stored interactions."""
    try:
        from .store import Store
        from .main import _embed
    except ImportError:
        from store import Store
        from main import _embed
    store = Store(DB_PATH)
    embedding = asyncio.run(_embed(query))
    if not embedding:
        print("embedding failed")
        return
    results = store.search(embedding, limit=10)
    if not results:
        print("no results")
        return
    for i, r in enumerate(results):
        score = max(0, 1.0 - r["distance"])
        content = r["content"][:200]
        print(f"\n[{i+1}] {r['role']}  score={score:.3f}  conv={r['conversation_id'][:8]}")
        print(f"    {r['timestamp']}")
        print(f"    {content}")


def _list_conversations():
    """List recent conversations with message counts and previews."""
    try:
        from .store import Store
    except ImportError:
        from store import Store
    store = Store(DB_PATH)
    convs = store.get_conversations(50)
    if not convs:
        print("no conversations")
        return
    for c in convs:
        print(f"{c['id'][:8]}  {c['created_at']}  {c['message_count']:>3} msgs  {c['model']}  {c['first_message'][:80]}")


def _show_conversation(conv_id: str):
    """Display all messages in a conversation with metadata."""
    try:
        from .store import Store
    except ImportError:
        from store import Store
    store = Store(DB_PATH)
    msgs = store.get_messages(conv_id)
    if not msgs:
        print(f"conversation {conv_id} not found")
        return
    print(f"conversation {conv_id} ({len(msgs)} messages)\n")
    for m in msgs:
        content = m["content"][:500]
        extra = ""
        if m.get("reasoning_content"):
            extra += f" [reasoning: {len(m['reasoning_content'])} chars]"
        if m.get("tool_calls") and m["tool_calls"] != "[]":
            extra += " [tools]"
        if m.get("finish_reason"):
            extra += f" [{m['finish_reason']}]"
        hdrs = m.get("response_headers", "{}")
        if hdrs != "{}":
            try:
                h = json.loads(hdrs)
                extra += f" [provider={h.get('x-generation-id','?')[:12]}]"
            except Exception:
                pass
        print(f"[{m['role']}] {m.get('model','')}  tokens={m.get('token_count',0)}  latency={m.get('latency_ms')}{extra}")
        print(content)
        print()


async def _chat():
    """Chat via the local loom proxy (not OpenRouter directly) so every
    turn actually gets logged and embedded — this is the whole point of loom."""
    model = os.environ.get("LOOM_MODEL", "openai/gpt-4o")
    messages = []
    conv_id = ""
    print(f"loom chat — model: {model}  (via {LOOM_URL})")
    print("type /quit to exit\n")

    while True:
        try:
            user_input = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input.strip():
            continue
        if user_input.strip() == "/quit":
            break

        messages.append({"role": "user", "content": user_input})

        headers = {"Content-Type": "application/json"}
        if conv_id:
            headers["X-Loom-Conversation-ID"] = conv_id

        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{LOOM_URL}/v1/chat/completions",
                headers=headers,
                json={"model": model, "messages": messages, "stream": True},
            ) as resp:
                conv_id = resp.headers.get("X-Loom-Conversation-ID", conv_id)
                full = []
                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    chunk_str = line[6:]
                    if chunk_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_str)
                        delta = (chunk.get("choices") or [{}])[0].get("delta", {})
                        c = delta.get("content", "")
                        if c:
                            full.append(c)
                            print(c, end="", flush=True)
                    except json.JSONDecodeError:
                        pass

        messages.append({"role": "assistant", "content": "".join(full)})
        print()


if __name__ == "__main__":
    main()
