#!/usr/bin/env python3
"""Loom MCP Server — exposes search and conversation browsing as standard tools."""
import asyncio
import json
import os
import sys
from pathlib import Path

try:
    from .main import _embed
    from .store import Store
except ImportError:
    from main import _embed
    from store import Store

DB_PATH = str(Path(os.environ.get("LOOM_DB_PATH", str(Path.home() / ".loom" / "loom.db"))).expanduser().resolve())
store = None


async def search_memories(query: str, limit: int = 5, role: str = None) -> dict:
    """Semantic search over stored LLM interactions."""
    global store
    if not store:
        store = Store(DB_PATH)

    if not query.strip():
        return {"results": [], "error": "query cannot be empty"}

    try:
        embedding = await _embed(query)
        if not embedding:
            return {"results": [], "error": "embedding failed"}

        results = store.search(embedding, limit=min(limit, 100), role=role)
        return {
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        return {"results": [], "error": str(e)}


async def list_conversations(limit: int = 10) -> dict:
    """Get recent conversations."""
    global store
    if not store:
        store = Store(DB_PATH)

    try:
        convs = store.get_conversations(limit=min(limit, 100))
        return {
            "conversations": convs,
            "count": len(convs)
        }
    except Exception as e:
        return {"conversations": [], "error": str(e)}


async def get_conversation(conversation_id: str) -> dict:
    """Get all messages in a specific conversation."""
    global store
    if not store:
        store = Store(DB_PATH)

    try:
        messages = store.get_messages(conversation_id)
        if not messages:
            return {"messages": [], "error": "conversation not found"}
        return {
            "id": conversation_id,
            "messages": messages,
            "count": len(messages)
        }
    except Exception as e:
        return {"messages": [], "error": str(e)}


async def get_stats() -> dict:
    """Get Loom storage statistics."""
    global store
    if not store:
        store = Store(DB_PATH)

    try:
        return store.get_stats()
    except Exception as e:
        return {"error": str(e)}


# MCP Tool Definitions
TOOLS = {
    "loom_search": {
        "description": "Search stored LLM interactions by meaning (semantic search)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for (e.g., 'how do I deploy?')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default 5)",
                    "default": 5
                },
                "role": {
                    "type": "string",
                    "enum": ["user", "assistant"],
                    "description": "Filter by message role (optional)"
                }
            },
            "required": ["query"]
        }
    },
    "loom_conversations": {
        "description": "List recent conversations",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Max conversations to return (default 10)",
                    "default": 10
                }
            }
        }
    },
    "loom_conversation": {
        "description": "Get all messages in a specific conversation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {
                    "type": "string",
                    "description": "The conversation ID"
                }
            },
            "required": ["conversation_id"]
        }
    },
    "loom_stats": {
        "description": "Get Loom storage statistics (conversation count, message count, total tokens)",
        "inputSchema": {
            "type": "object"
        }
    }
}


async def handle_tool_call(tool_name: str, tool_input: dict) -> str:
    """Handle MCP tool invocation."""
    if tool_name == "loom_search":
        result = await search_memories(
            query=tool_input.get("query", ""),
            limit=tool_input.get("limit", 5),
            role=tool_input.get("role")
        )
    elif tool_name == "loom_conversations":
        result = await list_conversations(limit=tool_input.get("limit", 10))
    elif tool_name == "loom_conversation":
        result = await get_conversation(tool_input.get("conversation_id", ""))
    elif tool_name == "loom_stats":
        result = await get_stats()
    else:
        result = {"error": f"unknown tool: {tool_name}"}

    return json.dumps(result, indent=2)


async def run_mcp_server():
    """Run stdio-based MCP server."""
    loop = asyncio.get_event_loop()

    while True:
        try:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break

            request = json.loads(line)
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")

            if method == "tools/list":
                response = {
                    "id": request_id,
                    "result": {"tools": list(TOOLS.values())}
                }
            elif method == "tools/call":
                tool_name = params.get("name")
                tool_input = params.get("arguments", {})
                result = await handle_tool_call(tool_name, tool_input)
                response = {
                    "id": request_id,
                    "result": {"content": [{"type": "text", "text": result}]}
                }
            else:
                response = {
                    "id": request_id,
                    "error": {"code": -32601, "message": f"unknown method: {method}"}
                }

            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError:
            pass
        except Exception as e:
            sys.stderr.write(f"error: {e}\n")
            sys.stderr.flush()


if __name__ == "__main__":
    asyncio.run(run_mcp_server())
