"""Loom CLI: run the server, or search a running one."""
import json
import os
import sys

import httpx

LOOM_URL = os.environ.get("LOOM_URL", f"http://127.0.0.1:{os.environ.get('LOOM_PORT', '9741')}")
USAGE = """loom serve                 start the server
loom search <query>        search accepted memories
loom events <query>        search trusted event text
loom stats                 counts"""


def main():
    args = sys.argv[1:] or ["serve"]
    cmd, rest = args[0], " ".join(args[1:]).strip()
    if cmd == "serve":
        try:
            from .main import serve
        except ImportError:
            from main import serve
        serve()
    elif cmd in ("search", "events") and rest:
        path = "/memories/search" if cmd == "search" else "/events/search"
        r = httpx.post(LOOM_URL + path, json={"query": rest, "limit": 10}, timeout=600)
        for hit in r.json().get("results", []):
            where = hit.get("id") or f"{hit.get('session')}#{hit.get('seq')}"
            print(f"{hit.get('distance', 0):.3f}  {where}  {(hit.get('text') or '')[:200]}")
    elif cmd == "stats":
        print(json.dumps(httpx.get(LOOM_URL + "/stats", timeout=30).json(), indent=2))
    else:
        print(USAGE)
        sys.exit(0 if cmd in ("help", "-h", "--help") else 1)


if __name__ == "__main__":
    main()
