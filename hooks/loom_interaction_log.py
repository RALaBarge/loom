#!/usr/bin/env python3
"""Log LLM interactions to Loom for any hook-based harness.

This script reads from the harness transcript and sends the most recent
interaction to Loom's /log endpoint. It works with any harness that:
- Provides TRANSCRIPT_PATH environment variable
- Stores messages as newline-delimited JSON

Supported harnesses:
- Claude Code (CLAUDE_TRANSCRIPT_PATH)
- Pi (if available)
- OpenCode (if available)
- Any custom harness with compatible transcript format
"""
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

LOOM_URL = os.environ.get("LOOM_URL", "http://127.0.0.1:9741")
TIMEOUT = 10


def find_transcript_path():
    """Try to locate transcript from supported harnesses."""
    candidates = [
        os.environ.get("CLAUDE_TRANSCRIPT_PATH"),
        os.environ.get("PI_TRANSCRIPT_PATH"),
        os.environ.get("OPENCODE_TRANSCRIPT_PATH"),
        os.environ.get("TRANSCRIPT_PATH"),
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path

    return None


def read_latest_turn(transcript_path: str) -> dict | None:
    """Extract the most recent user+assistant turn from transcript."""
    try:
        with open(transcript_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return None

    if not lines:
        return None

    turns = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg = obj.get("message") or {}
        role = msg.get("role")

        if role in ("user", "assistant"):
            turns.append({
                "role": role,
                "content": msg.get("content", ""),
                "timestamp": obj.get("timestamp", "")
            })

        if len(turns) >= 2:
            break

    if len(turns) < 2:
        return None

    turns = list(reversed(turns))
    user_msg = next((t for t in turns if t["role"] == "user"), None)
    asst_msg = next((t for t in turns if t["role"] == "assistant"), None)

    if not user_msg or not asst_msg:
        return None

    return {
        "user": user_msg,
        "assistant": asst_msg
    }


def send_to_loom(turn: dict) -> bool:
    """POST the interaction to Loom /log endpoint."""
    payload = {
        "request": {
            "messages": [{"role": "user", "content": turn["user"]["content"]}]
        },
        "response": {
            "choices": [{"message": {"content": turn["assistant"]["content"]}}]
        },
        "metadata": {
            "provider": os.environ.get("LOOM_HARNESS", "unknown"),
            "model": os.environ.get("LOOM_MODEL", ""),
            "source": "harness-hook"
        }
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{LOOM_URL}/log",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            result = json.loads(resp.read().decode())
            return result.get("stored", False)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def main():
    """Find transcript, extract last turn, send to Loom."""
    transcript_path = find_transcript_path()
    if not transcript_path:
        return

    turn = read_latest_turn(transcript_path)
    if turn:
        send_to_loom(turn)


if __name__ == "__main__":
    main()
