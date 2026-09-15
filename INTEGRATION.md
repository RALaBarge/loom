# Loom Integration Guide

Integrate Loom with your favorite LLM harness. Loom works as:
- **HTTP service** (REST API)
- **MCP server** (standard tools)
- **Hook receiver** (automatic logging)

## Quick Start

```bash
# 1. Install Loom
pip install -e .

# 2. Start the server
python -m loom serve

# 3. Integrate with your harness (see below)
```

## Claude Code

### Option A: Automatic logging (recommended)

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "python3 ~/.loom/hooks/loom_interaction_log.py 2>/dev/null || true",
            "timeout": 15,
            "async": true,
            "env": {
              "LOOM_HARNESS": "claude-code"
            }
          }
        ]
      }
    ]
  }
}
```

Every Claude Code session automatically logs to Loom.

### Option B: MCP tools

Add to Claude Code settings:

```json
{
  "mcpServers": {
    "loom": {
      "command": "python3",
      "args": ["/path/to/loom/mcp_server.py"]
    }
  }
}
```

Claude Code can now search Loom via MCP tools.

## Pi

### Automatic logging

Set environment variables and add hook:

```bash
export LOOM_HARNESS=pi
export LOOM_URL=http://localhost:9741
```

Configure Pi to run hook at session end:
```
hook:session_end = python3 ~/.loom/hooks/loom_interaction_log.py
```

## OpenCode

### Automatic logging

```bash
export LOOM_HARNESS=opencode
```

Add to OpenCode config:
```
onSessionEnd: [
  {command: "python3 ~/.loom/hooks/loom_interaction_log.py"}
]
```

## Continue.dev (IDE plugin)

### MCP integration

In `.continue/config.json`:

```json
{
  "models": [...],
  "mcpServers": {
    "loom": {
      "command": "python3",
      "args": ["/path/to/loom/mcp_server.py"],
      "env": {
        "LOOM_DB_PATH": "~/.loom/loom.db"
      }
    }
  }
}
```

Continue will have access to Loom search and conversation tools.

## Custom Harness

If your harness:
1. **Provides a transcript** with newline-delimited JSON messages
2. **Supports hooks** (shell commands at session end)

Then integrate like Pi/OpenCode:

```bash
# Set transcript path and harness name
export TRANSCRIPT_PATH=/path/to/transcript.jsonl
export LOOM_HARNESS=my_harness

# Add to your harness config
hook:on_session_end = python3 ~/.loom/hooks/loom_interaction_log.py
```

The hook will auto-detect your transcript format and send to Loom.

## Direct HTTP API

If your harness can make HTTP requests, POST directly to `/log`:

```bash
curl -X POST http://localhost:9741/log \
  -H "Content-Type: application/json" \
  -d '{
    "request": {"messages": [{"role": "user", "content": "hello"}]},
    "response": {"choices": [{"message": {"content": "hi"}}]},
    "metadata": {"harness": "my_app", "model": "gpt-4"}
  }'
```

## Search from Command Line

Once running:

```bash
loom search "how do I deploy?"
loom conversations
loom conversation <id>
```

## Troubleshooting

**"Loom not found"**
- Make sure `python -m loom serve` is running
- Check `http://localhost:9741/health`

**"Hook not firing"**
- Check hook command syntax for your harness
- Verify `LOOM_URL` environment variable
- Test manually: `python3 ~/.loom/hooks/loom_interaction_log.py`

**"Embedding failed"**
- Make sure Ollama is running: `ollama serve`
- Make sure model is installed: `ollama pull nomic-embed-text`
- Check `LOOM_OLLAMA_URL` environment variable

**"No results in search"**
- Interactions are logged but may not be embedded yet
- Check `loom stats` to see embedding count
- Give Ollama time to embed large batches
