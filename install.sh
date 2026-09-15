#!/bin/bash
set -e

# Loom installation script
# Installs Loom and configures it for supported harnesses

LOOM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOOM_HOME="${LOOM_HOME:-$HOME/.loom}"
LOOM_DB="$LOOM_HOME/loom.db"
LOOM_PORT="${LOOM_PORT:-9741}"

echo "🧵 Installing Loom..."

# Create Loom home directory
mkdir -p "$LOOM_HOME"

# Install Python dependencies
pip install -q -r "$LOOM_DIR/requirements.txt"

# Create .env if doesn't exist
if [ ! -f "$LOOM_DIR/.env" ]; then
    cp "$LOOM_DIR/.env.example" "$LOOM_DIR/.env"
    echo "✓ Created .env (configure as needed)"
fi

# Create systemd service (Linux)
if command -v systemctl &> /dev/null; then
    SERVICE_FILE="$HOME/.config/systemd/user/loom.service"
    mkdir -p "$(dirname "$SERVICE_FILE")"
    cat > "$SERVICE_FILE" << EOF
[Unit]
Description=Loom Memory Server
After=network.target

[Service]
Type=simple
ExecStart=$LOOM_DIR/run.py serve
Restart=always
RestartSec=10
Environment="LOOM_DB_PATH=$LOOM_DB"
Environment="LOOM_PORT=$LOOM_PORT"

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    echo "✓ Created systemd service (enable with: systemctl --user enable --now loom)"
fi

# Configure Claude Code
if [ -f "$HOME/.claude/settings.json" ]; then
    echo "✓ Claude Code detected"
    echo "  Add this to ~/.claude/settings.json hooks.SessionEnd:"
    cat << 'EOF'
{
  "type": "command",
  "command": "python3 ~/.loom/loom_interaction_log.py 2>/dev/null || true",
  "timeout": 15,
  "async": true
}
EOF
fi

# Configure MCP (for harnesses supporting MCP)
if [ -d "$HOME/.config/claude" ]; then
    echo "✓ MCP support available"
    echo "  Add to your Claude config:"
    cat << EOF
"mcpServers": {
  "loom": {
    "command": "python3",
    "args": ["$LOOM_DIR/mcp_server.py"]
  }
}
EOF
fi

echo ""
echo "✅ Loom installed!"
echo ""
echo "Next steps:"
echo "  1. Start Loom: python -m loom serve"
echo "  2. Configure your harness (see options above)"
echo "  3. Search: loom search 'your query'"
echo ""
