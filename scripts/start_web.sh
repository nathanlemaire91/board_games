#!/usr/bin/env bash
# Starts the game server (core/server.py, Awale by default) and the Cloudflare tunnel publishing it
# at https://my-board-games.com, both in the background, logging to server.log and tunnel.log at the
# repository root.
#
#     scripts/start_web.sh [--restart] [server arguments, such as --game]
#
# A running server or tunnel is left as is; --restart stops the server first, to pick up code or
# weight changes (the tunnel keeps running and reconnects to the new server). The tunnel's settings
# are in ~/.cloudflared/config.yml. Stop both with: pkill -f core.server; pkill -f "cloudflared tunnel run"
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SERVER='python3? -m core\.server' # uv's process and Python's

if [ "${1:-}" = "--restart" ]; then
    shift
    if pkill -f "$SERVER"; then
        for _ in $(seq 20); do pgrep -f "$SERVER" >/dev/null || break; sleep 0.5; done
        pkill -KILL -f "$SERVER" || true # Still there after 10 s
        echo "server stopped"
    fi
fi

if pgrep -f "$SERVER" >/dev/null; then
    echo "server already running"
else
    # localhost is enough for the tunnel; pass --host 0.0.0.0 to also serve the local network
    (cd "$ROOT" && setsid nohup uv run python -m core.server "$@" >"$ROOT/server.log" 2>&1 &)
fi

if pgrep -f "cloudflared tunnel run" >/dev/null; then
    echo "tunnel already running"
else
    setsid nohup "$HOME/.local/bin/cloudflared" tunnel run board-games >"$ROOT/tunnel.log" 2>&1 &
fi

for _ in $(seq 60); do grep -q " at http" "$ROOT/server.log" 2>/dev/null && break; sleep 0.5; done
grep -q " at http" "$ROOT/server.log" || { echo "server did not start, see server.log"; exit 1; }
echo "https://my-board-games.com/$(cat "$ROOT/.server_token")/"
