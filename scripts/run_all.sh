#!/usr/bin/env bash
# Start Body, wait for its WebSocket port to accept connections, then start
# Brain in the foreground. Ctrl+C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${BODY_PORT:-8765}"

python -m body.main --port "$PORT" &
BODY_PID=$!
trap 'kill "$BODY_PID" 2>/dev/null || true' EXIT

# Poll the port instead of sleeping a fixed interval: model/URDF load time
# varies a lot between machines, and a fixed sleep is either too short
# (Brain fails to connect) or wasted time.
for _ in $(seq 1 100); do
  if python -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); sys.exit(s.connect_ex(('127.0.0.1', $PORT)))" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$BODY_PID" 2>/dev/null; then
    echo "Body exited before its WebSocket came up" >&2
    exit 1
  fi
  sleep 0.2
done

python -m brain.main --body-uri "ws://127.0.0.1:$PORT"
