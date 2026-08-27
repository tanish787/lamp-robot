#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

python -m body.main &
BODY_PID=$!
trap "kill $BODY_PID" EXIT

sleep 2  # let Body's WebSocket server come up
python -m brain.main
