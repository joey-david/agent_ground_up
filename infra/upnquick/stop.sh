#!/usr/bin/env bash
# Stop the backend and hand the shared GPU back (run this ON the node).
set -uo pipefail

PORT="${PORT:-8011}"

# Bracketing one character keeps the pattern from matching the shell that is running this
# script, which would otherwise kill the SSH command before it does any work.
pattern="--port ${PORT%?}[${PORT#${PORT%?}}]"

pkill -f -- "$pattern" 2>/dev/null || true
sleep 10

if pgrep -f -- "$pattern" >/dev/null 2>&1; then
  >&2 echo "backend still running on port ${PORT}"
  exit 1
fi

echo "backend stopped; current GPU state:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
