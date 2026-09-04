#!/usr/bin/env bash
set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8000}"
REMOTE_PORT="${REMOTE_PORT:-8000}"
exec ssh -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" lamgate
