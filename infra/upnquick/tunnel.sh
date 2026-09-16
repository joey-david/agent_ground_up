#!/usr/bin/env bash
# Forward the laptop's local port to the backend running on a lamsade GPU node.
#
# The GPUs are on the compute nodes, not on the upnquick login host, so the forward targets the
# node's loopback and reaches it through the ProxyJump defined in ~/.ssh/config.
set -euo pipefail

LOCAL_PORT="${LOCAL_PORT:-8020}"
REMOTE_HOST="${REMOTE_HOST:-upnquick}"
REMOTE_PORT="${REMOTE_PORT:-8011}"

# A local port that is already taken makes ssh bind only one address family (or nothing at all)
# while still appearing to succeed, so the agent silently talks to whatever else is listening.
# Fail loudly instead.
if lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  >&2 echo "local port ${LOCAL_PORT} is already in use:"
  >&2 lsof -nP -iTCP:"${LOCAL_PORT}" -sTCP:LISTEN
  >&2 echo "pick another with LOCAL_PORT=... and point model.base_url at it"
  exit 1
fi

echo "forwarding 127.0.0.1:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT}"

# A long sweep keeps the forward busy for hours, and a silent drop surfaces as
# APIConnectionError in the middle of an episode rather than as an ssh error. Keepalives
# detect a dead peer, and ExitOnForwardFailure refuses to sit there forwarding nothing.
exec ssh -N \
  -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -o ExitOnForwardFailure=yes \
  -L "127.0.0.1:${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${REMOTE_HOST}"
