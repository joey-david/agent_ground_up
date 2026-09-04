#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
: "${VLLM_TP:=1}"
export VLLM_TP

docker compose pull
docker compose up -d

for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; then
    echo "lamgate backend ready on 127.0.0.1:8000"
    exit 0
  fi
  sleep 2
done

docker compose logs --tail=100 llm
>&2 echo "backend failed health check"
exit 1
