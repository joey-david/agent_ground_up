#!/usr/bin/env bash
# Start the vLLM backend natively on a lamsade GPU node (run this ON the node, e.g. upnquick).
#
# The lamsade compute nodes have no container runtime and the account is unprivileged, so the
# backend runs directly from the prepared virtualenv rather than from compose.yaml.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
PORT="${PORT:-8011}"
GPUS="${GPUS:-0}"                 # single A100 by default; these cards are shared
VENV="${VENV:-$HOME/.venvs/qwen36-vllm}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
RUNDIR="${RUNDIR:-$HOME/tmp/agent-vllm}"

if [ ! -x "$VENV/bin/vllm" ]; then
  >&2 echo "no vllm in $VENV -- use an existing node venv, do not create or install one"
  exit 1
fi

mkdir -p "$RUNDIR"

# Match on the port rather than on the model name: a pattern containing the launch command
# also matches the shell that is running this script, which yields a false "already running".
if pgrep -f -- "--port ${PORT}" >/dev/null 2>&1; then
  >&2 echo "something is already serving on port ${PORT}; refusing to start a second copy"
  exit 1
fi

CUDA_VISIBLE_DEVICES="$GPUS" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
  nohup "$VENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" \
  --tensor-parallel-size "$(printf '%s' "$GPUS" | awk -F, '{print NF}')" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --disable-uvicorn-access-log \
  > "$RUNDIR/server.log" 2>&1 &

echo "launched pid $! on GPU(s) ${GPUS}, log: $RUNDIR/server.log"

# Loading ~52 GB of weights off NFS takes several minutes on a cold cache.
for _ in $(seq 1 90); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "backend ready on 127.0.0.1:${PORT}"
    exit 0
  fi
  sleep 10
done

tail -40 "$RUNDIR/server.log"
>&2 echo "backend failed health check"
exit 1
