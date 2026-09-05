# upnquick backend — prepared infrastructure

This directory is deliberately **outside the 2–3 hour implementation exercise**. The video starts
with an OpenAI-compatible model endpoint already reachable at `http://127.0.0.1:8020/v1` through an
SSH tunnel.

## Topology

The login host has no GPUs. The cards live on the compute nodes reached through it (`upnquick` has
2x A100 80GB; `kaisertrot`, `ourasi`, `coktailjet` and others carry smaller cards). So the backend
runs on `upnquick` and the tunnel forwards to that node's loopback, hopping through the login host
via the `ProxyJump` already configured in `~/.ssh/config`.

These nodes are shared with other researchers and the account is unprivileged. Two consequences
shape everything here:

- **No containers.** There is no Docker group membership and no rootless runtime, so the backend is
  started natively from a virtualenv that already exists on the node. `compose.yaml` is kept only as
  a reference for container-capable hosts.
- **No elevated permissions, and take only what you need.** `serve.sh` defaults to a single GPU.

## Start the backend (on the node)

```bash
ssh upnquick
GPUS=0 ~/agent-ground-up/infra/upnquick/serve.sh
```

`serve.sh` launches vLLM from `~/.venvs/qwen36-vllm` on the requested GPU(s), binds to the node's
loopback, and waits for `/v1/models`. Loading Qwen3.6-27B (~52 GB over NFS) takes several minutes on
a cold cache. Override `MODEL`, `PORT`, `GPUS`, `MAX_MODEL_LEN` or `GPU_MEMORY_UTILIZATION` as needed;
`GPUS="0,1"` sets tensor parallelism to match.

## Open the tunnel (on the laptop)

```bash
./infra/upnquick/tunnel.sh
```

Defaults forward `127.0.0.1:8020` to `upnquick:8011`, matching `model.base_url` in
`configs/upnquick.yaml`. The local side is deliberately not `8000`: that port is a common default for
locally running inference servers, and a collision there is unusually nasty — `ssh` can bind only one
address family while still appearing to succeed, so the agent silently talks to whichever local model
is listening instead of the cluster, with no error anywhere.

The script therefore refuses to start if the local port is already bound, and names the process
holding it. To use a different port, override it and point `model.base_url` at the same number:

```bash
LOCAL_PORT=8030 ./infra/upnquick/tunnel.sh
```

## Free the GPU when done (on the node)

```bash
ssh upnquick '~/agent-ground-up/infra/upnquick/stop.sh'
```

The cards are shared, so stop the server as soon as the run is finished and confirm the memory came
back. `stop.sh` prints `nvidia-smi` afterwards for that check.

No backend implementation belongs in the timed reconstruction.
