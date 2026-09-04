# lamgate backend — prepared infrastructure

This directory is deliberately **outside the 2–3 hour implementation exercise**. The video starts with an OpenAI-compatible model endpoint already available at `http://127.0.0.1:8000/v1` through an SSH tunnel.

One-time server setup after copying this directory to `lamgate`:

```bash
ssh lamgate
cd ~/agent-ground-up/infra/lamgate
export HF_TOKEN=...        # only if the model requires it
export VLLM_TP=1           # set to the number of GPUs used by the model
docker compose pull
docker compose up -d
```

Check it remotely:

```bash
docker compose ps
curl http://127.0.0.1:8000/v1/models
```

On the laptop, keep the tunnel open:

```bash
./infra/lamgate/tunnel.sh
```

`configs/lamgate.yaml` already points the agent at the tunneled endpoint. No backend/container implementation belongs in the timed reconstruction.
