# lamgate backend — prepared infrastructure

This directory is deliberately **outside the 2–3 hour implementation exercise**. The video starts with an OpenAI-compatible model endpoint already available at `http://127.0.0.1:8000/v1` through an SSH tunnel.

One-time setup after the repo exists on `lamgate`:

```bash
ssh lamgate
cd ~/agent-ground-up
export HF_TOKEN=...        # only if the model requires it
export VLLM_TP=1           # set to the number of GPUs assigned to the model
./infra/lamgate/bootstrap.sh
```

`bootstrap.sh` pulls the pinned vLLM image, starts the Docker service, and waits for `/v1/models` to become healthy.

On the laptop, keep the tunnel open:

```bash
./infra/lamgate/tunnel.sh
```

`configs/lamgate.yaml` already points the agent at the tunneled endpoint. No Docker/vLLM/backend implementation belongs in the timed reconstruction.
