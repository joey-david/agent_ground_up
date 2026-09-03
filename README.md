# Agent Ground Up — self-evolving kernel

The main research object is now a **recursively specializing agent**, not just a coding-agent
harness. Starting from the same frozen model, it selects a task family near its capability frontier,
mutates its own agent/tools/memory/skills/improvement policy, evaluates descendants on held-out
siblings, and retains a population archive rather than greedily overwriting the parent.

```text
task-family frontier
        ↓
population archive → descendant mutation → train evidence
        ↑                                  ↓
        └──────── held-out evaluation ─────┘
```

Core additions:

- `memory.py`: append-only lifetime memory with bounded wake context, raw recall, and zoomable
  hierarchical summaries.
- `skills.py`: persistent generated skills that execute through the same workspace bash boundary.
- `tasks.py`: task families and frontier curriculum.
- `evaluate.py`: explicit train/held-out evaluation and a local candidate runner.
- `archive.py`: immutable descendant snapshots with performance + novelty/exploration selection.
- `improve.py` + `evolve.py`: DGM-style select → mutate → evaluate → archive loop.
- `improvement_policy.md`: **editable by descendants**, so selected children change how future
  children are produced while the external held-out evaluator remains fixed.
- `video/`: exact on-camera implementation boundary. Tests and infrastructure stay off-camera.
- `learning/`: reconstruction drills, seed retrieval cards, and a LazyVim FSRS configuration.
- `evolution_tasks/smoke_curriculum.json`: tiny end-to-end smoke curriculum.

Run the new focused tests:

```bash
uv run pytest -q \
  tests/test_memory.py tests/test_skills.py tests/test_tasks.py \
  tests/test_evaluate.py tests/test_archive.py tests/test_improve.py tests/test_agent_memory.py
```

Run the evolutionary smoke loop against the configured model endpoint:

```bash
uv run python evolve.py \
  --curriculum evolution_tasks/smoke_curriculum.json \
  --rounds 3
```

See `video/README.md` for what to implement from memory and what should be prepared off-camera.

---

## Previous inference/training extension

The existing SFT → verifier-RL → merge/quantization stack is deliberately retained as an extension.
It is useful for later weight-level adaptation, but it is not part of the code you should memorize
or type during the self-evolving-agent videos.

# Agent Ground Up

Building an agentic harness and training the associated LLM. Mostly inspired from [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) and [codex](https://github.com/openai/codex).

**Key features**:

- [x] Minimal agentic scaffold for multimodal LLMs with tools for bash and image inspection (web interaction can be achieved via bash + a web browsing skill). Automatic compaction at 90%.
- [x] Custom inference and training code via a detached 2xA100 80 GB vLLM server. QLoRA RL pipeline via a remote environment and SOTA DAPO loss.
- [x] Highly customizable and adaptable via a general-purpose `config.yaml`.
- [x] TUI interface with capped Markdown, shell, and diff output. No web server or browser required.

### Repository structure

```text
agent_ground_up/
├── config.yaml             # customize to your own needs
├── run.py                  # hosted/local agent entry point
├── train.py                # SFT, RL, merge, and quantization stages
├── agent_ground_up/
│   ├── agent.py            # model loop and context compaction
│   ├── tools.py            # bash and image viewer
│   ├── ui.py               # capped Markdown, shell, and diff output
│   ├── loss.py             # custom Torch DAPO objective
│   ├── remote_env.py       # TRL to authenticated sandbox client
│   └── config.py           # YAML and secret loading
├── infra/
│   ├── sandbox_server.py   # disposable Docker envs
│   ├── start_sandbox.py
│   ├── start_vllm.py       # GPU RL rollout server
│   ├── vllm/               # isolated vLLM env
│   ├── quantize/           # isolated LLM Compressor env
│   └── tasks/              # task workspaces and verifiers
└── tests/                  # unit and Docker-sandbox tests
```

### Setup and checks

```bash
uv sync --extra dev --extra sandbox
uv run pytest -q
uv run ruff check . --exclude mini-swe-agent
```

Training on older Ampere nodes forced me to separate GPU processes into separate environments because the released Transformers-5 trainer, vLLM, and LLM Compressor dependency ranges don't resolve together:

```bash
uv sync --extra train
uv sync --project infra/vllm
uv sync --project infra/quantize
uv sync --extra sandbox                 # Mac only
```

You probably shouldn't merge the envs with `--no-deps` unless you know what you're doing.

### Video 1: implementation order

Target: 60 minutes. Keep tests and deployment infrastructure prepared off-camera.

| Time  | Implement         | Exact order                                                            |
| ----- | ----------------- | ---------------------------------------------------------------------- |
| 0–6   | Tablet diagram    | task → model → tools → observations; compaction; SFT → RL → W4A16      |
| 6–10  | Tree and YAML     | `pyproject.toml`, `config.yaml`, `config.py`                           |
| 10–23 | `tools.py`        | results → schemas → bash timeout → image viewer → head/tail truncation |
| 23–43 | `agent.py`        | prompts → loop → model request → dispatch → trajectories               |
| 43–50 | Compaction        | native history → checkpoint → canonical state + recent user turns      |
| 50–56 | `ui.py`, `run.py` | capped Rich renderer → Markdown/bash/diff → wire dependencies          |
| 56–60 | Demo              | repository edit → image task → forced compaction; show prepared tests  |

#### Lines worth explaining

- `Toolbox.bash`: every call starts in the workspace; filesystem state persists, shell-local state does not.
- `os.killpg`: a timeout kills child processes too.
- `_truncate`: binary-search tokenizer-sized head and tail; do not promise a token budget and slice characters.
- `if not tool_calls`: this is the only completion condition.
- `_trajectory_messages`: image bytes never enter saved trajectories.
- `_prompt_tokens`: compaction fails closed if the exact model processor cannot render the request.
- `_maybe_compact`: append the instruction to native history, then rebuild state plus a checkpoint and recent users.
- `crop_middle`: every displayed string keeps its beginning and end; Rich owns Markdown and syntax highlighting.

The checkpoint starts with the active plan and summarizes only episodic history. Old checkpoints are excluded; working directory, environment facts, tools, and live `AGENTS.md` instructions are recomputed. Recent user turns remain byte-for-byte intact under `agent.recent_user_tokens`.

### Run the Video 1 agent

Edit only these YAML fields:

```yaml
model:
  processor: Qwen/Qwen3.6-27B
  served_name: Qwen/Qwen3.6-27B
  base_url: https://YOUR-ENDPOINT/v1
agent:
  workdir: /tmp/agent-demo
  trajectory: runs/video1.json
  recent_user_tokens: 12000
ui:
  max_lines: 40
```

Then:

```bash
export API_KEY=...
uv run python run.py 'Fix the failing test and verify it.'
```

Use a disposable repository for the first run. For an image demo, place a PNG in the configured workspace and ask the agent to inspect it. To force compaction, temporarily set `model.context_window: 4096` and give it a task with substantial shell output. Confirm `result.compactions >= 1` in the trajectory.

The hosted endpoint must support OpenAI chat completions, tool calls, and structured `image_url` content. Using a text-only provider would require a second image channel and is out of scope.

### Video 2: implementation order

Target: 60 minutes, with sandbox/container files prepared off-camera.

| Time  | Implement       | Exact order                                                                  |
| ----- | --------------- | ---------------------------------------------------------------------------- |
| 0–5   | Training story  | successful trajectories → SFT → verifier RL                                  |
| 5–14  | `remote_env.py` | reset → tools → WSS request → score → infrastructure masking                 |
| 14–25 | QLoRA SFT       | NF4 → language LoRA targets → assistant-only SFT                             |
| 25–43 | `loss.py`       | masks → sequence ratio → asymmetric clip → optional KL → token normalization |
| 43–50 | Trainer bridge  | subclass TRL `_compute_loss`; call the hand-written loss                     |
| 50–56 | Launch/package  | Mac sandbox + GPU-1 vLLM + GPU-0 trainer → merge/GPTQ W4A16                  |
| 56–60 | Time-lapse/demo | base vs SFT vs RL through unchanged `run.py`                                 |

#### The hand-written objective

`dapo_loss` implements:

```text
log r_i = mean_t(log pi_theta - log pi_old) over valid tokens
r_i     = exp(log r_i)
L_clip  = -min(r_i A_i, clip(r_i, 1-eps_low, 1+eps_high) A_i)
L_DAPO  = sum_valid_tokens(L_clip + beta KL) / total_valid_tokens
```

Sequence-level ratios suit sequence-level verifier rewards. Token-level normalization avoids giving every long trajectory the same total weight as a short one. `tool_mask` removes environment text from policy gradients. Optional reverse-KL uses the unbiased exponential estimator. Unit tests prove padding has zero gradient and KL cannot be enabled without reference log-probabilities.

TRL still owns generation, multimodal forwarding, reward-standardized advantages, optimizer steps, and distributed synchronization. `build_dapo_trainer()` replaces only the loss boundary. Pin TRL 1.8 because this is intentionally a small dependency on its prepared-batch interface.

### Data

SFT JSONL contains successful canonical `messages` conversations. Image tool responses use structured blocks with a dataset-accessible image path. Reject failed, secret-bearing, or schema-incompatible trajectories. Include long-history → checkpoint → successful-continuation examples.

RL JSONL is prompt-only and selects a trusted Mac task:

```json
{
  "task_id": "example",
  "prompt": [
    {
      "role": "user",
      "content": "Create answer.txt containing exactly: agent ready"
    }
  ]
}
```

A local task is `infra/tasks/<id>/task.json` plus optional `workspace/`:

```json
{
  "image": "python:3.12-slim",
  "observation": "Create /workspace/answer.txt containing exactly: agent ready",
  "verifier": "test \"$(cat answer.txt)\" = \"agent ready\"",
  "cpus": 1,
  "memory": "1g",
  "pids": 64
}
```

The image, verifier, and workspace are trusted curator inputs. Model commands run only inside a no-network container with no host bind mounts or Docker socket.

Use SWE-Gym, curated SWE-smith, Endless Terminals, and SWE-bench Multimodal development data for training. Reserve Terminal-Bench 2 and SWE-bench test tasks for evaluation.

### Start the Mac environment

Set `sandbox.tasks_dir`, `sandbox.public_url`, and limits in `config.yaml`. Keep the bearer token outside YAML:

```bash
export AGENT_ENV_TOKEN="$(openssl rand -hex 32)"
uv run --extra sandbox python infra/start_sandbox.py
```

Smoke-test locally:

```bash
uv run --extra sandbox python -m agent_ground_up.remote_env example \
  --command "printf 'agent ready' > answer.txt"
```

Expected: `reward=1.0`, followed by no `agent-ground-up-*` container in `docker ps -a`.

For the node, expose the service through an authenticated HTTPS/WSS tunnel and change only:

```yaml
sandbox:
  public_url: https://agent-env.example.net
```

If using Cloudflare Access, export `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` on the node. The client sends those headers automatically.

On macOS, Homebrew's `docker` formula is only the client. This project was verified with Colima as its daemon:

```bash
brew install colima
colima start --cpu 2 --memory 4 --disk 20
docker version
# Later: colima stop
```

The real local smoke test must end with `reward=1.0` and no output from:

```bash
docker ps -a --filter 'name=agent-ground-up-' --format '{{.Names}} {{.Status}}'
```

### Serve the Video 1 model on `upnquick`

`upnquick` has 2×A100 80 GB and driver 550.163.01. It can try CUDA-12.9 binaries through CUDA 12.x minor-version compatibility, but may lack features that require the native 575 driver. Use a clean environment and stop if the CUDA check fails:

```bash
uv venv ~/.venvs/qwen36-vllm --python 3.12
uv pip install --python ~/.venvs/qwen36-vllm/bin/python \
  'vllm==0.19.1' --torch-backend=cu129

~/.venvs/qwen36-vllm/bin/python - <<'PY'
import torch, vllm
print('torch', torch.__version__, 'runtime', torch.version.cuda, 'vllm', vllm.__version__)
print('gpus', torch.cuda.device_count(), torch.cuda.get_device_name(0))
PY
```

Serve both GPUs without exposing the port publicly:

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_LOGGING_LEVEL=WARNING \
~/.venvs/qwen36-vllm/bin/vllm serve Qwen/Qwen3.6-27B \
  --host 127.0.0.1 --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-uvicorn-access-log
```

From the Mac, keep this tunnel open:

```bash
ssh -N -L 8000:127.0.0.1:8000 upnquick
```

Set `model.base_url: http://127.0.0.1:8000/v1` and keep `model.served_name: Qwen/Qwen3.6-27B`, then run `API_KEY=EMPTY uv run python run.py 'your task'`.

### Train on 2×A100

No model-generated command is executed on the node. Disable completion/request logging:

```bash
export AGENT_ENV_TOKEN=...
export CF_ACCESS_CLIENT_ID=...
export CF_ACCESS_CLIENT_SECRET=...
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
```

First run one-step gates by setting `sft.max_steps: 1`, then `rl.max_steps: 1`.

SFT on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra train python train.py sft
```

Start generation on GPU 1 after setting `rollout_server.model: outputs/sft/final`:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_LOGGING_LEVEL=WARNING \
  uv run --project infra/vllm python -m infra.start_vllm
```

RL on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra train python train.py rl
```

During RL, GPU 0 trains while GPU 1 generates rollouts. The one-step gate must prove four remote rollouts, multimodal tool parsing, QLoRA backward, custom-loss metrics, and vLLM adapter synchronization.

Suggested full curriculum: 800 short tasks, 150 repository tasks, 50 long-horizon tasks; 70% ≤8K, 25% ≤16K, 5% ≤32K. Treat 64K as inference-only curriculum/evaluation and 128K/262K as small evaluation tails.

### Merge, quantize, and test Video 2

After updating the `merge` and `quantize` YAML sections:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --project infra/quantize python train.py merge
CUDA_VISIBLE_DEVICES=0 uv run --project infra/quantize python train.py quantize
```

The calibration JSONL needs 512 representative agent conversations in its `text` field. GPTQ produces W4A16 weights: INT4 storage with BF16/FP16 activations, supported by Ampere and vLLM. The vision modules stay unquantized.

After training and quantization, stop the trainer and rollout server, then serve the final checkpoint across both A100s:

```bash
CUDA_VISIBLE_DEVICES=0,1 VLLM_LOGGING_LEVEL=WARNING \
~/.venvs/qwen36-vllm/bin/vllm serve outputs/agent-w4a16 \
  --host 127.0.0.1 --port 8000 \
  --served-model-name agent-w4a16 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --disable-uvicorn-access-log
```

Set `model.served_name: agent-w4a16` and `model.base_url: http://127.0.0.1:8000/v1`. Keep `model.processor: Qwen/Qwen3.6-27B` so the Mac can render and count the remote model's requests, then rerun the Video 1 task unchanged.

Compare base, SFT, and RL with identical prompts and sampling. Record verifier pass rate, valid dispatches, premature finishes, timeouts, image success, compaction continuity, and steps/tokens per success. Ship RL only if task success improves without reducing tool validity or regressing multimodal/long-context groups.

### What is verified here

```bash
uv run pytest -q
uv run ruff check . --exclude mini-swe-agent
python3 -m py_compile agent_ground_up/*.py run.py train.py infra/*.py
```

Unit tests cover shell results/timeouts/truncation, image confinement, tool-loop completion, forced compaction, invalid calls, trajectory sanitization, remote reward handling, and the Torch loss.

External checks still required: hosted Qwen compatibility, authenticated WSS, one-step QLoRA/vLLM synchronization, Qwen3.6 GPTQ conversion, two-A100 W4A16 serving, and benchmark results.
