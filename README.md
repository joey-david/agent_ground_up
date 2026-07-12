# Agent Ground Up

Two videos build one small multimodal coding agent:

1. Type the runtime and run Qwen3.6-27B through a hosted OpenAI-compatible endpoint.
2. Type its QLoRA SFT pipeline, remote environment, and DAPO loss; train on 2×A100; deploy NVFP4 on Blackwell.

The model sees only `bash` and `view_image`. Compaction is automatic at 90%. `config.yaml` is the interface for every stage; secrets remain environment variables.

## Shape

```text
run.py / train.py
        |
agent_ground_up/
  agent.py       model loop, compaction, trajectories
  tools.py       bash and image viewer
  loss.py        hand-written Torch DAPO objective
  remote_env.py  TRL -> authenticated Mac sandbox
  config.py      YAML loading

infra/
  sandbox_server.py   disposable Docker environments
  start_sandbox.py
  start_vllm.py
```

`mini-swe-agent/` is read-only reference material. We reuse its single query/execute idea and process-group timeout, not its framework layers or magic completion command.

## Setup and checks

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check . --exclude mini-swe-agent
```

GPU processes have separate environments because the released Transformers-5 trainer, vLLM, and LLM Compressor dependency ranges do not currently resolve together:

```bash
uv sync --extra train
uv sync --project infra/vllm
uv sync --project infra/quantize
uv sync --extra sandbox                 # Mac only
```

Do not merge these environments with `--no-deps`.

## Video 1: implementation order

Target: 80–90 minutes, about 300 live-written lines.

| Time | Implement | Exact order |
|---|---|---|
| 0–6 | Tablet diagram | task → model → tools → observations; add automatic compaction; zoom out to SFT → RL → NVFP4 |
| 6–12 | Tree and config | `pyproject.toml`, `config.yaml`, `config.py` |
| 12–30 | `tools.py` | result dataclasses → schemas → `Toolbox` → bash timeout → image validation → token head/tail truncation |
| 30–55 | `agent.py` | prompts → `RunResult` → loop → model request → tool dispatch → safe trajectory |
| 55–70 | Compaction | exact processor token count → 90% trigger → tools-off summary → rebuild history |
| 70–78 | `run.py` | load YAML → processor/client/tools/agent → run |
| 78–90 | Tests and demo | fake-model tests → repository edit → image task → forced compaction |

### Lines worth explaining

- `Toolbox.bash`: every call starts in the workspace; filesystem state persists, shell-local state does not.
- `os.killpg`: a timeout kills child processes too.
- `_truncate`: binary-search tokenizer-sized head and tail; do not promise a token budget and slice characters.
- `if not tool_calls`: this is the only completion condition.
- `_safe`: image bytes and credentials never enter saved trajectories.
- `_prompt_tokens`: compaction fails closed if the exact model processor cannot render the request.
- `_maybe_compact`: preserve the immutable task plus one continuation checkpoint; compaction is not a tool.

The system prompt stays short: inspect, edit, test, recover from tool failures, prefer minimal changes, and finish only with evidence. The checkpoint prompt explicitly preserves state, constraints, files, command results, failures, and next actions.

## Run the Video 1 agent

Edit only these YAML fields:

```yaml
model:
  name: Qwen/Qwen3.6-27B
  base_url: https://YOUR-ENDPOINT/v1
agent:
  workdir: /tmp/agent-demo
  trajectory: runs/video1.json
```

Then:

```bash
export API_KEY=...
uv run python run.py 'Fix the failing test and verify it.'
```

Use a disposable repository for the first run. For an image demo, place a PNG in the configured workspace and ask the agent to inspect it. To force compaction, temporarily set `model.context_window: 4096` and give it a task with substantial shell output. Confirm `result.compactions >= 1` in the trajectory.

The hosted endpoint must support OpenAI chat completions, tool calls, and structured `image_url` content. Using a text-only provider would require a second image channel and is out of scope.

## Video 2: implementation order

Target: 80–90 minutes, with sandbox/container files prepared off-camera.

| Time | Implement | Exact order |
|---|---|---|
| 0–8 | Training story | successful trajectories → SFT → verifier RL; explain why GRPO alone cannot discover the interface efficiently |
| 8–20 | `remote_env.py` | reset → typed tools → WSS request → score → infrastructure masking |
| 20–35 | QLoRA SFT | NF4 config → language-only LoRA targets → assistant-only SFT |
| 35–58 | `loss.py` | masks → sequence ratio → asymmetric clipping → advantages → optional KL → DAPO token normalization → metrics |
| 58–68 | Trainer bridge | subclass TRL `_compute_loss`; let TRL supply rollouts/log-probs, then call our loss |
| 68–76 | Remote launch | Mac sandbox + GPU-1 vLLM + GPU-0 trainer |
| 76–82 | Merge/NVFP4 | merge adapter → representative calibration → compress language layers |
| 82–90 | Time-lapse/demo | base vs SFT vs RL using unchanged `run.py` |

### The hand-written objective

`dapo_loss` implements:

```text
log r_i = mean_t(log pi_theta - log pi_old) over valid tokens
r_i     = exp(log r_i)
L_clip  = -min(r_i A_i, clip(r_i, 1-eps_low, 1+eps_high) A_i)
L_DAPO  = sum_valid_tokens(L_clip + beta KL) / total_valid_tokens
```

Sequence-level ratios suit sequence-level verifier rewards. Token-level normalization avoids giving every long trajectory the same total weight as a short one. `tool_mask` removes environment text from policy gradients. Optional reverse-KL uses the unbiased exponential estimator. Unit tests prove padding has zero gradient and KL cannot be enabled without reference log-probabilities.

TRL still owns generation, multimodal forwarding, reward-standardized advantages, optimizer steps, and distributed synchronization. `build_dapo_trainer()` replaces only the loss boundary. Pin TRL 1.8 because this is intentionally a small dependency on its prepared-batch interface.

## Data

SFT JSONL contains successful canonical `messages` conversations. Image tool responses use structured blocks with a dataset-accessible image path. Reject failed, secret-bearing, or schema-incompatible trajectories. Include long-history → checkpoint → successful-continuation examples.

RL JSONL is prompt-only and selects a trusted Mac task:

```json
{"task_id":"example","prompt":[{"role":"user","content":"Create answer.txt containing exactly: agent ready"}]}
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

## Start the Mac environment

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

## Train on 2×A100

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

Start generation on GPU 1 after setting `vllm.model: outputs/sft/final`:

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_LOGGING_LEVEL=WARNING \
  uv run --project infra/vllm python -m infra.start_vllm
```

RL on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --extra train python train.py rl
```

The one-step RL gate must prove four remote rollouts, multimodal tool parsing, QLoRA backward, custom-loss metrics, and vLLM adapter synchronization. If it fails, stop; do not silently reduce the model or context claim.

Suggested full curriculum: 800 short tasks, 150 repository tasks, 50 long-horizon tasks; 70% ≤8K, 25% ≤16K, 5% ≤32K. Treat 64K as inference-only curriculum/evaluation and 128K/262K as small evaluation tails.

## Merge, quantize, and test Video 2

After updating the `merge` and `quantize` YAML sections:

```bash
uv run --project infra/quantize python train.py merge
uv run --project infra/quantize python train.py quantize
```

The calibration JSONL needs a `text` field containing representative agent conversations. NVFP4 is deployment precision, not training precision. Validate the result on Blackwell; Qwen3.6 hybrid-attention compression support is a hard release gate.

Serve the checkpoint with an OpenAI-compatible vLLM server, change `model.name` and `model.base_url`, and rerun the same Video 1 task with unchanged agent code.

Compare base, SFT, and RL with identical prompts and sampling. Record verifier pass rate, valid dispatches, premature finishes, timeouts, image success, compaction continuity, and steps/tokens per success. Ship RL only if task success improves without reducing tool validity or regressing multimodal/long-context groups.

## What is verified here

```bash
uv run pytest -q
uv run ruff check . --exclude mini-swe-agent
python3 -m py_compile agent_ground_up/*.py run.py train.py infra/*.py
```

Unit tests cover shell results/timeouts/truncation, image confinement, tool-loop completion, forced compaction, invalid calls, trajectory sanitization, remote reward handling, and the Torch loss.

External checks still required: hosted Qwen compatibility, Docker lifecycle, authenticated WSS, one-step QLoRA/vLLM synchronization, NVFP4 conversion, Blackwell serving, and benchmark results.
