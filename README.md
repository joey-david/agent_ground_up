# Agent Ground Up

A compact reference implementation of a **recursively self-improving coding agent**, plus a
reconstruction curriculum for rebuilding it from memory.

```text
capability-frontier task family
            ↓
population archive → self-mutation → train evidence
        ↑                         ↓
        └──────── held-out evaluation
```

~2.5k lines of kernel: continuous provider state, append-only searchable experience, bounded
semantic memory, generated skills, a fixed held-out evaluator. `loss.py` derives the RL objective by
hand — causal shift → chosen-token log-probs → policy ratios → clipping/KL → token normalization —
so TRL stays rollout and optimizer plumbing instead of hiding the core.

```text
agent_ground_up/   kernel and recursive-improvement code
benchmarks/        verifiable coding tasks + system-prompt variants for prompt search
configs/           runtime profiles (`upnquick.yaml` is the default)
infra/             backend, sandbox, rollout, packaging — prepared, not reconstructed
practice/          two resettable recall workspaces
scripts/           run / evolve / train
tests/ docs/       reference tests; implementation boundary
```

## Run it

The backend runs natively on the `upnquick` compute node (2x A100 80GB, shared, unprivileged, no
container runtime); the laptop reaches it over an SSH tunnel.

**The node's checkout is not this working copy.** `~/agent-ground-up` on the node can lag behind
this branch and may not contain `infra/upnquick/` at all, in which case the launcher below fails
with `No such file or directory`. Ship the two scripts once, then launch from where you put them:

```bash
scp infra/upnquick/serve.sh infra/upnquick/stop.sh upnquick:~/tmp/agent-vllm/
ssh upnquick 'cd ~/tmp/agent-vllm && setsid nohup env GPUS=0 PORT=8011 \
    RUNDIR=$HOME/tmp/agent-vllm ./serve.sh </dev/null >serve.out 2>&1 &'
./infra/upnquick/tunnel.sh                       # 127.0.0.1:8020 -> upnquick:8011
```

`setsid nohup ... &` matters: `serve.sh` polls for readiness for up to fifteen minutes, so running
it in the foreground holds the SSH session open for the whole model load. Weights take a few
minutes off NFS on a cold cache; poll readiness through the tunnel with
`curl -s http://127.0.0.1:8020/v1/models`, not with more SSH.

A second card doubles throughput for evaluation sweeps. Give it its own port, run directory and
tunnel:

```bash
ssh upnquick 'cd ~/tmp/agent-vllm && setsid nohup env GPUS=1 PORT=8012 \
    RUNDIR=$HOME/tmp/agent-vllm2 ./serve.sh </dev/null >serve2.out 2>&1 &'
ssh -f -N -L 127.0.0.1:8022:127.0.0.1:8012 upnquick
```

Then drive it from the laptop. **Always export `HF_HUB_OFFLINE=1`**: without it the processor load
blocks on a Hugging Face Hub network call and the run can sit for tens of minutes producing nothing
but an "unauthenticated requests" warning, which looks exactly like a hung GPU.

```bash
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/run.py 'Fix the failing test and verify it.'
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/evolve.py --rounds 3 --unsafe-local
```

Hand both cards back when you are done; they are shared, and the stop script reports what is left:

```bash
ssh upnquick 'PORT=8011 ~/tmp/agent-vllm/stop.sh; PORT=8012 ~/tmp/agent-vllm/stop.sh'
```

Failure modes: `infra/upnquick/README.md`. `configs/astra.yaml` is the provider-native
continuous-state alternative. `--unsafe-local` runs candidate code on the host and suits the bundled
smoke fixtures only; real descendants belong inside the sandbox boundary.

## Measure it

`benchmarks/coding/` holds 41 verifiable tasks in five splits — `train`, `heldout`, `hard`, `hard2`
and `probe` — each a small workspace plus a verifier scored by exit code. `scripts/prompt_search.py`
runs prompt variants against them, one independent episode per (variant, case): fresh workspace,
fresh memory/experience/skill state, results written after every episode so an interrupted sweep is
still usable.

```bash
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/prompt_search.py \
    --split hard2 --repeats 2 --temperature 0 \
    --endpoints http://127.0.0.1:8020/v1,http://127.0.0.1:8022/v1

# ablate the compaction prompt instead, with a window small enough to make compaction fire
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/prompt_search.py \
    --vary compact --variants benchmarks/compact_variants.json \
    --split hard2 --context-window 8000 --max-tool-output-tokens 1200
```

Pin `--temperature 0` for any A/B: the served default is nonzero, and one sample per cell cannot
tell a prompt effect from a dice roll. `--context-window` must stay comfortably above
`--max-tool-output-tokens`, or a single large observation overflows the window on its own.

## Learn it

Three orders through the same ~2.5k lines. Pick by what you want out of it; each file is listed
where it first earns its keep.

**A — Follow one task through the system.** The recommended first pass: you watch the loop run
before reading anything that supports it.

1. `scripts/run.py` — how a config, a workspace, and a prompt become a live `Agent`.
2. `agent.py`, top half — `run` → `_run` → `_turn` → `_execute`. One loop drives both runtimes;
   everything else in the repo hangs off it. Read the tool schemas here too: they are the only
   description of the tools the model ever sees.
3. `tools.py` — the entire contact surface with the machine: one bash subprocess in its own
   process group, one image reader, one token budget.
4. `ui.py` — what a turn looks like as it happens. Optional, and the fastest way to make the
   loop legible while it runs.
5. `config.py`, `factory.py` — a YAML profile becomes a client, a processor, and a token counter.
6. `agent.py`, bottom half — `_canonical_state`, `_maybe_compact`, `_prompt_tokens`: what happens
   when the context fills up, which is the whole difficulty of long-horizon work.
7. `runtime.py` — the other way to hold state: replay provider-native items, including encrypted
   reasoning, and let the provider compact. Same loop, different memory of itself.

**B — Build it bottom-up.** Dependency order, each step runnable and testable on its own, and the
same order the drills use: `tools.py` → `memory.py` → `experience.py` → `skills.py` → `agent.py` →
`runtime.py`. The three storage layers are deliberately distinct and worth reading back to back:
`memory.py` is distilled and bounded, `experience.py` is exact and append-only, `skills.py` is
procedural and executable.

**C — The self-improvement half only.** Treat the agent loop as a black box that scores a task,
and read outward from the objective: `tasks.py` (families, splits, frontier) → `evaluate.py`
(fixed held-out scoring) → `archive.py` (immutable descendants, novelty) → `improve.py` (the
round) → `loss.py` (the clipped DAPO/GRPO objective, derived by hand) → `scripts/train.py`.

Whichever order you take, `tests/` is the specification. When a file stops making sense, read its
test first — that is what the reconstruction drills below are graded against.

## Replicate it

Everything under `infra/` is prepared off-camera and sits outside the ~140–160 minute budget
(`docs/video-plan.md`). You reconstruct the kernel, not the backend.

**First, make the reference green** so the tests are a trustworthy answer key:

```bash
uv sync --extra dev --extra sandbox && uv run pytest -q
```

**Then fill the templates in order.** Both workspaces reset with
`git restore practice/implementation practice/signatures`.

1. **`practice/implementation/`** — signatures, types and `NotImplementedError` bodies are given;
   write the bodies and the marked config values. Dependency order, each step usable on its own:
   `tools.py` (bash boundary, process-group timeout, output truncation) → `memory.py` (append-only
   memories, bounded wake context, summary-tree zoom) → `skills.py` (generated procedures over the
   same boundary) → `agent.py` (model/tool loop, dynamic tools, completion, persistent state) →
   `tasks.py`/`evaluate.py` (sibling train/held-out families, fixed evaluator) →
   `archive.py`/`improve.py` (immutable descendants, novelty selection, the round) → `loss.py` (the
   clipped DAPO/GRPO-style objective) → `config.yaml` (profile schema and values).
   Check with `uv run pytest -q practice/implementation/tests`.

2. **`practice/signatures/`** — the same files, empty. Write *only* signatures and annotations;
   bodies may be `...`. Tests compare the AST against drill 1, so they check names, kinds, defaults
   and types without executing anything. Check with `uv run pytest -q practice/signatures/tests`.

3. **Blank directory, timed, `agent_ground_up/` unopened.** The drills exist to make this boring.

Prompts in `practice/cards.md`, per-workspace notes in `practice/README.md`.

## Validation

```bash
uv sync --extra dev --extra sandbox
uv run pytest -q
uv run ruff check . --exclude practice
```
