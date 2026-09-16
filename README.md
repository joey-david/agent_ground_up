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

## Running

The backend runs natively on a default compute node (2x A100 80GB, shared, unprivileged, no
container runtime); in my case, my laptop reaches it via an ssh tunnel, but this can all be adjusted fairly easily, e.g. to run via API.

In my case, `~/agent-ground-up` on the node can lag behind
this branch and may not contain `infra/upnquick/` at all, in which case the launcher below fails
with `No such file or directory`. Ship the two scripts once, then launch from where you put them:

```bash
scp infra/upnquick/serve.sh infra/upnquick/stop.sh upnquick:~/tmp/agent-vllm/
ssh upnquick 'cd ~/tmp/agent-vllm && WAIT=0 GPUS=0 PORT=8011 \
    RUNDIR=$HOME/tmp/agent-vllm ./serve.sh'
./infra/upnquick/tunnel.sh                       # 127.0.0.1:8020 -> upnquick:8011
```

`WAIT=0` matters: without it `serve.sh` polls for readiness for up to fifteen minutes and the SSH
command sits there for the whole model load (the symptom is an `ssh` that prints `launched pid ...`
and then returns nothing for ten-odd minutes). `WAIT=0` returns as soon as the server process is
spawned. Weights take a few minutes off NFS on a cold cache, so poll readiness through the tunnel
instead of over SSH, with `curl -s http://127.0.0.1:8020/v1/models`.

Pick a free GPU before launching: `nvidia-smi --query-gpu=index,memory.used --format=csv` on the
node: these cards are shared, and `GPUS=0` above claims exactly one of them.

**Always export `HF_HUB_OFFLINE=1`**: without it the processor load
blocks on a Hugging Face Hub network call and the run can sit for tens of minutes producing nothing
but an "unauthenticated requests" warning and the stop script reports what is left:

```bash
ssh upnquick 'PORT=8011 ~/tmp/agent-vllm/stop.sh; PORT=8012 ~/tmp/agent-vllm/stop.sh'
```

Failure modes: `infra/upnquick/README.md`. `--unsafe-local` runs candidate code on the host and
suits the bundled smoke fixtures only; real descendants belong inside the sandbox boundary.

## Benchmarking

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

`scripts/harness_ab.py` scores the harness itself rather than the prompt. It runs the same cases
through four arms — `kernel` (the full agent), `kernel-bare` (the same loop and tools with memory,
experience and skills removed), `mini` (a reimplementation of the mini-swe-agent protocol: no tool
API, one bash command per turn parsed out of a fenced block, linear history, no compaction) and
`oneshot` (a single completion, with the image inlined for a visual task — the floor a harness has
to beat to justify itself).

```bash
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/harness_ab.py \
    --split heldout,hard,hard2,probe --repeats 2 --temperature 0 \
    --endpoints http://127.0.0.1:8020/v1
```

It reports pass rate, mean steps, peak prompt tokens and terminal-status counts per arm, plus a
per-case matrix — the per-case rows are the useful part, since the arms differ on specific
capabilities rather than uniformly.

Pin `--temperature 0` for any A/B: the served default is nonzero, and one sample per cell cannot
tell a prompt effect from a dice roll. `--context-window` must stay comfortably above
`--max-tool-output-tokens`, or a single large observation overflows the window on its own.

`--max-output-tokens` matters more than it looks on a reasoning model. At the config default of
4096 this model spends the whole budget thinking about a hard problem and returns an *empty*
message with no tool calls, which the loop reads as "task complete": the episode ends at two steps
having written nothing. Those are reported as `empty_completion` rather than `completed`, and the
cure is a budget the model can actually finish a thought in (12288 works for contest problems).

### Benchmarks beyond the bundled curriculum

The bundled coding curriculum is saturated for a capable model — all three agent arms score in the
90s and the differences come down to two or three probe cases. Three adapters build harder, less
code-shaped benchmarks as ordinary workspaces plus a curriculum file, so `harness_ab.py` consumes
them unchanged. Each keeps its answer key *outside* the workspace the agent can read.

```bash
# contest programming: one uniform stdin/stdout contract, graded on hidden tests
uv run python benchmarks/livecodebench/build.py --difficulty hard --limit 12
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/harness_ab.py \
    --curriculum benchmarks/livecodebench/curriculum_hard.json --split lcb \
    --max-output-tokens 12288 --max-steps 15

# visual search: small targets in large photographs, where cropping and looking again pays
uv run python benchmarks/vstar/build.py --limit 48
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/harness_ab.py \
    --curriculum benchmarks/vstar/curriculum.json --split vstar \
    --arms oneshot,mini,kernel-bare,kernel

# household text games whose six task types recur, so procedures can be reused
alfworld-download                                  # ~2.2GB, into $ALFWORLD_DATA
uv run python benchmarks/alfworld/build.py --limit 18 \
    --data ~/.cache/alfworld --interpreter ~/.venvs/alfworld/bin/python
```

ALFWorld needs its own interpreter: it pulls in textworld and spacy, which have no business in the
agent's environment. Its games are stateful while the agent's shell is not, so each workspace holds
an `act.py` client over a workspace-local daemon that keeps one game open across commands.

### Measuring self-improvement

By default every episode starts blank, which makes memory, experience and skills untestable —
they exist to pay off *across* tasks. `--persistent-state DIR` carries each arm's subsystem state
across the whole split, runs the episodes one at a time in curriculum order, and prints a learning
curve comparing each arm's first half to its second.

```bash
API_KEY=EMPTY HF_HUB_OFFLINE=1 uv run python scripts/harness_ab.py \
    --curriculum benchmarks/alfworld/curriculum.json --split alfworld \
    --arms kernel --persistent-state runs/alfworld_state
```

Use a benchmark whose tasks genuinely repeat, and compare against the same arm run without the
flag: a rising second half only means something next to a flat one.

## Implementing it yourself

**A — Follow one task through the system.**

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

## Replicate it

Everything under `infra/` is prepared off-camera and sits outside the ~140–160 minute budget
(`docs/video-plan.md`).

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

2. **`practice/signatures/`** — the same files, empty. Write _only_ signatures and annotations;
   bodies may be `...`. Tests compare the AST against drill 1, so they check names, kinds, defaults
   and types without executing anything. Check with `uv run pytest -q practice/signatures/tests`.

3. **Blank directory, timed, `agent_ground_up/` unopened.**

Prompts in `practice/cards.md`, per-workspace notes in `practice/README.md`.

## Validation

```bash
uv sync --extra dev --extra sandbox
uv run pytest -q
uv run ruff check . --exclude practice
```
