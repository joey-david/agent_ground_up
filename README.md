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
configs/           runtime profiles (`upnquick.yaml` is the default)
infra/             backend, sandbox, rollout, packaging — prepared, not reconstructed
practice/          two resettable recall workspaces
scripts/           run / evolve / train
tests/ docs/       reference tests; implementation boundary
```

## Run it

The backend runs natively on the `upnquick` compute node (2x A100 80GB, shared, unprivileged, no
container runtime); the laptop reaches it over an SSH tunnel.

```bash
ssh upnquick 'GPUS=0 ~/agent-ground-up/infra/upnquick/serve.sh'   # one A100; cards are shared
./infra/upnquick/tunnel.sh                                        # 127.0.0.1:8020 -> upnquick:8011
API_KEY=EMPTY uv run python scripts/run.py 'Fix the failing test and verify it.'
API_KEY=EMPTY uv run python scripts/evolve.py --rounds 3 --unsafe-local
ssh upnquick '~/agent-ground-up/infra/upnquick/stop.sh'           # hand the GPU back
```

Failure modes: `infra/upnquick/README.md`. `configs/astra.yaml` is the provider-native
continuous-state alternative. `--unsafe-local` runs candidate code on the host and suits the bundled
smoke fixtures only; real descendants belong inside the sandbox boundary.

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
