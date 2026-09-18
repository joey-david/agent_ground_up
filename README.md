# Agent Ground Up

A small research harness for studying long-horizon agent behavior, accumulated experience, and recursive improvement. It now runs **Prism's Ternary Bonsai 2 27B directly in-process with MLX**: no vLLM server, OpenAI compatibility layer, SSH tunnel, or remote coding service.

The core loop is deliberately simple:

```text
Bonsai 2 / MLX
      ↓  streamed tokens
    Agent ──→ tools
      ↑        ↓
      └── observations
```

Memory, exact experience logs, generated skills, held-out evaluation, and the descendant archive sit around that loop rather than inside a serving stack.

## Setup

Apple Silicon is required for this runtime. Create the environment and install the exact versions used by the published Bonsai pack:

```bash
uv sync --extra local --extra dev
```

The first run downloads the ~8.6 GB checkpoint from Hugging Face if it is not already cached:

```bash
uv run python scripts/run.py 'Inspect this repository and summarize its architecture.'
```

Generation is streamed to the terminal as `mlx-vlm` yields it. The model is loaded once per process and stays resident in unified memory.

The default profile is `configs/local.yaml`. To use a local checkpoint directory instead of the Hub, change `model.id`; to run a trained adapter, set `model.adapter`.

## Training

The packed ternary base stays frozen. `scripts/train.py` attaches differentiable low-rank MLX branches to its packed language projections and optimizes only those adapters:

```bash
uv sync --extra local --extra train
uv run python scripts/train.py --steps 20
```

Training data is JSONL with either a `messages` field or `prompt` / `response` fields. The default objective is next-token cross entropy. Research losses are first-class rather than hidden in a trainer:

```bash
uv run python scripts/train.py --loss my_experiment.losses:loss
```

A custom loss has the signature `loss(model, batch) -> scalar`. On a 16 GB Mac, keep sequences and adapter ranks modest; backpropagation needs substantially more working memory than inference.

## Recursive loop

```bash
uv run python scripts/evolve.py --rounds 3 --unsafe-local
```

`--unsafe-local` is explicit because descendants are executable code. The bundled evaluator is local and intended for trusted research fixtures; there is no remote sandbox service.

## Repository

```text
agent_ground_up/   agent, local model runtime, memory, tools, archive/evaluation
configs/           local runtime/training profile
scripts/           run, evolve, train
practice/          reconstruction exercises
tests/             kernel tests and smoke fixtures
docs/              implementation notes
```

## Validation

The portable kernel tests do not import MLX, so they also run in Linux CI:

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check . --exclude practice
```
