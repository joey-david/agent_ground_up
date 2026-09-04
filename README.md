# Agent Ground Up

A compact reference implementation of a **recursively self-improving coding agent** plus a reconstruction curriculum for learning it from memory.

The research loop is:

```text
capability-frontier task family
            ↓
population archive → self-mutation → train evidence
        ↑                         ↓
        └──────── held-out evaluation
```

The agent combines modern long-horizon primitives—provider-native continuous state when available, append-only searchable experience, bounded semantic memory, generated skills—and a fixed held-out evaluator. The training path also exposes the LM internals manually: causal logits → chosen-token log-probs → policy ratios → clipping/KL → token-normalized loss.

## Repository

```text
agent_ground_up/   reference kernel and recursive-improvement code
configs/           runtime profiles (`lamgate.yaml` is the default)
docs/              video / implementation boundary
infra/             prepared backend, sandbox, rollout and packaging infrastructure
practice/          two resettable active-recall workspaces
scripts/           run / evolve / train entry points
tests/             reference tests and smoke fixtures
```

The root intentionally contains only project metadata plus those six purpose-specific directories. Old ad hoc attempts, demo images/text, duplicate requirements, and loose TODO files are removed.

## Run against lamgate

`infra/lamgate/` is prepared infrastructure and is **not part of the 2–3 hour implementation clock**. After the remote vLLM container is up, open the tunnel:

```bash
./infra/lamgate/tunnel.sh
API_KEY=EMPTY uv run python scripts/run.py 'Fix the failing test and verify it.'
```

The default profile is `configs/lamgate.yaml`. The optional provider-native continuous-state profile is `configs/astra.yaml`.

## Recursive smoke loop

```bash
API_KEY=EMPTY uv run python scripts/evolve.py --rounds 3 --unsafe-local
```

The bundled local candidate runner is for trusted smoke fixtures only; serious self-modified descendants should execute as whole processes inside an external/container boundary.

## Manual LM objective

`agent_ground_up/loss.py` explicitly implements the causal shift, `log_softmax`, sampled-token gather, importance ratio, asymmetric clipping, optional reverse-KL, and active-token normalization. TRL remains rollout/distribution/optimizer plumbing rather than hiding the core objective.

## Practice / replication

- `practice/implementation/`: signatures and types are supplied; implement bodies until tests pass.
- `practice/signatures/`: files are empty; reconstruct signatures/types only, with AST tests.
- both include YAML/TOML recall tasks.

See `practice/README.md` and `docs/video-plan.md`.

## Validation

```bash
uv sync --extra dev
uv run pytest -q
uv run ruff check . --exclude practice
```
