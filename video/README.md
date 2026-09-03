# Video implementation boundary

The repository contains a complete reference implementation in `agent_ground_up/`. For the video,
**do not type the reference files verbatim**. Implement the conceptual pieces below from their
contracts, while tests, fixtures, model-serving setup, benchmark workspaces, and dependency glue stay
prepared off-camera.

## What you implement on camera

### Video A — the lifetime agent kernel (~60 min)

1. `tools.py` — bash process-group timeout, token head/tail truncation, image confinement
2. `memory.py` — append-only `remember`, bounded `wake`, regex `recall`, summary tree + `zoom`
3. `skills.py` — persistent generated-skill registry through the same bash boundary
4. `agent.py` — model/tool loop, dynamic schemas, token accounting, compaction, state reinjection
5. Demo — solve task, persist discovery, force compaction, start a second task and recall it

### Video B — recursive specialization (~60 min)

1. `tasks.py` — task families, explicit train/held-out siblings, frontier selection
2. `evaluate.py` — candidate runner protocol and held-out reports
3. `archive.py` — immutable snapshots, fingerprint/novelty, parent selection
4. `improve.py` — select → materialize → mutate → train-eval → heldout-eval → archive
5. `evolve.py` — wiring and demo

Draw this before code:

```text
frontier task family
       ↓
 archive parent ──→ mutate tools / memory / policy / agent
       ↑                         ↓
       └──── held-out eval ← descendant

persistent experimental memory spans rounds
```

`improvement_policy.md` is part of every descendant. A selected child therefore changes the
policy that produces its future children. The small archive/evaluation loop stays fixed: the system
may improve *how it improves* without being allowed to redefine the score.

## Prepared off-camera

Do not spend video time on:

- `tests/` and `evolution_tasks/`
- OpenAI-compatible endpoint / vLLM setup
- YAML parsing, TUI rendering, CLI plumbing
- `remote_env.py`, sandbox server, TRL, QLoRA, DAPO, merge/quantization
- benchmark downloads and third-party compatibility

Run prepared tests after each module:

```bash
uv run pytest -q tests/test_tools.py tests/test_memory.py tests/test_skills.py
uv run pytest -q tests/test_agent.py tests/test_agent_memory.py
uv run pytest -q tests/test_tasks.py tests/test_evaluate.py tests/test_archive.py
uv run pytest -q tests/test_improve.py
```

## Rules for a genuine reconstruction attempt

1. Start from an empty scratch file, not a half-erased reference file.
2. Write invariants/interfaces first in comments.
3. No reference peek until the relevant tests have failed at least once.
4. Record `time_to_green`, `reference_peeks`, and `tests_green_before_first_peek`.
5. Syntax/API lookup is allowed after the design has been recalled; source lookup counts as a peek.
6. A successful repetition means equivalent behavior, not byte-identical code.
