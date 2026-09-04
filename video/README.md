# Video implementation boundary

The repository contains a complete reference implementation in `agent_ground_up/`. The video still
has the same goal: build our own recursively self-improving agent using modern runtime, memory,
tool-generation, evaluation, and training ideas.

The hard constraint is now explicit: **the meaningful implementation must fit in 2–3 hours total**.
Target ~150–165 minutes of coding, not counting commands/tests running or explanation pauses.

Tests, fixtures, provider SDK compatibility glue, model-serving setup, benchmark workspaces, and
packaging stay prepared off-camera. The algorithmic pieces that demonstrate understanding stay on
camera.

## What you implement on camera

### Video A — lifetime agent kernel (~65–75 min)

1. `tools.py` core (~15 min)
   - bash subprocess in workspace
   - process-group timeout/kill
   - token-budget head/tail truncation
   - confined image read
2. `agent.py` core (~20–25 min)
   - model → tool → observation loop
   - tool dispatch and invalid-call recovery
   - dynamic tool schemas
   - bounded context / canonical-state reinjection
3. `memory.py` + `skills.py` core (~20 min)
   - append-only reusable memory
   - bounded wake + recall/zoom interface
   - persistent generated skills through the same bash boundary
4. Modern runtime seam (~5 min)
   - wire the agent to the small runtime interface
   - the provider-specific `runtime.py` adapter is prepared; do not spend the video retyping SDK object conversion
5. Demo (~10 min)
   - solve a task
   - persist a discovery / skill
   - cross a context boundary
   - reuse old experience on the next task

The important distinction to explain is:

```text
provider cognitive state   exact searchable history   distilled memory   generated skill
         │                         │                        │                │
         └─────────────────────────┴──────────┬─────────────┴────────────────┘
                                              ↓
                                         agent policy
```

### Video B — learning + recursive specialization (~75–90 min)

#### 1. Manual LM policy dynamics in `loss.py` (~20–25 min)

This is **not** delegated to TRL. Implement explicitly:

```text
model(input_ids).logits
        ↓ causal shift
log_softmax(vocabulary)
        ↓ gather sampled token
log πθ(a_t | s_t)
        ↓ subtract old log-prob
r_t = exp(log πθ - log πold)
        ↓
asymmetric clipped advantage
        ↓
optional sampled reverse-KL
        ↓
global active-token normalization
        ↓
loss.backward()
```

Functions to type:

- `completion_logps_from_logits`
- `dapo_loss`
- the small model-forward portion of `DAPOTrainer._compute_loss`

TRL may handle rollout collection, distributed batching, LoRA plumbing, and optimizer scheduling.
It must **not** hide the forward pass from logits to chosen-token probabilities or the policy loss.
The point is that a viewer can trace the gradient from a rewarded agent trajectory back into the
LM parameters.

#### 2. Recursive specialization (~45–55 min)

1. `tasks.py` essentials — task family + train/held-out siblings + frontier selection
2. `evaluate.py` essentials — candidate runner contract + held-out score
3. `archive.py` essentials — immutable snapshots + performance/novelty parent choice
4. `improve.py` core — select → materialize → mutate → train-eval → heldout-eval → archive
5. `evolve.py` — minimal wiring loop

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

#### 3. Demo (~10 min)

Show at least one full round:

```text
parent score → failure trace → source/tool change → held-out score → archive decision
```

If compute permits, show a tiny RL update separately: before/after chosen-token log-probability or
loss on one rewarded batch. The long training run itself should be precomputed.

## Typed-code budget

The target is roughly:

| Piece | Target typed lines | Time |
| --- | ---: | ---: |
| tools + agent core | 180–220 | 35–40 min |
| memory + skills | 90–120 | 20 min |
| manual LM log-probs + DAPO loss | 70–100 | 20–25 min |
| task/eval/archive/improve core | 180–220 | 45–55 min |
| wiring + demos | 40–60 | 20 min |
| **Total** | **~560–720** | **~140–160 min** |

The reference implementation can be larger because it includes validation and compatibility code.
The claim made by the video is that the **agent/retrieval/learning/self-improvement mechanisms** can
be reconstructed from scratch inside the time budget, not that you should retype YAML parsers and
vendor SDK normalization code for entertainment.

## Prepared off-camera

Do not spend video time on:

- `tests/` and `evolution_tasks/`
- provider-specific Responses object normalization in `runtime.py`
- OpenAI-compatible endpoint / vLLM setup
- YAML parsing, TUI rendering, CLI argument plumbing
- `remote_env.py`, sandbox server implementation
- TRL rollout/distributed plumbing, QLoRA configuration details
- merge/quantization code
- benchmark downloads and third-party compatibility

**Do implement `loss.py` manually.** Training infrastructure is prepared; the learning objective is not.

Run prepared tests after each section:

```bash
uv run pytest -q tests/test_tools.py tests/test_memory.py tests/test_skills.py
uv run pytest -q tests/test_agent.py tests/test_agent_memory.py tests/test_agent_continuous.py
uv run pytest -q tests/test_loss.py
uv run pytest -q tests/test_tasks.py tests/test_evaluate.py tests/test_archive.py tests/test_improve.py
```

## Rules for a genuine reconstruction attempt

1. Start from an empty scratch file, not a half-erased reference file.
2. Write invariants/interfaces first in comments.
3. No reference peek until the relevant tests have failed at least once.
4. Record `time_to_green`, `reference_peeks`, and `tests_green_before_first_peek`.
5. Syntax/API lookup is allowed after the design has been recalled; source lookup counts as a peek.
6. A successful repetition means equivalent behavior, not byte-identical code.
7. For `loss.py`, be able to derive every tensor shape and explain where gradients do and do not flow.
