# Video implementation boundary

The goal stays unchanged: implement a modern recursively self-improving agent from first principles, not present any particular frontier model. The typed-code budget is **about 140–160 minutes total**.

## Video A — lifetime agent kernel (~70–75 min)

1. `tools.py`: bash boundary, process-group timeout, token head/tail truncation, image confinement.
2. `memory.py`: append-only memories, bounded wake context, recall, summary-tree zoom.
3. `skills.py`: persistent generated procedures through the same tool boundary.
4. `agent.py`: model/tool loop, dynamic tools, completion condition, persistent state.
5. Demo: solve, persist a discovery/skill, reuse it on a second task.

The provider adapter / continuous-response plumbing is prepared infrastructure. Explain what state it preserves, but do not spend timed implementation minutes normalizing SDK objects.

## Video B — recursive specialization + LM dynamics (~70–85 min)

1. `tasks.py`: train/held-out sibling families and capability-frontier selection.
2. `evaluate.py`: fixed evaluator and reports.
3. `archive.py`: immutable descendants, fingerprints, novelty/exploration parent selection.
4. `improve.py`: select → mutate → train evidence → held-out evaluation → archive.
5. `loss.py`: manually derive causal chosen-token log-probs and the clipped DAPO/GRPO-style objective.
6. Demo: one self-improvement round plus a tiny loss/backprop sanity check.

Draw before code:

```text
frontier task family
       ↓
 archive parent ──→ mutate tools / memory / policy / agent
       ↑                         ↓
       └──── held-out eval ← descendant

persistent experimental memory spans rounds
```

The editable policy lives at `agent_ground_up/improvement_policy.md`; the held-out evaluator remains fixed.

## Prepared off-camera

- tests and benchmark fixtures
- config/CLI glue
- provider SDK adapter
- `infra/upnquick/` model backend and SSH tunnel
- sandbox server
- vLLM/TRL rollout plumbing, QLoRA, merge/quantization
- benchmark downloads

The backend on `ssh upnquick` is explicitly **not part of the three-hour clock**.

## Practice

Use `practice/implementation/` for body-level reconstruction and `practice/signatures/` for pure API/type recall. See `practice/README.md`.
