# Video implementation boundary

The project is now a single local stack: Bonsai 2 is loaded directly with its bundled Hadamard-aware runtime, MLX performs inference/training, and the agent calls it in-process.

## Part A — lifetime agent kernel

1. `bonsai.py`: load the packed checkpoint and optional adapters.
2. `runtime.py`: chat templating, streamed generation, native Qwen tool-call parsing.
3. `tools.py`: bash/image boundary and bounded observations.
4. `memory.py`, `experience.py`, `skills.py`: persistent state outside the context window.
5. `agent.py`: one model/tool loop plus explicit compaction.

There is no server/client section to implement: tokens flow from `mlx_vlm.stream_generate` directly into the terminal and the agent.

## Part B — recursive specialization

1. `tasks.py`: train/held-out sibling families and frontier selection.
2. `evaluate.py`: fixed local evaluator.
3. `archive.py`: immutable descendants and replay-based exploration.
4. `improve.py`: select → mutate → evaluate → archive.
5. `loss.py` / `scripts/train.py`: frozen packed base, MLX LoRA branches, replaceable loss.

## Prepared rather than reconstructed

- the published Bonsai checkpoint and its bundled Prism loader
- tests and benchmark fixtures
- config/CLI glue

No Lamgate, SSH tunnel, vLLM rollout server, remote coding environment, or container sandbox is part of the repository anymore.
