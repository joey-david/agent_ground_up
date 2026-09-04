# Seed cards — agent_ground_up

## Tool kernel

- Why must a timed-out bash call kill the process *group*, not only its parent process?
- State the six invariants of `Toolbox.bash` before writing any Python.
- Why truncate tool output by model tokens while preserving both head and tail?
- Why is image path confinement a tool-layer invariant rather than a prompt instruction?
- Drill: reconstruct `_truncate()` from its behavioral contract; make `test_tools.py` green.

## Agent loop

- What exact condition means the agent has completed a turn successfully?
- What state is episodic, what state is canonical, and why must compaction separate them?
- Why should prompt-token accounting fail closed when the exact processor cannot render the request?
- Reconstruct the complete message sequence for one assistant tool call and observation.
- Drill: implement the model/tool loop from blank `agent.py`, then add compaction.

## Lifetime memory

- What does “constant memory” mean here, and what does it explicitly *not* mean?
- Why are wake context and total persistent storage independent quantities?
- Explain `remember → summary tree → wake / recall / zoom` without referring to code.
- Why keep raw append-only memories even after summaries exist?
- Drill: recreate `memory.py` from only the four public operations.

## Skills

- What is the difference between a persistent memory and a generated skill?
- Why do generated skills execute through the same bash boundary rather than a privileged path?
- When should repeated reasoning be converted into a skill rather than another memory?

## LM policy dynamics

- For a causal LM, which logit position predicts completion token `t`, and why is a one-token shift required?
- Starting from logits `[B,T,V]`, derive the exact tensor operations that produce chosen-token log-probs `[B,C]`.
- Why does `old_logps = logps.detach()` produce a ratio numerically equal to one but still allow a policy gradient?
- Derive `r_t = exp(log pi_theta - log pi_old)` and explain why clipping is asymmetric when `epsilon_low != epsilon_high`.
- For positive vs negative advantage, which side of the trust region becomes active and why?
- Why does DAPO normalize over active tokens in the accumulated batch instead of averaging each response separately?
- Derive the sampled reverse-KL estimator `exp(ref_logp - logp) - (ref_logp - logp) - 1` and identify its minimum.
- Which tensors must be detached, and which must retain gradient, for the RL update to actually change the LM?
- Drill: blank `loss.py` → implement causal shift, gather chosen-token log-probs, clipping, KL, mask, and `backward()` with `test_loss.py` green.

## Evaluation and self-improvement

- Why must a task be a family with unseen sibling instances rather than one fixed task?
- How does frontier selection differ from selecting the hardest task?
- Why keep weaker but valid descendants in the archive?
- What makes this loop recursive rather than merely self-modifying?
- Why is `improvement_policy.md` editable while the held-out evaluator remains fixed?
- What information may training cases expose that held-out cases must not?
- Drill: reconstruct `SelfImprover.run_round` from the arrows in `video/README.md`.

## Full rebuild

- Drill: empty directory → tools + agent + memory + skills + manual loss + task/eval/archive/improve; all focused tests green with zero source peeks.
