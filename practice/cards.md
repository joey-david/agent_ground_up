# Active-recall prompts

## Tools
- Why kill a timed-out bash **process group** instead of only the parent?
- Why should truncation preserve both head and tail and measure model tokens rather than characters?
- Why is image confinement a tool invariant rather than a prompt instruction?

## Agent state
- What exact condition means an agent run is complete?
- Distinguish native/working context, exact experience, semantic memory, and generated skills.
- Which state should survive a new episode, and which should not?

## Memory / skills
- What does bounded active memory mean if persistent storage can keep growing?
- When does a repeated discovery become a memory versus a generated skill?

## Recursive improvement
- Why use sibling task families instead of one fixed benchmark instance?
- Why select near the capability frontier rather than always the hardest family?
- Why retain weaker valid descendants?
- What makes editing `improvement_policy.md` genuinely recursive while the evaluator stays fixed?
- Why can mutation see train evidence but not held-out case details?

## LM dynamics
- Why does logit position `t` score token `t+1`?
- Derive chosen-token log-probabilities from `[B,T,V]` logits.
- Derive `r_t = exp(log pi_theta - log pi_old)`.
- Explain asymmetric clipping for positive versus negative advantages.
- Why mask padding/tool/environment tokens before normalization?
- What does the sampled reverse-KL estimator penalize?

## Configuration
- Reconstruct the minimal runtime YAML sections from memory.
- Which dependencies belong in the core project versus training/dev extras?
- Why does production resolve config-relative paths against the project root rather than `configs/`?
