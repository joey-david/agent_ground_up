# Self-improvement policy

This file is part of every archived descendant. The mutation agent reads it before proposing the
next descendant, so editing this file changes how future self-improvement rounds are performed.
That is the recursive surface; the tiny outer archive/evaluator loop remains fixed so descendants
cannot redefine what counts as success.

1. Select a task family near the current capability frontier, not the easiest available family.
2. Diagnose failures before editing. Prefer one falsifiable hypothesis per descendant.
3. Prefer reusable changes: better tool interfaces, generated skills, memory retrieval, context
   management, planning, verification, or the improvement policy itself.
4. Never hard-code benchmark answers or inspect held-out verifier internals to synthesize outputs.
5. Run focused unit tests after mutation.
6. Keep weaker but valid descendants in the archive: they may be useful stepping stones.
7. Promote claims only from held-out sibling tasks. Training cases are for diagnosis, not selection.
8. If a procedure is repeatedly re-derived, turn it into a persistent skill; if a discovery is
   repeatedly forgotten, store it in persistent memory.
