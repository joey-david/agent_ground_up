# Reconstruction learning loop

The target skill is: **empty directory → working self-evolving kernel without consulting the
reference implementation**.

Use three kinds of retrieval:

1. **Concept cards (20–90 s):** explain an invariant or design choice.
2. **Function drills (5–15 min):** recreate one function from its contract and make focused tests green.
3. **System rebuilds (30–90 min):** recreate a whole module or the entire kernel.

Suggested rebuild spacing: day 0, 1, 3, 7, 14, 30, then every 1–2 months. Let FSRS schedule the
short cards; keep the larger rebuilds as explicit coding cards/tasks.

For every reconstruction, record:

```text
date | drill | minutes | reference_peeks | tests_before_peek | final_tests
```

Rate coding cards as:

- Again: could not reconstruct the governing idea.
- Hard: needed reference source or had a major conceptual error.
- Good: correct design; only normal syntax/debugging mistakes.
- Easy: blank file to green without source reference.

The seed questions in `cards.md` are about invariants rather than line-level trivia.
`lazyvim/mnemonic.lua` installs an FSRS review surface inside LazyVim.
