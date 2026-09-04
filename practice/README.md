# Reconstruction practice

This replaces the old `attempt_1/` / `attempt_2/` pattern with two deterministic drills.

## 1. `implementation/` — LeetCode-style implementation recall

The files already contain the imports, dataclasses, function/method signatures, argument types, return types, and `NotImplementedError` bodies. Fill only the bodies (plus the marked config/TOML values). Tests are the answer key.

Run from the repository root:

```bash
uv run pytest -q practice/implementation/tests
```

Recommended order: tools → memory → skills → agent → tasks/evaluate → archive/improve → loss → config.

## 2. `signatures/` — interface recall

The target Python/config/TOML files are empty. Add **only** class/function signatures and type annotations; bodies may be `...`. The tests compare the AST-level API against the implementation drill, so they check parameter names/kinds/default presence, annotations, and return types without executing your code. For YAML/TOML, the equivalent recall task is the schema: section/key structure, not production values.

```bash
uv run pytest -q practice/signatures/tests
```

## Reset

Both workspaces are committed starter states, so resetting is trivial:

```bash
git restore practice/implementation practice/signatures
```

Do not use the reference `agent_ground_up/` while timing a recall attempt. Record time-to-green and source peeks in your own notes/cards.
