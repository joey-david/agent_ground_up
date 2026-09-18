# Reconstruction practice

Two deterministic drills cover the agent kernel without reproducing model-serving infrastructure.

## 1. `implementation/`

Imports, dataclasses, function signatures, and types are supplied; fill the bodies and the marked local config/TOML values.

```bash
uv run pytest -q practice/implementation/tests
```

Recommended order: tools → memory → skills → agent → tasks/evaluate → archive/improve → local config.

## 2. `signatures/`

Reconstruct only interfaces and annotations. The tests compare AST-level APIs against the implementation drill. YAML/TOML exercises cover the local Bonsai/MLX schema, not HTTP endpoints.

```bash
uv run pytest -q practice/signatures/tests
```

Reset either workspace with `git restore practice/implementation practice/signatures`.
