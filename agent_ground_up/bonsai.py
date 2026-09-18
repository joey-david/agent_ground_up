from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_BONSAI_MODEL = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"
DEFAULT_LORA_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
}


def resolve_model_path(model: str | Path = DEFAULT_BONSAI_MODEL) -> Path:
    candidate = Path(model).expanduser()
    if candidate.exists():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(model))).resolve()


def load_bonsai(
    model: str | Path = DEFAULT_BONSAI_MODEL,
) -> tuple[Any, Any, dict[str, Any], Path]:
    """Load Bonsai 2 with its Hadamard-aware runtime and corrected tokenizer."""
    model_path = resolve_model_path(model)
    runtime_path = model_path / "runtime"
    loader = runtime_path / "vision_artifact.py"
    if not loader.is_file():
        raise FileNotFoundError(
            f"{loader} is missing; {model_path} is not a complete Bonsai 2 MLX pack"
        )

    if str(runtime_path) not in sys.path:
        sys.path.insert(0, str(runtime_path))

    from transformers import AutoTokenizer
    from transformers import logging as transformers_logging

    transformers_logging.set_verbosity_error()
    original_from_pretrained = AutoTokenizer.from_pretrained

    def fixed_from_pretrained(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("fix_mistral_regex", True)
        return original_from_pretrained(*args, **kwargs)

    with patch.object(AutoTokenizer, "from_pretrained", side_effect=fixed_from_pretrained):
        from vision_artifact import load_vl_model

        loaded_model, processor, config = load_vl_model(str(model_path))

    return loaded_model, processor, config, model_path


def _resolve_module(root: Any, name: str) -> tuple[Any, str]:
    parts = name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def attach_lora(
    model: Any,
    *,
    rank: int = 8,
    alpha: float = 16.0,
    module_names: list[str] | None = None,
) -> list[str]:
    """Add trainable low-rank branches around frozen Prism Packed projections."""
    if rank < 1:
        raise ValueError("rank must be positive")

    import mlx.core as mx
    from mlx import nn
    from runtime import Packed

    class PackedLoRA(nn.Module):
        def __init__(self, base: Any) -> None:
            super().__init__()
            self.base = base
            self.base.freeze()
            in_features = int(base.scales.shape[-1]) * 128
            out_features = int(base.weight.shape[0])
            self.lora_a = nn.Linear(in_features, rank, bias=False)
            self.lora_b = nn.Linear(rank, out_features, bias=False)
            self.lora_a.weight = (
                mx.random.normal(self.lora_a.weight.shape) / math.sqrt(in_features)
            ).astype(base.dtype)
            self.lora_b.weight = mx.zeros(self.lora_b.weight.shape, dtype=base.dtype)
            self.scale = float(alpha) / rank

        def __call__(self, x: Any) -> Any:
            return self.base(x) + self.scale * self.lora_b(self.lora_a(x))

    language = model.language_model
    wanted = set(module_names or ())
    candidates = list(language.named_modules())
    selected: list[str] = []
    model.freeze()

    for name, module in candidates:
        if not isinstance(module, Packed) or getattr(module, "embedding", False):
            continue
        if wanted:
            if name not in wanted:
                continue
        elif name.rsplit(".", 1)[-1] not in DEFAULT_LORA_TARGETS:
            continue
        parent, leaf = _resolve_module(language, name)
        setattr(parent, leaf, PackedLoRA(module))
        selected.append(name)

    if wanted - set(selected):
        missing = ", ".join(sorted(wanted - set(selected)))
        raise ValueError(f"adapter targets no longer match the checkpoint: {missing}")
    if not selected:
        raise ValueError("no packed language projections matched the LoRA targets")
    return selected


def save_adapters(
    model: Any,
    output: str | Path,
    *,
    rank: int,
    alpha: float,
    module_names: list[str],
) -> Path:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    directory = Path(output).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "format": "agent-ground-up-packed-lora-v1",
        "rank": rank,
        "alpha": alpha,
        "modules": module_names,
    }
    (directory / "adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(directory / "adapters.safetensors"), weights)
    return directory


def load_adapters(model: Any, adapter_path: str | Path) -> None:
    import mlx.core as mx

    directory = Path(adapter_path).expanduser().resolve()
    config = json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("format") != "agent-ground-up-packed-lora-v1":
        raise ValueError("unsupported adapter format")
    attach_lora(
        model,
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        module_names=list(config["modules"]),
    )
    model.load_weights(str(directory / "adapters.safetensors"), strict=False)
    model.eval()
    mx.eval(model.parameters())
