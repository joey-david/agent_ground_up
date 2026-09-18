from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

# These must be set before importing transformers. The Prism pack deliberately
# advertises a custom model_type while instantiating the compatible Qwen3.5 VLM
# graph, which makes Transformers emit an irrelevant architecture warning.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_BONSAI_MODEL = "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"


def resolve_model_path(model: str | Path = DEFAULT_BONSAI_MODEL) -> Path:
    """Return a local checkpoint directory, downloading the HF repo if necessary."""
    candidate = Path(model).expanduser()
    if candidate.exists():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=str(model))).resolve()


def load_bonsai(model: str | Path = DEFAULT_BONSAI_MODEL) -> tuple[Any, Any, dict[str, Any], Path]:
    """Load Bonsai 2 with its Hadamard-aware runtime and a correctly patched tokenizer.

    The bundled loader currently omits fix_mistral_regex=True. Patch only that
    construction call in memory rather than mutating the Hugging Face cache, so
    Prism's runtime files remain byte-for-byte intact for checksum verification.
    """
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

    # Import inside the patch as well: this remains correct if a future pack moves
    # tokenizer construction from load_vl_model() to module import time.
    with patch.object(AutoTokenizer, "from_pretrained", side_effect=fixed_from_pretrained):
        from vision_artifact import load_vl_model

        loaded_model, processor, config = load_vl_model(str(model_path))

    return loaded_model, processor, config, model_path
