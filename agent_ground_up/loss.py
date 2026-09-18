from __future__ import annotations

from typing import Any


def causal_lm_loss(model: Any, batch: dict[str, Any]) -> Any:
    """Standard next-token cross entropy for the local MLX model."""
    import mlx.core as mx
    import mlx.nn as nn

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    model_ids = input_ids[:, :-1]
    labels = input_ids[:, 1:]
    mask = attention_mask[:, 1:].astype(mx.float32)

    outputs = model(model_ids, None, attention_mask[:, :-1])
    logits = outputs.logits.astype(mx.float32)
    if logits.shape[1] != labels.shape[1]:
        logits = logits[:, -labels.shape[1] :, :]
    losses = nn.losses.cross_entropy(logits, labels)
    return (losses * mask).sum() / mx.maximum(mask.sum(), 1)
