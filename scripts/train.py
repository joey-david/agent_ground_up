from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from agent_ground_up.bonsai import attach_lora, load_bonsai, save_adapters
from agent_ground_up.config import DEFAULT_CONFIG, load_config, path, section

LossFn = Callable[[Any, dict[str, Any]], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train local MLX adapters on the Bonsai base")
    parser.add_argument("--config", default=os.getenv("AGENT_CONFIG", str(DEFAULT_CONFIG)))
    parser.add_argument(
        "--loss",
        default="agent_ground_up.loss:causal_lm_loss",
        help="custom MLX loss as module:function",
    )
    parser.add_argument("--steps", type=int, help="override training.steps")
    return parser.parse_args()


def load_loss(spec: str) -> LossFn:
    module_name, separator, function_name = spec.partition(":")
    if not separator:
        raise ValueError("--loss must be module:function")
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"{spec} is not callable")
    return function


def load_examples(source: Path) -> list[list[dict[str, Any]]]:
    examples: list[list[dict[str, Any]]] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        messages = item.get("messages")
        if messages is None:
            if "prompt" not in item or "response" not in item:
                raise ValueError("each JSONL row needs messages or prompt/response")
            messages = [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["response"]},
            ]
        examples.append(messages)
    if not examples:
        raise ValueError(f"no training examples in {source}")
    return examples


def encode_example(
    processor: Any,
    config: dict[str, Any],
    messages: list[dict[str, Any]],
    max_seq_length: int,
) -> list[int]:
    from mlx_vlm.prompt_utils import apply_chat_template
    from vision_artifact import chat_config

    prompt = apply_chat_template(
        processor,
        chat_config(config),
        messages,
        add_generation_prompt=False,
        num_images=0,
        enable_thinking=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(token_ids) < 2:
        raise ValueError("training example tokenized to fewer than two tokens")
    return token_ids[:max_seq_length]


def train(config: dict[str, Any], loss_fn: LossFn, *, steps_override: int | None = None) -> Path:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    model_values = section(config, "model")
    values = section(config, "training")
    model, processor, model_config, _ = load_bonsai(model_values["id"])

    rank = int(values.get("rank", 8))
    alpha = float(values.get("alpha", 16.0))
    module_names = attach_lora(model, rank=rank, alpha=alpha)
    model.train()

    examples = load_examples(path(config, values["data"]))
    max_seq_length = int(values.get("max_seq_length", 512))
    encoded = [
        encode_example(processor, model_config, messages, max_seq_length)
        for messages in examples
    ]

    optimizer = optim.AdamW(learning_rate=float(values.get("learning_rate", 2e-5)))
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    steps = int(steps_override or values.get("steps", 100))

    for step in range(1, steps + 1):
        ids = encoded[(step - 1) % len(encoded)]
        input_ids = mx.array([ids])
        batch = {
            "input_ids": input_ids,
            "attention_mask": mx.ones(input_ids.shape, dtype=mx.int32),
        }
        loss, gradients = loss_and_grad(model, batch)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)
        if step == 1 or step % int(values.get("report_every", 10)) == 0 or step == steps:
            print(f"step={step} loss={float(loss.item()):.4f}")

    output = path(config, values.get("output", "outputs/bonsai-lora"))
    return save_adapters(
        model,
        output,
        rank=rank,
        alpha=alpha,
        module_names=module_names,
    )


def main() -> None:
    args = parse_args()
    output = train(load_config(args.config), load_loss(args.loss), steps_override=args.steps)
    print(f"saved adapter to {output}")


if __name__ == "__main__":
    main()
