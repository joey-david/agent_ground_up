from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from openai import OpenAI
from transformers import AutoProcessor

from .config import secret
from .runtime import ContinuousResponsesRuntime


@dataclass(frozen=True, slots=True)
class ModelRuntimeBundle:
    client: Any
    processor: Any | None
    runtime: ContinuousResponsesRuntime | None
    token_counter: Callable[[str], int]


def build_model_runtime(model_config: dict[str, Any]) -> ModelRuntimeBundle:
    """Build either the local Chat Completions runtime or provider-native Responses runtime."""
    client = OpenAI(
        base_url=model_config.get("base_url") or None,
        api_key=secret(model_config, "api_key_env"),
    )
    mode = model_config.get("runtime", "chat_completions")
    if mode == "chat_completions":
        processor_name = model_config.get("processor")
        if not processor_name:
            raise ValueError("chat_completions runtime requires model.processor")
        processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=False)

        def token_counter(text: str) -> int:
            return len(processor.tokenizer.encode(text, add_special_tokens=False))

        return ModelRuntimeBundle(client, processor, None, token_counter)

    if mode == "responses_continuous":
        runtime = ContinuousResponsesRuntime(
            client,
            model_config["served_name"],
            reasoning_effort=model_config.get("reasoning_effort", "high"),
            compact_threshold=int(model_config.get("compact_threshold", 175_000)),
            reasoning_summary=model_config.get("reasoning_summary", "auto"),
        )

        # Tool-output truncation does not need exact prompt accounting on this path because the
        # provider reports usage and manages context. Four characters/token is a deliberately
        # simple local estimate used only to cap individual shell observations.
        def token_counter(text: str) -> int:
            return max(1, (len(text) + 3) // 4)

        return ModelRuntimeBundle(client, None, runtime, token_counter)

    raise ValueError(f"unknown model.runtime: {mode}")
