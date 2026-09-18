from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bonsai import DEFAULT_BONSAI_MODEL, load_bonsai

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_RE = re.compile(
    r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


@dataclass(frozen=True, slots=True)
class RuntimeTurn:
    message: dict[str, Any]
    input_tokens: int
    output_tokens: int
    raw_text: str


def _tool_properties(tools: list[dict[str, Any]] | None, name: str) -> dict[str, Any]:
    for tool in tools or ():
        function = tool.get("function") or {}
        if function.get("name") == name:
            return (function.get("parameters") or {}).get("properties") or {}
    return {}


def _coerce_argument(value: str, schema: dict[str, Any]) -> Any:
    kind = schema.get("type")
    value = value.strip()
    if value.lower() == "null":
        return None
    if kind == "string" or kind is None:
        return value
    if kind == "integer":
        return int(value)
    if kind == "number":
        return float(value)
    if kind == "boolean":
        lowered = value.lower()
        if lowered not in {"true", "false"}:
            raise ValueError(f"invalid boolean {value!r}")
        return lowered == "true"
    if kind in {"object", "array"}:
        return json.loads(value)
    return value


def parse_assistant(text: str, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Turn Bonsai/Qwen's native XML tool syntax into the agent's structured message."""
    reasoning = ""
    visible = text
    if "</think>" in visible:
        reasoning, visible = visible.split("</think>", 1)
        reasoning = reasoning.removeprefix("<think>").strip()
    elif visible.lstrip().startswith("<think>"):
        reasoning = visible.split("<think>", 1)[1].strip()
        visible = ""

    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(visible):
        name, body = match.group(1).strip(), match.group(2)
        properties = _tool_properties(tools, name)
        arguments: dict[str, Any] = {}
        for parameter in _PARAMETER_RE.finditer(body):
            key, value = parameter.group(1).strip(), parameter.group(2)
            arguments[key] = _coerce_argument(value, properties.get(key, {}))
        calls.append(
            {
                "id": f"call-{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )

    content = _TOOL_CALL_RE.sub("", visible).strip()
    message: dict[str, Any] = {"role": "assistant", "content": content or None}
    if reasoning:
        message["reasoning_content"] = reasoning
    if calls:
        message["tool_calls"] = calls
    return message


class LocalBonsaiRuntime:
    """In-process MLX runtime for the Prism Bonsai 2 checkpoint."""

    def __init__(
        self,
        model: str | Path = DEFAULT_BONSAI_MODEL,
        *,
        adapter_path: str | Path | None = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
    ) -> None:
        self.model_id = str(model)
        self.model, self.processor, self.config, self.model_path = load_bonsai(model)
        if adapter_path:
            from .bonsai import load_adapters

            load_adapters(self.model, adapter_path)
        from vision_artifact import chat_config

        self.chat_config = chat_config(self.config)
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        vision = self.config.get("vision_config") or {}
        self.patch_size = int(vision.get("patch_size", 16))

    def _images(self, messages: list[dict[str, Any]]) -> list[str]:
        images: list[str] = []
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "image_url":
                    continue
                image_url = block.get("image_url") or {}
                url = image_url.get("url")
                if isinstance(url, str) and url:
                    images.append(url)
        return images

    def prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str, list[str]]:
        from mlx_vlm.prompt_utils import apply_chat_template

        images = self._images(messages)
        prompt = apply_chat_template(
            self.processor,
            self.chat_config,
            messages,
            num_images=len(images),
            tools=tools or None,
            enable_thinking=True,
        )
        return prompt, images

    def prompt_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> int:
        prompt, _ = self.prompt(messages, tools)
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return len(tokenizer.encode(prompt, add_special_tokens=False))

    def count_text(self, text: str) -> int:
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        return len(tokenizer.encode(text, add_special_tokens=False))

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None,
        max_output_tokens: int,
        on_token: Callable[[str], None] | None = None,
    ) -> RuntimeTurn:
        from mlx_vlm import stream_generate

        prompt, images = self.prompt(messages, tools)
        input_tokens = self.prompt_tokens(messages, tools)
        pieces: list[str] = []
        last = None
        for chunk in stream_generate(
            self.model,
            self.processor,
            prompt=prompt,
            image=images or None,
            max_tokens=max_output_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            verbose=False,
        ):
            piece = getattr(chunk, "text", "")
            if piece:
                pieces.append(piece)
                if on_token:
                    on_token(piece)
            last = chunk

        raw = "".join(pieces)
        output_tokens = int(getattr(last, "generation_tokens", 0) or 0)
        if not output_tokens:
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            output_tokens = len(tokenizer.encode(raw, add_special_tokens=False))
        return RuntimeTurn(parse_assistant(raw, tools), input_tokens, output_tokens, raw)
