from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RuntimeTurn:
    """One provider turn normalized to the agent's readable message format."""

    message: dict[str, Any]
    input_tokens: int
    output_tokens: int
    compactions: int


class ContinuousResponsesRuntime:
    """Stateless Responses API loop that preserves provider-native reasoning state.

    The runtime replays accepted native response.output items, including encrypted reasoning,
    instead of reducing every turn to visible assistant text. Provider compaction items replace the
    older replay prefix when present, keeping effective context bounded without an LLM-authored
    plaintext checkpoint.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        *,
        reasoning_effort: str = "high",
        compact_threshold: int = 175_000,
        reasoning_summary: str = "auto",
    ) -> None:
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.compact_threshold = compact_threshold
        self.reasoning_summary = reasoning_summary
        self.history: list[dict[str, Any]] = []
        self.compactions = 0

    def reset(self, task: str) -> None:
        self.history = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": task}],
            }
        ]
        self.compactions = 0

    def complete(
        self,
        *,
        instructions: str,
        tools: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> RuntimeTurn:
        request: dict[str, Any] = {
            "model": self.model,
            "instructions": instructions,
            # Snapshot at the provider boundary: later compaction is allowed to mutate our local
            # replay state, but it must never retroactively mutate the request that was just sent.
            "input": copy.deepcopy(self.history),
            "tools": self._response_tools(tools),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "max_output_tokens": max_output_tokens,
            "store": False,
            "include": ["reasoning.encrypted_content"],
            "reasoning": {
                "effort": self.reasoning_effort,
                "summary": self.reasoning_summary,
                "context": "auto",
            },
        }
        if self.compact_threshold > 0:
            request["context_management"] = [
                {"type": "compaction", "compact_threshold": self.compact_threshold}
            ]

        response = self.client.responses.create(**request)
        status = getattr(response, "status", "completed") or "completed"
        if status in {"failed", "cancelled", "incomplete"}:
            details = getattr(response, "incomplete_details", None) or getattr(response, "error", None)
            raise RuntimeError(f"Responses API turn ended with status={status}: {details}")

        output = [self._sanitize_output_item(self._dump(item)) for item in response.output]
        self.history.extend(output)
        new_compactions = sum(item.get("type") == "compaction" for item in output)
        self.compactions += new_compactions
        if new_compactions:
            self._prune_to_latest_compaction()

        usage = getattr(response, "usage", None)
        return RuntimeTurn(
            message=self._mirror(output),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            compactions=self.compactions,
        )

    def submit_tool_output(
        self,
        *,
        call_id: str,
        name: str,
        content: str | list[dict[str, Any]],
    ) -> None:
        self.history.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                "name": name,
                "output": self._response_tool_output(content),
            }
        )

    def _prune_to_latest_compaction(self) -> None:
        latest = max(
            index for index, item in enumerate(self.history) if item.get("type") == "compaction"
        )
        self.history = self.history[latest:]

    @staticmethod
    def _response_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for schema in tools:
            function = schema.get("function") or {}
            converted.append(
                {
                    "type": "function",
                    "name": function["name"],
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {"type": "object"}),
                    # Existing local schemas contain optional arguments (e.g. timeout_s), which
                    # are not valid strict-mode schemas unless every property is required.
                    "strict": bool(function.get("strict", False)),
                }
            )
        return converted

    @staticmethod
    def _response_tool_output(
        content: str | list[dict[str, Any]],
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        converted: list[dict[str, Any]] = []
        for block in content:
            if block.get("type") == "text":
                converted.append({"type": "input_text", "text": block.get("text", "")})
            elif block.get("type") == "image_url":
                image_url = block.get("image_url") or {}
                converted.append(
                    {
                        "type": "input_image",
                        "image_url": image_url.get("url", ""),
                        "detail": "auto",
                    }
                )
        return converted or "(tool returned no readable output)"

    @staticmethod
    def _mirror(output: list[dict[str, Any]]) -> dict[str, Any]:
        text_parts: list[str] = []
        calls: list[dict[str, Any]] = []
        for item in output:
            kind = item.get("type")
            if kind == "message":
                for block in item.get("content") or []:
                    if block.get("type") in {"output_text", "text"} and block.get("text"):
                        text_parts.append(block["text"])
            elif kind == "function_call":
                calls.append(
                    {
                        "id": item.get("call_id") or item.get("id") or "missing-call-id",
                        "type": "function",
                        "function": {
                            "name": item.get("name", "unknown"),
                            "arguments": item.get("arguments") or "{}",
                        },
                    }
                )
        message: dict[str, Any] = {"role": "assistant", "content": "\n".join(text_parts)}
        if calls:
            message["tool_calls"] = calls
        return message

    @staticmethod
    def _sanitize_output_item(item: dict[str, Any]) -> dict[str, Any]:
        # The stateless replay input rejects these response-only fields for the corresponding
        # opaque provider items. Other fields (including phase) must remain untouched.
        if item.get("type") == "reasoning":
            item.pop("status", None)
        if item.get("type") == "compaction":
            item.pop("created_by", None)
        return item

    @staticmethod
    def _dump(item: Any) -> dict[str, Any]:
        if isinstance(item, dict):
            return dict(item)
        if hasattr(item, "model_dump"):
            return item.model_dump(exclude_none=True)
        raise TypeError(f"unsupported Responses output item: {type(item).__name__}")
