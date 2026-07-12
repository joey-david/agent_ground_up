from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from agent_ground_up.agent import COMPACT_PROMPT, Agent
from agent_ground_up.tools import Toolbox


class FakeProcessor:
    def __init__(self, force_compaction: bool = False) -> None:
        self.force_compaction = force_compaction

    def count_tokens(self, messages: list[dict[str, Any]], _: Any) -> int:
        if self.force_compaction and len(messages) == 2 and messages[0].get("content") != COMPACT_PROMPT:
            return 91
        return 20


class FakeCompletions:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = self.messages.pop(0)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeClient:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(messages))


def tool_call(command: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": command})},
            }
        ],
    }


def test_agent_executes_tool_then_finishes(tmp_path: Path) -> None:
    trajectory = tmp_path / "run.json"
    client = FakeClient([tool_call("printf hello"), {"role": "assistant", "content": "done"}])
    agent = Agent(client, "fake", Toolbox(tmp_path), FakeProcessor(), context_window=100, trajectory_path=trajectory)
    result = agent.run("inspect the project")

    assert result.status == "completed"
    assert result.answer == "done"
    assert result.valid_tool_calls == 1
    assert agent.messages[-2]["content"] == "hello\n[exit code: 0]"
    assert json.loads(trajectory.read_text())["result"]["status"] == "completed"


def test_agent_compacts_at_ninety_percent(tmp_path: Path) -> None:
    client = FakeClient(
        [{"role": "assistant", "content": "preserve next step"}, {"role": "assistant", "content": "finished"}]
    )
    agent = Agent(client, "fake", Toolbox(tmp_path), FakeProcessor(force_compaction=True), context_window=100)
    result = agent.run("force the checkpoint")

    assert result.compactions == 1
    assert result.status == "completed"
    compact_call = client.chat.completions.calls[0]
    assert "tools" not in compact_call
    assert compact_call["messages"][0]["content"] == COMPACT_PROMPT
    assert agent.messages[1]["content"] == "force the checkpoint"
    assert "preserve next step" in agent.messages[2]["content"]


def test_invalid_call_becomes_observation(tmp_path: Path) -> None:
    invalid = {
        "role": "assistant",
        "tool_calls": [{"id": "bad", "type": "function", "function": {"name": "unknown", "arguments": "{}"}}],
    }
    agent = Agent(
        FakeClient([invalid, {"role": "assistant", "content": "recovered"}]),
        "fake",
        Toolbox(tmp_path),
        FakeProcessor(),
        context_window=100,
    )
    result = agent.run("recover")
    assert result.invalid_tool_calls == 1
    assert "unknown tool" in agent.messages[-2]["content"]


def test_safe_trajectory_removes_image_bytes() -> None:
    value = {"type": "image_url", "image_url": {"url": "data:image/png;base64,secret"}}
    assert "secret" not in json.dumps(Agent._safe(value))
