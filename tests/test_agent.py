from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

from agent_ground_up.agent import CHECKPOINT_PREFIX, Agent
from agent_ground_up.runtime import RuntimeTurn
from agent_ground_up.tools import Toolbox
from agent_ground_up.ui import TUI


def tool_call(command: str) -> dict:
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


class FakeRuntime:
    model_id = "fake-local"

    def __init__(self, messages: list[dict], *, force_compaction: bool = False) -> None:
        self.messages = list(messages)
        self.force_compaction = force_compaction
        self.calls: list[dict] = []
        self._budget_checks = 0

    def prompt_tokens(self, messages, tools=None) -> int:
        if self.force_compaction and tools is not None and self._budget_checks == 0:
            self._budget_checks += 1
            return 91
        return 20

    def count_text(self, text: str) -> int:
        return len(text)

    def complete(self, messages, *, tools, max_output_tokens, on_token=None):
        message = self.messages.pop(0)
        self.calls.append({"messages": messages, "tools": tools, "max_output_tokens": max_output_tokens})
        content = message.get("content") or ""
        if on_token and content:
            for piece in (content[:2], content[2:]):
                if piece:
                    on_token(piece)
        return RuntimeTurn(message, 20, len(content), content)


def test_agent_streams_executes_tool_then_finishes(tmp_path: Path) -> None:
    trajectory = tmp_path / "run.json"
    runtime = FakeRuntime([tool_call("printf hello"), {"role": "assistant", "content": "done"}])
    ui = Mock(spec=TUI)
    agent = Agent(
        runtime,
        Toolbox(tmp_path),
        context_window=100,
        trajectory_path=trajectory,
        ui=ui,
    )

    result = agent.run("inspect the project")

    assert result.status == "completed"
    assert result.answer == "done"
    assert result.valid_tool_calls == 1
    assert agent.messages[-2]["content"] == "hello\n[exit code: 0]"
    assert json.loads(trajectory.read_text())["runtime"] == "local_mlx"
    ui.begin_assistant.assert_called()
    assert ui.token.call_count >= 1
    ui.tool.assert_called_once_with("bash", "hello\n[exit code: 0]")


def test_agent_compacts_locally(tmp_path: Path) -> None:
    runtime = FakeRuntime(
        [
            {"role": "assistant", "content": "preserve next step"},
            {"role": "assistant", "content": "finished"},
        ],
        force_compaction=True,
    )
    agent = Agent(runtime, Toolbox(tmp_path), context_window=100)

    result = agent.run("force the checkpoint")

    assert result.status == "completed"
    assert result.compactions == 1
    assert agent.messages[1]["content"] == f"{CHECKPOINT_PREFIX}\npreserve next step"
    assert agent.messages[2] == {"role": "user", "content": "force the checkpoint"}
    assert runtime.calls[0]["tools"] is None


def test_invalid_call_becomes_observation(tmp_path: Path) -> None:
    invalid = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "bad", "type": "function", "function": {"name": "unknown", "arguments": "{}"}}
        ],
    }
    runtime = FakeRuntime([invalid, {"role": "assistant", "content": "recovered"}])
    agent = Agent(runtime, Toolbox(tmp_path), context_window=100)

    result = agent.run("recover")

    assert result.invalid_tool_calls == 1
    assert "unknown tool" in agent.messages[-2]["content"]


def test_safe_trajectory_removes_image_bytes(tmp_path: Path) -> None:
    runtime = FakeRuntime([])
    agent = Agent(runtime, Toolbox(tmp_path))
    value = {"type": "image_url", "image_url": {"url": "data:image/png;base64,secret"}}
    agent.messages = [{"role": "tool", "content": [value]}]

    assert "secret" not in json.dumps(agent._trajectory_messages())
