from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from agent_ground_up.agent import (
    CANONICAL_PREFIX,
    CHECKPOINT_PREFIX,
    COMPACT_PROMPT,
    SYSTEM_PROMPT,
    Agent,
)
from agent_ground_up.skills import SkillRegistry
from agent_ground_up.tools import Toolbox
from agent_ground_up.ui import TUI


class FakeProcessor:
    def __init__(self, force_compaction: bool = False) -> None:
        self.force_compaction = force_compaction

    def apply_chat_template(self, messages: list[dict[str, Any]], **_: Any) -> dict[str, list[int]]:
        for message in messages:
            for call in message.get("tool_calls") or []:
                assert isinstance(call["function"]["arguments"], dict)
        if (
            self.force_compaction
            and len(messages) == 2
            and messages[0].get("content") != COMPACT_PROMPT
        ):
            return {"input_ids": [list(range(91))]}
        return {"input_ids": [list(range(20))]}


class FakeMessage:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return self.data.copy()


class FakeCompletions:
    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = FakeMessage(self.messages.pop(0))
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
    ui = Mock(spec=TUI)
    agent = Agent(
        client,
        "fake",
        Toolbox(tmp_path),
        FakeProcessor(),
        context_window=100,
        trajectory_path=trajectory,
        ui=ui,
    )
    result = agent.run("inspect the project")

    assert result.status == "completed"
    assert result.answer == "done"
    assert result.valid_tool_calls == 1
    assert agent.messages[-2]["content"] == "hello\n[exit code: 0]"
    sent_arguments = client.chat.completions.calls[1]["messages"][2]["tool_calls"][0]["function"]
    assert isinstance(sent_arguments["arguments"], str)
    assert json.loads(trajectory.read_text())["result"]["status"] == "completed"
    ui.user.assert_called_once_with("inspect the project")
    assert ui.assistant.call_count == 2
    ui.tool.assert_called_once_with("bash", "hello\n[exit code: 0]")


def test_agent_compacts_at_ninety_percent(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Always run focused tests.\n")
    client = FakeClient(
        [
            {"role": "assistant", "content": "preserve next step"},
            {"role": "assistant", "content": "finished"},
        ]
    )
    agent = Agent(
        client, "fake", Toolbox(tmp_path), FakeProcessor(force_compaction=True), context_window=100
    )
    result = agent.run("force the checkpoint")

    assert result.compactions == 1
    assert result.status == "completed"
    compact_call = client.chat.completions.calls[0]
    assert "tools" not in compact_call
    assert compact_call["messages"] == [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "force the checkpoint"},
        {"role": "user", "content": COMPACT_PROMPT},
    ]
    assert agent.messages[1]["content"].startswith(CANONICAL_PREFIX)
    assert str(tmp_path) in agent.messages[1]["content"]
    assert "Always run focused tests." in agent.messages[1]["content"]
    assert agent.messages[2]["content"] == f"{CHECKPOINT_PREFIX}\npreserve next step"
    assert agent.messages[3] == {"role": "user", "content": "force the checkpoint"}


def test_compaction_drops_old_checkpoints_and_budgets_recent_users(tmp_path: Path) -> None:
    agent = Agent(
        FakeClient([]),
        "fake",
        Toolbox(tmp_path),
        FakeProcessor(),
        recent_user_tokens=10,
    )
    older = {"role": "user", "content": "older request"}
    newest = {"role": "user", "content": "newest request"}
    agent.messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"{CANONICAL_PREFIX}\nold"},
        {"role": "assistant", "content": f"{CHECKPOINT_PREFIX}\nold"},
        older,
        newest,
    ]

    assert agent._episodic_history() == [
        {"role": "system", "content": SYSTEM_PROMPT},
        older,
        newest,
    ]
    assert agent._recent_user_messages() == [newest]


def test_invalid_call_becomes_observation(tmp_path: Path) -> None:
    invalid = {
        "role": "assistant",
        "tool_calls": [
            {"id": "bad", "type": "function", "function": {"name": "unknown", "arguments": "{}"}}
        ],
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
    agent = Agent(FakeClient([]), "fake", Toolbox("."), FakeProcessor())
    agent.messages = [{"role": "tool", "content": [value]}]
    assert "secret" not in json.dumps(agent._trajectory_messages())


def test_leading_system_messages_are_merged_on_the_wire(tmp_path: Path) -> None:
    """Qwen chat templates reject a system message that is not the very first one.

    The kernel deliberately keeps the base prompt and the canonical state apart so that
    compaction can drop the canonical block, so the merge has to happen where messages
    leave the agent, not in `agent.messages` itself.
    """
    client = FakeClient([{"role": "assistant", "content": "done"}])
    agent = Agent(
        client,
        "fake",
        Toolbox(tmp_path),
        FakeProcessor(),
        skills=SkillRegistry(tmp_path / "skills"),
    )
    result = agent.run("merge the system blocks")

    assert result.status == "completed"
    sent = client.chat.completions.calls[0]["messages"]
    assert [message["role"] for message in sent[:2]] == ["system", "user"]
    assert sent[0]["content"].startswith(SYSTEM_PROMPT)
    assert CANONICAL_PREFIX in sent[0]["content"]
    # The split is preserved internally, so episodic history can still drop the canonical block.
    assert [message["role"] for message in agent.messages[:3]] == ["system", "system", "user"]
    assert agent.messages[1]["content"].startswith(CANONICAL_PREFIX)
