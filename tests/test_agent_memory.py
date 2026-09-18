import json
from pathlib import Path

from agent_ground_up.agent import Agent
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.runtime import RuntimeTurn
from agent_ground_up.skills import SkillRegistry
from agent_ground_up.tools import Toolbox


def tool_call(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


class Runtime:
    model_id = "fake"

    def __init__(self, messages):
        self.messages = list(messages)
        self.calls = []

    def prompt_tokens(self, messages, tools=None):
        return 20

    def complete(self, messages, *, tools, max_output_tokens, on_token=None):
        self.calls.append({"messages": messages, "tools": tools})
        message = self.messages.pop(0)
        return RuntimeTurn(message, 20, 1, message.get("content") or "")


def test_agent_exposes_persistent_memory_and_skill_tools(tmp_path: Path) -> None:
    memory = ConstantMemory(tmp_path / "memory")
    skills = SkillRegistry(tmp_path / "skills")
    skills.register("echo_arg", "echo one argument", "main() { printf '%s' \"$1\"; }")
    runtime = Runtime(
        [
            tool_call("remember", {"text": "parser failures need utf8 replacement", "tags": ["io"]}),
            tool_call("skill", {"name": "echo_arg", "argument": "hello"}),
            {"role": "assistant", "content": "done"},
        ]
    )
    agent = Agent(
        runtime,
        Toolbox(tmp_path),
        context_window=1000,
        memory=memory,
        skills=skills,
    )

    result = agent.run("do it")

    assert result.status == "completed"
    assert any(record.text.startswith("parser failures") for record in memory.records())
    assert any(record.text.startswith("Completed task") for record in memory.records())
    tool_names = {
        schema["function"]["name"] for schema in runtime.calls[0]["tools"]
    }
    assert {"bash", "view_image", "remember", "recall", "zoom", "create_skill", "skill"} <= tool_names
    assert "hello" in agent.messages[-2]["content"]
