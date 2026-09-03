import json
from pathlib import Path
from types import SimpleNamespace

from agent_ground_up.agent import Agent
from agent_ground_up.memory import ConstantMemory
from agent_ground_up.skills import SkillRegistry
from agent_ground_up.tools import Toolbox


class Processor:
    def apply_chat_template(self, messages, **kwargs):
        return {"input_ids": list(range(20))}


class Message:
    def __init__(self, data):
        self.data = data

    def model_dump(self, **kwargs):
        return self.data.copy()


class Completions:
    def __init__(self, messages):
        self.messages = messages
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=Message(self.messages.pop(0)))])


class Client:
    def __init__(self, messages):
        self.chat = SimpleNamespace(completions=Completions(messages))


def tool_call(name: str, arguments: dict) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }


def test_agent_exposes_persistent_memory_and_skill_tools(tmp_path: Path) -> None:
    memory = ConstantMemory(tmp_path / "memory")
    skills = SkillRegistry(tmp_path / "skills")
    skills.register("echo_arg", "echo one argument", "main() { printf '%s' \"$1\"; }")
    client = Client(
        [
            tool_call("remember", {"text": "parser failures need utf8 replacement", "tags": ["io"]}),
            tool_call("skill", {"name": "echo_arg", "argument": "hello"}),
            {"role": "assistant", "content": "done"},
        ]
    )
    agent = Agent(
        client,
        "fake",
        Toolbox(tmp_path),
        Processor(),
        context_window=1000,
        memory=memory,
        skills=skills,
    )
    result = agent.run("do it")
    assert result.status == "completed"
    assert any(record.text.startswith("parser failures") for record in memory.records())
    assert any(record.text.startswith("Completed task") for record in memory.records())
    tool_names = {
        schema["function"]["name"] for schema in client.chat.completions.calls[0]["tools"]
    }
    assert {"bash", "view_image", "remember", "recall", "zoom", "skill"} <= tool_names
    assert "hello" in agent.messages[-2]["content"]
