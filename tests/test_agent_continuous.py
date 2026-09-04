import json
from pathlib import Path

from agent_ground_up.agent import Agent
from agent_ground_up.experience import ExperienceLog
from agent_ground_up.runtime import RuntimeTurn
from agent_ground_up.tools import Toolbox


class FakeRuntime:
    def __init__(self):
        self.tasks = []
        self.calls = []
        self.outputs = []
        self.turns = [
            RuntimeTurn(
                message={
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": json.dumps({"command": "printf hello"}),
                            },
                        }
                    ],
                },
                input_tokens=50,
                output_tokens=10,
                compactions=0,
            ),
            RuntimeTurn(
                message={"role": "assistant", "content": "finished"},
                input_tokens=75,
                output_tokens=5,
                compactions=1,
            ),
        ]

    def reset(self, task: str) -> None:
        self.tasks.append(task)

    def complete(self, *, instructions, tools, max_output_tokens):
        self.calls.append((instructions, tools, max_output_tokens))
        return self.turns.pop(0)

    def submit_tool_output(self, *, call_id, name, content):
        self.outputs.append((call_id, name, content))


def test_agent_uses_continuous_runtime_and_records_exact_experience(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    experience = ExperienceLog(tmp_path / "experience")
    agent = Agent(
        client=None,
        model="gpt-6-astra",
        tools=Toolbox(tmp_path),
        processor=None,
        context_window=262_144,
        max_steps=4,
        runtime=runtime,
        experience=experience,
    )

    result = agent.run("do work")
    assert result.status == "completed"
    assert result.answer == "finished"
    assert result.prompt_tokens == 75
    assert result.compactions == 1
    assert runtime.tasks == ["do work"]
    assert runtime.outputs[0][0:2] == ("call-1", "bash")
    assert "hello" in runtime.outputs[0][2]
    assert "Searchable exact experience log" in runtime.calls[0][0]
    assert any(event.kind == "tool_result" for event in experience.events())
    assert any(event.kind == "run_result" for event in experience.events())
