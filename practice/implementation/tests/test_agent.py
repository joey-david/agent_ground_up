from pathlib import Path

from kernel.agent import Agent
from kernel.experience import ExperienceLog
from kernel.tools import Toolbox


class Runtime:
    def __init__(self) -> None:
        self.turn = 0

    def complete(self, messages, tools, max_output_tokens):
        self.turn += 1
        if self.turn == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "1", "function": {"name": "bash", "arguments": '{"command":"printf done"}'}}],
            }
        return {"role": "assistant", "content": "finished"}


def test_agent_loops_through_tool_then_finishes(tmp_path: Path) -> None:
    experience = ExperienceLog(tmp_path / "history")
    agent = Agent(Runtime(), Toolbox(tmp_path, token_counter=len), experience=experience, max_steps=4)
    result = agent.run("do it")
    assert result.status == "completed"
    assert result.answer == "finished"
    assert result.valid_tool_calls == 1
    assert experience.count() >= 3
