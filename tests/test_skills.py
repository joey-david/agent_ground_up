from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_ground_up.skills import SkillRegistry


class FakeRunner:
    def __init__(self) -> None:
        self.calls = []

    def bash(self, command: str, timeout_s: int = 120):
        self.calls.append((command, timeout_s))
        return SimpleNamespace(output="ok")


def test_skill_registry_round_trip_and_run(tmp_path: Path) -> None:
    registry = SkillRegistry(tmp_path / "skills")
    registry.register("inspect_repo", "summarize repository", "main() { printf '%s' \"$1\"; }")
    assert registry.get("inspect_repo").description == "summarize repository"
    assert "inspect_repo" in registry.prompt_catalog()

    runner = FakeRunner()
    registry.run("inspect_repo", runner, argument="a b", timeout_s=9)
    command, timeout = runner.calls[0]
    assert "main 'a b'" in command
    assert timeout == 9

    with pytest.raises(ValueError):
        registry.register("../bad", "x", "main() { :; }")
