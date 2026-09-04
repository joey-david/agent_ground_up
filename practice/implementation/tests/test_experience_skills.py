from pathlib import Path

from kernel.experience import ExperienceLog
from kernel.skills import SkillRegistry


class Runner:
    def bash(self, command: str, timeout_s: int = 120):
        self.command = command
        self.timeout_s = timeout_s
        return command


def test_exact_experience_and_generated_skills(tmp_path: Path) -> None:
    log = ExperienceLog(tmp_path / "history")
    log.append("observation", {"text": "alpha"})
    log.append("tool_result", {"text": "beta"})
    assert log.count() == 2
    assert log.search("beta")[0].kind == "tool_result"
    assert "alpha" in log.format(log.read(0, 1))

    skills = SkillRegistry(tmp_path / "skills")
    skills.register("echo_arg", "echo the argument", "main() { printf '%s' \"$1\"; }")
    runner = Runner()
    skills.run("echo_arg", runner, argument="hello")
    assert "main" in runner.command and "hello" in runner.command
    assert "echo_arg" in skills.prompt_catalog()
