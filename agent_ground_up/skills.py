from __future__ import annotations

import json
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

_NAME = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class ShellRunner(Protocol):
    def bash(self, command: str, timeout_s: int = 120) -> Any: ...


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    script: str


class SkillRegistry:
    """Persistent registry of agent-written shell tools that run inside the normal workspace."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def register(self, name: str, description: str, script: str) -> Skill:
        self._validate_name(name)
        if not description.strip() or not script.strip():
            raise ValueError("skill description and script are required")
        skill = Skill(name=name, description=description.strip(), script=script.rstrip() + "\n")
        path = self.root / f"{name}.json"
        path.write_text(
            json.dumps(asdict(skill), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return skill

    def get(self, name: str) -> Skill:
        self._validate_name(name)
        path = self.root / f"{name}.json"
        if not path.exists():
            raise KeyError(f"unknown skill: {name}")
        return Skill(**json.loads(path.read_text(encoding="utf-8")))

    def list(self) -> list[Skill]:
        skills = []
        for path in sorted(self.root.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            skills.append(Skill(**data))
        return skills

    def remove(self, name: str) -> None:
        self._validate_name(name)
        path = self.root / f"{name}.json"
        if path.exists():
            path.unlink()

    def run(
        self,
        name: str,
        runner: ShellRunner,
        *,
        argument: str = "",
        timeout_s: int = 120,
    ) -> Any:
        """Execute the stored script with one quoted argument via the existing bash boundary."""
        skill = self.get(name)
        command = "set -euo pipefail\n" + skill.script + "\n"
        command += f"\nmain {shlex.quote(argument)}\n"
        return runner.bash(command, timeout_s=timeout_s)

    def prompt_catalog(self) -> str:
        skills = self.list()
        if not skills:
            return "Generated skills: none."
        rows = ["Generated skills (call skill by name when useful):"]
        rows.extend(f"- {skill.name}: {skill.description}" for skill in skills)
        return "\n".join(rows)

    @staticmethod
    def _validate_name(name: str) -> None:
        if not _NAME.fullmatch(name):
            raise ValueError("skill names must match [a-z][a-z0-9_-]{0,63}")
