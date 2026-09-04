from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


class ShellRunner(Protocol):
    def bash(self, command: str, timeout_s: int = 120) -> Any: ...


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    script: str


class SkillRegistry:
    def __init__(self, root: str | Path) -> None:
        raise NotImplementedError

    def register(self, name: str, description: str, script: str) -> Skill:
        raise NotImplementedError

    def get(self, name: str) -> Skill:
        raise NotImplementedError

    def list(self) -> list[Skill]:
        raise NotImplementedError

    def remove(self, name: str) -> None:
        raise NotImplementedError

    def run(
        self,
        name: str,
        runner: ShellRunner,
        *,
        argument: str = "",
        timeout_s: int = 120,
    ) -> Any:
        raise NotImplementedError

    def prompt_catalog(self) -> str:
        raise NotImplementedError

    @staticmethod
    def _validate_name(name: str) -> None:
        raise NotImplementedError
