from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .experience import ExperienceLog
from .memory import ConstantMemory
from .skills import SkillRegistry
from .tools import Toolbox


class Runtime(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_output_tokens: int,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class RunResult:
    status: str
    answer: str
    steps: int
    valid_tool_calls: int
    invalid_tool_calls: int


class Agent:
    def __init__(
        self,
        runtime: Runtime,
        tools: Toolbox,
        *,
        memory: ConstantMemory | None = None,
        experience: ExperienceLog | None = None,
        skills: SkillRegistry | None = None,
        max_output_tokens: int = 4096,
        max_steps: int = 80,
    ) -> None:
        raise NotImplementedError

    def run(self, task: str) -> RunResult:
        raise NotImplementedError

    def _tool_schemas(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    def _execute(self, call: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def _canonical_state(self) -> str:
        raise NotImplementedError

    def _record(self, kind: str, payload: Any) -> None:
        raise NotImplementedError
