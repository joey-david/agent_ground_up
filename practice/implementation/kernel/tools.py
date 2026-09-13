from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class ToolResult:
    output: str
    returncode: int
    timed_out: bool = False
    omitted_tokens: int = 0

    def as_text(self) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class ImageResult:
    path: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    data_url: str

    def content(self) -> list[dict[str, Any]]:
        raise NotImplementedError


def arg(type_: str, description: str = "", **extra: Any) -> dict[str, Any]:
    raise NotImplementedError


def tool(name: str, description: str, /, **properties: dict[str, Any]) -> dict[str, Any]:
    raise NotImplementedError


TOOL_SCHEMAS: list[dict[str, Any]] = []  # bash and view_image, built with tool()/arg()


class Toolbox:
    def __init__(
        self,
        workdir: str | Path,
        *,
        max_output_tokens: int = 8192,
        patch_size: int = 32,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        raise NotImplementedError

    def bash(self, command: str, timeout_s: int = 120) -> ToolResult:
        raise NotImplementedError

    def view_image(self, path: str) -> ImageResult:
        raise NotImplementedError

    def _truncate(self, text: str) -> tuple[str, int]:
        raise NotImplementedError

    def _longest_fit(self, text: str, budget: int, *, tail: bool = False) -> str:
        raise NotImplementedError
