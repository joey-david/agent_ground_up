from __future__ import annotations

from typing import Any
from collections.abc import Callable
import base64
from pathlib import Path
from PIL import Image
from dataclasses import dataclass

from agent_ground_up.tools import TOOL_SCHEMAS
# dataclass for tool and image result


@dataclass(slots=True)
class ToolResult:
    """Captured result of a bash process"""

    output: str
    returncode: int
    timed_out: bool = False
    omitted_tokens: int = 0

    # create a text representation with the exit code and a timeout indicator
    # and the output first, of course
    def as_text(self) -> str:
        suffix = f"\exit code: {self.returncode}"
        if self.timed_out:
            suffix += " [timed_out] "
        return self.output + suffix


@dataclass(slots=True)
class ImageResult:
    """Validated image metadata and model-ready bytes"""

    path: str
    mime_type: str
    data_url: str
    width: int
    height: int
    size_bytes: int

    # return a quick description of the metadaata of the image, plus its url


def content(self) -> list[dict[str, Any]]:
    description = f"Image: {self.path} ({self.width} x {self.height}), {self.mime_type}"
    return [
        {"type": "text", "text": description},
        {"type": "image_url", "image_url": self.data_url},
    ]


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command in the workspace and return its combined output and exit code"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
                    "timeout_s": {
                        "type": "integer",
                        "description": "Maximum runtime in seconds",
                        "default": 120,
                    },
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_image",
            "description": (
                "Open an image file from the workspace and show it to the multimodal model running the agent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace."},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


class Toolbox:
    """The two operations available to the coding agent"""

    def __init__(
        self,
        workdir: str | Path,
        *,
        max_output_tokens: int = 8192,
        token_counter: Callable[[str], int] | None = None,
        max_image_bytes: int = 20 * 2048 * 2048,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
