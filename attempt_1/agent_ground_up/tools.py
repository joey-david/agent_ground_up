from __future__ import annotations

import base64
import mimetypes
import os
import signal  # for sigkill
from typing import Any
from pathlib import Path
from collections.abc import Callable
from dataclasses import dataclass
import subprocess  # NOTE: spawn processes, collect input/output/pipes, etc

from PIL import Image


# NOTE: a dataclass avoids typing a bunch of boilerplate for a class that
# mainly functions as a data holder. It notably auto-implements the
# following python dunders (built-in protocols):
# 1. __init__, which initializes the properties of an already created (
# with __new__, only overriden for immutable types, etc) instance.
# 2. __repr__, which is the developper/debug representation/print.
# 3. __eq__, for == checking.
@dataclass(slots=True)  # NOTE: slots makes the mutation of attributes more
# rigid, e.g. fixed mem alloc, faster.
class ToolResults:
    """Captured result of a bash process"""

    # includes output text, return code for tracking errors and successes
    # a bool for timed_out to keep the agent moving, and ommited tokens for?
    output: str
    returncode: str
    timed_out: bool = False
    omitted_toks: int = 0

    def as_text(self) -> str:
        suffix = f"\n[exit code: {self.returncode}]"
        if self.timed_out:
            suffix += " [timed out]"
        return self.output + suffix


@dataclass(slots=True)
class ImageResult:
    """Validated image metadata and model-ready bytes"""

    path: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    data_url: str

    def content(self) -> list[dict[str, Any]]:
        description = f"Image: {self.path} \
        ({self.width}x{self.height}, {self.mime_type})"
        return [
            {"type": "text", "text": description},
            {"type": "image_url", "image_url": {"url": self.data_url}},
        ]


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the workspace \
            and return combined results and exit code",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                    "timeout_s": {
                        "type": "integer",
                        "description": "Maximum runtime in seconds.",
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
            "description": "Open an image file from the workspace and \
            show it to the multimodal model",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


class Toolbox:
    # here, we should specify that we chose not to make compaction a tool available to the agent
    # we should probably also justify why end_turn or final_answer isn't a final tool.
    """The two operations available to the coding agent."""

    def __init__(
        self,
        workdir: str | Path,
        *,  # NOTE: force all parameters after this to be passed by name
        max_output_tokens: int = 8192,  # max length of a tool output
        # Callable describes a string signature: takes in str, returns an int
        token_counter: Callable[[str], int] | None = None,
        max_image_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        # .expanduser to add ~ support, .resolve to make the path absolute
        self.workdir = Path(workdir).expanduser().resolve()
        self.max_output_tokens = max_output_tokens
        self.token_counter = token_counter or (lambda text: len(text.encode("utf-8")))
        self.max_image_bytes = max_image_bytes

    def bash(self, command: str, timeout_s: int = 120) -> ToolResult:
        """Run a command in the workspace.

        Args:
            command: Bash source to execute.
            timeout_s: Maximum runtime in seconds.

            Returns:
            Combined stdout/stderr, exit code, and timeout metadata.
        """
        if not 1 <= timeout_s <= 3600:
            return ToolResult("bash: timeout_s must be between 1 and 3600")

        # bash commands are ran through the Popen subprocess interface.
        # start a process running the command with bash in a new session
        process = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cmd=self.workdir,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            # if the process times out, kill it via sigkill with os
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()

        output, ommited = self._truncate(output)

    def view_image(self, path: str) -> ImageResult:
        """Read an image from the workspace.

        Args:
            path: Image path relative to the workspace.

        Returns:
            Image metadata and a data URL for the model request.
        """
        image_path = (self.workdir / path).resolve(strict=True)
        if not image_path.is_relative_to(self.workdir):
            raise ValueError("image must be in workspace")
        size = image_path.stat().st_size
        if size > self.max_image_bytes:
            raise ValueError(
                f"Image exceeds the maximum byte limit of \
            {self.max_image_bytes} bytes"
            )
        with Image.open(image_path) as image:
            image.verify()
            width, height = image.size
            mime = (
                Image.MIME.get(image.format)
                or mimetypes.guess_type(image_path.name)[0]
                or "image/png"
            )
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return ImageResult(
            str(image_path.relative_to(self.workdir)),
            mime,
            width,
            height,
            size,
            f"data:{mime};base64,{encoded}",
        )

    def _truncate(self, text: str) -> tuple[str, int]:
        """Fit output to a token budget while preserving head and tail"""
        count = self.token_counter(text)
        if count <= self.max_output_tokens:
            return text, 0

        target = self.max_output_tokens // 2
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.token_counter(text[:mid]) <= target:
                lo = mid
            else:
                hi = mid - 1
        head = text[:lo]

        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.token_counter(text[len(text) - mid :]) <= target:
                lo = mid
            else:
                hi = mid - 1
        tail = text[len(text) - lo :]
        ommited = max(0, count - self.token_counter(head) - self.token_counter(tail))
        marker = f"\n... [{ommited} tokens omitted] ...\n"
        return head + marker + tail, ommited
