from __future__ import annotations

import base64
import mimetypes
import os
import signal
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(slots=True)
class ToolResult:
    output: str
    returncode: int
    timed_out: bool = False
    omitted_tokens: int = 0

    def as_text(self) -> str:
        suffix = f"\n[exit code: {self.returncode}]"
        if self.timed_out:
            suffix += " [timed out]"
        return self.output + suffix

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImageResult:
    path: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    data_url: str

    def content(self) -> list[dict[str, Any]]:
        description = f"Image: {self.path} ({self.width}x{self.height}, {self.mime_type})"
        return [
            {"type": "text", "text": description},
            {"type": "image_url", "image_url": {"url": self.data_url}},
        ]

    def safe_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key != "data_url"}


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a bash command in the workspace and return combined output plus its exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run."},
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
            "description": "Open an image file from the workspace and show it to the multimodal model.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Path relative to the workspace."}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
]


class Toolbox:
    def __init__(
        self,
        workdir: str | Path,
        *,
        max_output_tokens: int = 8192,
        token_counter: Callable[[str], int] | None = None,
        max_image_bytes: int = 20 * 1024 * 1024,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve(strict=True)
        if not self.workdir.is_dir():
            raise ValueError(f"Workspace is not a directory: {self.workdir}")
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
        if not command.strip():
            return ToolResult("bash: empty command", 2)
        if not 1 <= timeout_s <= 3600:
            return ToolResult("bash: timeout_s must be between 1 and 3600", 2)

        process = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=self.workdir,
            env=os.environ.copy(),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        )
        timed_out = False
        try:
            output, _ = process.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            output, _ = process.communicate()

        output, omitted = self._truncate(output)
        return ToolResult(output, process.returncode if not timed_out else -1, timed_out, omitted)

    def view_image(self, path: str) -> ImageResult:
        """Read an image from the workspace.

        Args:
            path: Image path relative to the workspace.

        Returns:
            Image metadata and a data URL for the model request.
        """
        image_path = (self.workdir / path).resolve(strict=True)
        if not image_path.is_relative_to(self.workdir) or not image_path.is_file():
            raise ValueError("Image must be a file inside the workspace")
        size = image_path.stat().st_size
        if size > self.max_image_bytes:
            raise ValueError(f"Image exceeds {self.max_image_bytes} bytes")
        with Image.open(image_path) as image:
            image.verify()
            width, height = image.size
            mime = Image.MIME.get(image.format) or mimetypes.guess_type(image_path.name)[0] or "image/png"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return ImageResult(
            str(image_path.relative_to(self.workdir)), mime, width, height, size, f"data:{mime};base64,{encoded}"
        )

    def _truncate(self, text: str) -> tuple[str, int]:
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
        omitted = max(0, count - self.token_counter(head) - self.token_counter(tail))
        marker = f"\n... [{omitted} tokens omitted] ...\n"
        return head + marker + tail, omitted
