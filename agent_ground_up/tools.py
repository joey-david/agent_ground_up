from __future__ import annotations

import base64
import io
import mimetypes
import os
import signal
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


def arg(kind: str, description: str = "", **extra: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": kind}
    if description:
        schema["description"] = description
    schema.update(extra)
    return schema


def tool(tool_name: str, tool_description: str, **properties: dict[str, Any]) -> dict[str, Any]:
    required = [name for name, schema in properties.items() if "default" not in schema]
    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


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


@dataclass(slots=True)
class ImageResult:
    path: str
    mime_type: str
    width: int
    height: int
    size_bytes: int
    data_url: str

    def content(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "text",
                "text": f"Image: {self.path} ({self.width}x{self.height}, {self.mime_type})",
            },
            {"type": "image_url", "image_url": {"url": self.data_url}},
        ]


TOOL_SCHEMAS = [
    tool(
        "bash",
        "Run a bash command in the workspace and return combined output plus its exit code.",
        command=arg("string", "The command to run."),
        timeout_s=arg("integer", "Maximum runtime in seconds.", default=120),
    ),
    tool(
        "view_image",
        "Open an image file from the workspace and show it to the model.",
        path=arg("string", "Path relative to the workspace."),
    ),
]


class Toolbox:
    def __init__(
        self,
        workdir: str | Path,
        *,
        max_output_tokens: int = 8192,
        patch_size: int = 16,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.max_output_tokens = max_output_tokens
        self.patch_size = patch_size
        self.token_counter = token_counter or (lambda text: len(text.encode("utf-8")))

    def bash(self, command: str, timeout_s: int = 120) -> ToolResult:
        process = subprocess.Popen(
            command,
            shell=True,
            executable="/bin/bash",
            cwd=self.workdir,
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
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate()

        output, omitted = self._truncate(output)
        return ToolResult(output, process.returncode if not timed_out else -1, timed_out, omitted)

    def view_image(self, path: str) -> ImageResult:
        image_path = (self.workdir / path).resolve(strict=True)
        if not image_path.is_relative_to(self.workdir):
            raise ValueError("Image must be inside the workspace")
        with Image.open(image_path) as image:
            image.verify()
            width, height = image.size
            mime = (
                Image.MIME.get(image.format)
                or mimetypes.guess_type(image_path.name)[0]
                or "image/png"
            )

        if self._patch_count(width, height) <= self.max_output_tokens:
            data = image_path.read_bytes()
        else:
            data, width, height, mime = self._shrink(image_path, width, height)

        encoded = base64.b64encode(data).decode("ascii")
        return ImageResult(
            str(image_path.relative_to(self.workdir)),
            mime,
            width,
            height,
            len(data),
            f"data:{mime};base64,{encoded}",
        )

    def _patch_count(self, width: int, height: int) -> int:
        patch = self.patch_size
        return -(-width // patch) * -(-height // patch)

    def _shrink(self, image_path: Path, width: int, height: int) -> tuple[bytes, int, int, str]:
        patch = self.patch_size
        scale = ((self.max_output_tokens * patch * patch) / (width * height)) ** 0.5
        width = max(1, int(width * min(scale, 1.0)))
        height = max(1, int(height * min(scale, 1.0)))
        while self._patch_count(width, height) > self.max_output_tokens:
            if width >= height:
                width = max(1, width - patch)
            else:
                height = max(1, height - patch)

        with Image.open(image_path) as original:
            image = original.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue(), width, height, "image/jpeg"

    def _truncate(self, text: str) -> tuple[str, int]:
        count = self.token_counter(text)
        if count <= self.max_output_tokens:
            return text, 0
        budget = self.max_output_tokens // 2
        head = self._longest_fit(text, budget)
        tail = self._longest_fit(text, budget, tail=True)
        omitted = max(0, count - self.token_counter(head) - self.token_counter(tail))
        return f"{head}\n... [{omitted} tokens omitted] ...\n{tail}", omitted

    def _longest_fit(self, text: str, budget: int, *, tail: bool = False) -> str:
        take = (lambda n: text[len(text) - n :]) if tail else (lambda n: text[:n])
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.token_counter(take(middle)) <= budget:
                low = middle
            else:
                high = middle - 1
        return take(low)
