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


@dataclass(slots=True)
class ToolResult:
    """Captured result of one bash process."""

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
    """Validated image metadata and model-ready bytes."""

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


def arg(type_: str, description: str = "", **extra: Any) -> dict[str, Any]:
    """Describe one tool parameter; `default` marks it optional and documents the fallback."""
    spec: dict[str, Any] = {"type": type_}
    if description:
        spec["description"] = description
    spec.update(extra)
    return spec


def tool(name: str, description: str, /, **properties: dict[str, Any]) -> dict[str, Any]:
    """Build one Chat Completions tool schema from flat parameter specs.

    This is only boilerplate removal: the model still receives the exact JSON Schema written
    out longhand below the wrapper. A parameter is required unless its spec carries a
    `default`, which is the convention every tool here already followed by hand.
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": [key for key, spec in properties.items() if "default" not in spec],
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS = [
    tool(
        "bash",
        "Run a bash command in the workspace and return combined output plus its exit code.",
        command=arg("string", "The command to run."),
        timeout_s=arg("integer", "Maximum runtime in seconds.", default=120),
    ),
    tool(
        "view_image",
        "Open an image file from the workspace and show it to the multimodal model.",
        path=arg("string", "Path relative to the workspace."),
    ),
]


class Toolbox:
    """The two operations available to the coding agent."""

    def __init__(
        self,
        workdir: str | Path,
        *,
        max_output_tokens: int = 8192,
        patch_size: int = 32,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.workdir = Path(workdir).expanduser().resolve()
        self.max_output_tokens = max_output_tokens
        # Vision tiling grid: an image's cost scales with patch_size x patch_size pixel patches,
        # so the value follows the configured model's encoder rather than the workspace.
        self.patch_size = patch_size
        self.token_counter = token_counter or (lambda text: len(text.encode("utf-8")))

    def bash(self, command: str, timeout_s: int = 120) -> ToolResult:
        """Run a command in the workspace.

        Args:
            command: Bash source to execute.
            timeout_s: Maximum runtime in seconds.

        Returns:
            Combined stdout/stderr, exit code, and timeout metadata.
        """
        process = subprocess.Popen(
            command,  # you want to run something, right?
            shell=True,  # but this is python, it can't run anything by itself,
            # it needs to go through a shell
            executable="/bin/bash",  # what program do you want to run in your shell
            cwd=self.workdir,  # and where?
            text=True,  # stdout is bytes by default, we want text instead
            encoding="utf-8",  # how do you want your text, then?
            errors="replace",  # what if you can't decode the text? don't crash replace with <?>.
            stdout=subprocess.PIPE,  # capture stdout instead of default projecting it to screen
            stderr=subprocess.STDOUT,  # merge stderr into stdout, that way we get both
            start_new_session=True,  # create a dedicated process group \
            # that way timeout/kills don't leave orphans.
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
        """Read an image from the workspace.

        Args:
            path: Image path relative to the workspace.

        Returns:
            Image metadata and a data URL for the model request.
        """
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
        """Approximate vision-token cost the way tiling encoders bill it: by patch, not byte."""
        patch = self.patch_size
        return -(-width // patch) * -(-height // patch)

    def _shrink(self, image_path: Path, width: int, height: int) -> tuple[bytes, int, int, str]:
        """Resize once to the largest size whose patch grid fits the token budget.

        Solved analytically from the patch-count budget (area scales with the square of the
        scale factor) instead of repeatedly halving and re-encoding to see what fits. The area
        formula alone can overshoot by a fraction of a patch at the grid edges, so the scale is
        nudged down to the nearest whole patch per side before it's applied.
        """
        patch = self.patch_size
        scale = ((self.max_output_tokens * patch * patch) / (width * height)) ** 0.5
        patches_wide, patches_high = width * scale / patch, height * scale / patch
        scale *= min(int(patches_wide) / patches_wide, int(patches_high) / patches_high)
        width, height = max(1, int(width * scale)), max(1, int(height * scale))
        with Image.open(image_path) as original:
            image = original.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue(), width, height, "image/jpeg"

    def _truncate(self, text: str) -> tuple[str, int]:
        """Fit output to a token budget while preserving its head and tail."""
        count = self.token_counter(text)
        if count <= self.max_output_tokens:
            return text, 0
        budget = self.max_output_tokens // 2
        head = self._longest_fit(text, budget)
        tail = self._longest_fit(text, budget, tail=True)
        omitted = max(0, count - self.token_counter(head) - self.token_counter(tail))
        return f"{head}\n... [{omitted} tokens omitted] ...\n{tail}", omitted

    def _longest_fit(self, text: str, budget: int, *, tail: bool = False) -> str:
        """Longest head (or tail) of `text` that fits the token budget.

        Tokens are not characters, so the character count that fits has to be searched for.
        The token counter is monotonic in slice length, which makes binary search valid and
        costs log2(len(text)) tokenizations instead of one per character.
        """
        take = (lambda n: text[len(text) - n :]) if tail else (lambda n: text[:n])
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self.token_counter(take(middle)) <= budget:
                low = middle
            else:
                high = middle - 1
        return take(low)
