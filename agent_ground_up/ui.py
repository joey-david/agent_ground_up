from __future__ import annotations

import json
from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text


def crop_middle(text: str, max_lines: int) -> str:
    """Keep the beginning and end of a long string, removing its middle."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    head = max_lines // 2
    tail = max_lines - head - 1
    marker = f"... {len(lines) - head - tail} lines omitted ..."
    kept = [*lines[:head], marker]
    if tail:
        kept.extend(lines[-tail:])
    return "\n".join(kept)


class TUI:
    """Render agent turns and tool activity to a terminal."""

    def __init__(self, max_lines: int = 40, console: Console | None = None) -> None:
        self.max_lines = max_lines
        self.console = console or Console(highlight=False)

    def user(self, text: str) -> None:
        """Render a Markdown user turn."""
        self._show("User", text, "green", markdown=True)

    def assistant(self, message: dict[str, Any], title: str = "Agent") -> None:
        """Render an assistant turn and each tool call it requested."""
        self.console.print(Rule(f"[bold magenta]{title}[/]"))
        if content := message.get("content"):
            self.console.print(Markdown(crop_middle(content, self.max_lines)))
        for call in message.get("tool_calls", []):
            function = call["function"]
            raw_arguments = function["arguments"]
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            name = function["name"]
            source = raw_arguments
            if isinstance(arguments, dict):
                source = arguments.get("command") or arguments.get("path") or raw_arguments
            language = "bash" if name == "bash" else "text"
            self.console.print(f"[bold cyan]→ {name}[/]")
            self.console.print(Syntax(crop_middle(source, self.max_lines), language, word_wrap=True))

    def tool(self, name: str, text: str) -> None:
        """Render a tool observation, highlighting unified diffs when present."""
        is_diff = any(line.startswith(("diff --git", "@@ ", "+++ ", "--- ")) for line in text.splitlines())
        language = "diff" if is_diff else None
        self._show(f"Tool · {name}", text, "cyan", language=language)

    def status(self, text: str) -> None:
        """Render the final run status."""
        self.console.print(f"[dim]{text}[/]")

    def _show(
        self,
        title: str,
        text: str,
        color: str,
        *,
        markdown: bool = False,
        language: str | None = None,
    ) -> None:
        """Render one capped string through the requested Rich formatter."""
        self.console.print(Rule(f"[bold {color}]{title}[/]"))
        text = crop_middle(text, self.max_lines)
        if markdown:
            self.console.print(Markdown(text))
        elif language:
            self.console.print(Syntax(text, language, word_wrap=True))
        else:
            self.console.print(Text(text))
