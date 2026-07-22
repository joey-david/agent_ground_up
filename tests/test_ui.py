from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from agent_ground_up.ui import TUI, crop_middle


def test_crop_middle_preserves_both_ends() -> None:
    text = "\n".join(f"line {number}" for number in range(10))
    cropped = crop_middle(text, 5)

    assert cropped.splitlines() == [
        "line 0",
        "line 1",
        "... 6 lines omitted ...",
        "line 8",
        "line 9",
    ]


def test_tui_prints_all_event_types() -> None:
    stream = StringIO()
    ui = TUI(max_lines=4, console=Console(file=stream, width=80, color_system=None))
    call = {
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": "git diff"}),
        }
    }

    ui.user("**change** the code")
    ui.assistant({"content": "I will inspect it.", "tool_calls": [call]})
    ui.tool("bash", "diff --git a/a.py b/a.py\n-old = 1\n+new = 2")
    ui.status("status=completed")

    output = stream.getvalue()
    assert all(
        text in output for text in ("User", "Agent", "→ bash", "Tool · bash", "status=completed")
    )
