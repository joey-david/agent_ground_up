from pathlib import Path

from kernel.tools import Toolbox, ToolResult


def test_tool_result_and_head_tail_truncation(tmp_path: Path) -> None:
    assert ToolResult("ok", 0).as_text().endswith("[exit code: 0]")
    toolbox = Toolbox(tmp_path, max_output_tokens=10, token_counter=len)
    text, omitted = toolbox._truncate("abcdefghijklmnopqrstuvwxyz")
    assert text.startswith("abc")
    assert text.endswith("xyz")
    assert omitted > 0


def test_bash_runs_in_workspace(tmp_path: Path) -> None:
    toolbox = Toolbox(tmp_path, token_counter=len)
    result = toolbox.bash("printf hello")
    assert result.output == "hello"
    assert result.returncode == 0
