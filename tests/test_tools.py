from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from agent_ground_up.tools import Toolbox


def test_bash_returns_combined_output_and_exit_code(tmp_path: Path) -> None:
    result = Toolbox(tmp_path).bash("printf out; printf err >&2; exit 7")
    assert result.output == "outerr"
    assert result.returncode == 7
    assert "[exit code: 7]" in result.as_text()


def test_bash_timeout_and_head_tail_truncation(tmp_path: Path) -> None:
    toolbox = Toolbox(tmp_path, max_output_tokens=10, token_counter=len)
    truncated = toolbox.bash("printf 12345678901234567890")
    assert truncated.output.startswith("12345")
    assert truncated.output.endswith("67890")
    assert truncated.omitted_tokens == 10

    timed_out = toolbox.bash("sleep 2", timeout_s=1)
    assert timed_out.timed_out
    assert timed_out.returncode == -1


def test_view_image_is_multimodal_and_confined(tmp_path: Path) -> None:
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (2, 3), "red").save(image_path)
    result = Toolbox(tmp_path).view_image("pixel.png")
    assert (result.width, result.height) == (2, 3)
    assert result.data_url.startswith("data:image/png;base64,")
    assert result.content()[1]["type"] == "image_url"

    outside = tmp_path.parent / "outside.png"
    Image.new("RGB", (1, 1)).save(outside)
    with pytest.raises(ValueError, match="inside the workspace"):
        Toolbox(tmp_path).view_image("../outside.png")
