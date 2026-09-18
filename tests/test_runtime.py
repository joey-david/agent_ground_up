import json

from agent_ground_up.runtime import parse_assistant
from agent_ground_up.tools import TOOL_SCHEMAS


def test_qwen_native_tool_call_is_parsed_without_a_server() -> None:
    text = """<think>
I should inspect the repository.
</think>
<tool_call>
<function=bash>
<parameter=command>
pytest -q
</parameter>
<parameter=timeout_s>
30
</parameter>
</function>
</tool_call>"""

    message = parse_assistant(text, TOOL_SCHEMAS)

    assert "inspect the repository" in message["reasoning_content"]
    assert message["content"] is None
    call = message["tool_calls"][0]
    assert call["function"]["name"] == "bash"
    assert json.loads(call["function"]["arguments"]) == {
        "command": "pytest -q",
        "timeout_s": 30,
    }


def test_plain_answer_stays_plain() -> None:
    message = parse_assistant("<think>work</think>finished", TOOL_SCHEMAS)
    assert message["content"] == "finished"
    assert message["reasoning_content"] == "work"
    assert "tool_calls" not in message
