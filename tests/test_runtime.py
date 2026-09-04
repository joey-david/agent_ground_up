from types import SimpleNamespace

from agent_ground_up.runtime import ContinuousResponsesRuntime


class Item:
    def __init__(self, **data):
        self.data = data

    def model_dump(self, **kwargs):
        return dict(self.data)


class Responses:
    def __init__(self):
        self.calls = []
        self.outputs = [
            [
                Item(type="reasoning", id="r1", encrypted_content="opaque", status="completed"),
                Item(
                    type="function_call",
                    id="fc1",
                    call_id="call-1",
                    name="bash",
                    arguments='{"command":"pwd"}',
                    status="completed",
                    phase="analysis",
                ),
            ],
            [
                Item(type="compaction", id="cmp1", encrypted_content="compact", created_by="server"),
                Item(
                    type="message",
                    id="m1",
                    role="assistant",
                    content=[{"type": "output_text", "text": "done"}],
                    status="completed",
                ),
            ],
        ]

    def create(self, **kwargs):
        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return SimpleNamespace(
            output=output,
            usage=SimpleNamespace(input_tokens=321, output_tokens=45),
        )


class Client:
    def __init__(self):
        self.responses = Responses()


def tool_schema():
    return {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "run shell",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }


def test_continuous_runtime_replays_native_state_and_provider_compaction() -> None:
    client = Client()
    runtime = ContinuousResponsesRuntime(client, "gpt-6-astra", compact_threshold=175_000)
    runtime.reset("fix it")

    first = runtime.complete(instructions="agent", tools=[tool_schema()], max_output_tokens=1000)
    request = client.responses.calls[0]
    assert request["store"] is False
    assert request["include"] == ["reasoning.encrypted_content"]
    assert request["reasoning"]["context"] == "auto"
    assert request["context_management"] == [
        {"type": "compaction", "compact_threshold": 175_000}
    ]
    assert request["tools"][0]["name"] == "bash"
    assert "function" not in request["tools"][0]
    assert first.message["tool_calls"][0]["id"] == "call-1"
    assert first.input_tokens == 321
    assert runtime.history[1]["type"] == "reasoning"
    assert "status" not in runtime.history[1]
    assert runtime.history[2]["phase"] == "analysis"

    runtime.submit_tool_output(call_id="call-1", name="bash", content="/tmp\n[exit code: 0]")
    second = runtime.complete(instructions="agent", tools=[tool_schema()], max_output_tokens=1000)
    replay = client.responses.calls[1]["input"]
    assert any(item.get("type") == "reasoning" for item in replay)
    assert any(item.get("type") == "function_call" for item in replay)
    assert replay[-1]["type"] == "function_call_output"
    assert replay[-1]["call_id"] == "call-1"
    assert second.message["content"] == "done"
    assert second.compactions == 1
    assert runtime.history[0]["type"] == "compaction"
    assert "created_by" not in runtime.history[0]
