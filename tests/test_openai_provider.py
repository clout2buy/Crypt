from __future__ import annotations

import json

from core.api import OpenAIProvider, ToolUseProgress, TurnEnd


class _FakeStreamResponse:
    status_code = 200
    text = ""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return b""

    def iter_text(self):
        yield from self._chunks


class _FakeHTTP:
    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks
        self.calls: list[dict] = []

    def stream(self, method, url, *, json, headers):
        self.calls.append({"method": method, "url": url, "json": json, "headers": headers})
        return _FakeStreamResponse(self._chunks)


def _sse(data: dict) -> str:
    return "data: " + json.dumps(data) + "\n\n"


def test_openai_tool_progress_includes_partial_json():
    first_args = '{"path":"demo.html"'
    second_args = ',"content":"<p>hi</p>"}'
    chunks = [
        _sse({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_1",
                        "function": {
                            "name": "write_file",
                            "arguments": first_args,
                        },
                    }],
                },
                "finish_reason": None,
            }],
        }),
        _sse({
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "function": {"arguments": second_args},
                    }],
                },
                "finish_reason": None,
            }],
        }),
        _sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]\n\n",
    ]
    provider = OpenAIProvider(
        model="gpt-test",
        api_key="test-key",
        base_url="http://openai.test/v1",
    )
    provider._http = _FakeHTTP(chunks)

    out = list(provider.stream_turn(messages=[], tools=[], system="sys"))

    progress = [event for event in out if isinstance(event, ToolUseProgress)]
    assert progress[-1].name == "write_file"
    assert progress[-1].call_id == "call_1"
    assert progress[-1].partial_json == first_args + second_args
    assert progress[-1].argument_chars == len(first_args + second_args)

    turn_end = [event for event in out if isinstance(event, TurnEnd)][0]
    assert turn_end.stop_reason == "tool_use"
    assert turn_end.message["content"][0]["input"] == {
        "path": "demo.html",
        "content": "<p>hi</p>",
    }
