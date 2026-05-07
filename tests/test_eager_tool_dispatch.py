from __future__ import annotations

from core import loop, runtime
import pytest

from core.api import (
    ThinkingDelta,
    TextDelta,
    ToolUseProgress,
    ToolUseReady,
    TurnEnd,
    _finalized_assistant_message,
    _process_event,
)
from tools import registry
from tools.types import Tool


class _FakeProvider:
    name = "fake"
    model = "fake-model"
    is_oauth = False

    def stream_turn(self, messages, tools, system):
        yield TextDelta("I'll do that.")
        yield TurnEnd(
            stop_reason="tool_use",
            message={
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I'll do that."},
                    {
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    },
                ],
            },
        )


def test_stream_one_turn_cuts_over_to_ready_tool(workspace):
    provider = _FakeProvider()
    runtime.configure(provider, str(workspace), session=None)

    end = loop._stream_one_turn(
        provider,
        messages=[],
        tools=[],
        loader=loop._SilentLoader(),
        render=False,
    )

    assert end.stop_reason == "tool_use"
    assert end.message["content"][1]["name"] == "read_file"
    assert end.message["content"][1]["input"] == {"path": "README.md"}


def test_anthropic_block_stop_emits_tool_ready_before_full_message():
    content_blocks = []
    usage = {"input_tokens": 10, "output_tokens": 3}

    list(_process_event(
        "content_block_start",
        {
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "read_file",
            },
        },
        content_blocks,
        usage,
    ))
    list(_process_event(
        "content_block_delta",
        {
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"path":"README.md"}',
            },
        },
        content_blocks,
        usage,
    ))

    events = list(_process_event(
        "content_block_stop",
        {"index": 0},
        content_blocks,
        usage,
    ))

    ready = [event for event in events if isinstance(event, ToolUseReady)]
    assert len(ready) == 1
    assert ready[0].message["content"][0]["input"] == {"path": "README.md"}
    assert _finalized_assistant_message(content_blocks)["content"] == [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "read_file",
            "input": {"path": "README.md"},
        }
    ]


def test_anthropic_multiple_tool_blocks_survive_until_final_message():
    content_blocks = []
    usage = {"input_tokens": 10, "output_tokens": 3}

    for idx, tool_id, path in [
        (0, "toolu_1", "a.py"),
        (1, "toolu_2", "b.py"),
    ]:
        list(_process_event(
            "content_block_start",
            {
                "index": idx,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "read_file",
                },
            },
            content_blocks,
            usage,
        ))
        list(_process_event(
            "content_block_delta",
            {
                "index": idx,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": f'{{"path":"{path}"}}',
                },
            },
            content_blocks,
            usage,
        ))
        events = list(_process_event(
            "content_block_stop",
            {"index": idx},
            content_blocks,
            usage,
        ))
        ready = [event for event in events if isinstance(event, ToolUseReady)]
        assert len(ready) == 1
        assert ready[0].message["content"][-1]["input"] == {"path": path}

    msg = _finalized_assistant_message(content_blocks)
    assert [b["id"] for b in msg["content"]] == ["toolu_1", "toolu_2"]
    assert [b["input"]["path"] for b in msg["content"]] == ["a.py", "b.py"]


def test_multi_tool_assistant_message_uses_parallel_dispatch(monkeypatch):
    class Provider:
        is_oauth = False

    calls: list[list[str]] = []

    def fake_parallel(blocks, *, render):
        calls.append([block["id"] for block in blocks])
        return [
            {
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": "ok",
                "is_error": False,
            }
            for block in blocks
        ]

    tool = Tool(
        name="parallel_stub",
        description="stub",
        schema={"type": "object", "properties": {}, "required": []},
        permission="auto",
        run=lambda args: "ok",
        parallel_safe=True,
    )
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    monkeypatch.setattr(loop, "_dispatch_parallel", fake_parallel)

    assistant_msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "call_1", "name": "parallel_stub", "input": {}},
            {"type": "tool_use", "id": "call_2", "name": "parallel_stub", "input": {}},
        ],
    }
    messages = [assistant_msg]

    loop._dispatch_tool_uses(Provider(), assistant_msg, messages, record=False, render=False)

    assert calls == [["call_1", "call_2"]]
    assert [b["tool_use_id"] for b in messages[-1]["content"]] == ["call_1", "call_2"]


def test_anthropic_json_delta_emits_tool_progress():
    content_blocks = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    start_events = list(_process_event(
        "content_block_start",
        {
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "write_file",
            },
        },
        content_blocks,
        usage,
    ))
    delta_events = list(_process_event(
        "content_block_delta",
        {
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"path":"demo.html"',
            },
        },
        content_blocks,
        usage,
    ))

    progress = [event for event in start_events + delta_events if isinstance(event, ToolUseProgress)]
    assert len(progress) == 2
    assert progress[-1].name == "write_file"
    assert progress[-1].call_id == "toolu_1"
    assert progress[-1].argument_chars == len('{"path":"demo.html"')
    assert progress[-1].partial_json == '{"path":"demo.html"'


def test_tool_progress_detail_surfaces_partial_write_file():
    event = ToolUseProgress(
        name="write_file",
        call_id="toolu_1",
        argument_chars=62,
        partial_json='{"path":"demo.html","content":"<html>\\n<body>hi',
    )

    detail = loop._tool_progress_detail(event)

    assert "demo.html" in detail
    assert "2 line(s)" in detail
    assert "chars" in detail


def test_tool_progress_preview_surfaces_tail_of_partial_file():
    event = ToolUseProgress(
        name="write_file",
        call_id="toolu_1",
        argument_chars=120,
        partial_json='{"path":"demo.html","content":"<html>\\n<body>\\n<canvas id=\\"c\\">\\n<script>tick()',
    )

    preview = loop._tool_progress_preview(event)

    assert preview[-2:] == ['<canvas id="c">', "<script>tick()"]


def test_hidden_reasoning_stall_aborts(monkeypatch, workspace):
    class SlowThinkingProvider:
        name = "fake"
        model = "fake-model"
        is_oauth = False

        def stream_turn(self, messages, tools, system):
            yield ThinkingDelta("still thinking")
            yield ThinkingDelta("still thinking")

    times = iter([0.0, 0.0, 46.0])
    monkeypatch.setenv("CRYPT_REASONING_STALL_SECONDS", "45")
    monkeypatch.setattr(loop.time, "monotonic", lambda: next(times))
    runtime.configure(SlowThinkingProvider(), str(workspace), session=None)
    runtime.set_show_thinking(False)

    with pytest.raises(RuntimeError, match="too long in reasoning"):
        loop._stream_one_turn(
            SlowThinkingProvider(),
            messages=[{"role": "user", "content": "make a file"}],
            tools=[],
            loader=loop._SilentLoader(),
            render=True,
        )


class _TextThenToolProvider:
    name = "fake"
    model = "fake-model"
    is_oauth = False

    def __init__(self) -> None:
        self.calls = 0
        self.seen_messages: list[list[dict]] = []

    def stream_turn(self, messages, tools, system):
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls == 1:
            yield TextDelta("```html\n")
            raise AssertionError("artifact text stream should be cut immediately")
        if self.calls == 2:
            yield TurnEnd(
                stop_reason="tool_use",
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "stub",
                            "input": {"x": "ok"},
                        }
                    ],
                },
            )
            return
        yield TextDelta("done")
        yield TurnEnd(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        )


class _TextThenToolNoRenderProvider(_TextThenToolProvider):
    def stream_turn(self, messages, tools, system):
        self.calls += 1
        self.seen_messages.append(list(messages))
        if self.calls == 1:
            text = "```html\n" + ("<section>artifact</section>\n" * 20) + "```"
            yield TextDelta(text)
            yield TurnEnd(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"type": "text", "text": text}]},
            )
            return
        if self.calls == 2:
            yield TurnEnd(
                stop_reason="tool_use",
                message={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "stub",
                            "input": {"x": "ok"},
                        }
                    ],
                },
            )
            return
        yield TextDelta("done")
        yield TurnEnd(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        )


def test_text_only_artifact_answer_is_retried_with_tools(monkeypatch, workspace):
    provider = _TextThenToolNoRenderProvider()
    runtime.configure(provider, str(workspace), session=None)
    tool = Tool(
        name="stub",
        description="stub",
        schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        permission="auto",
        run=lambda args: f"ran {args['x']}",
    )
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)

    messages = [{"role": "user", "content": "make me a super advanced html"}]
    loop._run_until_done(provider, messages, 0, 0, render=False)

    assert provider.calls == 3
    assert any(
        "pasted the artifact in chat" in block.get("text", "")
        for msg in messages
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict)
    )
    assert any(
        block.get("type") == "tool_result" and "ran ok" in block.get("content", "")
        for msg in messages
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict)
    )


def test_rendered_artifact_stream_is_cut_before_full_generation(monkeypatch, workspace):
    provider = _TextThenToolProvider()
    runtime.configure(provider, str(workspace), session=None)
    tool = Tool(
        name="stub",
        description="stub",
        schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        permission="auto",
        run=lambda args: f"ran {args['x']}",
    )
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)

    messages = [{"role": "user", "content": "make me a super advanced html"}]
    loop._run_until_done(provider, messages, 0, 0, render=True)

    assert provider.calls == 3
    assert any(
        "pasted the artifact in chat" in block.get("text", "")
        for msg in messages
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict)
    )
    assert any(
        block.get("type") == "tool_result" and "ran ok" in block.get("content", "")
        for msg in messages
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict)
    )


class _PlanThenToolProvider:
    name = "fake"
    model = "fake-model"
    is_oauth = False

    def __init__(self) -> None:
        self.calls = 0

    def stream_turn(self, messages, tools, system):
        self.calls += 1
        if self.calls == 1:
            yield TurnEnd(
                stop_reason="tool_use",
                message={
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "plan_1",
                        "name": "present_plan",
                        "input": {"title": "Demo", "plan": "Create one HTML file."},
                    }],
                },
            )
            return
        if self.calls == 2:
            yield TurnEnd(
                stop_reason="tool_use",
                message={
                    "role": "assistant",
                    "content": [{
                        "type": "tool_use",
                        "id": "call_1",
                        "name": "stub",
                        "input": {"x": "ok"},
                    }],
                },
            )
            return
        yield TurnEnd(
            stop_reason="end_turn",
            message={"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        )


def test_present_plan_runs_for_simple_artifact_request(monkeypatch, workspace):
    provider = _PlanThenToolProvider()
    runtime.configure(provider, str(workspace), session=None)
    tool = Tool(
        name="stub",
        description="stub",
        schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        permission="auto",
        run=lambda args: f"ran {args['x']}",
    )
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)

    messages = [{"role": "user", "content": "make me a super advanced html and open it"}]
    loop._run_until_done(provider, messages, 0, 0, render=False)

    assert provider.calls == 3
    assert any(
        block.get("tool_use_id") == "plan_1"
        and "approved. proceed" in block.get("content", "")
        for msg in messages
        for block in (msg.get("content") if isinstance(msg.get("content"), list) else [])
        if isinstance(block, dict)
    )


def test_artifact_request_shows_real_stream_activity(monkeypatch, workspace):
    provider = _FakeProvider()
    runtime.configure(provider, str(workspace), session=None)
    seen: list[str] = []

    monkeypatch.setattr(loop.ui, "activity", lambda label: seen.append(label))
    monkeypatch.setattr(loop.ui, "tool_progress_clear", lambda: None)

    loop._stream_one_turn(
        provider,
        messages=[{"role": "user", "content": "make me an animated html"}],
        tools=[],
        loader=loop._SilentLoader(),
        render=True,
    )

    assert "waiting for provider response" in seen
    assert "receiving text stream" in seen
    assert "response complete" in seen


def test_real_stream_activity_survives_after_plan_approval(monkeypatch, workspace):
    provider = _FakeProvider()
    runtime.configure(provider, str(workspace), session=None)
    seen: list[str] = []

    monkeypatch.setattr(loop.ui, "activity", lambda label: seen.append(label))
    monkeypatch.setattr(loop.ui, "tool_progress_clear", lambda: None)

    loop._stream_one_turn(
        provider,
        messages=[
            {"role": "user", "content": "make me an animated html and open it"},
            {
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "plan_1",
                    "name": "present_plan",
                    "input": {"title": "Demo", "plan": "Create one HTML file."},
                }],
            },
            {
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "plan_1",
                    "content": "approved. proceed with execution.",
                }],
            },
        ],
        tools=[],
        loader=loop._SilentLoader(),
        render=True,
    )

    assert "waiting for provider response" in seen
    assert "receiving text stream" in seen
    assert "response complete" in seen
