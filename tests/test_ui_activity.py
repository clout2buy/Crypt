from __future__ import annotations

import pytest

from core import ui


def test_status_uses_real_activity_label(monkeypatch):
    monkeypatch.setattr(ui.time, "monotonic", lambda: 105.0)
    ui._state["loader_start"] = 100.0
    ui._state["activity"] = "waiting for provider response"
    try:
        rendered = ui._build_status()
        assert "waiting for provider response" in rendered.plain
        assert "5s" in rendered.plain
        assert "thinking" not in rendered.plain
        assert "writing" not in rendered.plain
    finally:
        ui._state["loader_start"] = 0.0
        ui._state["activity"] = "idle"


def _reset_lifecycle():
    ui._state["tool_lifecycle"] = {}
    ui._state["tool_lifecycle_order"] = []


def test_tool_lifecycle_registers_in_flight_row():
    _reset_lifecycle()
    try:
        ui.tool_begin("call_1", "bash", "echo hi")
        assert "call_1" in ui._state["tool_lifecycle"]
        entry = ui._state["tool_lifecycle"]["call_1"]
        assert entry["state"] == "queued"
        assert entry["name"] == "bash"
        assert ui._state["tool_lifecycle_order"] == ["call_1"]
    finally:
        _reset_lifecycle()


def test_tool_lifecycle_state_transitions():
    _reset_lifecycle()
    try:
        ui.tool_begin("call_1", "bash", "echo hi")
        ui.tool_set_state("call_1", "running")
        assert ui._state["tool_lifecycle"]["call_1"]["state"] == "running"
        assert ui._state["tool_lifecycle"]["call_1"]["detail"] == "running"

        ui.tool_set_state("call_1", "approval", "awaiting approval (rm -rf)")
        assert ui._state["tool_lifecycle"]["call_1"]["state"] == "approval"
        assert "rm -rf" in ui._state["tool_lifecycle"]["call_1"]["detail"]
    finally:
        _reset_lifecycle()


def test_tool_end_removes_from_in_flight():
    _reset_lifecycle()
    try:
        ui.tool_begin("call_1", "bash", "echo hi")
        ui.tool_end("call_1", ok=True, output="")
        assert "call_1" not in ui._state["tool_lifecycle"]
        assert "call_1" not in ui._state["tool_lifecycle_order"]
    finally:
        _reset_lifecycle()


def test_tool_lifecycle_clear_drops_leftover_rows():
    _reset_lifecycle()
    try:
        ui.tool_begin("call_1", "a", "x")
        ui.tool_begin("call_2", "b", "y")
        ui.tool_lifecycle_clear()
        assert ui._state["tool_lifecycle"] == {}
        assert ui._state["tool_lifecycle_order"] == []
    finally:
        _reset_lifecycle()


def test_in_flight_render_shows_state_and_elapsed(monkeypatch):
    _reset_lifecycle()
    try:
        # t=100: tool begins (queued)
        monkeypatch.setattr(ui.time, "monotonic", lambda: 100.0)
        ui.tool_begin("call_1", "bash", "echo hi")
        # t=104: user has been deliberating for 4s; not approved yet.
        # State is still 'queued', and the displayed elapsed is 0 because
        # running_at hasn't been stamped — approval delay is not run time.
        monkeypatch.setattr(ui.time, "monotonic", lambda: 104.0)
        ui.tool_set_state("call_1", "running")  # approved; running clock starts here
        # t=107: tool has been running for 3s.
        monkeypatch.setattr(ui.time, "monotonic", lambda: 107.0)

        group = ui._build_in_flight_tools()
        assert group is not None
        plain = "\n".join(t.plain for t in group.renderables)
        assert "running" in plain
        assert "(3s)" in plain  # 107 - 104, NOT 107 - 100
    finally:
        _reset_lifecycle()


def test_in_flight_render_returns_none_when_empty():
    _reset_lifecycle()
    assert ui._build_in_flight_tools() is None


def test_tool_set_state_on_unknown_id_is_safe():
    _reset_lifecycle()
    # Should not raise even though the id was never registered.
    ui.tool_set_state("missing", "running")
    assert "missing" not in ui._state["tool_lifecycle"]


def test_tool_begin_without_id_falls_back_to_legacy(monkeypatch):
    """No tool_use_id → legacy single-shot tool_call, no lifecycle entry."""
    _reset_lifecycle()
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(ui, "tool_call", lambda name, summary: seen.append((name, summary)))
    ui.tool_begin("", "bash", "echo hi")
    assert seen == [("bash", "echo hi")]
    assert ui._state["tool_lifecycle"] == {}


def test_elapsed_excludes_approval_typing_time(monkeypatch):
    """Approval delay must NOT inflate the displayed run time.

    Regression for the case where 'echo hi (4.8s)' was really 'approved
    after 4.7s, ran in 0.1s' — the user reads the timer as execution speed.
    """
    _reset_lifecycle()
    try:
        # t=0: tool begin (queued)
        monkeypatch.setattr(ui.time, "monotonic", lambda: 0.0)
        ui.tool_begin("call_1", "bash", "echo hi")

        # t=5: user finally typed 'y'; tool now running.
        monkeypatch.setattr(ui.time, "monotonic", lambda: 5.0)
        ui.tool_set_state("call_1", "running")

        # t=5.1: tool finished. Footer should show 0.1s, not 5.1s.
        monkeypatch.setattr(ui.time, "monotonic", lambda: 5.1)
        entry = ui._state["tool_lifecycle"]["call_1"]
        assert ui._elapsed_since(entry) == pytest.approx(0.1, abs=0.01)
    finally:
        _reset_lifecycle()


def test_elapsed_falls_back_to_started_at_when_never_ran():
    """If the tool was denied at approval, no running_at is stamped — the
    elapsed should still be meaningful (queue/approval duration)."""
    _reset_lifecycle()
    try:
        ui.tool_begin("call_1", "bash", "echo hi")
        entry = ui._state["tool_lifecycle"]["call_1"]
        assert "running_at" not in entry
        # Should not crash, should return a small non-negative number.
        elapsed = ui._elapsed_since(entry)
        assert elapsed >= 0.0
    finally:
        _reset_lifecycle()


def test_spinner_advances_with_time(monkeypatch):
    """The status spinner must advance with the wall clock so the user
    sees motion during silent provider waits (no chunks arriving)."""
    monkeypatch.setattr(ui.time, "monotonic", lambda: 0.0)
    a = ui._spinner_frame()
    monkeypatch.setattr(ui.time, "monotonic", lambda: 0.3)
    b = ui._spinner_frame()
    # Both glyphs are valid spinner frames, and time has advanced enough
    # for the index to change (10 fps * 0.3s = 3 frames forward).
    assert a in ui._SPINNER_FRAMES
    assert b in ui._SPINNER_FRAMES
    assert a != b


def test_status_includes_spinner(monkeypatch):
    monkeypatch.setattr(ui.time, "monotonic", lambda: 105.0)
    ui._state["loader_start"] = 100.0
    ui._state["activity"] = "waiting for provider response"
    try:
        rendered = ui._build_status()
        # Spinner glyph appears as the very first non-space char.
        first_glyph = rendered.plain.lstrip()[0]
        assert first_glyph in ui._SPINNER_FRAMES
    finally:
        ui._state["loader_start"] = 0.0
        ui._state["activity"] = "idle"


def test_activity_does_not_refresh_when_label_is_unchanged(monkeypatch):
    class Live:
        def __init__(self):
            self.calls = 0

        def update(self, *args, **kwargs):
            self.calls += 1

    live = Live()
    ui._state["live"] = live
    ui._state["activity"] = "planning file tool call"
    try:
        ui.activity("planning file tool call")
        assert live.calls == 0
        ui.activity("receiving tool args: write_file")
        assert live.calls == 1
    finally:
        ui._state["live"] = None
        ui._state["activity"] = "idle"


def test_status_shows_abort_hint_after_long_wait(monkeypatch):
    """After 15s of no chunks, the status surfaces ctrl+c / /model hint."""
    monkeypatch.setattr(ui.time, "monotonic", lambda: 120.0)
    ui._state["loader_start"] = 100.0  # 20s elapsed
    ui._state["activity"] = "waiting for provider response"
    ui._state["stream_chars"] = 0  # nothing received yet
    try:
        rendered = ui._build_status().plain
        assert "Ctrl+C to abort" in rendered
        assert "/model to switch" in rendered
    finally:
        ui._state["loader_start"] = 0.0
        ui._state["activity"] = "idle"


def test_status_hides_abort_hint_once_chunks_arrive(monkeypatch):
    """Once any chunk has been received, the wait hint goes away — we're
    no longer 'stuck', the model is producing output."""
    monkeypatch.setattr(ui.time, "monotonic", lambda: 120.0)
    ui._state["loader_start"] = 100.0
    ui._state["activity"] = "receiving text stream"
    ui._state["stream_chars"] = 50
    try:
        rendered = ui._build_status().plain
        assert "Ctrl+C to abort" not in rendered
    finally:
        ui._state["loader_start"] = 0.0
        ui._state["activity"] = "idle"
        ui._state["stream_chars"] = 0


def test_artifact_hidden_thinking_surfaces_collapsed_liveness(monkeypatch, workspace):
    from core import loop, runtime
    from core.api import ThinkingDelta, TurnEnd

    class Provider:
        name = "fake"
        model = "fake-model"
        is_oauth = False

        def stream_turn(self, messages, tools, system):
            yield ThinkingDelta("private planning")
            yield TurnEnd(
                stop_reason="end_turn",
                message={"role": "assistant", "content": [{"type": "text", "text": "done"}]},
            )

    runtime.configure(Provider(), str(workspace), session=None)
    runtime.set_show_thinking(False)
    seen: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        loop.ui,
        "stream_delta",
        lambda kind, text, activity=None: seen.append((kind, text, activity or "")),
    )
    monkeypatch.setattr(loop.ui, "stream_clear", lambda: None)
    monkeypatch.setattr(loop.ui, "tool_progress_clear", lambda: None)

    loop._stream_one_turn(
        Provider(),
        messages=[{"role": "user", "content": "make an interactive html"}],
        tools=[],
        loader=loop._SilentLoader(),
        render=True,
    )

    assert seen == [("thinking", "private planning", "planning file tool call")]


def test_stream_delta_can_preserve_activity_label():
    ui._state["stream_kind"] = ""
    ui._state["stream_chars"] = 0
    ui._state["activity"] = "idle"
    try:
        ui.stream_delta("thinking", "abc", activity="planning file tool call")

        assert ui._state["stream_kind"] == "thinking"
        assert ui._state["stream_chars"] == 3
        assert ui._state["activity"] == "planning file tool call"
    finally:
        ui._state["stream_kind"] = ""
        ui._state["stream_chars"] = 0
        ui._state["activity"] = "idle"


def test_tool_progress_renders_detail():
    try:
        ui.tool_progress(
            "write_file",
            argument_chars=2048,
            call_id="toolu_123456789",
            detail="demo.html - 80 line(s), 4,200 chars",
            preview=["<canvas id=\"stage\">", "requestAnimationFrame(tick)"],
        )

        rendered = ui._build_tool_progress()

        assert rendered is not None
        plain = rendered.plain
        assert "assembling write_file" in plain
        assert "demo.html" in plain
        assert "80 line(s)" in plain
        assert "requestAnimationFrame" in plain
    finally:
        ui.tool_progress_clear()
