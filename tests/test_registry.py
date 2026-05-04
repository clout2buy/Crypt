"""Registry dispatch — permission integration, classify hooks, missing args."""
from __future__ import annotations

import pytest

from tools import registry
from tools.types import Tool


def _stub_tool(
    name: str = "stub",
    permission: str = "auto",
    classify=None,
    runner=lambda args: f"ran {args}",
    parallel_safe: bool = False,
) -> Tool:
    return Tool(
        name=name,
        description="stub",
        schema={"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        permission=permission,
        run=runner,
        classify=classify,
        parallel_safe=parallel_safe,
        summary=lambda args: str(args.get("x", "")),
    )


def test_unknown_tool_returns_error():
    ok, msg = registry.dispatch("nonexistent_tool", {}, render=False)
    assert ok is False
    assert "unknown" in msg


def test_missing_required_arg(monkeypatch):
    tool = _stub_tool(permission="auto")
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    ok, msg = registry.dispatch(tool.name, {}, render=False)
    assert ok is False
    assert "missing required input" in msg


def test_auto_tool_runs_without_prompt(monkeypatch):
    tool = _stub_tool(permission="auto")
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    ok, output = registry.dispatch(tool.name, {"x": "hi"}, render=False)
    assert ok is True
    assert "ran" in output


def test_ask_tool_blocked_in_subagent(monkeypatch):
    """render=False simulates the subagent path. Ask-tools must return a
    failure with a clear message instead of silently calling input()."""
    tool = _stub_tool(permission="ask")
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    ok, output = registry.dispatch(tool.name, {"x": "hi"}, render=False)
    assert ok is False
    assert "approval required" in output


def test_classify_safe_skips_prompt(monkeypatch):
    tool = _stub_tool(permission="ask", classify=lambda args: "safe")
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    ok, output = registry.dispatch(tool.name, {"x": "hi"}, render=False)
    assert ok is True


def test_classify_danger_blocks_subagent(monkeypatch):
    tool = _stub_tool(permission="auto", classify=lambda args: "danger")
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    ok, output = registry.dispatch(tool.name, {"x": "hi"}, render=False)
    assert ok is False
    assert "approval required" in output


def test_permissions_deny_short_circuits(monkeypatch, tmp_path):
    """A deny rule must reject the call even before classify runs."""
    from core import permissions
    pf = tmp_path / "permissions.json"
    pf.write_text('{"deny": ["stub:hi"]}')
    monkeypatch.setattr(permissions, "PERMISSIONS_PATH", pf)
    monkeypatch.setattr(permissions, "_CACHE", None)
    monkeypatch.setattr(permissions, "_CACHE_MTIME", None)

    tool = _stub_tool(permission="auto")  # would normally run
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    ok, msg = registry.dispatch(tool.name, {"x": "hi"}, render=False)
    assert ok is False
    assert "permissions rule" in msg


def test_run_exception_surfaces_as_failure(monkeypatch):
    def bad_run(args):
        raise RuntimeError("boom")
    tool = _stub_tool(permission="auto", runner=bad_run)
    monkeypatch.setitem(registry.REGISTRY._tools, tool.name, tool)
    ok, msg = registry.dispatch(tool.name, {"x": "hi"}, render=False)
    assert ok is False
    assert "RuntimeError" in msg
    assert "boom" in msg
