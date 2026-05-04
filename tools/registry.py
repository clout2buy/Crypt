from __future__ import annotations

import importlib
import sys
from pathlib import Path

from core import redact, ui

from .types import Tool


_SKIP_MODULES = {"__init__", "fs", "registry", "types", "bash_safety"}


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._load_failures: list[tuple[str, str]] = []

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def record_load_failure(self, module: str, error: str) -> None:
        self._load_failures.append((module, error))

    def load_failures(self) -> list[tuple[str, str]]:
        return list(self._load_failures)

    def schemas(self, *, for_subagent: bool = False) -> list[dict]:
        out: list[dict] = []
        for tool in self._ordered_tools():
            if for_subagent:
                if not tool.available_in_subagent:
                    continue
                # Subagents are intentionally non-interactive. Give them the
                # read-only/auto tool surface and keep approval-gated actions in
                # the main agent where the user can see and approve them.
                if tool.permission != "auto":
                    continue
            out.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.schema,
            })
        return out

    def prompts(self) -> str:
        chunks = []
        for tool in self._ordered_tools():
            if tool.prompt:
                chunks.append(f"## {tool.name}\n\n{tool.prompt.strip()}")
        return "\n\n".join(chunks)

    def before_prompt(self) -> None:
        for tool in self._ordered_tools():
            if tool.before_prompt:
                tool.before_prompt()

    def reset_state(self) -> None:
        for tool in self._ordered_tools():
            if tool.reset:
                tool.reset()

    def _ordered_tools(self) -> list[Tool]:
        return sorted(self._tools.values(), key=lambda t: (t.priority, t.name))


REGISTRY = Registry()


def _load_tools() -> None:
    package = __package__
    for path in sorted(Path(__file__).parent.glob("*.py")):
        name = path.stem
        if name in _SKIP_MODULES or name.startswith("_"):
            continue
        try:
            module = importlib.import_module(f"{package}.{name}")
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            REGISTRY.record_load_failure(name, detail)
            print(
                f"  warning: failed to load tool {name}: {detail}",
                file=sys.stderr,
            )
            continue
        tool = getattr(module, "TOOL", None)
        if isinstance(tool, Tool):
            REGISTRY.register(tool)


def dispatch(name: str, args: dict, *, render: bool = True) -> tuple[bool, str]:
    from core import permissions, runtime

    tool = REGISTRY.get(name)
    if tool is None:
        return False, f"unknown tool: {name}"

    quiet = tool.quiet
    summary_text = _summary(tool, args)
    if render and not quiet:
        ui.tool_call(name, summary_text)

    missing = _missing_required(tool, args)
    if missing:
        msg = f"missing required input: {', '.join(missing)}"
        if render and quiet:
            ui.tool_call(name, summary_text)
        if render:
            ui.tool_result(False, msg)
        return False, msg

    preflight_error = _preflight(tool, args)
    if preflight_error:
        if render and quiet:
            ui.tool_call(name, summary_text)
        if render:
            ui.tool_result(False, preflight_error)
        return False, preflight_error

    # User-defined rules first. Deny always wins; explicit allow trumps the
    # danger prompt because the user opted in by writing the rule.
    rule_decision, rule_pattern = permissions.check(name, summary_text)
    if rule_decision == "deny":
        msg = f"denied by permissions rule: {rule_pattern}"
        if render and quiet:
            ui.tool_call(name, summary_text)
        if render:
            ui.tool_result(False, msg)
        return False, msg

    classification = tool.classify(args) if tool.classify else None
    if rule_decision == "allow":
        # Explicit user pre-approval — skip every prompt below.
        if render:
            ui.info(f"auto-approved by rule: {rule_pattern}")
    elif classification == "danger":
        # Danger always confirms unless explicitly allow-listed above.
        if not render:
            return False, "approval required: destructive tool call unavailable in non-interactive subagent"
        reason = _danger_reason(tool, args) or "destructive operation"
        if render:
            _render_preview(tool, args)
        approved, feedback = ui.confirm(f"DANGER ({reason}) - run anyway?")
        if not approved:
            msg = "denied by user"
            if feedback:
                msg += f": {feedback}"
            ui.tool_result(False, msg)
            return False, msg
    elif classification == "safe":
        # Read-only / harmless. Skip the prompt regardless of mode.
        pass
    elif tool.permission == "ask" and not runtime.can_auto_approve(tool.name):
        if not render:
            return False, "approval required: interactive tool call unavailable in non-interactive subagent"
        if render:
            _render_preview(tool, args)
        approved, feedback = ui.confirm("run this?")
        if not approved:
            msg = "denied by user"
            if feedback:
                msg += f": {feedback}"
            ui.tool_result(False, msg)
            return False, msg

    try:
        with runtime.tool_render(render):
            out = tool.run(args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if render and quiet:
            ui.tool_call(name, _summary(tool, args))
        if render:
            ui.tool_result(False, msg)
        return False, msg

    display = redact.text(_display_output(out))
    if render and not quiet:
        ui.tool_result(True, display)
    return True, redact.content(_model_output(out))


def _summary(tool: Tool, args: dict) -> str:
    if tool.summary:
        return tool.summary(args)
    return _one_line(str(args), 80)


def _one_line(text: str, limit: int = 80) -> str:
    text = str(text).replace("\n", " | ")
    return text if len(text) <= limit else text[: limit - 1] + "."


def _display_output(out) -> str:
    if isinstance(out, dict) and out.get("__crypt_tool_result__"):
        return str(out.get("display", "(structured result)"))
    return str(out)


def _model_output(out):
    if isinstance(out, dict) and out.get("__crypt_tool_result__"):
        return out.get("content", "")
    return out


def _render_preview(tool: Tool, args: dict) -> None:
    """Show a tool-specific preview before the approval prompt. Tool's
    preview() can raise — we never let preview problems block the prompt."""
    if not tool.preview:
        return
    try:
        text = tool.preview(args)
    except Exception:
        return
    if text:
        ui.diff_preview(text)


def _missing_required(tool: Tool, args: dict) -> list[str]:
    required = tool.schema.get("required") or []
    if not isinstance(required, list):
        return []
    return [
        name for name in required
        if name not in args or args.get(name) in (None, "")
    ]


def _danger_reason(tool: Tool, args: dict) -> str | None:
    """Best-effort human-readable reason. Bash-aware today; harmless for others."""
    if tool.name in {"bash", "bash_start"}:
        from . import bash_safety
        return bash_safety.is_destructive(str(args.get("command", "")))
    return None


def _preflight(tool: Tool, args: dict) -> str | None:
    if tool.permission != "ask" or "path" not in args:
        return None
    try:
        from .fs import resolve

        resolve(str(args["path"]))
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


_load_tools()
