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


def dispatch(
    name: str,
    args: dict,
    *,
    render: bool = True,
    tool_use_id: str = "",
) -> tuple[bool, str]:
    """Run a registered tool with permission/preview/render plumbing.

    When ``tool_use_id`` is given, the UI uses the new per-tool lifecycle
    (header on entry, animated row in the live region, ``└─ ✓/✗`` footer
    on exit). When it is empty, we fall back to the legacy one-shot
    ``tool_call`` + ``tool_result`` calls so callers that don't have an ID
    (CLI invocations, ad-hoc test paths) still render correctly.
    """
    from core import permissions, runtime

    tool = REGISTRY.get(name)
    if tool is None:
        return False, f"unknown tool: {name}"

    quiet = tool.quiet
    using_lifecycle = bool(render and tool_use_id)
    # `header_printed` tracks whether the visible header for this tool has
    # already landed in scrollback. Quiet tools start without a header so a
    # silent success leaves nothing behind; on failure we lazily print one.
    header_printed = False

    def _begin(summary: str) -> None:
        """Print the header (or register the lifecycle row) once."""
        nonlocal header_printed
        if header_printed or not render:
            return
        if using_lifecycle:
            ui.tool_begin(tool_use_id, name, summary)
        else:
            ui.tool_call(name, summary)
        header_printed = True

    def _emit_failure(msg: str) -> None:
        """Surface a failure cleanly regardless of which UI path we're on."""
        if not render:
            return
        if using_lifecycle:
            # Quiet tool that failed before printing a header: print one now
            # so the user sees what went wrong instead of a phantom error.
            if not header_printed:
                ui.tool_begin(tool_use_id, name, _summary_or_invalid(args, tool))
            ui.tool_end(tool_use_id, ok=False, output=msg)
        else:
            if not header_printed and quiet:
                ui.tool_call(name, _summary_or_invalid(args, tool))
            ui.tool_result(False, msg)

    if not isinstance(args, dict):
        msg = f"invalid tool input: expected JSON object, got {_json_type(args)}"
        if render:
            if using_lifecycle:
                ui.tool_begin(tool_use_id, name, "<invalid input>")
                ui.tool_end(tool_use_id, ok=False, output=msg)
            else:
                ui.tool_call(name, "<invalid input>")
                ui.tool_result(False, msg)
        return False, msg

    summary_text = _summary(tool, args)
    if render and not quiet:
        _begin(summary_text)

    validation_errors = _validate_args(tool, args)
    if validation_errors:
        shown = "; ".join(validation_errors[:4])
        if len(validation_errors) > 4:
            shown += f"; +{len(validation_errors) - 4} more"
        msg = f"schema validation failed: {shown}. Fix the arguments and retry once; do not repeat the same invalid call."
        _emit_failure(msg)
        return False, msg

    preflight_error = _preflight(tool, args)
    if preflight_error:
        _emit_failure(preflight_error)
        return False, preflight_error

    # User-defined rules first. Deny always wins; explicit allow trumps the
    # danger prompt because the user opted in by writing the rule.
    rule_decision, rule_pattern = permissions.check(name, summary_text)
    if rule_decision == "deny":
        msg = f"denied by permissions rule: {rule_pattern}"
        _emit_failure(msg)
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
            if using_lifecycle:
                ui.tool_set_state(tool_use_id, "approval", f"awaiting approval ({reason})")
            else:
                ui.activity(f"waiting for approval: {tool.name}")
            _render_preview(tool, args)
        approved, feedback = ui.confirm(f"DANGER ({reason}) - run anyway?")
        if not approved:
            msg = "denied by user"
            if feedback:
                msg += f": {feedback}"
            _emit_failure(msg)
            return False, msg
    elif classification == "safe":
        # Read-only / harmless. Skip the prompt regardless of mode.
        pass
    elif tool.permission == "ask" and not runtime.can_auto_approve(tool.name):
        if not render:
            return False, "approval required: interactive tool call unavailable in non-interactive subagent"
        if render:
            if using_lifecycle:
                ui.tool_set_state(tool_use_id, "approval")
            else:
                ui.activity(f"waiting for approval: {tool.name}")
            _render_preview(tool, args)
        approved, feedback = ui.confirm("run this?")
        if not approved:
            msg = "denied by user"
            if feedback:
                msg += f": {feedback}"
            _emit_failure(msg)
            return False, msg

    try:
        if render:
            if using_lifecycle:
                ui.tool_set_state(tool_use_id, "running")
            else:
                ui.activity(f"running tool: {tool.name}")
        with runtime.tool_render(render):
            out = tool.run(args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        _emit_failure(msg)
        return False, msg

    display = redact.text(_display_output(out))
    if render:
        if using_lifecycle and not quiet:
            ui.tool_end(tool_use_id, ok=True, output=display)
        elif using_lifecycle and quiet:
            # Quiet success: clear the lifecycle row without leaving any
            # transcript noise. Pass empty output so tool_end skips the body
            # but still drops the in-flight row.
            ui.tool_end(tool_use_id, ok=True, output="")
        elif not quiet:
            ui.tool_result(True, display)
    return True, redact.content(_model_output(out))


def _summary_or_invalid(args, tool: Tool) -> str:
    """Best-effort summary used when emitting a header for a failed tool."""
    if not isinstance(args, dict):
        return "<invalid input>"
    try:
        return _summary(tool, args)
    except Exception:
        return "<invalid input>"


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
        if name not in args
    ]


def _validate_args(tool: Tool, args: dict) -> list[str]:
    return _validate_object(tool.schema, args, path="")


def _validate_object(schema: dict, value, *, path: str) -> list[str]:
    if not isinstance(value, dict):
        label = path or "input"
        return [f"{label}: expected object, got {_json_type(value)}"]

    errors: list[str] = []
    required = schema.get("required") or []
    if isinstance(required, list):
        missing = [
            str(name)
            for name in required
            if name not in value
        ]
        if missing:
            label = f"{path}." if path else ""
            errors.append(f"missing required input: {', '.join(label + m for m in missing)}")

    properties = schema.get("properties") or {}
    if isinstance(properties, dict):
        for key, item in value.items():
            prop_schema = properties.get(key)
            if isinstance(prop_schema, dict):
                child_path = f"{path}.{key}" if path else str(key)
                errors.extend(_validate_value(prop_schema, item, child_path))
    return errors


def _validate_value(schema: dict, value, path: str) -> list[str]:
    expected = schema.get("type")
    if expected and not _matches_json_type(value, expected):
        return [f"{path}: expected {_type_label(expected)}, got {_json_type(value)}"]

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        allowed = ", ".join(repr(v) for v in enum[:6])
        if len(enum) > 6:
            allowed += ", ..."
        return [f"{path}: expected one of {allowed}, got {value!r}"]

    if isinstance(value, dict):
        return _validate_object(schema, value, path=path)

    if isinstance(value, list):
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            return []
        errors: list[str] = []
        for idx, item in enumerate(value):
            errors.extend(_validate_value(item_schema, item, f"{path}[{idx}]"))
            if len(errors) >= 8:
                break
        return errors

    return []


def _matches_json_type(value, expected) -> bool:
    if isinstance(expected, list):
        return any(_matches_json_type(value, item) for item in expected)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True


def _type_label(expected) -> str:
    if isinstance(expected, list):
        return " or ".join(str(item) for item in expected)
    return str(expected)


def _json_type(value) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


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
