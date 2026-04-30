from __future__ import annotations

import importlib
import sys
from pathlib import Path

from core import ui

from .types import Tool


_SKIP_MODULES = {"__init__", "fs", "registry", "types"}


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self, *, for_subagent: bool = False) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.schema,
            }
            for t in self._ordered_tools()
            if not for_subagent or t.available_in_subagent
        ]

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
            print(
                f"  warning: failed to load tool {name}: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            continue
        tool = getattr(module, "TOOL", None)
        if isinstance(tool, Tool):
            REGISTRY.register(tool)


def dispatch(name: str, args: dict) -> tuple[bool, str]:
    from core import runtime

    tool = REGISTRY.get(name)
    if tool is None:
        return False, f"unknown tool: {name}"

    quiet = tool.quiet
    if not quiet:
        ui.tool_call(name, _summary(tool, args))

    missing = _missing_required(tool, args)
    if missing:
        msg = f"missing required input: {', '.join(missing)}"
        if quiet:
            ui.tool_call(name, _summary(tool, args))
        ui.tool_result(False, msg)
        return False, msg

    if tool.permission == "ask" and not runtime.yolo():
        if not ui.ask("run this?"):
            return False, "denied by user"

    try:
        out = tool.run(args)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if quiet:
            ui.tool_call(name, _summary(tool, args))
        ui.tool_result(False, msg)
        return False, msg

    if not quiet:
        ui.tool_result(True, out)
    return True, out


def _summary(tool: Tool, args: dict) -> str:
    if tool.summary:
        return tool.summary(args)
    return _one_line(str(args), 80)


def _one_line(text: str, limit: int = 80) -> str:
    text = str(text).replace("\n", " | ")
    return text if len(text) <= limit else text[: limit - 1] + "."


def _missing_required(tool: Tool, args: dict) -> list[str]:
    required = tool.schema.get("required") or []
    if not isinstance(required, list):
        return []
    return [
        name for name in required
        if name not in args or args.get(name) in (None, "")
    ]


_load_tools()
