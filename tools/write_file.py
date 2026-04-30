from __future__ import annotations

from .fs import rel, resolve
from .types import Tool


def run(args: dict) -> str:
    path = resolve(args["path"])
    if path.exists():
        raise FileExistsError("file exists; use edit_file")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    return f"created {rel(path)}"


def summary(args: dict) -> str:
    return str(args.get("path", ""))


TOOL = Tool(
    "write_file",
    "Create a new UTF-8 text file inside the workspace. Refuses to overwrite.",
    {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    "ask",
    run,
    priority=50,
    summary=summary,
)
