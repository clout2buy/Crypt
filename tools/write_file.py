from __future__ import annotations

from core import file_state

from .fs import rel, resolve
from .types import Tool


def run(args: dict) -> str:
    path = resolve(args["path"])
    overwrite = bool(args.get("overwrite"))
    existed = path.exists()
    if existed and not overwrite:
        raise FileExistsError(
            "file exists; pass overwrite=true to replace it, or use edit_file"
        )
    if existed:
        file_state.assert_fresh_for_edit(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(args["content"], encoding="utf-8")
    file_state.record_write(path)
    return f"{'overwrote' if existed else 'created'} {rel(path)}"


def summary(args: dict) -> str:
    p = str(args.get("path", ""))
    if args.get("overwrite"):
        return f"{p} (overwrite)"
    return p


TOOL = Tool(
    "write_file",
    (
        "Create a new UTF-8 text file inside the workspace. By default refuses "
        "to overwrite — use edit_file for existing files. Pass overwrite=true "
        "ONLY when you really want to replace the entire file content (rare; "
        "edit_file is almost always the right tool for changes)."
    ),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "overwrite": {
                "type": "boolean",
                "description": "If true, replace existing file content instead of erroring.",
            },
        },
        "required": ["path", "content"],
    },
    "ask",
    run,
    priority=50,
    summary=summary,
)
