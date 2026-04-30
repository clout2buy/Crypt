from __future__ import annotations

from .fs import clip, is_text_file, rel, resolve
from .types import Tool


def run(args: dict) -> str:
    path = resolve(args["path"])
    if not path.exists():
        raise FileNotFoundError(rel(path))
    if not path.is_file():
        raise IsADirectoryError(rel(path))
    if not is_text_file(path):
        raise ValueError(f"refusing to read binary file: {rel(path)}")

    text = path.read_text(encoding="utf-8", errors="replace")

    raw_offset = args.get("offset")
    raw_limit = args.get("limit")
    if raw_offset or raw_limit:
        all_lines = text.splitlines(keepends=True)
        start = max(1, int(raw_offset or 1)) - 1
        if raw_limit:
            end = start + max(1, int(raw_limit))
        else:
            end = len(all_lines)
        slice_ = all_lines[start:end]
        text = "".join(slice_)

    return clip(text, 50_000)


def summary(args: dict) -> str:
    path = str(args.get("path", ""))
    offset = args.get("offset")
    limit = args.get("limit")
    if offset or limit:
        end = (offset or 1) + (limit or 0) - 1 if limit else "end"
        return f"{path} L{offset or 1}-{end}"
    return path


TOOL = Tool(
    "read_file",
    (
        "Read a UTF-8 text file inside the workspace. "
        "Pass `offset` (1-indexed start line) and `limit` (line count) to read a slice "
        "instead of the whole file. Refuses to read binary files."
    ),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "1-indexed line to start from."},
            "limit": {"type": "integer", "description": "Number of lines to return."},
        },
        "required": ["path"],
    },
    "auto",
    run,
    priority=20,
    summary=summary,
)
