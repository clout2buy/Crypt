from __future__ import annotations

import difflib

from .fs import is_text_file, rel, resolve
from .types import Tool


def run(args: dict) -> str:
    path = resolve(args["path"])
    if not path.exists():
        raise FileNotFoundError(rel(path))
    if not is_text_file(path):
        raise ValueError(f"refusing to edit binary file: {rel(path)}")

    edits = _parse_edits(args)
    if not edits:
        raise ValueError("no edits provided (need 'old'+'new' or 'edits' list)")

    original = path.read_text(encoding="utf-8")

    # Validate every edit on the same starting text, applying as we go,
    # so multi-edit batches are atomic — either all apply or none do.
    cursor = original
    for i, (old, new) in enumerate(edits, 1):
        if not old:
            raise ValueError(f"edit {i}: 'old' cannot be empty")
        count = cursor.count(old)
        if count == 0:
            raise ValueError(
                f"edit {i}: no match for {_preview(old)}\n"
                f"   {_no_match_hint(cursor, old)}"
            )
        if count > 1:
            line_nums = _line_numbers_for(cursor, old, limit=5)
            raise ValueError(
                f"edit {i}: {count} matches for {_preview(old)}\n"
                f"   matched on line{'s' if len(line_nums) != 1 else ''} "
                f"{', '.join(str(n) for n in line_nums)}"
                f"{' (+ more)' if count > len(line_nums) else ''}\n"
                f"   add surrounding lines to make 'old' unique"
            )
        cursor = cursor.replace(old, new, 1)

    path.write_text(cursor, encoding="utf-8")

    diff = _short_diff(original, cursor, rel(path))
    suffix = f"\n{diff}" if diff else ""
    plural = "s" if len(edits) != 1 else ""
    return f"edited {rel(path)} ({len(edits)} edit{plural}){suffix}"


def summary(args: dict) -> str:
    path = str(args.get("path", ""))
    if "edits" in args and isinstance(args["edits"], list):
        n = len(args["edits"])
        return f"{path} ({n} edit{'s' if n != 1 else ''})"
    return path


def _parse_edits(args: dict) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if isinstance(args.get("edits"), list):
        for e in args["edits"]:
            if isinstance(e, dict):
                out.append((str(e.get("old", "")), str(e.get("new", ""))))
        return out
    if "old" in args:
        return [(str(args.get("old", "")), str(args.get("new", "")))]
    return out


def _preview(text: str, n: int = 60) -> str:
    s = text.replace("\n", "\\n").replace("\t", "\\t")
    return repr(s if len(s) <= n else s[:n] + "...")


def _line_numbers_for(text: str, needle: str, limit: int = 5) -> list[int]:
    first_line = needle.split("\n", 1)[0]
    out: list[int] = []
    for i, line in enumerate(text.splitlines(), 1):
        if first_line in line:
            out.append(i)
            if len(out) >= limit:
                break
    return out


def _no_match_hint(text: str, old: str) -> str:
    first = old.split("\n", 1)[0]
    if not first.strip():
        return "old text starts with whitespace; check leading indentation"
    if "\r" in old:
        return "old text contains \\r — file may use a different line ending"
    return f"first line of search text: {_preview(first, 80)}"


def _short_diff(old: str, new: str, label: str, max_chars: int = 1600) -> str:
    diff = list(difflib.unified_diff(
        old.splitlines(keepends=True),
        new.splitlines(keepends=True),
        fromfile=f"a/{label}",
        tofile=f"b/{label}",
        n=2,
        lineterm="",
    ))
    if not diff:
        return ""
    body = "".join(diff[2:])  # drop the file header lines
    body = body.rstrip()
    if len(body) > max_chars:
        body = body[:max_chars] + "\n... (diff truncated)"
    return body


TOOL = Tool(
    "edit_file",
    (
        "Edit one existing file by replacing exact substrings. Provide either "
        "`old`+`new` for a single edit or `edits: [{old, new}, ...]` for an atomic batch. "
        "All replacements must match exactly once across the file. "
        "Returns a unified diff snippet on success."
    ),
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string", "description": "Exact text to replace (single-edit mode)."},
            "new": {"type": "string", "description": "Replacement text (single-edit mode)."},
            "edits": {
                "type": "array",
                "description": "Batch of replacements; applied atomically.",
                "items": {
                    "type": "object",
                    "properties": {
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                    },
                    "required": ["old", "new"],
                },
            },
        },
        "required": ["path"],
    },
    "ask",
    run,
    priority=40,
    summary=summary,
)
