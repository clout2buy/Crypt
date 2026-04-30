from __future__ import annotations

import re

from .fs import files, int_arg, rel, resolve
from .types import Tool


def run(args: dict) -> str:
    pattern = re.compile(args["pattern"])
    limit = int_arg(args, "limit", 100, 500)
    matches: list[str] = []
    for path in files(resolve(args.get("path", "."))):
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                matches.append(f"{rel(path)}:{i}: {line.strip()}")
                if len(matches) >= limit:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches)"


def summary(args: dict) -> str:
    return f"{args.get('pattern', '')!r} in {args.get('path', '.')}"


TOOL = Tool(
    "grep",
    "Regex search inside workspace files.",
    {
        "type": "object",
        "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["pattern"],
    },
    "auto",
    run,
    priority=30,
    summary=summary,
)
