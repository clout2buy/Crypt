from __future__ import annotations

import os
import subprocess
import sys

from .fs import rel, resolve
from .types import Tool


def run(args: dict) -> str:
    path = resolve(args["path"])
    if not path.exists():
        raise FileNotFoundError(rel(path))
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
    return f"opened {rel(path)}"


def summary(args: dict) -> str:
    return str(args.get("path", ""))


TOOL = Tool(
    "open_file",
    "Open a workspace file in the system default app or browser.",
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "ask",
    run,
    priority=70,
    summary=summary,
)
