from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .fs import rel
from .types import Tool


def run(args: dict) -> str:
    raw = str(args["path"]).strip()
    if not raw:
        raise ValueError("path is required")

    p = Path(raw).expanduser()
    if not p.is_absolute():
        # Relative paths resolve under the workspace root.
        from .fs import resolve
        p = resolve(raw)
    else:
        p = p.resolve()

    if not p.exists():
        raise FileNotFoundError(str(p))

    if os.name == "nt":
        os.startfile(p)  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])

    # Show workspace-relative when inside, absolute when outside.
    try:
        return f"opened {rel(p)}"
    except Exception:
        return f"opened {p}"


def summary(args: dict) -> str:
    return str(args.get("path", ""))


TOOL = Tool(
    "open_file",
    (
        "Open a file in the system default app (image viewer, browser, etc). "
        "Read-only — does not modify anything. Accepts absolute paths anywhere "
        "on disk; relative paths resolve under the workspace."
    ),
    {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "ask",
    run,
    priority=70,
    summary=summary,
)
