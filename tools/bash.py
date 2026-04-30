from __future__ import annotations

import subprocess

from .fs import clip, int_arg, root
from .types import Tool


def run(args: dict) -> str:
    r = subprocess.run(
        args["command"],
        shell=True,
        cwd=root(),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
        timeout=int_arg(args, "timeout", 30, 120),
    )

    parts: list[str] = []
    if r.stdout and r.stdout.strip():
        parts.append(r.stdout.rstrip())
    if r.stderr and r.stderr.strip():
        parts.append(f"[stderr]\n{r.stderr.rstrip()}")
    body = "\n".join(parts) if parts else "(no output)"

    if r.returncode != 0:
        # Errors: keep head + tail so the model sees what was attempted
        # AND the failure message at the end of stderr.
        raise RuntimeError(f"exit {r.returncode}\n{_head_tail(body, 20, 30)}")
    return clip(body)


def _head_tail(text: str, head: int, tail: int) -> str:
    lines = text.splitlines()
    if len(lines) <= head + tail + 2:
        return text
    omitted = len(lines) - head - tail
    return "\n".join(lines[:head] + [f"... [{omitted} lines omitted]"] + lines[-tail:])


def summary(args: dict) -> str:
    return str(args.get("command", ""))[:120]


TOOL = Tool(
    "bash",
    "Run a shell command from the workspace. On Windows this uses cmd.exe.",
    {"type": "object", "properties": {"command": {"type": "string"}, "timeout": {"type": "integer"}}, "required": ["command"]},
    "ask",
    run,
    priority=60,
    summary=summary,
)
