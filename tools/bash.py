from __future__ import annotations

import os
import re
import subprocess
import time
import uuid
from pathlib import Path

from .bash_safety import classify as _classify_command, is_destructive
from .fs import int_arg, root
from .types import Tool


# Search utilities exit with code 1 when they simply found nothing — that's
# not a failure, just a useful "no" answer. Treat exit 1 as success when the
# command's leading verb matches one of these.
_NO_MATCH_OK = re.compile(r"^\s*(grep|egrep|fgrep|rg|ack|find|locate)\b")

# Cap on bytes returned to the model per stream. Mirrors Claude Code's
# bash-output policy: enough to see the shape of a build but small enough
# that one big test run can't blow the context. Full output spills to disk.
_STREAM_CAP = 30_000
_SPILL_DIR = Path.home() / ".crypt" / "runs"


def run(args: dict) -> str:
    command = args["command"]
    r = subprocess.run(
        command,
        shell=True,
        cwd=root(),
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        text=True,
        # Default 30s, capped at 10 minutes so `pip install` and friends fit.
        timeout=int_arg(args, "timeout", 30, 600),
    )

    body, spill = _format_output(command, r.stdout or "", r.stderr or "")

    if r.returncode != 0:
        # grep/find returning 1 is "no match", not an error. Pass through cleanly
        # so the model sees the truth instead of an alarming exception.
        if r.returncode == 1 and _NO_MATCH_OK.match(command):
            return body if body != "(no output)" else "(no matches)"
        # Errors: full body (already capped + spilled) so the model sees what
        # was attempted AND the failure message.
        suffix = f"\n[full output: {spill}]" if spill else ""
        raise RuntimeError(f"exit {r.returncode}\n{body}{suffix}")
    return body


def _format_output(command: str, stdout: str, stderr: str) -> tuple[str, str]:
    """Cap stdout/stderr to ~30 KB each. If anything overflows, spill the
    full streams to ~/.crypt/runs/<id>.log and point the model at the path.

    Returns (body_for_model, spill_path_or_empty).
    """
    parts: list[str] = []
    overflow = False
    if stdout.strip():
        head, did_clip = _clip_stream(stdout)
        overflow = overflow or did_clip
        parts.append(head)
    if stderr.strip():
        head, did_clip = _clip_stream(stderr)
        overflow = overflow or did_clip
        parts.append(f"[stderr]\n{head}")
    body = "\n".join(parts) if parts else "(no output)"

    if not overflow:
        return body, ""

    spill = _write_spill(command, stdout, stderr)
    # Forward-slash the path even on Windows — cleaner in tool output and
    # readable by every shell we'd hand it to.
    pretty = spill.replace("\\", "/") if spill else ""
    return body + f"\n[output truncated; full log: {pretty}]", pretty


def _clip_stream(text: str, cap: int = _STREAM_CAP) -> tuple[str, bool]:
    """Keep head + tail of a stream so the model sees both the start and the
    failure message. Returns (clipped, was_truncated)."""
    text = text.rstrip()
    if len(text) <= cap:
        return text, False
    half = cap // 2
    head = text[:half].rstrip()
    tail = text[-half:].lstrip()
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [{omitted:,} chars truncated]\n{tail}", True


def _write_spill(command: str, stdout: str, stderr: str) -> str:
    try:
        _SPILL_DIR.mkdir(parents=True, exist_ok=True)
        name = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.log"
        path = _SPILL_DIR / name
        with path.open("w", encoding="utf-8", errors="replace") as f:
            f.write(f"$ {command}\n\n")
            if stdout:
                f.write(stdout)
                if not stdout.endswith("\n"):
                    f.write("\n")
            if stderr:
                f.write("\n[stderr]\n")
                f.write(stderr)
                if not stderr.endswith("\n"):
                    f.write("\n")
        if os.name != "nt":
            try:
                os.chmod(path, 0o600)
            except OSError:
                pass
        return str(path)
    except OSError:
        return ""


def classify(args: dict) -> str | None:
    cmd = str(args.get("command", ""))
    return _classify_command(cmd)


def summary(args: dict) -> str:
    cmd = str(args.get("command", ""))
    reason = is_destructive(cmd)
    label = cmd[:120]
    return f"!{label}  ({reason})" if reason else label


_PROMPT = """
For shell work. Crypt classifies commands at runtime:
- read-only commands (`ls`, `pwd`, `git status`, `cat`, `grep`, `rg`, `find`,
  `git diff/log/show`, `python -V`, etc.) auto-approve in every mode.
- destructive commands (`rm -rf`, `git reset --hard`, `git push --force`,
  `git commit --amend`, `git clean`, `dd`, etc.) prompt with a warning even
  in yolo. Don't try to outsmart this — use the dedicated tools when they
  exist (`edit_file` instead of `sed -i`, etc.).
- `grep`/`find` returning exit 1 means "no matches" — that's success.
""".strip()


TOOL = Tool(
    "bash",
    "Run a shell command from the workspace. On Windows this uses cmd.exe.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {
                "type": "integer",
                "description": "Seconds before the process is killed (default 30, max 600).",
            },
        },
        "required": ["command"],
    },
    "ask",
    run,
    prompt=_PROMPT,
    priority=60,
    summary=summary,
    classify=classify,
)
