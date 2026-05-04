"""Runtime hooks shared between core and optional tools.

Holds session-scoped state that tools and the loop both need to read or write
without forming an import cycle. Configured once per session by core.loop.run().
"""
from __future__ import annotations

import os
import contextvars
from contextlib import contextmanager
from pathlib import Path
from typing import Callable


_provider = None
_cwd = "."
_subagent_runner: Callable[[str], str] | None = None
_session = None
_approval_mode = "normal"
_show_thinking = False
_render_tools = contextvars.ContextVar("crypt_render_tools", default=True)
_git_snapshot_cache: dict[str, str] = {}

APPROVAL_NORMAL = "normal"
APPROVAL_EDITS = "edits"
APPROVAL_ALL = "all"
APPROVAL_MODES = (APPROVAL_NORMAL, APPROVAL_EDITS, APPROVAL_ALL)
AUTO_EDIT_TOOLS = {
    "edit_file",
    "write_file",
    "open_file",
}


def configure(
    provider,
    cwd: str,
    subagent_runner: Callable[[str], str] | None = None,
    session=None,
) -> None:
    global _provider, _cwd, _subagent_runner, _session
    _provider = provider
    _cwd = cwd
    _subagent_runner = subagent_runner
    if session is not None:
        _session = session
    os.environ["CRYPT_ROOT"] = str(cwd)


def provider():
    return _provider


def session():
    return _session


def set_session(session_obj) -> None:
    global _session
    _session = session_obj


def session_id() -> str | None:
    return getattr(_session, "id", None)


def cwd() -> str:
    return _cwd


def set_cwd(path: str) -> Path:
    """Move the workspace mid-session. Updates CRYPT_ROOT so file tools and bash agree."""
    global _cwd
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"workspace does not exist: {p}")
    if not p.is_dir():
        raise NotADirectoryError(f"workspace is not a directory: {p}")
    _cwd = str(p)
    os.environ["CRYPT_ROOT"] = str(p)
    invalidate_git_snapshot()
    return p


def git_snapshot(cwd: str) -> str:
    """Cached startup git snapshot. Each cwd gets computed once per session
    so the system prompt stays cache-friendly and turns don't pay a 50-200ms
    git tax. Call invalidate_git_snapshot() if you need a refresh."""
    if cwd in _git_snapshot_cache:
        return _git_snapshot_cache[cwd]
    from . import prompt as _prompt  # avoid import cycle at module load

    snapshot = _prompt.compute_git_snapshot(cwd)
    _git_snapshot_cache[cwd] = snapshot
    return snapshot


def invalidate_git_snapshot() -> None:
    _git_snapshot_cache.clear()


def yolo() -> bool:
    return _approval_mode == APPROVAL_ALL


def set_yolo(on: bool) -> bool:
    set_approval_mode(APPROVAL_ALL if on else APPROVAL_NORMAL)
    return yolo()


def approval_mode() -> str:
    return _approval_mode


def approval_label() -> str:
    if _approval_mode == APPROVAL_EDITS:
        return "auto-edits"
    if _approval_mode == APPROVAL_ALL:
        return "yolo-all"
    return "manual"


def set_approval_mode(mode: str) -> str:
    global _approval_mode
    if mode not in APPROVAL_MODES:
        raise ValueError(f"unknown approval mode: {mode}")
    _approval_mode = mode
    return _approval_mode


def can_auto_approve(tool_name: str) -> bool:
    if _approval_mode == APPROVAL_ALL:
        return True
    if _approval_mode == APPROVAL_EDITS:
        return tool_name in AUTO_EDIT_TOOLS
    return False


def can_auto_approve_plan() -> bool:
    return _approval_mode == APPROVAL_ALL


def show_thinking() -> bool:
    return _show_thinking


def set_show_thinking(on: bool) -> bool:
    global _show_thinking
    _show_thinking = on
    return _show_thinking


def render_tools() -> bool:
    return bool(_render_tools.get())


@contextmanager
def tool_render(enabled: bool):
    token = _render_tools.set(bool(enabled))
    try:
        yield
    finally:
        _render_tools.reset(token)


def run_subagent(prompt: str, context: str | None = None) -> str:
    if _subagent_runner is None:
        return "spawn_agent unavailable: no active subagent runner"
    if context:
        return _subagent_runner(prompt, context)
    return _subagent_runner(prompt)


def background_job_summaries() -> list[str]:
    try:
        from . import background

        out: list[str] = []
        for job in background.list_jobs():
            out.append(f"{job.id}: {background.status(job)} - {job.command}")
        return out
    except Exception:
        return []
