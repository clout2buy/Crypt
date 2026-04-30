"""Runtime hooks shared between core and optional tools.

Holds session-scoped state that tools and the loop both need to read or write
without forming an import cycle. Configured once per session by core.loop.run().
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable


_provider = None
_cwd = "."
_subagent_runner: Callable[[str], str] | None = None
_yolo = False
_show_thinking = False


def configure(
    provider,
    cwd: str,
    subagent_runner: Callable[[str], str] | None = None,
) -> None:
    global _provider, _cwd, _subagent_runner
    _provider = provider
    _cwd = cwd
    _subagent_runner = subagent_runner
    os.environ["CRYPT_ROOT"] = str(cwd)


def provider():
    return _provider


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
    return p


def yolo() -> bool:
    return _yolo


def set_yolo(on: bool) -> bool:
    global _yolo
    _yolo = on
    return _yolo


def show_thinking() -> bool:
    return _show_thinking


def set_show_thinking(on: bool) -> bool:
    global _show_thinking
    _show_thinking = on
    return _show_thinking


def run_subagent(prompt: str) -> str:
    if _subagent_runner is None:
        return "spawn_agent unavailable: no active subagent runner"
    return _subagent_runner(prompt)
