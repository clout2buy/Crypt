from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    permission: str
    run: Callable[[dict], str]
    prompt: str = ""
    priority: int = 100
    summary: Callable[[dict], str] | None = None
    before_prompt: Callable[[], None] | None = None
    reset: Callable[[], None] | None = None
    available_in_subagent: bool = True
    quiet: bool = False
