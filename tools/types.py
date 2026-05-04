from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    permission: str
    run: Callable[[dict], Any]
    prompt: str = ""
    priority: int = 100
    summary: Callable[[dict], str] | None = None
    before_prompt: Callable[[], None] | None = None
    reset: Callable[[], None] | None = None
    available_in_subagent: bool = True
    quiet: bool = False
    # Optional per-call safety classifier. Returns one of:
    #   "safe"   - read-only / harmless; auto-approve in every mode
    #   "danger" - destructive; always confirm with a warning, even in yolo
    #   None or "ask" - default; use the tool's `permission` field
    # Letting tools self-classify keeps the registry generic.
    classify: Callable[[dict], str | None] | None = None
    # True only for deterministic read-only tools that can safely execute
    # beside each other when the model emits multiple tool calls in one turn.
    parallel_safe: bool = False
