from __future__ import annotations

from core import memory as mem

from .fs import clip
from .types import Tool


PROMPT = """
Use this tool only for durable facts that should affect future Crypt sessions:
user workflow preferences, project conventions, recurring commands, hard-won
debugging lessons, or explicit "remember this" requests.

Do not store secrets, credentials, private keys, transient task details, or
facts that are obvious from files in the repository.
""".strip()


def run(args: dict) -> str:
    action = str(args.get("action", "list")).strip().lower()
    if action == "add":
        return mem.add_memory(str(args.get("text", "")), str(args.get("scope", "Workflow")))
    if action == "list":
        return clip(mem.read_memory(), int(args.get("limit") or 12_000))
    if action == "search":
        needle = str(args.get("text", "")).strip().lower()
        if not needle:
            return "memory search requires text"
        lines = [
            line for line in mem.read_memory(80_000).splitlines()
            if needle in line.lower()
        ]
        return "\n".join(lines) or "(no matches)"
    raise ValueError("unknown memory action; use add, list, or search")


def classify(args: dict) -> str | None:
    action = str(args.get("action", "list")).lower()
    return "safe" if action in {"list", "search"} else None


def summary(args: dict) -> str:
    action = str(args.get("action", "list"))
    if action == "add":
        return "add"
    return action


TOOL = Tool(
    "memory",
    "Read, search, or add durable Crypt memory under ~/.crypt/memory/MEMORY.md.",
    {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "search"]},
            "text": {"type": "string"},
            "scope": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["action"],
    },
    "ask",
    run,
    prompt=PROMPT,
    priority=85,
    summary=summary,
    classify=classify,
)

