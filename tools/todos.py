from __future__ import annotations

from core import ui

from .types import Tool


PROMPT = """
Use this tool to create and manage a structured task list for the current
session. It helps the user see progress and helps you avoid dropping work.

## When to Use

Use proactively for:

1. Multi-step tasks with 3 or more distinct steps.
2. Non-trivial implementation work requiring several operations.
3. Explicit user requests for a todo list.
4. User requests that include multiple tasks.
5. New instructions that change scope or add requirements.
6. Starting real work: mark one item as doing before beginning.
7. Finishing work: mark items done immediately, not in a final batch.

## When NOT to Use

- A single, straightforward task.
- Trivial work where tracking adds no value.
- Pure conversation, Q&A, or explanation.
- Anything that is fewer than 3 simple steps.

## Rules

1. Pass the full list every call. The list is replaced, not merged.
2. Valid statuses: pending, doing, done.
3. Exactly one item should be doing while work is active.
4. Only mark done when the item is fully complete and verified when relevant.
5. If blocked, keep the item doing and add a pending item for the blocker.
6. Remove items that turn out to be irrelevant.

## Examples

GOOD:
User: "Add a dark mode toggle and run the tests"
todos: [
  {"text": "Add toggle component to Settings", "status": "doing"},
  {"text": "Wire theme state into the app context", "status": "pending"},
  {"text": "Add CSS variables for dark theme", "status": "pending"},
  {"text": "Run tests and fix failures", "status": "pending"}
]

BAD:
User: "Fix the typo in README"
One trivial step. Just edit the file.

BAD:
User: "What does git status do?"
Informational. No work to track.
""".strip()


_TODOS: list[dict] = []


def clear() -> None:
    _TODOS.clear()


def get_todos() -> list[dict]:
    return list(_TODOS)


def run(args: dict) -> str:
    items = args.get("todos") or []
    cleaned: list[dict] = []
    doing = 0

    for item in items:
        text = str(item.get("text", "")).strip()
        status = str(item.get("status", "pending")).strip().lower()
        if status not in ("pending", "doing", "done"):
            status = "pending"
        if status == "doing":
            doing += 1
        if text:
            cleaned.append({"text": text, "status": status})

    _TODOS.clear()
    _TODOS.extend(cleaned)
    ui.todos_panel(_TODOS)

    done = sum(1 for t in _TODOS if t["status"] == "done")
    note = "" if doing <= 1 else " warning: more than one item is doing"
    return f"todos: {done}/{len(_TODOS)} done{note}"


def summary(args: dict) -> str:
    items = args.get("todos") or []
    return f"{len(items)} item{'s' if len(items) != 1 else ''}"


TOOL = Tool(
    name="todos",
    description=(
        "Track multi-step work as a visible checklist. Pass the full list every "
        "call. Statuses: pending, doing, done. Keep exactly one item doing."
    ),
    schema={
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "doing", "done"],
                        },
                    },
                    "required": ["text", "status"],
                },
            },
        },
        "required": ["todos"],
    },
    permission="auto",
    run=run,
    prompt=PROMPT,
    priority=80,
    summary=summary,
    reset=clear,
)
