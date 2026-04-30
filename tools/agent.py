from __future__ import annotations

from core import runtime, ui

from .types import Tool


PROMPT = """
Use this tool to run a focused subagent with a fresh, isolated context. The
subagent receives only your prompt and returns only final text. Its intermediate
file reads and tool calls do not enter your context.

## When to Use

1. Open-ended codebase research that would spam your context with file reads.
2. Large searches where you need a compact report.
3. Independent investigations that can run without this conversation history.
4. Summarizing many files into a short answer.
5. Getting a second read before a risky design decision.

## When NOT to Use

- Tasks you can finish in a few tool calls yourself.
- Trivial lookups where read_file or grep is faster.
- Work requiring this conversation's hidden context.
- Step-by-step supervised work.
- Delegating synthesis you should do yourself.

## Writing the Prompt

Brief the subagent like a smart colleague who just walked in:

- State the goal and why it matters.
- Include what you already know or ruled out.
- Include exact paths, symbols, commands, or constraints when you have them.
- Say how short the report should be when you need brevity.
- For investigations, ask the question. Do not prescribe brittle steps.

Never write "based on your findings, fix the bug". That delegates
understanding. Ask for the evidence, then you decide what to do.

## Examples

GOOD:
- description: "Find auth refresh sites"
  prompt: "Search this workspace for code that refreshes or stores auth tokens.
  We are checking for stale-token bugs. Report file:line and one sentence per
  match. Keep it under 300 words."

BAD:
- description: "Help me"
  prompt: "Look around and tell me what is wrong"
""".strip()


def run(args: dict) -> str:
    description = str(args.get("description", "subagent")).strip() or "subagent"
    prompt = str(args.get("prompt", "")).strip()
    if not prompt:
        return "spawn_agent: prompt is required"

    ui.subagent_start(description)
    try:
        output = runtime.run_subagent(prompt)
    except Exception as e:
        ui.subagent_end(False, description)
        return f"subagent error: {type(e).__name__}: {e}"

    ok = not output.startswith("spawn_agent unavailable")
    ui.subagent_end(ok, description)
    return output or "(no output)"


def summary(args: dict) -> str:
    return str(args.get("description", ""))


TOOL = Tool(
    name="spawn_agent",
    description=(
        "Run a focused subagent with a fresh context and return only its final "
        "text. Use for research or large explorations that should not pollute "
        "the main context."
    ),
    schema={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Short user-facing label under 60 characters.",
            },
            "prompt": {
                "type": "string",
                "description": "Self-contained briefing with scope and expected report.",
            },
        },
        "required": ["description", "prompt"],
    },
    permission="auto",
    run=run,
    prompt=PROMPT,
    priority=110,
    summary=summary,
    available_in_subagent=False,
)
