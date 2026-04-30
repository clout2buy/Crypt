"""Agent loop: stream, dispatch tools, repeat."""
from __future__ import annotations

from . import runtime, ui
from .api import Provider, TextDelta, ThinkingDelta, TurnEnd
from tools import REGISTRY, dispatch
from tools.todos import get_todos


_BASE_SYSTEM = """You are crypt, a coding assistant in a minimal terminal harness.

Work like a careful engineer:
- inspect before editing
- prefer edit_file for existing files; write_file is for new files
- use open_file to open generated local files
- keep changes small, idiomatic, and easy to review
- run focused tests or checks after edits when practical
- be concise and report blockers directly
""".strip()


def _build_system_prompt() -> str:
    guidance = REGISTRY.prompts()
    if not guidance:
        return _BASE_SYSTEM
    return f"{_BASE_SYSTEM}\n\n# Tool guidance\n\n{guidance}"


SYSTEM_PROMPT = _build_system_prompt()


def run(provider: Provider, show_thinking: bool = True, cwd: str = ".") -> str | None:
    """Run the TAOR loop. Returns 'login'/'logout' if the user asked, or None on quit."""
    messages: list[dict] = []
    current_tokens = 0
    session_tokens = 0
    runtime.configure(provider, cwd, lambda prompt: _run_subagent(provider, prompt))
    runtime.set_show_thinking(show_thinking)

    while True:
        REGISTRY.before_prompt()

        status = ui.todos_status(get_todos())
        user_text = ui.user_prompt(status=status, yolo=runtime.yolo())
        if not user_text:
            continue
        if user_text in ("/quit", "/exit", "exit", "quit"):
            ui.info("bye")
            return None
        if user_text == "/help":
            _show_help()
            continue
        if user_text in ("/login", "/logout"):
            return user_text.lstrip("/")
        if user_text == "/clear":
            messages.clear()
            REGISTRY.reset_state()
            current_tokens = 0
            session_tokens = 0
            ui.info("context cleared")
            continue
        if user_text == "/yolo":
            on = runtime.set_yolo(not runtime.yolo())
            ui.info(f"yolo {'on - auto-approving all tools' if on else 'off'}")
            continue
        if user_text == "/thinking":
            on = runtime.set_show_thinking(not runtime.show_thinking())
            ui.info(f"thinking display: {'on' if on else 'off'}")
            continue
        if user_text.startswith("/cwd"):
            arg = user_text[4:].strip()
            if not arg:
                ui.info(f"cwd: {runtime.cwd()}")
                continue
            try:
                new_cwd = runtime.set_cwd(arg)
                ui.workspace_changed(str(new_cwd))
            except Exception as e:
                ui.error(f"{type(e).__name__}: {e}")
            continue
        if user_text == "/status":
            _show_status(provider, current_tokens, session_tokens)
            continue

        checkpoint = len(messages)
        messages.append({"role": "user", "content": user_text})

        try:
            current_tokens, session_tokens = _run_until_done(
                provider, messages, current_tokens, session_tokens,
            )
        except Exception as e:
            del messages[checkpoint:]
            _stub_dangling_tools(messages)
            ui.error(_format_error(e))


def _run_until_done(
    provider: Provider,
    messages: list[dict],
    current_tokens: int,
    session_tokens: int,
    max_turns: int = 50,
    is_subagent: bool = False,
) -> tuple[int, int]:
    tools = REGISTRY.schemas(for_subagent=is_subagent)

    for _ in range(max_turns):
        _ensure_tool_result_pairing(messages)
        loader = ui.Loader(base_tokens=current_tokens)
        loader.start()

        try:
            end = _stream_one_turn(provider, messages, tools, loader)
        except Exception:
            if loader.running:
                loader.stop()
            raise

        messages.append(end.message)

        if end.usage:
            turn = (
                end.usage.get("input_tokens", 0)
                + end.usage.get("output_tokens", 0)
            )
            current_tokens = turn
            session_tokens += turn

        if not _has_tool_use(end.message):
            if end.stop_reason == "max_tokens":
                ui.info("response truncated at max_tokens - bump --max-tokens or rephrase")
            if not is_subagent:
                window = getattr(provider, "context_window", 200_000)
                ctx_pct = min(99, int(current_tokens / window * 100))
                ui.footer(
                    provider.model, ctx_pct,
                    current_tokens, session_tokens,
                    runtime.cwd(),
                    yolo=runtime.yolo(),
                )
            return current_tokens, session_tokens

        _dispatch_tool_uses(end.message, messages)
        _ensure_tool_result_pairing(messages)

    _stub_dangling_tools(messages)
    _ensure_tool_result_pairing(messages)
    ui.error(f"max turns ({max_turns}) exceeded")
    return current_tokens, session_tokens


def _run_subagent(provider: Provider, prompt: str) -> str:
    messages: list[dict] = [{"role": "user", "content": prompt}]
    _run_until_done(provider, messages, 0, 0, is_subagent=True)
    return _extract_text(messages[-1] if messages else {})


def _stream_one_turn(
    provider: Provider,
    messages: list[dict],
    tools: list[dict],
    loader: ui.Loader,
) -> TurnEnd:
    show_thinking = runtime.show_thinking()
    thinking_open = False
    text_open = False

    try:
        for event in provider.stream_turn(messages, tools, SYSTEM_PROMPT):
            if loader.running:
                loader.stop()

            if isinstance(event, ThinkingDelta):
                if not show_thinking:
                    continue
                if not thinking_open:
                    ui.thinking_start()
                    thinking_open = True
                ui.thinking_chunk(event.text)
            elif isinstance(event, TextDelta):
                if thinking_open:
                    ui.thinking_end()
                    thinking_open = False
                if not text_open:
                    ui.assistant_start()
                    text_open = True
                ui.assistant_chunk(event.text)
            elif isinstance(event, TurnEnd):
                if thinking_open:
                    ui.thinking_end()
                if text_open:
                    ui.assistant_end()
                return event

        raise RuntimeError("provider stream ended without a TurnEnd event")

    except BaseException:
        if loader.running:
            loader.stop()
        if thinking_open:
            ui.thinking_end()
        if text_open:
            ui.assistant_end()
        raise


def _dispatch_tool_uses(assistant_msg: dict, messages: list[dict]) -> None:
    results: list[dict] = []
    for block in assistant_msg.get("content", []):
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        ok, output = dispatch(block["name"], block.get("input") or {})
        results.append({
            "type": "tool_result",
            "tool_use_id": block["id"],
            "content": output,
            "is_error": not ok,
        })
    if not results:
        raise RuntimeError("assistant stopped for tool_use without tool_use blocks")
    messages.append({"role": "user", "content": results})


def _has_tool_use(msg: dict) -> bool:
    return any(
        isinstance(b, dict) and b.get("type") == "tool_use"
        for b in msg.get("content", [])
    )


def _ensure_tool_result_pairing(messages: list[dict]) -> None:
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        expected = [
            b["id"] for b in msg.get("content", [])
            if isinstance(b, dict) and b.get("type") == "tool_use" and b.get("id")
        ]
        if not expected:
            continue
        if i + 1 >= len(messages):
            raise RuntimeError("assistant tool_use has no following tool_result message")
        nxt = messages[i + 1]
        if nxt.get("role") != "user" or not isinstance(nxt.get("content"), list):
            raise RuntimeError("assistant tool_use must be followed by user tool_result blocks")
        actual = [
            b.get("tool_use_id") for b in nxt["content"]
            if isinstance(b, dict) and b.get("type") == "tool_result"
        ]
        if sorted(expected) != sorted(actual):
            raise RuntimeError(
                "tool_result ids do not match tool_use ids: "
                f"expected {expected}, got {actual}"
            )


def _extract_text(message: dict) -> str:
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ).strip()


def _stub_dangling_tools(messages: list[dict]) -> None:
    if not messages or messages[-1].get("role") != "assistant":
        return
    tool_ids = [
        b["id"] for b in messages[-1].get("content", [])
        if b.get("type") == "tool_use"
    ]
    if tool_ids:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": "cancelled: max turns exceeded",
                    "is_error": True,
                }
                for tid in tool_ids
            ],
        })


def _show_help() -> None:
    ui.info("/quit              exit crypt")
    ui.info("/help              this message")
    ui.info("/status            show provider, auth, tools, todos")
    ui.info("/login   /logout   swap or sign out of Claude OAuth")
    ui.info("/clear             wipe context and todos")
    ui.info("/yolo              toggle auto-approve for all tools")
    ui.info("/thinking          toggle thinking display")
    ui.info("/cwd [path]        show or move workspace")


def _show_status(provider: Provider, current_tokens: int, session_tokens: int) -> None:
    from core import auth
    from tools.todos import get_todos

    cred = auth.resolve()
    todos = get_todos()
    done = sum(1 for t in todos if t.get("status") == "done")
    doing = sum(1 for t in todos if t.get("status") == "doing")
    pending = sum(1 for t in todos if t.get("status") == "pending")
    window = getattr(provider, "context_window", 200_000)
    ctx_pct = min(99, int(current_tokens / window * 100)) if current_tokens else 0

    auth_line = "none"
    if cred:
        if cred.kind == "oauth":
            auth_line = f"oauth · {cred.email or 'Claude OAuth'}"
            if cred.plan:
                auth_line += f" · {cred.plan}"
        else:
            auth_line = cred.kind

    ui.status_panel({
        "provider": provider.name,
        "model": provider.model,
        "auth": auth_line,
        "cwd": runtime.cwd(),
        "yolo": "on" if runtime.yolo() else "off",
        "thinking": "on" if runtime.show_thinking() else "off",
        "tools": f"{len(REGISTRY.schemas())} loaded",
        "todos": f"{done} done · {doing} doing · {pending} pending" if todos else "none",
        "context": f"{ctx_pct}% ({current_tokens:,} / {window:,})",
        "session": f"{session_tokens:,} tokens",
    })


def _format_error(exc: Exception) -> str:
    if type(exc).__name__ == "RateLimitError":
        return (
            f"{type(exc).__name__}: {exc}\n"
            "   Anthropic rate limits count the requested output cap before generation. "
            "Try --max-tokens 2048, --no-thinking, or wait for the retry window."
        )
    return f"{type(exc).__name__}: {exc}"

