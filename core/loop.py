"""Agent loop: stream, dispatch tools, repeat."""
from __future__ import annotations

import time

from . import runtime, ui
from .api import Provider, TextDelta, ThinkingDelta, TurnEnd
from tools import REGISTRY, dispatch


_BASE_SYSTEM = """You are crypt, a coding assistant in a minimal terminal harness.

Work like a careful engineer:
- inspect before editing
- prefer edit_file for existing files; write_file is for new files
- use open_file to open generated local files
- keep changes small, idiomatic, and easy to review
- run focused tests or checks after edits when practical
- when using todos, keep narration minimal and let the live todo panel show progress
- execute large requests one verified phase at a time unless the user explicitly says to do all phases now
- keep execution narration short; avoid long audits unless asked
- never claim a phase is complete without naming the verification that passed
- if an edit tool fails in a risky state, stop and explain the recovery instead of improvising shell hacks
- if a tool is denied or fails with a permission error, adapt to the user's feedback instead of retrying stale calls
- open_file only opens a file for the user; it does not let you inspect image/video contents
- be concise and report blockers directly
""".strip()


def _build_system_prompt() -> str:
    guidance = REGISTRY.prompts()
    if not guidance:
        return _BASE_SYSTEM
    return f"{_BASE_SYSTEM}\n\n# Tool guidance\n\n{guidance}"


SYSTEM_PROMPT = _build_system_prompt()


def run(
    provider: Provider,
    show_thinking: bool = False,
    cwd: str = ".",
    model_switcher=None,
) -> str | None:
    """Run the TAOR loop. Returns 'login'/'logout' if the user asked, or None on quit."""
    messages: list[dict] = []
    current_tokens = 0
    session_tokens = 0
    runtime.configure(provider, cwd, lambda prompt: _run_subagent(provider, prompt))
    runtime.set_show_thinking(show_thinking)

    while True:
        REGISTRY.before_prompt()
        user_text = ui.user_prompt(
            yolo=runtime.yolo(),
            approval=runtime.approval_label(),
        )
        if not user_text:
            continue
        if user_text in ("/quit", "/exit", "exit", "quit"):
            ui.info("bye")
            return None
        if user_text == "/help":
            _show_help()
            continue
        if user_text in ("/login", "/logout", "/model"):
            if user_text == "/model" and model_switcher:
                new_provider = model_switcher(provider)
                if new_provider is not None:
                    provider = new_provider
                    runtime.configure(
                        provider,
                        runtime.cwd(),
                        lambda prompt: _run_subagent(provider, prompt),
                    )
                continue
            return user_text.lstrip("/")
        if user_text == "/clear":
            messages.clear()
            REGISTRY.reset_state()
            current_tokens = 0
            session_tokens = 0
            ui.info("context cleared")
            continue
        if user_text.startswith("/yolo") or user_text == "/safe":
            _handle_yolo(user_text)
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
        except KeyboardInterrupt:
            del messages[checkpoint:]
            _stub_dangling_tools(messages)
            ui.info("interrupted - turn cancelled")
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
    loader = ui.Loader(base_tokens=current_tokens)
    loader.start()
    try:
        budget = max_turns
        used = 0
        while used < budget:
            _ensure_tool_result_pairing(messages)
            end = _stream_with_retry(provider, messages, tools, loader)
            messages.append(end.message)
            used += 1

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
                loader.stop()
                if not is_subagent:
                    window = getattr(provider, "context_window", 200_000)
                    ctx_pct = min(99, int(current_tokens / window * 100))
                    ui.footer(
                        provider.model, ctx_pct,
                        current_tokens, session_tokens,
                        runtime.cwd(),
                        yolo=runtime.yolo(),
                        approval=runtime.approval_label(),
                    )
                return current_tokens, session_tokens

            _dispatch_tool_uses(end.message, messages)
            _ensure_tool_result_pairing(messages)

            # Budget check: instead of dying silently at max_turns, ask the
            # user whether to extend. Subagents just stop at their cap.
            if used >= budget and not is_subagent:
                loader.stop()
                if ui.ask(f"used {used} turns on this task - continue for {max_turns} more?"):
                    budget += max_turns
                    loader.start()
                else:
                    break

        _stub_dangling_tools(messages)
        _ensure_tool_result_pairing(messages)
        ui.error(f"stopped after {used} turns")
        return current_tokens, session_tokens
    finally:
        if loader.running:
            loader.stop()


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
            if isinstance(event, ThinkingDelta):
                if not show_thinking:
                    continue
                if not thinking_open:
                    ui.thinking_start()
                    thinking_open = True
                ui.thinking_chunk(event.text)
            elif isinstance(event, TextDelta):
                if not text_open and not event.text.strip():
                    continue
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
        if thinking_open:
            ui.thinking_end()
        if text_open:
            ui.assistant_end()
        raise


def _dispatch_tool_uses(assistant_msg: dict, messages: list[dict]) -> None:
    results: list[dict] = []
    tool_blocks = [
        block
        for block in assistant_msg.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    for idx, block in enumerate(tool_blocks):
        ok, output = dispatch(block["name"], block.get("input") or {})
        results.append({
            "type": "tool_result",
            "tool_use_id": block["id"],
            "content": output,
            "is_error": not ok,
        })
        if _should_abort_tool_batch(ok, output):
            skipped = tool_blocks[idx + 1:]
            for pending in skipped:
                results.append({
                    "type": "tool_result",
                    "tool_use_id": pending["id"],
                    "content": (
                        "skipped: previous tool failed or was denied. "
                        "Read the failure/feedback and choose the next step."
                    ),
                    "is_error": True,
                })
            if skipped:
                ui.info(f"skipped {len(skipped)} queued tool call(s) after failure")
            break
    if not results:
        raise RuntimeError("assistant stopped for tool_use without tool_use blocks")
    messages.append({"role": "user", "content": results})


def _should_abort_tool_batch(ok: bool, output: str) -> bool:
    if ok:
        return False
    text = str(output)
    return (
        text.startswith("denied by user")
        or text.startswith("PermissionError:")
    )


_TRANSIENT_NAMES = {
    "RateLimitError",
    "InternalServerError",
    "APIConnectionError",
    "APITimeoutError",
    "ServiceUnavailableError",
    "OverloadedError",
    "ResponseError",
}
_TRANSIENT_HINTS = (
    "overloaded", "rate limit", "rate_limit",
    "503", "502", "504",
    "timeout", "timed out", "connection", "temporarily",
)


def _is_transient(exc: Exception) -> bool:
    if type(exc).__name__ in _TRANSIENT_NAMES:
        return True
    msg = str(exc).lower()
    return any(h in msg for h in _TRANSIENT_HINTS)


def _is_rate_limit(exc: Exception) -> bool:
    return type(exc).__name__ == "RateLimitError" or "rate limit" in str(exc).lower()


def _stream_with_retry(
    provider: Provider,
    messages: list[dict],
    tools: list[dict],
    loader: ui.Loader,
    delays: tuple[int, ...] = (2, 5, 12),
) -> TurnEnd:
    """Run one turn; on transient failure, back off and retry. The live
    region stays running across retries — the wait just shows above it."""
    last_err: Exception | None = None
    for attempt in range(len(delays) + 1):
        try:
            return _stream_one_turn(provider, messages, tools, loader)
        except Exception as e:
            last_err = e
            if _is_rate_limit(e):
                raise
            if not _is_transient(e) or attempt == len(delays):
                raise
            wait = delays[attempt]
            ui.info(
                f"transient {type(e).__name__} - retry "
                f"{attempt + 1}/{len(delays)} in {wait}s"
            )
            time.sleep(wait)
    raise last_err  # pragma: no cover (unreachable)


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
    ui.info("/model             switch provider/model in this session")
    ui.info("/login   /logout   swap or sign out of Claude OAuth")
    ui.info("/clear             wipe context and todos")
    ui.info("/yolo              auto-approve file edits only")
    ui.info("/yolo all          auto-approve every tool (dangerous)")
    ui.info("/yolo off /safe    return to manual approvals")
    ui.info("/thinking          toggle thinking display")
    ui.info("/cwd [path]        show or move workspace")


def _handle_yolo(command: str) -> None:
    arg = command[5:].strip().lower() if command.startswith("/yolo") else "off"
    if command == "/safe" or arg in ("off", "false", "0", "manual"):
        mode = runtime.set_approval_mode(runtime.APPROVAL_NORMAL)
    elif arg in ("all", "full", "danger"):
        mode = runtime.set_approval_mode(runtime.APPROVAL_ALL)
    elif arg in ("", "edit", "edits", "trusted"):
        current = runtime.approval_mode()
        next_mode = (
            runtime.APPROVAL_NORMAL
            if current == runtime.APPROVAL_EDITS
            else runtime.APPROVAL_EDITS
        )
        mode = runtime.set_approval_mode(next_mode)
    else:
        ui.info("usage: /yolo, /yolo all, /yolo off")
        return

    label = runtime.approval_label()
    if mode == runtime.APPROVAL_EDITS:
        ui.info("approval: auto-edits (file edits skip prompts; shell still asks)")
    elif mode == runtime.APPROVAL_ALL:
        ui.info("approval: yolo-all (all tool prompts bypassed)")
    else:
        ui.info(f"approval: {label}")


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
        "approval": runtime.approval_label(),
        "thinking": "on" if runtime.show_thinking() else "off",
        "tools": f"{len(REGISTRY.schemas())} loaded",
        "todos": f"{done} done · {doing} doing · {pending} pending" if todos else "none",
        "context": f"{ctx_pct}% ({current_tokens:,} / {window:,})",
        "session": f"{session_tokens:,} tokens",
    })


def _format_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "RateLimitError":
        return (
            f"{name}: {exc}\n"
            "   Anthropic rate limits count the requested output cap before generation. "
            "Use /model to switch to Ollama Cloud, or restart with --max-tokens 2048."
        )
    if _is_transient(exc):
        return (
            f"{name}: {exc}\n"
            "   Provider is overloaded or unreachable. Tried with backoff. "
            "Use /model to switch provider/model, or wait a minute and retry."
        )
    return f"{name}: {exc}"
