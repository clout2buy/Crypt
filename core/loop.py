"""Agent loop: stream, dispatch tools, repeat."""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import artifacts, background, compact, doctor, file_state, memory, prompt as prompt_builder, runtime
from . import session as sessions
from . import tracing, ui
from .api import Provider, TextDelta, ThinkingDelta, ToolUseProgress, ToolUseReady, TurnEnd
from tools import REGISTRY, dispatch


_BASE_SYSTEM = """You are Crypt, a local-first software engineering agent.

Work like a careful engineer:
- inspect before editing
- prefer edit_file for existing files; write_file is for new files
- use open_file to open generated local files
- keep changes small, idiomatic, and easy to review
- run focused tests or checks after edits when practical
- when using todos, keep narration minimal and let the live todo panel show progress
- execute large requests one verified phase at a time unless the user explicitly says to do all phases now
- when asked to create/build/write an artifact or file, use write_file/edit_file instead of pasting the full artifact in chat
- if a tool result says schema validation failed, fix the arguments once and then change approach if it fails again
- keep execution narration short; avoid long audits unless asked
- never claim a phase is complete without naming the verification that passed
- if an edit tool fails in a risky state, stop and explain the recovery instead of improvising shell hacks
- if a tool is denied or fails with a permission error, adapt to the user's feedback instead of retrying stale calls
- open_file only opens a file for the user; it does not let you inspect image/video contents
- be concise and report blockers directly
""".strip()


@dataclass
class RunResult:
    messages: list[dict]
    final_text: str
    current_tokens: int = 0
    session_tokens: int = 0


def _build_system_prompt() -> str:
    guidance = REGISTRY.prompts()
    if not guidance:
        return _BASE_SYSTEM
    return f"{_BASE_SYSTEM}\n\n# Tool guidance\n\n{guidance}"


def run_prompt(
    provider: Provider,
    user_text: str,
    *,
    cwd: str = ".",
    session_obj=None,
    max_turns: int = 50,
    approval_mode: str = runtime.APPROVAL_ALL,
    show_thinking: bool = False,
    render: bool = False,
) -> RunResult:
    """Run one prompt to completion without the interactive input loop.

    This is the production entry point for benchmarks, CI smoke checks, and
    scripted integrations. It uses the same provider/tool loop as the terminal
    UI, but approvals are governed by ``approval_mode`` instead of prompts.
    """
    previous_mode = runtime.approval_mode()
    previous_thinking = runtime.show_thinking()
    messages: list[dict] = session_obj.load_messages() if session_obj else []
    runtime.configure(provider, cwd, lambda prompt, context=None: _run_subagent(provider, prompt, context), session=session_obj)
    runtime.set_show_thinking(show_thinking)
    runtime.set_approval_mode(approval_mode)
    REGISTRY.before_prompt()
    user_msg = {"role": "user", "content": user_text}
    messages.append(user_msg)
    if runtime.session():
        runtime.session().record_message(user_msg)
    tracing.emit("run_start", mode="non_interactive", prompt=user_text, max_turns=max_turns)
    try:
        current_tokens, session_tokens = _run_until_done(
            provider,
            messages,
            0,
            compact.rough_tokens(messages),
            max_turns=max_turns,
            render=render,
        )
        final_text = _extract_text(messages[-1]) if messages else ""
        tracing.emit(
            "run_stop",
            mode="non_interactive",
            message_count=len(messages),
            current_tokens=current_tokens,
            session_tokens=session_tokens,
            final_text=final_text[:1000],
        )
        return RunResult(
            messages=messages,
            final_text=final_text,
            current_tokens=current_tokens,
            session_tokens=session_tokens,
        )
    except Exception as exc:
        tracing.emit("run_error", mode="non_interactive", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        runtime.set_approval_mode(previous_mode)
        runtime.set_show_thinking(previous_thinking)


def run(
    provider: Provider,
    show_thinking: bool = False,
    cwd: str = ".",
    model_switcher=None,
    session_obj=None,
) -> str | None:
    """Run the TAOR loop. Returns 'login'/'logout' if the user asked, or None on quit."""
    messages: list[dict] = session_obj.load_messages() if session_obj else []
    current_tokens = 0
    session_tokens = compact.rough_tokens(messages) if messages else 0
    runtime.configure(provider, cwd, lambda prompt, context=None: _run_subagent(provider, prompt, context), session=session_obj)
    runtime.set_show_thinking(show_thinking)
    tracing.emit("session_start", mode="interactive", resumed=bool(messages), message_count=len(messages))
    if session_obj and messages:
        _restore_tool_state(messages)
        ui.info(f"resumed session {session_obj.id} with {len(messages)} message(s)")

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
                        lambda prompt, context=None: _run_subagent(provider, prompt, context),
                        session=runtime.session(),
                    )
                continue
            return user_text.lstrip("/")
        if user_text == "/clear":
            messages.clear()
            REGISTRY.reset_state()
            file_state.clear()
            current_tokens = 0
            session_tokens = 0
            if runtime.session():
                runtime.session().append({"type": "event", "event": "clear"})
            ui.info("context cleared")
            continue
        if user_text.startswith("/sessions"):
            _show_sessions(runtime.cwd(), all_projects="--all" in user_text)
            continue
        if user_text.startswith("/resume"):
            query = user_text[len("/resume"):].strip() or None
            new_session = _resume_session(runtime.cwd(), provider, query)
            if new_session is not None:
                session_obj = new_session
                runtime.set_session(new_session)
                messages = new_session.load_messages()
                _restore_tool_state(messages)
                current_tokens = compact.rough_tokens(messages)
                session_tokens = current_tokens
                file_state.clear()
                ui.info(f"resumed {new_session.id} with {len(messages)} message(s)")
            continue
        if user_text == "/compact":
            if len(messages) < 4:
                ui.info("not enough context to compact")
                continue
            try:
                summary, compacted = compact.compact_messages(provider, messages)
                messages[:] = compacted
                current_tokens = compact.rough_tokens(messages)
                if runtime.session():
                    runtime.session().record_compaction(summary, len(messages), messages)
                ui.info(f"compacted context to {len(messages)} message(s)")
            except Exception as e:
                ui.error(f"compact failed: {type(e).__name__}: {e}")
            continue
        if user_text.startswith("/memory"):
            _handle_memory(user_text)
            continue
        if user_text.startswith("/background"):
            _show_background()
            continue
        if user_text == "/doctor":
            ui.info(doctor.run_doctor(runtime.cwd()))
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
        user_msg = {"role": "user", "content": user_text}
        messages.append(user_msg)
        if runtime.session():
            runtime.session().record_message(user_msg)

        try:
            if compact.should_compact(messages, getattr(provider, "context_window", 200_000)):
                ui.info("auto-compacting context before continuing")
                summary, compacted = compact.compact_messages(provider, messages)
                messages[:] = compacted
                current_tokens = compact.rough_tokens(messages)
                if runtime.session():
                    runtime.session().record_compaction(summary, len(messages), messages)
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
            if _is_rate_limit(e) and model_switcher:
                if ui.ask("switch model now?"):
                    new_provider = model_switcher(provider)
                    if new_provider is not None:
                        provider = new_provider
                        runtime.configure(
                            provider,
                            runtime.cwd(),
                            lambda prompt, context=None: _run_subagent(provider, prompt, context),
                            session=runtime.session(),
                        )


def _run_until_done(
    provider: Provider,
    messages: list[dict],
    current_tokens: int,
    session_tokens: int,
    max_turns: int = 50,
    is_subagent: bool = False,
    render: bool = True,
) -> tuple[int, int]:
    tools = REGISTRY.schemas(for_subagent=is_subagent)
    loader = ui.Loader(base_tokens=current_tokens) if render else _SilentLoader()
    if render:
        loader.start()
    try:
        budget = max_turns
        used = 0
        # One-shot per assistant turn: if the response truncates at the
        # default max_tokens cap, retry the SAME call once at the escalated
        # budget before giving up. Mirrors Claude Code's
        # max_output_tokens_escalate transition (query.ts:1199-1221).
        escalated_this_turn = False
        artifact_tool_retry_used = False
        while used < budget:
            _ensure_tool_result_pairing(messages)
            tracing.emit(
                "model_turn_start",
                turn=used + 1,
                message_count=len(messages),
                tool_count=len(tools),
                subagent=is_subagent,
            )
            end = _stream_with_retry(provider, messages, tools, loader, render=render)
            tracing.emit(
                "model_turn_end",
                turn=used + 1,
                stop_reason=end.stop_reason,
                tool_uses=_tool_use_names(end.message),
                usage=end.usage or {},
                subagent=is_subagent,
            )
            messages.append(end.message)
            if runtime.session() and not is_subagent:
                runtime.session().record_message(end.message)
            used += 1

            if end.usage:
                turn = (
                    end.usage.get("input_tokens", 0)
                    + end.usage.get("output_tokens", 0)
                )
                current_tokens = turn
                session_tokens += turn

            if not _has_tool_use(end.message):
                if (
                    not is_subagent
                    and not artifact_tool_retry_used
                    and artifacts.should_retry_text_only(messages, end.message)
                ):
                    artifact_tool_retry_used = True
                    correction = artifacts.tool_retry_message(messages)
                    messages.append(correction)
                    if runtime.session():
                        runtime.session().append({"type": "message", "message": correction})
                    if render:
                        ui.info("model pasted an artifact; retrying with file tools")
                    continue
                if (
                    end.stop_reason == "max_tokens"
                    and not escalated_this_turn
                    and hasattr(provider, "escalate_once")
                ):
                    # Pop the truncated message + restore the budget so this
                    # failed call doesn't count against max_turns. The session
                    # record gets the truncated message; that's intentional —
                    # leaves a forensic trail for /resume.
                    messages.pop()
                    used -= 1
                    new_cap = provider.escalate_once()
                    escalated_this_turn = True
                    if render:
                        ui.info(f"response truncated; retrying at max_tokens={new_cap:,}")
                    continue
                if end.stop_reason == "max_tokens":
                    if render:
                        ui.info("response truncated at escalated cap - rephrase or split work")
                if not is_subagent:
                    _maybe_complete_todos_after_final(messages)
                if render and getattr(end, "text_buffered", False):
                    _render_buffered_assistant_text(end.message)
                if render:
                    loader.stop()
                if render and not is_subagent:
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

            _dispatch_tool_uses(
                provider,
                end.message,
                messages,
                record=not is_subagent,
                render=render,
            )
            _ensure_tool_result_pairing(messages)

            # Micro-compaction: once context is half full, replace stale
            # bodies of re-runnable tool results with a marker. Cheap, no
            # LLM call, runs every turn so big-bash-output sessions don't
            # creep up to the 72% full-compaction threshold unnecessarily.
            window = getattr(provider, "context_window", 200_000)
            if current_tokens > window // 2:
                elided = compact.micro_compact(messages)
                if elided and render and not is_subagent:
                    ui.info(f"micro-compacted {elided} stale tool result(s)")

            # Budget check: instead of dying silently at max_turns, ask the
            # user whether to extend. Subagents just stop at their cap.
            if used >= budget and render and not is_subagent:
                loader.stop()
                if ui.ask(f"used {used} turns on this task - continue for {max_turns} more?"):
                    budget += max_turns
                    loader.start()
                else:
                    break

        _stub_dangling_tools(messages)
        _ensure_tool_result_pairing(messages)
        if render:
            ui.error(f"stopped after {used} turns")
        return current_tokens, session_tokens
    finally:
        if loader.running:
            loader.stop()


def _run_subagent(provider: Provider, prompt: str, context: str | None = None) -> str:
    state = file_state.snapshot()
    parts = [
        "You are a Crypt research subagent. Work silently and return only a "
        "concise final report. Use read-only tools only. Do not edit files, "
        "ask the user questions, update todos, or claim verification you did "
        "not run.",
    ]
    if context and context.strip():
        # Parent already digested these facts; passing them avoids the
        # subagent re-reading the same files. Treat as untrusted parent
        # output, not the user's voice.
        parts.append(
            "<parent_context>\n"
            + context.strip()
            + "\n</parent_context>"
        )
    parts.append("<task>\n" + prompt + "\n</task>")
    scoped_prompt = "\n\n".join(parts)
    messages: list[dict] = [{"role": "user", "content": scoped_prompt}]
    try:
        _run_until_done(provider, messages, 0, 0, is_subagent=True, render=False)
        return _extract_text(messages[-1] if messages else {})
    finally:
        file_state.restore(state)


def _stream_one_turn(
    provider: Provider,
    messages: list[dict],
    tools: list[dict],
    loader: ui.Loader,
    *,
    render: bool = True,
) -> TurnEnd:
    show_thinking = runtime.show_thinking() if render else False
    thinking_open = False
    text_open = False
    buffer_artifact_text = render and artifacts.creation_requested(messages)
    buffered_text: list[str] = []
    if render:
        ui.activity("waiting for provider response")

    try:
        system_prompt = prompt_builder.build_system_prompt(
            provider_name=getattr(provider, "name", "provider"),
            model=getattr(provider, "model", "model"),
            cwd=runtime.cwd(),
            tool_guidance=REGISTRY.prompts(),
        )
        for event in provider.stream_turn(messages, tools, system_prompt):
            if isinstance(event, ThinkingDelta):
                if render:
                    if show_thinking:
                        ui.activity("receiving reasoning stream")
                        ui.stream_delta("reasoning", event.text)
                    else:
                        ui.activity("model planning before next action")
                        ui.stream_delta("internal", event.text)
                if not show_thinking:
                    continue
                if not thinking_open:
                    ui.thinking_start()
                    thinking_open = True
                ui.thinking_chunk(event.text)
            elif isinstance(event, ToolUseProgress):
                if render:
                    ui.activity(f"receiving tool args: {event.name}")
                    ui.tool_progress(
                        event.name,
                        argument_chars=event.argument_chars,
                        call_id=event.call_id,
                        detail=_tool_progress_detail(event),
                    )
            elif isinstance(event, TextDelta):
                if not render:
                    continue
                ui.activity("receiving text stream")
                ui.stream_delta("text", event.text)
                if buffer_artifact_text:
                    if thinking_open:
                        ui.thinking_end()
                        thinking_open = False
                    buffered_text.append(event.text)
                    partial = "".join(buffered_text)
                    if artifacts.looks_like_artifact_start(partial):
                        return TurnEnd(
                            stop_reason="text_artifact",
                            message={
                                "role": "assistant",
                                "content": [{"type": "text", "text": partial}],
                            },
                            text_buffered=True,
                        )
                    continue
                if not text_open and not event.text.strip():
                    continue
                if thinking_open:
                    ui.thinking_end()
                    thinking_open = False
                if not text_open:
                    ui.assistant_start()
                    text_open = True
                ui.assistant_chunk(event.text)
            elif isinstance(event, ToolUseReady):
                if render:
                    ui.activity("tool call ready")
                    ui.stream_clear()
                    ui.tool_progress_clear()
                if thinking_open:
                    ui.thinking_end()
                    thinking_open = False
                if text_open:
                    ui.assistant_end()
                    text_open = False
                return TurnEnd(
                    stop_reason="tool_use",
                    message=event.message,
                    usage=event.usage,
                )
            elif isinstance(event, TurnEnd):
                if render:
                    ui.activity("response complete")
                    ui.stream_clear()
                    ui.tool_progress_clear()
                if thinking_open:
                    ui.thinking_end()
                if text_open:
                    ui.assistant_end()
                if buffered_text and _extract_text(event.message):
                    event.text_buffered = True
                return event

        raise RuntimeError("provider stream ended without a TurnEnd event")

    except BaseException:
        if render:
            ui.stream_clear()
            ui.tool_progress_clear()
        if thinking_open:
            ui.thinking_end()
        if text_open:
            ui.assistant_end()
        raise


class _SilentLoader:
    running = False

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def _dispatch_tool_uses(
    provider: Provider,
    assistant_msg: dict,
    messages: list[dict],
    record: bool = True,
    render: bool = True,
) -> None:
    results: list[dict] = []
    tool_blocks = [
        block
        for block in assistant_msg.get("content", [])
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    if _can_dispatch_parallel(tool_blocks):
        # OAuth limits concurrency strictly (often 1 or 2); sequential is safer
        # to avoid generic 429s.
        if provider.is_oauth:
            if render:
                ui.info("sequential execution: OAuth concurrency is restricted")
            for block in tool_blocks:
                ok, output = _dispatch_one(block, render=render)
                results.append(_tool_result_block(block, ok, output))
        else:
            results = _dispatch_parallel(tool_blocks, render=render)
    else:
        for idx, block in enumerate(tool_blocks):
            if render:
                ui.activity(f"executing tool: {block.get('name', 'tool')}")
            ok, output = _dispatch_one(block, render=render)
            results.append(_tool_result_block(block, ok, output))
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
                if skipped and render:
                    ui.info(f"skipped {len(skipped)} queued tool call(s) after failure")
                break
    if not results:
        raise RuntimeError("assistant stopped for tool_use without tool_use blocks")
    result_msg = {"role": "user", "content": results}
    messages.append(result_msg)
    if record and runtime.session():
        runtime.session().record_message(result_msg)


def _tool_progress_detail(event: ToolUseProgress) -> str:
    partial = event.partial_json or ""
    if not partial:
        return "tool call opened"
    parsed = _partial_tool_args(partial)
    if event.name in {"write_file", "edit_file"}:
        path = str(parsed.get("path") or "")
        content = str(parsed.get("content") or parsed.get("new") or "")
        if content:
            lines = content.count("\n") + 1
            label = f"{lines} line(s), {len(content):,} chars"
        else:
            label = f"{len(partial):,} arg chars"
        return f"{path} - {label}" if path else label
    if event.name == "multi_edit":
        path = str(parsed.get("path") or "")
        edits = parsed.get("edits")
        if isinstance(edits, list):
            label = f"{len(edits)} edit(s)"
        else:
            label = f"{len(partial):,} arg chars"
        return f"{path} - {label}" if path else label
    if event.name == "bash_start":
        cmd = str(parsed.get("command") or "")
        return cmd[:120] if cmd else f"{len(partial):,} arg chars"
    return f"{len(partial):,} arg chars"


def _partial_tool_args(partial: str) -> dict:
    try:
        value = json.loads(partial)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    out: dict[str, str] = {}
    for key in ("path", "command", "content", "new"):
        value = _partial_json_string_value(partial, key)
        if value:
            out[key] = value
    return out


def _partial_json_string_value(text: str, key: str) -> str:
    marker = f'"{key}"'
    idx = text.find(marker)
    if idx < 0:
        return ""
    colon = text.find(":", idx + len(marker))
    if colon < 0:
        return ""
    start = text.find('"', colon + 1)
    if start < 0:
        return ""
    chars: list[str] = []
    escaped = False
    for ch in text[start + 1:]:
        if escaped:
            chars.append(_unescape_json_char(ch))
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            break
        chars.append(ch)
    return "".join(chars)


def _unescape_json_char(ch: str) -> str:
    return {
        "n": "\n",
        "r": "\r",
        "t": "\t",
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
    }.get(ch, ch)


def _can_dispatch_parallel(tool_blocks: list[dict]) -> bool:
    if len(tool_blocks) < 2:
        return False
    for block in tool_blocks:
        tool = REGISTRY.get(block.get("name", ""))
        if tool is None or not tool.parallel_safe:
            return False
    return True


def _dispatch_parallel(tool_blocks: list[dict], *, render: bool) -> list[dict]:
    workers = min(8, len(tool_blocks))
    if render:
        names = ", ".join(str(block.get("name", "")) for block in tool_blocks)
        ui.activity(f"executing tools: {names}")
        ui.info(f"running {len(tool_blocks)} parallel-safe tool calls: {names}")

    def run_one(block: dict) -> dict:
        ok, output = _dispatch_one(block, render=False)
        return _tool_result_block(block, ok, output)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_one, tool_blocks))
    if render:
        failures = sum(1 for item in results if item.get("is_error"))
        suffix = f" ({failures} failed)" if failures else ""
        ui.info(f"parallel tool calls complete{suffix}")
    return results


def _dispatch_one(block: dict, *, render: bool) -> tuple[bool, str]:
    tool_name = str(block.get("name", ""))
    tool_id = str(block.get("id", ""))
    args = block.get("input") or {}
    tracing.emit("tool_start", tool_id=tool_id, tool=tool_name, args=args)
    started = time.perf_counter()
    ok, output = dispatch(
        tool_name,
        args,
        render=render,
        tool_use_id=tool_id if render else "",
    )
    tracing.emit(
        "tool_end",
        tool_id=tool_id,
        tool=tool_name,
        ok=ok,
        duration_ms=int((time.perf_counter() - started) * 1000),
        output=str(output)[:1000],
    )
    return ok, output


def _tool_result_block(block: dict, ok: bool, output) -> dict:
    return {
        "type": "tool_result",
        "tool_use_id": block["id"],
        "content": output,
        "is_error": not ok,
    }


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
    *,
    render: bool = True,
) -> TurnEnd:
    """Run one turn; on transient failure, back off and retry. The live
    region stays running across retries — the wait just shows above it."""
    last_err: Exception | None = None
    for attempt in range(len(delays) + 1):
        try:
            return _stream_one_turn(provider, messages, tools, loader, render=render)
        except Exception as e:
            last_err = e
            if _is_rate_limit(e):
                raise
            if not _is_transient(e) or attempt == len(delays):
                raise
            wait = delays[attempt]
            if render:
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


def _tool_use_names(msg: dict) -> list[str]:
    return [
        str(b.get("name", ""))
        for b in msg.get("content", [])
        if isinstance(b, dict) and b.get("type") == "tool_use"
    ]


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


def _maybe_complete_todos_after_final(messages: list[dict]) -> None:
    try:
        from tools import todos

        current = todos.get_todos()
        if not current or all(item.get("status") == "done" for item in current):
            return
        if not artifacts.creation_requested(messages):
            return
        successful = artifacts.successful_tool_names_since_last_request(messages)
        if successful.intersection({"write_file", "edit_file", "multi_edit", "open_file"}):
            todos.complete_all()
    except Exception:
        return

def _render_buffered_assistant_text(message: dict) -> None:
    text = _extract_text(message)
    if not text:
        return
    ui.assistant_start()
    ui.assistant_chunk(text)
    ui.assistant_end()


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
    ui.info("/sessions [--all]  list resumable sessions")
    ui.info("/resume [id|text]  resume latest or matching session")
    ui.info("/compact           summarize old context and keep working")
    ui.info("/memory            show durable Crypt memory")
    ui.info("/memory add <txt>  save durable memory")
    ui.info("/background        list background shell jobs")
    ui.info("/doctor            run local Crypt harness self-checks")
    ui.info("/model             switch provider/model in this session")
    ui.info("/login   /logout   swap or sign out of Anthropic OAuth")
    ui.info("/clear             wipe context and todos")
    ui.info("/yolo              auto-approve file edits only")
    ui.info("/yolo all          auto-approve every tool (dangerous)")
    ui.info("/yolo off /safe    return to manual approvals")
    ui.info("/thinking          toggle thinking display")
    ui.info("/cwd [path]        show or move workspace")


def _restore_tool_state(messages: list[dict]) -> None:
    try:
        from tools.todos import run as todos_run
    except Exception:
        return
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "tool_use"
                and block.get("name") == "todos"
                and isinstance(block.get("input"), dict)
            ):
                todos_run(block["input"])
                return


def _show_sessions(cwd: str, all_projects: bool = False) -> None:
    infos = sessions.list_sessions(cwd, all_projects=all_projects)
    if not infos:
        ui.info("no sessions found")
        return
    rows = {}
    for info in infos[:12]:
        age = time.strftime("%Y-%m-%d %H:%M", time.localtime(info.updated_at or info.created_at))
        label = info.session_id[:8]
        rows[label] = f"{age} · {info.title or '(untitled)'} · {info.message_count} messages"
    ui.status_panel(rows)


def _resume_session(cwd: str, provider: Provider, query: str | None):
    info = sessions.find_session(cwd, query)
    if info is None:
        ui.info("no matching session found")
        return None
    return sessions.load_session(
        info.cwd or cwd,
        info.session_id,
        provider=getattr(provider, "name", ""),
        model=getattr(provider, "model", ""),
    )


def _handle_memory(command: str) -> None:
    arg = command[len("/memory"):].strip()
    try:
        if not arg:
            ui.status_panel({"memory": memory.MEMORY_INDEX, "content": memory.read_memory(4000)})
            return
        if arg.startswith("add "):
            ui.info(memory.add_memory(arg[4:].strip()))
            return
        if arg.startswith("search "):
            needle = arg[len("search "):].strip().lower()
            lines = [line for line in memory.read_memory(80_000).splitlines() if needle in line.lower()]
            ui.info("\n".join(lines) if lines else "(no matches)")
            return
        ui.info("usage: /memory, /memory add <text>, /memory search <text>")
    except Exception as e:
        ui.error(f"memory failed: {type(e).__name__}: {e}")


def _show_background() -> None:
    jobs = background.list_jobs()
    if not jobs:
        ui.info("no background jobs")
        return
    rows = {
        job.id: f"{background.status(job)} · {job.command} · {job.output_path}"
        for job in jobs
    }
    ui.status_panel(rows)


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
            auth_line = f"oauth · {cred.email or 'Anthropic OAuth'}"
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
        "session_id": runtime.session_id() or "none",
        "background": f"{len(background.list_jobs())} job(s)",
        "context": f"{ctx_pct}% ({current_tokens:,} / {window:,})",
        "session": f"{session_tokens:,} tokens",
    })


def _format_error(exc: Exception) -> str:
    name = type(exc).__name__
    if name == "RateLimitError":
        return _rate_limit_help(exc)
    if _is_transient(exc):
        return (
            f"{name}: {exc}\n"
            "   Provider is overloaded or unreachable. Tried with backoff. "
            "Use /model to switch provider/model, or wait a minute and retry."
        )
    return f"{name}: {exc}"


# Header names from `services/rateLimitMocking.ts` and `services/claudeAiLimits.ts`
# in the Claude Code reference. Anthropic returns these on every response (not
# just 429s); the unified-* family is the Claude.ai 5h/7d window quota, which
# is what OAuth users almost always trip on.
_RATE_HEADERS = (
    "anthropic-ratelimit-unified-status",
    "anthropic-ratelimit-unified-5h-status",
    "anthropic-ratelimit-unified-5h-resets-at",
    "anthropic-ratelimit-unified-5h-utilization",
    "anthropic-ratelimit-unified-7d-status",
    "anthropic-ratelimit-unified-7d-resets-at",
    "anthropic-ratelimit-unified-7d-utilization",
    "retry-after",
)


def _rate_limit_help(exc: Exception) -> str:
    """Inspect Anthropic response headers to give a real diagnosis instead of
    blanket-blaming max_tokens. Handles two header families:
      - Claude.ai unified-* (OAuth, 5h/7d windows)
      - Standard API per-minute (requests / tokens / input-tokens / output-tokens)
    Falls back to a raw header dump if we see rate-limit shaped headers we
    don't recognize, so we can debug instead of guessing."""
    headers = _extract_headers(exc)
    name = type(exc).__name__
    parts: list[str] = [f"{name}: {exc}"]
    found_anything = False
    saw_exhausted_window = False

    # Family 1: Claude.ai unified windows (OAuth users). A status of "rejected"
    # or "exceeded" means that window's quota is gone — only the clock fixes it.
    unified_status = headers.get("anthropic-ratelimit-unified-status")
    if unified_status:
        found_anything = True
        parts.append(f"   unified status: {unified_status}")
        if unified_status.lower() in ("rejected", "exceeded"):
            saw_exhausted_window = True

    for tag, label in (("5h", "5-hour"), ("7d", "7-day")):
        status = (headers.get(f"anthropic-ratelimit-unified-{tag}-status") or "").lower()
        if not status:
            continue
        found_anything = True
        util = headers.get(f"anthropic-ratelimit-unified-{tag}-utilization")
        resets_at = headers.get(f"anthropic-ratelimit-unified-{tag}-resets-at")
        try:
            util_pct = f"{float(util) * 100:.0f}%" if util else "?"
        except ValueError:
            util_pct = "?"
        when = _format_reset(resets_at)
        parts.append(
            f"   {label} window: {status}, ~{util_pct} used"
            + (f", resets {when}" if when else "")
        )
        if status in ("rejected", "exceeded"):
            saw_exhausted_window = True

    # Family 2: standard API per-minute rate limits (API-key users, but also
    # surfaces on OAuth when the unified family isn't populated).
    for kind, label in (
        ("requests", "requests/min"),
        ("input-tokens", "input tok/min"),
        ("output-tokens", "output tok/min"),
        ("tokens", "tokens/min"),
    ):
        limit = headers.get(f"anthropic-ratelimit-{kind}-limit")
        remaining = headers.get(f"anthropic-ratelimit-{kind}-remaining")
        reset = headers.get(f"anthropic-ratelimit-{kind}-reset")
        if limit is None and remaining is None:
            continue
        found_anything = True
        when = _format_reset(reset) if reset else ""
        parts.append(
            f"   {label}: {remaining or '?'}/{limit or '?'} remaining"
            + (f", resets {when}" if when else "")
        )

    retry_after = headers.get("retry-after")
    if retry_after:
        found_anything = True
        try:
            secs = int(float(retry_after))
            parts.append(f"   server suggests retry in {secs}s")
        except ValueError:
            parts.append(f"   server suggests retry-after: {retry_after}")

    # If we recognized nothing but rate-limit-shaped headers exist, dump them
    # raw so we can see what we're actually dealing with instead of guessing.
    if not found_anything:
        rl_keys = sorted(
            k for k in headers
            if any(h in k for h in ("ratelimit", "retry-after", "quota", "limit"))
        )
        if rl_keys:
            parts.append("   unrecognized rate-limit headers (raw):")
            for k in rl_keys[:12]:
                parts.append(f"     {k}: {headers[k]}")
        elif headers:
            # If we still found nothing but headers exist, dump some basics.
            parts.append("   no rate-limit headers found. debug headers:")
            for k in sorted(headers.keys())[:8]:
                parts.append(f"     {k}: {headers[k]}")

    if saw_exhausted_window:
        parts.append(
            "   Claude.ai window is exhausted; only the clock fixes this. "
            "Reducing --max-tokens won't help."
        )
    parts.append("   Use /model to switch to Ollama Cloud, or wait it out.")
    return "\n".join(parts)


def _extract_headers(exc: Exception) -> dict[str, str]:
    """Pull headers off an Anthropic SDK error object, robustly. The SDK
    attaches the httpx Response under .response on APIStatusError subclasses;
    older versions used .body or .request_id. We fish for whatever is there."""
    response = getattr(exc, "response", None)
    raw = getattr(response, "headers", None) if response is not None else None
    if not raw:
        # Some versions or mocks put them on the exception directly.
        raw = getattr(exc, "headers", None)

    if not raw:
        return {}

    # httpx Headers is dict-like but case-sensitive on some platforms; normalize.
    # It might also be a list of tuples or a standard dict.
    try:
        items = raw.items() if hasattr(raw, "items") else raw
        return {str(k).lower(): str(v) for k, v in items}
    except Exception:
        return {}


def _format_reset(resets_at: str | None) -> str:
    """Render a unix-epoch (or ISO-8601) reset time as a relative string."""
    if not resets_at:
        return ""
    try:
        ts = float(resets_at)
    except ValueError:
        return resets_at  # ISO string — surface raw
    delta = max(0, int(ts - time.time()))
    if delta < 60:
        return f"in {delta}s"
    if delta < 3600:
        return f"in {delta // 60}m"
    return f"in {delta // 3600}h{(delta % 3600) // 60:02d}m"
