"""Terminal UI for crypt — TRANSMISSION style, Rich-powered.

Architecture:
- One Rich `Console` shared across the module.
- A live region (Rich `Live`) pinned at the bottom while the model works.
  It renders the todos panel + animated status line.
- All other output (tool calls, assistant text, etc.) goes through
  `console.print`, which scrolls above the live region naturally.
- A small typewriter pipe smooths chunked streaming into a steady
  character flow so the model's text doesn't appear in jarring bursts.

Public API stays stable: welcome, footer, user_prompt, info, error, ask,
ask_choice, splash_choice, todos_panel, todos_status, plan_panel,
status_panel, workspace_changed, subagent_start, subagent_end,
assistant_*, thinking_*, tool_call, tool_result, clear_screen,
read_multiline, feedback_prompt, Loader.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from datetime import datetime
from typing import Iterable

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.text import Text


if os.name == "nt":
    os.system("")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


# ─── palette ─────────────────────────────────────────────────────────────
INK = "rgb(200,230,212)"
MUTED = "rgb(106,138,120)"
FAINT = "rgb(58,84,72)"
EDGE = "rgb(26,40,32)"
ACCENT = "rgb(38,255,156)"
ALT = "rgb(84,224,255)"
WARN = "rgb(255,184,64)"
ERR = "rgb(255,64,96)"
HOT = "rgb(255,126,192)"

ACCENT_BG = "on rgb(16,74,48)"
WARN_BG = "on rgb(74,54,16)"
ERR_BG = "on rgb(74,20,32)"
INPUT_BG = "on rgb(28,40,32)"

# Backward-compat constants used by tools/plan.py and similar.
RST = ""  # legacy raw-ANSI marker; safe no-op now
GOLD = WARN


# ─── console singleton ──────────────────────────────────────────────────
console = Console(
    highlight=False,
    soft_wrap=False,
    force_terminal=True,
    legacy_windows=False,
)


# ─── module state ───────────────────────────────────────────────────────
_state = {
    "live": None,           # active Live instance (or None)
    "loader_start": 0.0,    # monotonic clock when current turn started
    "loader_tokens": 0,     # base token count for the running loader
    "todos": [],            # current todo list (mirrored from todos tool)
    "pipe": None,           # typewriter pipe (when streaming text/thinking)
}

_WAVE = "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
_VERBS = ("thinking", "reading", "planning", "writing", "checking", "working")


# ─── tiny formatters ─────────────────────────────────────────────────────
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _short_path(path: str, limit: int = 44) -> str:
    s = str(path)
    if len(s) <= limit:
        return s
    keep = max(8, limit - 3)
    return "..." + s[-keep:]


def _short_model(model: str) -> str:
    m = model.removeprefix("claude-")
    parts = m.split("-")
    if parts and len(parts[-1]) >= 8 and parts[-1].isdigit():
        parts.pop()
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        parts[-2:] = [f"{parts[-2]}.{parts[-1]}"]
    return " ".join(parts)


def _trim(text: str, limit: int) -> str:
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: max(1, limit - 1)] + "."


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.0f}k"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n}"


def _ctx_color(pct: int) -> str:
    if pct < 50:
        return ACCENT
    if pct < 80:
        return WARN
    return ERR


# ─── packet header (─[ XX · time ]──────────) ───────────────────────────
def _packet_text(code: str, color: str) -> Text:
    t = Text("  ─[ ", style=FAINT)
    t.append(code, style=f"bold {color}")
    t.append(" · ", style=FAINT)
    t.append(_now(), style=MUTED)
    t.append(" ]", style=FAINT)
    rule_w = max(2, console.size.width - len(t.plain) - 2)
    t.append("─" * rule_w, style=FAINT)
    return t


def _packet(code: str, color: str) -> None:
    console.print(_packet_text(code, color))


# ─── live region ────────────────────────────────────────────────────────
def _build_status() -> Text:
    elapsed = max(1, int(time.monotonic() - _state["loader_start"]))
    offset = (elapsed * 4) % len(_WAVE)
    wave = (_WAVE + _WAVE)[offset : offset + 14]
    verb = _VERBS[(elapsed // 2) % len(_VERBS)]

    t = Text("  ")
    t.append(wave, style=ACCENT)
    t.append("  ")
    t.append(verb, style=MUTED)
    t.append(f"  ({elapsed}s", style=FAINT)
    if _state["loader_tokens"]:
        t.append(f" · {_state['loader_tokens']:,} tok", style=FAINT)
    t.append(")", style=FAINT)
    return t


def _build_todos_block() -> RenderableType | None:
    items = _state["todos"]
    if not items:
        return None
    rows: list[Text] = [_packet_text("JB", WARN)]
    for it in items:
        status = it.get("status", "pending")
        text = it.get("text", "")
        t = Text("  ")
        if status == "done":
            t.append("✓ ", style=ACCENT)
            t.append(text, style=f"strike {FAINT}")
        elif status == "doing":
            t.append("■ ", style=WARN)
            t.append(text, style=f"bold {INK}")
        else:
            t.append("□ ", style=FAINT)
            t.append(text, style=MUTED)
        rows.append(t)
    return Group(*rows)


class _LiveRenderable:
    """Rebuilt by Rich on each refresh — pulls fresh state."""
    def __rich__(self) -> RenderableType:
        parts: list[RenderableType] = []
        todos = _build_todos_block()
        if todos is not None:
            parts.append(todos)
            parts.append(Text(""))
        parts.append(_build_status())
        return Group(*parts)


def _live_start() -> None:
    if _state["live"] is not None:
        return
    live = Live(
        _LiveRenderable(),
        console=console,
        refresh_per_second=12,
        transient=True,
    )
    live.start(refresh=True)
    _state["live"] = live


def _live_stop() -> None:
    live = _state["live"]
    if live is None:
        return
    live.stop()
    _state["live"] = None


def _suspend_live():
    """Context manager: pause the live region so input() works cleanly."""
    class _Ctx:
        def __enter__(self_):
            self_.was = _state["live"] is not None
            if self_.was:
                _live_stop()
            return self_
        def __exit__(self_, *exc):
            if self_.was:
                _live_start()
    return _Ctx()


# ─── typewriter pipe ────────────────────────────────────────────────────
class _Typewriter:
    """Drains chunks into a per-character stream at adaptive cadence.

    Chunks arrive in bursts; the worker thread emits one char at a time
    so output feels typed rather than dumped. Cadence speeds up when the
    backlog grows so long responses finish promptly.
    """
    def __init__(self, on_char) -> None:
        self._on_char = on_char
        self._q: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2)
        # Flush anything left.
        while self._q:
            self._on_char(self._q.popleft())

    def write(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._q.extend(text)
        self._wake.set()

    def _loop(self) -> None:
        while self._running:
            self._wake.wait(timeout=0.05)
            self._wake.clear()
            while self._q and self._running:
                with self._lock:
                    backlog = len(self._q)
                    ch = self._q.popleft() if self._q else None
                if ch is None:
                    break
                self._on_char(ch)
                # Adaptive: faster when backlog grows.
                if backlog > 400:
                    delay = 0.0008
                elif backlog > 120:
                    delay = 0.003
                elif backlog > 30:
                    delay = 0.006
                else:
                    delay = 0.012
                time.sleep(delay)


def _emit_with_rail(rail: Text, ch: str, ink_style: str) -> None:
    """Write one char honouring the rail prefix on newlines."""
    if ch == "\n":
        console.print()
        console.print(rail, end="")
    else:
        console.print(ch, style=ink_style, end="", markup=False, highlight=False)


# ─── welcome / chrome ───────────────────────────────────────────────────
def welcome(
    provider: str,
    model: str,
    auth_kind: str | None = None,
    auth_email: str | None = None,
    auth_plan: str | None = None,
    cwd: str = ".",
) -> None:
    friendly = _short_model(model)

    auth_line = "no auth"
    if auth_kind == "oauth":
        auth_line = auth_email or "Claude OAuth"
        if auth_plan:
            auth_line += f"  ·  {auth_plan}"
    elif auth_kind:
        auth_line = auth_kind

    width = console.size.width

    # top edge with logo
    top = Text("  ╭─", style=FAINT)
    top.append(" CRYPT ", style=f"bold {ACCENT}")
    top.append("── ", style=FAINT)
    top.append("transmission console ", style=MUTED)
    top.append("── ", style=FAINT)
    rule_w = max(2, width - len(top.plain) - 2)
    top.append("─" * rule_w, style=FAINT)

    # info rows
    link = Text("  │  ", style=FAINT)
    link.append("●", style=ACCENT)
    link.append("  ")
    link.append("LINK   ", style=MUTED)
    link.append(friendly, style=f"bold {INK}")
    link.append(" via ", style=FAINT)
    link.append(provider, style=MUTED)

    auth = Text("  │       ", style=FAINT)
    auth.append(auth_line, style=MUTED)

    cwd_row = Text("  │  ", style=FAINT)
    cwd_row.append("◌", style=MUTED)
    cwd_row.append("  ")
    cwd_row.append("CWD    ", style=MUTED)
    cwd_row.append(_short_path(cwd, 60), style=INK)

    tools_row = Text("  │  ", style=FAINT)
    tools_row.append("▰", style=ACCENT)
    tools_row.append("  ")
    tools_row.append("TOOLS  ", style=MUTED)
    tools_row.append(f"{_tool_count()} loaded", style=INK)
    tools_row.append("  ·  /status for details", style=FAINT)

    bottom = Text("  ╰─", style=FAINT)
    bottom.append("─" * max(2, width - 2), style=FAINT)

    cmds = Text("  ▸  ", style=ACCENT)
    sep = Text("  ·  ", style=FAINT)
    for i, cmd in enumerate(("/help", "/status", "/login", "/logout", "/quit")):
        if i:
            cmds.append_text(sep)
        cmds.append(cmd, style=MUTED)

    console.print()
    console.print(top)
    console.print(link)
    console.print(auth)
    console.print(cwd_row)
    console.print(tools_row)
    console.print(bottom)
    console.print()
    console.print(cmds)
    console.print()


def _tool_count() -> int:
    try:
        from tools import REGISTRY
        return len(REGISTRY.schemas())
    except Exception:
        return 0


def clear_screen() -> None:
    console.clear()


# ─── messages ───────────────────────────────────────────────────────────
def info(text: str) -> None:
    t = Text("  · ", style=ALT)
    t.append(text, style=MUTED)
    console.print(t)


def error(text: str) -> None:
    for i, line in enumerate(str(text).splitlines() or [""]):
        t = Text("  ! " if i == 0 else "    ", style=f"bold {ERR}")
        t.append(line, style=ERR)
        console.print(t)


def workspace_changed(path: str) -> None:
    _packet("CW", ALT)
    t = Text("  ")
    t.append("▸ ", style=ALT)
    t.append(_short_path(path, 70), style=INK)
    console.print(t)
    console.print()


# ─── prompts ────────────────────────────────────────────────────────────
def user_prompt(yolo: bool = False) -> str:
    width = console.size.width
    sep = Text("  ╶" + "─" * max(2, width - 4), style=FAINT)
    console.print(sep)

    bg = ERR_BG if yolo else INPUT_BG
    fg = ERR if yolo else ACCENT
    prompt_text = Text("  ")
    prompt_text.append(" >> ", style=f"bold {fg} {bg}")
    prompt_text.append("  ")

    with _suspend_live():
        try:
            return console.input(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return "/quit"


def ask(question: str) -> bool:
    q = Text("  ? ", style=f"bold {WARN}")
    q.append(question, style=INK)
    q.append("  (y/N) ", style=MUTED)
    with _suspend_live():
        try:
            ans = console.input(q).strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False
    return ans in ("y", "yes")


def ask_choice(question: str, options: list[str]) -> str:
    console.print()
    _packet("PK", WARN)
    head = Text("  ▌ ", style=WARN)
    head.append(question, style=INK)
    console.print(head)

    if not options:
        with _suspend_live():
            try:
                return console.input(Text("  ╘══ ▶ ", style=ACCENT)).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return ""

    for i, opt in enumerate(options, 1):
        row = Text("  ╞══  ", style=FAINT)
        row.append(f"{i} · ", style=MUTED)
        row.append(opt, style=INK)
        console.print(row)

    while True:
        with _suspend_live():
            try:
                raw = console.input(Text("  ╘══ ▶ ", style=ACCENT)).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return options[0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        for opt in options:
            if raw.lower() == opt.lower():
                return opt
        error(f"choose 1-{len(options)} or type a listed option")


def splash_choice(label: str, options: list[tuple[str, str]], default_idx: int = 1) -> str:
    console.print()
    _packet("SETUP", ALT)
    head = Text("  ▌ ", style=ALT)
    head.append(label, style=f"bold {INK}")
    console.print(head)
    console.print()

    for i, (_, desc) in enumerate(options, 1):
        marker = Text("▶", style=ACCENT) if i == default_idx else Text("▷", style=FAINT)
        row = Text("  ╞══  ", style=FAINT)
        row.append_text(marker)
        row.append(f"  {i:>2} · ", style=MUTED)
        row.append(desc, style=INK)
        console.print(row)
    console.print()

    while True:
        prompt = Text("  ╘══ ▶ ", style=ACCENT)
        prompt.append(f"[{default_idx}] ", style=FAINT)
        with _suspend_live():
            try:
                raw = console.input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                console.print()
                return options[default_idx - 1][0]
        if not raw:
            return options[default_idx - 1][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        for value, _ in options:
            if raw == value:
                return value
        error(f"choose 1-{len(options)} or type a listed value")


def read_multiline(prompt: str = "") -> str:
    """Read input until a blank line. Ctrl+D / Ctrl+Z also ends. Ctrl+C aborts."""
    if prompt:
        sys.stdout.write(prompt)
        sys.stdout.flush()
    lines: list[str] = []
    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip():
                break
            lines.append(line)
    except KeyboardInterrupt:
        return ""
    return "\n".join(lines).strip()


def feedback_prompt() -> str:
    """Prompt used by present_plan rejection — keeps colour out of plan.py."""
    head = Text("  > feedback ", style=GOLD)
    head.append("(paste or type, blank line to send, ctrl+c to abort)", style=FAINT)
    console.print(head)
    with _suspend_live():
        return read_multiline("  ")


# ─── streaming ──────────────────────────────────────────────────────────
def _start_stream(packet_code: str, packet_color: str, rail_color: str, ink_style: str) -> None:
    _packet(packet_code, packet_color)
    rail = Text("  ▌ ", style=rail_color)
    console.print(rail, end="")

    def on_char(ch: str) -> None:
        _emit_with_rail(rail, ch, ink_style)

    pipe = _Typewriter(on_char)
    pipe.start()
    _state["pipe"] = pipe


def _stop_stream() -> None:
    pipe: _Typewriter | None = _state.get("pipe")
    if pipe is not None:
        pipe.stop()
        _state["pipe"] = None
    console.print()
    console.print()


def thinking_start() -> None:
    _start_stream("TH", MUTED, MUTED, f"italic {FAINT}")


def thinking_chunk(text: str) -> None:
    pipe: _Typewriter | None = _state.get("pipe")
    if pipe is not None:
        pipe.write(text)


def thinking_end() -> None:
    _stop_stream()


def assistant_start() -> None:
    _start_stream("RP", ACCENT, ACCENT, INK)


def assistant_chunk(text: str) -> None:
    pipe: _Typewriter | None = _state.get("pipe")
    if pipe is not None:
        pipe.write(text)


def assistant_end() -> None:
    _stop_stream()


# ─── tools ──────────────────────────────────────────────────────────────
def tool_call(name: str, summary: str) -> None:
    width = console.size.width
    t = Text("  ")
    t.append("▸ ", style=ACCENT)
    t.append(name, style=INK)
    t.append("  ")
    t.append(_trim(summary, width - len(name) - 8), style=MUTED)
    console.print(t)


def tool_result(ok: bool, output: str = "") -> None:
    text = (output or "").rstrip()
    if not text or text == "(no output)":
        return
    width = console.size.width
    lines = text.splitlines()
    if not ok:
        head = Text("  ! ", style=f"bold {ERR}")
        head.append(_trim(lines[0], width - 6), style=ERR)
        console.print(head)
        rest, cap = lines[1:], 5
    else:
        rest, cap = lines, 5
    for line in rest[:cap]:
        row = Text("    ")
        row.append(_trim(line, width - 6), style=FAINT)
        console.print(row)
    if len(rest) > cap:
        more = Text("    ")
        more.append(f"+ {len(rest) - cap} more", style=FAINT)
        console.print(more)


# ─── panels ─────────────────────────────────────────────────────────────
def todos_panel(items: list[dict]) -> None:
    """Mirror the model's todos into the live region.

    When a live region is active, the next refresh picks up the new state
    automatically. When idle, render the panel inline so the user sees
    the change."""
    _state["todos"] = list(items)
    if _state["live"] is None:
        block = _build_todos_block()
        if block is not None:
            console.print(block)
            console.print()


def todos_status(items: Iterable[dict]) -> str:
    items = list(items)
    if not items:
        return ""
    done = sum(1 for t in items if t.get("status") == "done")
    doing = next((t for t in items if t.get("status") == "doing"), None)
    base = f"todos {done}/{len(items)} done"
    if not doing:
        return base
    return f"{base} · doing: {doing.get('text', '')}"


def plan_panel(title: str, body: str) -> None:
    console.print()
    _packet("PL", WARN)
    head = Text("  ▌ ", style=WARN)
    head.append(_trim(title, console.size.width - 4), style=f"bold {INK}")
    console.print(head)
    for line in body.splitlines() or [""]:
        row = Text("  ▌ ", style=WARN)
        row.append(line, style=INK)
        console.print(row)
    console.print()


def status_panel(rows: dict) -> None:
    console.print()
    _packet("STATUS", ALT)
    width = max((len(k) for k in rows), default=8)
    for k, v in rows.items():
        row = Text("  ")
        row.append(f"{k:<{width}}", style=MUTED)
        row.append("  ")
        row.append(str(v), style=INK)
        console.print(row)
    console.print()


def subagent_start(description: str) -> None:
    _packet("AG", HOT)
    t = Text("  ")
    t.append("▸ ", style=HOT)
    t.append(_trim(description, console.size.width - 4), style=INK)
    console.print(t)


def subagent_end(ok: bool, description: str) -> None:
    label = "✓ done" if ok else "✗ failed"
    color = ACCENT if ok else ERR
    t = Text("  ")
    t.append(label, style=color)
    t.append("  ")
    t.append(_trim(description, console.size.width - 12), style=FAINT)
    console.print(t)
    console.print()


def footer(
    model: str,
    ctx_pct: int,
    ctx_tokens: int,
    session_tokens: int,
    cwd: str,
    yolo: bool = False,
) -> None:
    friendly = _short_model(model)
    bar_w = 10
    filled = int(bar_w * max(0, min(100, ctx_pct)) / 100)
    if ctx_pct and filled == 0:
        filled = 1
    ctx_color = _ctx_color(ctx_pct)

    t = Text("  ─[ ", style=FAINT)
    t.append("STA", style=MUTED)
    t.append(" · ", style=FAINT)
    t.append(friendly, style=MUTED)
    if yolo:
        t.append(" · ", style=FAINT)
        t.append(" YOLO ", style=f"bold {ERR} {ERR_BG}")
    t.append(" · ", style=FAINT)
    t.append("▰" * filled, style=ctx_color)
    t.append("▱" * (bar_w - filled), style=FAINT)
    t.append(f" {ctx_pct:>2}%", style=f"bold {ctx_color}")
    t.append(" · ", style=FAINT)
    t.append(f"{_fmt_tokens(ctx_tokens)} / {_fmt_tokens(session_tokens)}", style=MUTED)
    t.append(" · ", style=FAINT)
    t.append(_short_path(cwd, 32), style=MUTED)
    t.append(" ]", style=FAINT)
    rule_w = max(2, console.size.width - len(t.plain) - 2)
    t.append("─" * rule_w, style=FAINT)

    console.print()
    console.print(t)
    console.print()


# ─── loader (the live region) ───────────────────────────────────────────
class Loader:
    """Owns the live region: pinned status + todos panel.

    The region stays up the entire turn — streaming text scrolls above
    it via `console.print`, so the user always sees a live status at the
    bottom and the current todos right above it.
    """
    def __init__(self, base_tokens: int = 0) -> None:
        self._base = base_tokens

    @property
    def running(self) -> bool:
        return _state["live"] is not None

    def start(self) -> None:
        _state["loader_start"] = time.monotonic()
        _state["loader_tokens"] = self._base
        _live_start()

    def stop(self) -> None:
        _live_stop()
