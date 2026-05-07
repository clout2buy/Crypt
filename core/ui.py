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
    "tool_progress": None,  # current streaming tool-call assembly state
    "activity": "idle",     # last real observed harness/provider state
    "stream_kind": "",      # current streamed provider payload type
    "stream_chars": 0,      # exact chars received for the current stream kind
    "pipe": None,           # typewriter pipe (when streaming text/thinking)
    # Per-tool lifecycle entries shown in the live region while a tool is
    # in flight. Mirrors the shape of Claude Code's inProgressToolUseIDs:
    # each tool gets its own animated row (queued → running/approval → ok/err).
    "tool_lifecycle": {},   # tool_use_id -> {name, summary, state, detail, started_at}
    "tool_lifecycle_order": [],  # render order; preserved across state changes
}

_TODO_DWELL_SECONDS = 0.75


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


def _fmt_bytes(n: int) -> str:
    n = max(0, int(n or 0))
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f} MB"
    if n >= 10_000:
        return f"{n / 1_000:.0f} KB"
    if n >= 1_000:
        return f"{n / 1_000:.1f} KB"
    return f"{n} B"


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
def _elapsed() -> int:
    start = _state["loader_start"]
    if not start:
        return 0
    return max(1, int(time.monotonic() - start))


# Braille spinner — 10 frames, advances every ~100ms at our 8 fps refresh.
# The point is to give the user proof-of-life motion even when no chunks
# are arriving (e.g. Ollama is silent while the model composes a long
# tool_call internally — Crypt has nothing to render but the spinner says
# "still alive").
_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _spinner_frame() -> str:
    """Return the spinner glyph for the current monotonic time."""
    return _SPINNER_FRAMES[int(time.monotonic() * 10) % len(_SPINNER_FRAMES)]


def _build_status() -> Text:
    elapsed = _elapsed() or 1
    activity = str(_state.get("activity") or "waiting")
    stream_kind = str(_state.get("stream_kind") or "")
    stream_chars = int(_state.get("stream_chars") or 0)
    t = Text("  ")
    # Animated spinner instead of a static arrow: the user sees the prefix
    # advance every refresh, so a long silent generation phase doesn't read
    # as a frozen UI.
    t.append(_spinner_frame() + " ", style=ACCENT)
    t.append(_trim(activity, max(16, console.size.width - 42)), style=MUTED)
    t.append(f"  ({elapsed}s", style=FAINT)
    if stream_chars:
        t.append(f" · ↓ {_fmt_bytes(stream_chars)}", style=FAINT)
        if stream_kind:
            t.append(f" {stream_kind}", style=FAINT)
    if _state["loader_tokens"]:
        t.append(f" · {_state['loader_tokens']:,} tok", style=FAINT)
    t.append(")", style=FAINT)
    # If no chunks have arrived after 15 seconds, surface an abort hint.
    # Big cloud models (Qwen3-coder:480b, etc.) can take 30-120s to prefill
    # a long system+tools prompt — the wait is real, but the user should
    # know they can ctrl+C or /model out at any time.
    if stream_chars == 0 and elapsed >= 15:
        t.append("  · Ctrl+C to abort · /model to switch", style=WARN)
    return t


def _build_tool_progress() -> Text | None:
    progress = _state.get("tool_progress")
    if not progress:
        return None
    width = console.size.width
    name = _trim(progress.get("name") or "tool", 28)
    chars = int(progress.get("argument_chars") or 0)
    call_id = str(progress.get("call_id") or "")
    detail = str(progress.get("detail") or "")
    preview = progress.get("preview") or []
    label = f"assembling {name}"
    if chars:
        label += f" args {_fmt_bytes(chars)}"

    t = Text("  ")
    t.append("▸ ", style=ACCENT)
    t.append(_trim(label, max(12, width - 8)), style=f"bold {INK}")
    if call_id:
        t.append(f" · {call_id[:10]}", style=FAINT)
    if detail:
        detail_row = Text("\n    ", style=FAINT)
        detail_row.append(_trim(detail, max(12, width - 8)), style=MUTED)
        t.append_text(detail_row)
    for line in list(preview)[-3:]:
        preview_row = Text("\n    ", style=FAINT)
        preview_row.append(_trim(str(line), max(12, width - 8)), style=FAINT)
        t.append_text(preview_row)
    return t


def _build_todos_block() -> RenderableType | None:
    items = _state["todos"]
    if not items:
        return None

    done = sum(1 for it in items if it.get("status") == "done")
    active = next((it for it in items if it.get("status") == "doing"), None)
    elapsed = _elapsed()

    head = Text("  ")
    head.append("✓ " if done == len(items) else "● ", style=ACCENT if done == len(items) else WARN)
    if active:
        label = _trim(active.get("text", "working"), max(16, console.size.width - 34))
        head.append(f"{label}...", style=f"bold {WARN}")
    elif done == len(items):
        head.append(f"All {len(items)} todos completed", style=INK)
    else:
        head.append(f"{done}/{len(items)} todos", style=INK)
    if _state["live"] is not None and elapsed:
        head.append(f" ({elapsed}s", style=FAINT)
        if _state["loader_tokens"]:
            head.append(f" · {_fmt_tokens(_state['loader_tokens'])} tok", style=FAINT)
        head.append(")", style=FAINT)

    rows: list[Text] = [head]
    visible = items[:6]
    for it in visible:
        status = it.get("status", "pending")
        text = it.get("text", "")
        t = Text("    ")
        if status == "done":
            t.append("✓ ", style=ACCENT)
            t.append(text, style=f"strike {FAINT}")
        elif status == "doing":
            t.append("■ ", style=ACCENT)
            t.append(text, style=f"bold {INK}")
        else:
            t.append("□ ", style=FAINT)
            t.append(text, style=MUTED)
        rows.append(t)
    if len(items) > len(visible):
        more = Text("    ... ", style=FAINT)
        more.append(f"+{len(items) - len(visible)} more", style=MUTED)
        rows.append(more)
    return Group(*rows)


# ─── tool lifecycle ─────────────────────────────────────────────────────
# State machine (mirrors Claude Code's tool states):
#   queued    — header printed, dispatch hasn't started yet
#   approval  — waiting for the user's y/N
#   running   — tool.run() is executing
#   ok / err  — terminal; row leaves the live region and a final
#               "└─ ✓ ok (1.2s)" / "└─ ✗ failed (0.3s)" stamp lands in scrollback.
#
# Bullet glyph + color per state. The detail string is the default child
# label when no caller-supplied detail is given.
_BULLET_BY_STATE = {
    "queued":   ("○", FAINT,  "queued"),
    "approval": ("●", WARN,   "awaiting approval"),
    "running":  ("●", WARN,   "running"),
    "ok":       ("✓", ACCENT, "ok"),
    "err":      ("✗", ERR,    "failed"),
}


def _blink_bullet(bullet: str, state: str) -> str:
    """Blink the bullet for active states so the user sees motion at 4 fps."""
    if state in ("approval", "running"):
        # ~2 Hz toggle: visible on, then off, then on...
        if int(time.monotonic() * 2) % 2 == 0:
            return " "
    return bullet


def _elapsed_since(entry: dict) -> float:
    """Wall-clock time since this tool actually started running.

    The displayed timer must reflect *execution* time, not queue + approval
    typing time — otherwise a fast tool that sits 5s at an approval prompt
    looks like it took 5s to run. We use ``running_at`` when set (post-
    approval) and fall back to ``started_at`` for tools that never reached
    the running phase (denied / validation failed).
    """
    ref = entry.get("running_at") or entry.get("started_at")
    if not ref:
        return 0.0
    return max(0.0, time.monotonic() - ref)


def _build_in_flight_tools() -> RenderableType | None:
    """Render the live row for each tool currently in flight."""
    order = _state.get("tool_lifecycle_order") or []
    lifecycle = _state.get("tool_lifecycle") or {}
    rows: list[Text] = []
    for tid in order:
        entry = lifecycle.get(tid)
        if not entry:
            continue
        bullet, color, default_detail = _BULLET_BY_STATE.get(
            entry.get("state", "queued"), _BULLET_BY_STATE["queued"]
        )
        bullet = _blink_bullet(bullet, entry.get("state", "queued"))
        detail = entry.get("detail") or default_detail
        elapsed = int(_elapsed_since(entry))
        t = Text("    ")
        t.append("└─ ", style=FAINT)
        t.append(bullet + " ", style=f"bold {color}")
        t.append(_trim(detail, 38), style=MUTED)
        if elapsed >= 1:
            t.append(f"  ({elapsed}s)", style=FAINT)
        rows.append(t)
    return Group(*rows) if rows else None


def _refresh_live() -> None:
    """Push a fresh frame to the live region, if it's running."""
    live = _state.get("live")
    if live is not None:
        live.update(_LiveRenderable(), refresh=True)


def tool_begin(tool_id: str, name: str, summary: str) -> None:
    """Print the tool header line and register an in-flight row.

    The header (`▸ name  summary`) goes straight to scrollback so it stays
    pinned in the transcript above the live region. The in-flight row in
    the live region is what animates while the tool runs.
    """
    if not tool_id:
        # Backward compat for callers without a tool_use_id.
        tool_call(name, summary)
        return
    width = console.size.width
    t = Text("  ")
    t.append("▸ ", style=ACCENT)
    t.append(name, style=INK)
    t.append("  ")
    t.append(_trim(summary, max(8, width - len(name) - 8)), style=MUTED)
    console.print(t)

    _state.setdefault("tool_lifecycle", {})
    _state.setdefault("tool_lifecycle_order", [])
    _state["tool_lifecycle"][tool_id] = {
        "name": name,
        "summary": summary,
        "state": "queued",
        "detail": "",
        "started_at": time.monotonic(),
    }
    if tool_id not in _state["tool_lifecycle_order"]:
        _state["tool_lifecycle_order"].append(tool_id)
    _refresh_live()


def tool_set_state(tool_id: str, state: str, detail: str = "") -> None:
    """Move a tracked tool to a new state; refresh the live row."""
    if not tool_id:
        return
    entry = (_state.get("tool_lifecycle") or {}).get(tool_id)
    if entry is None:
        return
    entry["state"] = state
    if detail:
        entry["detail"] = detail
    elif state in _BULLET_BY_STATE:
        entry["detail"] = _BULLET_BY_STATE[state][2]
    # Stamp the running clock once: this is what the displayed timer should
    # measure. Approval typing time stays out of the elapsed counter.
    if state == "running" and "running_at" not in entry:
        entry["running_at"] = time.monotonic()
    _refresh_live()


def tool_end(tool_id: str, ok: bool, output: str = "") -> None:
    """Finalize a tool: print result body + a `└─ ✓/✗` footer; drop from live.

    Visual shape (success):
        ▸ bash  echo hi
            hi
            └─ ✓ ok  (0.2s)

    Visual shape (failure):
        ▸ bash  bogus
          ! command not found
            └─ ✗ failed  (0.1s)
    """
    if not tool_id:
        tool_result(ok, output)
        return
    lifecycle = _state.get("tool_lifecycle") or {}
    entry = lifecycle.pop(tool_id, None)
    order = _state.get("tool_lifecycle_order") or []
    if tool_id in order:
        order.remove(tool_id)

    if output:
        tool_result(ok, output)

    if entry is not None:
        elapsed = _elapsed_since(entry)
        bullet = "✓" if ok else "✗"
        color = ACCENT if ok else ERR
        label = "ok" if ok else "failed"
        line = Text("    ")
        line.append("└─ ", style=FAINT)
        line.append(bullet + " ", style=f"bold {color}")
        line.append(label, style=color)
        line.append(f"  ({elapsed:.1f}s)", style=FAINT)
        console.print(line)
    _refresh_live()


def tool_lifecycle_clear() -> None:
    """Drop any leftover lifecycle entries (defensive end-of-turn cleanup)."""
    _state["tool_lifecycle"] = {}
    _state["tool_lifecycle_order"] = []
    _refresh_live()


class _LiveRenderable:
    """Rebuilt by Rich on each refresh — pulls fresh state."""
    def __rich__(self) -> RenderableType:
        parts: list[RenderableType] = []
        in_flight = _build_in_flight_tools()
        if in_flight is not None:
            parts.append(in_flight)
        progress = _build_tool_progress()
        if progress is not None:
            parts.append(progress)
        todos = _build_todos_block()
        if todos is not None:
            parts.append(todos)
        else:
            parts.append(_build_status())
        return Group(*parts)


def _live_start() -> None:
    if _state["live"] is not None:
        return
    live = Live(
        _LiveRenderable(),
        console=console,
        # 8 fps: smooth enough that the per-tool bullet blink reads as
        # animation, low enough that idle CPU stays negligible.
        refresh_per_second=8,
        transient=True,
    )
    live.start(refresh=True)
    _state["live"] = live


def _live_stop(render_todos: bool = False) -> None:
    live = _state["live"]
    if live is None:
        return
    items = list(_state["todos"])
    live.stop()
    _state["live"] = None
    if render_todos:
        block = _build_todos_block()
        if block is not None:
            console.print(block)
            console.print()
    if items and all(it.get("status") == "done" for it in items):
        _state["todos"] = []


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
                # Adaptive cadence. The typewriter only exists to *smooth
                # bursts* — when chunks arrive in a steady drip (Qwen
                # Cloud's small-chunk pattern), we should not throttle.
                # Old base delay (12 ms) capped output at ~83 chars/sec
                # which made small-chunk providers feel sluggish.
                if backlog > 400:
                    delay = 0.0
                elif backlog > 120:
                    delay = 0.0005
                elif backlog > 30:
                    delay = 0.002
                else:
                    delay = 0.004  # ~250 chars/sec at low backlog
                if delay:
                    time.sleep(delay)


def _emit_with_rail(rail: Text, ch: str, ink_style: str) -> None:
    """Write one char honouring the rail prefix on newlines."""
    if ch == "\n":
        console.print()
        console.print(rail, end="")
    else:
        console.print(ch, style=ink_style, end="", markup=False, highlight=False)


# ─── welcome / chrome ───────────────────────────────────────────────────
# Calvin-S box-drawing brand. Tasteful, fits the aesthetic, 3 rows × 17 cols.
_BRAND = (
    "  ╔═╗╦═╗╦ ╦╔═╗╔╦╗",
    "  ║  ╠╦╝╚╦╝╠═╝ ║ ",
    "  ╚═╝╩╚═ ╩ ╩   ╩ ",
)

# Tiny rotating welcome lines so a fresh launch feels alive without being noisy.
_TAGLINES = (
    "transmission console",
    "ready when you are",
    "local-first coding harness",
    "TAOR loop online",
    "tools loaded · standing by",
    "channel open",
)


def _is_animated_tty() -> bool:
    if os.environ.get("CRYPT_NO_ANIMATION"):
        return False
    try:
        return bool(console.is_terminal and sys.stdout.isatty())
    except Exception:
        return False


def _tagline() -> str:
    # Deterministic per-launch but varies day to day so it doesn't feel repetitive.
    idx = int(time.time() // 60) % len(_TAGLINES)
    return _TAGLINES[idx]


def _info_rows(
    provider: str,
    model: str,
    auth_kind: str | None,
    auth_email: str | None,
    auth_plan: str | None,
    cwd: str,
) -> list[Text]:
    friendly = _short_model(model)
    auth_line = "no auth"
    if auth_kind == "oauth":
        auth_line = auth_email or "Anthropic OAuth"
        if auth_plan:
            auth_line += f"  ·  {auth_plan}"
    elif auth_kind:
        auth_line = auth_kind

    link = Text("    ", style=FAINT)
    link.append("●  ", style=ACCENT)
    link.append("LINK   ", style=MUTED)
    link.append(friendly, style=f"bold {INK}")
    link.append(" via ", style=FAINT)
    link.append(provider, style=MUTED)

    auth = Text("    ", style=FAINT)
    auth.append("◌  ", style=MUTED)
    auth.append("AUTH   ", style=MUTED)
    auth.append(auth_line, style=INK)

    cwd_row = Text("    ", style=FAINT)
    cwd_row.append("▸  ", style=MUTED)
    cwd_row.append("CWD    ", style=MUTED)
    cwd_row.append(_short_path(cwd, 60), style=INK)

    tools_row = Text("    ", style=FAINT)
    tools_row.append("▰  ", style=ACCENT)
    tools_row.append("TOOLS  ", style=MUTED)
    tools_row.append(f"{_tool_count()} loaded", style=INK)
    tools_row.append("   /status for details", style=FAINT)

    return [link, auth, cwd_row, tools_row]


def _brand_frame(reveal: int, scan_row: int | None) -> Group:
    """Render the brand at frame `reveal`. Lines 0..reveal are bright; the
    line at scan_row glows brightest (the scanline)."""
    rows: list[Text] = []
    for i, line in enumerate(_BRAND):
        if i == scan_row:
            t = Text(line, style=f"bold {ACCENT} {ACCENT_BG}")
        elif i <= reveal:
            t = Text(line, style=f"bold {ACCENT}")
        else:
            t = Text(line, style=FAINT)
        rows.append(t)
    return Group(*rows)


def welcome(
    provider: str,
    model: str,
    auth_kind: str | None = None,
    auth_email: str | None = None,
    auth_plan: str | None = None,
    cwd: str = ".",
) -> None:
    rows = _info_rows(provider, model, auth_kind, auth_email, auth_plan, cwd)
    width = console.size.width

    tag = Text("  ", style=FAINT)
    tag.append("─ ", style=FAINT)
    tag.append(_tagline(), style=MUTED)
    tag.append(" ", style=FAINT)
    tag.append("─" * max(2, width - len(tag.plain) - 2), style=FAINT)

    cmds = Text("  ▸  ", style=ACCENT)
    sep = Text("  ·  ", style=FAINT)
    for i, cmd in enumerate(("/help", "/status", "/model", "/yolo", "/quit")):
        if i:
            cmds.append_text(sep)
        cmds.append(cmd, style=MUTED)

    hint = Text("  ", style=FAINT)
    hint.append("type a request to begin · ", style=FAINT)
    hint.append("Ctrl+C", style=MUTED)
    hint.append(" interrupts a turn", style=FAINT)

    console.print()

    if not _is_animated_tty():
        # Headless / piped: skip animation entirely.
        console.print(_brand_frame(reveal=len(_BRAND) - 1, scan_row=None))
        console.print()
        for row in rows:
            console.print(row)
        console.print()
        console.print(tag)
        console.print()
        console.print(cmds)
        console.print()
        console.print(hint)
        console.print()
        return

    # Scanline reveal: brand fades in row-by-row with a brighter "scan"
    # row riding the wavefront. ~210ms total, easy to tolerate.
    with Live(
        _brand_frame(reveal=-1, scan_row=None),
        console=console,
        refresh_per_second=30,
        transient=False,
    ) as live:
        for i in range(len(_BRAND)):
            live.update(_brand_frame(reveal=i, scan_row=i))
            time.sleep(0.07)
        live.update(_brand_frame(reveal=len(_BRAND) - 1, scan_row=None))

    # Materialize info rows with a tiny stagger so the eye tracks them.
    console.print()
    for row in rows:
        console.print(row)
        time.sleep(0.045)
    console.print()
    console.print(tag)
    console.print()
    console.print(cmds)
    console.print()
    console.print(hint)
    console.print()


def _tool_count() -> int:
    try:
        from tools import REGISTRY
        return len(REGISTRY.schemas())
    except Exception:
        return 0


def clear_screen() -> None:
    # Rich's console.clear() only wipes the visible viewport. We also send
    # \x1b[3J so the scrollback buffer (Windows Terminal / iTerm / gnome-
    # terminal) is cleared — otherwise setup screens linger one scroll up.
    sys.stdout.write("\x1b[2J\x1b[3J\x1b[H")
    sys.stdout.flush()


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
def _drain_buffered_lines(timeout_ms: int = 80) -> list[str]:
    """After input() returns, drain any extra lines still in stdin from a paste.

    `input()` returns at the first \\n, but a multi-line paste delivers all the
    lines into the input buffer at once. This reads whatever's still queued so
    a paste can be assembled into a single message. Empty lines inside the
    paste are preserved."""
    lines: list[str] = []
    cur: list[str] = []
    deadline = time.monotonic() + timeout_ms / 1000

    if os.name == "nt":
        import msvcrt
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch == "\r":
                    continue
                if ch == "\n":
                    lines.append("".join(cur))
                    cur = []
                    deadline = time.monotonic() + timeout_ms / 1000
                else:
                    cur.append(ch)
                    deadline = time.monotonic() + timeout_ms / 1000
            else:
                time.sleep(0.001)
        if cur:
            lines.append("".join(cur))
        return lines

    import select
    while time.monotonic() < deadline:
        ready, _, _ = select.select([sys.stdin], [], [], 0.020)
        if not ready:
            continue
        line = sys.stdin.readline()
        if not line:
            break
        lines.append(line.rstrip("\r\n"))
        deadline = time.monotonic() + timeout_ms / 1000
    return lines


def _terminal_width() -> int:
    try:
        return max(20, console.size.width)
    except Exception:
        return 80


def _erase_recent_input_echo(lines: list[str], prompt_width: int = 8) -> None:
    """Best-effort cleanup for huge pasted input.

    Standard terminal input echoes pasted text before Python receives it. For
    long prompts that wrap badly, clear the echoed rows and replace them with a
    compact paste summary.
    """
    width = _terminal_width()
    rows = 0
    for i, line in enumerate(lines):
        prefix = prompt_width if i == 0 else 0
        rows += max(1, (prefix + len(line)) // width + 1)
    rows = max(1, min(rows, 80))
    sys.stdout.write(f"\x1b[{rows}A")
    for i in range(rows):
        sys.stdout.write("\x1b[2K")
        if i < rows - 1:
            sys.stdout.write("\x1b[1B")
    if rows > 1:
        sys.stdout.write(f"\x1b[{rows - 1}A")
    sys.stdout.flush()


def user_prompt(yolo: bool = False, approval: str = "manual") -> str:
    width = console.size.width
    sep = Text("  ╶" + "─" * max(2, width - 4), style=FAINT)
    console.print(sep)

    if yolo or approval == "yolo-all":
        bg, fg = ERR_BG, ERR
    elif approval in {"auto-edits", "auto-work"}:
        bg, fg = WARN_BG, WARN
    else:
        bg, fg = INPUT_BG, ACCENT
    prompt_text = Text("  ")
    prompt_text.append(" >> ", style=f"bold {fg} {bg}")
    prompt_text.append("  ")

    with _suspend_live():
        try:
            first = console.input(prompt_text)
        except (EOFError, KeyboardInterrupt):
            console.print()
            return "/quit"
        extras = _drain_buffered_lines(80)

    is_large = extras or len(first) > 220
    if is_large:
        # Multi-line paste detected — assemble the full message and show a
        # compact placeholder so the scrollback isn't flooded by the paste.
        lines = [first, *extras]
        _erase_recent_input_echo(lines)
        full = "\n".join(lines).strip()
        total_lines = 1 + len(extras)
        non_empty_chars = sum(len(line) for line in lines)
        placeholder = Text("  ")
        placeholder.append(
            f"[{total_lines} lines · {non_empty_chars} chars pasted]",
            style=f"italic {ALT}",
        )
        console.print(placeholder)
        return full
    return first.strip()


def ask(question: str) -> bool:
    approved, _ = confirm(question)
    return approved


def confirm(question: str) -> tuple[bool, str]:
    q = Text("  ? ", style=f"bold {WARN}")
    q.append(question, style=INK)
    q.append("  (y/N, or type feedback) ", style=MUTED)
    with _suspend_live():
        try:
            ans = console.input(q).strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return False, ""
    normalized = ans.lower()
    if normalized in ("y", "yes"):
        return True, ""
    if normalized in ("", "n", "no"):
        return False, ""
    return False, ans


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
# Live region must be paused while we stream typewriter text. Rich's Live
# treats partial-line writes (console.print(..., end="")) as part of the
# live region and erases them on the next refresh. Without pausing we get
# empty packet headers stacking up while the model's actual text vanishes.
def _start_stream(packet_code: str, packet_color: str, rail_color: str, ink_style: str) -> None:
    _state["streaming_paused_live"] = _state["live"] is not None
    if _state["streaming_paused_live"]:
        _live_stop()

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
    if _state.pop("streaming_paused_live", False):
        _live_start()


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


def tool_status(label: str, *, ok: bool | None = None) -> None:
    style = MUTED
    marker_style = FAINT
    if ok is True:
        style = ACCENT
        marker_style = ACCENT
    elif ok is False:
        style = ERR
        marker_style = ERR
    t = Text("    ")
    t.append("└ ", style=marker_style)
    t.append(_trim(label, console.size.width - 8), style=style)
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
def tool_progress(
    name: str,
    *,
    argument_chars: int = 0,
    call_id: str = "",
    detail: str = "",
    preview: list[str] | None = None,
) -> None:
    _state["tool_progress"] = {
        "name": name,
        "argument_chars": max(0, int(argument_chars or 0)),
        "call_id": call_id,
        "detail": detail,
        "preview": list(preview or []),
    }
    live = _state["live"]
    if live is not None:
        live.update(_LiveRenderable(), refresh=True)


def tool_progress_clear() -> None:
    if not _state.get("tool_progress"):
        return
    _state["tool_progress"] = None
    live = _state["live"]
    if live is not None:
        live.update(_LiveRenderable(), refresh=True)


def activity(label: str) -> None:
    next_label = str(label or "waiting")
    if _state.get("activity") == next_label:
        return
    _state["activity"] = next_label
    live = _state["live"]
    if live is not None and _state.get("tool_progress") is None:
        live.update(_LiveRenderable(), refresh=True)


def stream_delta(kind: str, text: str, *, activity: str | None = None) -> None:
    kind = str(kind or "stream")
    if _state.get("stream_kind") != kind:
        _state["stream_kind"] = kind
        _state["stream_chars"] = 0
    _state["stream_chars"] = int(_state.get("stream_chars") or 0) + len(text or "")
    _state["activity"] = str(activity or f"receiving {kind} stream")
    live = _state["live"]
    if live is not None and _state.get("tool_progress") is None:
        live.update(_LiveRenderable(), refresh=True)


def stream_clear() -> None:
    _state["stream_kind"] = ""
    _state["stream_chars"] = 0
    live = _state["live"]
    if live is not None and _state.get("tool_progress") is None:
        live.update(_LiveRenderable(), refresh=True)


def todos_panel(items: list[dict]) -> None:
    """Mirror the model's todos into the live region.

    The active live region owns rendering. When idle, this only updates
    state so completed/old lists do not get reprinted into the transcript."""
    _state["todos"] = list(items)
    live = _state["live"]
    if live is None:
        return

    # Force the transition onto the terminal before the model continues.
    # Without this, rapid todo calls can race Rich's auto-refresh and only
    # the final state becomes visible.
    live.update(_LiveRenderable(), refresh=True)
    if items:
        time.sleep(_TODO_DWELL_SECONDS)


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


def diff_preview(text: str) -> None:
    """Render a unified diff with colored +/- lines, dimmed context.

    Called by the tool dispatcher right before an approval prompt so the
    user sees the actual change instead of just a path. Truncates long
    diffs to ~40 visible lines with an "N more" footer."""
    if not text:
        return
    lines = text.splitlines()
    cap = 40
    visible = lines[:cap]
    omitted = max(0, len(lines) - cap)

    head = Text("  ┐ ", style=FAINT)
    head.append("preview", style=ALT)
    console.print(head)
    for line in visible:
        if line.startswith("+++") or line.startswith("---"):
            row = Text("  │ ", style=FAINT)
            row.append(line, style=MUTED)
        elif line.startswith("@@"):
            row = Text("  │ ", style=FAINT)
            row.append(line, style=ALT)
        elif line.startswith("+"):
            row = Text("  │ ", style=FAINT)
            row.append(line, style=ACCENT)
        elif line.startswith("-"):
            row = Text("  │ ", style=FAINT)
            row.append(line, style=ERR)
        else:
            row = Text("  │ ", style=FAINT)
            row.append(line, style=MUTED)
        console.print(row)
    if omitted:
        more = Text("  │ ", style=FAINT)
        more.append(f"... +{omitted} more line(s)", style=FAINT)
        console.print(more)
    foot = Text("  ┘", style=FAINT)
    console.print(foot)


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
    approval: str = "manual",
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
    if approval in {"auto-edits", "auto-work"}:
        t.append(" · ", style=FAINT)
        t.append(" AUTO-WORK ", style=f"bold {WARN} {WARN_BG}")
    elif yolo or approval == "yolo-all":
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
        _state["tool_progress"] = None
        _state["activity"] = "request sent"
        _state["stream_kind"] = ""
        _state["stream_chars"] = 0
        _live_start()

    def stop(self) -> None:
        _live_stop(render_todos=bool(_state["todos"]))
        # Defensive end-of-turn cleanup: if a tool was in flight when the
        # turn aborted (KeyboardInterrupt, exception), its row would
        # otherwise leak into the next turn's live region.
        tool_lifecycle_clear()
