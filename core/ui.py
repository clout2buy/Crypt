"""Terminal UI for crypt — TRANSMISSION style.

Truecolor (24-bit) ANSI; falls back to whatever the terminal renders. Public
API kept stable so the rest of the harness doesn't care about the redesign.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time


if os.name == "nt":
    os.system("")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


RST = "\033[0m"
B = "\033[1m"
DIM = "\033[2m"
IT = "\033[3m"
CLR = "\033[K"


def _rgb(r: int, g: int, b: int) -> str:
    return f"\033[38;2;{r};{g};{b}m"


def _bg(r: int, g: int, b: int) -> str:
    return f"\033[48;2;{r};{g};{b}m"


# Palette — TRANSMISSION (greenscale + amber + red)
INK = _rgb(0xc8, 0xe6, 0xd4)
MUTED = _rgb(0x6a, 0x8a, 0x78)
FAINT = _rgb(0x3a, 0x54, 0x48)
EDGE = _rgb(0x1a, 0x28, 0x20)
ACCENT = _rgb(0x26, 0xff, 0x9c)
ALT = _rgb(0x54, 0xe0, 0xff)
WARN = _rgb(0xff, 0xb8, 0x40)
ERR = _rgb(0xff, 0x40, 0x60)
HOT = _rgb(0xff, 0x7e, 0xc0)

# Dim backgrounds for badges
ACCENT_BG = _bg(0x10, 0x4a, 0x30)
WARN_BG = _bg(0x4a, 0x36, 0x10)
ERR_BG = _bg(0x4a, 0x14, 0x20)

# Backward-compat aliases (other modules import these)
GREEN = ACCENT
GREEN_BRT = ACCENT
GREEN_DARK = MUTED
CYAN = ALT
GOLD = WARN
RED = ERR
RED_SOFT = _rgb(0xb4, 0x32, 0x4c)


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")
_WAVE = "▁▂▃▄▅▆▇█▇▆▅▄▃▂▁"
_MIN_WIDTH = 60
_MAX_WIDTH = 96


def _w(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def clear_screen() -> None:
    """Wipe the terminal and home the cursor."""
    _w("\033[2J\033[3J\033[H")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _width() -> int:
    cols = shutil.get_terminal_size((90, 24)).columns
    return max(_MIN_WIDTH, min(_MAX_WIDTH, cols - 4))


def _trim(text: str, limit: int | None = None) -> str:
    limit = limit or _width()
    clean = " ".join(str(text).split())
    return clean if len(clean) <= limit else clean[: max(1, limit - 1)] + "."


def _short_model(model: str) -> str:
    m = model.removeprefix("claude-")
    parts = m.split("-")
    if parts and len(parts[-1]) >= 8 and parts[-1].isdigit():
        parts.pop()
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        parts[-2:] = [f"{parts[-2]}.{parts[-1]}"]
    return " ".join(parts)


def _short_path(path: str, limit: int = 44) -> str:
    path = str(path)
    if len(path) <= limit:
        return path
    keep = max(8, limit - 3)
    return "..." + path[-keep:]


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


def _seg_bar(pct: int, width: int = 10) -> str:
    pct = max(0, min(100, pct))
    filled = int(width * pct / 100)
    if pct and filled == 0:
        filled = 1
    on = ACCENT if pct < 50 else (WARN if pct < 80 else ERR)
    return f"{on}{'▰' * filled}{FAINT}{'▱' * (width - filled)}{RST}"


def _packet(code: str, color: str, ctx_pct: int | None = None) -> None:
    """Render a `─[ XX · time · CTX N% ]──────` strip flush against the right edge."""
    ts = time.strftime("%H:%M:%S")
    head_parts = [
        f"{FAINT}─[{RST} ",
        f"{color}{B}{code}{RST}",
        f" {FAINT}·{RST} {MUTED}{ts}{RST}",
    ]
    if ctx_pct is not None:
        head_parts.append(f" {FAINT}·{RST} {FAINT}CTX {ctx_pct}%{RST}")
    head_parts.append(f" {FAINT}]{RST}")
    head = "".join(head_parts)
    rule_w = max(2, _width() - len(_strip_ansi(head)) - 2)
    print(f"  {head}{FAINT}{'─' * rule_w}{RST}")


# ─── welcome ──────────────────────────────────────────────────────────────

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

    width = _width()
    tool_count = _tool_count()

    print()
    # Top edge with logo
    head = f"  {FAINT}╭─{RST}{ACCENT}{B} CRYPT {RST}{FAINT}── {MUTED}transmission console {FAINT}── "
    head_w = len(_strip_ansi(head)) - 2
    print(head + f"{FAINT}{'─' * max(2, width - head_w)}{RST}")

    # Three info rows with status bullets
    print(
        f"  {FAINT}│  {ACCENT}●{RST}  {MUTED}LINK   {RST}"
        f"{INK}{B}{friendly}{RST} {FAINT}via{RST} {MUTED}{provider}{RST}"
    )
    print(
        f"  {FAINT}│       {RST}{MUTED}{auth_line}{RST}"
    )
    print(
        f"  {FAINT}│  ◌  {MUTED}CWD    {RST}{INK}{_short_path(cwd, 60)}{RST}"
    )
    print(
        f"  {FAINT}│  ▰  {MUTED}TOOLS  {RST}{INK}{tool_count} loaded{RST}"
        f"  {FAINT}·{RST}  {FAINT}/status for details{RST}"
    )

    # Bottom edge
    print(f"  {FAINT}╰─{'─' * max(2, width - 2)}{RST}")
    print()
    # Command rail
    cmds = ["/help", "/status", "/login", "/logout", "/quit"]
    rail = f"  {ACCENT}▸{RST}  " + f"  {FAINT}·{RST}  ".join(f"{MUTED}{c}{RST}" for c in cmds)
    print(rail)
    print()


def _tool_count() -> int:
    """Lazy import to avoid circular dependency with tools.registry."""
    try:
        from tools import REGISTRY
        return len(REGISTRY.schemas())
    except Exception:
        return 0


# ─── prompt + status ──────────────────────────────────────────────────────

def user_prompt(status: str | None = None, yolo: bool = False) -> str:
    if status:
        print(f"  {FAINT}{status}{RST}")
    bg = ERR_BG if yolo else ACCENT_BG
    fg = ERR if yolo else ACCENT
    try:
        return input(f"  {bg}{fg}{B} >> {RST}  ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "/quit"


def todos_status(items: list[dict]) -> str:
    if not items:
        return ""
    done = sum(1 for t in items if t.get("status") == "done")
    total = len(items)
    doing = next((t for t in items if t.get("status") == "doing"), None)
    base = f"todos {done}/{total}"
    if not doing:
        return base
    text = " ".join(doing.get("text", "").split())
    room = _width() - len(base) - 12
    if room < 8:
        return base
    if len(text) > room:
        text = text[: room - 1] + "."
    return f"{base} · doing: {text}"


# ─── messages ─────────────────────────────────────────────────────────────

def info(text: str) -> None:
    print(f"  {ALT}·{RST} {MUTED}{text}{RST}")


def error(text: str) -> None:
    for i, line in enumerate(str(text).splitlines() or [""]):
        lead = f"{ERR}{B}!{RST}" if i == 0 else f" "
        print(f"  {lead} {ERR}{line}{RST}")


def workspace_changed(path: str) -> None:
    _packet("CW", ALT)
    print(f"  {ALT}▸{RST} {INK}{_short_path(path, 70)}{RST}")
    print()


# ─── streaming ────────────────────────────────────────────────────────────

def thinking_start() -> None:
    _packet("TH", MUTED)
    _w(f"  {FAINT}{IT}")


def thinking_chunk(text: str) -> None:
    _w(text.replace("\n", f"{RST}\n  {FAINT}{IT}"))


def thinking_end() -> None:
    _w(f"{RST}\n\n")


def assistant_start() -> None:
    _packet("RP", ACCENT)
    _w(f"  {ACCENT}▌{RST} {INK}")


def assistant_chunk(text: str) -> None:
    _w(text.replace("\n", f"{RST}\n  {ACCENT}▌{RST} {INK}"))


def assistant_end() -> None:
    _w(f"{RST}\n\n")


# ─── tools ────────────────────────────────────────────────────────────────

def tool_call(name: str, summary: str) -> None:
    summary = _trim(summary, _width() - len(name) - 8)
    print(f"  {ACCENT}▸{RST} {INK}{name}{RST}  {MUTED}{summary}{RST}")


def tool_result(ok: bool, output: str = "") -> None:
    text = (output or "").rstrip()
    if not text or text == "(no output)":
        return
    lines = text.splitlines()

    if not ok:
        head = _trim(lines[0], _width() - 6)
        print(f"  {ERR}!{RST} {ERR}{head}{RST}")
        rest = lines[1:]
        cap = 5
    else:
        rest = lines
        cap = 5

    for line in rest[:cap]:
        clean = _trim(line, _width() - 6)
        print(f"    {FAINT}{clean}{RST}")
    if len(rest) > cap:
        print(f"    {FAINT}+ {len(rest) - cap} more{RST}")


# ─── y/n + choice ─────────────────────────────────────────────────────────

def _getkey() -> str | None:
    try:
        if not sys.stdin.isatty():
            return None
        if os.name == "nt":
            import msvcrt
            ch = msvcrt.getwch()
            if ch == "\x03":
                raise KeyboardInterrupt
            return ch.lower()
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch.lower()
    except KeyboardInterrupt:
        raise
    except Exception:
        return None


def ask(question: str) -> bool:
    _w(f"  {WARN}?{RST} {INK}{question}{RST} {MUTED}(y/N){RST} ")
    while True:
        try:
            ch = _getkey()
        except KeyboardInterrupt:
            print()
            return False
        if ch is None:
            try:
                ans = input("").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                return False
            if ans in ("y", "yes"):
                return True
            if ans in ("", "n", "no"):
                return False
            _w(f"  {WARN}?{RST} {INK}{question}{RST} {MUTED}(y/N){RST} ")
            continue
        if ch == "y":
            print(f"{ACCENT}y{RST}")
            return True
        if ch in ("n", "\r", "\n", " ", "\x1b"):
            print(f"{MUTED}n{RST}")
            return False


def splash_choice(label: str, options: list[tuple[str, str]], default_idx: int = 1) -> str:
    """Setup-style picker. Options are (value, display) tuples. Returns value."""
    print()
    _packet("SETUP", ALT)
    print(f"  {ALT}▌{RST} {INK}{B}{label}{RST}")
    print()
    for i, (_, desc) in enumerate(options, 1):
        marker = f"{ACCENT}▶{RST}" if i == default_idx else f"{FAINT}▷{RST}"
        print(f"  {FAINT}╞══{RST}  {marker}  {MUTED}{i:>2} ·{RST} {INK}{desc}{RST}")
    print()
    while True:
        try:
            raw = input(f"  {ACCENT}╘══ ▶{RST} {FAINT}[{default_idx}]{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return options[default_idx - 1][0]
        if not raw:
            return options[default_idx - 1][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        for value, _ in options:
            if raw == value:
                return value
        print(f"  {ERR}!{RST} {ERR}choose 1-{len(options)} or type a listed value{RST}")


def ask_choice(question: str, options: list[str]) -> str:
    print()
    _packet("PK", WARN)
    print(f"  {WARN}▌{RST} {INK}{question}{RST}")
    if not options:
        try:
            return input(f"  {ACCENT}╘══ ▶{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""

    for i, opt in enumerate(options, 1):
        print(f"  {FAINT}╞══{RST}  {MUTED}{i} ·{RST} {INK}{opt}{RST}")
    while True:
        try:
            raw = input(f"  {ACCENT}╘══ ▶{RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return options[0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        for opt in options:
            if raw.lower() == opt.lower():
                return opt
        print(f"  {ERR}!{RST} {ERR}choose 1-{len(options)} or type a listed option{RST}")


# ─── panels ───────────────────────────────────────────────────────────────

def todos_panel(items: list[dict]) -> None:
    if not items:
        return
    print()
    _packet("JB", WARN)
    for it in items:
        status = it.get("status", "pending")
        text = _trim(it.get("text", ""), _width() - 22)
        if status == "done":
            bars = f"{ACCENT}▰▰▰▰▰{RST}"
            label = f"{FAINT}{text}{RST}"
            badge = f"{ACCENT_BG}{ACCENT}{B} DONE {RST}"
        elif status == "doing":
            bars = f"{WARN}▰▰▱▱▱{RST}"
            label = f"{INK}{text}{RST}"
            badge = f"{WARN_BG}{WARN}{B}  NOW {RST}"
        else:
            bars = f"{FAINT}▱▱▱▱▱{RST}"
            label = f"{MUTED}{text}{RST}"
            badge = f"{FAINT}next{RST}"
        print(f"  {bars}  {label}  {badge}")
    print()


def plan_panel(title: str, body: str) -> None:
    print()
    _packet("PL", WARN)
    print(f"  {WARN}▌{RST} {INK}{B}{_trim(title, _width() - 4)}{RST}")
    for line in body.splitlines() or [""]:
        print(f"  {WARN}▌{RST} {INK}{line}{RST}")
    print()


def status_panel(rows: dict) -> None:
    print()
    _packet("STATUS", ALT)
    width = max(len(k) for k in rows.keys()) if rows else 8
    for k, v in rows.items():
        print(f"  {MUTED}{k:<{width}}{RST}  {INK}{v}{RST}")
    print()


# ─── subagent markers ─────────────────────────────────────────────────────

def subagent_start(description: str) -> None:
    _packet("AG", HOT)
    print(f"  {HOT}▸{RST} {INK}{_trim(description, _width() - 4)}{RST}")


def subagent_end(ok: bool, description: str) -> None:
    label = f"{ACCENT}✓ done{RST}" if ok else f"{ERR}✗ failed{RST}"
    print(f"  {label}  {FAINT}{_trim(description, _width() - 12)}{RST}")
    print()


# ─── footer ───────────────────────────────────────────────────────────────

def footer(
    model: str,
    ctx_pct: int,
    ctx_tokens: int,
    session_tokens: int,
    cwd: str,
    yolo: bool = False,
) -> None:
    friendly = _short_model(model)
    ctx_c = _ctx_color(ctx_pct)
    yolo_tag = f"{FAINT} · {ERR_BG}{ERR}{B} YOLO {RST}" if yolo else ""
    bar = _seg_bar(ctx_pct)

    parts = [
        f"{FAINT}─[ {RST}",
        f"{MUTED}STA{RST}",
        f"{FAINT} · {RST}",
        f"{MUTED}{friendly}{RST}",
        yolo_tag,
        f"{FAINT} · {RST}",
        bar,
        f" {ctx_c}{B}{ctx_pct:>2}%{RST}",
        f"{FAINT} · {RST}",
        f"{MUTED}{_fmt_tokens(ctx_tokens)} / {_fmt_tokens(session_tokens)}{RST}",
        f"{FAINT} · {RST}",
        f"{MUTED}{_short_path(cwd, 32)}{RST}",
        f"{FAINT} ]{RST}",
    ]
    head = "".join(parts)
    visible = len(_strip_ansi(head))
    rule_w = max(2, _width() - visible - 2)
    print()
    print(f"  {head}{FAINT}{'─' * rule_w}{RST}")
    print()


# ─── animated loader ──────────────────────────────────────────────────────

class Loader:
    def __init__(self, base_tokens: int = 0) -> None:
        self._base_tokens = base_tokens
        self._running = False
        self._thread: threading.Thread | None = None
        self._tick = 0
        self._start = 0.0

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self._tick = 0
        self._start = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        _w(f"\r{CLR}")

    def _loop(self) -> None:
        wave_len = 14
        verbs = ("thinking", "reading", "planning", "writing", "checking", "working")
        while self._running:
            offset = self._tick % len(_WAVE)
            wave = (_WAVE + _WAVE)[offset : offset + wave_len]
            verb = verbs[(self._tick // 16) % len(verbs)]  # ~1.6s per verb
            elapsed = int(time.monotonic() - self._start) + 1
            tok = f" · {self._base_tokens:,} tok" if self._base_tokens else ""
            _w(
                f"\r{CLR}  {ACCENT}{wave}{RST}  "
                f"{MUTED}{verb}{RST} {FAINT}({elapsed}s{tok}){RST}"
            )
            time.sleep(0.10)
            self._tick += 1
