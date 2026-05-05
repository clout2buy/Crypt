# Crypt

A local-first coding agent harness for serious software work. Small enough
to read in an afternoon, built around the workflow primitives that make an
agent useful day to day.

```text
  ╔═╗╦═╗╦ ╦╔═╗╔╦╗
  ║  ╠╦╝╚╦╝╠═╝ ║
  ╚═╝╩╚═ ╩ ╩   ╩
```

Crypt runs in your terminal, owns your workspace, and keeps every turn on
disk. It speaks Anthropic, OpenAI-compatible endpoints, and Ollama (via
Ollama's native Anthropic-compatible API) with the same set of 25 tools
and a per-tool live lifecycle inspired by Anthropic's reference CLI —
eager dispatch, animated bullets, real reasoning streams, no fake
spinners.

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py setup       # interactive: pick provider, model, workspace
python main.py doctor      # 11 self-checks; verifies your environment
python main.py             # launch the agent
```

That's it. Crypt remembers your choices in `~/.crypt/config.json`, so the
second launch goes straight to the prompt.

---

## Providers

Crypt has three first-class transports. All four use the same internal
message format (Anthropic-style content blocks), so switching providers
mid-session via `/model` is a one-keystroke operation.

### Anthropic (OAuth or API key)

```bash
python main.py login                                          # OAuth
python main.py --provider anthropic --model claude-opus-4-7
```

OAuth tokens go to `~/.crypt/auth.json` (mode `0600`). Crypt uses raw
`httpx` and impersonates the official CLI's identity headers because the
Anthropic SDK injects Stainless fingerprints that the OAuth edge silently
rate-limits.

### OpenAI (or any OpenAI-compatible endpoint)

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://api.together.xyz/v1   # optional
python main.py --provider openai --model gpt-5
```

Works with Together, Fireworks, LM Studio, vLLM, and any other server that
serves Chat Completions.

### Ollama (local or cloud)

```bash
export OLLAMA_API_KEY=...                            # cloud only
python main.py --provider ollama --model qwen3-coder:480b-cloud
```

Crypt uses the official `anthropic` SDK pointed at Ollama's
Anthropic-compatible `/v1/messages` endpoint — the same surface Ollama
documents for the Anthropic SDK. You get eager tool dispatch (tool names
appear in the live UI before args finish streaming), per-byte arg
progress, automatic retries, and live thinking traces from Qwen3, Kimi
K2, GLM, and DeepSeek.

The Ollama host is normalized: `0.0.0.0`, bare `localhost`, missing scheme
all resolve to `http://localhost:11434`. `https://ollama.com` is left
alone.

---

## Live UI

Crypt's terminal renders every tool call as its own row that animates
through `queued → approval → running → ok / failed`, with a Braille
spinner glyph during silent provider waits and an inline reasoning stream
when the model thinks.

```text
▸ write_file  holographic-cube.html
┐ preview
│ +<!DOCTYPE html>
│ +<html lang="en">
│ ... +200 more line(s)
┘
? run this?  (y/N, or type feedback) y
    created holographic-cube.html
    └─ ✓ ok  (0.1s)
```

Highlights:

- **Per-tool live row.** Each tool gets a status bullet that blinks
  during pending states and freezes into the transcript with elapsed time
  on completion.
- **Eager dispatch.** As soon as the provider emits `content_block_start`
  for a tool, the row appears — no waiting for full args to stream.
- **Honest timer.** Elapsed time measures execution only, not approval
  typing time. A `write_file` that ran in 50ms shows `(0.0s)` even if
  you spent 30s reading the diff.
- **Spinner during silent waits.** The bottom status bar advances every
  refresh so a long prefill phase doesn't read as a frozen UI. After 15s
  with no chunks, you get an inline `Ctrl+C to abort · /model to switch`
  hint.
- **Thinking on by default.** Reasoning trace streams live in faint
  italic. Pass `--no-show-thinking` if you'd rather skip it.
- **Diff previews.** `edit_file`, `multi_edit`, and `write_file` render a
  unified diff with colored `+/-` lines before asking for approval.

---

## Tools

Crypt loads 25 tools dynamically from `tools/*.py` — drop a file, add it
to the registry, ship.

| Category | Tools |
|---|---|
| Read | `read_file`, `read_media`, `list_files`, `glob`, `grep` |
| Write | `edit_file`, `multi_edit`, `write_file` |
| Shell | `bash`, `bash_start`, `bash_poll`, `bash_kill` |
| Git | `git`, `git_branch`, `git_stage`, `git_commit` |
| Web | `web_search`, `web_fetch` |
| Plan / track | `present_plan`, `todos`, `ask_user`, `memory` |
| Workflow | `spawn_agent`, `set_workspace`, `open_file` |

### Bash

`bash` caps model-visible output at ~30 KB head + 30 KB tail. Anything
larger spills to `~/.crypt/runs/<timestamp>-<id>.log` and the path is
returned so the model can grep or tail without re-running. When a command
fails with no captured output (Windows `2>nul`, missing POSIX tools),
Crypt adds a `[hint: ...]` line diagnosing the likely cause.

### Multi-file edits

`multi_edit` applies edits across files atomically. All edits are
validated dry-run first; if any single edit fails, nothing is written.

### Read-before-edit

Existing files must be read before editing. Stale reads (file changed on
disk after the read) are rejected so the model can't over-write a manual
change it never observed.

### Web fetch sandbox

`web_fetch` refuses private and loopback addresses (RFC1918, link-local,
reserved). Override with `CRYPT_WEB_ALLOW_PRIVATE=1`. For host-level
control: `CRYPT_WEB_ALLOWED_HOSTS` (allowlist with `*.example.com`
wildcards) or `CRYPT_WEB_DENIED_HOSTS`.

### Subagents

`spawn_agent` runs a fresh-context, read-only subagent and returns its
final report. Pass excerpts via `context` so the subagent doesn't re-read
what the parent already has.

---

## Sessions, Memory, and Compaction

Every turn lands in `~/.crypt/projects/<workspace>/<session>.jsonl` as
append-only JSON. The slash commands below operate on that store:

| Command | Effect |
|---|---|
| `/sessions [--all]` | List resumable sessions |
| `/resume [id\|text]` | Swap the live thread to a previous session |
| `/compact` | Summarize old context into a continuation snapshot |
| `/memory` | Read durable memory from `~/.crypt/memory/MEMORY.md` |
| `/memory add <text>` | Save a durable workflow/project fact |
| `/background` | List background shell jobs |
| `/doctor` | Run 11 local harness self-checks |

Long sessions are also **micro-compacted** automatically: stale tool
results (old reads, greps, oversized bash output) get elided once the
context is half full, before the full-compaction threshold.

---

## Permissions

Three approval modes, switchable mid-session:

| Mode | Trigger | What it does |
|---|---|---|
| Manual | default | Every shell or edit asks once |
| Auto-edits | `/yolo` | File edits skip prompts; shell still asks |
| YOLO-all | `/yolo all` | All tool prompts bypassed |
| Safe | `/safe` | Return to manual |

Destructive operations (`rm -rf`, `git reset --hard`, `git push --force`,
`git clean`, `dd`, ...) **always** confirm even in YOLO. The detection
runs on the actual command string.

### Allow / deny rules

Drop `~/.crypt/permissions.json`:

```json
{
  "allow": [
    "bash:git status*",
    "bash:rg *",
    "bash:ls *"
  ],
  "deny": [
    "bash:rm -rf /*",
    "bash:git push --force*"
  ]
}
```

Each rule is `<tool>:<glob>` matching the tool's user-facing summary.
Globs use `*` and `?`. Deny wins over allow; allow skips every prompt
including the danger prompt (you opted in by name).

---

## Project Instructions

Crypt auto-loads project guidance from the workspace and parents:

- `CRYPT.md` — native Crypt instructions
- `AGENTS.md` — generic agent file (used by several tools)
- `CLAUDE.md` — Anthropic-CLI projects work without porting
- `.crypt/instructions.md` — alternate location

---

## Configuration

| Env var | Default | Effect |
|---|---|---|
| `CRYPT_ROOT` | saved or cwd | Workspace root for tools |
| `CRYPT_PROVIDER` | saved | `anthropic`, `openai`, or `ollama` |
| `CRYPT_NO_ANIMATION` | unset | Disables the startup splash |
| `CRYPT_WEB_ALLOW_PRIVATE` | unset | Lets `web_fetch` hit RFC1918 |
| `CRYPT_WEB_ALLOWED_HOSTS` | unset | Comma-separated allowlist |
| `CRYPT_WEB_DENIED_HOSTS` | unset | Comma-separated denylist |
| `ANTHROPIC_MODEL` | `claude-opus-4-7` | Default Anthropic model |
| `ANTHROPIC_MAX_TOKENS` | 4096 | Anthropic response cap |
| `ANTHROPIC_THINKING_BUDGET` | 512 | Anthropic thinking budget |
| `OPENAI_BASE_URL` | api.openai.com | Override for compat servers |
| `OPENAI_MAX_TOKENS` | 4096 | OpenAI response cap |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama URL (auto-normalized) |
| `OLLAMA_API_KEY` | `ollama` | Bearer token; required for cloud |
| `OLLAMA_MAX_TOKENS` | 16384 | Output budget |
| `OLLAMA_THINKING_BUDGET` | 8000 | Reasoning budget; 0 disables |
| `OLLAMA_TIMEOUT` | 600 | Request timeout in seconds |

---

## Architecture

```text
main.py              CLI, setup, provider/model/session startup
core/
  loop.py            interactive think-act-observe loop
  prompt.py          modular system prompt builder
  session.py         append-only JSONL transcripts and resume lookup
  compact.py         conversation + per-tool-result micro-compaction
  memory.py          durable memory and project instruction loading
  permissions.py     ~/.crypt/permissions.json allow/deny rules
  redact.py          best-effort secret redaction for tool outputs
  doctor.py          local harness self-checks
  file_state.py      read-before-edit and stale-file protection
  background.py      background shell task manager
  api.py             Anthropic, OpenAI, Ollama provider adapters
  runtime.py         session-scoped runtime hooks shared with tools
  ui.py              terminal UI (Rich live region + per-tool lifecycle)
  oauth.py           Anthropic OAuth login flow
  auth.py            credential resolution
  settings.py        env + config + saved defaults
tools/
  *.py               one tool per file, auto-registered
tests/
  test_*.py          pytest coverage for the load-bearing pieces
```

The agent loop is a straight TAOR cycle: the model emits a turn, Crypt
dispatches any `tool_use` blocks (in parallel when safe, sequentially
when approval-gated), appends `tool_result` blocks, and asks the model
to continue. The message array is the only state.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

111 tests covering the load-bearing pieces:

- Provider streaming contracts (Anthropic SSE, OpenAI Chat Completions,
  Ollama via the Anthropic SDK)
- Eager tool dispatch, partial-arg progress, and turn-end finalization
- Per-tool lifecycle UI (queued / approval / running / ok / err)
- Full + micro compaction
- Permission rule grammar (allow / deny / classify / danger)
- Read-before-edit invariants
- Registry dispatch (validation, preflight, schema enforcement)
- `edit_file` matching and atomic batches
- `multi_edit` all-or-nothing semantics
- `bash` output cap and Windows-friendly error diagnostics
- `web_fetch` SSRF sandbox
- Todos completion and lifecycle cleanup
- Status bar spinner, abort hints, elapsed-time honesty

---

## Files Crypt Writes

```text
~/.crypt/
  auth.json            OAuth tokens (mode 0600)
  config.json          provider, model, workspace defaults
  permissions.json     optional allow/deny rules (you create this)
  memory/MEMORY.md     durable user memory index + entries
  projects/<slug>/     per-workspace session JSONLs
  runs/                bash output spill files for oversized commands
  tasks/<sid>/         background shell job logs
```

---

## Design Rules

- Crypt identifies as Crypt.
- The system prompt is assembled from modular Crypt-native sections.
- The git snapshot in the system prompt is computed once per session per
  workspace so turns stay cache-friendly.
- Existing files must be read before editing; stale files are rejected.
- Long commands should run as background jobs instead of blocking the loop.
- Independent read-only tool calls and subagents can run in parallel.
- Tool results from the web are treated as untrusted external data.
- Common secrets in tool output are redacted before they enter the
  transcript.
- Durable memory is opt-in through the memory tool or `/memory add`.

---

## Status

Crypt is built for migration from heavier coding harnesses, but parity
should be judged by real workflow tests: resume a task after restart,
compact a long session, edit safely after reads, run background checks,
fetch docs, inspect media, and verify changes before reporting completion.
