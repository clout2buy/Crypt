# Crypt

Crypt is a local-first coding agent harness for serious software work. It is
small enough to understand, but built around the workflow primitives that make
an agent useful day to day: durable sessions, resume, project instructions,
memory, safe file edits, background shell jobs, web research, media reads,
todos, plans, git inspection, and subagents.

## Quick Start

```bash
pip install -r requirements.txt
python main.py setup
python main.py doctor
python main.py
```

Anthropic OAuth:

```bash
python main.py login
python main.py --provider anthropic --model claude-opus-4-7
```

OpenAI (or any OpenAI-compatible endpoint — Together, Fireworks, LM Studio, vLLM):

```bash
set OPENAI_API_KEY=...
# Optional: point at a compatible server instead of api.openai.com
set OPENAI_BASE_URL=https://api.together.xyz/v1
python main.py --provider openai --model gpt-5
```

Ollama Cloud:

```bash
set OLLAMA_API_KEY=...
python main.py --provider ollama --model gpt-oss:120b-cloud
```

Resume the latest session for the workspace:

```bash
python main.py --resume
```

## Recommended Setup

For the best experience, install [ripgrep](https://github.com/BurntSushi/ripgrep)
(`rg`) on your PATH. Crypt's `grep` tool detects it and uses it for an order-of-
magnitude speedup on real repos. A pure-Python fallback runs when `rg` is absent
so the tool still works on a fresh box. `python main.py doctor` reports which
backend is in use.

## Core Workflow

- Every session is written to `~/.crypt/projects/<project>/<session>.jsonl`.
- `/sessions` lists resumable conversations for the workspace.
- `/resume [id|text]` swaps the live thread to a previous session.
- `/compact` summarizes older context into a durable continuation snapshot.
- Long sessions are also micro-compacted automatically: stale tool results
  (old reads, greps, bash output) get elided once the context is half full.
- `/memory` reads durable memory from `~/.crypt/memory/MEMORY.md`.
- `/memory add <fact>` saves durable workflow/project facts.
- `/background` lists background shell jobs.
- `/doctor` runs local harness self-checks for sessions, tools, prompt identity,
  file safety, background jobs, ripgrep, and writable app dir.

## Tools

Crypt loads tools dynamically from `tools/*.py`.

- `read_file`, `read_media`, `list_files`, `glob`, `grep`
- `edit_file`, `multi_edit`, `write_file`
- `bash`, `bash_start`, `bash_poll`, `bash_kill`
- `git`, `git_branch`, `git_stage`, `git_commit`
- `web_search`, `web_fetch`
- `todos`, `present_plan`, `ask_user`, `memory`
- `spawn_agent`, `set_workspace`, `open_file`

### Bash Output

`bash` caps the model-visible output at ~30 KB head + 30 KB tail. Anything
larger spills to `~/.crypt/runs/<timestamp>-<id>.log` and the path is
returned in the result so the model can grep or tail it without re-running
the command. This stops a single noisy build from draining the context
window in one tool call.

When a command fails with no captured output (common on Windows when
`2>nul` swallows the error, or when a POSIX command like `wc` isn't
installed), Crypt adds a `[hint: ...]` line diagnosing the likely cause
so the model can self-correct without burning a turn on guessing.

### Multi-File Edits

`multi_edit` applies edits across one or more files atomically. All edits
are validated dry-run first; if any single edit fails (no match, ambiguous
match, missing file), nothing is written. Use it for renames, type-narrowing
sweeps, or any change where partial application would leave the project
broken.

### Edit Approval

In manual approval mode, `edit_file`, `multi_edit`, and `write_file` show
a unified-diff preview before asking for approval, so you see what's about
to change instead of approving a path. The same renderer applies to any
tool that exposes a `preview()` method.

### Web Fetch Sandbox

`web_fetch` refuses to hit private or loopback addresses (RFC1918, link-
local, reserved). Set `CRYPT_WEB_ALLOW_PRIVATE=1` to override (rare). For
finer-grained control, set `CRYPT_WEB_ALLOWED_HOSTS` (comma-separated, with
`*.example.com` wildcards) or `CRYPT_WEB_DENIED_HOSTS`. Allow rules are
enforced as an allowlist when set; deny rules always apply.

### Subagents

`spawn_agent` runs a fresh-context, read-only subagent and returns only its
final report. The optional `context` field lets the parent pass excerpts
(file snippets, prior findings) so the subagent does not re-read what the
parent already has.

## Permissions

Crypt has three approval modes:

- `manual` (default) — every shell or edit asks once
- `/yolo` — file edits skip prompts, shell still asks
- `/yolo all` — all tool prompts bypassed
- `/safe` — return to manual

Destructive operations (`rm -rf`, `git reset --hard`, `git push --force`,
`git clean`, `dd`, ...) always confirm even in yolo. Use the dedicated tools
when they exist (`edit_file` instead of `sed -i`, etc.) so this detection
stays accurate.

### Allow / Deny Rules

Drop a `~/.crypt/permissions.json` file to pre-approve specific tool calls
or hard-block others. Format:

```json
{
  "allow": [
    "bash:git status*",
    "bash:git diff*",
    "bash:rg *",
    "bash:ls *"
  ],
  "deny": [
    "bash:rm -rf /*",
    "bash:git push --force*"
  ]
}
```

Each rule is `<tool_name>:<glob>`. The glob matches the tool's user-facing
summary string (the same text shown in the approval prompt). Globs use `*`
and `?`, no regex. Precedence: deny wins over allow, allow skips every
prompt below it including the danger prompt (you opted in by name).

## Project Instructions

Crypt automatically loads project guidance from the current workspace and
parents:

- `CRYPT.md`
- `AGENTS.md`
- `CLAUDE.md`
- `.crypt/instructions.md`

Use `CRYPT.md` for native Crypt instructions. `CLAUDE.md` is supported so
existing projects can migrate without losing their local rules.

## Design Rules

- Crypt identifies as Crypt.
- The prompt is assembled from modular Crypt-native sections in
  `core/prompt.py`.
- The git snapshot in the system prompt is computed once per session per
  workspace so turns stay cache-friendly and fast.
- Existing files must be read before editing; stale files are rejected.
- Long commands should run as background jobs instead of blocking the loop.
- Independent read-only tool calls and subagents can run in parallel.
- Tool results from the web are treated as untrusted external data.
- Common secrets from the environment are redacted from tool output before they
  enter the transcript.
- Durable memory is opt-in through the memory tool or `/memory add`.

## Architecture

```text
main.py              CLI, setup, provider/model/session startup
core/
  loop.py            interactive think-act-observe loop
  prompt.py          modular Crypt-native system prompt builder
  session.py         append-only JSONL transcripts and resume lookup
  compact.py         conversation + per-tool-result micro-compaction
  memory.py          durable memory and project instruction loading
  permissions.py     ~/.crypt/permissions.json allow/deny rules
  redact.py          best-effort secret redaction for tool outputs
  doctor.py          local harness self-checks
  file_state.py      read-before-edit and stale-file protection
  background.py      background shell task manager
  api.py             Anthropic and Ollama provider adapters
  runtime.py         session-scoped runtime hooks shared with tools
  ui.py              terminal UI (animated splash + Rich live region)
tools/
  *.py               one tool per file, auto-registered
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers the load-bearing pieces: full + micro compaction,
permission rule grammar, read-before-edit invariants, registry dispatch
(allow/deny/classify), edit_file matching and atomic batches, multi_edit
all-or-nothing semantics, bash output cap and diagnostics, and web_fetch
sandbox. Add tests alongside any new core/* or tools/* logic — the doctor
is a smoke check, not coverage.

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

Set `CRYPT_NO_ANIMATION=1` to disable the startup animation if you prefer a
static splash (e.g. for slow terminals or screencast recording).

## Status

Crypt is built for migration from heavier coding harnesses, but parity should
be judged by real workflow tests: resume a task after restart, compact a long
session, edit safely after reads, run background checks, fetch docs, inspect
media, and verify changes before reporting completion.
