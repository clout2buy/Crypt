# crypt

minimal Claude-Code–style coding agent. ~2k lines of Python, no framework, no
magic. think of it as the kind of thing you can read in an afternoon and then
fork into your own.

```
─[ STA · opus 4.7 · ▰▰▱▱▱▱▱▱▱▱  9% · 16k / 70k · ~/code ]──
```

## what it is

a terminal coding agent that runs the same Think → Act → Observe loop the big
ones do, with a real toolbox: file ops, bash, glob, grep, git, todos, plans,
subagents. plug in Claude via OAuth (no API key needed if you have a Max plan)
or any Ollama Cloud model.

built so people can learn from it, hack on it, or use it as a starting point
for their own harness. nothing's hidden — every tool is one file in `tools/`,
the loop is ~250 lines, the UI is one module.

## quick start

```bash
git clone https://github.com/clout2buy/Crypt.git
cd Crypt
pip install -r requirements.txt
python main.py
```

first run drops you into setup — pick a workspace, provider, and model. done.

```bash
# Claude — OAuth login (uses your Max plan, no API key)
python main.py login
python main.py

# Ollama Cloud
export OLLAMA_API_KEY=...
python main.py --provider ollama --model gpt-oss:120b-cloud
```

## the toolbox (14 tools)

| read-only       | edit / run         | meta            |
| --------------- | ------------------ | --------------- |
| `list_files`    | `edit_file` *      | `todos`         |
| `read_file` †   | `write_file`       | `present_plan`  |
| `glob`          | `bash`             | `ask_user`      |
| `grep`          | `open_file`        | `spawn_agent`   |
| `git` (ro)      |                    | `set_workspace` |

\* atomic batch edits + line-numbered failure hints + diff snippet on success
† line-range slicing via `offset` + `limit`, refuses binaries

each tool is one Python file. drop a `TOOL = Tool(...)` in `tools/foo.py` and
it auto-registers — no import edits anywhere.

## slash commands

```
/help              cheatsheet
/status            provider · auth · tools · todos · ctx
/yolo              auto-approve every tool (prompt turns red)
/thinking          toggle thinking-stream display
/cwd [path]        show or move workspace mid-session
/clear             wipe context + todos
/login   /logout   swap or sign out of Claude OAuth
```

## architecture

```
main.py            CLI + setup wizard
core/
  loop.py          the TAOR loop
  api.py           Anthropic + Ollama adapters (Anthropic-style internal format)
  oauth.py         PKCE browser-redirect flow for Claude.ai
  auth.py          token storage + auto-refresh
  runtime.py       session state shared with tools (yolo, cwd, subagent runner)
  ui.py            terminal UI — truecolor ANSI, no deps
  settings.py      paths, defaults, config persistence
tools/
  *.py             one tool per file, dynamic discovery via importlib
  fs.py            shared workspace-safe helpers (path resolve, binary detect)
```

three rules that keep it modular:

1. tools never import from `core.loop` — they go through `core.runtime`
2. `ui.py` has no app state, just renders what it's told
3. each tool owns its own teaching prompt — concatenated into the system prompt
   at startup, so the model learns when to use what

## design notes

- **no patch tool.** tried it, models produce broken unified diffs. `edit_file`
  does atomic batch substring replacements with helpful failure messages instead
  (line numbers of partial matches, hint when `\r` is in the search string).
- **no autorun test/check tool.** the model already has `bash` and knows what
  `pytest` is. adding a "smart check" tool just hands it more ways to be
  confidently wrong.
- **plan mode is model-driven.** there's no `/plan` slash command — the model
  calls `present_plan` itself when the task warrants it (3+ steps, multi-file,
  refactors). user approves, model executes.
- **OAuth uses the public Claude Code client_id.** same one Anthropic's
  official CLI uses. tokens land in `~/.crypt/auth.json` (mode 600 on Unix),
  never in the repo.

## not yet

- `git commit` / staging — only read-only git for now (status / diff / log / show)
- session persistence across runs
- project indexing / vector search
- hot-reload tools without restart

## license

MIT. fork it, ship it, change everything.
