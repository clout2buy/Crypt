# Crypt UI-Agent Boundary

Crypt is the product UI-agent repository. Its job is to make the assistant feel
usable, visual, and integrated while relying on CryptCore for the runtime.

## Crypt Owns

- Electron/desktop renderer and product navigation.
- Chat, Agents, Code, Design, Preview, Mission Control, and future product
  surfaces.
- Visual design, animations, icons, splash screens, packaged app assets, and
  installer behavior.
- Product-agent roadmaps such as Agent D, WebUI reconstruction, personal
  operator flows, and desktop/mobile UX.
- UI adapters for the core CLI, package, daemon, or future app-server protocol.

## Crypt Should Consume From CryptCore

- `python -m crypt` terminal entrypoint behavior.
- Provider/model routing and auth.
- Tool orchestration, approvals, redaction, and safety policy.
- Sessions, memory, evidence, traces, artifacts, and task events.
- Subagents, skills, MCP bridge, project intelligence, benchmarks, and
  verification gates.

## Migration Target

The current repository still contains a Python core copy. Treat that as a
transition state. The target shape is:

- CryptCore remains the tested CLI/core package.
- Crypt depends on CryptCore by local path during development and by package or
  release artifact for production.
- Crypt talks to the core through `python -m crypt app-daemon` or a future
  formal protocol instead of importing private internals.
- Product UI work happens here, not in CryptCore.

## Do Not Put In Crypt

- New core runtime policy without a CryptCore change.
- Provider/tool behavior forks that drift from CryptCore.
- Credentials, generated traces, local caches, or packaged output committed as
  source.

If work changes how the assistant thinks, routes providers, edits files, runs
commands, stores sessions, or enforces safety, land the source-of-truth change
in CryptCore first. If work changes how the assistant looks, launches, presents
state, or feels as a product, land it here.
