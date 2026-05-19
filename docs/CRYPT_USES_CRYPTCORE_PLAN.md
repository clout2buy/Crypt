# Crypt Uses CryptCore Plan

This is the migration plan for making Crypt the actual UI-agent product while
CryptCore remains the pro CLI/core.

## Goal

Crypt should not compete with CryptCore for runtime ownership. Crypt should make
the assistant usable through desktop/UI surfaces and call the core through a
stable boundary.

## Phase 1 - Boundary Lock [started]

Done when:

- README identifies Crypt as the product UI-agent shell.
- `docs/CRYPT_UI_AGENT_BOUNDARY.md` defines what belongs here.
- Architecture docs state the current embedded core is transitional.

Progress:

- Added the UI-agent boundary doc.
- Added README and architecture pointers to CryptCore as the source-of-truth
  runtime.

## Phase 2 - Local Development Link

Done when:

- Development can point Crypt at a local `D:\CryptCore` checkout.
- Electron can launch the selected core through `CRYPT_BACKEND_ROOT` or an
  equivalent explicit setting.
- The UI displays which CryptCore path/version it is using.

## Phase 3 - Protocol Adapter

Done when:

- UI code talks to the core daemon/protocol through one adapter module.
- Renderer components do not assume private core internals.
- Contract fixtures cover chat events, tool events, approvals, sessions,
  artifacts, provider metadata, and errors.

## Phase 4 - Product Roadmap Migration

Done when:

- Legacy product-agent roadmaps currently parked in CryptCore docs are copied or
  moved here.
- CryptCore keeps only CLI/core parity and protocol roadmaps.
- Future WebUI, Agent D, desktop/mobile, visual QA, and product-shell work is
  tracked in this repository.

## Phase 5 - Core Copy Reduction

Done when:

- Any duplicated runtime modules are either removed from Crypt or clearly
  marked as compatibility shims.
- Runtime behavior changes land in CryptCore first.
- Crypt tests validate UI behavior against the core boundary, not a forked
  runtime.
