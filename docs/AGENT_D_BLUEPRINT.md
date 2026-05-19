# Agent D Blueprint

Agent D is a simple, professional assistant product built on Crypt's Python
runtime. The target is not "more autonomous" for its own sake. The target is a
trusted operator that helps a user move through real work: navigating sites,
sending email, setting up accounts, launching websites, preparing business
assets, and keeping the user oriented while it works.

## Product Shape

Agent D should feel like a capable operator with a clean control room:

- simple chat-first workflow
- visible task state, tool calls, and pending approvals
- clear resume context after interruptions
- one-click handoff between planning, acting, verifying, and reviewing
- professional web UI for non-terminal operation
- local-first session history and project memory

The assistant should be direct and useful. It should explain assumptions when
they affect user trust, but avoid flooding the user with internal reasoning.

## Runtime Decision

Python remains the best runtime choice for the core agent:

- Crypt is already Python and has the safety-critical pieces in place.
- Browser automation, file work, local process control, email APIs, and backend
  integrations are practical from Python.
- The runtime can stay local-first while exposing a thin app API for the UI.
- Existing Crypt guardrails can be reused instead of rebuilt.

The web UI should not become the agent brain. The UI should be a client over the
Python runtime.

Recommended split:

| Layer | Technology | Responsibility |
|---|---|---|
| Agent runtime | Python | loop, tools, permissions, sessions, subagents |
| App API | Python service | events, task commands, auth connectors, session access |
| Web UI | React or similar | chat, task board, approvals, browser/email/account panels |
| Desktop shell | Existing Electron path | local app wrapper and preview surfaces |

## Core Capabilities

### Navigation

Agent D should use browser automation for ordinary web navigation, form filling,
research, and setup workflows. It should keep the browser state visible in the
UI and report when it needs user input.

Required controls:

- inspect current page
- click/type/select/upload
- capture page evidence
- pause for login, CAPTCHA, payment, or sensitive data
- resume after user intervention

### Email

Email should be connector-based, not password-based. Agent D can draft, search,
summarize, label, and send emails only through approved providers or user-owned
authenticated sessions.

Required controls:

- draft before send
- recipient and attachment confirmation
- visible sent/not-sent status
- no silent mass messaging

### Accounts And Setup

Agent D can help create accounts and configure services, but must treat identity,
payment, phone verification, CAPTCHA, legal acceptance, and terms-sensitive steps
as user-confirmed actions.

Required controls:

- collect required setup fields
- generate strong credentials only through an approved secret flow
- pause for verification codes and human-only checks
- record what was created and where

### Websites And Business Work

Agent D should support practical business setup tasks:

- domain and hosting setup guidance
- website scaffolding
- copy, layout, asset, and form generation
- email/domain records checklist
- launch checklist and verification
- CRM, calendar, docs, and spreadsheet setup through connectors

The first useful version should focus on guided execution, not full unsupervised
business creation.

## Safety Contract

Agent D can act, but it should not pretend actions are safe just because they are
possible.

Hard confirmation required:

- sending email
- creating or deleting accounts
- submitting payment or legal forms
- publishing a website
- changing DNS, billing, auth, or security settings
- destructive filesystem or repo operations
- storing secrets

Hard stop required:

- bypassing CAPTCHA or access controls
- impersonation
- credential harvesting
- spam or mass account creation
- actions that violate a site's stated terms

## UX Principles

The web UI should be simple, calm, and operational:

- left rail for sessions/projects
- central chat/task stream
- right panel for active tool state, approvals, browser preview, and artifacts
- compact cards only for repeated task items or approval blocks
- clear status labels: planning, waiting, acting, verifying, done, blocked
- no decorative landing page as the primary product surface

The UI should make it obvious what Agent D is doing, what it is waiting on, and
what the user needs to approve.

## Crypt Fit

Crypt is a strong base because it already has:

- Python loop and provider adapters
- typed tools
- permissions and risky command classification
- write policy and read-before-edit behavior
- durable sessions and compaction
- subagent lanes for planning, exploration, implementation, and verification
- Electron desktop surface backed by the Python daemon

Agent D should extend this rather than replace it.

## First Build Order

1. Define Agent D task/event schema.
2. Add web-visible task states and approval objects.
3. Add browser automation as a governed tool group.
4. Add email connector abstractions with draft-first behavior.
5. Add setup workflow templates for accounts, websites, and business launch.
6. Add project memory for preferences, identities, domains, brands, and active
   setup state.
7. Build the professional web UI around chat, tasks, approvals, and browser
   evidence.
8. Add regression tests for approval gates and action logging.

## Non-Goals For MVP

- public SDK
- unsupervised account creation
- silent email sending
- payment automation without explicit user action
- bypassing human verification
- replacing Crypt's safety model

The MVP should prove that Agent D can reliably help a user complete a real setup
workflow with fewer mistakes, clear approvals, and durable memory.
