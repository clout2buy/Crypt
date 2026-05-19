import { useState } from "react";
import {
  Bot,
  Box,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleAlert,
  Clock3,
  ShieldAlert,
  Terminal,
  User,
  Wrench,
  XCircle
} from "lucide-react";

const HIDDEN_CHAT_EVENTS = new Set([
  "ready",
  "snapshot",
  "sessionReset",
  "taskStarted",
  "taskProgress",
  "daemonRestarting"
]);

export function visibleChatEvents(events) {
  return events.filter((event) => !HIDDEN_CHAT_EVENTS.has(event.event));
}

export function EventStream({ compact = false, empty, events, send, transcriptRef }) {
  const visibleEvents = visibleChatEvents(events);

  return (
    <section className={compact ? "event-feed compact" : "event-feed"} ref={transcriptRef}>
      {visibleEvents.length ? visibleEvents.map((event) => (
        <EventBubble compact={compact} event={event} key={event.id} send={send} />
      )) : empty}
    </section>
  );
}

export function EventBubble({ compact = false, event, send }) {
  const Icon = iconFor(event);
  const effect = effectFor(event);

  if (event.event === "approval") {
    return (
      <div className={`message-row ${event.tone}`}>
        <div className="avatar"><Icon size={16} /></div>
        <ApprovalCard event={event} send={send} />
      </div>
    );
  }

  if (event.event === "tool") {
    return (
      <div className={`message-row ${event.tone}`}>
        <div className="avatar"><Icon size={16} /></div>
        <ToolCard compact={compact} effect={effect} event={event} />
      </div>
    );
  }

  if (event.event === "thinking") {
    return (
      <div className="message-row thinking">
        <div className="avatar"><Icon size={16} /></div>
        <ThinkingCard event={event} />
      </div>
    );
  }

  return (
    <div className={`message-row ${event.tone}`}>
      <div className="avatar"><Icon size={16} /></div>
      <article className={`${compact ? "message-card compact-message" : "message-card"} ${effect}`}>
        <header>
          <strong>{event.label}</strong>
          <time>{event.time}</time>
        </header>
        {event.text ? <pre>{event.text}</pre> : null}
      </article>
    </div>
  );
}

function ApprovalCard({ event, send }) {
  const resolved = event.rawEvent === "approvalResolved";
  const [sent, setSent] = useState(false);
  const disabled = resolved || sent || !event.approvalId || !send;
  const submit = (approved) => {
    if (disabled) return;
    setSent(true);
    send({ type: "approvalResponse", approvalId: event.approvalId, approved });
  };

  return (
    <article className={event.danger ? "message-card approval-card danger" : "message-card approval-card"}>
      <header>
        <div className="tool-title">
          <span className={event.danger ? "tool-glyph failed" : "tool-glyph"}><ShieldAlert size={14} /></span>
          <span>
            <strong>{event.label}</strong>
            <small>{resolved ? event.text || "resolved" : event.reason || "permission required"}</small>
          </span>
        </div>
        <time>{event.time}</time>
      </header>
      {resolved ? null : <pre>{event.text}</pre>}
      {!resolved ? (
        <div className="approval-actions">
          <button className="primary-button compact" type="button" disabled={disabled} onClick={() => submit(true)}>
            Approve
          </button>
          <button className="text-button danger" type="button" disabled={disabled} onClick={() => submit(false)}>
            Deny
          </button>
        </div>
      ) : null}
    </article>
  );
}

export function ToolCard({ compact = false, effect = "", event }) {
  const [open, setOpen] = useState(false);
  const ok = event.rawEvent === "toolResult" && event.ok !== false;
  const failed = event.rawEvent === "toolResult" && event.ok === false;
  const active = event.rawEvent === "toolStarted" || event.rawEvent === "toolProgress";
  const queued = !ok && !failed && !active;
  const StatusIcon = failed ? XCircle : ok ? CheckCircle2 : Clock3;
  const status = failed ? "Failed" : ok ? "Complete" : active ? "Running" : "Queued";
  const body = event.text || "";
  const collapsed = compact || (!failed && body.length > 520);

  return (
    <article className={`${failed ? "message-card tool-card failed" : "message-card tool-card"} ${active ? "active" : ""} ${queued ? "queued" : ""} ${effect}`}>
      <header>
        <div className="tool-title">
          <span className={active ? "tool-glyph active" : "tool-glyph"}><Wrench size={14} /></span>
          <span>
            <strong>{event.tool || "tool"}</strong>
            <small>{event.rawEvent || "tool"}</small>
          </span>
        </div>
        <div className="tool-meta">
          <span className={failed ? "tool-status failed" : "tool-status"}>
            <StatusIcon size={13} />
            {status}
          </span>
          <time>{event.time}</time>
        </div>
      </header>

      {body ? (
        <>
          <pre className={collapsed && !open ? "collapsed-output" : ""}>{body}</pre>
          {collapsed ? (
            <button className="inline-disclosure" type="button" onClick={() => setOpen((value) => !value)}>
              <ChevronDown className={open ? "open" : ""} size={14} />
              {open ? "Collapse output" : "Expand output"}
            </button>
          ) : null}
        </>
      ) : (
        <div className="tool-skeleton">
          <StatusIcon size={14} />
          {active ? "Tool is running" : "Queued by model"}
        </div>
      )}
    </article>
  );
}

export function ThinkingCard({ event }) {
  const [open, setOpen] = useState(false);
  return (
    <article className="message-card thinking-card">
      <header>
        <button className="thinking-toggle" type="button" onClick={() => setOpen((value) => !value)}>
          <ChevronDown className={open ? "open" : ""} size={15} />
          <strong>{event.label}</strong>
          <span>{open ? "expanded" : "collapsed"}</span>
          <span className="thinking-wave" aria-hidden="true"><i /><i /><i /></span>
        </button>
        <time>{event.time}</time>
      </header>
      {open && event.text ? <pre>{event.text}</pre> : null}
    </article>
  );
}

function iconFor(event) {
  if (event.tone === "user") return User;
  if (event.tone === "assistant") return Bot;
  if (event.tone === "tool") return Wrench;
  if (event.tone === "thinking") return BrainCircuit;
  if (event.tone === "error") return CircleAlert;
  if (event.event === "approval") return ShieldAlert;
  if (event.event === "commandResult") return Box;
  return Terminal;
}

function effectFor(event) {
  const text = `${event.label || ""}\n${event.text || ""}`.toLowerCase();
  if (text.includes("verification note:")) return "";
  if (
    text.includes("pytest") ||
    text.includes("vitest") ||
    text.includes("npm test") ||
    text.includes("test suite") ||
    /\btests?\s+(passed|failed|skipped|errored)\b/.test(text) ||
    /\b\d+\s+(passed|failed|skipped|errors?)\b/.test(text)
  ) {
    return "test-result-card";
  }
  return "";
}
