import { useState } from "react";
import {
  Activity,
  Bot,
  Box,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  CircleAlert,
  Clock3,
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

export function EventStream({ compact = false, empty, events, transcriptRef }) {
  const visibleEvents = visibleChatEvents(events);

  return (
    <section className={compact ? "event-feed compact" : "event-feed"} ref={transcriptRef}>
      {visibleEvents.length ? visibleEvents.map((event) => <EventBubble compact={compact} event={event} key={event.id} />) : empty}
    </section>
  );
}

export function EventBubble({ compact = false, event }) {
  const Icon = iconFor(event);

  if (event.event === "tool") {
    return (
      <div className={`message-row ${event.tone}`}>
        <div className="avatar"><Icon size={16} /></div>
        <ToolCard compact={compact} event={event} />
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
      <article className={compact ? "message-card compact-message" : "message-card"}>
        <header>
          <strong>{event.label}</strong>
          <time>{event.time}</time>
        </header>
        {event.text ? <pre>{event.text}</pre> : null}
      </article>
    </div>
  );
}

export function ToolCard({ compact = false, event }) {
  const [open, setOpen] = useState(false);
  const ok = event.rawEvent === "toolResult" && event.ok !== false;
  const failed = event.rawEvent === "toolResult" && event.ok === false;
  const running = !ok && !failed;
  const StatusIcon = failed ? XCircle : ok ? CheckCircle2 : Activity;
  const status = failed ? "Failed" : ok ? "Complete" : "Running";
  const body = event.text || "";
  const collapsed = compact || (!failed && body.length > 520);

  return (
    <article className={failed ? "message-card tool-card failed" : "message-card tool-card"}>
      <header>
        <div className="tool-title">
          <span className={running ? "tool-glyph running" : "tool-glyph"}><Wrench size={14} /></span>
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
          <Clock3 size={14} />
          Waiting for result
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
  if (event.event === "commandResult") return Box;
  return Terminal;
}
