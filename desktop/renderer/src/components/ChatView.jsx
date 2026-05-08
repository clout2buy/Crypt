import { useState } from "react";
import {
  Bot,
  ChevronDown,
  CircleAlert,
  Eraser,
  LoaderCircle,
  RefreshCcw,
  SendHorizontal,
  Shield,
  Sparkles,
  Terminal,
  User,
  Wrench
} from "lucide-react";

const LANES = [
  { id: "planner", label: "Plan" },
  { id: "builder", label: "Code" }
];

const APPROVAL = [
  { id: "normal", label: "Manual" },
  { id: "edits", label: "Auto-work" },
  { id: "all", label: "Bypass" }
];

export function ChatView({
  activeRoute,
  events,
  lane,
  onClearView,
  onLaneChange,
  onNewSession,
  onRestart,
  onUserMessage,
  running,
  send,
  snapshot,
  transcriptRef
}) {
  const [prompt, setPrompt] = useState("");
  const visibleEvents = events.filter((event) => event.event !== "snapshot" && event.event !== "ready");

  const submit = () => {
    const text = prompt.trim();
    if (!text || running) return;
    setPrompt("");
    onUserMessage(text);
    send({ type: "sendPrompt", text, route: lane });
  };

  return (
    <main className="chat-shell">
      <header className="chat-topbar">
        <div>
          <h1>Crypt</h1>
          <span>{activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend"}</span>
        </div>
        <div className="top-actions">
          <button className="ghost-button" type="button" onClick={() => send({ type: "runCommand", command: "doctor" })}>
            <Terminal size={15} />
            Doctor
          </button>
          <button className="icon-button" type="button" onClick={onRestart} title="Restart backend">
            <RefreshCcw size={16} />
          </button>
        </div>
      </header>

      <section className="chat-feed" ref={transcriptRef}>
        {visibleEvents.length ? visibleEvents.map((event) => <EventBubble key={event.id} event={event} />) : <EmptyChat onNewSession={onNewSession} />}
        {running ? (
          <div className="message-row system">
            <div className="avatar"><LoaderCircle className="spin" size={16} /></div>
            <div className="message-card compact">Working through the shared Crypt engine.</div>
          </div>
        ) : null}
      </section>

      <footer className="composer-bar">
        <div className="composer-tools">
          <Segmented value={lane} options={LANES} onChange={onLaneChange} />
          <ApprovalToggle value={snapshot?.approvalMode || "edits"} onChange={(mode) => send({ type: "setApproval", mode })} />
          <button className="text-button" type="button" onClick={onClearView}>
            <Eraser size={14} />
            Clear view
          </button>
        </div>

        <div className="composer-input">
          <textarea
            value={prompt}
            disabled={running}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            placeholder="Ask Crypt to code, inspect, debug, or plan..."
            rows={1}
          />
          <button className="send-button" type="button" disabled={!prompt.trim() || running} onClick={submit}>
            {running ? <LoaderCircle className="spin" size={18} /> : <SendHorizontal size={18} />}
          </button>
        </div>
      </footer>
    </main>
  );
}

function EventBubble({ event }) {
  if (event.event === "snapshot" || event.event === "ready") return null;
  const Icon = iconFor(event);
  return (
    <div className={`message-row ${event.tone}`}>
      <div className="avatar"><Icon size={16} /></div>
      <article className={event.event === "toolCall" || event.event === "toolResult" ? "message-card tool-card" : "message-card"}>
        <header>
          <strong>{event.label}</strong>
          <time>{event.time}</time>
        </header>
        {event.text ? <pre>{event.text}</pre> : null}
      </article>
    </div>
  );
}

function EmptyChat({ onNewSession }) {
  return (
    <div className="empty-chat">
      <Sparkles size={24} />
      <h2>What should Crypt work on?</h2>
      <p>Same engine as the terminal. Cleaner cockpit.</p>
      <button className="primary-button" type="button" onClick={onNewSession}>Start fresh</button>
    </div>
  );
}

function Segmented({ value, options, onChange }) {
  return (
    <div className="segmented-control">
      {options.map((option) => (
        <button key={option.id} type="button" className={value === option.id ? "active" : ""} onClick={() => onChange(option.id)}>
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ApprovalToggle({ value, onChange }) {
  return (
    <label className="select-pill">
      <Shield size={14} />
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {APPROVAL.map((item) => (
          <option key={item.id} value={item.id}>{item.label}</option>
        ))}
      </select>
      <ChevronDown size={14} />
    </label>
  );
}

function iconFor(event) {
  if (event.tone === "user") return User;
  if (event.tone === "assistant") return Bot;
  if (event.tone === "tool") return Wrench;
  if (event.tone === "error") return CircleAlert;
  return Terminal;
}
