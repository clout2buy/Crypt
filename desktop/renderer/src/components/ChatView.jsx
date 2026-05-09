import { useEffect, useState } from "react";
import {
  Activity,
  Bot,
  ChevronDown,
  CheckCircle2,
  CircleAlert,
  Eraser,
  FolderOpen,
  BrainCircuit,
  LoaderCircle,
  RefreshCcw,
  Save,
  SendHorizontal,
  Shield,
  Sparkles,
  Terminal,
  User,
  Wrench,
  XCircle
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

const THINKING = [
  { id: "fast", label: "Fast" },
  { id: "think", label: "Think" },
  { id: "ultra", label: "Ultra" }
];

const HIDDEN_CHAT_EVENTS = new Set([
  "ready",
  "snapshot",
  "sessionReset",
  "taskStarted",
  "taskProgress",
  "daemonRestarting"
]);

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
  const [workspace, setWorkspace] = useState(snapshot?.workspace || "");
  const visibleEvents = events.filter((event) => !HIDDEN_CHAT_EVENTS.has(event.event));

  useEffect(() => {
    setWorkspace(snapshot?.workspace || "");
  }, [snapshot?.workspace]);

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
      </section>

      <footer className="composer-bar">
        <div className="composer-tools">
          <WorkspaceDock workspace={workspace} setWorkspace={setWorkspace} send={send} />
          <Segmented value={lane} options={LANES} onChange={onLaneChange} />
          <ApprovalToggle value={snapshot?.approvalMode || "edits"} onChange={(mode) => send({ type: "setApproval", mode })} />
          <div className="thinking-mode">
            <BrainCircuit size={14} />
            <Segmented value={snapshot?.thinkingMode || "fast"} options={THINKING} onChange={(mode) => send({ type: "setThinking", mode })} />
          </div>
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
        {running ? <div className="run-status">Running on {activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "active model"}...</div> : null}
      </footer>
    </main>
  );
}

function EventBubble({ event }) {
  const Icon = iconFor(event);
  if (event.event === "tool") {
    return (
      <div className={`message-row ${event.tone}`}>
        <div className="avatar"><Icon size={16} /></div>
        <ToolCard event={event} />
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
      <article className="message-card">
        <header>
          <strong>{event.label}</strong>
          <time>{event.time}</time>
        </header>
        {event.text ? <pre>{event.text}</pre> : null}
      </article>
    </div>
  );
}

function ToolCard({ event }) {
  const ok = event.rawEvent === "toolResult" && event.ok !== false;
  const failed = event.rawEvent === "toolResult" && event.ok === false;
  const StatusIcon = failed ? XCircle : ok ? CheckCircle2 : Activity;
  const status = failed ? "Failed" : ok ? "Complete" : "Running";

  return (
    <article className={failed ? "message-card tool-card failed" : "message-card tool-card"}>
      <header>
        <div className="tool-title">
          <span className="tool-glyph"><Wrench size={14} /></span>
          <span>
            <strong>{event.tool || "tool"}</strong>
            <small>{status}</small>
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
      {event.text ? <pre>{event.text}</pre> : null}
    </article>
  );
}

function ThinkingCard({ event }) {
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

function WorkspaceDock({ workspace, setWorkspace, send }) {
  return (
    <div className="workspace-dock">
      <FolderOpen size={14} />
      <input
        aria-label="Workspace path"
        value={workspace}
        onChange={(event) => setWorkspace(event.target.value)}
        spellCheck={false}
      />
      <button
        className="dock-icon"
        type="button"
        title="Browse workspace"
        onClick={async () => {
          const path = await window.crypt?.chooseDirectory?.();
          if (path) {
            setWorkspace(path);
            send({ type: "setWorkspace", path });
          }
        }}
      >
        <FolderOpen size={13} />
      </button>
      <button className="dock-icon" type="button" title="Save workspace" onClick={() => send({ type: "setWorkspace", path: workspace })}>
        <Save size={13} />
      </button>
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
  if (event.tone === "thinking") return BrainCircuit;
  if (event.tone === "error") return CircleAlert;
  return Terminal;
}
