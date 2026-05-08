import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  Bot,
  Brain,
  CheckCircle2,
  ChevronDown,
  CircleDot,
  Cpu,
  Database,
  FolderOpen,
  Gauge,
  GitBranch,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  MessageSquareText,
  Play,
  RefreshCcw,
  Route,
  SendHorizontal,
  Settings2,
  Shield,
  Sparkles,
  Terminal,
  Trash2,
  TriangleAlert,
  Workflow,
  Zap
} from "lucide-react";

const MAX_EVENTS = 160;
const ROUTE_LABELS = {
  planner: "Planner",
  builder: "Builder",
  reviewer: "Reviewer",
  fast: "Fast Lane",
  fallback: "Fallback"
};

const APPROVAL_MODES = [
  { id: "normal", label: "Manual", icon: LockKeyhole },
  { id: "edits", label: "Auto-work", icon: Shield },
  { id: "all", label: "Yolo", icon: Zap }
];

const QUICK_COMMANDS = [
  { label: "Status", icon: Activity, command: "status" },
  { label: "Doctor", icon: Gauge, command: "doctor" },
  { label: "Fresh Session", icon: Sparkles, command: "clear" }
];
let eventSequence = 0;

export function App() {
  const [snapshot, setSnapshot] = useState(null);
  const [events, setEvents] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [connected, setConnected] = useState(false);
  const [engineDraft, setEngineDraft] = useState({ provider: "", model: "" });
  const [workspaceDraft, setWorkspaceDraft] = useState("");
  const [activeRoute, setActiveRoute] = useState("planner");
  const [routeDraft, setRouteDraft] = useState({ provider: "", model: "" });
  const transcriptRef = useRef(null);

  useEffect(() => {
    if (!window.crypt) {
      setConnected(false);
      appendEvent(setEvents, {
        event: "error",
        text: "Electron bridge is unavailable. Run through npm run electron:dev or npm run electron:preview."
      });
      return undefined;
    }

    const unsubscribe = window.crypt.onEvent((payload) => {
      if (payload?.snapshot) setSnapshot(payload.snapshot);
      if (payload?.event === "ready" || payload?.event === "snapshot") setConnected(true);
      if (payload?.event === "daemonExit") setConnected(false);
      if (payload?.event === "daemonRestarting") setConnected(false);
      if (payload?.event === "sessionReset") {
        setEvents([normalizeEvent(payload)]);
        return;
      }
      appendEvent(setEvents, payload);
    });

    window.crypt.send({ type: "hello" });
    return unsubscribe;
  }, []);

  useEffect(() => {
    if (!snapshot) return;
    setEngineDraft({ provider: snapshot.provider, model: snapshot.model });
    setWorkspaceDraft(snapshot.workspace || "");
  }, [snapshot?.provider, snapshot?.model, snapshot?.workspace]);

  const providers = snapshot?.providers || [];
  const activeProvider = providers.find((provider) => provider.id === snapshot?.provider);
  const selectedRoute = useMemo(
    () => (snapshot?.routes || []).find((item) => item.role === activeRoute) || snapshot?.routes?.[0],
    [activeRoute, snapshot?.routes]
  );

  useEffect(() => {
    if (!selectedRoute) return;
    setRouteDraft({ provider: selectedRoute.provider, model: selectedRoute.model });
  }, [selectedRoute?.role, selectedRoute?.provider, selectedRoute?.model]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [events.length]);

  const running = Boolean(snapshot?.activeTask);
  const send = (command) => window.crypt?.send(command);
  const modelsFor = (providerId) => providers.find((provider) => provider.id === providerId)?.models || [];

  const submitPrompt = () => {
    const text = prompt.trim();
    if (!text || running) return;
    setPrompt("");
    appendEvent(setEvents, { event: "user", text });
    send({ type: "sendPrompt", text });
  };

  const runCommand = (command) => {
    if (command === "clear") {
      setEvents([]);
    }
    send({ type: "runCommand", command });
  };

  const applyEngine = () => {
    send({ type: "setProviderModel", provider: engineDraft.provider, model: engineDraft.model });
  };

  const applyRoute = () => {
    send({
      type: "setRoute",
      role: activeRoute,
      provider: routeDraft.provider,
      model: routeDraft.model
    });
  };

  const chooseWorkspace = async () => {
    const path = await window.crypt?.chooseDirectory?.();
    if (path) {
      setWorkspaceDraft(path);
      send({ type: "setWorkspace", path });
    }
  };

  const applyWorkspace = () => {
    const path = workspaceDraft.trim();
    if (path) send({ type: "setWorkspace", path });
  };

  return (
    <div className="app-shell">
      <div className="ambient-layer" aria-hidden="true" />
      <header className="top-chrome">
        <div className="brand-lockup">
          <img src="./crypt-logo.svg" alt="" className="brand-mark" />
          <div>
            <div className="eyebrow">Crypt Desktop</div>
            <h1>Operating Console</h1>
          </div>
        </div>
        <div className="chrome-status">
          <StatusPill connected={connected} running={running} />
          <button className="icon-button" type="button" onClick={() => window.crypt?.restart?.()} title="Restart engine">
            <RefreshCcw size={16} />
          </button>
        </div>
      </header>

      <main className="workspace-grid">
        <section className="left-stack">
          <HeroBar snapshot={snapshot} activeProvider={activeProvider} running={running} />

          <SpotlightPanel className="command-panel">
            <div className="panel-heading">
              <div>
                <span className="eyebrow">Command Channel</span>
                <h2>Shared engine session</h2>
              </div>
              <div className="quick-actions">
                {QUICK_COMMANDS.map((item) => (
                  <button key={item.command} type="button" className="soft-button" onClick={() => runCommand(item.command)}>
                    <item.icon size={15} />
                    {item.label}
                  </button>
                ))}
                <button type="button" className="soft-button danger-soft" onClick={() => setEvents([])}>
                  <Trash2 size={15} />
                  Clear View
                </button>
              </div>
            </div>

            <Transcript events={events} refTarget={transcriptRef} running={running} />

            <div className="composer">
              <div className="speaker-tag">You</div>
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={(event) => {
                  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
                    event.preventDefault();
                    submitPrompt();
                  }
                }}
                placeholder="Type a request. Slash commands route through the same backend."
                rows={2}
                disabled={running}
              />
              <button className="send-button" type="button" onClick={submitPrompt} disabled={!prompt.trim() || running}>
                {running ? <LoaderCircle className="spin" size={18} /> : <SendHorizontal size={18} />}
              </button>
            </div>
          </SpotlightPanel>
        </section>

        <aside className="right-stack">
          <EnginePanel
            snapshot={snapshot}
            providers={providers}
            engineDraft={engineDraft}
            setEngineDraft={setEngineDraft}
            modelsFor={modelsFor}
            onApply={applyEngine}
          />

          <ApprovalPanel snapshot={snapshot} send={send} />

          <RoutePanel
            snapshot={snapshot}
            providers={providers}
            activeRoute={activeRoute}
            setActiveRoute={setActiveRoute}
            routeDraft={routeDraft}
            setRouteDraft={setRouteDraft}
            modelsFor={modelsFor}
            onApply={applyRoute}
          />

          <WorkspacePanel
            snapshot={snapshot}
            workspaceDraft={workspaceDraft}
            setWorkspaceDraft={setWorkspaceDraft}
            chooseWorkspace={chooseWorkspace}
            applyWorkspace={applyWorkspace}
          />
        </aside>
      </main>
    </div>
  );
}

function HeroBar({ snapshot, activeProvider, running }) {
  return (
    <SpotlightPanel className="hero-panel">
      <div className="hero-copy">
        <span className="eyebrow">Local-first engineering harness</span>
        <div className="hero-title">
          <span>Crypt</span>
          <span className="gradient-word">Control Plane</span>
        </div>
      </div>
      <div className="metric-grid">
        <Metric icon={Cpu} label="Model" value={snapshot?.model || "loading"} />
        <Metric icon={Bot} label="Provider" value={snapshot?.provider || "pending"} badge={activeProvider?.status} />
        <Metric icon={LockKeyhole} label="Auth" value={snapshot?.auth || "checking"} good={snapshot?.authOk} />
        <Metric icon={Database} label="Tools" value={`${snapshot?.tools ?? 0} armed`} />
      </div>
      <div className="hero-footer">
        <div className="pulse-line">
          <span className={running ? "pulse-dot live" : "pulse-dot"} />
          {running ? "engine running" : "ready"}
        </div>
        <div className="workspace-path">{snapshot?.workspace || "workspace unresolved"}</div>
      </div>
    </SpotlightPanel>
  );
}

function EnginePanel({ snapshot, providers, engineDraft, setEngineDraft, modelsFor, onApply }) {
  const models = modelsFor(engineDraft.provider);
  const current = providers.find((provider) => provider.id === engineDraft.provider);
  const dirty = engineDraft.provider !== snapshot?.provider || engineDraft.model !== snapshot?.model;

  return (
    <SpotlightPanel className="side-panel">
      <PanelTitle icon={Settings2} eyebrow="Engine" title="Provider Core" />
      <div className="field-stack">
        <label>
          <span>Provider</span>
          <Select
            value={engineDraft.provider}
            onChange={(value) => {
              const nextModels = modelsFor(value);
              setEngineDraft({ provider: value, model: nextModels[0] || "" });
            }}
          >
            {providers.map((provider) => (
              <option key={provider.id} value={provider.id}>
                {provider.label}
              </option>
            ))}
          </Select>
        </label>
        <label>
          <span>Model</span>
          <Select value={engineDraft.model} onChange={(value) => setEngineDraft((draft) => ({ ...draft, model: value }))}>
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
            {engineDraft.model && !models.includes(engineDraft.model) ? (
              <option value={engineDraft.model}>{engineDraft.model}</option>
            ) : null}
          </Select>
        </label>
      </div>
      {current ? <ProviderCard provider={current} active={current.id === snapshot?.provider} /> : null}
      <button className="primary-button" type="button" onClick={onApply} disabled={!dirty}>
        <Play size={15} />
        Apply Engine
      </button>
    </SpotlightPanel>
  );
}

function ApprovalPanel({ snapshot, send }) {
  return (
    <SpotlightPanel className="side-panel">
      <PanelTitle icon={Shield} eyebrow="Approval" title="Execution Guard" />
      <div className="segmented-row">
        {APPROVAL_MODES.map((mode) => {
          const selected = snapshot?.approvalMode === mode.id;
          return (
            <button
              key={mode.id}
              type="button"
              className={selected ? "segment-button selected" : "segment-button"}
              onClick={() => send({ type: "setApproval", mode: mode.id })}
            >
              <mode.icon size={14} />
              {mode.label}
            </button>
          );
        })}
      </div>
      <button
        className={snapshot?.showThinking ? "toggle-row on" : "toggle-row"}
        type="button"
        onClick={() => send({ type: "setThinking", enabled: !snapshot?.showThinking })}
      >
        <Brain size={15} />
        Thinking Trace
        <span>{snapshot?.showThinking ? "On" : "Off"}</span>
      </button>
    </SpotlightPanel>
  );
}

function RoutePanel({ snapshot, providers, activeRoute, setActiveRoute, routeDraft, setRouteDraft, modelsFor, onApply }) {
  const models = modelsFor(routeDraft.provider);
  const selectedRoute = snapshot?.routes?.find((route) => route.role === activeRoute);
  const dirty = selectedRoute && (selectedRoute.provider !== routeDraft.provider || selectedRoute.model !== routeDraft.model);

  return (
    <SpotlightPanel className="side-panel route-panel">
      <PanelTitle icon={Route} eyebrow="Routing" title="Role Matrix" />
      <div className="route-tabs">
        {(snapshot?.routes || []).map((route) => (
          <button
            key={route.role}
            type="button"
            className={activeRoute === route.role ? "route-tab active" : "route-tab"}
            onClick={() => setActiveRoute(route.role)}
          >
            <span>{ROUTE_LABELS[route.role] || route.role}</span>
            <small>{route.model}</small>
          </button>
        ))}
      </div>
      <div className="field-stack two">
        <label>
          <span>Provider</span>
          <Select
            value={routeDraft.provider}
            onChange={(value) => {
              const nextModels = modelsFor(value);
              setRouteDraft({ provider: value, model: nextModels[0] || "" });
            }}
          >
            {providers.map((provider) => (
              <option key={provider.id} value={provider.id}>
                {provider.label}
              </option>
            ))}
          </Select>
        </label>
        <label>
          <span>Model</span>
          <Select value={routeDraft.model} onChange={(value) => setRouteDraft((draft) => ({ ...draft, model: value }))}>
            {models.map((model) => (
              <option key={model} value={model}>
                {model}
              </option>
            ))}
            {routeDraft.model && !models.includes(routeDraft.model) ? <option value={routeDraft.model}>{routeDraft.model}</option> : null}
          </Select>
        </label>
      </div>
      <button className="secondary-button" type="button" onClick={onApply} disabled={!dirty}>
        <Workflow size={15} />
        Save Route
      </button>
    </SpotlightPanel>
  );
}

function WorkspacePanel({ snapshot, workspaceDraft, setWorkspaceDraft, chooseWorkspace, applyWorkspace }) {
  return (
    <SpotlightPanel className="side-panel">
      <PanelTitle icon={FolderOpen} eyebrow="Workspace" title="Project Root" />
      <div className="workspace-control">
        <input value={workspaceDraft} onChange={(event) => setWorkspaceDraft(event.target.value)} onKeyDown={(event) => {
          if (event.key === "Enter") applyWorkspace();
        }} />
        <button className="icon-button" type="button" onClick={chooseWorkspace} title="Choose workspace">
          <FolderOpen size={16} />
        </button>
      </div>
      <div className="mini-grid">
        <Metric icon={GitBranch} label="Session" value={snapshot?.sessionId || "fresh"} />
        <Metric icon={Layers3} label="Mode" value={snapshot?.approval || "pending"} />
      </div>
    </SpotlightPanel>
  );
}

function Transcript({ events, refTarget, running }) {
  if (!events.length) {
    return (
      <div className="transcript empty" ref={refTarget}>
        <Terminal size={24} />
        <span>Ready for the first instruction.</span>
      </div>
    );
  }

  return (
    <div className="transcript" ref={refTarget}>
      {events.map((item) => (
        <EventRow key={item.id} item={item} />
      ))}
      {running ? (
        <div className="event-row active-row">
          <div className="event-icon"><LoaderCircle className="spin" size={16} /></div>
          <div>
            <div className="event-meta">running</div>
            <div className="event-text">Crypt is working through the shared engine.</div>
          </div>
        </div>
      ) : null}
    </div>
  );
}

function EventRow({ item }) {
  const iconMap = {
    user: MessageSquareText,
    taskStarted: LoaderCircle,
    taskProgress: Activity,
    taskFinished: Bot,
    commandResult: Terminal,
    taskFailed: TriangleAlert,
    error: TriangleAlert,
    daemonLog: Terminal,
    daemonExit: TriangleAlert,
    daemonRestarting: RefreshCcw,
    sessionReset: Sparkles,
    snapshot: CircleDot,
    ready: CheckCircle2
  };
  const Icon = iconMap[item.kind] || Terminal;

  return (
    <div className={`event-row ${item.tone}`}>
      <div className="event-icon">
        <Icon className={item.kind === "taskStarted" || item.kind === "daemonRestarting" ? "spin" : ""} size={16} />
      </div>
      <div className="event-body">
        <div className="event-meta">
          <span>{item.label}</span>
          <time>{item.time}</time>
        </div>
        <pre className="event-text">{item.text}</pre>
      </div>
    </div>
  );
}

function ProviderCard({ provider, active }) {
  return (
    <div className={active ? "provider-card active" : "provider-card"}>
      <div>
        <strong>{provider.label}</strong>
        <span>{provider.id}</span>
      </div>
      <Badge status={provider.status} />
      {provider.note ? <p>{provider.note}</p> : null}
    </div>
  );
}

function Metric({ icon: Icon, label, value, badge, good }) {
  return (
    <div className="metric">
      <Icon size={16} />
      <span>{label}</span>
      <strong>{value}</strong>
      {badge ? <Badge status={badge} /> : null}
      {good === false ? <TriangleAlert className="metric-warning" size={14} /> : null}
    </div>
  );
}

function PanelTitle({ icon: Icon, eyebrow, title }) {
  return (
    <div className="panel-title">
      <Icon size={16} />
      <div>
        <span className="eyebrow">{eyebrow}</span>
        <h3>{title}</h3>
      </div>
    </div>
  );
}

function Select({ children, value, onChange }) {
  return (
    <div className="select-wrap">
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {children}
      </select>
      <ChevronDown size={15} />
    </div>
  );
}

function Badge({ status }) {
  const label = status === "construction" ? "Construction" : status || "ready";
  return <span className={`badge ${status || "ready"}`}>{label}</span>;
}

function StatusPill({ connected, running }) {
  const label = running ? "Running" : connected ? "Linked" : "Offline";
  return (
    <div className={connected ? "status-pill online" : "status-pill"}>
      <span />
      {label}
    </div>
  );
}

function SpotlightPanel({ className = "", children }) {
  const onMouseMove = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    event.currentTarget.style.setProperty("--mouse-x", `${event.clientX - rect.left}px`);
    event.currentTarget.style.setProperty("--mouse-y", `${event.clientY - rect.top}px`);
  };
  return (
    <div className={`glass-panel ${className}`} onMouseMove={onMouseMove}>
      {children}
    </div>
  );
}

function appendEvent(setEvents, payload) {
  setEvents((current) => [...current, normalizeEvent(payload)].slice(-MAX_EVENTS));
}

function normalizeEvent(payload = {}) {
  const kind = payload.event || "system";
  const tone = kind.includes("Failed") || kind === "error" || kind === "daemonError" || kind === "daemonExit" ? "danger" : kind === "user" ? "user-tone" : "";
  const time = formatTime(payload.ts);
  const baseId = payload.id || "event";
  eventSequence += 1;
  return {
    id: `${baseId}-${kind}-${eventSequence}`,
    kind,
    tone,
    label: eventLabel(payload),
    time,
    text: eventText(payload)
  };
}

function eventLabel(payload) {
  if (payload.event === "user") return "you";
  if (payload.event === "taskFinished") return "assistant";
  if (payload.event === "commandResult") return payload.command || "command";
  if (payload.event === "taskProgress") return payload.phase || "progress";
  return (payload.event || "system").replace(/([A-Z])/g, " $1").toLowerCase();
}

function eventText(payload) {
  if (payload.text) return String(payload.text).trim();
  if (payload.error) return String(payload.error).trim();
  if (payload.prompt) return String(payload.prompt).trim();
  if (payload.event === "ready") return "Daemon connected.";
  if (payload.event === "snapshot") return "State synchronized.";
  if (payload.event === "daemonRestarting") return "Restarting backend daemon.";
  if (payload.event === "daemonExit") return `Daemon exited (${payload.code ?? "unknown"}${payload.signal ? `, ${payload.signal}` : ""}).`;
  return JSON.stringify(payload, null, 2);
}

function formatTime(ts) {
  const date = ts ? new Date(ts * 1000) : new Date();
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export default App;
