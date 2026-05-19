import {
  Bot,
  Cloud,
  Code2,
  GitBranch,
  MessageSquare,
  Palette,
  Plus,
  Settings2,
  Shield,
  Wrench
} from "lucide-react";

export function Sidebar({
  activeSessionId,
  activeView,
  connected,
  onChangeView,
  onNewSession,
  onSelectSession,
  sessions = [],
  snapshot
}) {
  const modes = [
    { id: "chat", label: "Chat", icon: MessageSquare },
    { id: "agents", label: "Agents", icon: GitBranch },
    { id: "code", label: "Code", icon: Code2 },
    { id: "design", label: "Design", icon: Palette }
  ];
  const nav = [
    { id: "providers", label: "Providers", icon: Cloud }
  ];

  return (
    <aside className={connected ? "sidebar linked" : "sidebar offline"}>
      <div className="app-brand">
        <div className="brand-glyph">
          <img src="./crypt-logo.svg" alt="" />
          <span aria-hidden="true" />
        </div>
        <div>
          <strong>Crypt</strong>
          <span>{connected ? "Linked" : "Offline"}</span>
        </div>
      </div>

      <div className="mode-tabs">
        {modes.map((item) => (
          <button
            key={item.id}
            className={activeView === item.id ? "active" : ""}
            type="button"
            onClick={() => onChangeView(item.id)}
          >
            <item.icon size={15} />
            {item.label}
          </button>
        ))}
      </div>

      <button className="new-session" type="button" onClick={onNewSession}>
        <Plus size={16} />
        New session
      </button>

      <nav className="nav-list">
        {nav.map((item) => (
          <button
            key={item.id}
            className={activeView === item.id ? "nav-item active" : "nav-item"}
            type="button"
            onClick={() => onChangeView(item.id)}
          >
            <item.icon size={16} />
            {item.label}
          </button>
        ))}
      </nav>

      <div className="sidebar-section">
        <span className="section-label">Current</span>
        <InfoLine icon={Bot} label="Model" value={snapshot?.model || "Loading"} />
        <InfoLine icon={Settings2} label="Provider" value={snapshot?.provider || "Pending"} />
        <InfoLine icon={Shield} label="Permissions" value={snapshot?.approval || "Manual"} />
        <InfoLine icon={Wrench} label="Tools" value={`${snapshot?.tools ?? 0} armed`} />
      </div>

      <div className="sidebar-section recents">
        <span className="section-label">Sessions</span>
        {sessions.length ? (
          sessions.map((session) => (
            <button
              className={session.id === activeSessionId ? "session-item active" : "session-item"}
              key={session.id}
              title={session.name}
              type="button"
              onClick={() => onSelectSession(session.id)}
            >
              <span>{session.view}</span>
              <strong>{session.name}</strong>
              <small>{session.events?.filter((event) => event.event === "user").length || 0}</small>
            </button>
          ))
        ) : (
          <span className="empty-sidebar">No sessions yet</span>
        )}
      </div>

      <div className="sidebar-footer">
        <span className={connected ? "dot online" : "dot"} />
        <span>{snapshot?.workspace || "Workspace not set"}</span>
      </div>
    </aside>
  );
}

function InfoLine({ icon: Icon, label, value }) {
  return (
    <div className="info-line">
      <Icon size={14} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
