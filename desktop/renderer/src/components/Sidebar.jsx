import { Bot, Cloud, MessageSquare, Plus, Settings2, Shield, Wrench } from "lucide-react";

export function Sidebar({ activeView, connected, onChangeView, onNewSession, snapshot }) {
  const nav = [
    { id: "chat", label: "Chat", icon: MessageSquare },
    { id: "providers", label: "Providers", icon: Cloud }
  ];

  return (
    <aside className="sidebar">
      <div className="app-brand">
        <img src="./crypt-logo.svg" alt="" />
        <div>
          <strong>Crypt</strong>
          <span>{connected ? "Linked" : "Offline"}</span>
        </div>
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
