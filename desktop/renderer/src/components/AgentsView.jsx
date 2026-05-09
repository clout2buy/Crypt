import { Bot, BrainCircuit, GitBranch, Network, Terminal, Wrench } from "lucide-react";
import { Composer } from "./Composer.jsx";
import { TopBar } from "./ChatView.jsx";
import { EventBubble, visibleChatEvents } from "./EventStream.jsx";

export function AgentsView({
  activeRoute,
  events,
  lane,
  onClearView,
  onLaneChange,
  onRestart,
  onUserMessage,
  running,
  send,
  snapshot,
  transcriptRef
}) {
  const visible = visibleChatEvents(events);
  const tools = visible.filter((event) => event.event === "tool").slice(-12).reverse();
  const thinking = visible.filter((event) => event.event === "thinking").slice(-3).reverse();
  const route = activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend";

  return (
    <main className="workspace-shell">
      <TopBar eyebrow={`Operations dock - ${route}`} onRestart={onRestart} send={send} title="Agents" />

      <section className="agents-grid" ref={transcriptRef}>
        <Panel title="Active Run" icon={Terminal}>
          <div className="run-card">
            <span className={running ? "pulse-dot online" : "pulse-dot"} />
            <div>
              <strong>{running ? "Engine running" : "Ready"}</strong>
              <p>{snapshot?.activeTask ? `task ${snapshot.activeTask}` : "No active task"}</p>
            </div>
          </div>
          <div className="metric-row">
            <Metric label="Approval" value={snapshot?.approval || "Manual"} />
            <Metric label="Thinking" value={snapshot?.thinkingMode || "fast"} />
            <Metric label="Tools" value={`${snapshot?.tools ?? 0}`} />
          </div>
        </Panel>

        <Panel title="Model Routes" icon={GitBranch}>
          <div className="route-list">
            {(snapshot?.routes || []).map((item) => (
              <div className="route-pill" key={item.role}>
                <span>{item.role}</span>
                <strong>{item.model}</strong>
                <small>{item.provider}</small>
              </div>
            ))}
          </div>
        </Panel>

        <Panel className="wide-panel" title="Tool Calls" icon={Wrench}>
          <div className="ops-list">
            {tools.length ? tools.map((event) => <EventBubble compact event={event} key={event.id} />) : <EmptyOps text="No tool calls in this session yet." />}
          </div>
        </Panel>

        <Panel title="Thinking" icon={BrainCircuit}>
          <div className="ops-list">
            {thinking.length ? thinking.map((event) => <EventBubble compact event={event} key={event.id} />) : <EmptyOps text="Switch to Think or Ultra to stream reasoning when a provider supports it." />}
          </div>
        </Panel>

        <Panel title="Subagents" icon={Network}>
          <div className="agent-stack">
            <AgentRole icon={Bot} label="Explorer" value={routeFor(snapshot, "planner")} />
            <AgentRole icon={Bot} label="Worker" value={routeFor(snapshot, "builder")} />
            <AgentRole icon={Bot} label="Reviewer" value={routeFor(snapshot, "reviewer")} />
          </div>
        </Panel>
      </section>

      <Composer
        activeRoute={activeRoute}
        lane={lane}
        onClearView={onClearView}
        onLaneChange={onLaneChange}
        onUserMessage={onUserMessage}
        placeholder="Ask Crypt to coordinate tools, subagents, checks, or review..."
        running={running}
        send={send}
        snapshot={snapshot}
      />
    </main>
  );
}

function Panel({ children, className = "", icon: Icon, title }) {
  return (
    <article className={`surface-panel ${className}`}>
      <header>
        <span><Icon size={16} /> {title}</span>
      </header>
      {children}
    </article>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AgentRole({ icon: Icon, label, value }) {
  return (
    <div className="agent-role">
      <Icon size={16} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyOps({ text }) {
  return <div className="empty-ops">{text}</div>;
}

function routeFor(snapshot, role) {
  const route = (snapshot?.routes || []).find((item) => item.role === role);
  return route ? `${route.provider} / ${route.model}` : "default";
}
