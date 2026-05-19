import { Activity, GitBranch, Terminal, Wrench } from "lucide-react";
import { Composer } from "./Composer.jsx";
import { TopBar } from "./ChatView.jsx";
import { EventBubble, visibleChatEvents } from "./EventStream.jsx";
import { MissionHud } from "./MissionControl.jsx";

export function AgentsView({
  activeRoute,
  events,
  lane,
  onClearView,
  onLaneChange,
  onRestart,
  onUserMessage,
  running,
  sessionName,
  send,
  snapshot,
  transcriptRef
}) {
  const visible = visibleChatEvents(events);
  const tools = visible.filter((event) => event.event === "tool").slice(-6).reverse();
  const route = activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend";
  const activeTask = snapshot?.activeTask ? `Task ${snapshot.activeTask}` : "No task running";

  return (
    <main className="workspace-shell agents-shell">
      <TopBar eyebrow={`${sessionName || "Agents"} - ${route}`} onRestart={onRestart} send={send} title="Agents" />

      <section className="agents-simple" ref={transcriptRef}>
        <MissionHud events={events} running={running} snapshot={snapshot} />

        <div className="agent-overview">
          <article className={running ? "agent-focus running" : "agent-focus"}>
            <div className="agent-focus-orb" aria-hidden="true">
              <Activity size={20} />
            </div>
            <div>
              <span>Current work</span>
              <strong>{running ? "Crypt is working" : "Ready when you are"}</strong>
              <p>{activeTask}</p>
            </div>
          </article>

          <article className="agent-focus">
            <div className="agent-focus-orb" aria-hidden="true">
              <GitBranch size={20} />
            </div>
            <div>
              <span>Model route</span>
              <strong>{activeRoute?.model || snapshot?.model || "Default model"}</strong>
              <p>{activeRoute?.provider || snapshot?.provider || "Provider pending"}</p>
            </div>
          </article>
        </div>

        <article className="agent-log">
          <header>
            <span><Wrench size={16} /> Recent actions</span>
          </header>
          <div className="ops-list">
            {tools.length ? tools.map((event) => <EventBubble compact event={event} key={event.id} />) : (
              <div className="empty-ops">
                Tool calls, checks, and file work will appear here while Crypt runs.
              </div>
            )}
          </div>
        </article>

        <article className="agent-log compact-routes">
          <header>
            <span><Terminal size={16} /> Routes</span>
          </header>
          <div className="route-list">
            {(snapshot?.routes || []).slice(0, 3).map((item) => (
              <div className="route-pill" data-role={item.role} key={item.role}>
                <span>{roleLabel(item.role)}</span>
                <strong>{item.model}</strong>
                <small>{item.provider}</small>
              </div>
            ))}
          </div>
        </article>
      </section>

      <Composer
        activeRoute={activeRoute}
        lane={lane}
        onClearView={onClearView}
        onLaneChange={onLaneChange}
        onUserMessage={onUserMessage}
        placeholder="Ask Crypt to inspect, coordinate tools, or check the run..."
        running={running}
        send={send}
        snapshot={snapshot}
      />
    </main>
  );
}

function roleLabel(role) {
  if (role === "planner") return "Plan";
  if (role === "builder") return "Code";
  if (role === "reviewer") return "Review";
  return role;
}
