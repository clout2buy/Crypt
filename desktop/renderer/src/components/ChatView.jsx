import { RefreshCcw, Sparkles, Terminal } from "lucide-react";
import { Composer } from "./Composer.jsx";
import { EventStream } from "./EventStream.jsx";

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
  return (
    <main className="workspace-shell">
      <TopBar
        eyebrow={activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend"}
        onRestart={onRestart}
        send={send}
        title="Crypt"
      />

      <EventStream
        empty={<EmptyChat onNewSession={onNewSession} />}
        events={events}
        transcriptRef={transcriptRef}
      />

      <Composer
        activeRoute={activeRoute}
        lane={lane}
        onClearView={onClearView}
        onLaneChange={onLaneChange}
        onUserMessage={onUserMessage}
        running={running}
        send={send}
        snapshot={snapshot}
      />
    </main>
  );
}

export function TopBar({ eyebrow, onRestart, send, title }) {
  return (
    <header className="topbar">
      <div>
        <h1>{title}</h1>
        <span>{eyebrow}</span>
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
  );
}

function EmptyChat({ onNewSession }) {
  return (
    <div className="empty-chat">
      <Sparkles size={25} />
      <h2>What should Crypt work through?</h2>
      <p>Same terminal engine. Cleaner cockpit.</p>
      <div className="empty-actions">
        <button className="primary-button" type="button" onClick={onNewSession}>New session</button>
      </div>
    </div>
  );
}
