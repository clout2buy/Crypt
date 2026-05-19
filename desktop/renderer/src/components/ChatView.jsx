import { RefreshCcw, Terminal } from "lucide-react";
import { Composer } from "./Composer.jsx";
import { CryptGlyph } from "./CryptGlyph.jsx";
import { EventStream } from "./EventStream.jsx";
import { MissionHud, RitualBoard } from "./MissionControl.jsx";

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
  sessionName,
  send,
  snapshot,
  transcriptRef
}) {
  const launchRitual = (text) => {
    if (!text || running) return;
    onUserMessage(text);
    send({ type: "sendPrompt", text, route: lane });
  };

  return (
    <main className="workspace-shell chat-shell">
      <TopBar
        eyebrow={`${sessionName || "Chat"} - ${activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend"}`}
        onRestart={onRestart}
        send={send}
        title="Crypt"
      />

      <section className="run-deck-shell">
        <MissionHud events={events} running={running} snapshot={snapshot} />
        <EventStream
          empty={<EmptyChat onLaunchRitual={launchRitual} onNewSession={onNewSession} running={running} />}
          events={events}
          send={send}
          transcriptRef={transcriptRef}
        />
      </section>

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

function EmptyChat({ onLaunchRitual, onNewSession, running }) {
  return (
    <div className="empty-chat">
      <div className="empty-emblem" aria-hidden="true">
        <CryptGlyph name="sigil" size={34} />
        <span />
      </div>
      <h2>What are we building?</h2>
      <p>Pick a starting point or type exactly what you want.</p>
      <RitualBoard disabled={running} onLaunch={onLaunchRitual} />
      <div className="empty-actions">
        <button className="primary-button" type="button" onClick={onNewSession}>New session</button>
      </div>
    </div>
  );
}
