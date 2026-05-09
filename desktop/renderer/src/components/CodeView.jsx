import { Composer } from "./Composer.jsx";
import { TopBar } from "./ChatView.jsx";
import { EventStream } from "./EventStream.jsx";
import { PreviewPanel } from "./PreviewPanel.jsx";

export function CodeView({
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
  return (
    <main className="workspace-shell code-shell">
      <TopBar
        eyebrow={activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend"}
        onRestart={onRestart}
        send={send}
        title="Code"
      />

      <section className="code-workbench">
        <div className="code-chat">
          <EventStream
            compact
            empty={<div className="empty-ops">Start a code task and Crypt will stream edits, tools, and output here.</div>}
            events={events}
            transcriptRef={transcriptRef}
          />
          <Composer
            activeRoute={activeRoute}
            lane={lane}
            onClearView={onClearView}
            onLaneChange={onLaneChange}
            onUserMessage={onUserMessage}
            placeholder="Build, fix, test, or open a preview..."
            running={running}
            send={send}
            snapshot={snapshot}
          />
        </div>
        <PreviewPanel />
      </section>
    </main>
  );
}
