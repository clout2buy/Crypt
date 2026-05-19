import { useEffect, useMemo, useState } from "react";
import { Monitor } from "lucide-react";
import { Composer } from "./Composer.jsx";
import { TopBar } from "./ChatView.jsx";
import { EventStream } from "./EventStream.jsx";
import { MissionHud } from "./MissionControl.jsx";
import { PreviewPanel } from "./PreviewPanel.jsx";
import { hasWebProjectActivity, mergePreviewArtifacts, previewArtifactsFromEvents } from "../lib/artifacts.js";

export function CodeView({
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
  const workspace = snapshot?.workspace || "";
  const [workspaceArtifacts, setWorkspaceArtifacts] = useState([]);
  const [dismissedArtifactIds, setDismissedArtifactIds] = useState([]);
  const eventArtifacts = useMemo(() => previewArtifactsFromEvents(events, workspace), [events, workspace]);
  const artifacts = useMemo(
    () => mergePreviewArtifacts(eventArtifacts, workspaceArtifacts).filter((item) => !dismissedArtifactIds.includes(item.id)),
    [dismissedArtifactIds, eventArtifacts, workspaceArtifacts]
  );
  const launchSignal = useMemo(() => hasWebProjectActivity(events), [events]);
  const [previewPinned, setPreviewPinned] = useState(false);
  const [closedPreviewKey, setClosedPreviewKey] = useState("");
  const previewKey = `${artifacts.map((item) => item.id).join("|")}::${launchSignal ? "launch" : ""}`;
  const previewWanted = previewPinned || artifacts.length > 0;
  const showPreview = previewWanted && previewKey !== closedPreviewKey;

  useEffect(() => {
    let alive = true;
    window.crypt?.listPreviewArtifacts?.(workspace).then((items) => {
      if (alive) setWorkspaceArtifacts(Array.isArray(items) ? items : []);
    });
    return () => {
      alive = false;
    };
  }, [workspace, events.length]);

  return (
    <main className="workspace-shell code-shell">
      <TopBar
        eyebrow={`${sessionName || "Code"} - ${activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend"}`}
        onRestart={onRestart}
        send={send}
        title="Code"
      />

      <section className={showPreview ? "code-workbench has-preview" : "code-workbench"}>
        <div className="code-chat decked">
          {!showPreview ? (
            <button
              className="preview-peek"
              type="button"
              onClick={() => {
                setClosedPreviewKey("");
                setPreviewPinned(true);
              }}
            >
              <Monitor size={16} />
              {launchSignal ? "Open app runner" : "Open preview when you need one"}
            </button>
          ) : null}
          <MissionHud compact events={events} running={running} snapshot={snapshot} />
          <EventStream
            compact
            empty={<div className="empty-ops">Start a code task and Crypt will stream edits, tools, and output here.</div>}
            events={events}
            send={send}
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
        {showPreview ? (
          <PreviewPanel
            artifacts={artifacts}
            autoStart={previewPinned}
            events={events}
            onClose={() => {
              setPreviewPinned(false);
              setClosedPreviewKey(previewKey);
            }}
            onDismissArtifact={(artifactId) => {
              setDismissedArtifactIds((ids) => (ids.includes(artifactId) ? ids : [...ids, artifactId]));
            }}
            workspace={workspace}
          />
        ) : null}
      </section>
    </main>
  );
}
