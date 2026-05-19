import {
  Monitor,
  Palette,
  SlidersHorizontal
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Composer } from "./Composer.jsx";
import { CryptGlyph } from "./CryptGlyph.jsx";
import { TopBar } from "./ChatView.jsx";
import { EventStream } from "./EventStream.jsx";
import { MissionHud } from "./MissionControl.jsx";
import { PreviewPanel } from "./PreviewPanel.jsx";
import { hasWebProjectActivity, mergePreviewArtifacts, previewArtifactsFromEvents } from "../lib/artifacts.js";
import {
  buildDesignPrompt,
  DESIGN_DEVICES,
  DESIGN_DIRECTIONS,
  DESIGN_SURFACES
} from "../lib/designBrief.js";

export function DesignView({
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
  const route = activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "Shared backend";
  const workspace = snapshot?.workspace || "";
  const [surface, setSurface] = useState(DESIGN_SURFACES[0]);
  const [direction, setDirection] = useState(DESIGN_DIRECTIONS[0]);
  const [device, setDevice] = useState(DESIGN_DEVICES[0]);
  const [workspaceArtifacts, setWorkspaceArtifacts] = useState([]);
  const [dismissedArtifactIds, setDismissedArtifactIds] = useState([]);
  const eventArtifacts = useMemo(() => previewArtifactsFromEvents(events, workspace), [events, workspace]);
  const artifacts = useMemo(
    () => mergePreviewArtifacts(eventArtifacts, workspaceArtifacts).filter((item) => !dismissedArtifactIds.includes(item.id)),
    [dismissedArtifactIds, eventArtifacts, workspaceArtifacts]
  );
  const launchSignal = useMemo(() => hasWebProjectActivity(events), [events]);

  useEffect(() => {
    let alive = true;
    window.crypt?.listPreviewArtifacts?.(workspace).then((items) => {
      if (alive) setWorkspaceArtifacts(Array.isArray(items) ? items : []);
    });
    return () => {
      alive = false;
    };
  }, [workspace, events.length]);

  const sendDesign = (command = {}) => {
    if (command.type !== "sendPrompt") return send(command);
    return send({
      ...command,
      text: buildDesignPrompt({
        device,
        direction,
        surface,
        text: command.text
      })
    });
  };

  return (
    <main className="workspace-shell design-shell">
      <TopBar eyebrow={`${sessionName || "Design"} - ${route}`} onRestart={onRestart} send={send} title="Design" />

      <section className="design-workbench">
        <div className="design-chat decked">
          <div className="design-brief-bar">
            <BriefSelect icon={SlidersHorizontal} label="Surface" options={DESIGN_SURFACES} value={surface} onChange={setSurface} />
            <BriefSelect icon={Palette} label="Direction" options={DESIGN_DIRECTIONS} value={direction} onChange={setDirection} />
            <BriefSelect icon={Monitor} label="Frame" options={DESIGN_DEVICES} value={device} onChange={setDevice} />
          </div>
          <MissionHud compact events={events} running={running} snapshot={snapshot} />
          <EventStream
            compact
            empty={<DesignEmpty />}
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
            placeholder="Describe the design artifact, interaction, screen, or visual direction..."
            running={running}
            send={sendDesign}
            snapshot={snapshot}
          />
        </div>

        <aside className="design-studio">
          <div className="design-guide">
            <div>
              <CryptGlyph name="beauty" size={19} />
              <strong>Design pass</strong>
              <span>{surface} / {direction} / {device}</span>
            </div>
            <small>Generated pages and local apps preview here.</small>
          </div>

          <PreviewPanel
            artifacts={artifacts}
            autoStart={launchSignal}
            events={events}
            onDismissArtifact={(artifactId) => {
              setDismissedArtifactIds((ids) => (ids.includes(artifactId) ? ids : [...ids, artifactId]));
            }}
            workspace={workspace}
          />
        </aside>
      </section>
    </main>
  );
}

function DesignEmpty() {
  return (
    <div className="design-empty">
      <CryptGlyph name="beauty" size={30} />
      <h2>Ready for a design pass</h2>
      <p>Describe the screen, component, or reference you want to work from.</p>
    </div>
  );
}

function BriefSelect({ icon: Icon, label, onChange, options, value }) {
  return (
    <label className="design-brief-control">
      <span><Icon size={14} /> {label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
    </label>
  );
}
