import { useEffect, useState } from "react";
import {
  BrainCircuit,
  ChevronDown,
  Eraser,
  FolderOpen,
  LoaderCircle,
  Save,
  SendHorizontal,
  Shield
} from "lucide-react";

const LANES = [
  { id: "planner", label: "Plan" },
  { id: "builder", label: "Code" }
];

const APPROVAL = [
  { id: "normal", label: "Manual" },
  { id: "edits", label: "Auto-work" },
  { id: "all", label: "Bypass" }
];

const THINKING = [
  { id: "fast", label: "Fast" },
  { id: "think", label: "Think" },
  { id: "ultra", label: "Ultra" }
];

export function Composer({
  activeRoute,
  lane,
  onClearView,
  onLaneChange,
  onUserMessage,
  placeholder = "Ask Crypt to code, inspect, debug, or plan...",
  running,
  send,
  snapshot
}) {
  const [prompt, setPrompt] = useState("");
  const [workspace, setWorkspace] = useState(snapshot?.workspace || "");

  useEffect(() => {
    setWorkspace(snapshot?.workspace || "");
  }, [snapshot?.workspace]);

  const submit = () => {
    const text = prompt.trim();
    if (!text || running) return;
    setPrompt("");
    onUserMessage(text);
    send({ type: "sendPrompt", text, route: lane });
  };

  return (
    <footer className="composer-bar">
      <div className="composer-tools">
        <WorkspaceDock workspace={workspace} setWorkspace={setWorkspace} send={send} />
        <Segmented value={lane} options={LANES} onChange={onLaneChange} />
        <ApprovalToggle value={snapshot?.approvalMode || "edits"} onChange={(mode) => send({ type: "setApproval", mode })} />
        <div className="thinking-mode">
          <BrainCircuit size={14} />
          <Segmented value={snapshot?.thinkingMode || "fast"} options={THINKING} onChange={(mode) => send({ type: "setThinking", mode })} />
        </div>
        <button className="text-button" type="button" onClick={onClearView}>
          <Eraser size={14} />
          Clear
        </button>
      </div>

      <div className="composer-input">
        <textarea
          value={prompt}
          disabled={running}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
          placeholder={placeholder}
          rows={1}
        />
        <button className="send-button" type="button" disabled={!prompt.trim() || running} onClick={submit}>
          {running ? <LoaderCircle className="spin" size={18} /> : <SendHorizontal size={18} />}
        </button>
      </div>
      {running ? <div className="run-status">Running on {activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "active model"}...</div> : null}
    </footer>
  );
}

function WorkspaceDock({ workspace, setWorkspace, send }) {
  return (
    <div className="workspace-dock">
      <FolderOpen size={14} />
      <input
        aria-label="Workspace path"
        value={workspace}
        onChange={(event) => setWorkspace(event.target.value)}
        spellCheck={false}
      />
      <button
        className="dock-icon"
        type="button"
        title="Browse workspace"
        onClick={async () => {
          const path = await window.crypt?.chooseDirectory?.();
          if (path) {
            setWorkspace(path);
            send({ type: "setWorkspace", path });
          }
        }}
      >
        <FolderOpen size={13} />
      </button>
      <button className="dock-icon" type="button" title="Save workspace" onClick={() => send({ type: "setWorkspace", path: workspace })}>
        <Save size={13} />
      </button>
    </div>
  );
}

function Segmented({ value, options, onChange }) {
  return (
    <div className="segmented-control">
      {options.map((option) => (
        <button key={option.id} type="button" className={value === option.id ? "active" : ""} onClick={() => onChange(option.id)}>
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ApprovalToggle({ value, onChange }) {
  return (
    <label className="select-pill">
      <Shield size={14} />
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {APPROVAL.map((item) => (
          <option key={item.id} value={item.id}>{item.label}</option>
        ))}
      </select>
      <ChevronDown size={14} />
    </label>
  );
}
