import { useEffect, useRef, useState } from "react";
import {
  BrainCircuit,
  ChevronDown,
  Eraser,
  FolderOpen,
  LoaderCircle,
  Mic,
  Save,
  SendHorizontal,
  Shield,
  Square
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
  const [voiceState, setVoiceState] = useState("idle");
  const [voiceInterim, setVoiceInterim] = useState("");
  const audioStreamRef = useRef(null);
  const manuallyStoppingVoiceRef = useRef(false);
  const recognitionRef = useRef(null);
  const voiceHadResultRef = useRef(false);
  const voiceRetryRef = useRef(0);
  const voiceStartedAtRef = useRef(0);
  const canSend = Boolean(prompt.trim()) && !running;
  const voiceActive = voiceState === "starting" || voiceState === "recording";

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

  useEffect(() => {
    return () => {
      recognitionRef.current?.stop?.();
      releaseAudioStream(audioStreamRef);
    };
  }, []);

  const startVoice = async ({ retry = false } = {}) => {
    if (running || (!retry && voiceActive)) return;
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) {
      setVoiceState("unsupported");
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceState("unsupported");
      return;
    }

    manuallyStoppingVoiceRef.current = false;
    voiceHadResultRef.current = false;
    if (!retry) voiceRetryRef.current = 0;
    setVoiceState("starting");
    setVoiceInterim("");
    releaseAudioStream(audioStreamRef);

    try {
      audioStreamRef.current = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      setVoiceState(error?.name === "NotAllowedError" ? "blocked" : "unavailable");
      return;
    }

    const recognition = new Recognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.maxAlternatives = 1;
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      voiceStartedAtRef.current = Date.now();
      setVoiceState("recording");
    };
    recognition.onerror = (event) => {
      releaseAudioStream(audioStreamRef);
      recognitionRef.current = null;
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        setVoiceState("blocked");
      } else if (event.error === "no-speech") {
        setVoiceState("no-speech");
      } else {
        setVoiceState("unavailable");
      }
      setVoiceInterim("");
    };
    recognition.onend = () => {
      recognitionRef.current = null;
      const elapsed = Date.now() - voiceStartedAtRef.current;
      const stoppedByUser = manuallyStoppingVoiceRef.current;
      const endedTooFast = elapsed > 0 && elapsed < 1200 && !voiceHadResultRef.current;

      if (!stoppedByUser && endedTooFast && voiceRetryRef.current < 1) {
        voiceRetryRef.current += 1;
        window.setTimeout(() => startVoice({ retry: true }), 260);
        return;
      }

      releaseAudioStream(audioStreamRef);
      manuallyStoppingVoiceRef.current = false;
      setVoiceState((state) => {
        if (stoppedByUser) return "idle";
        if (state === "recording" || state === "starting") return voiceHadResultRef.current ? "idle" : "no-speech";
        return state;
      });
      setVoiceInterim("");
    };
    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";
      for (let index = event.resultIndex; index < event.results.length; index += 1) {
        const result = event.results[index];
        const text = result[0]?.transcript || "";
        if (result.isFinal) {
          finalText += text;
        } else {
          interimText += text;
        }
      }
      if (finalText.trim()) {
        voiceHadResultRef.current = true;
        setPrompt((current) => appendDictation(current, finalText));
      }
      if (interimText.trim()) voiceHadResultRef.current = true;
      setVoiceInterim(interimText.trim());
    };

    try {
      recognition.start();
    } catch {
      releaseAudioStream(audioStreamRef);
      recognitionRef.current = null;
      setVoiceState("unavailable");
    }
  };

  const stopVoice = () => {
    manuallyStoppingVoiceRef.current = true;
    try {
      recognitionRef.current?.stop?.();
    } catch {
      // The recognizer may already have ended.
    }
    recognitionRef.current = null;
    releaseAudioStream(audioStreamRef);
    setVoiceState("idle");
    setVoiceInterim("");
  };

  return (
    <footer className={running ? "composer-bar running" : "composer-bar"}>
      <div className="composer-tools">
        <WorkspaceDock workspace={workspace} setWorkspace={setWorkspace} send={send} />
        <Segmented value={lane} options={LANES} onChange={onLaneChange} />
        <ApprovalToggle value={snapshot?.approvalMode || "edits"} onChange={(mode) => send({ type: "setApproval", mode })} />
        <ThinkingControl value={snapshot?.thinkingMode || "fast"} onChange={(mode) => send({ type: "setThinking", mode })} />
        <button className="text-button" type="button" onClick={onClearView}>
          <Eraser size={14} />
          Clear
        </button>
      </div>

      <div className="composer-input">
        <span className="composer-glow" aria-hidden="true" />
        <button
          className={voiceActive ? "voice-button recording" : "voice-button"}
          disabled={running}
          title={voiceTitle(voiceState)}
          type="button"
          onClick={voiceActive ? stopVoice : startVoice}
        >
          {voiceActive ? <Square size={16} /> : <Mic size={17} />}
        </button>
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
        <button className={canSend ? "send-button ready" : "send-button"} type="button" disabled={!canSend} onClick={submit}>
          {running ? <LoaderCircle className="spin" size={18} /> : <SendHorizontal size={18} />}
        </button>
        {voiceInterim ? <div className="voice-interim">Listening: {voiceInterim}</div> : null}
      </div>
      {voiceState === "unsupported" ? <div className="voice-note">Voice input is not available in this Electron build.</div> : null}
      {voiceState === "blocked" ? <div className="voice-note">Microphone permission was blocked.</div> : null}
      {voiceState === "unavailable" ? <div className="voice-note">Voice input could not start. Check Windows microphone access, then try again.</div> : null}
      {voiceState === "no-speech" ? <div className="voice-note">I did not catch audio. Tap the mic and speak again, then hit the square to stop.</div> : null}
      {running ? <div className="run-status">Running on {activeRoute ? `${activeRoute.provider} / ${activeRoute.model}` : "active model"}...</div> : null}
    </footer>
  );
}

function appendDictation(current, spoken) {
  const text = String(spoken || "").trim();
  if (!text) return current;
  const prefix = current.trim() ? `${current.trimEnd()} ` : "";
  return `${prefix}${text}`;
}

function voiceTitle(state) {
  if (state === "starting") return "Starting voice input";
  if (state === "recording") return "Stop voice input";
  if (state === "unsupported") return "Voice input unavailable";
  if (state === "blocked") return "Microphone permission blocked";
  if (state === "unavailable") return "Voice input could not start";
  return "Start voice input";
}

function releaseAudioStream(streamRef) {
  const stream = streamRef.current;
  streamRef.current = null;
  for (const track of stream?.getTracks?.() || []) {
    track.stop();
  }
}

function ThinkingControl({ onChange, value }) {
  const enabled = value !== "fast";
  const depth = value === "ultra" ? "2" : "1";

  return (
    <div className={enabled ? "thinking-control active" : "thinking-control"}>
      <button type="button" onClick={() => onChange(enabled ? "fast" : "think")}>
        <BrainCircuit size={14} />
        Think
        <span>{enabled ? "On" : "Off"}</span>
      </button>
      <label>
        Depth
        <input
          aria-label="Thinking depth"
          disabled={!enabled}
          max="2"
          min="1"
          step="1"
          type="range"
          value={depth}
          onChange={(event) => onChange(event.target.value === "2" ? "ultra" : "think")}
        />
        <strong>{value === "ultra" ? "Deep" : "Normal"}</strong>
      </label>
    </div>
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
