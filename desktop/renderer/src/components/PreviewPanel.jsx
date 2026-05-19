import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  ExternalLink,
  FileCode2,
  Globe2,
  Monitor,
  PackageCheck,
  Play,
  RefreshCcw,
  Server,
  Smartphone,
  Square,
  Tablet,
  X
} from "lucide-react";
import { previewArtifactsFromEvents } from "../lib/artifacts.js";
import {
  displayPreviewTarget,
  normalizePreviewTarget
} from "../lib/previewTargets.js";

export function PreviewPanel({
  artifacts: providedArtifacts,
  autoStart = false,
  events = [],
  onClose,
  onDismissArtifact,
  workspace = ""
}) {
  const detectedArtifacts = useMemo(() => previewArtifactsFromEvents(events, workspace), [events, workspace]);
  const artifacts = providedArtifacts || detectedArtifacts;
  const latestArtifact = artifacts[0] || null;
  const [draftUrl, setDraftUrl] = useState("");
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [iframeKey, setIframeKey] = useState(0);
  const [screenshot, setScreenshot] = useState("");
  const [viewport, setViewport] = useState("desktop");
  const [autoLoadedId, setAutoLoadedId] = useState("");
  const [autoStartedCwd, setAutoStartedCwd] = useState("");
  const [serverState, setServerState] = useState(null);
  const stageRef = useRef(null);
  const currentTarget = historyIndex >= 0 ? history[historyIndex] : null;
  const currentUrl = historyIndex >= 0 ? history[historyIndex]?.url || "" : "";
  const currentLabel = historyIndex >= 0 ? history[historyIndex]?.label || "" : "";
  const canGoBack = historyIndex > 0;
  const canGoForward = historyIndex >= 0 && historyIndex < history.length - 1;
  const displayUrl = useMemo(() => displayPreviewTarget(currentUrl, currentTarget?.path), [currentTarget?.path, currentUrl]);

  const clearPreview = () => {
    setDraftUrl("");
    setHistory([]);
    setHistoryIndex(-1);
    setIframeKey((value) => value + 1);
    setScreenshot("");
  };

  useEffect(() => {
    if (!latestArtifact || latestArtifact.id === autoLoadedId) return;
    navigateToTarget(latestArtifact, { auto: true });
  }, [latestArtifact?.id]);

  useEffect(() => {
    let alive = true;
    window.crypt?.detectPreviewServer?.(workspace).then((state) => {
      if (alive) setServerState(state);
    });
    return () => {
      alive = false;
    };
  }, [workspace]);

  useEffect(() => {
    return window.crypt?.onPreviewEvent?.((payload) => {
      if (payload?.event === "previewState") setServerState(payload.state);
    });
  }, []);

  useEffect(() => {
    if (!autoStart || !serverState?.available || serverState.running || serverState.installing) return;
    if (serverState.needsInstall || autoStartedCwd === serverState.cwd) return;
    setAutoStartedCwd(serverState.cwd);
    window.crypt?.startPreviewServer?.(serverState.cwd);
  }, [
    autoStart,
    autoStartedCwd,
    serverState?.available,
    serverState?.cwd,
    serverState?.installing,
    serverState?.needsInstall,
    serverState?.running
  ]);

  useEffect(() => {
    if (!serverState?.url || serverState.url === currentUrl) return;
    navigateToTarget({
      id: `server-${serverState.url}`,
      label: serverState.url.replace(/^https?:\/\//, ""),
      path: serverState.url,
      url: serverState.url
    });
  }, [serverState?.url]);

  const navigate = (raw = draftUrl) => {
    const next = normalizePreviewTarget(raw);
    if (!next) return;
    navigateToTarget(next);
  };

  const navigateToTarget = (target, { auto = false } = {}) => {
    const next = typeof target === "string" ? normalizePreviewTarget(target) : target;
    if (!next?.url) return;
    if (auto) setAutoLoadedId(next.id || next.url);
    const nextHistory = history.slice(0, historyIndex + 1);
    if (nextHistory[nextHistory.length - 1]?.url !== next.url) {
      nextHistory.push(next);
    }
    setHistory(nextHistory);
    setHistoryIndex(nextHistory.length - 1);
    setDraftUrl(next.url);
    setIframeKey((value) => value + 1);
  };

  const pickFile = async () => {
    const filePath = await window.crypt?.choosePreviewFile?.();
    if (filePath) navigate(filePath);
  };

  const capture = async () => {
    const bounds = stageRef.current?.getBoundingClientRect();
    const image = await window.crypt?.captureScreenshot?.(bounds ? {
      x: Math.round(bounds.x),
      y: Math.round(bounds.y),
      width: Math.round(bounds.width),
      height: Math.round(bounds.height)
    } : null);
    if (image) setScreenshot(image);
  };

  const dismissArtifact = (artifact) => {
    const dismissingCurrent =
      artifact?.id === currentTarget?.id ||
      artifact?.url === currentTarget?.url ||
      artifact?.path === currentTarget?.path;
    onDismissArtifact?.(artifact.id);
    if (dismissingCurrent) clearPreview();
  };

  return (
    <aside className={currentUrl ? "preview-panel active-preview" : "preview-panel"}>
      <header className="preview-header">
        <div>
          <div className="preview-title-row">
            <span className="preview-leds" aria-hidden="true"><i /><i /><i /></span>
            <span><Monitor size={15} /> Preview</span>
          </div>
          <strong>{currentLabel || displayUrl || "Waiting for a web artifact"}</strong>
        </div>
        <div className="preview-header-actions">
          <button className="icon-button" type="button" title="Open externally" disabled={!currentUrl} onClick={() => window.crypt?.openExternal?.(currentTarget?.path || currentUrl)}>
            <ExternalLink size={15} />
          </button>
          {onClose ? (
            <button className="icon-button" type="button" title="Close preview" onClick={onClose}>
              <X size={16} />
            </button>
          ) : null}
        </div>
      </header>

      {artifacts.length ? (
        <div className="artifact-strip">
          {artifacts.slice(0, 4).map((artifact) => (
            <div
              key={artifact.id}
              className={artifact.url === currentUrl ? "artifact-chip active" : "artifact-chip"}
              title={artifact.path}
            >
              <button className="artifact-chip-main" type="button" onClick={() => navigateToTarget(artifact)}>
                <FileCode2 size={14} />
                <span>{artifact.label}</span>
              </button>
              <button className="artifact-dismiss" type="button" title={`Dismiss ${artifact.label}`} onClick={() => dismissArtifact(artifact)}>
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      ) : null}

      <ServerConsole
        hasCurrentUrl={Boolean(currentUrl)}
        serverState={serverState}
        workspace={workspace}
        onInstall={() => window.crypt?.installPreviewDeps?.(workspace)}
        onStart={() => window.crypt?.startPreviewServer?.(workspace)}
        onStop={() => window.crypt?.stopPreviewServer?.()}
      />

      <div className="preview-controls">
        <button className="icon-button" type="button" disabled={!canGoBack} title="Back" onClick={() => {
          setHistoryIndex((value) => value - 1);
          setIframeKey((value) => value + 1);
        }}>
          <ArrowLeft size={15} />
        </button>
        <button className="icon-button" type="button" disabled={!canGoForward} title="Forward" onClick={() => {
          setHistoryIndex((value) => value + 1);
          setIframeKey((value) => value + 1);
        }}>
          <ArrowRight size={15} />
        </button>
        <button className="icon-button" type="button" disabled={!currentUrl} title="Reload" onClick={() => setIframeKey((value) => value + 1)}>
          <RefreshCcw size={15} />
        </button>
        <div className="viewport-switch" aria-label="Preview frame">
          <button className={viewport === "desktop" ? "active" : ""} type="button" title="Desktop frame" onClick={() => setViewport("desktop")}>
            <Monitor size={14} />
          </button>
          <button className={viewport === "tablet" ? "active" : ""} type="button" title="Tablet frame" onClick={() => setViewport("tablet")}>
            <Tablet size={14} />
          </button>
          <button className={viewport === "phone" ? "active" : ""} type="button" title="Phone frame" onClick={() => setViewport("phone")}>
            <Smartphone size={14} />
          </button>
        </div>
        <label className="preview-address">
          <Globe2 size={15} />
          <input
            value={draftUrl}
            onChange={(event) => setDraftUrl(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") navigate();
            }}
            placeholder={latestArtifact ? "Preview target" : "localhost:3000 or generated HTML"}
            spellCheck={false}
          />
        </label>
        <button className="secondary-button" type="button" onClick={pickFile}>
          <FileCode2 size={15} />
          HTML
        </button>
        <button className="secondary-button" type="button" disabled={!currentUrl} onClick={capture}>
          <Camera size={15} />
          Shot
        </button>
      </div>

      <div className={`preview-stage frame-${viewport}`} ref={stageRef}>
        {currentUrl ? (
          <iframe key={`${iframeKey}-${currentUrl}`} src={currentUrl} title="Crypt preview" />
        ) : (
          <div className="preview-empty">
            <Monitor size={28} />
            <strong>No preview yet.</strong>
            <span>Generated HTML/SVG files and launched localhost apps open here automatically.</span>
          </div>
        )}
      </div>

      {screenshot ? (
        <div className="screenshot-strip">
          <span>Latest screenshot</span>
          <img src={screenshot} alt="Latest preview screenshot" />
        </div>
      ) : null}
    </aside>
  );
}

function ServerConsole({ hasCurrentUrl, onInstall, onStart, onStop, serverState, workspace }) {
  if (!serverState?.available) {
    return (
      <div className="server-console quiet">
        <div className="server-status">
          <Server size={15} />
          <span>{hasCurrentUrl ? "Static artifact preview" : workspace ? "No launchable app server detected" : "Workspace not ready"}</span>
        </div>
      </div>
    );
  }

  const logs = (serverState.logs || []).slice(-8).join("\n");
  const status = serverState.installing
    ? "Installing dependencies"
    : serverState.running
      ? serverState.url || "Starting localhost"
      : serverState.needsInstall
        ? "Dependencies needed"
        : "Ready to launch";

  return (
    <div className={serverState.running ? "server-console running" : "server-console"}>
      <div className="server-status">
        <Server size={15} />
        <div>
          <strong>{status}</strong>
          <span>{serverState.command} / {serverState.packageName}</span>
        </div>
      </div>
      <div className="server-actions">
        {serverState.needsInstall ? (
          <button className="secondary-button" type="button" disabled={serverState.installing} onClick={onInstall}>
            <PackageCheck size={14} />
            Install deps
          </button>
        ) : null}
        {serverState.running ? (
          <button className="secondary-button" type="button" onClick={onStop}>
            <Square size={14} />
            Stop
          </button>
        ) : (
          <button className="secondary-button" type="button" disabled={serverState.installing} onClick={onStart}>
            <Play size={14} />
            Start
          </button>
        )}
      </div>
      {logs ? <pre className="server-log">{logs}</pre> : null}
    </div>
  );
}
