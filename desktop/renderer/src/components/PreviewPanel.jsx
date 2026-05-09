import { useMemo, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Camera,
  ExternalLink,
  FileCode2,
  Globe2,
  Monitor,
  RefreshCcw
} from "lucide-react";

export function PreviewPanel() {
  const [draftUrl, setDraftUrl] = useState("http://localhost:5173");
  const [history, setHistory] = useState([]);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [iframeKey, setIframeKey] = useState(0);
  const [screenshot, setScreenshot] = useState("");
  const currentUrl = historyIndex >= 0 ? history[historyIndex] : "";
  const canGoBack = historyIndex > 0;
  const canGoForward = historyIndex >= 0 && historyIndex < history.length - 1;
  const displayUrl = useMemo(() => currentUrl.replace(/^file:\/\/\//, ""), [currentUrl]);

  const navigate = (raw = draftUrl) => {
    const next = normalizePreviewUrl(raw);
    if (!next) return;
    const nextHistory = history.slice(0, historyIndex + 1);
    nextHistory.push(next);
    setHistory(nextHistory);
    setHistoryIndex(nextHistory.length - 1);
    setDraftUrl(next);
    setIframeKey((value) => value + 1);
  };

  const pickFile = async () => {
    const filePath = await window.crypt?.choosePreviewFile?.();
    if (filePath) navigate(filePath);
  };

  const capture = async () => {
    const image = await window.crypt?.captureScreenshot?.();
    if (image) setScreenshot(image);
  };

  return (
    <aside className="preview-panel">
      <header className="preview-header">
        <div>
          <span><Monitor size={15} /> Preview</span>
          <strong>{displayUrl || "No target loaded"}</strong>
        </div>
        <button className="icon-button" type="button" title="Open externally" disabled={!currentUrl} onClick={() => window.crypt?.openExternal?.(currentUrl)}>
          <ExternalLink size={15} />
        </button>
      </header>

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
        <label className="preview-address">
          <Globe2 size={15} />
          <input
            value={draftUrl}
            onChange={(event) => setDraftUrl(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") navigate();
            }}
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

      <div className="preview-stage">
        {currentUrl ? (
          <iframe key={`${iframeKey}-${currentUrl}`} src={currentUrl} title="Crypt preview" />
        ) : (
          <div className="preview-empty">
            <Monitor size={28} />
            <strong>Open a localhost app or HTML file.</strong>
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

function normalizePreviewUrl(raw) {
  const value = String(raw || "").trim();
  if (!value) return "";
  if (/^[a-zA-Z]:[\\/]/.test(value)) {
    return `file:///${value.replace(/\\/g, "/")}`;
  }
  if (value.startsWith("/") || value.startsWith("~")) {
    return `file://${value}`;
  }
  if (/^https?:\/\//i.test(value) || /^file:\/\//i.test(value)) {
    return value;
  }
  if (/^(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(value)) {
    return `http://${value}`;
  }
  return `https://${value}`;
}
