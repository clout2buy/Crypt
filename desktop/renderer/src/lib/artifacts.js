const PREVIEWABLE_EXTENSIONS = new Set([".html", ".htm", ".svg"]);

export function previewArtifactsFromEvents(events, workspace) {
  const artifacts = [];
  const root = String(workspace || "").trim();

  for (const event of events) {
    const filePath = artifactPathFromEvent(event);
    if (!filePath || !isPreviewable(filePath)) continue;
    artifacts.push({
      id: `${event.id}-${filePath}`,
      label: basename(filePath),
      path: absolutePath(filePath, root),
      relativePath: filePath,
      source: event.tool || "artifact",
      time: event.time || ""
    });
  }

  return dedupeLatest(artifacts).map((artifact) => ({
    ...artifact,
    url: fileUrl(artifact.path)
  }));
}

export function hasWebProjectActivity(events) {
  return events.some((event) => {
    const text = `${event.text || ""} ${pathFromToolArgs(event.args) || ""}`.toLowerCase();
    return (
      text.includes("localhost:") ||
      text.includes("package.json") ||
      text.includes("vite.config") ||
      text.includes("next.config") ||
      text.includes("astro.config") ||
      /\bsrc[\\/].+\.(jsx|tsx|vue|svelte)\b/.test(text)
    );
  });
}

function artifactPathFromEvent(event) {
  if (!event || event.event !== "tool" || event.rawEvent !== "toolResult" || event.ok === false) return "";
  if (!["write_file", "edit_file", "multi_edit", "open_file"].includes(event.tool)) return "";

  const pathFromArgs = pathFromToolArgs(event.args);
  if (pathFromArgs) return pathFromArgs;

  return pathFromText(event.text);
}

function pathFromToolArgs(args) {
  if (!args || typeof args !== "object") return "";
  if (typeof args.path === "string") return args.path;
  if (Array.isArray(args.edits)) {
    const edit = args.edits.find((item) => item && typeof item.path === "string" && isPreviewable(item.path));
    return edit?.path || "";
  }
  if (Array.isArray(args.paths)) {
    return args.paths.find((item) => typeof item === "string" && isPreviewable(item)) || "";
  }
  return "";
}

function pathFromText(text) {
  const value = String(text || "");
  const match = value.match(/\b(?:created|overwrote|edited|opened)\s+(.+?\.(?:html?|svg))\b/i);
  return match ? match[1].trim().replace(/^["'`]|["'`]$/g, "") : "";
}

function isPreviewable(filePath) {
  const clean = stripQuery(filePath).toLowerCase();
  return [...PREVIEWABLE_EXTENSIONS].some((extension) => clean.endsWith(extension));
}

function absolutePath(filePath, workspace) {
  const value = filePath.replace(/\//g, "\\");
  if (/^[a-zA-Z]:\\/.test(value) || value.startsWith("\\\\")) return value;
  if (!workspace) return value;
  return `${workspace.replace(/[\\/]+$/, "")}\\${value.replace(/^[\\/]+/, "")}`;
}

function basename(filePath) {
  const parts = filePath.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || filePath;
}

function fileUrl(filePath) {
  const normalized = filePath.replace(/\\/g, "/");
  if (/^file:\/\//i.test(normalized) || /^https?:\/\//i.test(normalized)) return normalized;
  return `file:///${encodeURI(normalized).replace(/#/g, "%23")}`;
}

function stripQuery(filePath) {
  return String(filePath || "").split(/[?#]/, 1)[0];
}

function dedupeLatest(artifacts) {
  const seen = new Set();
  const result = [];
  for (const artifact of [...artifacts].reverse()) {
    const key = artifact.path.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(artifact);
  }
  return result;
}
