export function normalizePreviewTarget(raw) {
  const value = String(raw || "").trim();
  const url = normalizePreviewUrl(value);
  if (!url) return null;
  return {
    id: url,
    label: labelFromPreviewUrl(url),
    path: localPathFromPreviewUrl(url) || pathFromFileUrl(value) || value,
    url
  };
}

export function normalizePreviewUrl(raw) {
  const value = String(raw || "").trim();
  if (!value) return "";
  if (/^[a-zA-Z]:[\\/]/.test(value) || value.startsWith("/") || value.startsWith("~")) {
    return previewFileUrl(value);
  }
  if (/^file:\/\//i.test(value)) {
    return previewFileUrl(pathFromFileUrl(value));
  }
  if (/^https?:\/\//i.test(value) || /^crypt-preview:\/\//i.test(value)) {
    return value;
  }
  if (/^(localhost|127\.0\.0\.1|\[::1\])(:|\/|$)/i.test(value)) {
    return `http://${value}`;
  }
  return `https://${value}`;
}

export function labelFromPreviewUrl(url) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "file:" || parsed.protocol === "crypt-preview:") {
      const parts = decodeURIComponent(parsed.pathname).split("/");
      return parts[parts.length - 1] || "artifact";
    }
    return parsed.host;
  } catch {
    return url;
  }
}

export function displayPreviewTarget(url, path) {
  if (path && !/^https?:\/\//i.test(path) && !/^crypt-preview:\/\//i.test(path)) {
    return path;
  }
  const localPath = localPathFromPreviewUrl(url);
  if (localPath) return localPath;
  return String(url || "").replace(/^file:\/\/\//, "");
}

export function previewFileUrl(filePath) {
  let normalized = String(filePath || "").trim().replace(/\\/g, "/");
  if (!normalized) return "";
  if (/^\/[a-zA-Z]:\//.test(normalized)) normalized = normalized.slice(1);
  if (normalized.startsWith("~")) normalized = normalized.slice(1);
  const pathname = normalized.startsWith("/") ? normalized : `/${normalized}`;
  return `crypt-preview://local${encodeURI(pathname).replace(/#/g, "%23").replace(/\?/g, "%3F")}`;
}

export function pathFromFileUrl(raw) {
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== "file:") return "";
    let decoded = decodeURIComponent(parsed.pathname || "");
    if (/^\/[a-zA-Z]:\//.test(decoded)) decoded = decoded.slice(1);
    return /^[a-zA-Z]:\//.test(decoded) ? decoded.replace(/\//g, "\\") : decoded;
  } catch {
    return "";
  }
}

export function localPathFromPreviewUrl(raw) {
  try {
    const parsed = new URL(raw);
    if (parsed.protocol !== "crypt-preview:" || parsed.hostname !== "local") return "";
    let decoded = decodeURIComponent(parsed.pathname || "");
    if (/^\/[a-zA-Z]:\//.test(decoded)) decoded = decoded.slice(1);
    return /^[a-zA-Z]:\//.test(decoded) ? decoded.replace(/\//g, "\\") : decoded;
  } catch {
    return "";
  }
}
