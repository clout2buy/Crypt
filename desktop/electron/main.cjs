const { app, BrowserWindow, dialog, ipcMain, protocol, session, shell } = require("electron");
const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");
const { fileURLToPath } = require("node:url");
const { createPreviewManager } = require("./preview.cjs");

const isDev = !app.isPackaged;
const forceBuilt = process.env.CRYPT_ELECTRON_BUILT === "1";
const useDevServer = isDev && !forceBuilt && process.argv.includes("--dev");
const desktopRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(desktopRoot, "..");

let mainWindow = null;
let daemon = null;
let daemonBuffer = "";
let previewManager = null;
const previewRoots = new Set();

app.commandLine.appendSwitch("enable-features", "WebSpeechAPI");

protocol.registerSchemesAsPrivileged([
  {
    scheme: "crypt-preview",
    privileges: {
      standard: true,
      secure: true,
      supportFetchAPI: true,
      corsEnabled: true,
      stream: true
    }
  }
]);

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 980,
    minHeight: 760,
    backgroundColor: "#1f1f1e",
    title: "Crypt",
    icon: appIconPath(),
    titleBarStyle: "hidden",
    titleBarOverlay: {
      color: "#1f1f1e",
      symbolColor: "#ededf0",
      height: 40
    },
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    }
  });

  mainWindow.setMenuBarVisibility(false);

  if (useDevServer) {
    mainWindow.loadURL("http://127.0.0.1:5173");
  } else {
    mainWindow.loadFile(path.join(desktopRoot, "renderer", "dist", "index.html"));
  }

  mainWindow.webContents.on("render-process-gone", (_event, details) => {
    forwardEvent({
      event: "rendererExit",
      reason: details.reason,
      exitCode: details.exitCode
    });
  });

  if (process.env.CRYPT_ELECTRON_SCREENSHOT) {
    mainWindow.webContents.once("did-finish-load", () => {
      setTimeout(async () => {
        const image = await mainWindow.webContents.capturePage();
        fs.writeFileSync(process.env.CRYPT_ELECTRON_SCREENSHOT, image.toPNG());
        app.quit();
      }, Number(process.env.CRYPT_ELECTRON_SCREENSHOT_DELAY_MS || 1800));
    });
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

function startDaemon() {
  if (daemon) return;
  const python = process.env.CRYPT_PYTHON || "python";
  const cwd = backendCwd();
  daemonBuffer = "";
  try {
    daemon = spawn(python, ["-m", "crypt", "app-daemon"], {
      cwd,
      env: {
        ...process.env,
        CRYPT_FORCE_TERMINAL: "0",
        PYTHONIOENCODING: "utf-8",
        PYTHONUNBUFFERED: "1"
      },
      stdio: ["pipe", "pipe", "pipe"]
    });
  } catch (error) {
    daemon = null;
    forwardDaemonError(error, cwd);
    return;
  }
  const proc = daemon;

  proc.stdout.setEncoding("utf8");
  proc.stdout.on("data", (chunk) => {
    daemonBuffer += chunk;
    let idx = daemonBuffer.indexOf("\n");
    while (idx >= 0) {
      const line = daemonBuffer.slice(0, idx).trim();
      daemonBuffer = daemonBuffer.slice(idx + 1);
      if (line) forwardDaemonLine(line);
      idx = daemonBuffer.indexOf("\n");
    }
  });

  proc.stderr.setEncoding("utf8");
  proc.stderr.on("data", (chunk) => {
    forwardEvent({
      event: "daemonLog",
      stream: "stderr",
      text: String(chunk)
    });
  });

  proc.on("exit", (code, signal) => {
    forwardEvent({
      event: "daemonExit",
      code,
      signal
    });
    if (daemon === proc) {
      daemon = null;
    }
  });

  proc.on("error", (error) => {
    forwardDaemonError(error, cwd);
    if (daemon === proc) {
      daemon = null;
    }
  });
}

function stopDaemon() {
  if (!daemon) return;
  daemon.kill();
  daemon = null;
}

function restartDaemon() {
  forwardEvent({ event: "daemonRestarting" });
  stopDaemon();
  startDaemon();
  return true;
}

function forwardDaemonLine(line) {
  try {
    forwardEvent(JSON.parse(line));
  } catch {
    forwardEvent({
      event: "daemonLog",
      stream: "stdout",
      text: line
    });
  }
}

function forwardEvent(event) {
  if (event?.snapshot?.workspace) {
    rememberPreviewRoot(event.snapshot.workspace);
  }
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("crypt:event", event);
  }
}

function forwardPreviewEvent(event) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("crypt:preview", event);
  }
}

function sendCommand(command) {
  const body = {
    id: command.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    ...command
  };
  if (body.type === "setWorkspace") {
    rememberPreviewRoot(body.path);
  }
  if (!daemon || !daemon.stdin.writable) {
    startDaemon();
  }
  if (!daemon || !daemon.stdin.writable) {
    forwardEvent({
      event: "daemonError",
      text: "Crypt backend is not running. Set CRYPT_PYTHON or install the Python package, then restart the engine.",
      cwd: backendCwd()
    });
    return body.id;
  }
  daemon.stdin.write(`${JSON.stringify(body)}\n`);
  return body.id;
}

function backendCwd() {
  if (process.env.CRYPT_BACKEND_ROOT) {
    return process.env.CRYPT_BACKEND_ROOT;
  }
  if (!app.isPackaged) {
    return projectRoot;
  }
  return path.join(process.resourcesPath, "backend");
}

function appIconPath() {
  const ico = path.join(desktopRoot, "electron", "assets", "crypt.ico");
  if (fs.existsSync(ico)) return ico;
  return path.join(desktopRoot, "electron", "assets", "crypt-logo.svg");
}

function forwardDaemonError(error, cwd) {
  forwardEvent({
    event: "daemonError",
    text: `${error.name || "Error"}: ${error.message || String(error)}`,
    cwd
  });
}

async function chooseDirectory() {
  if (!mainWindow || mainWindow.isDestroyed()) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select Crypt workspace",
    properties: ["openDirectory", "createDirectory"]
  });
  if (result.canceled || !result.filePaths.length) return null;
  rememberPreviewRoot(result.filePaths[0]);
  return result.filePaths[0];
}

async function choosePreviewFile() {
  if (!mainWindow || mainWindow.isDestroyed()) return null;
  const result = await dialog.showOpenDialog(mainWindow, {
    title: "Select HTML preview",
    properties: ["openFile"],
    filters: [
      { name: "Web files", extensions: ["html", "htm", "svg"] },
      { name: "All files", extensions: ["*"] }
    ]
  });
  if (result.canceled || !result.filePaths.length) return null;
  rememberPreviewRoot(path.dirname(result.filePaths[0]));
  return result.filePaths[0];
}

async function captureScreenshot(bounds) {
  if (!mainWindow || mainWindow.isDestroyed()) return null;
  const rect = screenshotBounds(bounds);
  const image = await mainWindow.webContents.capturePage(rect || undefined);
  return image.toDataURL();
}

function registerPreviewProtocol() {
  protocol.handle("crypt-preview", async (request) => {
    const filePath = localPathFromPreviewUrl(request.url);
    if (!filePath) {
      return new Response("Invalid preview URL.", {
        status: 400,
        headers: { "content-type": "text/plain; charset=utf-8" }
      });
    }

    try {
      const resolvedPath = canonicalPath(filePath);
      if (!isAllowedPreviewPath(resolvedPath)) {
        return new Response("Preview file is outside the active workspace.", {
          status: 403,
          headers: { "content-type": "text/plain; charset=utf-8" }
        });
      }
      const stat = await fs.promises.stat(resolvedPath);
      if (!stat.isFile()) {
        return new Response("Preview target is not a file.", {
          status: 404,
          headers: { "content-type": "text/plain; charset=utf-8" }
        });
      }
      const data = await fs.promises.readFile(resolvedPath);
      return new Response(data, {
        headers: {
          "cache-control": "no-store",
          "content-type": mimeType(resolvedPath)
        }
      });
    } catch (error) {
      return new Response(errorText(error), {
        status: 404,
        headers: { "content-type": "text/plain; charset=utf-8" }
      });
    }
  });
}

async function openExternalTarget(rawUrl) {
  const value = String(rawUrl || "").trim();
  if (!value) return false;

  const previewPath = localPathFromPreviewUrl(value);
  if (previewPath) {
    await shell.openPath(previewPath);
    return true;
  }

  if (/^[a-zA-Z]:[\\/]/.test(value) || value.startsWith("\\\\")) {
    await shell.openPath(value);
    return true;
  }

  if (/^file:\/\//i.test(value)) {
    await shell.openPath(fileURLToPath(value));
    return true;
  }

  if (/^https?:\/\//i.test(value)) {
    await shell.openExternal(value);
    return true;
  }

  return false;
}

function localPathFromPreviewUrl(rawUrl) {
  let parsed;
  try {
    parsed = new URL(String(rawUrl || ""));
  } catch {
    return "";
  }
  if (parsed.protocol !== "crypt-preview:" || parsed.hostname !== "local") return "";

  let decoded = decodeURIComponent(parsed.pathname || "");
  if (process.platform === "win32" && /^\/[a-zA-Z]:\//.test(decoded)) {
    decoded = decoded.slice(1);
  }
  return path.resolve(decoded);
}

function rememberPreviewRoot(rawPath) {
  const value = String(rawPath || "").trim();
  if (!value) return;
  try {
    if (!fs.existsSync(value)) return;
    const stat = fs.statSync(value);
    const root = stat.isDirectory() ? value : path.dirname(value);
    previewRoots.add(canonicalPath(root));
  } catch {
    // A bad workspace path is reported by the daemon; preview just ignores it.
  }
}

function isAllowedPreviewPath(filePath) {
  if (!previewRoots.size) return false;
  const resolved = canonicalPath(filePath);
  for (const root of previewRoots) {
    const relative = path.relative(root, resolved);
    if (relative === "" || (relative && !relative.startsWith("..") && !path.isAbsolute(relative))) {
      return true;
    }
  }
  return false;
}

function canonicalPath(rawPath) {
  const resolved = path.resolve(String(rawPath || ""));
  try {
    return fs.realpathSync.native(resolved);
  } catch {
    return resolved;
  }
}

function screenshotBounds(bounds) {
  if (!bounds || typeof bounds !== "object") return null;
  const rect = {
    x: Math.max(0, Math.round(Number(bounds.x) || 0)),
    y: Math.max(0, Math.round(Number(bounds.y) || 0)),
    width: Math.max(1, Math.round(Number(bounds.width) || 0)),
    height: Math.max(1, Math.round(Number(bounds.height) || 0))
  };
  return rect.width > 1 && rect.height > 1 ? rect : null;
}

function mimeType(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const types = {
    ".avif": "image/avif",
    ".css": "text/css; charset=utf-8",
    ".gif": "image/gif",
    ".htm": "text/html; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".ico": "image/x-icon",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".map": "application/json; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".mp4": "video/mp4",
    ".png": "image/png",
    ".svg": "image/svg+xml; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".webm": "video/webm",
    ".webp": "image/webp",
    ".woff": "font/woff",
    ".woff2": "font/woff2"
  };
  return types[ext] || "application/octet-stream";
}

app.whenReady().then(() => {
  registerPermissionHandlers();
  rememberPreviewRoot(projectRoot);
  registerPreviewProtocol();
  createWindow();
  startDaemon();
  previewManager = createPreviewManager({ backendCwd, emit: forwardPreviewEvent });

  ipcMain.handle("crypt:command", (_event, command) => sendCommand(command || {}));
  ipcMain.handle("crypt:openExternal", (_event, url) => openExternalTarget(url));
  ipcMain.handle("crypt:restartDaemon", () => restartDaemon());
  ipcMain.handle("crypt:chooseDirectory", () => chooseDirectory());
  ipcMain.handle("crypt:choosePreviewFile", () => choosePreviewFile());
  ipcMain.handle("crypt:captureScreenshot", (_event, bounds) => captureScreenshot(bounds));
  ipcMain.handle("crypt:detectPreviewServer", (_event, cwd) => {
    rememberPreviewRoot(cwd);
    return previewManager.detect(cwd);
  });
  ipcMain.handle("crypt:startPreviewServer", (_event, cwd) => {
    rememberPreviewRoot(cwd);
    return previewManager.start(cwd);
  });
  ipcMain.handle("crypt:stopPreviewServer", () => previewManager.stop());
  ipcMain.handle("crypt:installPreviewDeps", (_event, cwd) => {
    rememberPreviewRoot(cwd);
    return previewManager.install(cwd);
  });
  ipcMain.handle("crypt:listPreviewArtifacts", (_event, cwd) => {
    rememberPreviewRoot(cwd);
    return previewManager.artifacts(cwd);
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

function registerPermissionHandlers() {
  const allowMedia = (_webContents, permission, callback, details = {}) => {
    if (permission === "media" && shouldAllowMedia(details)) {
      callback(true);
      return;
    }
    callback(false);
  };

  session.defaultSession.setPermissionRequestHandler(allowMedia);
  session.defaultSession.setPermissionCheckHandler((_webContents, permission, requestingOrigin, details = {}) => {
    return permission === "media" && shouldAllowMedia({ ...details, requestingOrigin });
  });
}

function shouldAllowMedia(details = {}) {
  const mediaTypes = details.mediaTypes || [];
  if (mediaTypes.length && !mediaTypes.includes("audio")) return false;
  const origin = String(details.requestingOrigin || details.securityOrigin || details.origin || "");
  return (
    origin.startsWith("file://") ||
    origin.startsWith("http://127.0.0.1:5173") ||
    origin.startsWith("http://localhost:5173")
  );
}

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  previewManager?.dispose();
  if (daemon) {
    stopDaemon();
  }
});
