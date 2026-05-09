const { app, BrowserWindow, dialog, ipcMain, shell } = require("electron");
const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const isDev = !app.isPackaged;
const forceBuilt = process.env.CRYPT_ELECTRON_BUILT === "1";
const useDevServer = isDev && !forceBuilt && process.argv.includes("--dev");
const desktopRoot = path.resolve(__dirname, "..");
const projectRoot = path.resolve(desktopRoot, "..");

let mainWindow = null;
let daemon = null;
let daemonBuffer = "";

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 980,
    minHeight: 760,
    backgroundColor: "#1f1f1e",
    title: "Crypt",
    icon: path.join(desktopRoot, "electron", "assets", "crypt-logo.svg"),
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
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("crypt:event", event);
  }
}

function sendCommand(command) {
  const body = {
    id: command.id || `${Date.now()}-${Math.random().toString(16).slice(2)}`,
    ...command
  };
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
  return app.getPath("home");
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
  return result.filePaths[0];
}

async function captureScreenshot() {
  if (!mainWindow || mainWindow.isDestroyed()) return null;
  const image = await mainWindow.webContents.capturePage();
  return image.toDataURL();
}

app.whenReady().then(() => {
  createWindow();
  startDaemon();

  ipcMain.handle("crypt:command", (_event, command) => sendCommand(command || {}));
  ipcMain.handle("crypt:openExternal", (_event, url) => shell.openExternal(url));
  ipcMain.handle("crypt:restartDaemon", () => restartDaemon());
  ipcMain.handle("crypt:chooseDirectory", () => chooseDirectory());
  ipcMain.handle("crypt:choosePreviewFile", () => choosePreviewFile());
  ipcMain.handle("crypt:captureScreenshot", () => captureScreenshot());

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  if (daemon) {
    stopDaemon();
  }
});
