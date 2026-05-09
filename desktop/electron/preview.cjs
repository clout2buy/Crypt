const fs = require("node:fs");
const path = require("node:path");
const { spawn } = require("node:child_process");

const ARTIFACT_SKIP_DIRS = new Set([
  ".git",
  ".venv",
  "__pycache__",
  "build",
  "coverage",
  "desktop",
  "dist",
  "docs",
  "node_modules",
  "venv"
]);

function createPreviewManager({ backendCwd, emit }) {
  let previewProcess = null;
  let previewInstallProcess = null;
  let previewState = emptyState();

  function detect(cwdArg) {
    const detection = detectPreviewServer(cwdArg || backendCwd());
    if (previewProcess && previewState.cwd === detection.cwd) {
      previewState = {
        ...previewState,
        available: Boolean(detection.available),
        command: detection.command || previewState.command,
        needsInstall: Boolean(detection.needsInstall),
        packageName: detection.packageName || previewState.packageName,
        reason: detection.reason || "",
        script: detection.script || previewState.script
      };
    } else {
      resetState(detection);
    }
    emitState();
    return status();
  }

  function start(cwdArg) {
    const detection = detectPreviewServer(cwdArg || backendCwd());
    if (!detection.available) {
      resetState(detection);
      emitState();
      return status();
    }

    if (previewProcess && previewState.running && previewState.cwd === detection.cwd) {
      emitState();
      return status();
    }

    stop();
    resetState(detection);
    previewState.running = true;
    previewState.url = inferPreviewUrl(detection.scriptBody);
    emitState();

    const proc = spawn(npmCommand(), ["run", detection.script], {
      cwd: detection.cwd,
      env: {
        ...process.env,
        BROWSER: "none",
        FORCE_COLOR: "0",
        HOST: process.env.HOST || "127.0.0.1"
      },
      stdio: ["ignore", "pipe", "pipe"]
    });
    previewProcess = proc;

    pipeProcess(proc, {
      onData: emitState,
      onError(error) {
        previewState.error = errorText(error);
        previewState.running = false;
        if (previewProcess === proc) previewProcess = null;
        emitState();
      },
      onExit(code, signal) {
        appendLog(`preview server exited (${signal || (code ?? "done")})`);
        previewState.running = false;
        if (previewProcess === proc) previewProcess = null;
        emitState();
      }
    });

    return status();
  }

  function stop() {
    if (previewProcess) {
      previewProcess.kill();
      previewProcess = null;
    }
    previewState.running = false;
    emitState();
    return status();
  }

  function install(cwdArg) {
    const detection = detectPreviewServer(cwdArg || backendCwd());
    if (!detection.available) {
      resetState(detection);
      emitState();
      return status();
    }
    if (previewInstallProcess) return status();

    resetState(detection);
    previewState.installing = true;
    previewState.command = "npm install";
    emitState();

    const proc = spawn(npmCommand(), ["install"], {
      cwd: detection.cwd,
      env: { ...process.env, FORCE_COLOR: "0" },
      stdio: ["ignore", "pipe", "pipe"]
    });
    previewInstallProcess = proc;

    pipeProcess(proc, {
      onData: emitState,
      onError(error) {
        previewState.error = errorText(error);
        previewState.installing = false;
        if (previewInstallProcess === proc) previewInstallProcess = null;
        emitState();
      },
      onExit(code, signal) {
        appendLog(`dependency install exited (${signal || (code ?? "done")})`);
        const nextDetection = detectPreviewServer(detection.cwd);
        previewState.installing = false;
        previewState.needsInstall = Boolean(nextDetection.needsInstall);
        previewState.command = nextDetection.command || detection.command || "";
        if (previewInstallProcess === proc) previewInstallProcess = null;
        emitState();
      }
    });

    return status();
  }

  function dispose() {
    if (previewInstallProcess) {
      previewInstallProcess.kill();
      previewInstallProcess = null;
    }
    if (previewProcess) {
      previewProcess.kill();
      previewProcess = null;
    }
  }

  function artifacts(cwdArg) {
    const cwd = path.resolve(String(cwdArg || backendCwd()));
    const files = [];
    scanArtifacts(cwd, files, 0, { visited: 0 });
    return files
      .sort((left, right) => right.mtimeMs - left.mtimeMs)
      .slice(0, 20)
      .map((item) => ({
        id: `workspace-${item.path}`,
        label: path.basename(item.path),
        path: item.path,
        relativePath: path.relative(cwd, item.path) || path.basename(item.path),
        source: "workspace",
        time: "",
        url: fileUrl(item.path)
      }));
  }

  function detectPreviewServer(cwdArg) {
    const cwd = path.resolve(String(cwdArg || backendCwd()));
    const packagePath = path.join(cwd, "package.json");
    if (!fs.existsSync(packagePath)) {
      return {
        available: false,
        command: "",
        cwd,
        error: "",
        needsInstall: false,
        packageName: "",
        reason: "No package.json found in this workspace.",
        script: "",
        url: ""
      };
    }

    try {
      const pkg = JSON.parse(fs.readFileSync(packagePath, "utf8"));
      const scripts = pkg.scripts || {};
      const script = ["dev", "start", "preview", "serve"].find((name) => typeof scripts[name] === "string");
      return {
        available: Boolean(script),
        command: script ? `npm run ${script}` : "",
        cwd,
        error: "",
        needsInstall: !fs.existsSync(path.join(cwd, "node_modules")),
        packageName: String(pkg.name || path.basename(cwd)),
        reason: script ? "" : "package.json has no dev/start/preview/serve script.",
        script: script || "",
        scriptBody: script ? String(scripts[script]) : "",
        url: ""
      };
    } catch (error) {
      return {
        available: false,
        command: "",
        cwd,
        error: errorText(error),
        needsInstall: false,
        packageName: path.basename(cwd),
        reason: "Could not read package.json.",
        script: "",
        url: ""
      };
    }
  }

  function resetState(detection = {}) {
    previewState = {
      ...emptyState(),
      available: Boolean(detection.available),
      command: detection.command || "",
      cwd: detection.cwd || "",
      error: detection.error || "",
      needsInstall: Boolean(detection.needsInstall),
      packageName: detection.packageName || "",
      reason: detection.reason || "",
      script: detection.script || "",
      url: detection.url || ""
    };
  }

  function pipeProcess(proc, handlers) {
    proc.stdout.setEncoding("utf8");
    proc.stdout.on("data", (chunk) => {
      appendLog(chunk);
      handlers.onData?.();
    });
    proc.stderr.setEncoding("utf8");
    proc.stderr.on("data", (chunk) => {
      appendLog(chunk);
      handlers.onData?.();
    });
    proc.on("error", handlers.onError);
    proc.on("exit", handlers.onExit);
  }

  function appendLog(text) {
    const clean = stripAnsi(String(text || "")).replace(/\r/g, "");
    if (!clean.trim()) return;
    const lines = clean.split("\n").filter(Boolean);
    previewState.logs = [...previewState.logs, ...lines].slice(-120);
    const url = firstLocalUrl(clean);
    if (url) previewState.url = url;
  }

  function emitState() {
    emit({ event: "previewState", state: status() });
  }

  function status() {
    return { ...previewState, logs: [...previewState.logs] };
  }

  return { artifacts, detect, dispose, install, start, status, stop };
}

function emptyState() {
  return {
    available: false,
    command: "",
    cwd: "",
    error: "",
    installing: false,
    logs: [],
    needsInstall: false,
    packageName: "",
    reason: "",
    running: false,
    script: "",
    url: ""
  };
}

function firstLocalUrl(text) {
  const match = String(text || "").match(/https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?[^\s"'<>)]*/i);
  return match ? match[0].replace(/\/$/, "") : "";
}

function inferPreviewUrl(scriptBody) {
  const script = String(scriptBody || "").toLowerCase();
  if (script.includes("next")) return "http://localhost:3000";
  if (script.includes("astro")) return "http://localhost:4321";
  if (script.includes("vite")) return "http://localhost:5173";
  return "";
}

function npmCommand() {
  return process.platform === "win32" ? "npm.cmd" : "npm";
}

function stripAnsi(text) {
  return String(text || "").replace(/\u001b\[[0-9;]*m/g, "");
}

function errorText(error) {
  return `${error.name || "Error"}: ${error.message || String(error)}`;
}

function scanArtifacts(current, files, depth, state) {
  if (depth > 3 || state.visited > 1000) return;
  let entries = [];
  try {
    entries = fs.readdirSync(current, { withFileTypes: true });
  } catch {
    return;
  }

  for (const entry of entries) {
    state.visited += 1;
    if (state.visited > 1000) return;
    if (entry.name.startsWith(".") || ARTIFACT_SKIP_DIRS.has(entry.name)) continue;
    const fullPath = path.join(current, entry.name);
    if (entry.isDirectory()) {
      scanArtifacts(fullPath, files, depth + 1, state);
      continue;
    }
    if (!entry.isFile() || !isPreviewable(entry.name)) continue;
    try {
      const stat = fs.statSync(fullPath);
      files.push({ mtimeMs: stat.mtimeMs, path: fullPath });
    } catch {
      // Ignore files that disappear while scanning.
    }
  }
}

function isPreviewable(fileName) {
  return /\.(html?|svg)$/i.test(fileName);
}

function fileUrl(filePath) {
  return `file:///${encodeURI(filePath.replace(/\\/g, "/")).replace(/#/g, "%23")}`;
}

module.exports = { createPreviewManager };
