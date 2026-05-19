const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("crypt", {
  send(command) {
    return ipcRenderer.invoke("crypt:command", command);
  },
  openExternal(url) {
    return ipcRenderer.invoke("crypt:openExternal", url);
  },
  restart() {
    return ipcRenderer.invoke("crypt:restartDaemon");
  },
  chooseDirectory() {
    return ipcRenderer.invoke("crypt:chooseDirectory");
  },
  choosePreviewFile() {
    return ipcRenderer.invoke("crypt:choosePreviewFile");
  },
  captureScreenshot(bounds) {
    return ipcRenderer.invoke("crypt:captureScreenshot", bounds || null);
  },
  detectPreviewServer(cwd) {
    return ipcRenderer.invoke("crypt:detectPreviewServer", cwd);
  },
  startPreviewServer(cwd) {
    return ipcRenderer.invoke("crypt:startPreviewServer", cwd);
  },
  stopPreviewServer() {
    return ipcRenderer.invoke("crypt:stopPreviewServer");
  },
  installPreviewDeps(cwd) {
    return ipcRenderer.invoke("crypt:installPreviewDeps", cwd);
  },
  listPreviewArtifacts(cwd) {
    return ipcRenderer.invoke("crypt:listPreviewArtifacts", cwd);
  },
  onPreviewEvent(callback) {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("crypt:preview", listener);
    return () => ipcRenderer.removeListener("crypt:preview", listener);
  },
  onEvent(callback) {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("crypt:event", listener);
    return () => ipcRenderer.removeListener("crypt:event", listener);
  }
});
