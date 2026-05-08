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
  onEvent(callback) {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on("crypt:event", listener);
    return () => ipcRenderer.removeListener("crypt:event", listener);
  }
});
