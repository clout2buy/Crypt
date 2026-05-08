import { useEffect, useMemo, useRef, useState } from "react";
import { ChatView } from "./components/ChatView.jsx";
import { ProviderSettings } from "./components/ProviderSettings.jsx";
import { Sidebar } from "./components/Sidebar.jsx";
import { normalizeEvent, pushEvent } from "./lib/events.js";

export function App() {
  const [snapshot, setSnapshot] = useState(null);
  const [events, setEvents] = useState([]);
  const [activeView, setActiveView] = useState("chat");
  const [connected, setConnected] = useState(false);
  const [lane, setLane] = useState("builder");
  const transcriptRef = useRef(null);

  useEffect(() => {
    if (!window.crypt) {
      pushEvent(setEvents, { event: "error", text: "Electron bridge unavailable." });
      return undefined;
    }

    const unsubscribe = window.crypt.onEvent((payload) => {
      if (payload?.snapshot) setSnapshot(payload.snapshot);
      if (payload?.event === "ready" || payload?.event === "snapshot") setConnected(true);
      if (payload?.event === "daemonExit" || payload?.event === "daemonRestarting") setConnected(false);
      if (payload?.event === "sessionReset") {
        setEvents([normalizeEvent(payload)]);
        return;
      }
      pushEvent(setEvents, payload);
    });

    window.crypt.send({ type: "hello" });
    return unsubscribe;
  }, []);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [events]);

  const send = (command) => window.crypt?.send(command);
  const running = Boolean(snapshot?.activeTask);
  const providers = snapshot?.providers || [];
  const activeRoute = useMemo(
    () => (snapshot?.routes || []).find((route) => route.role === lane),
    [lane, snapshot?.routes]
  );

  const startNewSession = () => {
    setEvents([]);
    send({ type: "newSession" });
  };

  return (
    <div className="app-frame">
      <Sidebar
        activeView={activeView}
        connected={connected}
        onChangeView={setActiveView}
        onNewSession={startNewSession}
        snapshot={snapshot}
      />

      {activeView === "providers" ? (
        <ProviderSettings providers={providers} send={send} snapshot={snapshot} />
      ) : (
        <ChatView
          activeRoute={activeRoute}
          events={events}
          lane={lane}
          onClearView={() => setEvents([])}
          onLaneChange={setLane}
          onNewSession={startNewSession}
          onRestart={() => window.crypt?.restart?.()}
          onUserMessage={(text) => pushEvent(setEvents, { event: "user", text })}
          running={running}
          send={send}
          snapshot={snapshot}
          transcriptRef={transcriptRef}
        />
      )}
    </div>
  );
}

export default App;
