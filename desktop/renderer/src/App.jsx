import { useEffect, useMemo, useRef, useState } from "react";
import { ChatView } from "./components/ChatView.jsx";
import { CodeView } from "./components/CodeView.jsx";
import { AgentsView } from "./components/AgentsView.jsx";
import { DesignView } from "./components/DesignView.jsx";
import { ProviderSettings } from "./components/ProviderSettings.jsx";
import { Sidebar } from "./components/Sidebar.jsx";
import { SplashScreen } from "./components/SplashScreen.jsx";
import { appendEvent } from "./lib/events.js";

const SESSION_VIEWS = new Set(["chat", "code", "design"]);
const SESSION_STORAGE_KEY = "crypt.desktop.sessions.v1";
const SESSION_EVENT_LIMIT = 180;

export function App() {
  const [snapshot, setSnapshot] = useState(null);
  const [sessionState, setSessionState] = useState(createInitialSessionState);
  const [activeView, setActiveView] = useState("chat");
  const [connected, setConnected] = useState(false);
  const [lane, setLane] = useState("builder");
  const [showSplash, setShowSplash] = useState(true);
  const [splashLeaving, setSplashLeaving] = useState(false);
  const transcriptRef = useRef(null);
  const frameRef = useRef(null);
  const activeSessionRef = useRef("");
  const taskSessionRef = useRef("");

  const activeSessionView = sessionViewFor(activeView);
  const activeSessionId = sessionState.activeByView[activeSessionView];
  const activeSession = sessionState.sessions.find((session) => session.id === activeSessionId) || sessionState.sessions[0];
  const events = activeSession?.events || [];
  activeSessionRef.current = activeSessionId;

  useEffect(() => {
    saveSessionState(sessionState);
  }, [sessionState]);

  useEffect(() => {
    let leaveTimer;
    let hideTimer;
    const startIntro = () => {
      leaveTimer = window.setTimeout(() => setSplashLeaving(true), 8800);
      hideTimer = window.setTimeout(() => setShowSplash(false), 10600);
    };
    if (document.readyState === "complete") {
      startIntro();
    } else {
      window.addEventListener("load", startIntro, { once: true });
    }
    return () => {
      window.removeEventListener("load", startIntro);
      window.clearTimeout(leaveTimer);
      window.clearTimeout(hideTimer);
    };
  }, []);

  function appendPayloadToSession(sessionId, payload) {
    if (!sessionId) return;
    setSessionState((state) => ({
      ...state,
      sessions: state.sessions.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              events: appendEvent(session.events, payload),
              updatedAt: Date.now()
            }
          : session
      )
    }));
  }

  function replaceSessionEvents(sessionId, nextEvents) {
    if (!sessionId) return;
    setSessionState((state) => ({
      ...state,
      sessions: state.sessions.map((session) =>
        session.id === sessionId
          ? {
              ...session,
              events: nextEvents,
              updatedAt: Date.now()
            }
          : session
      )
    }));
  }

  useEffect(() => {
    if (!window.crypt) {
      appendPayloadToSession(activeSessionRef.current, { event: "error", text: "Electron bridge unavailable." });
      return undefined;
    }

    const unsubscribe = window.crypt.onEvent((payload) => {
      if (payload?.snapshot) setSnapshot(payload.snapshot);
      if (payload?.event === "ready" || payload?.event === "snapshot") setConnected(true);
      if (payload?.event === "daemonExit" || payload?.event === "daemonRestarting") setConnected(false);

      if (payload?.event === "ready" || payload?.event === "snapshot") return;

      const targetSessionId = payload?.sessionKey || taskSessionRef.current || activeSessionRef.current;
      if (payload?.event === "sessionReset") {
        replaceSessionEvents(targetSessionId, []);
        return;
      }
      if (payload?.event === "taskStarted" && payload?.sessionKey) {
        taskSessionRef.current = payload.sessionKey;
      }
      appendPayloadToSession(targetSessionId, payload);
      if (payload?.event === "taskFinished" || payload?.event === "taskFailed") {
        taskSessionRef.current = "";
      }
    });

    window.crypt.send({ type: "hello" });
    return unsubscribe;
  }, []);

  useEffect(() => {
    activeSessionRef.current = activeSessionId;
  }, [activeSessionId]);

  useEffect(() => {
    transcriptRef.current?.scrollTo({
      top: transcriptRef.current.scrollHeight,
      behavior: "smooth"
    });
  }, [events]);

  const send = (command = {}) => {
    const sessionId = activeSessionRef.current || activeSessionId;
    const channel = activeSessionView;
    const enriched =
      command.type === "sendPrompt" || command.type === "newSession"
        ? { ...command, channel, sessionKey: sessionId }
        : command;
    if (command.type === "sendPrompt") {
      taskSessionRef.current = sessionId;
    }
    return window.crypt?.send(enriched);
  };
  const running = Boolean(snapshot?.activeTask);
  const providers = snapshot?.providers || [];
  const activeRoute = useMemo(
    () => (snapshot?.routes || []).find((route) => route.role === lane),
    [lane, snapshot?.routes]
  );

  const startNewSession = () => {
    const view = sessionViewFor(activeView);
    const nextIndex = sessionState.sessions.filter((session) => session.view === view).length + 1;
    const session = createSession(view, nextIndex);
    setSessionState((state) => ({
      sessions: [session, ...state.sessions],
      activeByView: { ...state.activeByView, [view]: session.id }
    }));
    activeSessionRef.current = session.id;
    taskSessionRef.current = "";
    setActiveView(view);
    window.crypt?.send({ type: "newSession", channel: view, sessionKey: session.id });
  };

  const selectSession = (sessionId) => {
    const session = sessionState.sessions.find((item) => item.id === sessionId);
    if (!session) return;
    taskSessionRef.current = "";
    activeSessionRef.current = session.id;
    setSessionState((state) => ({
      sessions: state.sessions,
      activeByView: { ...state.activeByView, [session.view]: session.id }
    }));
    setActiveView(session.view);
  };

  const clearActiveSession = () => {
    taskSessionRef.current = "";
    replaceSessionEvents(activeSessionId, []);
    window.crypt?.send({ type: "newSession", channel: activeSessionView, sessionKey: activeSessionId });
  };

  const recordUserMessage = (text) => {
    const sessionId = activeSessionRef.current || activeSessionId;
    setSessionState((state) => ({
      ...state,
      sessions: state.sessions.map((session) => {
        if (session.id !== sessionId) return session;
        return {
          ...session,
          name: isDefaultSessionName(session) ? titleFromPrompt(text) : session.name,
          events: appendEvent(session.events, { event: "user", text }),
          updatedAt: Date.now()
        };
      })
    }));
  };

  const moveLight = (event) => {
    const frame = frameRef.current;
    if (!frame) return;
    const rect = frame.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width) * 100;
    const y = ((event.clientY - rect.top) / rect.height) * 100;
    frame.style.setProperty("--mx", `${x.toFixed(2)}%`);
    frame.style.setProperty("--my", `${y.toFixed(2)}%`);
  };

  const skipSplash = () => {
    setSplashLeaving(true);
    window.setTimeout(() => setShowSplash(false), 1450);
  };

  if (showSplash && !splashLeaving) {
    return (
      <div className="app-frame splash-host" ref={frameRef} onPointerMove={moveLight}>
        <SplashScreen leaving={splashLeaving} onSkip={skipSplash} />
      </div>
    );
  }

  return (
    <div className={showSplash ? "app-frame app-reveal" : "app-frame"} ref={frameRef} onPointerMove={moveLight}>
      <Sidebar
        activeSessionId={activeSessionId}
        activeView={activeView}
        connected={connected}
        onChangeView={setActiveView}
        onNewSession={startNewSession}
        onSelectSession={selectSession}
        sessions={sessionState.sessions}
        snapshot={snapshot}
      />

      {activeView === "providers" ? (
        <ProviderSettings providers={providers} send={send} snapshot={snapshot} />
      ) : activeView === "agents" ? (
        <AgentsView
          activeRoute={activeRoute}
          events={events}
          lane={lane}
          onClearView={clearActiveSession}
          onLaneChange={setLane}
          onRestart={() => window.crypt?.restart?.()}
          onUserMessage={recordUserMessage}
          running={running}
          sessionName={activeSession?.name}
          send={send}
          snapshot={snapshot}
          transcriptRef={transcriptRef}
        />
      ) : activeView === "code" ? (
        <CodeView
          activeRoute={activeRoute}
          events={events}
          lane={lane}
          onClearView={clearActiveSession}
          onLaneChange={setLane}
          onRestart={() => window.crypt?.restart?.()}
          onUserMessage={recordUserMessage}
          running={running}
          sessionName={activeSession?.name}
          send={send}
          snapshot={snapshot}
          transcriptRef={transcriptRef}
        />
      ) : activeView === "design" ? (
        <DesignView
          activeRoute={activeRoute}
          events={events}
          lane={lane}
          onClearView={clearActiveSession}
          onLaneChange={setLane}
          onRestart={() => window.crypt?.restart?.()}
          onUserMessage={recordUserMessage}
          running={running}
          sessionName={activeSession?.name}
          send={send}
          snapshot={snapshot}
          transcriptRef={transcriptRef}
        />
      ) : (
        <ChatView
          activeRoute={activeRoute}
          events={events}
          lane={lane}
          onClearView={clearActiveSession}
          onLaneChange={setLane}
          onNewSession={startNewSession}
          onRestart={() => window.crypt?.restart?.()}
          onUserMessage={recordUserMessage}
          running={running}
          sessionName={activeSession?.name}
          send={send}
          snapshot={snapshot}
          transcriptRef={transcriptRef}
        />
      )}

      {showSplash ? <SplashScreen leaving={splashLeaving} onSkip={skipSplash} /> : null}
    </div>
  );
}

export default App;

function createInitialSessionState() {
  const saved = loadSessionState();
  if (saved) return saved;
  const chat = createSession("chat", 1);
  const code = createSession("code", 1);
  const design = createSession("design", 1);
  return {
    sessions: [chat, code, design],
    activeByView: {
      chat: chat.id,
      code: code.id,
      design: design.id
    }
  };
}

function createSession(view, index) {
  const now = Date.now();
  return {
    id: `${view}-${now}-${Math.random().toString(16).slice(2)}`,
    view,
    name: `${titleCase(view)} ${index}`,
    events: [],
    createdAt: now,
    updatedAt: now
  };
}

function sessionViewFor(view) {
  return SESSION_VIEWS.has(view) ? view : "chat";
}

function titleCase(value) {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

function titleFromPrompt(text) {
  const firstLine = String(text || "").trim().split(/\r?\n/, 1)[0] || "Untitled";
  return firstLine.length > 42 ? `${firstLine.slice(0, 39).trim()}...` : firstLine;
}

function isDefaultSessionName(session) {
  return new RegExp(`^${titleCase(session.view)} \\d+$`).test(session.name);
}

function loadSessionState() {
  try {
    const raw = window.localStorage?.getItem(SESSION_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.sessions) || !parsed.sessions.length) return null;
    const sessions = parsed.sessions
      .filter((session) => SESSION_VIEWS.has(session?.view) && session.id)
      .map((session) => ({
        id: String(session.id),
        view: session.view,
        name: String(session.name || titleCase(session.view)),
        events: Array.isArray(session.events) ? session.events.slice(-SESSION_EVENT_LIMIT) : [],
        createdAt: Number(session.createdAt) || Date.now(),
        updatedAt: Number(session.updatedAt) || Date.now()
      }));
    if (!sessions.length) return null;
    const activeByView = {};
    for (const view of SESSION_VIEWS) {
      const savedId = parsed.activeByView?.[view];
      activeByView[view] = sessions.find((session) => session.id === savedId && session.view === view)?.id
        || sessions.find((session) => session.view === view)?.id
        || sessions[0].id;
    }
    return { sessions, activeByView };
  } catch {
    return null;
  }
}

function saveSessionState(state) {
  try {
    const sessions = state.sessions.slice(0, 36).map((session) => ({
      ...session,
      events: (session.events || []).slice(-SESSION_EVENT_LIMIT)
    }));
    window.localStorage?.setItem(SESSION_STORAGE_KEY, JSON.stringify({ ...state, sessions }));
  } catch {
    // Losing local session history should never break the desktop shell.
  }
}
