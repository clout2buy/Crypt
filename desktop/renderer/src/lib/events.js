const MAX_EVENTS = 180;
let eventSequence = 0;

export function pushEvent(setEvents, payload) {
  setEvents((current) => [...current, normalizeEvent(payload)].slice(-MAX_EVENTS));
}

export function normalizeEvent(payload = {}) {
  const event = payload.event || "system";
  eventSequence += 1;
  return {
    id: `${payload.id || "event"}-${event}-${eventSequence}`,
    event,
    tone: toneFor(payload),
    label: labelFor(payload),
    time: formatTime(payload.ts),
    text: textFor(payload),
    tool: payload.tool || "",
    ok: payload.ok,
    args: payload.args || null
  };
}

function toneFor(payload) {
  const event = payload.event || "";
  if (event === "user") return "user";
  if (event === "toolCall") return "tool";
  if (event === "toolResult") return payload.ok === false ? "error" : "tool";
  if (event.includes("Failed") || event === "error" || event === "daemonError" || event === "daemonExit") return "error";
  if (event === "taskFinished") return "assistant";
  return "system";
}

function labelFor(payload) {
  if (payload.event === "user") return "You";
  if (payload.event === "taskFinished") return "Crypt";
  if (payload.event === "toolCall") return payload.tool || "Tool call";
  if (payload.event === "toolResult") return payload.ok === false ? "Tool failed" : "Tool result";
  if (payload.event === "commandResult") return payload.command || "Command";
  if (payload.event === "taskProgress") return payload.phase || "Progress";
  return String(payload.event || "System").replace(/([A-Z])/g, " $1").trim();
}

function textFor(payload) {
  if (payload.text) return String(payload.text).trim();
  if (payload.error) return String(payload.error).trim();
  if (payload.prompt) return String(payload.prompt).trim();
  if (payload.event === "ready") return "Backend connected.";
  if (payload.event === "snapshot") return "State synchronized.";
  if (payload.event === "daemonRestarting") return "Restarting backend.";
  if (payload.event === "daemonExit") return "Backend stopped.";
  return "";
}

function formatTime(ts) {
  const date = ts ? new Date(ts * 1000) : new Date();
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
