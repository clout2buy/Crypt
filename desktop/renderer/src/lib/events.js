const MAX_EVENTS = 180;
let eventSequence = 0;

export function pushEvent(setEvents, payload) {
  setEvents((current) => mergeEvent(current, normalizeEvent(payload)).slice(-MAX_EVENTS));
}

export function normalizeEvent(payload = {}) {
  const rawEvent = payload.event || "system";
  const event = normalizedEventName(payload);
  eventSequence += 1;
  return {
    id: eventId(payload, event, rawEvent),
    event,
    rawEvent,
    tone: toneFor(payload),
    label: labelFor(payload),
    time: formatTime(payload.ts),
    text: textFor(payload),
    tool: payload.tool || "",
    ok: payload.ok,
    args: payload.args || null,
    append: rawEvent === "assistantDelta",
    done: rawEvent === "taskFinished" || rawEvent === "toolResult"
  };
}

function mergeEvent(current, next) {
  if (!next.text && (next.event === "assistant" || next.event === "tool")) {
    return current;
  }

  const idx = current.findIndex((event) => event.id === next.id);
  if (idx < 0) {
    return [...current, next];
  }

  const updated = [...current];
  const prev = updated[idx];
  const text = next.append ? `${prev.text || ""}${next.text || ""}` : (next.text || prev.text || "");
  updated[idx] = {
    ...prev,
    ...next,
    text,
    time: next.time || prev.time
  };
  return updated;
}

function normalizedEventName(payload) {
  const event = payload.event || "system";
  if (event === "assistantDelta" || event === "taskFinished") return "assistant";
  if (event === "toolCall" || event === "toolStarted" || event === "toolProgress" || event === "toolResult") return "tool";
  if (event === "taskFailed") return "error";
  return event;
}

function eventId(payload, event, rawEvent) {
  const taskId = payload.id || "event";
  if (event === "assistant") return `${taskId}-assistant`;
  if (event === "tool") return `${taskId}-tool-${payload.callId || eventSequence}`;
  if (rawEvent === "snapshot" || rawEvent === "ready") return rawEvent;
  return `${taskId}-${event}-${eventSequence}`;
}

function toneFor(payload) {
  const event = payload.event || "";
  if (event === "user") return "user";
  if (event === "assistantDelta" || event === "taskFinished") return "assistant";
  if (event === "toolCall" || event === "toolStarted" || event === "toolProgress") return "tool";
  if (event === "toolResult") return payload.ok === false ? "error" : "tool";
  if (event.includes("Failed") || event === "error" || event === "daemonError" || event === "daemonExit") return "error";
  return "system";
}

function labelFor(payload) {
  if (payload.event === "user") return "You";
  if (payload.event === "assistantDelta" || payload.event === "taskFinished") return "Crypt";
  if (payload.event === "toolCall") return payload.tool || "Tool";
  if (payload.event === "toolStarted") return `Running - ${payload.tool || "tool"}`;
  if (payload.event === "toolProgress") return `Reading - ${payload.tool || "tool"}`;
  if (payload.event === "toolResult") return `${payload.ok === false ? "Failed" : "Complete"} - ${payload.tool || "tool"}`;
  if (payload.event === "commandResult") return payload.command || "Command";
  if (payload.event === "taskProgress") return payload.phase || "Progress";
  if (payload.event === "taskFailed") return "Failed";
  return String(payload.event || "System").replace(/([A-Z])/g, " $1").trim();
}

function textFor(payload) {
  if (payload.text) return String(payload.text);
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
