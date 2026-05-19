import { describe, expect, it } from "vitest";
import { appendEvent, normalizeEvent } from "./events.js";

describe("normalizeEvent", () => {
  it("formats approval prompts with the actual question and summary", () => {
    const event = normalizeEvent({
      event: "approvalRequested",
      id: "task-1",
      approvalId: "approval-1",
      tool: "open_file",
      question: "run this?",
      text: "super-animated-dashboard.html"
    });

    expect(event.event).toBe("approval");
    expect(event.text).toBe("run this?\nsuper-animated-dashboard.html");
    expect(event.approvalId).toBe("approval-1");
  });
});

describe("appendEvent", () => {
  it("merges approval resolution into the original approval card", () => {
    const requested = appendEvent([], {
      event: "approvalRequested",
      id: "task-1",
      approvalId: "approval-1",
      tool: "open_file",
      question: "run this?",
      text: "prototype.html"
    });
    const resolved = appendEvent(requested, {
      event: "approvalResolved",
      id: "task-1",
      approvalId: "approval-1",
      text: "approved"
    });

    expect(resolved).toHaveLength(1);
    expect(resolved[0]).toMatchObject({
      rawEvent: "approvalResolved",
      text: "approved"
    });
  });

  it("keeps user prompts when a run produces a lot of tool traffic", () => {
    let events = appendEvent([], { event: "user", text: "make the dashboard better" });

    for (let index = 0; index < 220; index += 1) {
      events = appendEvent(events, {
        event: "toolResult",
        id: "task-1",
        callId: `tool-${index}`,
        tool: "read_file",
        ok: true,
        text: `chunk ${index}`
      });
    }

    events = appendEvent(events, {
      event: "taskFinished",
      id: "task-1",
      text: "Done."
    });

    expect(events).toHaveLength(180);
    expect(events.some((event) => event.event === "user" && event.text === "make the dashboard better")).toBe(true);
    expect(events.at(-1)).toMatchObject({ event: "assistant", text: "Done." });
  });
});
