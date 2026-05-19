import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { EventStream } from "./EventStream.jsx";

afterEach(() => {
  cleanup();
});

describe("EventStream", () => {
  it("shows active tools without decorative trace or progress animations", () => {
    const { container } = render(
      <EventStream
        events={[{
          id: "tool-1",
          event: "tool",
          rawEvent: "toolStarted",
          tone: "tool",
          label: "Running - open_file",
          time: "2:09 AM",
          text: "super-animated-dashboard.html",
          tool: "open_file"
        }]}
      />
    );

    expect(screen.getByText("Running")).toBeTruthy();
    expect(container.querySelector(".tool-trace")).toBeNull();
    expect(container.querySelector(".tool-progress")).toBeNull();
  });

  it("wires approval buttons back to the daemon command channel", async () => {
    const send = vi.fn();
    const user = userEvent.setup();
    render(
      <EventStream
        send={send}
        events={[{
          id: "approval-1",
          event: "approval",
          rawEvent: "approvalRequested",
          tone: "tool",
          label: "Approval - open_file",
          time: "2:09 AM",
          text: "run this?\nsuper-animated-dashboard.html",
          tool: "open_file",
          approvalId: "approval-1"
        }]}
      />
    );

    await user.click(screen.getByRole("button", { name: "Approve" }));

    expect(send).toHaveBeenCalledWith({
      type: "approvalResponse",
      approvalId: "approval-1",
      approved: true
    });
  });

  it("does not style verification notes as test result cards", () => {
    const { container } = render(
      <EventStream
        events={[{
          id: "assistant-1",
          event: "assistant",
          rawEvent: "taskFinished",
          tone: "assistant",
          label: "Crypt",
          time: "2:06 AM",
          text: "Verification note: no runtime evidence recorded for tests or checks; treat this result as unverified."
        }]}
      />
    );

    expect(container.querySelector(".test-result-card")).toBeNull();
  });

  it("renders user messages as visible chat bubbles", () => {
    render(
      <EventStream
        events={[{
          id: "user-1",
          event: "user",
          rawEvent: "user",
          tone: "user",
          label: "You",
          time: "2:10 AM",
          text: "this should stay visible"
        }]}
      />
    );

    expect(screen.getByText("You")).toBeTruthy();
    expect(screen.getByText("this should stay visible")).toBeTruthy();
  });
});
