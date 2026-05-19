import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PreviewPanel } from "./PreviewPanel.jsx";

const artifact = {
  id: "artifact-1",
  label: "super-animated-dashboard.html",
  path: "C:\\site\\super-animated-dashboard.html",
  url: "crypt-preview://local/C:/site/super-animated-dashboard.html"
};

let previewListeners;

beforeEach(() => {
  previewListeners = [];
  window.crypt = {
    captureScreenshot: vi.fn(),
    choosePreviewFile: vi.fn(),
    detectPreviewServer: vi.fn().mockResolvedValue({ available: false }),
    installPreviewDeps: vi.fn(),
    onPreviewEvent: vi.fn((listener) => {
      previewListeners.push(listener);
      return () => {
        previewListeners = previewListeners.filter((item) => item !== listener);
      };
    }),
    openExternal: vi.fn(),
    startPreviewServer: vi.fn(),
    stopPreviewServer: vi.fn()
  };
});

afterEach(() => {
  cleanup();
  delete window.crypt;
});

describe("PreviewPanel", () => {
  it("auto-loads the newest generated artifact and changes responsive frames", async () => {
    const user = userEvent.setup();
    const { container } = render(<PreviewPanel artifacts={[artifact]} workspace="C:\\site" />);

    await waitFor(() => {
      expect(container.querySelector("iframe")?.getAttribute("src")).toBe(artifact.url);
    });

    await user.click(screen.getByTitle("Phone frame"));

    expect(container.querySelector(".preview-stage")?.className).toContain("frame-phone");
  });

  it("dismisses the active artifact preview without leaving a broken iframe", async () => {
    const user = userEvent.setup();
    const onDismissArtifact = vi.fn();
    const { container } = render(
      <PreviewPanel artifacts={[artifact]} onDismissArtifact={onDismissArtifact} workspace="C:\\site" />
    );

    await waitFor(() => {
      expect(container.querySelector("iframe")).toBeTruthy();
    });
    await user.click(screen.getByTitle("Dismiss super-animated-dashboard.html"));

    expect(onDismissArtifact).toHaveBeenCalledWith("artifact-1");
    expect(container.querySelector("iframe")).toBeNull();
    expect(screen.getByText("No preview yet.")).toBeTruthy();
  });

  it("starts an available background preview server when autoStart is enabled", async () => {
    window.crypt.detectPreviewServer.mockResolvedValue({
      available: true,
      command: "npm run dev",
      cwd: "C:\\site",
      installing: false,
      needsInstall: false,
      packageName: "site",
      running: false
    });

    render(<PreviewPanel autoStart artifacts={[]} workspace="C:\\site" />);

    await waitFor(() => {
      expect(window.crypt.startPreviewServer).toHaveBeenCalledWith("C:\\site");
    });
  });

  it("reacts to running preview server events and opens the localhost URL", async () => {
    const { container } = render(<PreviewPanel artifacts={[]} workspace="C:\\site" />);

    await waitFor(() => {
      expect(window.crypt.onPreviewEvent).toHaveBeenCalled();
    });
    await act(async () => {
      previewListeners[0]?.({
        event: "previewState",
        state: {
          available: true,
          command: "npm run dev",
          packageName: "site",
          running: true,
          url: "http://127.0.0.1:5173"
        }
      });
    });

    await waitFor(() => {
      expect(container.querySelector("iframe")?.getAttribute("src")).toBe("http://127.0.0.1:5173");
    });
  });
});
