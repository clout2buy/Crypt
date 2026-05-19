import { describe, expect, it } from "vitest";
import {
  mergePreviewArtifacts,
  previewArtifactsFromEvents
} from "./artifacts.js";

const workspace = "C:\\Users\\Clout\\OneDrive\\Desktop\\site";

describe("previewArtifactsFromEvents", () => {
  it("detects previewable files from multi_edit change args", () => {
    const artifacts = previewArtifactsFromEvents([
      {
        id: "evt-1",
        event: "tool",
        rawEvent: "toolResult",
        ok: true,
        tool: "multi_edit",
        args: {
          changes: [
            { path: "src/app.js", edits: [{ old: "a", new: "b" }] },
            { path: "prototype.html", edits: [{ old: "a", new: "b" }] }
          ]
        },
        text: "edited 1 file(s)"
      }
    ], workspace);

    expect(artifacts).toHaveLength(1);
    expect(artifacts[0]).toMatchObject({
      label: "prototype.html",
      path: `${workspace}\\prototype.html`,
      relativePath: "prototype.html"
    });
    expect(artifacts[0].url).toContain("crypt-preview://local/");
  });

  it("falls back to diff headers when restored history has no args", () => {
    const artifacts = previewArtifactsFromEvents([
      {
        id: "evt-2",
        event: "tool",
        rawEvent: "toolResult",
        ok: true,
        tool: "multi_edit",
        text: "edited 1 file(s):\n\n--- super-animated-dashboard.html ---\n@@ -1,2 +1,2 @@"
      }
    ], workspace);

    expect(artifacts[0].label).toBe("super-animated-dashboard.html");
    expect(artifacts[0].path).toBe(`${workspace}\\super-animated-dashboard.html`);
  });
});

describe("mergePreviewArtifacts", () => {
  it("keeps the first visible artifact for a path and drops duplicates", () => {
    const merged = mergePreviewArtifacts(
      [{ id: "a", path: "C:\\site\\prototype.html", url: "one" }],
      [{ id: "b", path: "C:\\site\\prototype.html", url: "two" }],
      [{ id: "c", path: "C:\\site\\dashboard.html", url: "three" }]
    );

    expect(merged.map((item) => item.id)).toEqual(["a", "c"]);
  });
});
