import {
  CircleCheck,
  FileCode2,
  ShieldCheck,
  Wrench
} from "lucide-react";
import { CryptGlyph } from "./CryptGlyph.jsx";

const RITUALS = [
  {
    id: "forge",
    title: "Build",
    tag: "build",
    glyph: "forge",
    prompt:
      "Build this into a complete usable result. Inspect the workspace first, choose the highest-leverage implementation path, make the changes, verify them, and finish with concrete evidence."
  },
  {
    id: "hunt",
    title: "Debug",
    tag: "debug",
    glyph: "hunt",
    prompt:
      "Debug this like a production issue. Reproduce or narrow it down, inspect the likely files, patch the cause, run focused verification, and explain the root cause clearly."
  },
  {
    id: "beauty",
    title: "Polish UI",
    tag: "visual",
    glyph: "beauty",
    prompt:
      "Do a serious visual polish pass. Improve layout, motion, color, typography, empty states, responsive behavior, and verify with a screenshot."
  },
  {
    id: "review",
    title: "Review",
    tag: "risk",
    glyph: "boss",
    prompt:
      "Review the current workspace like a strict senior engineer. Prioritize bugs, regressions, missing tests, and confusing UX. Fix the important issues that are safe to fix."
  },
  {
    id: "ship",
    title: "Ship",
    tag: "release",
    glyph: "ship",
    prompt:
      "Prepare this project to ship. Run the relevant checks, fix blockers, update any stale docs or obvious gaps, and leave a release-ready summary with what passed."
  },
  {
    id: "lore",
    title: "Map Repo",
    tag: "map",
    glyph: "lore",
    prompt:
      "Map this project for me. Explain the architecture, key flows, sharp edges, hidden opportunities, and the next three high-leverage improvements."
  }
];

export function MissionHud({ compact = false, events = [], running = false, snapshot }) {
  const stats = summarizeEvents(events);
  const status = running ? "Working" : stats.failed ? "Needs attention" : stats.tests ? "Verified" : stats.tools ? "Work recorded" : "Ready";
  const combo = Math.min(99, stats.tools + stats.artifacts + stats.tests + stats.approvals);
  const route = snapshot?.activeTask ? `task ${snapshot.activeTask}` : snapshot?.workspace ? basename(snapshot.workspace) : "no workspace";

  return (
    <section className={compact ? "mission-hud compact" : "mission-hud"}>
      <div className="mission-core">
        <div className={running ? "mission-orb live" : "mission-orb"} aria-hidden="true">
          <CryptGlyph name="sigil" size={24} />
        </div>
        <div>
          <span className="mission-kicker">Session</span>
          <strong>{status}</strong>
          <small>{route}</small>
        </div>
      </div>

      <div className="mission-meter" aria-label="Session activity">
        <span style={{ width: `${Math.min(100, 18 + combo * 5)}%` }} />
      </div>

      <div className="mission-stats">
        <Stat icon={Wrench} label="Actions" value={stats.tools} />
        <Stat icon={ShieldCheck} label="Checks" value={stats.tests} />
        <Stat icon={FileCode2} label="Files" value={stats.artifacts} />
        <Stat icon={CircleCheck} label="Approvals" value={stats.approvals} />
      </div>
    </section>
  );
}

export function RitualBoard({ disabled = false, onLaunch }) {
  return (
    <div className="ritual-board">
      <div className="ritual-board-title">
        <CryptGlyph name="shard" size={17} />
        <span>Start with one click</span>
      </div>
      <div className="ritual-grid">
        {RITUALS.map((ritual) => (
          <button
            className="ritual-card"
            disabled={disabled}
            key={ritual.id}
            type="button"
            onClick={() => onLaunch?.(ritual.prompt)}
          >
            <CryptGlyph name={ritual.glyph} size={20} />
            <span>{ritual.tag}</span>
            <strong>{ritual.title}</strong>
          </button>
        ))}
      </div>
    </div>
  );
}

function Stat({ icon: Icon, label, value }) {
  return (
    <div className="mission-stat">
      <Icon size={13} />
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function summarizeEvents(events) {
  const result = {
    approvals: 0,
    artifacts: 0,
    failed: 0,
    tests: 0,
    tools: 0,
    userMessages: 0
  };

  for (const event of events || []) {
    const text = `${event.label || ""}\n${event.text || ""}`.toLowerCase();
    if (event.event === "user") result.userMessages += 1;
    if (event.event === "approval") result.approvals += 1;
    if (event.event === "tool" && event.rawEvent === "toolResult") {
      result.tools += 1;
      if (event.ok === false) result.failed += 1;
    }
    if (/\b(pytest|ruff|vitest|npm run|tests?|passed|failed|build)\b/.test(text)) {
      result.tests += 1;
    }
    if (/\b(created|edited|wrote|overwrote|opened)\b.+\.(html?|svg|jsx?|tsx?|css|py|md)\b/.test(text)) {
      result.artifacts += 1;
    }
  }

  return result;
}

function basename(value) {
  const parts = String(value || "").replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || value;
}
