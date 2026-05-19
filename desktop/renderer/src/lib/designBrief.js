export const DESIGN_SURFACES = ["Prototype", "Landing page", "Dashboard", "Mobile app", "Slide", "Poster"];
export const DESIGN_DIRECTIONS = ["Tech utility", "Editorial", "Modern minimal", "Warm soft", "Brutalist"];
export const DESIGN_DEVICES = ["Responsive", "Desktop", "Tablet", "Phone"];

export function buildDesignPrompt({ device, direction, surface, text }) {
  return [
    "Design mode brief: create or revise a polished, browser-previewable design artifact in this workspace.",
    `Surface: ${surface}. Visual direction: ${direction}. Target frame: ${device}.`,
    "Prioritize real HTML/CSS/JS or React output, strong responsive states, visible interaction states, and a local preview artifact. Keep edits scoped and preserve previous useful work when iterating.",
    "If the user asks for pictures or media, use real loadable HTTPS image URLs after checking them, or create an honest local visual fallback. Do not label placeholder cards as the requested pictures.",
    "",
    "User brief:",
    text
  ].join("\n");
}
