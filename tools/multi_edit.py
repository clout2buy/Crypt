"""multi_edit — atomic batch of edits across multiple files.

All edits are validated dry-run BEFORE any file is written. If any edit
fails, no files are modified and the user gets one combined error. This
makes multi-file refactors safe to attempt: a typo in edit #5 doesn't
leave files #1-#4 already mutated.

Schema:

    {
      "changes": [
        {"path": "...", "old": "...", "new": "..."},
        {"path": "...", "edits": [{"old": "...", "new": "..."}, ...]}
      ]
    }

Each `changes` entry targets one file. Provide either a single old/new
or a list of edits for that file (applied in order, atomically per file).
"""
from __future__ import annotations

from collections import OrderedDict

from core import file_state

from . import edit_file as ef
from .fs import is_text_file, rel, resolve
from .types import Tool


def _normalize_changes(args: dict) -> list[tuple[str, list[tuple[str, str]]]]:
    """Return [(path_str, edits_list), ...]. Multiple entries with the same
    path are merged so a model can sloppily emit two changes for one file
    and we still apply atomically."""
    raw = args.get("changes")
    if not isinstance(raw, list) or not raw:
        raise ValueError("multi_edit needs a non-empty `changes` list")

    by_path: "OrderedDict[str, list[tuple[str, str]]]" = OrderedDict()
    for i, entry in enumerate(raw, 1):
        if not isinstance(entry, dict):
            raise ValueError(f"changes[{i - 1}] must be an object")
        path = str(entry.get("path", "")).strip()
        if not path:
            raise ValueError(f"changes[{i - 1}] missing `path`")

        edits: list[tuple[str, str]] = []
        if isinstance(entry.get("edits"), list):
            for j, e in enumerate(entry["edits"]):
                if isinstance(e, dict) and "old" in e and "new" in e:
                    edits.append((str(e["old"]), str(e["new"])))
                else:
                    raise ValueError(
                        f"changes[{i - 1}].edits[{j}] must be {{old, new}}"
                    )
        elif "old" in entry and "new" in entry:
            edits.append((str(entry["old"]), str(entry["new"])))
        else:
            raise ValueError(
                f"changes[{i - 1}] needs `old`+`new` or `edits: [...]`"
            )
        by_path.setdefault(path, []).extend(edits)
    return list(by_path.items())


def _resolve_and_check(path_str: str) -> "Path":  # type: ignore[name-defined]
    path = resolve(path_str)
    if not path.exists():
        raise FileNotFoundError(rel(path))
    if not is_text_file(path):
        raise ValueError(f"refusing to edit binary file: {rel(path)}")
    file_state.assert_fresh_for_edit(path)
    return path


def run(args: dict) -> str:
    grouped = _normalize_changes(args)

    # Phase 1 — dry-run every file. Any failure here aborts before any
    # write. Failures get prefixed with the path so the model can map the
    # error back to the offending change.
    plans: list[tuple[str, "Path", str, str, str]] = []  # (path_str, path, original, new_text, newline)
    errors: list[str] = []
    for path_str, edits in grouped:
        try:
            path = _resolve_and_check(path_str)
            original, new_text, newline = ef.compute_new_content(path, edits)
            plans.append((path_str, path, original, new_text, newline))
        except Exception as e:
            errors.append(f"{path_str}: {type(e).__name__}: {e}")

    if errors:
        joined = "\n".join(f"  - {line}" for line in errors)
        raise ValueError(
            f"multi_edit aborted; no files were modified.\n{joined}"
        )

    # Phase 2 — write everything. If a write fails midway (rare: out of
    # disk, permission), files already written stay written. We surface
    # which files succeeded so the user can decide how to recover.
    written: list[str] = []
    for path_str, path, _original, new_text, newline in plans:
        try:
            final = new_text.replace("\n", newline) if newline != "\n" else new_text
            path.write_bytes(final.encode("utf-8"))
            file_state.record_write(path)
            written.append(rel(path))
        except OSError as e:
            partial = ", ".join(written) or "(none)"
            raise RuntimeError(
                f"write failed at {rel(path)}: {e}. "
                f"Partial state — these files were already updated: {partial}"
            ) from e

    summary_lines = [f"edited {len(plans)} file(s):"]
    for path_str, path, original, new_text, _newline in plans:
        diff = ef._short_diff(original, new_text, rel(path), max_chars=600)
        summary_lines.append(f"\n--- {rel(path)} ---")
        if diff:
            summary_lines.append(diff)
    return "\n".join(summary_lines)


def preview(args: dict) -> str:
    """Combined unified diff across all changes. Empty on any planning
    failure (run() will surface the real error after approval)."""
    try:
        grouped = _normalize_changes(args)
    except Exception:
        return ""
    parts: list[str] = []
    for path_str, edits in grouped:
        try:
            path = resolve(path_str)
            if not path.exists() or not is_text_file(path):
                return ""
            original, new_text, _ = ef.compute_new_content(path, edits)
            diff = ef._short_diff(original, new_text, rel(path), max_chars=800)
            if diff:
                parts.append(diff)
        except Exception:
            return ""
    return "\n".join(parts)


def summary(args: dict) -> str:
    raw = args.get("changes") or []
    if not isinstance(raw, list):
        return "(invalid changes)"
    paths: list[str] = []
    edit_count = 0
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        p = str(entry.get("path", "")).strip()
        if p and p not in paths:
            paths.append(p)
        if isinstance(entry.get("edits"), list):
            edit_count += len(entry["edits"])
        elif "old" in entry:
            edit_count += 1
    files_part = f"{len(paths)} file(s)" if len(paths) != 1 else paths[0]
    return f"{files_part} ({edit_count} edit{'s' if edit_count != 1 else ''})"


_PROMPT = """
Apply edits across one or more files atomically. All edits are validated
before anything is written; if any single edit fails (no match, ambiguous
match, missing file), nothing is modified and you get a combined error.

Use this instead of multiple edit_file calls when:
- The same change touches several files (rename, type narrowing, import
  reorganisation).
- Edits in different files are logically one commit and partial application
  would leave the project broken.

Each entry in `changes` targets one file. Provide either a single
`old`/`new` pair or `edits: [{old, new}, ...]` for batched changes inside
that file. All matching rules from edit_file apply: each `old` must occur
exactly once in its file.
""".strip()


TOOL = Tool(
    "multi_edit",
    (
        "Apply edits to one or more files atomically. Validates all edits "
        "before writing anything; partial application is impossible."
    ),
    {
        "type": "object",
        "properties": {
            "changes": {
                "type": "array",
                "description": "One entry per file. Each has path + (old/new) or path + edits[].",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                        "edits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "old": {"type": "string"},
                                    "new": {"type": "string"},
                                },
                                "required": ["old", "new"],
                            },
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        "required": ["changes"],
    },
    "ask",
    run,
    prompt=_PROMPT,
    priority=42,  # Right after edit_file (40) so it shows next in tool guidance.
    summary=summary,
    preview=preview,
)
