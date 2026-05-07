from __future__ import annotations

import sys
import types

from tools import read_file, read_media
from tools import edit_file


def test_read_file_accepts_absolute_path_outside_workspace(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("hello\n", encoding="utf-8")
    monkeypatch.setenv("CRYPT_ROOT", str(workspace))

    out = read_file.run({"path": str(outside)})

    assert "hello" in out


def test_read_media_pdf_outside_workspace_includes_extracted_text(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "guide.pdf"
    outside.write_bytes(b"%PDF-1.4\n% fake test pdf\n")
    monkeypatch.setenv("CRYPT_ROOT", str(workspace))

    class _Page:
        def extract_text(self):
            return "Exam objective text"

    class _Reader:
        pages = [_Page()]

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        types.SimpleNamespace(PdfReader=lambda path: _Reader()),
    )

    out = read_media.run({"path": str(outside)})

    assert out["__crypt_tool_result__"] is True
    text = "\n".join(block.get("text", "") for block in out["content"] if block.get("type") == "text")
    assert "Exam objective text" in text
    assert any(block.get("type") == "document" for block in out["content"])


def test_range_covering_whole_file_counts_as_full_read(workspace):
    p = workspace / "deck.html"
    p.write_text("one\ntwo\n", encoding="utf-8")

    read_file.run({"path": str(p), "offset": 1, "limit": 2000})
    edit_file.run({"path": str(p), "old": "two", "new": "TWO"})

    assert p.read_text(encoding="utf-8") == "one\nTWO\n"
