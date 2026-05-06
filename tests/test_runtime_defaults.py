from __future__ import annotations

from core import runtime


def test_default_approval_mode_auto_approves_edits_only():
    previous = runtime.approval_mode()
    runtime.set_approval_mode(runtime.APPROVAL_EDITS)
    try:
        assert runtime.approval_label() == "auto-edits"
        assert runtime.can_auto_approve("write_file") is True
        assert runtime.can_auto_approve("edit_file") is True
        assert runtime.can_auto_approve("bash") is False
    finally:
        runtime.set_approval_mode(previous)
