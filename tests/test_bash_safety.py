from __future__ import annotations

import pytest

from tools.bash_safety import classify


@pytest.mark.parametrize(
    "command",
    [
        "git status --short",
        "git diff -- main.py",
        "git branch -a",
        "git remote -v",
        "git config --get user.email",
        "git tag --list",
        "git stash list",
    ],
)
def test_read_only_git_forms_are_safe(command: str):
    assert classify(command) == "safe"


@pytest.mark.parametrize(
    "command",
    [
        "git branch new-branch",
        "git remote add origin https://example.com/repo.git",
        "git config user.email x@example.com",
        "git tag v1.0.0",
        "git stash",
    ],
)
def test_mutating_git_forms_are_not_safe(command: str):
    assert classify(command) is None
