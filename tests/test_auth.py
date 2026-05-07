from __future__ import annotations

import time

from core import auth, settings


def test_provider_auth_records_do_not_clobber_each_other(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_PATH", tmp_path / "auth.json")
    monkeypatch.setattr(auth, "AUTH_PATH", tmp_path / "auth.json")

    auth.save_provider("anthropic", {"type": "oauth", "access": "anthropic-token"})
    auth.save_provider("openai-codex", {
        "type": "openai-codex",
        "access": "chatgpt-token",
        "refresh": "refresh",
        "expires": int(time.time() * 1000) + 60_000,
        "account_id": "account-123",
        "email": "me@example.com",
        "plan": "pro",
    })

    assert auth.load_provider("anthropic")["access"] == "anthropic-token"
    cred = auth.resolve_openai_codex()

    assert cred is not None
    assert cred.token == "chatgpt-token"
    assert cred.account_id == "account-123"
    assert cred.email == "me@example.com"
    assert cred.plan == "pro"


def test_legacy_anthropic_auth_file_still_resolves(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "AUTH_PATH", tmp_path / "auth.json")
    monkeypatch.setattr(auth, "AUTH_PATH", tmp_path / "auth.json")
    auth.save({"type": "oauth", "access": "anthropic-token"})

    cred = auth.resolve_anthropic()

    assert cred is not None
    assert cred.token == "anthropic-token"
