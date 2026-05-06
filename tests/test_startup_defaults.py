from __future__ import annotations

import argparse

import main
from core import settings
from core.doctor import _check_provider_auth


def _clear_provider_env(monkeypatch) -> None:
    for name in (
        "CRYPT_PROVIDER",
        "CRYPT_REQUIRE_SETUP",
        "CRYPT_PICKER",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OLLAMA_API_KEY",
        "OLLAMA_HOST",
    ):
        monkeypatch.delenv(name, raising=False)


def _args(**overrides):
    values = {
        "provider": None,
        "model": None,
        "cwd": None,
        "ollama_host": None,
        "no_picker": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fresh_clone_defaults_to_ollama_without_setup(monkeypatch):
    _clear_provider_env(monkeypatch)

    assert settings.provider_default({}) == settings.PROVIDER_OLLAMA
    assert main._needs_setup({}, _args()) is False


def test_env_and_saved_provider_precedence(monkeypatch):
    _clear_provider_env(monkeypatch)

    monkeypatch.setenv("OLLAMA_API_KEY", "ollama-key")
    assert settings.provider_default({}) == settings.PROVIDER_OLLAMA

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    assert settings.provider_default({"provider": settings.PROVIDER_ANTHROPIC}) == settings.PROVIDER_ANTHROPIC

    monkeypatch.setenv("CRYPT_PROVIDER", settings.PROVIDER_OPENAI)
    assert settings.provider_default({"provider": settings.PROVIDER_ANTHROPIC}) == settings.PROVIDER_OPENAI


def test_startup_choice_does_not_prompt_on_first_run(monkeypatch):
    _clear_provider_env(monkeypatch)

    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(main.sys, "stdin", _Tty())

    provider, model = main._startup_choice({}, _args(), skip=False)

    assert provider == settings.PROVIDER_OLLAMA
    assert model is None


def test_ollama_auth_label_distinguishes_cloud_and_local(monkeypatch):
    _clear_provider_env(monkeypatch)

    cloud = main._provider_auth_label(
        settings.PROVIDER_OLLAMA,
        _args(),
        {"ollama_host": "https://ollama.com"},
        None,
    )
    local = main._provider_auth_label(
        settings.PROVIDER_OLLAMA,
        _args(),
        {"ollama_host": "http://localhost:11434"},
        None,
    )

    assert cloud == "missing OLLAMA_API_KEY"
    assert local == "local Ollama"


def test_doctor_reports_missing_ollama_cloud_key(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("CRYPT_PROVIDER", settings.PROVIDER_OLLAMA)
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")

    check = _check_provider_auth()

    assert check.ok is False
    assert "OLLAMA_API_KEY" in check.detail
    assert "Anthropic login is not used for Ollama" in check.detail
