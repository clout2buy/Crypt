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
        "no_thinking": False,
        "show_thinking": False,
        "max_tokens": None,
        "thinking_budget": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fresh_clone_defaults_to_ollama_without_setup(monkeypatch):
    _clear_provider_env(monkeypatch)

    assert settings.provider_default({}) == settings.PROVIDER_OLLAMA
    assert settings.ollama_host(saved={}) == "http://localhost:11434"
    assert settings.model_default(settings.PROVIDER_OLLAMA, {}) == "gpt-oss:20b"
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
        _args(ollama_host="https://ollama.com"),
        {},
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


def test_saved_cloud_ollama_without_key_falls_back_to_local(monkeypatch):
    _clear_provider_env(monkeypatch)
    saved = {
        "provider": settings.PROVIDER_OLLAMA,
        "ollama_host": "https://ollama.com",
        "ollama_model": "glm-5.1:cloud",
    }

    assert settings.ollama_host(saved=saved) == "http://localhost:11434"
    assert settings.model_default(settings.PROVIDER_OLLAMA, saved) == "gpt-oss:20b"


def test_saved_cloud_ollama_with_key_stays_cloud(monkeypatch):
    _clear_provider_env(monkeypatch)
    monkeypatch.setenv("OLLAMA_API_KEY", "key")
    saved = {
        "provider": settings.PROVIDER_OLLAMA,
        "ollama_host": "https://ollama.com",
        "ollama_model": "glm-5.1:cloud",
    }

    assert settings.ollama_host(saved=saved) == "https://ollama.com"
    assert settings.model_default(settings.PROVIDER_OLLAMA, saved) == "glm-5.1:cloud"


def test_ollama_model_choices_match_host():
    local = settings.ollama_models_for_host("http://localhost:11434")
    cloud = settings.ollama_models_for_host("https://ollama.com")

    assert "gpt-oss:20b" in local
    assert "glm-5.1:cloud" not in local
    assert "glm-5.1:cloud" in cloud
    assert "gpt-oss:20b" not in cloud


def test_cloud_model_routes_to_cloud_host():
    assert settings.ollama_host_for_model("glm-5.1:cloud", "http://localhost:11434") == "https://ollama.com"
    assert settings.ollama_host_for_model("gpt-oss:20b", "https://ollama.com") == "https://ollama.com"


def test_ollama_provider_does_not_think_unless_thinking_is_shown(monkeypatch):
    _clear_provider_env(monkeypatch)

    provider = main._provider(_args(), {}, settings.PROVIDER_OLLAMA)
    shown_provider = main._provider(_args(show_thinking=True), {}, settings.PROVIDER_OLLAMA)

    assert provider._think is False
    assert shown_provider._think is True


def test_provider_routes_cloud_model_to_cloud_host(monkeypatch):
    _clear_provider_env(monkeypatch)

    provider = main._provider(_args(model="glm-5.1:cloud"), {}, settings.PROVIDER_OLLAMA)

    assert provider.model == "glm-5.1:cloud"
    assert provider._base_url == "https://ollama.com"


def test_doctor_reports_missing_ollama_cloud_key(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("CRYPT_PROVIDER", settings.PROVIDER_OLLAMA)
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")

    check = _check_provider_auth()

    assert check.ok is False
    assert "OLLAMA_API_KEY" in check.detail
    assert "Anthropic login is not used for Ollama" in check.detail


def test_doctor_accepts_default_local_ollama_without_key(monkeypatch, tmp_path):
    _clear_provider_env(monkeypatch)
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setenv("CRYPT_PROVIDER", settings.PROVIDER_OLLAMA)

    check = _check_provider_auth()

    assert check.ok is True
    assert "local Ollama" in check.detail
    assert "no key required" in check.detail
