from __future__ import annotations

from core import app_daemon, auth, runtime, settings
from core.api import OllamaProvider


def test_app_daemon_snapshot_uses_shared_provider_inventory(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(settings, "AUTH_PATH", tmp_path / "auth.json")
    monkeypatch.setattr(auth, "AUTH_PATH", tmp_path / "auth.json")
    monkeypatch.delenv("CRYPT_PROVIDER", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    daemon = app_daemon.AppDaemon(emit=lambda event: None, cwd=str(tmp_path))
    snapshot = daemon.snapshot()

    assert snapshot["workspace"] == str(tmp_path)
    assert snapshot["provider"] == settings.PROVIDER_OLLAMA
    assert snapshot["approval"] == runtime.approval_label()
    gemini = next(provider for provider in snapshot["providers"] if provider["id"] == settings.PROVIDER_GEMINI)
    assert gemini["status"] == "construction"
    assert snapshot["routes"][0]["role"] == "planner"


def test_app_daemon_set_approval_emits_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    events: list[dict] = []
    previous = runtime.approval_mode()
    daemon = app_daemon.AppDaemon(emit=events.append, cwd=str(tmp_path))

    try:
        daemon.handle_command({"type": "setApproval", "mode": runtime.APPROVAL_ALL, "id": "cmd-1"})

        assert runtime.approval_mode() == runtime.APPROVAL_ALL
        assert events[-1]["event"] == "snapshot"
        assert events[-1]["id"] == "cmd-1"
        assert events[-1]["snapshot"]["approval"] == "yolo-all"
    finally:
        runtime.set_approval_mode(previous)


def test_app_daemon_sets_provider_and_model(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(settings, "AUTH_PATH", tmp_path / "auth.json")
    monkeypatch.setattr(auth, "AUTH_PATH", tmp_path / "auth.json")
    events: list[dict] = []
    daemon = app_daemon.AppDaemon(emit=events.append, cwd=str(tmp_path))

    daemon.handle_command(
        {
            "type": "setProviderModel",
            "provider": settings.PROVIDER_OPENAI_CODEX,
            "model": "gpt-5.5",
            "id": "engine-1",
        }
    )

    saved = settings.load_config()
    assert saved["provider"] == settings.PROVIDER_OPENAI_CODEX
    assert saved["openai_codex_model"] == "gpt-5.5"
    assert events[-1]["event"] == "snapshot"
    assert events[-1]["snapshot"]["provider"] == settings.PROVIDER_OPENAI_CODEX
    assert events[-1]["snapshot"]["model"] == "gpt-5.5"


def test_app_daemon_persists_route_matrix(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    events: list[dict] = []
    daemon = app_daemon.AppDaemon(emit=events.append, cwd=str(tmp_path))

    daemon.handle_command(
        {
            "type": "setRoute",
            "role": "builder",
            "provider": settings.PROVIDER_OLLAMA,
            "model": "qwen2.5-coder:14b",
            "id": "route-1",
        }
    )

    saved = settings.load_config()
    builder = next(route for route in saved["desktop_routes"] if route["role"] == "builder")
    assert builder["provider"] == settings.PROVIDER_OLLAMA
    assert builder["model"] == "qwen2.5-coder:14b"
    snapshot_builder = next(route for route in events[-1]["snapshot"]["routes"] if route["role"] == "builder")
    assert snapshot_builder["model"] == "qwen2.5-coder:14b"


def test_app_daemon_route_factory_maps_worker_to_builder(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    settings.update_config(
        desktop_routes=[
            {
                "role": "builder",
                "provider": settings.PROVIDER_OLLAMA,
                "model": "qwen2.5-coder:14b",
                "status": "active",
            }
        ]
    )

    provider = app_daemon._provider_for_route(
        settings.load_config(),
        "worker",
        fallback=OllamaProvider(model="fallback", host="http://localhost:11434"),
    )

    assert provider.name == settings.PROVIDER_OLLAMA
    assert provider.model == "qwen2.5-coder:14b"


def test_app_daemon_timeline_events_render_tool_cards():
    events = app_daemon._timeline_events(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "ok",
                        "is_error": False,
                    }
                ],
            },
        ]
    )

    assert events[0]["event"] == "toolCall"
    assert events[0]["tool"] == "read_file"
    assert "README.md" in events[0]["text"]
    assert events[1]["event"] == "toolResult"
    assert events[1]["ok"] is True


def test_desktop_ollama_cloud_selection_uses_cloud_host():
    assert app_daemon._desktop_ollama_host("glm-5.1:cloud", {}) == "https://ollama.com"
    assert app_daemon._desktop_ollama_host("qwen2.5-coder:14b", {}) == settings.OLLAMA_HOST


def test_app_daemon_routes_slash_status_to_command_result(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    events: list[dict] = []
    daemon = app_daemon.AppDaemon(emit=events.append, cwd=str(tmp_path))

    daemon.handle_command({"type": "sendPrompt", "text": "/status", "id": "slash-1"})

    assert events[0]["event"] == "commandResult"
    assert events[0]["id"] == "slash-1"
    assert "Crypt desktop status" in events[0]["text"]
    assert all(event["event"] != "taskStarted" for event in events)


def test_app_daemon_rejects_empty_prompt(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CONFIG_PATH", tmp_path / "config.json")
    events: list[dict] = []
    daemon = app_daemon.AppDaemon(emit=events.append, cwd=str(tmp_path))

    daemon.handle_command({"type": "sendPrompt", "text": " ", "id": "cmd-2"})

    assert events[-1]["event"] == "error"
    assert events[-1]["id"] == "cmd-2"
    assert "empty" in events[-1]["error"]
