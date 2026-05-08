from __future__ import annotations

import json

from core import gemini_oauth


class _FakeCredentials:
    def to_json(self) -> str:
        return json.dumps({
            "token": "token",
            "refresh_token": "refresh",
        })


class _FakeFlow:
    def __init__(self) -> None:
        self.run_kwargs: dict = {}

    def run_local_server(self, **kwargs):
        self.run_kwargs = kwargs
        return _FakeCredentials()


def test_gemini_login_opens_browser_and_uses_client_project(monkeypatch, tmp_path):
    client_secret = tmp_path / "client_secret.json"
    client_secret.write_text(
        json.dumps({
            "installed": {
                "project_id": "client-project",
                "client_id": "client-id",
                "client_secret": "client-secret",
            },
        }),
        encoding="utf-8",
    )
    fake_flow = _FakeFlow()

    import google_auth_oauthlib.flow

    monkeypatch.setenv("GEMINI_CLIENT_SECRET_FILE", str(client_secret))
    monkeypatch.delenv("GEMINI_PROJECT_ID", raising=False)
    monkeypatch.setattr(gemini_oauth, "APP_DIR", tmp_path)
    monkeypatch.setattr(gemini_oauth, "restrict_file_permissions", lambda path: None)
    monkeypatch.setattr(
        google_auth_oauthlib.flow.InstalledAppFlow,
        "from_client_secrets_file",
        staticmethod(lambda path, scopes: fake_flow),
    )

    result = gemini_oauth.login()

    assert result["project_id"] == "client-project"
    assert result["credentials"]["project_id"] == "client-project"
    assert result["credentials"]["scopes"] == list(gemini_oauth.GEMINI_OAUTH_SCOPES)
    assert fake_flow.run_kwargs["open_browser"] is True
    assert fake_flow.run_kwargs["access_type"] == "offline"
    saved = json.loads((tmp_path / "gemini_token.json").read_text(encoding="utf-8"))
    assert saved["project_id"] == "client-project"
