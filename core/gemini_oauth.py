"""Google OAuth helper for Gemini API credentials."""
from __future__ import annotations

import json
import os
from pathlib import Path

from .settings import APP_DIR, GEMINI_OAUTH_SCOPES, restrict_file_permissions


def client_secret_path() -> Path:
    explicit = os.getenv("GEMINI_CLIENT_SECRET_FILE")
    candidates = [
        Path(explicit).expanduser() if explicit else None,
        APP_DIR / "gemini_client_secret.json",
        Path.cwd() / "client_secret.json",
    ]
    for item in candidates:
        if item and item.exists():
            return item.resolve()
    return APP_DIR / "gemini_client_secret.json"


def _client_secret_project_id(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    for key in ("installed", "web"):
        section = data.get(key)
        if isinstance(section, dict) and section.get("project_id"):
            return str(section["project_id"])
    return str(data.get("project_id") or "")


def login(on_status=None) -> dict:
    """Run Google's desktop OAuth flow and return serializable credentials."""
    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except Exception as exc:  # pragma: no cover - exercised in CLI smoke only
        raise RuntimeError(
            "Gemini OAuth requires google-auth-oauthlib. "
            "Install dependencies with `python -m pip install -e .[dev]`."
        ) from exc

    path = client_secret_path()
    if not path.exists():
        raise FileNotFoundError(
            "Gemini OAuth needs a Google desktop OAuth client secret JSON. "
            f"Set GEMINI_CLIENT_SECRET_FILE or place it at {path}."
        )
    if on_status:
        on_status(f"opening Google OAuth flow using {path}")
    flow = InstalledAppFlow.from_client_secrets_file(str(path), scopes=list(GEMINI_OAUTH_SCOPES))
    creds = flow.run_local_server(
        port=0,
        open_browser=True,
        prompt="consent",
        access_type="offline",
    )
    data = json.loads(creds.to_json())
    data["scopes"] = list(GEMINI_OAUTH_SCOPES)
    project_id = os.getenv("GEMINI_PROJECT_ID") or _client_secret_project_id(path) or data.get("quota_project_id") or ""
    if project_id:
        data["project_id"] = project_id
    token_path = APP_DIR / "gemini_token.json"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    restrict_file_permissions(token_path)
    return {
        "credentials": data,
        "project_id": project_id,
    }
