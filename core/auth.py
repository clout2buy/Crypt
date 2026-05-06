"""Credential storage and resolution."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .settings import AUTH_PATH, restrict_file_permissions


@dataclass
class Credential:
    kind: str
    token: str
    email: str | None = None
    plan: str | None = None


def _auth_path() -> Path:
    return AUTH_PATH


def load() -> dict | None:
    path = _auth_path()
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save(data: dict) -> None:
    path = _auth_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)
    restrict_file_permissions(path)


def delete() -> None:
    path = _auth_path()
    if path.exists():
        path.unlink()


def resolve() -> Credential | None:
    """Resolve the best available credential. OAuth first, then env var."""
    stored = load()
    if stored and stored.get("type") == "oauth":
        access = stored.get("access")
        refresh_tok = stored.get("refresh")
        expires = stored.get("expires", 0)

        now_ms = int(time.time() * 1000)
        if now_ms >= expires and refresh_tok:
            try:
                from .oauth import refresh
                tokens = refresh(refresh_tok)
                access = tokens["access_token"]
                new_refresh = tokens.get("refresh_token", refresh_tok)
                expires_in = tokens.get("expires_in", 3600)
                save({
                    "type": "oauth",
                    "access": access,
                    "refresh": new_refresh,
                    "expires": now_ms + expires_in * 1000 - 5 * 60 * 1000,
                    "email": stored.get("email"),
                    "plan": stored.get("plan"),
                })
            except Exception:
                return None

        if access:
            return Credential(
                kind="oauth",
                token=access,
                email=stored.get("email"),
                plan=stored.get("plan"),
            )

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        kind = "oauth" if api_key.startswith("sk-ant-oat") else "api"
        return Credential(kind=kind, token=api_key)

    return None
