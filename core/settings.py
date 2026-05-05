"""Small shared settings layer for crypt.

This is intentionally boring: one place for defaults, OAuth constants,
saved preferences, and workspace resolution. Runtime code should import
from here instead of copying constants across modules.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


APP_DIR = Path.home() / ".crypt"
AUTH_PATH = APP_DIR / "auth.json"
CONFIG_PATH = APP_DIR / "config.json"

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OLLAMA = "ollama"
PROVIDER_OPENAI = "openai"
PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OLLAMA, PROVIDER_OPENAI)

ANTHROPIC_MODEL = "claude-opus-4-7"
# Anthropic counts the full requested max_tokens against per-minute rate
# limits BEFORE the response is generated, so reserving more is not free.
# 8k is roughly 2x the p99 of real coding-agent responses; turns that need
# more get one auto-retry at ESCALATED_MAX_TOKENS instead of paying the
# rate-limit tax on every single request. Mirrors Claude Code's
# CAPPED_DEFAULT_MAX_TOKENS / ESCALATED_MAX_TOKENS pattern.
ANTHROPIC_MAX_TOKENS = 8_000
ANTHROPIC_ESCALATED_MAX_TOKENS = 32_000
# Anthropic requires `thinking.budget_tokens >= 1024` when thinking is enabled
# AND `budget_tokens < max_tokens`. 4k leaves ~4k for the response itself
# under the default cap, which fits the median real reply.
ANTHROPIC_THINKING_BUDGET = 4_000
ANTHROPIC_MODELS = (
    "claude-opus-4-7",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
)

OPENAI_MODEL = "gpt-5"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_MAX_TOKENS = 8_000
OPENAI_MODELS = (
    "gpt-5",
    "gpt-5-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "o3",
    "o3-mini",
)

OLLAMA_MODEL = "gpt-oss:120b-cloud"
OLLAMA_HOST = "https://ollama.com"
OLLAMA_MODELS = (
    "gpt-oss:120b-cloud",
    "gpt-oss:20b-cloud",
    "qwen3-coder:480b-cloud",
    "qwen3-coder:30b-cloud",
    "kimi-k2:latest",
    "kimi-k2:cloud",
    "kimi-k2.6:cloud",
    "deepseek-v3.1:cloud",
    "glm-4.6:cloud",
    "llama3.3:70b-cloud",
)

CRYPT_VERSION = "0.3.0"
CRYPT_IDENTITY = "You are Crypt, a local-first software engineering agent."
# Anthropic's OAuth tokens are issued to Claude Code's app id; the edge
# rate-limits OAuth traffic that doesn't present these exact headers
# (429 with no anthropic-ratelimit-* headers). Bump version with the real CLI.
ANTHROPIC_OAUTH_USER_AGENT = "claude-cli/2.1.75"
ANTHROPIC_OAUTH_X_APP = "cli"
ANTHROPIC_BASE_BETAS = (
    "fine-grained-tool-streaming-2025-05-14,"
    "interleaved-thinking-2025-05-14"
)
ANTHROPIC_OAUTH_BETAS = (
    f"claude-code-20250219,oauth-2025-04-20,{ANTHROPIC_BASE_BETAS}"
)

OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
OAUTH_AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
OAUTH_TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
OAUTH_CALLBACK_PORT = 53692
OAUTH_CALLBACK_PATH = "/callback"
OAUTH_REDIRECT_URI = f"http://localhost:{OAUTH_CALLBACK_PORT}{OAUTH_CALLBACK_PATH}"
OAUTH_SCOPES = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)
OAUTH_TIMEOUT = 300


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(data: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(CONFIG_PATH)
    if os.name != "nt":
        os.chmod(CONFIG_PATH, 0o600)


def update_config(**values: object) -> dict:
    data = load_config()
    data.update({k: v for k, v in values.items() if v is not None})
    save_config(data)
    return data


def env(name: str, default: str) -> str:
    return os.getenv(name) or default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def provider_default(saved: dict | None = None) -> str:
    saved = saved or load_config()
    value = os.getenv("CRYPT_PROVIDER") or saved.get("provider") or PROVIDER_ANTHROPIC
    return value if value in PROVIDERS else PROVIDER_ANTHROPIC


def model_default(provider: str, saved: dict | None = None) -> str:
    saved = saved or load_config()
    if provider == PROVIDER_ANTHROPIC:
        return os.getenv("ANTHROPIC_MODEL") or saved.get("anthropic_model") or ANTHROPIC_MODEL
    if provider == PROVIDER_OPENAI:
        return os.getenv("OPENAI_MODEL") or saved.get("openai_model") or OPENAI_MODEL
    return os.getenv("OLLAMA_MODEL") or saved.get("ollama_model") or OLLAMA_MODEL


def openai_base_url(saved: dict | None = None) -> str:
    """Returns the configured OpenAI base URL (env > saved > default).
    Lets users point at OpenAI-compatible endpoints (Together, Fireworks,
    LM Studio, vLLM, etc.) without code changes."""
    saved = saved or load_config()
    return os.getenv("OPENAI_BASE_URL") or saved.get("openai_base_url") or OPENAI_BASE_URL


def ollama_host(cli_host: str | None = None, saved: dict | None = None) -> str:
    saved = saved or load_config()
    return client_host(cli_host or os.getenv("OLLAMA_HOST") or saved.get("ollama_host") or OLLAMA_HOST)


def client_host(host: str) -> str:
    host = host.strip()
    raw = host if "://" in host else f"http://{host}"
    parts = urlsplit(raw)
    if parts.hostname in {"0.0.0.0", "::", "[::]"}:
        port = parts.port or 11434
        return urlunsplit((parts.scheme or "http", f"127.0.0.1:{port}", "", "", ""))
    return host


def resolve_workspace(cli_root: str | None = None, saved: dict | None = None) -> Path:
    saved = saved or load_config()
    explicit = cli_root or os.getenv("CRYPT_ROOT")
    if explicit:
        return _valid_workspace(Path(explicit))

    saved_root = saved.get("workspace")
    if saved_root:
        saved_path = Path(str(saved_root)).expanduser()
        if saved_path.exists() and saved_path.is_dir():
            return saved_path.resolve()

    cwd = Path.cwd().resolve()
    if not _is_bad_launch_cwd(cwd):
        return cwd

    home = Path.home().resolve()
    return home if home.exists() else cwd


def _valid_workspace(path: Path) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"workspace does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"workspace is not a directory: {path}")
    return path


def _is_bad_launch_cwd(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    text = str(path).lower()
    return (
        "windows" in parts and "system32" in parts
    ) or "\\program files\\windowsapps" in text
