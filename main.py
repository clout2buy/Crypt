"""crypt - local-first coding harness.

Examples:
    python main.py
    python main.py setup
    python main.py login
    python main.py --provider ollama --model gpt-oss:120b-cloud
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from core import auth, runtime, session as sessions, settings, ui
from core.api import AnthropicProvider, OllamaProvider, OpenAIProvider
from core.loop import run


def main() -> int:
    load_dotenv(Path(__file__).with_name(".env"))
    load_dotenv()
    saved = settings.load_config()

    p = argparse.ArgumentParser(prog="crypt", description="local-first coding harness")
    p.add_argument("--provider", choices=settings.PROVIDERS, help="anthropic oauth or ollama cloud")
    p.add_argument("--model", help="model id; overrides saved/default model")
    p.add_argument("--cwd", help="workspace root for tools")
    p.add_argument("--max-tokens", type=int, help="Anthropic response token cap")
    p.add_argument("--thinking-budget", type=int, help="Anthropic thinking token budget")
    p.add_argument("--ollama-host", help="Ollama host, usually https://ollama.com")
    p.add_argument(
        "--show-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="display the model's raw thinking stream (default: on; pass --no-show-thinking to hide)",
    )
    p.add_argument("--no-thinking", action="store_true", help="disable extended thinking")
    p.add_argument("--no-picker", action="store_true", help="skip startup provider/model picker")
    p.add_argument("--resume", action="store_true", help="resume the latest Crypt session for this workspace")
    p.add_argument("--session", help="resume a specific Crypt session id or title prefix")
    p.add_argument("command", nargs="?", choices=["login", "logout", "setup", "doctor"], help="login | logout | setup | doctor")
    args = p.parse_args()

    if args.command == "login":
        return _do_login()
    if args.command == "logout":
        return _do_logout()
    if args.command == "setup":
        _do_setup(saved, args)
        return 0
    if args.command == "doctor":
        from core.doctor import run_doctor

        print(run_doctor(settings.resolve_workspace(args.cwd, saved)))
        return 0
    did_setup = False
    if _needs_setup(saved, args):
        saved = _do_setup(saved, args)
        did_setup = True

    cwd = settings.resolve_workspace(args.cwd, saved)
    os.environ["CRYPT_ROOT"] = str(cwd)

    try:
        provider_name, model_override = _startup_choice(saved, args, skip=did_setup)
        cred = _credential(provider_name)
        provider = _provider(args, saved, provider_name, cred, model_override)
        _save_runtime_choice(args, saved, provider_name, provider.model, cwd)
        session_obj = _session_for_startup(args, cwd, provider)
        ui.clear_screen()
        _welcome(provider, cred, str(cwd))

        def switch_model(active_provider):
            nonlocal saved, provider_name, model_override, cred, provider

            saved = settings.load_config()
            ui.info("switch provider/model")
            provider_name = _pick(
                "provider",
                [
                    (settings.PROVIDER_ANTHROPIC, "Anthropic OAuth"),
                    (settings.PROVIDER_OPENAI, "OpenAI (or compatible)"),
                    (settings.PROVIDER_OLLAMA, "Ollama Cloud"),
                ],
                getattr(active_provider, "name", provider_name),
            )
            model_override = _pick_model(provider_name, saved)

            if provider_name == settings.PROVIDER_OLLAMA and not os.getenv("OLLAMA_API_KEY"):
                ui.info("OLLAMA_API_KEY is not set; Ollama Cloud calls may fail until it is configured")

            cred = _credential(provider_name)
            if provider_name == settings.PROVIDER_ANTHROPIC and not cred:
                if ui.ask("log in to Anthropic OAuth now?"):
                    _do_login()
                    cred = _credential(provider_name)

            provider = _provider(args, saved, provider_name, cred, model_override)
            _save_runtime_choice(args, saved, provider_name, provider.model, Path(runtime.cwd()))
            ui.status_panel({
                "provider": provider.name,
                "model": provider.model,
                "auth": cred.kind if cred else "none",
                "approval": runtime.approval_label(),
            })
            return provider

        while True:
            result = run(
                provider,
                show_thinking=args.show_thinking,
                cwd=str(cwd),
                model_switcher=switch_model,
                session_obj=session_obj,
            )
            if result == "login":
                _do_login()
            elif result == "logout":
                _do_logout()
            elif result == "model":
                provider = switch_model(provider)
                continue
            else:
                break

            cred = _credential(provider_name)
            provider = _provider(args, settings.load_config(), provider_name, cred, model_override)
            ui.clear_screen()
            _welcome(provider, cred, str(cwd))

    except KeyboardInterrupt:
        print()
        ui.info("interrupted")
        return 130
    except Exception as e:
        ui.error(f"{type(e).__name__}: {e}")
        return 1
    return 0


def _needs_setup(saved: dict, args: argparse.Namespace) -> bool:
    if args.provider or args.model or args.cwd or os.getenv("CRYPT_PROVIDER"):
        return False
    return not saved.get("provider") or not saved.get("workspace")


def _session_for_startup(args: argparse.Namespace, cwd: Path, provider) -> sessions.Session:
    query = args.session or ("latest" if args.resume else None)
    if query:
        info = sessions.find_session(cwd, query)
        if info:
            return sessions.load_session(
                info.cwd or str(cwd),
                info.session_id,
                provider=getattr(provider, "name", ""),
                model=getattr(provider, "model", ""),
            )
        ui.info(f"no matching session for {query!r}; starting a new one")
    return sessions.Session(
        cwd,
        provider=getattr(provider, "name", ""),
        model=getattr(provider, "model", ""),
    )


def _do_setup(saved: dict, args: argparse.Namespace) -> dict:
    ui.info("setup: choose workspace, provider, and model")
    workspace = _ask_workspace(args.cwd, saved)
    provider = args.provider or _pick(
        "provider",
        [
            ("anthropic", "Anthropic OAuth"),
            ("openai", "OpenAI (or compatible)"),
            ("ollama", "Ollama Cloud"),
        ],
        saved.get("provider") or settings.PROVIDER_ANTHROPIC,
    )

    if provider == settings.PROVIDER_ANTHROPIC:
        model = args.model or _pick_model(provider, saved)
        values = {
            "workspace": str(workspace),
            "provider": provider,
            "anthropic_model": model,
        }
    elif provider == settings.PROVIDER_OPENAI:
        model = args.model or _pick_model(provider, saved)
        values = {
            "workspace": str(workspace),
            "provider": provider,
            "openai_model": model,
        }
        if not os.getenv("OPENAI_API_KEY"):
            ui.info("OPENAI_API_KEY is not set; set it before using OpenAI")
    else:
        model = args.model or _pick_model(provider, saved)
        host = settings.client_host(args.ollama_host or saved.get("ollama_host") or settings.OLLAMA_HOST)
        values = {
            "workspace": str(workspace),
            "provider": provider,
            "ollama_model": model,
            "ollama_host": host,
        }
        if not os.getenv("OLLAMA_API_KEY"):
            ui.info("OLLAMA_API_KEY is not set; set it before using Ollama Cloud")

    new_saved = settings.update_config(**values)
    os.environ["CRYPT_ROOT"] = str(workspace)
    ui.info(f"saved setup in {settings.CONFIG_PATH}")

    if provider == settings.PROVIDER_ANTHROPIC and not auth.resolve() and ui.ask("log in to Anthropic OAuth now?"):
        _do_login()
    return new_saved


def _startup_choice(saved: dict, args: argparse.Namespace, skip: bool = False) -> tuple[str, str | None]:
    provider_name = args.provider or settings.provider_default(saved)
    model_override = args.model
    if skip or args.no_picker or args.provider or args.model or not sys.stdin.isatty():
        return provider_name, model_override

    ui.info("choose provider and model")
    provider_name = _pick(
        "provider",
        [
            (settings.PROVIDER_ANTHROPIC, "Anthropic OAuth"),
            (settings.PROVIDER_OLLAMA, "Ollama Cloud"),
        ],
        provider_name,
    )
    model_override = _pick_model(provider_name, saved)
    return provider_name, model_override


def _ask_workspace(cli_root: str | None, saved: dict) -> Path:
    default = settings.resolve_workspace(cli_root, saved)
    raw = input(f"       workspace [{default}]: ").strip()
    path = Path(raw).expanduser() if raw else default
    path = path.resolve()
    if not path.exists():
        if ui.ask(f"create workspace {path}?"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            return _ask_workspace(cli_root, saved)
    if not path.is_dir():
        ui.error(f"not a directory: {path}")
        return _ask_workspace(cli_root, saved)
    return path


def _pick(label: str, options: list[tuple[str, str]], default: str) -> str:
    default_idx = 1
    for i, (value, _) in enumerate(options, 1):
        if value == default:
            default_idx = i
            break
    return ui.splash_choice(label, options, default_idx)


def _pick_model(provider: str, saved: dict) -> str:
    if provider == settings.PROVIDER_ANTHROPIC:
        models = settings.ANTHROPIC_MODELS
    elif provider == settings.PROVIDER_OPENAI:
        models = settings.OPENAI_MODELS
    else:
        models = settings.OLLAMA_MODELS
    default = settings.model_default(provider, saved)
    options = [(m, m) for m in models]
    custom_value = "__custom__"
    if default not in models:
        options.insert(0, (default, f"{default} (saved custom)"))
    options.append((custom_value, "custom model id"))

    choice = _pick("model", options, default if default in [m for m, _ in options] else options[0][0])
    if choice != custom_value:
        return choice

    while True:
        raw = input("       model id: ").strip()
        if raw:
            return raw
        ui.error("model id cannot be empty")


def _credential(provider_name: str) -> auth.Credential | None:
    return auth.resolve() if provider_name == settings.PROVIDER_ANTHROPIC else None


def _provider(
    args: argparse.Namespace,
    saved: dict,
    provider_name: str,
    cred: auth.Credential | None = None,
    model_override: str | None = None,
):
    if provider_name == settings.PROVIDER_ANTHROPIC:
        kwargs: dict = {
            "model": model_override or args.model or settings.model_default(provider_name, saved),
            "max_tokens": args.max_tokens or settings.env_int("ANTHROPIC_MAX_TOKENS", settings.ANTHROPIC_MAX_TOKENS),
            "thinking_budget": 0 if args.no_thinking else (
                args.thinking_budget
                or settings.env_int("ANTHROPIC_THINKING_BUDGET", settings.ANTHROPIC_THINKING_BUDGET)
            ),
        }
        if cred and cred.kind == "oauth":
            kwargs["auth_token"] = cred.token
        return AnthropicProvider(**kwargs)

    if provider_name == settings.PROVIDER_OPENAI:
        return OpenAIProvider(
            model=model_override or args.model or settings.model_default(provider_name, saved),
            max_tokens=args.max_tokens or settings.env_int("OPENAI_MAX_TOKENS", settings.OPENAI_MAX_TOKENS),
            base_url=settings.openai_base_url(saved),
        )

    return OllamaProvider(
        model=model_override or args.model or settings.model_default(provider_name, saved),
        host=settings.ollama_host(args.ollama_host, saved),
        think=not args.no_thinking,
    )


def _save_runtime_choice(
    args: argparse.Namespace,
    saved: dict,
    provider_name: str,
    model: str,
    cwd: Path,
) -> None:
    values: dict[str, object] = {"provider": provider_name, "workspace": str(cwd)}
    if provider_name == settings.PROVIDER_ANTHROPIC:
        values["anthropic_model"] = model
    elif provider_name == settings.PROVIDER_OPENAI:
        values["openai_model"] = model
    else:
        values["ollama_model"] = model
        values["ollama_host"] = settings.ollama_host(args.ollama_host, saved)
    settings.update_config(**values)


def _welcome(provider, cred: auth.Credential | None, cwd: str) -> None:
    ui.welcome(
        provider=provider.name,
        model=provider.model,
        auth_kind=cred.kind if cred else None,
        auth_email=cred.email if cred else None,
        auth_plan=cred.plan if cred else None,
        cwd=cwd,
    )


def _do_login() -> int:
    try:
        from core.oauth import login

        tokens = login(on_status=lambda m: ui.info(m))
        now_ms = int(time.time() * 1000)
        expires_in = tokens.get("expires_in", 3600)
        auth.save({
            "type": "oauth",
            "access": tokens["access_token"],
            "refresh": tokens.get("refresh_token"),
            "expires": now_ms + expires_in * 1000 - 5 * 60 * 1000,
        })
        ui.info("logged in")
        return 0
    except Exception as e:
        ui.error(f"login failed: {e}")
        return 1


def _do_logout() -> int:
    auth.delete()
    ui.info("logged out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
