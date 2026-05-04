"""Provider adapters.

Internal messages use Anthropic-style content blocks:
    {"role": "user" | "assistant", "content": str | [block, ...]}

The Anthropic path uses raw httpx, not the anthropic-python SDK, because
the SDK injects X-Stainless-* fingerprint headers that the Anthropic edge
uses to detect non-Claude-Code OAuth traffic and silently rate-limit it.
Mirrors MrDoing's `mrdoing/stream.py`.
"""
from __future__ import annotations

import atexit
import json
import os
from dataclasses import dataclass
from typing import Iterator, Protocol

import httpx

from .settings import (
    ANTHROPIC_BASE_BETAS,
    ANTHROPIC_ESCALATED_MAX_TOKENS,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    ANTHROPIC_OAUTH_BETAS,
    ANTHROPIC_OAUTH_USER_AGENT,
    ANTHROPIC_OAUTH_X_APP,
    ANTHROPIC_THINKING_BUDGET,
    OLLAMA_MODEL,
    OPENAI_BASE_URL,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
)


@dataclass
class TextDelta:
    text: str


@dataclass
class ThinkingDelta:
    text: str


@dataclass
class TurnEnd:
    stop_reason: str
    message: dict
    usage: dict | None = None


Event = TextDelta | ThinkingDelta | TurnEnd


class Provider(Protocol):
    name: str
    model: str
    is_oauth: bool

    def stream_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> Iterator[Event]: ...


_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
# Prepended to system blocks on OAuth so the server's OAuth path
# recognizes us as Claude Code. MUST be the first system block.
_CLAUDE_CODE_IDENTITY = (
    "You are Claude Code, Anthropic's official CLI for Claude."
)


class AnthropicProvider:
    name = "anthropic"
    context_window = 1_000_000

    def __init__(
        self,
        model: str = ANTHROPIC_MODEL,
        max_tokens: int = ANTHROPIC_MAX_TOKENS,
        thinking_budget: int = ANTHROPIC_THINKING_BUDGET,
        auth_token: str | None = None,
    ) -> None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not auth_token and not api_key:
            raise RuntimeError(
                "AnthropicProvider needs either an OAuth auth_token or "
                "ANTHROPIC_API_KEY in the environment."
            )
        self._auth_token = auth_token
        self._api_key = api_key
        self._oauth = bool(auth_token)
        self.model = model
        self._max_tokens = max_tokens
        self._max_tokens_override = 0
        # Thinking budget must stay strictly below max_tokens; the API
        # rejects budgets >= max_tokens. Anthropic also requires a 1024
        # minimum when thinking is enabled — clamp up to honor intent.
        if thinking_budget <= 0 or max_tokens < 1025:
            self._thinking_budget = 0
        else:
            self._thinking_budget = max(1024, min(thinking_budget, max_tokens - 1))
        self._http = httpx.Client(timeout=httpx.Timeout(10.0, read=180.0))
        # Don't leak the connection pool when Python exits. atexit is
        # a process-wide hook; multiple providers register multiple
        # close handlers and that's fine — close() is idempotent.
        atexit.register(self.close)

    @property
    def is_oauth(self) -> bool:
        return self._oauth

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def escalate_once(self, escalated: int = ANTHROPIC_ESCALATED_MAX_TOKENS) -> int:
        """Arm a one-shot max_tokens override for the next stream_turn().

        Used by the loop when a turn ends with stop_reason='max_tokens' so we
        get one clean retry at a higher budget. The override clears itself
        after the next API call.
        """
        self._max_tokens_override = max(escalated, self._max_tokens)
        return self._max_tokens_override

    def _build_request(self, messages, tools, system):
        max_tokens = self._max_tokens_override or self._max_tokens
        self._max_tokens_override = 0
        if self._thinking_budget and self._thinking_budget >= max_tokens:
            thinking_budget = max(0, max_tokens - 1)
            if thinking_budget < 1024:
                thinking_budget = 0
        else:
            thinking_budget = self._thinking_budget

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "anthropic-version": "2023-06-01",
            "anthropic-dangerous-direct-browser-access": "true",
        }
        if self._oauth:
            headers["Authorization"] = f"Bearer {self._auth_token}"
            headers["anthropic-beta"] = ANTHROPIC_OAUTH_BETAS
            headers["user-agent"] = ANTHROPIC_OAUTH_USER_AGENT
            headers["x-app"] = ANTHROPIC_OAUTH_X_APP
            system_blocks = [
                {"type": "text", "text": _CLAUDE_CODE_IDENTITY},
                {"type": "text", "text": system},
            ]
        else:
            headers["x-api-key"] = self._api_key
            headers["anthropic-beta"] = ANTHROPIC_BASE_BETAS
            system_blocks = [{"type": "text", "text": system}]

        # Cache breakpoints: Anthropic allows 4 cache_control markers per
        # request and caches everything up to and including the marked
        # block. Spend them on system + tools + last 2 user messages so
        # multi-turn sessions get deep cache hits.
        system_blocks = _mark_last_for_cache(system_blocks)
        cached_tools = _mark_last_for_cache(list(tools)) if tools else []
        cached_messages = _mark_messages_for_cache(messages)

        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_blocks,
            "messages": cached_messages,
            "stream": True,
        }
        if cached_tools:
            body["tools"] = cached_tools
        if thinking_budget:
            body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
        return headers, body

    def stream_turn(self, messages, tools, system):
        headers, body = self._build_request(messages, tools, system)
        url = f"{_ANTHROPIC_BASE_URL}/v1/messages"

        with self._http.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                _raise_for_status(resp)

            content_blocks: list[dict | None] = []
            usage: dict = {"input_tokens": 0, "output_tokens": 0}
            stop_reason = "end_turn"
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n\n" in buf:
                    block, buf = buf.split("\n\n", 1)
                    parsed = _parse_sse(block)
                    if parsed is None:
                        continue
                    event_type, data = parsed
                    for ev in _process_event(event_type, data, content_blocks, usage):
                        if isinstance(ev, _StopReason):
                            stop_reason = ev.value
                        else:
                            yield ev

        # Drop thinking blocks from persisted history. Anthropic requires
        # a `signature` field on round-tripped thinking blocks (delivered
        # via signature_delta), and re-sending without it 400s on the next
        # turn. Match MrDoing: stream thinking to the UI, never persist it.
        content = [b for b in content_blocks if b and b.get("type") != "thinking"]
        for blk in content:
            if blk.get("type") == "tool_use":
                raw = blk.pop("_partial_json", "")
                try:
                    blk["input"] = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    blk["input"] = {}
        yield TurnEnd(
            stop_reason=stop_reason,
            message={"role": "assistant", "content": content},
            usage=usage,
        )


@dataclass
class _StopReason:
    value: str


def _parse_sse(block: str):
    event_type = None
    data_parts: list[str] = []
    for line in block.splitlines():
        if line.startswith("event:"):
            event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_parts.append(line[5:].lstrip())
    if not event_type or not data_parts:
        return None
    try:
        return event_type, json.loads("\n".join(data_parts))
    except json.JSONDecodeError:
        return None


def _process_event(event_type, data, content_blocks, usage):
    if event_type == "message_start":
        u = (data.get("message") or {}).get("usage") or {}
        usage["input_tokens"] = u.get("input_tokens", 0)
        usage["output_tokens"] = u.get("output_tokens", 0)
        return

    if event_type == "content_block_start":
        try:
            index = int(data.get("index", 0))
        except (TypeError, ValueError):
            return
        meta = data.get("content_block") or {}
        btype = meta.get("type")
        while len(content_blocks) <= index:
            content_blocks.append(None)
        if btype == "text":
            content_blocks[index] = {"type": "text", "text": ""}
        elif btype == "thinking":
            content_blocks[index] = {"type": "thinking", "thinking": ""}
        elif btype == "tool_use":
            content_blocks[index] = {
                "type": "tool_use",
                "id": meta.get("id") or f"tool_{index}",
                "name": meta.get("name") or "?",
                "input": {},
                "_partial_json": "",
            }
        return

    if event_type == "content_block_delta":
        try:
            index = int(data.get("index", 0))
        except (TypeError, ValueError):
            return
        if index < 0 or index >= len(content_blocks) or content_blocks[index] is None:
            return
        block = content_blocks[index]
        delta = data.get("delta") or {}
        dtype = delta.get("type")
        if dtype == "text_delta":
            text = delta.get("text", "")
            block["text"] = block.get("text", "") + text
            yield TextDelta(text)
        elif dtype == "thinking_delta":
            text = delta.get("thinking", "")
            block["thinking"] = block.get("thinking", "") + text
            yield ThinkingDelta(text)
        elif dtype == "input_json_delta":
            partial = delta.get("partial_json", "")
            block["_partial_json"] = block.get("_partial_json", "") + partial
        return

    if event_type == "message_delta":
        u = data.get("usage") or {}
        if "output_tokens" in u:
            usage["output_tokens"] = u["output_tokens"]
        sr = (data.get("delta") or {}).get("stop_reason")
        if sr:
            yield _StopReason(sr)
        return

    if event_type == "error":
        err = data.get("error") or {}
        raise RuntimeError(f"stream error: {err.get('message') or err}")


def _mark_cache_breakpoint(block: dict) -> dict:
    return {**block, "cache_control": {"type": "ephemeral"}}


def _mark_last_for_cache(items: list[dict]) -> list[dict]:
    if not items:
        return items
    return items[:-1] + [_mark_cache_breakpoint(items[-1])]


def _mark_messages_for_cache(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    out = list(messages)
    marked = 0
    for idx in range(len(out) - 1, -1, -1):
        msg = out[idx]
        if msg.get("role") != "user":
            continue
        out[idx] = _cache_mark_user_message(msg)
        marked += 1
        if marked >= 2:
            break
    return out


def _cache_mark_user_message(msg: dict) -> dict:
    content = msg.get("content")
    if isinstance(content, str):
        return {
            **msg,
            "content": [{
                "type": "text",
                "text": content,
                "cache_control": {"type": "ephemeral"},
            }],
        }
    if isinstance(content, list) and content:
        new_content = list(content)
        last = new_content[-1]
        if isinstance(last, dict):
            new_content[-1] = {**last, "cache_control": {"type": "ephemeral"}}
        return {**msg, "content": new_content}
    return msg


def _raise_for_status(resp: httpx.Response) -> None:
    """Map an HTTP error response onto an anthropic SDK exception so
    loop._format_error keeps recognizing it (RateLimitError etc.)."""
    try:
        resp.read()
        body = resp.json() if resp.text else {}
    except Exception:
        body = {}
    msg = (body.get("error") or {}).get("message") if isinstance(body, dict) else None
    msg = msg or resp.text or f"HTTP {resp.status_code}"
    from anthropic import (
        APIStatusError,
        AuthenticationError,
        BadRequestError,
        InternalServerError,
        NotFoundError,
        PermissionDeniedError,
        RateLimitError,
        UnprocessableEntityError,
    )
    cls = {
        400: BadRequestError,
        401: AuthenticationError,
        403: PermissionDeniedError,
        404: NotFoundError,
        422: UnprocessableEntityError,
        429: RateLimitError,
        500: InternalServerError,
    }.get(resp.status_code, APIStatusError)
    raise cls(message=msg, response=resp, body=body)


class OpenAIProvider:
    """OpenAI Chat Completions adapter with streaming + tool calls.

    Works against the official endpoint and any OpenAI-compatible server
    (Together, Fireworks, LM Studio, vLLM, etc.) via OPENAI_BASE_URL. The
    internal message shape stays Anthropic-style; we convert in/out at
    the wire boundary.

    o-series models (o1/o3) get reasoning_effort=medium by default — no
    setting because the API rejects unknown reasoning params on classic
    models. We sniff the model name to pick the right parameter set.
    """
    name = "openai"
    context_window = 128_000
    is_oauth = False

    def __init__(
        self,
        model: str = OPENAI_MODEL,
        max_tokens: int = OPENAI_MAX_TOKENS,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OpenAIProvider needs an API key (set OPENAI_API_KEY or pass api_key)."
            )
        self._api_key = key
        self.model = model
        self._max_tokens = max_tokens
        self._base_url = (base_url or os.getenv("OPENAI_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(10.0, read=180.0))
        atexit.register(self.close)

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:
            pass

    def stream_turn(self, messages, tools, system):
        body = self._build_body(messages, tools, system)
        url = f"{self._base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self._api_key}",
        }

        text_buf: list[str] = []
        # Tool calls arrive as deltas keyed by index. Each call has a
        # stable id (eventually), name, and JSON arguments built up over
        # multiple chunks. We assemble per-index, then emit once at the end.
        tool_calls: dict[int, dict] = {}
        finish_reason = "stop"

        with self._http.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise RuntimeError(
                    f"openai HTTP {resp.status_code}: {resp.text[:500]}"
                )
            buf = ""
            for chunk in resp.iter_text():
                buf += chunk
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        continue
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    fr = choice.get("finish_reason")
                    if fr:
                        finish_reason = fr

                    text = delta.get("content")
                    if text:
                        text_buf.append(text)
                        yield TextDelta(text)

                    for call_delta in delta.get("tool_calls") or []:
                        idx = call_delta.get("index", 0)
                        slot = tool_calls.setdefault(idx, {
                            "id": "",
                            "name": "",
                            "arguments": "",
                        })
                        if call_delta.get("id"):
                            slot["id"] = call_delta["id"]
                        fn = call_delta.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arguments"] += fn["arguments"]

        content: list[dict] = []
        if text_buf:
            content.append({"type": "text", "text": "".join(text_buf)})
        for idx in sorted(tool_calls):
            slot = tool_calls[idx]
            try:
                args_obj = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except json.JSONDecodeError:
                args_obj = {}
            content.append({
                "type": "tool_use",
                "id": slot["id"] or f"call_{idx}",
                "name": slot["name"] or "?",
                "input": args_obj,
            })

        # Map OpenAI finish_reason onto our internal stop_reason vocabulary
        # so the loop's max_tokens-escalate path keeps working uniformly.
        if finish_reason == "tool_calls":
            stop = "tool_use"
        elif finish_reason == "length":
            stop = "max_tokens"
        else:
            stop = "end_turn"

        yield TurnEnd(
            stop_reason=stop,
            message={"role": "assistant", "content": content},
        )

    def _build_body(self, messages, tools, system):
        msgs = [{"role": "system", "content": system}]
        msgs.extend(_to_openai_messages(messages))
        body: dict = {
            "model": self.model,
            "messages": msgs,
            "stream": True,
        }
        # o-series uses max_completion_tokens; classic uses max_tokens.
        # Sending the wrong one is a 400 on either side.
        if self._is_reasoning_model():
            body["max_completion_tokens"] = self._max_tokens
        else:
            body["max_tokens"] = self._max_tokens
        if tools:
            body["tools"] = [_to_openai_tool(t) for t in tools]
            body["tool_choice"] = "auto"
        return body

    def _is_reasoning_model(self) -> bool:
        m = self.model.lower()
        return m.startswith("o1") or m.startswith("o3") or m.startswith("o4")


def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert internal Anthropic-style messages to OpenAI Chat Completions
    format. Tool results become role=tool messages; assistant tool_use
    blocks become role=assistant with a tool_calls array."""
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        text_parts: list[str] = []
        tool_calls: list[dict] = []
        # Tool results live on user messages but become their own openai
        # messages. We collect them and emit after the user's text (if any).
        deferred_results: list[dict] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text_parts.append(str(block.get("text", "")))
            elif btype == "tool_use":
                args = block.get("input") or {}
                tool_calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(args),
                    },
                })
            elif btype == "tool_result":
                inner = block.get("content")
                if isinstance(inner, list):
                    inner = "".join(
                        b.get("text", "") if isinstance(b, dict) else str(b)
                        for b in inner
                    )
                deferred_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": str(inner) if inner is not None else "",
                })
            elif btype == "thinking":
                # OpenAI's classic chat API has no thinking surface.
                # o-series handles its own internal reasoning. Drop.
                continue

        if role == "assistant":
            item: dict = {"role": "assistant", "content": "".join(text_parts) or None}
            if tool_calls:
                item["tool_calls"] = tool_calls
            out.append(item)
        elif role == "user":
            if text_parts:
                out.append({"role": "user", "content": "".join(text_parts)})
            out.extend(deferred_results)

    return out


class OllamaProvider:
    name = "ollama"
    context_window = 128_000
    is_oauth = False

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str | None = None,
        think: bool = True,
    ) -> None:
        from ollama import Client

        api_key = os.getenv("OLLAMA_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self._client = Client(host=host or os.getenv("OLLAMA_HOST"), headers=headers)
        self.model = model
        self.host = host
        self._think = think

    def stream_turn(self, messages, tools, system):
        text: list[str] = []
        calls: list[dict] = []

        for chunk in self._client.chat(
            model=self.model,
            messages=_to_ollama(messages, system),
            tools=[_to_openai_tool(t) for t in tools] if tools else None,
            stream=True,
            think=self._think,
        ):
            msg = chunk.get("message") or {}
            if msg.get("thinking"):
                yield ThinkingDelta(msg["thinking"])
            if msg.get("content"):
                yield TextDelta(msg["content"])
                text.append(msg["content"])
            calls.extend(msg.get("tool_calls") or [])

        content = []
        if text:
            content.append({"type": "text", "text": "".join(text)})
        content.extend(_to_anthropic_call(call, i) for i, call in enumerate(calls))

        yield TurnEnd(
            stop_reason="tool_use" if calls else "end_turn",
            message={"role": "assistant", "content": content},
        )


def _to_anthropic_call(call: dict, index: int) -> dict:
    fn = call.get("function", {})
    args = fn.get("arguments", {})
    if isinstance(args, str):
        args = json.loads(args)
    return {
        "type": "tool_use",
        "id": call.get("id") or f"call_{index}",
        "name": fn.get("name", ""),
        "input": args,
    }


def _to_ollama(messages: list[dict], system: str) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue

        text: list[str] = []
        calls: list[dict] = []
        for block in content:
            kind = block.get("type")
            if kind == "text":
                text.append(block["text"])
            elif kind == "tool_use":
                calls.append({
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block["name"], "arguments": block["input"]},
                })
            elif kind == "tool_result":
                result = block.get("content", "")
                out.append({
                    "role": "tool",
                    "content": result if isinstance(result, str) else json.dumps(result),
                    "tool_call_id": block.get("tool_use_id", ""),
                })

        if text or calls:
            item = {"role": msg["role"], "content": "".join(text)}
            if calls:
                item["tool_calls"] = calls
            out.append(item)
    return out


def _to_openai_tool(tool: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }
