"""Provider adapters.

Internal messages use Anthropic-style content blocks:
    {"role": "user" | "assistant", "content": str | [block, ...]}
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator, Protocol

from .settings import (
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    ANTHROPIC_OAUTH_BETAS,
    ANTHROPIC_THINKING_BUDGET,
    CLAUDE_CODE_IDENTITY,
    CLAUDE_CODE_VERSION,
    OLLAMA_MODEL,
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

    def stream_turn(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str,
    ) -> Iterator[Event]: ...


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
        from anthropic import Anthropic

        kwargs: dict = {}
        if auth_token:
            kwargs["auth_token"] = auth_token
            kwargs["default_headers"] = {
                "anthropic-beta": ANTHROPIC_OAUTH_BETAS,
                "anthropic-dangerous-direct-browser-access": "true",
                "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                "x-app": "cli",
            }
        self._client = Anthropic(**kwargs)
        if auth_token:
            self._client.api_key = None
        self._oauth = bool(auth_token)
        self.model = model
        self._max_tokens = max_tokens
        self._thinking_budget = min(thinking_budget, max(0, max_tokens - 1))

    def stream_turn(self, messages, tools, system):
        kwargs = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": messages,
        }
        if self._oauth:
            kwargs["system"] = [
                {"type": "text", "text": CLAUDE_CODE_IDENTITY},
                {"type": "text", "text": system},
            ]
        if tools:
            kwargs["tools"] = tools
        if self._thinking_budget:
            kwargs["thinking"] = {"type": "enabled", "budget_tokens": self._thinking_budget}

        with self._client.messages.stream(**kwargs) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                if delta.type == "text_delta":
                    yield TextDelta(delta.text)
                elif delta.type == "thinking_delta":
                    yield ThinkingDelta(delta.thinking)

            final = stream.get_final_message()

        usage = None
        if final.usage:
            usage = {
                "input_tokens": final.usage.input_tokens,
                "output_tokens": final.usage.output_tokens,
            }
        yield TurnEnd(
            stop_reason=final.stop_reason or "end_turn",
            message={
                "role": "assistant",
                "content": [block.model_dump(exclude_none=True) for block in final.content],
            },
            usage=usage,
        )


class OllamaProvider:
    name = "ollama"
    context_window = 128_000

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
