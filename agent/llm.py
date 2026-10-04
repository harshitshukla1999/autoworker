"""Model-agnostic LLM layer.

The agent loop keeps its conversation in a small neutral format and this module
translates it to whichever provider is configured. That keeps the loop free of
vendor-specific code and lets the same agent run on Claude, GPT, Gemini (via its
OpenAI-compatible endpoint) or a scripted fake model in tests.

Neutral history items:
    {"role": "user", "text": str}
    {"role": "assistant", "text": str | None, "call": {"id", "name", "args"}}
    {"role": "tool", "id": str, "name": str, "text": str}
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    name: str
    args: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])


@dataclass
class LLMResponse:
    text: str | None
    call: ToolCall | None
    usage: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    pass


# ---- rate limiting -------------------------------------------------------
# Free API tiers allow only a few requests per minute (Gemini free tier: 5/min).
# LLM_MAX_RPM in .env spaces calls out so the agent never hits the limit.
_MIN_INTERVAL = 0.0
_last_call = 0.0


def set_rate_limit(max_rpm: float | None) -> None:
    global _MIN_INTERVAL
    _MIN_INTERVAL = 60.0 / max_rpm if max_rpm and max_rpm > 0 else 0.0


def _throttle() -> None:
    global _last_call
    wait = _MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        print(f"   \033[2m⏳ waiting {wait:.0f}s (API rate limit)\033[0m", flush=True)
        time.sleep(wait)
    _last_call = time.time()


def _retry_after(exc) -> float | None:
    """Seconds the provider asked us to wait, if it said so (e.g. 'retry in 44.2s')."""
    import re
    m = re.search(r"retry in ([\d.]+)s|retryDelay['\"]?: ?['\"]?(\d+)s", str(exc))
    if m:
        return float(m.group(1) or m.group(2)) + 1
    return None


def _with_retries(fn, attempts: int = 6):
    delay = 2.0
    for i in range(attempts):
        _throttle()
        try:
            return fn()
        except Exception as exc:  # network errors, rate limits, 5xx
            status = getattr(exc, "status_code", None)
            permanent = status is not None and 400 <= status < 500 and status not in (408, 409, 429)
            if permanent or i == attempts - 1:
                raise LLMError(f"LLM call failed ({i + 1} attempt(s)): {exc}") from exc
            wait = _retry_after(exc) if status == 429 else None
            wait = wait or (delay * 8 if status == 429 else delay)
            print(f"   \033[2m↻ model API error ({status or type(exc).__name__}), "
                  f"retrying in {wait:.0f}s\033[0m", flush=True)
            time.sleep(wait)
            delay *= 2


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None):
        import anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(self, system: str, history: list[dict], tools: list[dict],
                 max_tokens: int = 1500) -> LLMResponse:
        messages: list[dict] = []

        def push(role, block):
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": role, "content": [block]})

        for h in history:
            if h["role"] == "user":
                push("user", {"type": "text", "text": h["text"]})
            elif h["role"] == "assistant":
                if h.get("text"):
                    push("assistant", {"type": "text", "text": h["text"]})
                if h.get("call"):
                    c = h["call"]
                    push("assistant", {"type": "tool_use", "id": c["id"], "name": c["name"],
                                       "input": c["args"]})
            elif h["role"] == "tool":
                push("user", {"type": "tool_result", "tool_use_id": h["id"], "content": h["text"]})

        anth_tools = [{"name": t["name"], "description": t["description"],
                       "input_schema": t["parameters"]} for t in tools]
        resp = _with_retries(lambda: self.client.messages.create(
            model=self.model, system=system, messages=messages, tools=anth_tools,
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
            max_tokens=max_tokens))
        text = "\n".join(b.text for b in resp.content if b.type == "text").strip() or None
        call = next((ToolCall(b.name, dict(b.input), b.id) for b in resp.content
                     if b.type == "tool_use"), None)
        usage = {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens}
        return LLMResponse(text, call, usage)


class OpenAICompatLLM:
    """Works with OpenAI and any OpenAI-compatible endpoint (Gemini, Groq, Together, Ollama...)."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        import openai
        self.model = model
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        # Providers differ slightly; adapt once if the endpoint rejects a parameter.
        self.tool_choice = "required"
        self.token_param = "max_tokens"
        # Provider-specific data attached to a tool call that must be sent back
        # unchanged on later turns (e.g. Gemini 3 "thought signatures"), keyed by call id.
        self._call_extras: dict[str, dict] = {}

    def _create(self, messages, tools, max_tokens):
        for _ in range(3):
            kwargs = {"model": self.model, "messages": messages, "tools": tools,
                      self.token_param: max_tokens}
            if self.tool_choice:
                kwargs["tool_choice"] = self.tool_choice
            try:
                return _with_retries(lambda: self.client.chat.completions.create(**kwargs))
            except LLMError as exc:
                msg = str(exc.__cause__ or exc)
                if "max_tokens" in msg and self.token_param == "max_tokens":
                    self.token_param = "max_completion_tokens"
                elif "tool_choice" in msg and self.tool_choice:
                    self.tool_choice = None
                else:
                    raise
        raise LLMError("Could not find request parameters accepted by this provider.")

    def complete(self, system: str, history: list[dict], tools: list[dict],
                 max_tokens: int = 1500) -> LLMResponse:
        messages: list[dict] = [{"role": "system", "content": system}]
        for h in history:
            if h["role"] == "user":
                messages.append({"role": "user", "content": h["text"]})
            elif h["role"] == "assistant":
                m = {"role": "assistant", "content": h.get("text") or ""}
                if h.get("call"):
                    c = h["call"]
                    tc = {"id": c["id"], "type": "function", "function": {
                        "name": c["name"], "arguments": json.dumps(c["args"])}}
                    tc.update(self._call_extras.get(c["id"], {}))
                    m["tool_calls"] = [tc]
                messages.append(m)
            elif h["role"] == "tool":
                messages.append({"role": "tool", "tool_call_id": h["id"], "content": h["text"]})
        oa_tools = [{"type": "function", "function": t} for t in tools]
        resp = self._create(messages, oa_tools, max_tokens)
        msg = resp.choices[0].message
        call = None
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_malformed_arguments": tc.function.arguments}
            call = ToolCall(tc.function.name, args, tc.id or ToolCall("x", {}).id)
            extras = {k: v for k, v in (getattr(tc, "model_extra", None) or {}).items()
                      if k not in ("id", "type", "function") and v is not None}
            if extras:
                self._call_extras[call.id] = extras
        usage = {}
        if resp.usage:
            usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
        return LLMResponse((msg.content or "").strip() or None, call, usage)


def make_llm(settings):
    import os
    rpm = os.environ.get("LLM_MAX_RPM")
    set_rate_limit(float(rpm) if rpm else None)
    if settings.provider == "anthropic":
        return AnthropicLLM(settings.model, settings.api_key)
    if settings.provider in ("openai", "gemini", "groq", "openai-compatible"):
        return OpenAICompatLLM(settings.model, settings.api_key, settings.base_url)
    raise ValueError(f"Unknown LLM_PROVIDER {settings.provider!r}")