"""
llm_client.py — LLM integration layer.

The rest of the agent depends only on the abstract `LLMClient` and the neutral
types in models.py. `AnthropicClient` is the concrete adapter for Claude.

Swapping providers: write e.g. `OpenAIClient(LLMClient)` that converts
models.Message <-> OpenAI chat format and wraps tool specs as
{"type": "function", "function": {...}}, then return it from create_client().
Nothing else in the codebase changes.

Retries are implemented HERE (not by the SDK, which we set to max_retries=0) so
the backoff policy is explicit, configurable, visible in the UI, and testable.

Run `python llm_client.py` for a live smoke test (needs ANTHROPIC_API_KEY).
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from models import LLMResponse, Message, TextBlock, ToolCall, ToolResult


# ---- Errors (user-facing messages) -----------------------------------------

class LLMError(Exception):
    """Base class. `str(err)` is safe to show the user."""


class LLMAuthError(LLMError):
    """Bad / missing API key or no permission. Not retried."""


class LLMRequestError(LLMError):
    """The request itself is invalid (bad model name, bad params). Not retried."""


class LLMRetryExhaustedError(LLMError):
    """A transient error persisted through every retry."""


# ---- Retry policy -----------------------------------------------------------

@dataclass
class RetryPolicy:
    max_retries: int = 3          # retries AFTER the first attempt (4 attempts total)
    base_delay: float = 1.0       # seconds
    max_delay: float = 30.0       # cap for any single wait
    # Injectable for tests so they run instantly and deterministically.
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)
    rng: random.Random = field(default_factory=random.Random, repr=False)

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """
        Exponential backoff with "equal jitter":
            d = min(max_delay, base * 2**attempt);  wait = d/2 + uniform(0, d/2)
        Jitter keeps many clients from retrying in lockstep. If the server sent
        a Retry-After header, we wait at least that long.
        """
        d = min(self.max_delay, self.base_delay * (2 ** attempt))
        wait = d / 2 + self.rng.uniform(0, d / 2)
        if retry_after is not None:
            wait = max(wait, retry_after)
        return min(wait, self.max_delay)


#: Called before each retry sleep: (attempt_number, max_retries, delay_seconds, reason)
RetryCallback = Callable[[int, int, float, str], None]


# ---- Abstract interface -----------------------------------------------------

class LLMClient(ABC):
    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
    ) -> LLMResponse:
        """Send the conversation and return a normalized response. Raises LLMError."""


# ---- Anthropic adapter ------------------------------------------------------

class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 2048,
        retry: RetryPolicy | None = None,
        on_retry: RetryCallback | None = None,
        timeout: float = 60.0,
        sdk_client: Any = None,  # tests pass a fake here
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.retry = retry or RetryPolicy()
        self.on_retry = on_retry
        self._client = sdk_client or anthropic.Anthropic(
            api_key=api_key, max_retries=0, timeout=timeout
        )

    # -- format conversion (the only provider-specific code) --

    @staticmethod
    def _to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
        out = []
        for msg in messages:
            blocks = []
            for b in msg.content:
                if isinstance(b, TextBlock):
                    blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ToolCall):
                    blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                elif isinstance(b, ToolResult):
                    blocks.append({
                        "type": "tool_result",
                        "tool_use_id": b.tool_call_id,
                        "content": b.content,
                        "is_error": b.is_error,
                    })
            out.append({"role": msg.role, "content": blocks})
        return out

    @staticmethod
    def _from_anthropic(resp: Any) -> LLMResponse:
        texts: list[str] = []
        calls: list[ToolCall] = []
        raw: list = []
        for block in resp.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                texts.append(block.text)
                raw.append(TextBlock(block.text))
            elif kind == "tool_use":
                call = ToolCall(id=block.id, name=block.name, input=dict(block.input or {}))
                calls.append(call)
                raw.append(call)
            # other block types (e.g. thinking) are ignored
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text="\n".join(texts).strip(),
            tool_calls=calls,
            stop_reason=getattr(resp, "stop_reason", None),
            raw_content=raw,
            usage={
                "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                "output_tokens": getattr(usage, "output_tokens", 0) or 0,
            },
        )

    # -- error classification --

    @staticmethod
    def _retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        header = response.headers.get("retry-after") if response is not None else None
        try:
            return float(header) if header else None
        except ValueError:
            return None

    def _classify(self, exc: Exception) -> tuple[bool, str]:
        """Return (is_retryable, user_message)."""
        if isinstance(exc, anthropic.AuthenticationError):
            raise LLMAuthError("Authentication failed — check ANTHROPIC_API_KEY in your .env.") from exc
        if isinstance(exc, anthropic.PermissionDeniedError):
            raise LLMAuthError("Your API key doesn't have permission for this request/model.") from exc
        if isinstance(exc, anthropic.NotFoundError):
            raise LLMRequestError(
                f"Model '{self.model}' was not found — check ANTHROPIC_MODEL in your .env."
            ) from exc
        if isinstance(exc, anthropic.RateLimitError):
            return True, "rate limited (429)"
        if isinstance(exc, anthropic.APITimeoutError):
            return True, "request timed out"
        if isinstance(exc, anthropic.APIConnectionError):
            return True, "network connection error"
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code >= 500:  # includes 529 "overloaded"
                return True, f"server error ({exc.status_code})"
            raise LLMRequestError(f"Invalid request ({exc.status_code}): {exc.message}") from exc
        raise LLMError(f"Unexpected error calling the LLM: {exc}") from exc

    # -- main entry point --

    def complete(self, system: str, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": self._to_anthropic(messages),
        }
        if tools:
            payload["tools"] = tools

        last_reason = ""
        for attempt in range(self.retry.max_retries + 1):
            try:
                return self._from_anthropic(self._client.messages.create(**payload))
            except anthropic.APIError as exc:
                retryable, last_reason = self._classify(exc)  # raises if not retryable
                if attempt == self.retry.max_retries:
                    break
                delay = self.retry.delay_for(attempt, self._retry_after(exc))
                if self.on_retry:
                    self.on_retry(attempt + 1, self.retry.max_retries, delay, last_reason)
                self.retry.sleep(delay)

        raise LLMRetryExhaustedError(
            f"The LLM API is unavailable ({last_reason}) after "
            f"{self.retry.max_retries + 1} attempts. Please try again in a minute."
        )


def create_client(settings, on_retry: RetryCallback | None = None) -> LLMClient:
    """Factory used by main.py. Change this one function to switch providers."""
    return AnthropicClient(
        api_key=settings.require_api_key(),
        model=settings.model,
        max_tokens=settings.max_tokens,
        on_retry=on_retry,
    )


# ---- Live smoke test: python llm_client.py ----------------------------------

if __name__ == "__main__":
    from config import load_settings

    settings = load_settings()
    client = create_client(
        settings,
        on_retry=lambda n, m, d, why: print(f"  retry {n}/{m} in {d:.1f}s ({why})"),
    )
    print(f"Model: {settings.model}")

    print("\n1) Plain text:")
    r = client.complete("Be brief.", [Message.user_text("Say hello in five words.")], [])
    print(f"  {r.text!r}  stop={r.stop_reason}  usage={r.usage}")

    print("\n2) Tool calling:")
    echo_tool = {
        "name": "echo",
        "description": "Echo a message back.",
        "input_schema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    }
    r = client.complete(
        "Use the echo tool when asked.",
        [Message.user_text("Use the echo tool to echo 'it works'.")],
        [echo_tool],
    )
    for call in r.tool_calls:
        print(f"  tool call -> {call.name}({call.input})")
    print("  PASS" if r.tool_calls else "  (no tool call returned)")
