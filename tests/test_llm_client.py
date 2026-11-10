"""Tests for llm_client.py using a fake Anthropic SDK (no network, no API key)."""

from __future__ import annotations

import random
from types import SimpleNamespace as NS

import anthropic
import httpx
import pytest

from llm_client import (
    AnthropicClient, LLMAuthError, LLMRequestError, LLMRetryExhaustedError, RetryPolicy,
)
from models import Message, TextBlock, ToolCall, ToolResult

_REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(cls, status: int, headers: dict | None = None):
    return cls("simulated", response=httpx.Response(status, request=_REQ, headers=headers or {}), body=None)


OK = NS(
    content=[NS(type="text", text="Hello"), NS(type="tool_use", id="tu_1", name="read_file", input={"path": "a"})],
    stop_reason="tool_use",
    usage=NS(input_tokens=12, output_tokens=7),
)


class FakeSDK:
    """Mimics `anthropic.Anthropic().messages.create` with a scripted sequence."""

    def __init__(self, sequence):
        self.sequence = list(sequence)
        self.calls: list[dict] = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def make_client():
    def _make(sequence, max_retries=3):
        sdk = FakeSDK(sequence)
        sleeps: list[float] = []
        retries: list[tuple] = []
        client = AnthropicClient(
            api_key="test", model="test-model",
            retry=RetryPolicy(max_retries=max_retries, sleep=sleeps.append, rng=random.Random(0)),
            on_retry=lambda *a: retries.append(a),
            sdk_client=sdk,
        )
        return client, sdk, sleeps, retries
    return _make


# ---- Parsing / conversion ---------------------------------------------------

def test_parses_text_tool_calls_and_usage(make_client):
    client, sdk, _, _ = make_client([OK])
    r = client.complete("sys", [Message.user_text("hi")], [{"name": "read_file"}])
    assert r.text == "Hello"
    assert r.tool_calls == [ToolCall("tu_1", "read_file", {"path": "a"})]
    assert r.usage == {"input_tokens": 12, "output_tokens": 7}
    assert sdk.calls[0]["tools"] == [{"name": "read_file"}]
    assert sdk.calls[0]["system"] == "sys"


def test_converts_history_to_anthropic_format(make_client):
    client, sdk, _, _ = make_client([OK])
    history = [
        Message.user_text("hi"),
        Message("assistant", [TextBlock("plan"), ToolCall("tu_0", "list_dir", {})]),
        Message("user", [ToolResult("tu_0", "boom", is_error=True)]),
    ]
    client.complete("sys", history, [])
    sent = sdk.calls[0]["messages"]
    assert sent[1]["content"][1] == {"type": "tool_use", "id": "tu_0", "name": "list_dir", "input": {}}
    assert sent[2]["content"][0] == {"type": "tool_result", "tool_use_id": "tu_0",
                                     "content": "boom", "is_error": True}
    assert "tools" not in sdk.calls[0]  # empty tool list is omitted


# ---- Retry logic ------------------------------------------------------------

def test_retries_rate_limit_then_succeeds(make_client):
    client, sdk, sleeps, retries = make_client([
        api_error(anthropic.RateLimitError, 429),
        api_error(anthropic.InternalServerError, 500),
        OK,
    ])
    assert client.complete("s", [], []).text == "Hello"
    assert len(sdk.calls) == 3 and len(sleeps) == 2
    assert retries[0][3] == "rate limited (429)" and retries[1][3] == "server error (500)"


def test_retry_after_header_is_respected(make_client):
    client, _, sleeps, _ = make_client([api_error(anthropic.RateLimitError, 429, {"retry-after": "7"}), OK])
    client.complete("s", [], [])
    assert sleeps == [7.0]


def test_timeouts_and_connection_errors_are_retried(make_client):
    client, sdk, _, _ = make_client([anthropic.APITimeoutError(_REQ), anthropic.APIConnectionError(request=_REQ), OK])
    assert client.complete("s", [], []).text == "Hello"
    assert len(sdk.calls) == 3


def test_gives_up_after_max_retries(make_client):
    client, sdk, sleeps, _ = make_client([api_error(anthropic.InternalServerError, 503)] * 4)
    with pytest.raises(LLMRetryExhaustedError, match="after 4 attempts"):
        client.complete("s", [], [])
    assert len(sdk.calls) == 4 and len(sleeps) == 3


@pytest.mark.parametrize("cls, status, err, msg", [
    (anthropic.AuthenticationError, 401, LLMAuthError, "ANTHROPIC_API_KEY"),
    (anthropic.PermissionDeniedError, 403, LLMAuthError, "permission"),
    (anthropic.NotFoundError, 404, LLMRequestError, "ANTHROPIC_MODEL"),
    (anthropic.BadRequestError, 400, LLMRequestError, "Invalid request"),
])
def test_non_retryable_errors_fail_immediately(make_client, cls, status, err, msg):
    client, sdk, sleeps, _ = make_client([api_error(cls, status)])
    with pytest.raises(err, match=msg):
        client.complete("s", [], [])
    assert len(sdk.calls) == 1 and sleeps == []


def test_backoff_grows_and_is_capped():
    policy = RetryPolicy(base_delay=1.0, max_delay=10.0, rng=random.Random(42))
    for attempt in range(8):
        d = policy.delay_for(attempt)
        expected = min(10.0, 2 ** attempt)
        assert expected / 2 <= d <= expected  # equal jitter range
    assert policy.delay_for(0, retry_after=100) == 10.0  # retry-after still capped
