"""
Shared pytest fixtures. Every test runs in a temporary workspace and uses fakes
for the LLM and HTTP, so the suite needs no API key and no network access.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from actions import ActionContext, build_default_registry
from agent_loop import Agent
from llm_client import LLMClient
from logger import AuditLogger
from models import LLMResponse, Message, TextBlock, ToolCall
from safety import SafetyValidator


# ---- Fake LLM ---------------------------------------------------------------

class ScriptedLLM(LLMClient):
    """
    Returns pre-written responses in order. Each step is either:
      * (text, [ToolCall, ...])   — a normal response
      * an Exception instance     — raised to simulate an API failure
    Every conversation it receives is recorded in `.received` for assertions.
    """

    def __init__(self, steps: list[Any]):
        self.steps = list(steps)
        self.received: list[list[Message]] = []

    def complete(self, system, messages, tools) -> LLMResponse:
        self.received.append(list(messages))
        if not self.steps:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        text, calls = step
        raw = ([TextBlock(text)] if text else []) + list(calls)
        return LLMResponse(text=text, tool_calls=list(calls),
                           stop_reason="tool_use" if calls else "end_turn", raw_content=raw)


def call(name: str, _id: str | None = None, **params: Any) -> ToolCall:
    """Shorthand: call('read_file', path='a.txt')."""
    return ToolCall(id=_id or f"call_{name}_{abs(hash(str(params))) % 10_000}", name=name, input=params)


# ---- Fake HTTP --------------------------------------------------------------

def fake_api(request: httpx.Request) -> httpx.Response:
    host, path = request.url.host, request.url.path
    if host == "geocoding-api.open-meteo.com":
        name = request.url.params.get("name", "")
        if name.lower() == "nowhereville":
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"results": [
            {"name": "Paris", "country": "France", "latitude": 48.85, "longitude": 2.35}
        ]})
    if host == "api.open-meteo.com":
        return httpx.Response(200, json={"current": {
            "temperature_2m": 12.5, "relative_humidity_2m": 70,
            "wind_speed_10m": 8.0, "weather_code": 3, "time": "2026-11-09T10:00",
        }})
    if host == "catfact.ninja" and path == "/fact":
        return httpx.Response(200, json={"fact": "Cats sleep a lot."})
    return httpx.Response(404, json={"error": "not found"})


# ---- Fixtures ---------------------------------------------------------------

@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws.resolve()


@pytest.fixture
def ctx(workspace: Path) -> ActionContext:
    return ActionContext(workspace=workspace, command_timeout=5,
                         http_transport=httpx.MockTransport(fake_api))


@pytest.fixture
def registry():
    return build_default_registry()


@pytest.fixture
def validator(registry, ctx) -> SafetyValidator:
    return SafetyValidator(registry, ctx)


@pytest.fixture
def audit(tmp_path: Path) -> AuditLogger:
    return AuditLogger(tmp_path / "logs" / "audit.jsonl")


@pytest.fixture
def make_agent(registry, ctx, audit):
    """Factory: make_agent(steps, answers=[True, False], auto_yes=False, ...)."""

    def _make(steps, answers=None, **kwargs):
        answers = list(answers or [])
        events = []
        llm = ScriptedLLM(steps)
        agent = Agent(
            llm, registry, ctx, audit,
            confirm=lambda c, v: answers.pop(0) if answers else False,
            on_event=events.append,
            **kwargs,
        )
        agent.events = events  # attach for assertions
        return agent, llm

    return _make
