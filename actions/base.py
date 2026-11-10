"""
actions/base.py — The Action contract every tool implements.

To add a new tool: subclass Action, fill in the class attributes, implement
execute(), and register it in actions/registry.py. That's it — the LLM sees it
automatically, and the safety layer validates its params via input_schema.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from models import ActionResult, RiskLevel

if TYPE_CHECKING:
    import httpx

    from config import Settings


@dataclass
class ActionContext:
    """Runtime resources passed to every action (no global state)."""

    workspace: Path
    command_timeout: int = 10
    http_timeout: int = 10
    max_output_chars: int = 8000
    # Tests inject httpx.MockTransport here so nothing touches the network.
    http_transport: "httpx.BaseTransport | None" = None

    @property
    def db_path(self) -> Path:
        return self.workspace / "agent.db"

    @classmethod
    def from_settings(cls, s: "Settings") -> "ActionContext":
        return cls(
            workspace=s.workspace,
            command_timeout=s.command_timeout,
            http_timeout=s.http_timeout,
            max_output_chars=s.max_output_chars,
        )


class Action(ABC):
    """Base class for all tools."""

    name: ClassVar[str]
    description: ClassVar[str]            # shown to the LLM — be specific
    input_schema: ClassVar[dict[str, Any]]  # JSON Schema for params
    risk: ClassVar[RiskLevel]

    def effective_risk(self, params: dict[str, Any], ctx: ActionContext) -> RiskLevel:
        """Risk for THESE params. Override when risk depends on input
        (e.g. writing a new file vs. overwriting an existing one)."""
        return self.risk

    def safety_check(self, params: dict[str, Any], ctx: ActionContext) -> str | None:
        """Domain-specific checks after schema validation.
        Return an error message, raise SafetyError, or return None if OK."""
        return None

    def describe(self, params: dict[str, Any], ctx: ActionContext) -> str:
        """One-line, human-readable preview shown before confirmation."""
        return f"{self.name}({json.dumps(params, ensure_ascii=False)})"

    @abstractmethod
    def execute(self, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
        """Do the work. May raise; run_action() turns exceptions into failures."""


def run_action(action: Action, params: dict[str, Any], ctx: ActionContext) -> ActionResult:
    """Execute an action and guarantee it never raises — one bad action must
    not crash the agent. The error goes back to the LLM instead."""
    try:
        return action.execute(params, ctx)
    except Exception as exc:  # noqa: BLE001 — intentional catch-all boundary
        return ActionResult.fail(f"{type(exc).__name__}: {exc}")
