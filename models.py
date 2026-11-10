"""
models.py — Provider-neutral data types shared across the agent.

The agent loop, safety layer, and actions only ever see these types. Each LLM
adapter (e.g. AnthropicClient) translates to/from its SDK's format, which is
what makes swapping providers a one-class change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Union


class RiskLevel(str, Enum):
    """How dangerous an action is; drives the confirmation policy."""

    SAFE = "safe"                # read-only, no side effects
    MODERATE = "moderate"        # side effects, but recoverable (write, insert, run)
    DESTRUCTIVE = "destructive"  # irreversible (delete) — always confirmed


# ---- Conversation content blocks -------------------------------------------

@dataclass
class TextBlock:
    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCall:
    """An action the LLM wants to run."""

    id: str                      # provider-assigned id, echoed back in the result
    name: str                    # tool name, e.g. "read_file"
    input: dict[str, Any]        # raw params from the LLM (UNTRUSTED until validated)
    type: Literal["tool_call"] = "tool_call"


@dataclass
class ToolResult:
    """The outcome of a ToolCall, sent back to the LLM."""

    tool_call_id: str
    content: str
    is_error: bool = False
    type: Literal["tool_result"] = "tool_result"


ContentBlock = Union[TextBlock, ToolCall, ToolResult]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock]

    @classmethod
    def user_text(cls, text: str) -> "Message":
        return cls(role="user", content=[TextBlock(text)])


# ---- LLM response -----------------------------------------------------------

@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    text: str                                   # concatenated text blocks (may be "")
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None              # e.g. "end_turn", "tool_use", "max_tokens"
    raw_content: list[ContentBlock] = field(default_factory=list)  # for history
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---- Action execution -------------------------------------------------------

@dataclass
class ActionResult:
    """What an Action.execute() returns."""

    success: bool
    output: str = ""
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)  # optional structured extras

    @classmethod
    def ok(cls, output: str, **data: Any) -> "ActionResult":
        return cls(success=True, output=output, data=data)

    @classmethod
    def fail(cls, error: str) -> "ActionResult":
        return cls(success=False, error=error)

    def to_llm_text(self, max_chars: int) -> str:
        """Render for the LLM, truncating so huge outputs can't flood context."""
        text = self.output if self.success else f"ERROR: {self.error}"
        if len(text) > max_chars:
            omitted = len(text) - max_chars
            text = text[:max_chars] + f"\n...[truncated {omitted} characters]"
        return text
