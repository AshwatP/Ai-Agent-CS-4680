"""
agent_loop.py — The core "think -> act -> observe" loop.

For each user request:
  1. Send conversation + tool specs to the LLM
  2. If the LLM returns a final answer (no tool calls) -> done
  3. For each tool call: validate -> confirm if needed -> execute -> record result
  4. Send results back to the LLM and repeat (up to max_iterations)

The loop never talks to the terminal directly. It reports progress through two
callbacks, so the same Agent works with the rich UI, tests, or any other frontend:
  * on_event(AgentEvent)                 — status updates for display
  * confirm(ToolCall, ValidationResult)  — returns True to approve an action

Run `python agent_loop.py "your request"` for a quick test without the full UI.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Callable

from actions import ActionContext, ActionRegistry, run_action
from llm_client import LLMClient, LLMError
from logger import AuditLogger
from models import LLMResponse, Message, RiskLevel, TextBlock, ToolCall, ToolResult
from safety import SafetyValidator, ValidationResult, needs_confirmation


SYSTEM_PROMPT = """You are an AI agent that completes tasks by calling tools.

Environment:
- All file paths are relative to a sandboxed workspace folder.
- A local SQLite database is available (call db_list_tables to see its schema).
- Shell commands are limited to a small read-only allowlist.
- Operating system: {os_name}.
- Today's date is {today}.

How to work:
- Before calling tools, write ONE short sentence describing your plan.
- Use tools to get real information; never invent file contents, data, or results.
- Prefer reading/listing before writing or deleting.
- Some actions require the user's confirmation. If an action is denied by a safety
  check or declined by the user, do NOT try to work around it with a different
  tool — explain what happened and ask how to proceed.
- If a tool returns an error, use the message to fix your call or explain the problem.
- When finished, give a concise answer that says what you did and the outcome."""


class EventKind(str, Enum):
    THINKING = "thinking"            # about to call the LLM (UI shows a spinner)
    PLAN = "plan"                    # LLM's text alongside tool calls
    PROPOSED = "proposed"
    DENIED = "denied"                # failed validation
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    DECLINED = "declined"            # user said no
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"            # emitted by the UI's LLM on_retry hook
    FINAL = "final"
    ERROR = "error"                  # LLM/API failure for the whole request
    LIMIT = "limit"                  # max_iterations reached


@dataclass
class AgentEvent:
    kind: EventKind
    request_id: str
    message: str = ""
    call: ToolCall | None = None
    risk: RiskLevel | None = None
    preview: str = ""


@dataclass
class ActionRecord:
    """One tool call and what happened to it (for /history and tests)."""
    tool: str
    params: dict[str, Any]
    status: str                      # success | failed | denied | declined
    risk: str | None = None
    detail: str = ""


@dataclass
class AgentRunResult:
    request_id: str
    final_text: str
    actions: list[ActionRecord] = field(default_factory=list)
    stop_reason: str = "final"       # final | error | limit
    iterations: int = 0


ConfirmFn = Callable[[ToolCall, ValidationResult], bool]
EventFn = Callable[[AgentEvent], None]

_LOG_STR_LIMIT = 1000


def _clip(value: Any) -> Any:
    """Shorten long strings inside params before logging (e.g. big file contents)."""
    if isinstance(value, str) and len(value) > _LOG_STR_LIMIT:
        return value[:_LOG_STR_LIMIT] + f"...[{len(value) - _LOG_STR_LIMIT} more chars]"
    if isinstance(value, dict):
        return {k: _clip(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v) for v in value]
    return value


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ActionRegistry,
        ctx: ActionContext,
        audit: AuditLogger,
        *,
        confirm: ConfirmFn | None = None,
        on_event: EventFn | None = None,
        auto_yes: bool = False,
        max_iterations: int = 10,
        system_prompt: str | None = None,
    ):
        self.llm = llm
        self.registry = registry
        self.ctx = ctx
        self.audit = audit
        self.validator = SafetyValidator(registry, ctx)
        # Safe default: with no UI attached, anything needing confirmation is declined.
        self.confirm = confirm or (lambda call, v: False)
        self.on_event = on_event or (lambda e: None)
        self.auto_yes = auto_yes
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or SYSTEM_PROMPT.format(
            today=date.today().isoformat(), os_name=platform.system() or "unknown"
        )
        self.history: list[Message] = []

    def reset(self) -> None:
        """Forget the conversation (used by /clear)."""
        self.history.clear()

    def _emit(self, kind: EventKind, rid: str, **kw: Any) -> None:
        try:
            self.on_event(AgentEvent(kind=kind, request_id=rid, **kw))
        except Exception:  # noqa: BLE001 — a UI bug must not break the agent
            pass

    # ---- main loop ----------------------------------------------------------

    def run(self, user_request: str) -> AgentRunResult:
        rid = self.audit.new_request_id()
        self.audit.log(AuditLogger.USER_REQUEST, rid, text=user_request)
        result = AgentRunResult(request_id=rid, final_text="")

        checkpoint = len(self.history)  # roll back here if the request fails
        self.history.append(Message.user_text(user_request))
        tools = self.registry.tool_specs()

        for iteration in range(1, self.max_iterations + 1):
            result.iterations = iteration
            self.audit.log(AuditLogger.LLM_CALL, rid, iteration=iteration)
            self._emit(EventKind.THINKING, rid, message=f"step {iteration}")
            try:
                resp: LLMResponse = self.llm.complete(self.system_prompt, self.history, tools)
            except LLMError as exc:
                self.audit.log(AuditLogger.LLM_ERROR, rid, error=str(exc), error_type=type(exc).__name__)
                self._emit(EventKind.ERROR, rid, message=str(exc))
                # Drop the partial turn so the conversation stays valid for the next request.
                del self.history[checkpoint:]
                result.final_text, result.stop_reason = str(exc), "error"
                return result

            self.history.append(
                Message("assistant", resp.raw_content or [TextBlock(resp.text or "(no response)")])
            )

            # No tool calls -> this is the final answer.
            if not resp.tool_calls:
                text = resp.text or "(The model returned an empty response.)"
                if resp.stop_reason == "max_tokens":
                    text += "\n\n[Response was cut off — the output token limit was reached.]"
                self.audit.log(AuditLogger.FINAL_ANSWER, rid, text=_clip(text), usage=resp.usage)
                self._emit(EventKind.FINAL, rid, message=text)
                result.final_text = text
                return result

            if resp.text:
                self._emit(EventKind.PLAN, rid, message=resp.text)

            tool_results = [self._handle_call(rid, call, result) for call in resp.tool_calls]
            self.history.append(Message("user", tool_results))

        # Ran out of iterations.
        msg = (f"Stopped after {self.max_iterations} steps without finishing. "
               "Try breaking the request into smaller parts.")
        self.audit.log(AuditLogger.LOOP_LIMIT, rid, max_iterations=self.max_iterations)
        self._emit(EventKind.LIMIT, rid, message=msg)
        # Close the turn with an assistant message so roles keep alternating.
        self.history.append(Message("assistant", [TextBlock(f"[{msg}]")]))
        result.final_text, result.stop_reason = msg, "limit"
        return result

    # ---- one tool call ------------------------------------------------------

    def _handle_call(self, rid: str, call: ToolCall, run: AgentRunResult) -> ToolResult:
        aid = call.id
        self.audit.log(AuditLogger.ACTION_PROPOSED, rid, aid, tool=call.name, params=_clip(call.input))
        self._emit(EventKind.PROPOSED, rid, call=call)

        # 1) Validate
        v = self.validator.validate(call)
        self.audit.log(
            AuditLogger.ACTION_VALIDATED, rid, aid, tool=call.name, allowed=v.allowed,
            reason=v.reason, risk=v.risk.value if v.risk else None,
        )
        if not v.allowed:
            self.audit.log(AuditLogger.ACTION_DENIED, rid, aid, tool=call.name, reason=v.reason)
            self._emit(EventKind.DENIED, rid, call=call, message=v.reason or "")
            run.actions.append(ActionRecord(call.name, call.input, "denied", detail=v.reason or ""))
            return ToolResult(aid, f"DENIED by safety check: {v.reason}", is_error=True)

        risk = v.risk or RiskLevel.SAFE

        # 2) Confirm if the risk level requires it
        if needs_confirmation(risk, self.auto_yes):
            self._emit(EventKind.AWAITING_CONFIRMATION, rid, call=call, risk=risk, preview=v.preview)
            try:
                approved = bool(self.confirm(call, v))
            except Exception:  # noqa: BLE001 — e.g. Ctrl+C / EOF at the prompt
                approved = False
            self.audit.log(AuditLogger.CONFIRMATION, rid, aid, tool=call.name,
                           approved=approved, risk=risk.value, preview=v.preview)
            if not approved:
                self._emit(EventKind.DECLINED, rid, call=call, risk=risk, preview=v.preview)
                run.actions.append(ActionRecord(call.name, call.input, "declined", risk.value))
                return ToolResult(
                    aid, "The user DECLINED this action. Do not retry it; ask how to proceed.",
                    is_error=True,
                )

        # 3) Execute (run_action never raises)
        self._emit(EventKind.RUNNING, rid, call=call, risk=risk, preview=v.preview)
        start = time.perf_counter()
        outcome = run_action(v.action, call.input, self.ctx)
        ms = round((time.perf_counter() - start) * 1000)

        llm_text = outcome.to_llm_text(self.ctx.max_output_chars)
        self.audit.log(
            AuditLogger.ACTION_EXECUTED if outcome.success else AuditLogger.ACTION_FAILED,
            rid, aid, tool=call.name, params=_clip(call.input), risk=risk.value,
            success=outcome.success, duration_ms=ms,
            output=_clip(outcome.output), error=_clip(outcome.error),
        )
        kind = EventKind.SUCCESS if outcome.success else EventKind.FAILED
        self._emit(kind, rid, call=call, risk=risk, preview=v.preview, message=llm_text)
        run.actions.append(ActionRecord(
            call.name, call.input, "success" if outcome.success else "failed", risk.value,
            detail=(outcome.output if outcome.success else outcome.error or "")[:200],
        ))
        return ToolResult(aid, llm_text, is_error=not outcome.success)


def build_agent(settings, llm: LLMClient, **kwargs: Any) -> Agent:
    """Wire up an Agent from Settings (used by main.py)."""
    from actions import build_default_registry

    return Agent(
        llm=llm,
        registry=build_default_registry(),
        ctx=ActionContext.from_settings(settings),
        audit=AuditLogger(settings.log_path),
        max_iterations=settings.max_iterations,
        **kwargs,
    )


# ---- Quick test without the full UI: python agent_loop.py "list my tasks" -----

if __name__ == "__main__":
    import sys

    from config import load_settings
    from llm_client import create_client

    request = " ".join(sys.argv[1:]) or "What tasks are in the database?"
    settings = load_settings()

    def show(e: AgentEvent) -> None:
        if e.kind is EventKind.PLAN:
            print(f"[plan] {e.message}")
        elif e.kind in (EventKind.SUCCESS, EventKind.FAILED, EventKind.DENIED, EventKind.DECLINED):
            print(f"[{e.kind.value}] {e.call.name if e.call else ''} {e.preview or e.message[:120]}")
        elif e.kind in (EventKind.ERROR, EventKind.LIMIT):
            print(f"[{e.kind.value}] {e.message}")

    def ask(call: ToolCall, v: ValidationResult) -> bool:
        return input(f"Allow [{v.risk.value}] {v.preview}? [y/N] ").strip().lower() == "y"

    agent = build_agent(settings, create_client(settings), confirm=ask, on_event=show)
    print(f"> {request}\n")
    res = agent.run(request)
    print(f"\n{res.final_text}\n\n(request_id={res.request_id}, steps={res.iterations})")
