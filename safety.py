"""
safety.py — Everything that decides whether an LLM-proposed action may run.

Layers of defense (each one alone should stop the obvious attacks):
  1. Tool must exist in the registry          -> blocks made-up tools
  2. Params must match the tool's JSON Schema -> blocks malformed / extra params
  3. Domain checks (paths, commands, URLs, SQL)-> blocks traversal, dangerous cmds
  4. Confirmation policy by risk level        -> human in the loop for side effects
Actions ALSO re-check paths/commands inside execute() (defense in depth), so
calling an action directly can't skip the checks.

This module imports only `models`, never the actions package, to avoid
circular imports; actions import the helper functions from here.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from jsonschema import Draft202012Validator

from models import RiskLevel, ToolCall

if TYPE_CHECKING:  # type hints only — no runtime import cycle
    from actions.base import Action, ActionContext
    from actions.registry import ActionRegistry


class SafetyError(Exception):
    """Raised by safety checks; the message is shown to the user and the LLM."""


# ---- Path confinement -------------------------------------------------------

def resolve_in_workspace(path_str: str, workspace: Path) -> Path:
    """
    Resolve `path_str` and guarantee the result is inside `workspace`.

    Handles relative paths, absolute paths, `..` segments, `~`, and symlinks
    (resolve() follows symlinks, so a link pointing outside is rejected).
    """
    if not isinstance(path_str, str) or not path_str.strip():
        raise SafetyError("Path must be a non-empty string.")
    if "\x00" in path_str:
        raise SafetyError("Path contains a null byte.")

    ws = workspace.resolve()
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = ws / candidate
    resolved = candidate.resolve()

    if resolved != ws and not resolved.is_relative_to(ws):
        raise SafetyError(
            f"Path '{path_str}' resolves outside the workspace and is not allowed."
        )
    return resolved


# ---- Shell command policy ---------------------------------------------------

#: Only these programs may run. All are read-only / informational.
ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {"ls", "pwd", "echo", "cat", "date", "whoami", "wc", "head", "tail", "python", "python3"}
)

#: Commands whose non-flag arguments are file paths that must stay in the workspace.
PATH_ARG_COMMANDS: frozenset[str] = frozenset({"ls", "cat", "wc", "head", "tail"})

#: Checked against the raw string BEFORE parsing. Commands run with shell=False,
#: so operators wouldn't work anyway — rejecting them gives the LLM a clear reason.
BLOCKED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\b"), "file deletion via shell is not allowed (use delete_file)"),
    (re.compile(r"\b(sudo|su|doas)\b"), "privilege escalation is not allowed"),
    (re.compile(r"[|;&<>`]"), "shell operators (| ; & < > `) are not allowed"),
    (re.compile(r"\$\("), "command substitution is not allowed"),
    (re.compile(r"\b(mkfs|dd|shutdown|reboot|chmod|chown|kill|curl|wget|nc)\b"),
     "this command is blocked"),
]

MAX_COMMAND_LENGTH = 500


def check_command(command: str, workspace: Path) -> list[str]:
    """Validate a shell command string and return the parsed argv list."""
    if len(command) > MAX_COMMAND_LENGTH:
        raise SafetyError(f"Command is longer than {MAX_COMMAND_LENGTH} characters.")

    for pattern, reason in BLOCKED_PATTERNS:
        if pattern.search(command):
            raise SafetyError(f"Blocked command: {reason}.")

    try:
        argv = shlex.split(command)
    except ValueError as exc:  # e.g. unbalanced quotes
        raise SafetyError(f"Could not parse command: {exc}.") from exc
    if not argv:
        raise SafetyError("Command is empty.")

    program = argv[0]
    if "/" in program or "\\" in program:
        raise SafetyError("Use a bare command name (e.g. 'ls'), not a path to a program.")
    if program not in ALLOWED_COMMANDS:
        allowed = ", ".join(sorted(ALLOWED_COMMANDS))
        raise SafetyError(f"'{program}' is not on the allowlist. Allowed: {allowed}.")

    # python is only allowed to report its version — never to run code.
    if program in {"python", "python3"} and argv[1:] not in (["--version"], ["-V"]):
        raise SafetyError("python may only be run as 'python --version'.")

    # File-reading commands can't escape the workspace (e.g. `cat /etc/passwd`).
    if program in PATH_ARG_COMMANDS:
        for arg in argv[1:]:
            if not arg.startswith("-"):
                resolve_in_workspace(arg, workspace)

    return argv


# ---- Confirmation policy ----------------------------------------------------

def needs_confirmation(risk: RiskLevel, auto_yes: bool = False) -> bool:
    """
    safe        -> never ask
    moderate    -> ask, unless --yes was given
    destructive -> ALWAYS ask; --yes can never bypass this
    """
    if risk is RiskLevel.DESTRUCTIVE:
        return True
    if risk is RiskLevel.MODERATE:
        return not auto_yes
    return False


# ---- Validator --------------------------------------------------------------

@dataclass
class ValidationResult:
    allowed: bool
    reason: str | None = None          # why it was denied (None if allowed)
    action: "Action | None" = None
    risk: RiskLevel | None = None      # effective risk for THESE params
    preview: str = ""                  # human-readable "what will happen"


class SafetyValidator:
    """Runs every check on a ToolCall before anything executes."""

    def __init__(self, registry: "ActionRegistry", ctx: "ActionContext"):
        self.registry = registry
        self.ctx = ctx
        self._validators: dict[str, Draft202012Validator] = {}

    def _schema_validator(self, action: "Action") -> Draft202012Validator:
        if action.name not in self._validators:
            self._validators[action.name] = Draft202012Validator(action.input_schema)
        return self._validators[action.name]

    def validate(self, call: ToolCall) -> ValidationResult:
        # 1. Tool exists?
        action = self.registry.get(call.name)
        if action is None:
            return ValidationResult(False, f"Unknown tool '{call.name}'.")

        # 2. Params match schema?
        if not isinstance(call.input, dict):
            return ValidationResult(False, "Tool input must be a JSON object.", action)
        errors = sorted(
            self._schema_validator(action).iter_errors(call.input),
            key=lambda e: list(e.path),
        )
        if errors:
            details = "; ".join(
                f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors[:5]
            )
            return ValidationResult(False, f"Invalid parameters: {details}", action)

        # 3. Domain-specific checks (paths, commands, URLs, SQL).
        try:
            problem = action.safety_check(call.input, self.ctx)
        except SafetyError as exc:
            problem = str(exc)
        if problem:
            return ValidationResult(False, problem, action)

        # Passed. Compute risk + preview for the confirmation step.
        return ValidationResult(
            allowed=True,
            action=action,
            risk=action.effective_risk(call.input, self.ctx),
            preview=action.describe(call.input, self.ctx),
        )
