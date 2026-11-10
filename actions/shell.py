"""
actions/shell.py — Run an allowlisted command (no shell, timeout, workspace cwd).
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from typing import Any

import safety
from actions.base import Action, ActionContext
from models import ActionResult, RiskLevel
from safety import ALLOWED_COMMANDS, WINDOWS_BUILTINS, check_command

_SECRET_HINTS = ("KEY", "SECRET", "TOKEN", "PASSWORD")


def _scrubbed_env() -> dict[str, str]:
    """Child processes never see API keys or other secrets."""
    return {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_HINTS)}


class RunCommand(Action):
    name = "run_command"
    description = (
        f"Run a simple read-only system command in the workspace folder. "
        f"This computer runs {platform.system() or 'an unknown OS'}, so use commands for that OS "
        "(Windows: dir, type, where, hostname, ver; macOS/Linux: ls, cat, wc, head, tail). "
        f"Full allowlist: {', '.join(sorted(ALLOWED_COMMANDS))}. "
        "No pipes, redirects, or chaining. File arguments must be inside the workspace."
    )
    risk = RiskLevel.MODERATE
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "minLength": 1,
                "maxLength": 500,
                "description": "The command line, e.g. 'wc -l notes.txt'.",
            }
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        check_command(p["command"], ctx.workspace)  # raises SafetyError
        return None

    def describe(self, p: dict[str, Any], ctx: ActionContext) -> str:
        return f"RUN `{p['command']}` in workspace"

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        argv = check_command(p["command"], ctx.workspace)  # defense in depth
        if safety.IS_WINDOWS and argv[0] in WINDOWS_BUILTINS:
            # dir/type/echo/ver live inside cmd.exe. /d skips AutoRun scripts.
            # Shell operators and cmd special characters were rejected by check_command.
            exec_argv = ["cmd", "/d", "/c", *argv]
        elif shutil.which(argv[0]) is None:
            return ActionResult.fail(f"'{argv[0]}' is not installed on this system.")
        else:
            exec_argv = argv
        try:
            proc = subprocess.run(
                exec_argv,
                shell=False,               # never interpret shell syntax
                cwd=ctx.workspace,
                capture_output=True,
                text=True,
                timeout=ctx.command_timeout,
                env=_scrubbed_env(),
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            return ActionResult.fail(f"Command timed out after {ctx.command_timeout}s.")

        report = (
            f"exit code: {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout.rstrip() or '(none)'}\n"
            f"--- stderr ---\n{proc.stderr.rstrip() or '(none)'}"
        )
        if proc.returncode == 0:
            return ActionResult.ok(report, exit_code=0)
        return ActionResult(success=False, output=report, error=report,
                            data={"exit_code": proc.returncode})
