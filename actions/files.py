"""
actions/files.py — File operations, all confined to the workspace directory.
"""

from __future__ import annotations

from typing import Any

from actions.base import Action, ActionContext
from models import ActionResult, RiskLevel
from safety import SafetyError, resolve_in_workspace

MAX_READ_BYTES = 1_000_000
MAX_WRITE_CHARS = 1_000_000
MAX_LIST_ENTRIES = 200

_PATH_PROP = {
    "type": "string",
    "minLength": 1,
    "maxLength": 500,
    "description": "Path relative to the workspace root, e.g. 'notes/todo.txt'.",
}


def _rel(path, ctx: ActionContext) -> str:
    """Display paths relative to the workspace (never leak absolute paths)."""
    rel = path.relative_to(ctx.workspace.resolve())
    return str(rel) if str(rel) != "." else "."


class ReadFile(Action):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace and return its contents."
    risk = RiskLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {"path": _PATH_PROP},
        "required": ["path"],
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        resolve_in_workspace(p["path"], ctx.workspace)
        return None

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        path = resolve_in_workspace(p["path"], ctx.workspace)
        if not path.exists():
            return ActionResult.fail(f"File not found: {p['path']}")
        if path.is_dir():
            return ActionResult.fail(f"'{p['path']}' is a directory; use list_dir.")
        size = path.stat().st_size
        if size > MAX_READ_BYTES:
            return ActionResult.fail(f"File is {size} bytes; limit is {MAX_READ_BYTES}.")
        return ActionResult.ok(path.read_text(encoding="utf-8", errors="replace"), bytes=size)


class ListDir(Action):
    name = "list_dir"
    description = (
        "List files and folders in a workspace directory. "
        "Folders end with '/'. Use path '.' for the workspace root."
    )
    risk = RiskLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {"path": {**_PATH_PROP, "default": "."}},
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        resolve_in_workspace(p.get("path", "."), ctx.workspace)
        return None

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        path = resolve_in_workspace(p.get("path", "."), ctx.workspace)
        if not path.exists():
            return ActionResult.fail(f"Directory not found: {p.get('path', '.')}")
        if not path.is_dir():
            return ActionResult.fail(f"'{p.get('path')}' is a file, not a directory.")

        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
        lines = []
        for entry in entries[:MAX_LIST_ENTRIES]:
            if entry.is_dir():
                lines.append(f"{entry.name}/")
            else:
                lines.append(f"{entry.name}  ({entry.stat().st_size} bytes)")
        if len(entries) > MAX_LIST_ENTRIES:
            lines.append(f"...and {len(entries) - MAX_LIST_ENTRIES} more")
        header = f"Contents of {_rel(path, ctx)} ({len(entries)} entries):"
        return ActionResult.ok("\n".join([header, *lines]) if lines else f"{header}\n(empty)")


class WriteFile(Action):
    name = "write_file"
    description = (
        "Write text to a file in the workspace. Creates parent folders as needed. "
        "mode='overwrite' replaces the file; mode='append' adds to the end."
    )
    risk = RiskLevel.MODERATE
    input_schema = {
        "type": "object",
        "properties": {
            "path": _PATH_PROP,
            "content": {"type": "string", "maxLength": MAX_WRITE_CHARS},
            "mode": {"type": "string", "enum": ["overwrite", "append"], "default": "overwrite"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        path = resolve_in_workspace(p["path"], ctx.workspace)
        if path.is_dir():
            return f"'{p['path']}' is a directory."
        return None

    def effective_risk(self, p: dict[str, Any], ctx: ActionContext) -> RiskLevel:
        # Overwriting an existing file destroys its old contents -> destructive.
        path = resolve_in_workspace(p["path"], ctx.workspace)
        if path.exists() and p.get("mode", "overwrite") == "overwrite":
            return RiskLevel.DESTRUCTIVE
        return RiskLevel.MODERATE

    def describe(self, p: dict[str, Any], ctx: ActionContext) -> str:
        path = resolve_in_workspace(p["path"], ctx.workspace)
        mode = p.get("mode", "overwrite")
        note = ""
        if path.exists():
            note = " (OVERWRITES existing file)" if mode == "overwrite" else " (append)"
        return f"WRITE {len(p['content'])} chars to {p['path']}{note}"

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        path = resolve_in_workspace(p["path"], ctx.workspace)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = p.get("mode", "overwrite")
        with path.open("a" if mode == "append" else "w", encoding="utf-8") as f:
            f.write(p["content"])
        verb = "Appended" if mode == "append" else "Wrote"
        return ActionResult.ok(f"{verb} {len(p['content'])} chars to {_rel(path, ctx)}")


class DeleteFile(Action):
    name = "delete_file"
    description = "Permanently delete a single file in the workspace (not folders)."
    risk = RiskLevel.DESTRUCTIVE
    input_schema = {
        "type": "object",
        "properties": {"path": _PATH_PROP},
        "required": ["path"],
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        path = resolve_in_workspace(p["path"], ctx.workspace)
        if path == ctx.workspace.resolve():
            raise SafetyError("Refusing to delete the workspace root.")
        if not path.exists():
            return f"File not found: {p['path']}"
        if path.is_dir():
            return "delete_file only deletes files, not folders."
        return None

    def describe(self, p: dict[str, Any], ctx: ActionContext) -> str:
        path = resolve_in_workspace(p["path"], ctx.workspace)
        size = path.stat().st_size if path.exists() else 0
        return f"DELETE {p['path']} ({size} bytes) — cannot be undone"

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        problem = self.safety_check(p, ctx)  # re-check: file may have changed
        if problem:
            return ActionResult.fail(problem)
        path = resolve_in_workspace(p["path"], ctx.workspace)
        path.unlink()
        return ActionResult.ok(f"Deleted {_rel(path, ctx)}")
