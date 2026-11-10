"""
logger.py — Append-only JSON Lines audit log.

Every significant event (user request, proposed action, validation decision,
confirmation, execution outcome, LLM errors) is written as one JSON object per
line. Each event carries:

  * request_id — one per user request; ties every action back to its cause
  * action_id  — one per tool call (when applicable)

Example line:
  {"ts": "2026-11-09T18:02:11.123Z", "event": "action_executed",
   "request_id": "req_ab12...", "action_id": "toolu_01...", "tool": "read_file",
   "params": {"path": "notes.txt"}, "success": true}
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    """Thread-safe, append-only JSONL audit logger."""

    # Event names used across the codebase (kept here as the single source of truth).
    USER_REQUEST = "user_request"
    LLM_CALL = "llm_call"
    LLM_ERROR = "llm_error"
    ACTION_PROPOSED = "action_proposed"
    ACTION_VALIDATED = "action_validated"
    ACTION_DENIED = "action_denied"
    CONFIRMATION = "confirmation"
    ACTION_EXECUTED = "action_executed"
    ACTION_FAILED = "action_failed"
    FINAL_ANSWER = "final_answer"
    LOOP_LIMIT = "loop_limit_reached"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def new_request_id() -> str:
        return f"req_{uuid.uuid4().hex[:12]}"

    def log(
        self,
        event: str,
        request_id: str,
        action_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        """Append one event. Never raises — a logging failure must not crash the agent."""
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "event": event,
            "request_id": request_id,
        }
        if action_id is not None:
            record["action_id"] = action_id
        record.update(fields)

        try:
            line = json.dumps(record, default=str, ensure_ascii=False)
            with self._lock, self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        except OSError as exc:  # disk full, permissions, etc.
            # Fall back to stderr so the event isn't silently lost.
            import sys
            print(f"[audit-log write failed: {exc}] {record}", file=sys.stderr)
        return record

    def tail(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the last n events (used by the /log command)."""
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        events = []
        for line in lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip a corrupted line rather than fail
        return events

    def events_for_request(self, request_id: str) -> list[dict[str, Any]]:
        """Return every event for one request — the audit-trace view."""
        if not self.path.exists():
            return []
        out = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("request_id") == request_id:
                    out.append(rec)
        return out
