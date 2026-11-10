"""
actions/database.py — SQLite operations on workspace/agent.db.

Safety approach:
  * db_query runs on a READ-ONLY connection and only accepts SELECT / WITH,
    so even a cleverly-crafted query cannot modify data.
  * db_insert never puts user values into SQL text — values are bound
    parameters, and table/column names are checked against the real schema.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any

from actions.base import Action, ActionContext
from models import ActionResult, RiskLevel
from safety import SafetyError

MAX_ROWS = 100
_IDENT = r"^[A-Za-z_][A-Za-z0-9_]*$"
_IDENT_RE = re.compile(_IDENT)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'todo' CHECK (status IN ('todo', 'doing', 'done')),
    priority   INTEGER NOT NULL DEFAULT 2 CHECK (priority BETWEEN 1 AND 3),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""
_SEED_ROWS = [
    ("Write project README", "doing", 1),
    ("Add unit tests", "todo", 1),
    ("Record demo video", "todo", 2),
    ("Set up repository", "done", 3),
]


def ensure_database(db_path: Path) -> None:
    """Create the demo schema and seed rows on first use."""
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA_SQL)
        if conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0:
            conn.executemany(
                "INSERT INTO tasks (title, status, priority) VALUES (?, ?, ?)", _SEED_ROWS
            )


def _readonly(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True)


def _user_tables(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA table_info("{table}")').fetchall()]


def _format_table(columns: list[str], rows: list[tuple]) -> str:
    """Plain-text table that reads well for both humans and the LLM."""
    if not rows:
        return " | ".join(columns) + "\n(no rows)"
    cells = [[("NULL" if v is None else str(v)) for v in row] for row in rows]
    widths = [max(len(c), *(len(r[i]) for r in cells)) for i, c in enumerate(columns)]
    line = lambda vals: " | ".join(v.ljust(w) for v, w in zip(vals, widths))  # noqa: E731
    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([line(columns), sep, *(line(r) for r in cells)])


class DbListTables(Action):
    name = "db_list_tables"
    description = "List tables and their columns in the local SQLite database."
    risk = RiskLevel.SAFE
    input_schema = {"type": "object", "properties": {}, "additionalProperties": False}

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        ensure_database(ctx.db_path)
        with _readonly(ctx.db_path) as conn:
            lines = [f"{t}({', '.join(_columns(conn, t))})" for t in _user_tables(conn)]
        return ActionResult.ok("\n".join(lines) or "(no tables)")


class DbQuery(Action):
    name = "db_query"
    description = (
        "Run a read-only SQL SELECT on the local SQLite database. "
        f"Returns up to {MAX_ROWS} rows. Use '?' placeholders with 'params' for values."
    )
    risk = RiskLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {
            "sql": {"type": "string", "minLength": 1, "maxLength": 2000},
            "params": {
                "type": "array",
                "items": {"type": ["string", "number", "boolean", "null"]},
                "maxItems": 50,
            },
        },
        "required": ["sql"],
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        sql = p["sql"].strip().rstrip(";").strip()
        first = sql.split(None, 1)[0].upper() if sql else ""
        if first not in {"SELECT", "WITH"}:
            raise SafetyError("db_query only allows SELECT statements (use db_insert to add rows).")
        if ";" in sql:
            raise SafetyError("Only one SQL statement is allowed per query.")
        return None

    def describe(self, p: dict[str, Any], ctx: ActionContext) -> str:
        return f"SQL (read-only): {p['sql']}"

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        self.safety_check(p, ctx)
        ensure_database(ctx.db_path)
        sql = p["sql"].strip().rstrip(";")
        try:
            with _readonly(ctx.db_path) as conn:  # read-only: writes fail at the engine
                cur = conn.execute(sql, p.get("params") or [])
                columns = [d[0] for d in cur.description or []]
                rows = cur.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as exc:
            return ActionResult.fail(f"SQL error: {exc}")

        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        text = _format_table(columns, rows)
        if truncated:
            text += f"\n...(showing first {MAX_ROWS} rows)"
        return ActionResult.ok(text, row_count=len(rows))


class DbInsert(Action):
    name = "db_insert"
    description = (
        "Insert one row into a table in the local SQLite database. "
        "Call db_list_tables first if you don't know the columns."
    )
    risk = RiskLevel.MODERATE
    input_schema = {
        "type": "object",
        "properties": {
            "table": {"type": "string", "pattern": _IDENT, "maxLength": 64},
            "values": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 50,
                "propertyNames": {"pattern": _IDENT},
                "additionalProperties": {"type": ["string", "number", "boolean", "null"]},
            },
        },
        "required": ["table", "values"],
        "additionalProperties": False,
    }

    def safety_check(self, p: dict[str, Any], ctx: ActionContext) -> str | None:
        # Validate BEFORE asking the user to confirm, so they never approve a doomed insert.
        ensure_database(ctx.db_path)
        with _readonly(ctx.db_path) as conn:
            tables = _user_tables(conn)
            if p["table"] not in tables:
                return f"Unknown table '{p['table']}'. Existing tables: {', '.join(tables) or 'none'}."
            cols = set(_columns(conn, p["table"]))
        unknown = [c for c in p["values"] if c not in cols]
        if unknown:
            return f"Unknown column(s) for {p['table']}: {', '.join(unknown)}. Columns: {', '.join(sorted(cols))}."
        return None

    def describe(self, p: dict[str, Any], ctx: ActionContext) -> str:
        return f"INSERT into {p['table']}: {p['values']}"

    def execute(self, p: dict[str, Any], ctx: ActionContext) -> ActionResult:
        problem = self.safety_check(p, ctx)
        if problem:
            return ActionResult.fail(problem)
        cols = list(p["values"])
        assert all(_IDENT_RE.match(c) for c in cols) and _IDENT_RE.match(p["table"])
        col_sql = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join("?" for _ in cols)
        try:
            with sqlite3.connect(ctx.db_path) as conn:
                cur = conn.execute(
                    f'INSERT INTO "{p["table"]}" ({col_sql}) VALUES ({placeholders})',
                    [p["values"][c] for c in cols],
                )
                row_id = cur.lastrowid
        except sqlite3.Error as exc:  # e.g. CHECK / NOT NULL constraint
            return ActionResult.fail(f"SQL error: {exc}")
        return ActionResult.ok(f"Inserted row id {row_id} into {p['table']}.", row_id=row_id)
