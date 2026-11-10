"""Tests for each action domain: files, database, HTTP (mocked), shell."""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from actions import run_action
from actions.base import Action
from models import ActionResult, RiskLevel


def run(registry, ctx, name, **params) -> ActionResult:
    return run_action(registry.get(name), params, ctx)


# ---- Files ------------------------------------------------------------------

def test_write_read_list_append_delete(registry, ctx, workspace):
    assert run(registry, ctx, "write_file", path="notes/a.txt", content="hello\n").success
    assert (workspace / "notes" / "a.txt").read_text() == "hello\n"

    assert run(registry, ctx, "write_file", path="notes/a.txt", content="world\n", mode="append").success
    assert run(registry, ctx, "read_file", path="notes/a.txt").output == "hello\nworld\n"

    listing = run(registry, ctx, "list_dir", path="notes")
    assert listing.success and "a.txt" in listing.output

    root = run(registry, ctx, "list_dir")
    assert "notes/" in root.output

    assert run(registry, ctx, "delete_file", path="notes/a.txt").success
    assert not (workspace / "notes" / "a.txt").exists()


def test_read_missing_file_fails_cleanly(registry, ctx):
    r = run(registry, ctx, "read_file", path="nope.txt")
    assert not r.success and "not found" in r.error


def test_read_directory_fails_cleanly(registry, ctx, workspace):
    (workspace / "d").mkdir()
    r = run(registry, ctx, "read_file", path="d")
    assert not r.success and "directory" in r.error


def test_overwrite_is_destructive_but_new_file_is_moderate(registry, ctx, workspace):
    wf = registry.get("write_file")
    assert wf.effective_risk({"path": "x.txt", "content": "a"}, ctx) is RiskLevel.MODERATE
    (workspace / "x.txt").write_text("old")
    assert wf.effective_risk({"path": "x.txt", "content": "a"}, ctx) is RiskLevel.DESTRUCTIVE
    assert wf.effective_risk({"path": "x.txt", "content": "a", "mode": "append"}, ctx) is RiskLevel.MODERATE
    assert "OVERWRITES" in wf.describe({"path": "x.txt", "content": "a"}, ctx)


def test_delete_refuses_directories(registry, ctx, workspace):
    (workspace / "d").mkdir()
    r = run(registry, ctx, "delete_file", path="d")
    assert not r.success and (workspace / "d").exists()


def test_actions_recheck_paths_even_if_called_directly(registry, ctx):
    """Defense in depth: bypassing the validator still can't escape the workspace."""
    r = run(registry, ctx, "read_file", path="../../etc/passwd")
    assert not r.success and "outside the workspace" in r.error


def test_output_truncated_for_llm(registry, ctx, workspace):
    (workspace / "big.txt").write_text("x" * 20_000)
    text = run(registry, ctx, "read_file", path="big.txt").to_llm_text(max_chars=1000)
    assert len(text) < 1100 and "truncated" in text


# ---- Database ---------------------------------------------------------------

def test_db_seeded_and_listable(registry, ctx):
    r = run(registry, ctx, "db_list_tables")
    assert r.success and "tasks(id, title, status, priority, created_at)" in r.output


def test_db_query_with_params(registry, ctx):
    r = run(registry, ctx, "db_query", sql="SELECT title FROM tasks WHERE priority = ?", params=[1])
    assert r.success and r.data["row_count"] == 2 and "Write project README" in r.output


def test_db_insert_then_query(registry, ctx):
    r = run(registry, ctx, "db_insert", table="tasks", values={"title": "Test task", "priority": 3})
    assert r.success and r.data["row_id"] == 5
    q = run(registry, ctx, "db_query", sql="SELECT title FROM tasks WHERE id = 5")
    assert "Test task" in q.output


def test_db_insert_unknown_column_and_table(registry, ctx):
    assert "Unknown column" in run(registry, ctx, "db_insert", table="tasks", values={"bogus": 1}).error
    assert "Unknown table" in run(registry, ctx, "db_insert", table="users", values={"x": 1}).error


def test_db_constraint_violation_fails_cleanly(registry, ctx):
    r = run(registry, ctx, "db_insert", table="tasks", values={"title": "x", "status": "invalid"})
    assert not r.success and "CHECK constraint" in r.error


def test_db_query_connection_is_read_only(registry, ctx):
    """Even a write disguised as WITH ... passes the keyword check but is blocked by SQLite."""
    run(registry, ctx, "db_list_tables")  # ensure DB exists
    r = run(registry, ctx, "db_query",
            sql="WITH x AS (SELECT 1) DELETE FROM tasks WHERE id IN (SELECT * FROM x)")
    assert not r.success and "readonly" in r.error.lower()
    with sqlite3.connect(ctx.db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 4


def test_db_sql_syntax_error_fails_cleanly(registry, ctx):
    r = run(registry, ctx, "db_query", sql="SELECT * FROM no_such_table")
    assert not r.success and "SQL error" in r.error


# ---- HTTP (mocked — no network) ---------------------------------------------

def test_get_weather(registry, ctx):
    r = run(registry, ctx, "get_weather", city="Paris")
    assert r.success and "Paris, France" in r.output and "12.5°C" in r.output and "overcast" in r.output


def test_get_weather_unknown_city(registry, ctx):
    r = run(registry, ctx, "get_weather", city="Nowhereville")
    assert not r.success and "Could not find" in r.error


def test_http_get_json(registry, ctx):
    r = run(registry, ctx, "http_get", url="https://catfact.ninja/fact")
    assert r.success and "HTTP 200" in r.output and "Cats sleep" in r.output


def test_http_get_non_2xx_is_failure(registry, ctx):
    r = run(registry, ctx, "http_get", url="https://catfact.ninja/missing")
    assert not r.success and "HTTP 404" in r.error


def test_http_get_blocked_domain_even_if_called_directly(registry, ctx):
    r = run(registry, ctx, "http_get", url="https://evil.example.com/")
    assert not r.success and "allowlist" in r.error


# ---- Shell ------------------------------------------------------------------

def _python_cmd():
    for name in ("python", "python3"):
        if shutil.which(name):
            return name
    pytest.skip("no python on PATH")


def test_run_allowed_command(registry, ctx):
    r = run(registry, ctx, "run_command", command=f"{_python_cmd()} --version")
    assert r.success and "exit code: 0" in r.output and "Python" in r.output


def test_run_blocked_command_even_if_called_directly(registry, ctx):
    r = run(registry, ctx, "run_command", command="rm -rf /")
    assert not r.success and "Blocked" in r.error


def test_shell_env_has_no_secrets(monkeypatch):
    from actions.shell import _scrubbed_env
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("MY_TOKEN", "t")
    env = _scrubbed_env()
    assert "ANTHROPIC_API_KEY" not in env and "MY_TOKEN" not in env


# ---- run_action never raises ------------------------------------------------

class Boom(Action):
    name = "boom"
    description = "always crashes"
    risk = RiskLevel.SAFE
    input_schema = {"type": "object"}

    def execute(self, params, ctx):
        raise RuntimeError("kaboom")


def test_run_action_converts_exceptions(ctx):
    r = run_action(Boom(), {}, ctx)
    assert not r.success and "RuntimeError: kaboom" in r.error


def test_registry_rejects_duplicates_and_bad_schemas(registry):
    with pytest.raises(ValueError):
        registry.register(registry.get("read_file").__class__())

    class BadSchema(Boom):
        name = "bad"
        input_schema = {"type": "not-a-type"}

    from jsonschema.exceptions import SchemaError
    with pytest.raises(SchemaError):
        registry.register(BadSchema())
