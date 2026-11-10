"""Tests for safety.py: path confinement, command policy, schema validation, confirmation."""

from __future__ import annotations

import os

import pytest

import safety
from models import RiskLevel, ToolCall
from safety import SafetyError, check_command, needs_confirmation, resolve_in_workspace


# ---- Path traversal ---------------------------------------------------------

@pytest.mark.parametrize("bad", [
    "../secret.txt",
    "../../etc/passwd",
    "notes/../../outside.txt",
    "~/.ssh/id_rsa",
    "/etc/passwd",
    "C:\\Windows\\win.ini" if os.name == "nt" else "/root/.bashrc",
])
def test_paths_outside_workspace_are_rejected(workspace, bad):
    with pytest.raises(SafetyError, match="outside the workspace"):
        resolve_in_workspace(bad, workspace)


@pytest.mark.parametrize("good", ["a.txt", "notes/todo.md", "./x/../y.txt", "."])
def test_paths_inside_workspace_are_allowed(workspace, good):
    resolved = resolve_in_workspace(good, workspace)
    assert resolved == workspace or resolved.is_relative_to(workspace)


def test_absolute_path_inside_workspace_is_allowed(workspace):
    assert resolve_in_workspace(str(workspace / "a.txt"), workspace) == workspace / "a.txt"


@pytest.mark.parametrize("bad", ["", "   ", "a\x00b"])
def test_empty_or_null_paths_rejected(workspace, bad):
    with pytest.raises(SafetyError):
        resolve_in_workspace(bad, workspace)


def test_symlink_escaping_workspace_is_rejected(workspace, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not permitted on this system")
    with pytest.raises(SafetyError):
        resolve_in_workspace("link.txt", workspace)


# ---- Shell commands ---------------------------------------------------------

@pytest.mark.parametrize("cmd, reason", [
    ("rm -rf /", "deletion"),
    ("sudo ls", "privilege"),
    ("ls | sh", "operators"),
    ("ls; whoami", "operators"),
    ("echo hi && whoami", "operators"),
    ("echo hi > file.txt", "operators"),
    ("echo `whoami`", "operators"),
    ("echo $(whoami)", "substitution"),
    ("curl https://example.com", "blocked"),
    ("chmod 777 a.txt", "blocked"),
])
def test_dangerous_commands_blocked(workspace, cmd, reason):
    with pytest.raises(SafetyError, match=reason):
        check_command(cmd, workspace)


@pytest.mark.parametrize("cmd", [
    "nmap localhost", "powershell -c x", "bash -c ls", "del a.txt", "format c:",
])
def test_commands_not_on_allowlist_rejected(workspace, cmd):
    with pytest.raises(SafetyError, match="allowlist"):
        check_command(cmd, workspace)


def test_program_paths_rejected(workspace):
    with pytest.raises(SafetyError, match="bare command"):
        check_command("/bin/ls", workspace)


@pytest.mark.parametrize("cmd", ["python -c 'print(1)'", "python script.py", "python3"])
def test_python_only_version(workspace, cmd):
    with pytest.raises(SafetyError, match="--version"):
        check_command(cmd, workspace)


@pytest.mark.parametrize("cmd", ["cat /etc/passwd", "ls ..", "head -n 5 ../x.txt", "type ../x"])
def test_command_file_args_confined(workspace, cmd):
    with pytest.raises(SafetyError, match="outside the workspace"):
        check_command(cmd, workspace)


@pytest.mark.parametrize("cmd, argv", [
    ("ls -la", ["ls", "-la"]),
    ("wc -l notes.txt", ["wc", "-l", "notes.txt"]),
    ("echo 'hello world'", ["echo", "hello world"]),
    ("python --version", ["python", "--version"]),
    ("where python", ["where", "python"]),
])
def test_allowed_commands_parse(workspace, cmd, argv):
    assert check_command(cmd, workspace) == argv


def test_where_rejects_flags(workspace):
    with pytest.raises(SafetyError, match="program names"):
        check_command("where /r C:\\ secrets", workspace)


def test_unbalanced_quotes(workspace):
    with pytest.raises(SafetyError, match="parse"):
        check_command("echo 'oops", workspace)


class TestWindowsRules:
    """Simulate Windows so these run on any OS."""

    @pytest.fixture(autouse=True)
    def _windows(self, monkeypatch):
        monkeypatch.setattr(safety, "IS_WINDOWS", True)

    def test_switches_allowed(self, workspace):
        assert check_command("dir /b /a:h", workspace) == ["dir", "/b", "/a:h"]

    def test_backslash_paths_preserved(self, workspace):
        assert check_command("type notes\\a.txt", workspace) == ["type", "notes\\a.txt"]

    def test_long_slash_arg_is_treated_as_path(self, workspace):
        with pytest.raises(SafetyError, match="outside the workspace"):
            check_command("type /etc/passwd", workspace)

    @pytest.mark.parametrize("cmd", ["echo %PATH%", "echo a^b", "echo hi!"])
    def test_cmd_special_chars_blocked(self, workspace, cmd):
        with pytest.raises(SafetyError, match="not allowed"):
            check_command(cmd, workspace)


# ---- Confirmation policy ----------------------------------------------------

@pytest.mark.parametrize("risk, auto_yes, expected", [
    (RiskLevel.SAFE, False, False),
    (RiskLevel.SAFE, True, False),
    (RiskLevel.MODERATE, False, True),
    (RiskLevel.MODERATE, True, False),
    (RiskLevel.DESTRUCTIVE, False, True),
    (RiskLevel.DESTRUCTIVE, True, True),   # --yes can NEVER skip destructive
])
def test_confirmation_policy(risk, auto_yes, expected):
    assert needs_confirmation(risk, auto_yes) is expected


# ---- SafetyValidator (schema + domain checks) -------------------------------

def _v(validator, name, params):
    return validator.validate(ToolCall("t1", name, params))


def test_unknown_tool_rejected(validator):
    r = _v(validator, "format_hard_drive", {})
    assert not r.allowed and "Unknown tool" in r.reason


@pytest.mark.parametrize("name, params, fragment", [
    ("read_file", {}, "'path' is a required property"),
    ("read_file", {"path": 123}, "is not of type 'string'"),
    ("read_file", {"path": "a", "extra": True}, "Additional properties"),
    ("write_file", {"path": "a", "content": "x", "mode": "truncate"}, "is not one of"),
    ("db_insert", {"table": "tasks; DROP TABLE tasks", "values": {"title": "x"}}, "does not match"),
    ("db_insert", {"table": "tasks", "values": {}}, "should be non-empty"),
])
def test_schema_violations_rejected(validator, name, params, fragment):
    r = _v(validator, name, params)
    assert not r.allowed
    assert "Invalid parameters" in r.reason and fragment in r.reason


def test_non_dict_input_rejected(validator):
    r = validator.validate(ToolCall("t1", "read_file", ["not", "a", "dict"]))  # type: ignore[arg-type]
    assert not r.allowed


def test_domain_check_runs_after_schema(validator):
    r = _v(validator, "read_file", {"path": "../../etc/passwd"})
    assert not r.allowed and "outside the workspace" in r.reason


@pytest.mark.parametrize("url, fragment", [
    ("https://evil.example.com/x", "not on the allowlist"),
    ("http://api.github.com/", "https"),
    ("https://user:pw@api.github.com/", "credentials"),
    ("https://api.github.com:8443/", "ports"),
])
def test_url_checks(validator, url, fragment):
    r = _v(validator, "http_get", {"url": url})
    assert not r.allowed and fragment in r.reason


@pytest.mark.parametrize("sql, fragment", [
    ("DELETE FROM tasks", "only allows SELECT"),
    ("DROP TABLE tasks", "only allows SELECT"),
    ("SELECT 1; DROP TABLE tasks", "one SQL statement"),
])
def test_sql_checks(validator, sql, fragment):
    r = _v(validator, "db_query", {"sql": sql})
    assert not r.allowed and fragment in r.reason


def test_allowed_call_has_risk_and_preview(validator):
    r = _v(validator, "delete_file", {"path": "missing.txt"})
    assert not r.allowed  # file doesn't exist -> rejected before confirmation
    r = _v(validator, "write_file", {"path": "new.txt", "content": "hi"})
    assert r.allowed and r.risk is RiskLevel.MODERATE and "WRITE 2 chars" in r.preview
