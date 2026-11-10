"""End-to-end tests of the agent loop with a scripted (fake) LLM."""

from __future__ import annotations

from agent_loop import EventKind
from llm_client import LLMRetryExhaustedError
from models import ToolResult
from tests.conftest import call


def kinds(agent):
    return [e.kind for e in agent.events]


def tool_results_sent(llm, turn):
    """ToolResult blocks the agent sent on LLM call number `turn` (0-based)."""
    return [b for b in llm.received[turn][-1].content if isinstance(b, ToolResult)]


def test_end_to_end_multi_step(make_agent, workspace):
    """Read DB, write a file (confirmed), then give a final answer."""
    agent, llm = make_agent(
        [
            ("Checking tasks.", [call("db_query", "q1", sql="SELECT title FROM tasks WHERE status='todo'")]),
            ("Saving them.", [call("write_file", "w1", path="todo.txt", content="Add unit tests\nRecord demo video\n")]),
            ("Saved 2 todo tasks to todo.txt.", []),
        ],
        answers=[True],
    )
    result = agent.run("Save my todo tasks to a file")

    assert result.stop_reason == "final" and result.iterations == 3
    assert result.final_text == "Saved 2 todo tasks to todo.txt."
    assert [(a.tool, a.status) for a in result.actions] == [("db_query", "success"), ("write_file", "success")]
    assert (workspace / "todo.txt").read_text().startswith("Add unit tests")

    # The DB result was fed back to the LLM on the second call.
    first_result = tool_results_sent(llm, 1)[0]
    assert first_result.tool_call_id == "q1" and "Record demo video" in first_result.content

    assert EventKind.AWAITING_CONFIRMATION in kinds(agent)  # write needed confirmation
    assert kinds(agent)[-1] is EventKind.FINAL


def test_safe_actions_do_not_ask(make_agent):
    agent, _ = make_agent([("", [call("list_dir")]), ("done", [])], answers=[])
    agent.run("list")
    assert EventKind.AWAITING_CONFIRMATION not in kinds(agent)


def test_denied_action_is_reported_to_llm_not_executed(make_agent):
    agent, llm = make_agent([("", [call("read_file", "r1", path="../../etc/passwd")]), ("Blocked.", [])])
    result = agent.run("read passwd")
    assert result.actions[0].status == "denied"
    sent = tool_results_sent(llm, 1)[0]
    assert sent.is_error and "DENIED" in sent.content


def test_unknown_tool_is_denied(make_agent):
    agent, _ = make_agent([("", [call("launch_missiles")]), ("ok", [])])
    assert agent.run("x").actions[0].status == "denied"


def test_declined_destructive_action_is_not_executed(make_agent, workspace):
    (workspace / "keep.txt").write_text("important")
    agent, llm = make_agent([("", [call("delete_file", "d1", path="keep.txt")]), ("Not deleted.", [])],
                            answers=[False])
    result = agent.run("delete keep.txt")
    assert (workspace / "keep.txt").exists()
    assert result.actions[0].status == "declined"
    assert "DECLINED" in tool_results_sent(llm, 1)[0].content


def test_auto_yes_skips_moderate_but_not_destructive(make_agent, workspace):
    (workspace / "f.txt").write_text("x")
    agent, _ = make_agent(
        [("", [call("write_file", path="new.txt", content="a"), call("delete_file", path="f.txt")]), ("ok", [])],
        answers=[False],  # only ONE prompt should appear — for the delete
        auto_yes=True,
    )
    result = agent.run("x")
    assert [(a.tool, a.status) for a in result.actions] == [("write_file", "success"), ("delete_file", "declined")]
    assert kinds(agent).count(EventKind.AWAITING_CONFIRMATION) == 1
    assert (workspace / "f.txt").exists()


def test_failed_action_does_not_crash_agent(make_agent):
    agent, llm = make_agent([("", [call("read_file", path="missing.txt")]), ("File does not exist.", [])])
    result = agent.run("read missing")
    assert result.stop_reason == "final" and result.actions[0].status == "failed"
    assert tool_results_sent(llm, 1)[0].is_error


def test_llm_error_ends_request_and_rolls_back_history(make_agent):
    agent, _ = make_agent([("Hi!", []), LLMRetryExhaustedError("API down")])
    agent.run("hello")
    before = len(agent.history)
    result = agent.run("second request")
    assert result.stop_reason == "error" and "API down" in result.final_text
    assert len(agent.history) == before  # failed turn removed
    assert EventKind.ERROR in kinds(agent)


def test_iteration_limit(make_agent):
    agent, _ = make_agent([("", [call("list_dir", f"l{i}")]) for i in range(3)], max_iterations=3)
    result = agent.run("loop forever")
    assert result.stop_reason == "limit" and result.iterations == 3
    assert agent.history[-1].role == "assistant"  # roles still alternate


def test_conversation_memory_across_requests(make_agent):
    agent, llm = make_agent([("First answer.", []), ("Second answer.", [])])
    agent.run("first")
    agent.run("second")
    assert [m.role for m in llm.received[1]] == ["user", "assistant", "user"]
    agent.reset()
    assert agent.history == []


def test_ui_callback_errors_do_not_break_agent(make_agent):
    agent, _ = make_agent([("done", [])])
    agent.on_event = lambda e: (_ for _ in ()).throw(RuntimeError("UI bug"))
    assert agent.run("x").final_text == "done"


def test_audit_log_traces_every_action_to_its_request(make_agent, audit):
    agent, _ = make_agent(
        [("", [call("db_list_tables", "a1"), call("read_file", "a2", path="../x")]), ("done", [])]
    )
    result = agent.run("trace me")
    events = audit.events_for_request(result.request_id)
    names = [e["event"] for e in events]

    assert names[0] == "user_request" and names[-1] == "final_answer"
    assert events[0]["text"] == "trace me"
    assert {"action_proposed", "action_validated", "action_executed", "action_denied"} <= set(names)
    # Every action event carries both ids.
    for e in events:
        if e["event"].startswith("action_"):
            assert e["request_id"] == result.request_id and e["action_id"] in {"a1", "a2"}
