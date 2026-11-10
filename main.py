"""
main.py — Terminal user interface (REPL) for the AI agent.

Usage:
    python main.py            # moderate actions ask for confirmation
    python main.py --yes      # skip confirmation for MODERATE actions only;
                              # destructive actions (delete/overwrite) still ask

Commands inside the REPL: /help  /history  /log [n]  /clear  /quit
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from agent_loop import Agent, AgentEvent, AgentRunResult, EventKind, build_agent
from config import ConfigError, Settings, load_settings
from llm_client import create_client
from models import RiskLevel, ToolCall
from safety import ValidationResult

RISK_STYLE = {
    RiskLevel.SAFE: "green",
    RiskLevel.MODERATE: "yellow",
    RiskLevel.DESTRUCTIVE: "bold red",
}

HELP_TEXT = """\
Type a request in plain English, for example:
  • What tasks are in the database?
  • Get the weather in Tokyo and save it to weather.txt
  • Add a task called "Study for exam" with priority 1
  • Delete weather.txt

Commands:
  /help          show this help
  /history       requests in this session and the actions they ran
  /log [n]       last n audit-log events (default 15)
  /clear         forget the conversation (start fresh)
  /quit          exit (Ctrl+C also works)

Safety: safe actions run immediately; moderate actions ask y/N (skipped with --yes);
destructive actions ALWAYS ask. All file access is limited to the workspace folder."""


@dataclass
class HistoryEntry:
    request: str
    result: AgentRunResult


class TerminalUI:
    """Renders agent events and asks for confirmations."""

    def __init__(self, console: Console):
        self.console = console
        self._status = None  # active "Thinking..." spinner, if any

    # -- spinner helpers --

    def _stop_spinner(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _start_spinner(self, text: str) -> None:
        self._stop_spinner()
        self._status = self.console.status(f"[cyan]{text}[/]", spinner="dots")
        self._status.start()

    # -- callbacks passed to the Agent / LLM client --

    def on_retry(self, attempt: int, max_retries: int, delay: float, reason: str) -> None:
        self.console.print(
            f"  [yellow]LLM API {reason} — retrying in {delay:.1f}s ({attempt}/{max_retries})[/]"
        )

    def on_event(self, e: AgentEvent) -> None:
        c = self.console
        if e.kind is EventKind.THINKING:
            self._start_spinner(f"Thinking ({e.message})...")
            return
        self._stop_spinner()

        name = e.call.name if e.call else ""
        if e.kind is EventKind.PLAN:
            c.print(Text(f"Plan: {e.message}", style="italic cyan"))
        elif e.kind is EventKind.PROPOSED:
            params = json.dumps(e.call.input, ensure_ascii=False) if e.call else ""
            if len(params) > 120:
                params = params[:117] + "..."
            c.print(f"  [dim][PENDING][/]  {name} [dim]{params}[/]")
        elif e.kind is EventKind.DENIED:
            c.print(f"  [bold red][DENIED][/]   {name}: {e.message}")
        elif e.kind is EventKind.RUNNING:
            style = RISK_STYLE.get(e.risk, "white")
            c.print(f"  [blue][RUNNING][/]  {e.preview} [{style}]({e.risk.value})[/]")
        elif e.kind is EventKind.SUCCESS:
            c.print(f"  [green][SUCCESS][/]  {name}")
            self._print_output(e.message)
        elif e.kind is EventKind.FAILED:
            c.print(f"  [red][FAILED][/]   {name}")
            self._print_output(e.message, style="red")
        elif e.kind is EventKind.DECLINED:
            c.print(f"  [yellow][DECLINED][/] {e.preview} — not executed")
        elif e.kind is EventKind.ERROR:
            c.print(Panel(e.message, title="Error", border_style="red"))
        elif e.kind is EventKind.LIMIT:
            c.print(Panel(e.message, title="Step limit reached", border_style="yellow"))
        # FINAL and AWAITING_CONFIRMATION are rendered elsewhere.

    def _print_output(self, text: str, style: str = "dim", max_lines: int = 6) -> None:
        lines = text.splitlines() or [""]
        shown = lines[:max_lines]
        for line in shown:
            self.console.print(Text(f"             {line[:140]}", style=style))
        if len(lines) > max_lines:
            self.console.print(Text(f"             ... ({len(lines) - max_lines} more lines)", style="dim"))

    def confirm(self, call: ToolCall, v: ValidationResult) -> bool:
        """Show exactly what will run, then ask y/N (default: No)."""
        self._stop_spinner()
        risk = v.risk or RiskLevel.MODERATE
        style = RISK_STYLE[risk]
        body = Text()
        body.append(f"{v.preview}\n\n", style="bold")
        body.append(f"tool:   {call.name}\n")
        body.append(f"params: {json.dumps(call.input, indent=2, ensure_ascii=False)[:1500]}")
        if risk is RiskLevel.DESTRUCTIVE:
            body.append("\n\nThis action is DESTRUCTIVE and cannot be undone.", style="bold red")
        self.console.print(Panel(body, title=f"Confirm {risk.value} action", border_style=style))
        try:
            return Confirm.ask("Allow this action?", default=False, console=self.console)
        except (KeyboardInterrupt, EOFError):
            self.console.print("[yellow]No answer — treating as declined.[/]")
            return False

    def final(self, text: str) -> None:
        self._stop_spinner()
        self.console.print(Panel(Markdown(text), title="Agent", border_style="green"))


class App:
    """The REPL: reads input, dispatches commands, runs the agent."""

    def __init__(self, agent: Agent, ui: TerminalUI, settings: Settings, auto_yes: bool):
        self.agent = agent
        self.ui = ui
        self.console = ui.console
        self.settings = settings
        self.auto_yes = auto_yes
        self.history: list[HistoryEntry] = []

    def banner(self) -> None:
        info = Table.grid(padding=(0, 2))
        info.add_row("Model", self.settings.model)
        info.add_row("Workspace", str(self.settings.workspace))
        info.add_row("Audit log", str(self.settings.log_path))
        info.add_row("Tools", ", ".join(a.name for a in self.agent.registry.all()))
        info.add_row(
            "Confirmations",
            "moderate: skipped (--yes) · destructive: always" if self.auto_yes
            else "moderate + destructive",
        )
        self.console.print(Panel(info, title="AI Action Agent", subtitle="/help for commands",
                                 border_style="cyan"))

    # -- commands --

    def cmd_history(self) -> None:
        if not self.history:
            self.console.print("[dim]No requests yet this session.[/]")
            return
        t = Table(title="Session history", show_lines=True)
        t.add_column("#", justify="right")
        t.add_column("Request")
        t.add_column("Actions")
        t.add_column("Outcome")
        t.add_column("request_id", style="dim")
        colors = {"success": "green", "failed": "red", "denied": "red", "declined": "yellow"}
        for i, h in enumerate(self.history, 1):
            actions = "\n".join(
                f"[{colors.get(a.status, 'white')}]{a.tool}: {a.status}[/]" for a in h.result.actions
            ) or "[dim](none)[/]"
            t.add_row(str(i), h.request, actions, h.result.stop_reason, h.result.request_id)
        self.console.print(t)

    def cmd_log(self, arg: str) -> None:
        n = int(arg) if arg.isdigit() else 15
        events = self.agent.audit.tail(n)
        if not events:
            self.console.print("[dim]Audit log is empty.[/]")
            return
        t = Table(title=f"Last {len(events)} audit events ({self.settings.log_path.name})")
        t.add_column("Time (UTC)", style="dim")
        t.add_column("Event")
        t.add_column("request_id", style="dim")
        t.add_column("Tool")
        t.add_column("Detail")
        for e in events:
            detail = (e.get("reason") or e.get("error") or e.get("text")
                      or ("approved" if e.get("approved") else "declined" if "approved" in e else "")
                      or ("ok" if e.get("success") else ""))
            t.add_row(e.get("ts", "")[11:19], e.get("event", ""), e.get("request_id", ""),
                      e.get("tool", ""), str(detail)[:60])
        self.console.print(t)

    def cmd_clear(self) -> None:
        self.agent.reset()
        self.console.print("[cyan]Conversation cleared. The agent no longer remembers earlier requests.[/]")

    # -- main loop --

    def handle(self, line: str) -> bool:
        """Process one input line. Returns False to quit."""
        line = line.strip()
        if not line:
            return True
        if line.startswith("/"):
            cmd, _, arg = line.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/quit", "/exit", "/q"):
                return False
            elif cmd == "/help":
                self.console.print(Panel(HELP_TEXT, title="Help", border_style="cyan"))
            elif cmd == "/history":
                self.cmd_history()
            elif cmd == "/log":
                self.cmd_log(arg.strip())
            elif cmd == "/clear":
                self.cmd_clear()
            else:
                self.console.print(f"[red]Unknown command {cmd}.[/] Type /help.")
            return True

        checkpoint = len(self.agent.history)
        try:
            result = self.agent.run(line)
        except KeyboardInterrupt:
            self.ui._stop_spinner()
            del self.agent.history[checkpoint:]  # keep the conversation valid
            self.console.print("[yellow]Interrupted. That request was cancelled.[/]")
            return True
        if result.stop_reason == "final":
            self.ui.final(result.final_text)
        self.history.append(HistoryEntry(line, result))
        return True

    def run(self) -> None:
        self.banner()
        while True:
            try:
                line = self.console.input("\n[bold cyan]You>[/] ")
            except (KeyboardInterrupt, EOFError):
                break
            if not self.handle(line):
                break
        self.console.print("[cyan]Goodbye![/]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AI agent that turns LLM output into safe actions.")
    parser.add_argument("--yes", action="store_true",
                        help="auto-approve MODERATE actions (destructive actions still ask)")
    args = parser.parse_args(argv)

    console = Console()
    try:
        settings = load_settings()
        ui = TerminalUI(console)
        llm = create_client(settings, on_retry=ui.on_retry)  # checks the API key
    except ConfigError as exc:
        console.print(Panel(str(exc), title="Configuration error", border_style="red"))
        return 1

    agent = build_agent(settings, llm, confirm=ui.confirm, on_event=ui.on_event, auto_yes=args.yes)
    App(agent, ui, settings, auto_yes=args.yes).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
