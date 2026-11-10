"""
actions/registry.py — Holds every available action and exports tool specs.
"""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator

from actions.base import Action


class ActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, Action] = {}

    def register(self, action: Action) -> None:
        # Catch developer mistakes at startup, not when the LLM calls the tool.
        Draft202012Validator.check_schema(action.input_schema)
        if action.name in self._actions:
            raise ValueError(f"Duplicate action name: {action.name}")
        self._actions[action.name] = action

    def get(self, name: str) -> Action | None:
        return self._actions.get(name)

    def all(self) -> list[Action]:
        return list(self._actions.values())

    def __contains__(self, name: str) -> bool:
        return name in self._actions

    def __len__(self) -> int:
        return len(self._actions)

    def tool_specs(self) -> list[dict[str, Any]]:
        """Provider-neutral tool list: name, description, input_schema.
        (This happens to match Anthropic's format; other adapters re-wrap it.)"""
        return [
            {"name": a.name, "description": a.description, "input_schema": a.input_schema}
            for a in self._actions.values()
        ]


def build_default_registry() -> ActionRegistry:
    """Registry with every built-in action."""
    from actions.database import DbInsert, DbListTables, DbQuery
    from actions.files import DeleteFile, ListDir, ReadFile, WriteFile
    from actions.http_api import GetWeather, HttpGet
    from actions.shell import RunCommand

    registry = ActionRegistry()
    for action in (
        ReadFile(), ListDir(), WriteFile(), DeleteFile(),
        RunCommand(),
        GetWeather(), HttpGet(),
        DbListTables(), DbQuery(), DbInsert(),
    ):
        registry.register(action)
    return registry
