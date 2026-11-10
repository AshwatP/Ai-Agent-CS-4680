"""
config.py — Centralized, validated settings for the agent.

All configuration comes from environment variables (optionally loaded from a
`.env` file). Nothing secret is ever hardcoded. Settings are immutable once
loaded so no module can change them mid-run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


class ConfigError(Exception):
    """Raised when configuration is missing or invalid."""


def _int_env(name: str, default: int, minimum: int = 1) -> int:
    """Read a positive integer from the environment with a clear error message."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


@dataclass(frozen=True)
class Settings:
    api_key: str | None
    model: str
    workspace: Path
    log_path: Path
    max_iterations: int
    max_tokens: int
    command_timeout: int
    http_timeout: int
    max_output_chars: int

    def require_api_key(self) -> str:
        """Return the API key, or raise a user-friendly error if it's missing."""
        if not self.api_key or self.api_key.startswith("sk-ant-your-key"):
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or export ANTHROPIC_API_KEY in your shell."
            )
        return self.api_key


def load_settings(env_file: str | Path | None = ".env") -> Settings:
    """
    Load settings from the environment (and `.env` if present).

    The API key is NOT required here, so tests and offline tools can load
    settings without one. Call `settings.require_api_key()` where it's needed.
    Directories for the workspace and log file are created if missing.
    """
    if env_file and Path(env_file).exists():
        # override=False: real environment variables win over the .env file.
        load_dotenv(env_file, override=False)

    workspace = Path(os.getenv("AGENT_WORKSPACE", "./workspace")).expanduser().resolve()
    log_path = Path(os.getenv("AGENT_LOG_PATH", "./logs/audit.jsonl")).expanduser().resolve()

    workspace.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        workspace=workspace,
        log_path=log_path,
        max_iterations=_int_env("AGENT_MAX_ITERATIONS", 10),
        max_tokens=_int_env("AGENT_MAX_TOKENS", 2048, minimum=64),
        command_timeout=_int_env("AGENT_COMMAND_TIMEOUT", 10),
        http_timeout=_int_env("AGENT_HTTP_TIMEOUT", 10),
        max_output_chars=_int_env("AGENT_MAX_OUTPUT_CHARS", 8000, minimum=100),
    )