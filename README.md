# AI Action Agent

A terminal-based AI agent that turns plain-English requests into real actions — reading and writing files, querying a SQLite database, calling public web APIs, and running system commands — powered by Anthropic's Claude.

> **Core idea:** the LLM decides *what* to do; the code decides *whether* it's allowed and *how* it runs.

Built as a course project for **CS 4680**.

---

## Features

- **Natural-language control** — ask for something like *"Get the weather in Los Angeles and save a summary to weather.txt"* and the agent plans and chains the needed tool calls.
- **10 tools across 4 domains** — files, shell, HTTP APIs, and a local database.
- **Layered safety** — every action is schema-validated, sandboxed, and risk-rated before it runs.
- **Human-in-the-loop confirmation** — actions with side effects show an exact preview and ask before running.
- **Conversation memory** — follow-ups like *"now delete that file"* work within a session.
- **Full audit trail** — every request, proposed action, safety decision, and result is logged to `logs/audit.jsonl` and linked by a request ID.
- **Rich terminal UI** — live status labels, colored confirmation panels, `/history` and `/log` commands.

## How it works

```
 You ──► main.py (terminal UI)
            │
            ▼
       agent_loop.py ──► llm_client.py ──► Claude (tool use)
            │                                  │
            │◄──────── proposed tool call ─────┘
            ▼
        safety.py   (schema check, sandbox, allowlists, risk level)
            │
            ▼
     confirmation (if moderate / destructive)
            │
            ▼
        actions/*   (execute)  ──► result sent back to Claude
            │
            ▼
        logger.py   (audit.jsonl)
```

The loop repeats until Claude gives a final answer or the iteration limit is reached.

## Tools

| Domain   | Tool              | Risk                                  | What it does                                         |
| -------- | ----------------- | ------------------------------------- | ---------------------------------------------------- |
| Files    | `read_file`       | safe                                  | Read a UTF-8 text file in the workspace              |
| Files    | `list_dir`        | safe                                  | List a workspace folder                              |
| Files    | `write_file`      | moderate (destructive if overwriting) | Create or overwrite a file                           |
| Files    | `delete_file`     | destructive                           | Permanently delete one file                          |
| Shell    | `run_command`     | moderate                              | Run an allowlisted, read-only command                |
| HTTP     | `get_weather`     | safe                                  | Current weather for a city (Open-Meteo)              |
| HTTP     | `http_get`        | safe                                  | GET from an allowlisted public API                   |
| Database | `db_list_tables`  | safe                                  | Show tables and columns                              |
| Database | `db_query`        | safe                                  | Run a read-only `SELECT`                             |
| Database | `db_insert`       | moderate                              | Insert a row (parameterized)                         |

## Safety model

| Risk level      | Default        | With `--yes`       |
| --------------- | -------------- | ------------------ |
| Safe            | runs           | runs               |
| Moderate        | asks y/N       | runs               |
| Destructive     | asks y/N       | **still asks**     |

Guardrails enforced in code, regardless of what the LLM proposes:

- **File sandbox** — every path is resolved (symlinks included) and must stay inside `workspace/`.
- **Shell** — command allowlist (`ls`, `cat`, `dir`, `type`, `echo`, `python --version`, …), no real shell, no pipes/chaining/substitution, blocked commands (`rm`, `sudo`, `curl`, …), timeout, and API keys scrubbed from the child environment.
- **HTTP** — `https` only, domain allowlist, no redirects, no embedded credentials, default port only.
- **Database** — queries run on a read-only connection; inserts use bound parameters and table/column names are checked against the real schema.
- **Input validation** — every tool call is checked against its JSON Schema before anything runs.
- **Declines are respected** — if you answer "n", the agent is told not to retry another way.

## Getting started

### Requirements

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)

### Install

```bash
git clone <this-repo-url>
cd Ai-Agent-CS-4680
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Configure

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-5-5
```

Optional settings (defaults shown):

| Variable                 | Default              | Purpose                                 |
| ------------------------ | -------------------- | --------------------------------------- |
| `AGENT_WORKSPACE`        | `./workspace`        | Sandbox folder for file actions         |
| `AGENT_LOG_PATH`         | `./logs/audit.jsonl` | Audit log location                      |
| `AGENT_MAX_ITERATIONS`   | `10`                 | Max LLM ↔ tool rounds per request       |
| `AGENT_MAX_TOKENS`       | `2048`               | Max tokens per LLM response             |
| `AGENT_COMMAND_TIMEOUT`  | `10`                 | Shell command timeout (seconds)         |
| `AGENT_HTTP_TIMEOUT`     | `10`                 | HTTP request timeout (seconds)          |
| `AGENT_MAX_OUTPUT_CHARS` | `8000`               | Max tool output sent back to the LLM    |

Check the LLM connection (optional):

```bash
python llm_client.py
```

### Run

```bash
python main.py          # confirm moderate and destructive actions
python main.py --yes    # auto-approve moderate actions (destructive still asks)
```

## Usage

Example requests:

```
What tasks are in the database?
Get the weather in Los Angeles and save a one-line summary to weather.txt
What version of Python is installed?
Add a task called "Record demo video" with priority 1
Now delete that file
```

Built-in commands:

| Command       | Description                                    |
| ------------- | ---------------------------------------------- |
| `/history`    | Requests this session and the actions each ran |
| `/log [n]`    | Last *n* audit-log events                      |
| `/clear`      | Forget the conversation                        |
| `/help`       | Usage, examples, and safety summary            |
| `/quit`       | Exit (Ctrl+C cancels just the current request) |

For a full walkthrough, including blocked-action examples and how to trace a request through the audit log, see **[DEMO.md](DEMO.md)**.

## Testing

```bash
pytest
```

Tests cover the safety rules, each action, the agent loop, and the LLM client, and run without an API key.

## Project structure

```
├── main.py            # Terminal UI and entry point
├── agent_loop.py      # Plan → validate → confirm → execute loop
├── llm_client.py      # Anthropic API client with retries
├── safety.py          # Sandboxing, allowlists, risk/confirmation policy
├── config.py          # Settings loaded from environment / .env
├── logger.py          # JSONL audit logger
├── models.py          # Shared data types
├── actions/
│   ├── base.py        # Action base class
│   ├── registry.py    # Tool registry
│   ├── files.py       # File tools
│   ├── shell.py       # run_command
│   ├── http_api.py    # get_weather, http_get
│   └── database.py    # SQLite tools
├── tests/             # pytest suite
├── DEMO.md            # Step-by-step demo guide
└── requirements.txt
```

## Tech stack

[Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) · [httpx](https://www.python-httpx.org/) · [Rich](https://github.com/Textualize/rich) · [python-dotenv](https://github.com/theskumar/python-dotenv) · [jsonschema](https://python-jsonschema.readthedocs.io/) · [pytest](https://pytest.org/)
