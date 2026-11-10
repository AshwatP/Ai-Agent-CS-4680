# Demo Guide: AI Action Agent

This guide walks through a full demonstration of the agent using the terminal interface (`main.py`). It shows every required feature: LLM integration, interpreting and executing actions across domains, the user interface, and error handling and safety.

> **Core idea:** the LLM decides *what* to do; the code decides *whether* it's allowed and *how* it runs.

---

## 1. Setup (once)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your key:

```
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-5-5
```

Optional check that the LLM connection works:

```bash
python llm_client.py
```

Expected: a short plain-text reply, then `tool call -> echo({'message': 'it works'})` and `PASS`.

---

## 2. Start the agent

```bash
python main.py
```

A banner shows the model, the workspace folder, the audit log path, the 10 available tools, and the confirmation mode. Type requests at the `You>` prompt.

| Status label                 | Meaning                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------- |
| `Plan:`                    | The LLM's one-sentence plan before it acts                                   |
| `[PENDING]`                | The LLM proposed an action                                                   |
| `[RUNNING]`                | The action passed validation and is executing (risk level shown in brackets) |
| `[SUCCESS]` / `[FAILED]` | The result; the first lines of output are shown                              |
| `[DENIED]`                 | Blocked by a safety check. Never executed.                                   |
| `[DECLINED]`               | You answered "n" at the confirmation prompt. Never executed.                 |

The final answer appears in a green **Agent** panel.

> The LLM's wording and exact tool choices can vary slightly between runs. The safety behavior does not vary.

---

## 3. Demo script

Run these in order **in one session**. The agent remembers the conversation, so step 5 can refer to "that file".

### Scenario 1: Database read (safe, multi-step)

```
What tasks are in the database?
```

- **Expected:** Claude calls `db_list_tables` to learn the schema, then `db_query` with a `SELECT`. Both are **safe**, so there is no confirmation. The answer lists the 4 seeded tasks.
- **Point out:** the agent discovered the schema on its own. The query runs on a read-only database connection.

### Scenario 2: Two domains in one request (API + file, moderate)

```
Get the weather in Los Angeles and save a one-line summary to weather.txt
```

- **Expected:** `get_weather` runs right away (**safe**, calls the Open-Meteo API). Then `write_file` shows a yellow **Confirm moderate action** panel with the exact preview: `WRITE N chars to weather.txt`. Type `y`.
- **Point out:** one natural-language request became a chain of actions. Any action with side effects asks first, and the panel shows exactly what will happen.

### Scenario 3: System command (allowlisted)

```
What version of Python is installed?
```

- **Expected:** `run_command` with `python --version`. It's **moderate**, so type `y`. The output shows `exit code: 0` and the version.
- **Point out:** commands run without a real shell (no pipes or chaining), with a timeout, inside the workspace folder, and with API keys removed from the environment.

### Scenario 4: Database write (moderate)

```
Add a task called "Record demo video" with priority 1
```

- **Expected:** `db_insert` shows a confirmation panel; type `y`. Reply: `Inserted row id 5 into tasks.` Re-run Scenario 1 to show the new row.
- **Point out:** values are bound as parameters (no SQL injection), and table and column names are checked against the real schema *before* you're asked to confirm.

### Scenario 5: Destructive action with follow-up memory (always confirms)

```
Now delete that file
```

- **Expected:** Claude knows "that file" means `weather.txt` from Scenario 2. A **red** panel appears: `DELETE weather.txt (N bytes) — cannot be undone`.
  1. First type `n`. You'll see `[DECLINED] ... not executed`, and Claude acknowledges it without trying another way.
  2. Then ask again (`Ok, delete it`) and type `y` to show it succeeding.
- **Point out:** destructive actions always ask, even with `--yes` (see Scenario 8). When you decline, Claude is told not to retry.

### Scenario 6: Blocked: path traversal

```
Read the file ../../secret.txt
```

- **Expected:** `[DENIED] read_file: Path '../../secret.txt' resolves outside the workspace and is not allowed.` There is no confirmation prompt, because it never gets that far. Claude explains that the file is outside the sandbox.
- **Point out:** every file path is resolved, symlinks included, and must stay inside `workspace/`.

### Scenario 7: Blocked: dangerous command

```
Try running this command: curl https://example.com
```

- **Expected:** `[DENIED] run_command: Blocked command: this command is blocked.`
- Other examples that are blocked: `rm -rf workspace`, `ls | sort`, `sudo whoami`, `cat /etc/passwd`.
- **Note:** Claude sometimes refuses a clearly dangerous request on its own, without calling the tool. If that happens, say so in the video: nothing runs either way. Phrasing it as "Try running…" usually gets Claude to attempt it, so you can show the safety layer blocking it.

### Scenario 8: The `--yes` flag (optional)

Quit with `/quit`, then restart with:

```bash
python main.py --yes
```

```
Create notes.txt containing "hello"
Delete notes.txt
```

- **Expected:** the write runs **without** a prompt, since `--yes` skips moderate confirmations. The delete **still asks**.

| Risk level                                     | Default  | With `--yes`       |
| ---------------------------------------------- | -------- | -------------------- |
| Safe (read, list, query, weather)              | runs     | runs                 |
| Moderate (write new file, insert, run command) | asks y/N | runs                 |
| Destructive (delete, overwrite existing file)  | asks y/N | **still asks** |

---

## 4. Built-in commands

| Command      | What to show                                                                                                                      |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `/history` | A table of every request this session, the actions each ran with their status (success, denied, declined), and the `request_id` |
| `/log 15`  | The last 15 audit events: time, event, request ID, tool, detail                                                                   |
| `/clear`   | Forgets the conversation. Ask "what did I just delete?" afterwards to show it no longer knows.                                    |
| `/help`    | Usage, examples, and the safety summary                                                                                           |
| `/quit`    | Exit (Ctrl+C also works; Ctrl+C during a request cancels only that request)                                                       |

---

## 5. Auditability: trace an action back to its request

Every event is appended to `logs/audit.jsonl`, one JSON object per line. Each line carries a `request_id`, and each tool call also carries an `action_id`.

1. Run `/history` and copy the `request_id` for Scenario 5.
2. Open `logs/audit.jsonl` and search for that ID.

You'll see the full trail of that request. The lines below are an example; IDs and times will differ:

```json
{"ts": "...", "event": "user_request",     "request_id": "req_4502e29d6df2", "text": "Now delete that file"}
{"ts": "...", "event": "llm_call",         "request_id": "req_4502e29d6df2", "iteration": 1}
{"ts": "...", "event": "action_proposed",  "request_id": "req_4502e29d6df2", "action_id": "toolu_01...", "tool": "delete_file", "params": {"path": "weather.txt"}}
{"ts": "...", "event": "action_validated", "request_id": "req_4502e29d6df2", "action_id": "toolu_01...", "allowed": true, "risk": "destructive"}
{"ts": "...", "event": "confirmation",     "request_id": "req_4502e29d6df2", "action_id": "toolu_01...", "approved": false}
{"ts": "...", "event": "llm_call",         "request_id": "req_4502e29d6df2", "iteration": 2}
{"ts": "...", "event": "final_answer",     "request_id": "req_4502e29d6df2", "text": "Okay, I won't delete weather.txt..."}
```

**Point out:** the log records the original request, what the LLM proposed, the safety decision, the user's answer, and the outcome, all linked by one ID.

---

## 6. Troubleshooting

| Problem                                                               | Fix                                                                                                       |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'actions'`                    | Run from the project root, and check that `actions/` contains `__init__.py` plus the six action files |
| `ModuleNotFoundError: No module named 'httpx'` (or another package) | Activate the virtual environment, then `python -m pip install -r requirements.txt`                      |
| A deprecation warning about the model                                 | Set `ANTHROPIC_MODEL=claude-sonnet-5-5` in `.env`                                                     |
| `'cat' is not installed on this system` (Windows)                   | Expected: Unix tools don't exist on Windows. Ask for the Python version, or use `dir` / `type`        |
| Timestamps look hours off                                             | The audit log and database store UTC on purpose                                                           |
