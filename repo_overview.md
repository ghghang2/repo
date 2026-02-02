| Relative path | Function | Description |
|---------------|----------|-------------|
| app/chat.py | build_messages | Return the list of messages to send to the chat model.  Parameters ---------- history     List of ``(user, assistant)`` pairs that have already happened. system_prompt     The system message that sets the model behaviour. repo_docs     Optional Markdown string that contains the extracted repo source. user_input     The new user message that will trigger the assistant reply. |
| app/chat.py | stream_and_collect | Stream the assistant response while capturing tool calls.  The function writes the incremental assistant content to the supplied Streamlit ``placeholder`` and returns a tuple of the complete assistant text and a list of tool calls (or ``None`` if no tool call was emitted). |
| app/chat.py | process_tool_calls | Execute each tool that the model requested and keep asking the model for further replies until it stops calling tools.  Parameters ---------- client     The OpenAI client used to stream assistant replies. messages     The conversation history that will be extended with the tool‑call     messages and the tool replies. tools     The list of OpenAI‑compatible tool definitions that will be passed     to the ``chat.completions.create`` call. placeholder     Streamlit placeholder that will receive the intermediate     assistant output. tool_calls     The list of tool‑call objects produced by     :func:`stream_and_collect`.  The function may return a new     list of calls that the model wants to make after the tool     result is sent back.  Returns ------- tuple     ``(full_text, remaining_tool_calls)``.  *full_text* contains     the cumulative assistant reply **including** the text produced     by the tool calls.  *remaining_tool_calls* is ``None`` when the     model finished asking for tools; otherwise it is the list of calls     that still need to be handled. |
| app/client.py | get_client | Return a client that talks to the local OpenAI‑compatible server. |
| app/db.py | init_db | Create the database file and the chat_log table if they do not exist.  The function is idempotent – calling it repeatedly has no adverse effect.  It should be invoked once during application startup. |
| app/db.py | log_message | Persist a single chat line.  Parameters ---------- session_id     Identifier of the chat session – e.g. a user ID or a UUID. role     Either ``"user"`` or ``"assistant"``. content     The raw text sent or received. |
| app/db.py | load_history | Return the last *limit* chat pairs for the given session.  The return value is a list of ``(user_msg, assistant_msg)`` tuples. If *limit* is ``None`` the entire conversation is returned. |
| app/db.py | get_session_ids | Return a list of all distinct session identifiers stored in the DB. |
| app/docs_extractor.py | walk_python_files | Return all *.py files sorted alphabetically. |
| app/docs_extractor.py | write_docs | Append file path + code to *out*. |
| app/docs_extractor.py | extract | Extract the repo into a Markdown file and return the path. |
| app/docs_extractor.py | main | CLI entry point. |
| app/push_to_github.py | main | Create/attach the remote, pull, commit and push. |
| app/remote.py | _token | Return the GitHub PAT from the environment. |
| app/remote.py | _remote_url | HTTPS URL that contains the PAT – used only for git push. |
| app/tools/__init__.py | _generate_schema |  |
| app/tools/__init__.py | get_tools | Return the list of tools formatted for chat.completions.create. |
| app/tools/apply_patch.py | _apply_patch | Apply *patch_text* to the file or directory specified by *path*.  Parameters ---------- path:     File or directory path relative to the repository root. patch_text:     Unified diff string.  Returns ------- str     JSON string with either ``result`` or ``error``. |
| app/tools/create_file.py | _safe_resolve | Resolve ``rel_path`` against ``repo_root`` and ensure the result does **not** escape the repository root (prevents directory traversal). |
| app/tools/create_file.py | _create_file | Create a new file at ``path`` (relative to the repository root) with the supplied ``content``.  Parameters ---------- path     File path relative to the repo root.  ``path`` may contain     directory separators but **must not** escape the root. content     Raw text to write into the file.  Returns ------- str     JSON string.  On success:      .. code-block:: json          { "result": "File created: <path>" }      On failure:      .. code-block:: json          { "error": "<exception message>" } |
| app/tools/generate_table.py | walk_python_files |  |
| app/tools/generate_table.py | extract_functions_from_file | Return a list of (function_name, docstring) for top‑level functions.  Functions defined inside classes or other functions are ignored. |
| app/tools/generate_table.py | build_markdown_table |  |
| app/tools/generate_table.py | func | Generate a markdown table of all Python functions in the repo.  The table is written to ``function_table.md`` in the repository root. |
| app/tools/get_stock_price.py | _get_stock_price | Return the current stock price for *ticker* as a JSON string.  Parameters ---------- ticker: str     Stock symbol (e.g. ``"AAPL"``).  Case‑insensitive.  Returns ------- str     JSON string with ``ticker`` and ``price`` keys.  If the ticker     is unknown, ``price`` is set to ``"unknown"``. |
| app/tools/get_weather.py | _get_weather | Return a short weather description for *city*.  Parameters ---------- city : str     The name of the city to query (e.g. "Taipei").  Returns ------- str     JSON string. On success:          {"city":"Taipei","weather":"☀️  +61°F"}      On error:          {"error":"<error message>"} |
| app/tools/run_command.py | _safe_resolve | Resolve ``rel_path`` against ``repo_root`` and ensure the result does **not** escape the repository root (prevents directory traversal). |
| app/tools/run_command.py | _run_command | Execute ``command`` in the repository root (or a sub‑directory if ``cwd`` is provided) and return a JSON string with:     * ``stdout``     * ``stderr``     * ``exit_code`` Any exception is converted to an error JSON. |
| app/tools/run_tests.py | _run_tests | Execute `pytest -q` in the repository root and return JSON. |
| app/utils.py | build_api_messages | Convert local chat history into the format expected by the OpenAI API, optionally adding a tool list. |
| app/utils.py | stream_response | Yield the cumulative assistant reply while streaming. Also returns any tool call(s) that the model requested. |
| app.py | refresh_docs | Run the repository extractor and return its Markdown output. |
| app.py | is_repo_up_to_date | Return ``True`` if the local HEAD equals ``origin/main`` and the working tree is clean. |
| app.py | main |  |
| run.py | _run | Convenience wrapper around subprocess.run. |
| run.py | _is_port_free | Return True if the port is not currently bound. |
| run.py | _wait_for | Poll a URL until it returns 200 or timeout expires. |
| run.py | _save_service_info | Persist the running process IDs and the public tunnel URL. |
| run.py | main | Start all services and record their state. |
| run.py | _load_service_info |  |
| run.py | status | Print a quick report of the running services. |
| run.py | stop | Terminate all services and clean up. |
| tests/test_apply_patch_tool.py | test_apply_patch_tool_success | Test that the apply_patch tool can apply a simple patch.  The test creates a file in the repository root, writes a patch that changes its content, and ensures that the file content is updated. |
| tests/test_apply_patch_tool.py | test_apply_patch_tool_error | Test that the tool reports an error when git apply fails.  The patch references a non-existent file, which should cause the ``git apply`` command to fail. |
| tests/test_basic.py | test_basic |  |
| tests/test_create_file_tool.py | test_create_file_creates_file_with_content | Test that the create_file tool creates a file with the specified content. |
| tests/test_get_stock_price.py | test_known_ticker |  |
| tests/test_get_stock_price.py | test_unknown_ticker |  |
| tests/test_run_command_tool.py | test_run_command_basic | Verify that a simple command returns stdout, stderr and exit code. |
| tests/test_run_command_tool.py | test_run_command_with_cwd | Verify that the cwd argument correctly changes the working directory. |
| tests/test_run_command_tool.py | test_run_command_error | Verify that a non-existent command returns a non-zero exit code. |