| Relative path | Function | Description |
|---------------|----------|-------------|
| app/chat.py | build_messages | Return the list of messages to send to the chat model.  Parameters ---------- history     List of ``(user, assistant, tool_id, tool_name, tool_args)``     tuples that have already happened. system_prompt     The system message that sets the model behaviour. user_input     The new user message that will trigger the assistant reply. |
| app/chat.py | stream_and_collect | Stream the assistant response while capturing tool calls.  The function writes the incremental assistant content to a placeholder and returns a tuple of the complete assistant text, a list of tool calls (or ``None`` if no tool call was emitted), a boolean indicating if the assistant finished, and any reasoning text. |
| app/chat.py | process_tool_calls | Execute each tool that the model requested.  Parameters ---------- client     The OpenAI client used to stream assistant replies. messages     The conversation history that will be extended with the tool‑call     messages and the tool replies. session_id     Identifier of the chat session. history     Mutable list of ``(role, content, tool_id, tool_name, tool_args)``     tuples used to build the chat history. tools     The list of OpenAI‑compatible tool definitions. tool_calls     The list of tool‑call objects produced by :func:`stream_and_collect`. finished     Boolean indicating whether the assistant already finished a turn. assistant_text     Text that the assistant returned before the first tool call. reasoning_text     Any reasoning content produced by the model.  Returns ------- list     Updated history list. |
| app/client.py | get_client | Return a client that talks to the local OpenAI‑compatible server. |
| app/db.py | init_db | Create the database file and the chat_log table if they do not exist.  The function is idempotent – calling it repeatedly has no adverse effect.  It should be invoked once during application startup. |
| app/db.py | log_message | Persist a single chat line.  Parameters ---------- session_id     Identifier of the chat session – e.g. a user ID or a UUID. role     . content     The raw text sent or received. |
| app/db.py | log_tool_msg | Persist a single chat line.  Parameters ---------- session_id     Identifier of the chat session – e.g. a user ID or a UUID. role     . content     The raw text sent or received. |
| app/db.py | load_history | Return the last *limit* chat pairs for the given session.  The return value is a list of chat history. If *limit* is ``None`` the entire conversation is returned. |
| app/db.py | get_session_ids | Return a list of all distinct session identifiers stored in the DB. |
| app/docs_extractor.py | walk_python_files | Return all *.py files sorted alphabetically. |
| app/docs_extractor.py | write_docs | Append file path + code to *out*. |
| app/docs_extractor.py | extract | Extract the repo into a Markdown file and return the path. |
| app/docs_extractor.py | main | CLI entry point. |
| app/metrics_ui.py | changed_files |  |
| app/metrics_ui.py | parse_log | Reads the log and returns a formatted markdown string. |
| app/metrics_ui.py | _background_parser | Constantly updates the state so it's ready for any UI component. |
| app/metrics_ui.py | metrics_fragment | This refreshes every second automatically when the script is IDLE. |
| app/metrics_ui.py | display_metrics_panel | Main entry point for app.py sidebar. |
| app/push_to_github.py | main | Create/attach the remote, pull, commit and push. |
| app/remote.py | _token | Return the GitHub PAT from the environment. |
| app/remote.py | _remote_url | HTTPS URL that contains the PAT – used only for git push. |
| app/tools/__init__.py | _generate_schema |  |
| app/tools/__init__.py | get_tools | Return the list of tools formatted for chat.completions.create. |
| app/tools/apply_patch.py | _safe_resolve | Resolve ``rel_path`` against ``repo_root`` and guard against directory traversal. Normalize to NFKC to ensure characters like '..' or '/' aren't spoofed |
| app/tools/apply_patch.py | _extract_payload |  |
| app/tools/apply_patch.py | apply_diff |  |
| app/tools/apply_patch.py | _trim_overlap | Trims the end of ins_lines if they already exist at the start of following_lines. Prevents duplicate 'stitching' when the diff and file overlap. |
| app/tools/apply_patch.py | _normalize_diff_lines | Clean the diff and strip Unified Diff metadata headers. |
| app/tools/apply_patch.py | _detect_newline |  |
| app/tools/apply_patch.py | _is_done |  |
| app/tools/apply_patch.py | _read_str |  |
| app/tools/apply_patch.py | _parse_create_diff |  |
| app/tools/apply_patch.py | _parse_update_diff |  |
| app/tools/apply_patch.py | _advance_cursor |  |
| app/tools/apply_patch.py | _read_section |  |
| app/tools/apply_patch.py | _equals_slice | Helper to compare a slice of lines using a transformation function (like strip). |
| app/tools/apply_patch.py | _find_context |  |
| app/tools/apply_patch.py | _apply_chunks |  |
| app/tools/apply_patch.py | apply_patch | Apply a unified diff to a file inside the repository.  Parameters ---------- path : str     Relative file path (under the repo root). op_type : str     One of ``create``, ``update`` or ``delete``. diff : str     Unified diff string (ignored for ``delete``).  Returns ------- str     JSON string with either ``result`` or ``error``. |
| app/tools/browser.py | browser | Visit a webpage, perform optional interactions, and extract content.  This tool is STATELESS: It opens a browser, runs your commands, and closes. You cannot "keep" the browser open between calls.  Parameters ---------- url : str     The URL to visit. actions : List[Dict], optional     A list of interactions to perform before extracting data.     Supported action types:     - {"type": "click", "selector": "..."}     - {"type": "type", "selector": "...", "text": "..."}     - {"type": "wait", "selector": "..."} (or "timeout": ms)     - {"type": "screenshot", "path": "..."} selector : str, optional     A specific CSS selector to extract text from. If None, returns the full page text. **kwargs :     Handles "hallucinated" nested JSON arguments from some LLMs.  Returns ------- str     JSON string containing the extracted text, source, or operation results. |
| app/tools/create_file.py | _safe_resolve | Resolve ``rel_path`` against ``repo_root`` and ensure the result does **not** escape the repository root (prevents directory traversal). |
| app/tools/create_file.py | _create_file | Create a new file at ``path`` (relative to the repository root) with the supplied ``content``.  Parameters ---------- path     File path relative to the repo root.  ``path`` may contain     directory separators but **must not** escape the root. content     Raw text to write into the file.  Returns ------- str     JSON string.  On success:      .. code-block:: json          { "result": "File created: <path>" }      On failure:      .. code-block:: json          { "error": "<exception message>" } |
| app/tools/get_weather.py | _geocode_city | Return latitude and longitude for a given city name.  The function queries the OpenMeteo geocoding API and returns the first result.  It raises a :class:`ValueError` if the city cannot be found. |
| app/tools/get_weather.py | _fetch_weather | Fetch current and daily forecast weather data for the given coordinates and date.  Parameters ---------- lat, lon: float     Latitude and longitude of the location. date: str     ISO 8601 formatted date string (YYYY-MM-DD).  The API expects a     single day, so ``start_date`` and ``end_date`` are identical. |
| app/tools/get_weather.py | _get_weather | Retrieve current and forecast weather information for a given city and date.  Parameters ---------- city: str     The name of the city to look up. date: str, optional     The date for which to retrieve forecast data (ISO format YYYY-MM-DD).     If omitted or empty, today's date is used. |
| app/tools/repo_overview.py | walk_python_files | Return a sorted list of all ``.py`` files under *root*. |
| app/tools/repo_overview.py | extract_functions_from_file | Return a list of (function_name, docstring) for top‑level functions.  Functions defined inside classes or other functions are ignored. |
| app/tools/repo_overview.py | build_markdown_table |  |
| app/tools/repo_overview.py | func | Generate a markdown table of all top‑level Python functions.  The table is written to ``repo_overview.md`` in the repository root. |
| app/tools/run_command.py | _safe_resolve | Resolve ``rel_path`` against ``repo_root`` and ensure the result does **not** escape the repository root (prevents directory traversal). |
| app/tools/run_command.py | _run_command | Execute ``command`` in the repository root and return a JSON string with:     * ``stdout``     * ``stderr``     * ``exit_code`` Any exception is converted to an error JSON.  The ``cwd`` argument is accepted for backward compatibility but ignored; the command is always executed in the repository root. |
| app/tools/run_tests.py | _run_tests | Execute `pytest -q` in the repository root and return JSON. |
| app/utils.py | build_api_messages | Convert local chat history into the format expected by the OpenAI API, optionally adding a tool list. |
| app/utils.py | stream_response | Yield the cumulative assistant reply while streaming. Also returns any tool call(s) that the model requested. |
| app.py | refresh_docs | Run the repository extractor and return its Markdown output. |
| app.py | is_repo_up_to_date |  |
| app.py | main |  |
| run.py | _run | Convenience wrapper around subprocess.run. |
| run.py | _is_port_free | Return True if the port is not currently bound. |
| run.py | _wait_for | Poll a URL until it returns 200 or timeout expires. |
| run.py | _save_service_info | Persist the running process IDs and the public tunnel URL. |
| run.py | main | Start all services and record their state. |
| run.py | _load_service_info |  |
| run.py | status | Print a quick report of the running services. |
| run.py | stop | Terminate all services and clean up. |
| tests/test_apply_patch_all.py | _relative_path | Return a path relative to the repository root. |
| tests/test_apply_patch_all.py | test_apply_patch_update_with_context | Update a file using a diff that includes context lines. |
| tests/test_apply_patch_all.py | test_apply_patch_create_nested_dirs | Creating a file in nested directories should create the directories. |
| tests/test_apply_patch_all.py | test_apply_patch_error_invalid_diff | Providing a create diff that does not start with '+' should error. |
| tests/test_apply_patch_all.py | test_apply_patch_update_no_change | Updating with a diff that results in no changes should succeed. |
| tests/test_apply_patch_all.py | test_apply_patch_delete_nonexistent | Deleting a non‑existent file should succeed without error. |
| tests/test_apply_patch_all.py | test_apply_patch_idempotency_import | Re-applying the same 'import errno' patch should not result in duplicates. |
| tests/test_apply_patch_all.py | test_apply_patch_stutter_overlap | Applying a diff that overlaps with existing code should not 'stutter' (triple the lines). |
| tests/test_create_file_tool.py | test_create_file_creates_file_with_content | Test that the create_file tool creates a file with the specified content. |
| tests/test_get_weather_tool.py | test_get_weather_success | Test that the get_weather tool returns a result for a valid city and date. |
| tests/test_get_weather_tool.py | test_get_weather_error | Test that the get_weather tool returns an error for an invalid city. |
| tests/test_get_weather_tool.py | test_get_weather_date_omitted | Test that omitting the date defaults to today. |
| tests/test_run_command_tool.py | test_run_command_basic | Verify that a simple command returns stdout, stderr and exit code. |
| tests/test_run_command_tool.py | test_run_command_in_repo_root | Verify that the command always runs in the repository root.  The :mod:`app.tools.run_command` implementation no longer accepts a ``cwd`` argument – it always executes in the repository root.  This test checks that the working directory reported by ``pwd`` matches the repository root path. |
| tests/test_run_command_tool.py | test_run_command_error | Verify that a non-existent command returns a non-zero exit code. |