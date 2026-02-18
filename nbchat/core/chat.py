"""ChatUI implementation.

This module contains the :class:`ChatUI` class extracted from the
legacy ``nbchat_v2.py``.  All helper functions that were previously
defined in that file are moved to :mod:`nbchat.core.utils` and imported
here.  The rest of the code is unchanged.
"""

import ipywidgets as widgets
import json
import re
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import markdown  # for markdown rendering
from IPython.display import display

from nbchat.core.utils import lazy_import, md_to_html

# ----------------------------------------------------------------------
# Main Chat UI class
# ----------------------------------------------------------------------
class ChatUI:
    """Chat interface with streaming, reasoning, and safe tool execution."""

    MAX_TOOL_TURNS = 5

    def __init__(self):
        db = lazy_import("app.db")
        db.init_db()

        config = lazy_import("app.config")
        self.default_system_prompt = config.DEFAULT_SYSTEM_PROMPT
        self.model_name = config.MODEL_NAME

        self.session_id = str(uuid.uuid4())
        # history: (role, content, tool_id, tool_name, tool_args)
        # role can be: "user", "assistant", "analysis", "tool", "assistant_full"
        self.history: List[Tuple[str, str, str, str, str]] = []
        self.system_prompt = self.default_system_prompt

        self.session_ids = db.get_session_ids()

        self._create_widgets()
        self._start_metrics_updater()
        self._load_history()
        display(self.layout)