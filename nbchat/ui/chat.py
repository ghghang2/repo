"""ChatUI wrapper.

This module re‑exports the :class:`ChatUI` class from the legacy
``nbchat_v2.py`` module and provides a small ``run_chat`` helper that
instantiates the UI.
"""

from nbchat_v2 import ChatUI

def run_chat():
    """Instantiate and display the chat UI.

    The function is a thin wrapper around :class:`ChatUI`.  Keeping it
    separate makes it easier to replace the implementation in the
    future without touching the rest of the code base.
    """

    ChatUI()