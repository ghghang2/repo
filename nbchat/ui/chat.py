"""ChatUI wrapper.

This module provides the :class:`ChatUI` class and a small ``run_chat`` helper
that instantiates the UI.
"""

from nbchat.ui.chatui import ChatUI


def run_chat():
    """Instantiate and display the chat UI.

    The function is a thin wrapper around :class:`ChatUI`.
    """
    ChatUI()