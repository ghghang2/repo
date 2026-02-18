"""Metrics updater helper.

The original implementation lives inside ``ChatUI``.  To keep the
behaviour identical we simply expose a function that forwards to the
original logic.  This module can later be extended to provide a fully
independent implementation.
"""

from nbchat_v2 import ChatUI

def start_metrics_updater(metrics_output_widget):
    """Start the background thread that updates *metrics_output_widget*.

    The function currently delegates to the legacy implementation
    because the UI logic is tightly coupled with the widget.  It is
    left here as a placeholder for future refactoring.
    """

    # The original logic is inside ChatUI._start_metrics_updater.
    # We create a dummy ChatUI instance and replace its metrics_output
    # widget with the one passed in.
    ui = ChatUI()
    ui.metrics_output = metrics_output_widget
    ui._start_metrics_updater()