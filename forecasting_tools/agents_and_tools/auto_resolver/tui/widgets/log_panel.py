"""Global log panel for displaying captured logging and stderr output."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import RichLog, Static


class LogPanel(Vertical):
    """Full-screen panel showing captured Python logging and stderr output.

    Toggled via the ``l`` keybinding.  All log records from every logger
    in the process (including third-party libraries like litellm, httpx,
    and the OpenAI SDK) are routed here instead of to stderr so they do
    not corrupt the TUI display.
    """

    DEFAULT_CSS = """
    LogPanel {
        width: 1fr;
        height: 1fr;
    }
    LogPanel > .log-panel-header {
        dock: top;
        height: auto;
        padding: 0 2;
        background: $surface;
        border-bottom: solid $primary;
        text-align: center;
        text-style: bold;
    }
    LogPanel > #log-output {
        height: 1fr;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Logs  [dim](press [bold]l[/bold] to return)[/dim]",
            classes="log-panel-header",
        )
        yield RichLog(
            id="log-output",
            highlight=True,
            markup=True,
            wrap=True,
        )

    @property
    def log_output(self) -> RichLog:
        return self.query_one("#log-output", RichLog)

    def append_log(self, text: str) -> None:
        """Append a pre-formatted log line to the panel."""
        self.log_output.write(text)

    def clear_logs(self) -> None:
        """Clear all log output."""
        self.log_output.clear()
