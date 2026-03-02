"""Entry point for the Auto Resolver TUI.

Usage:
    python -m forecasting_tools.agents_and_tools.auto_resolver.tui
    python -m forecasting_tools.agents_and_tools.auto_resolver.tui --tournament 32813
    python -m forecasting_tools.agents_and_tools.auto_resolver.tui --tournament fall-aib-2025
    python -m forecasting_tools.agents_and_tools.auto_resolver.tui --question 12345 --question 67890
    python -m forecasting_tools.agents_and_tools.auto_resolver.tui --concurrency 5
"""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto Resolver TUI -- interactive agentic question resolution",
    )
    def _parse_tournament_id(value: str) -> int | str:
        """Parse a tournament ID as int if numeric, otherwise keep as string slug."""
        try:
            return int(value)
        except ValueError:
            return value

    parser.add_argument(
        "--tournament",
        type=_parse_tournament_id,
        action="append",
        default=[],
        help="Tournament ID or slug to load on startup (can be repeated)",
    )
    parser.add_argument(
        "--question",
        type=int,
        action="append",
        default=[],
        help="Question post ID to load on startup (can be repeated)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Max concurrent resolutions (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging to stderr",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING

    # Do NOT call logging.basicConfig() here -- it would install a
    # StreamHandler on sys.stderr which corrupts the Textual TUI display.
    # Instead, set the root logger level and let the App install a custom
    # handler that routes records into a TUI log panel.
    logging.getLogger().setLevel(log_level)

    from forecasting_tools.agents_and_tools.auto_resolver.tui.app import (
        AutoResolverApp,
    )

    app = AutoResolverApp(
        max_concurrency=args.concurrency,
        initial_tournaments=args.tournament,
        initial_questions=args.question,
        log_level=log_level,
    )
    app.run()


if __name__ == "__main__":
    main()
