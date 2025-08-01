"""Command-line interface components."""

from .error_handler import handle_cli_errors, handle_table_display_errors
from .main import app

__all__ = [
    "app",
    "handle_cli_errors",
    "handle_table_display_errors",
]
