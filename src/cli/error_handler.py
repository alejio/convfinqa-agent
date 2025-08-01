"""
Centralized error handling for CLI commands.
"""

import functools
from collections.abc import Callable
from typing import Any, TypeVar

from rich import print as rich_print

from ..core.exceptions import (
    ConvFinQAError,
    DataLoadingError,
    RecordNotFoundError,
    SchemaDetectionError,
    TableValidationError,
)
from ..core.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def handle_cli_errors(func: F) -> F:  # noqa: UP047
    """
    Decorator to handle CLI errors consistently across all commands.

    This decorator catches all ConvFinQA exceptions and formats them
    consistently with rich output, reducing code duplication.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except DataLoadingError as e:
            _handle_data_loading_error(e, func.__name__)
        except RecordNotFoundError as e:
            _handle_record_not_found_error(e, func.__name__)
        except SchemaDetectionError as e:
            _handle_schema_detection_error(e, func.__name__)
        except TableValidationError as e:
            _handle_table_validation_error(e, func.__name__)
        except ConvFinQAError as e:
            _handle_convfinqa_error(e, func.__name__)
        except Exception as e:
            _handle_unexpected_error(e, func.__name__)

    return wrapper  # type: ignore


def _handle_data_loading_error(error: DataLoadingError, command_name: str) -> None:
    """Handle DataLoadingError with file path and details."""
    rich_print(f"[red]Dataset Loading Error:[/red] {error.message}")

    if error.file_path:
        rich_print(f"[dim]File: {error.file_path}[/dim]")

    if error.details:
        rich_print(f"[dim]Details: {error.details}[/dim]")

    # Add helpful suggestions
    if "not found" in error.message.lower():
        rich_print(
            "[yellow]Suggestion:[/yellow] Check that the dataset file exists in the data/ directory"
        )
    elif "permission" in error.message.lower():
        rich_print(
            "[yellow]Suggestion:[/yellow] Check file permissions for the dataset file"
        )
    elif "json" in error.message.lower():
        rich_print(
            "[yellow]Suggestion:[/yellow] Verify the dataset file contains valid JSON"
        )


def _handle_record_not_found_error(
    error: RecordNotFoundError, command_name: str
) -> None:
    """Handle RecordNotFoundError with similar records and suggestions."""
    rich_print(f"[red]Record Not Found:[/red] {error.message}")

    if error.details.get("similar_records"):
        rich_print("[yellow]Similar records found:[/yellow]")
        for similar in error.details["similar_records"]:
            rich_print(f"  - {similar}")

    rich_print("[dim]Use 'list-records' to see all available records[/dim]")

    if error.details.get("available_count"):
        rich_print(
            f"[dim]Total available records: {error.details['available_count']}[/dim]"
        )


def _handle_schema_detection_error(
    error: SchemaDetectionError, command_name: str
) -> None:
    """Handle SchemaDetectionError with table name and schema details."""
    rich_print(f"[red]Schema Detection Error:[/red] {error.message}")

    if error.table_name:
        rich_print(f"[dim]Table Name: {error.table_name}[/dim]")

    if error.details:
        rich_print(f"[dim]Details: {error.details}[/dim]")

    # Add helpful suggestions based on error type
    if "empty" in error.message.lower():
        rich_print("[yellow]Suggestion:[/yellow] The table contains no data to analyze")
    elif "type detection" in error.message.lower():
        rich_print(
            "[yellow]Suggestion:[/yellow] The table contains mixed or invalid data types"
        )


def _handle_table_validation_error(
    error: TableValidationError, command_name: str
) -> None:
    """Handle TableValidationError with table ID and validation details."""
    rich_print(f"[red]Table Validation Error:[/red] {error.message}")

    if error.table_id:
        rich_print(f"[dim]Table ID: {error.table_id}[/dim]")

    if error.details:
        rich_print(f"[dim]Details: {error.details}[/dim]")

    # Add helpful suggestions
    if "convert" in error.message.lower():
        rich_print(
            "[yellow]Suggestion:[/yellow] The table data cannot be converted to DataFrame format"
        )
    elif "extract" in error.message.lower():
        rich_print(
            "[yellow]Suggestion:[/yellow] The table data structure is invalid or corrupted"
        )


def _handle_convfinqa_error(error: ConvFinQAError, command_name: str) -> None:
    """Handle generic ConvFinQAError with message and details."""
    rich_print(f"[red]Application Error:[/red] {error.message}")

    if error.details:
        rich_print(f"[dim]Details: {error.details}[/dim]")


def _handle_unexpected_error(error: Exception, command_name: str) -> None:
    """Handle unexpected errors with logging and user-friendly message."""
    rich_print(f"[red]Unexpected Error:[/red] {error}")
    rich_print(
        "[dim]This appears to be an unexpected error. Please check the logs for more details.[/dim]"
    )

    # Log the full error with context
    logger.error(f"Unexpected error in {command_name} command: {error}", exc_info=True)

    # Add helpful suggestion
    rich_print(
        "[yellow]Suggestion:[/yellow] If this error persists, please report it as a bug"
    )


def handle_table_display_errors(func: F) -> F:  # noqa: UP047
    """
    Decorator specifically for table display operations.

    This is a lighter-weight decorator for operations that might fail
    during table display but shouldn't crash the entire command.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except (TableValidationError, SchemaDetectionError) as e:
            rich_print(f"[red]Table Display Error:[/red] {e.message}")
            if (
                hasattr(e, "table_id")
                and getattr(e, "table_id", None)
                and isinstance(e, TableValidationError)
            ):
                rich_print(f"[dim]Table ID: {e.table_id}[/dim]")
            elif (
                hasattr(e, "table_name")
                and getattr(e, "table_name", None)
                and isinstance(e, SchemaDetectionError)
            ):
                rich_print(f"[dim]Table Name: {e.table_name}[/dim]")
            if e.details:
                rich_print(f"[dim]Details: {e.details}[/dim]")
        except Exception as e:
            rich_print(f"[red]Error displaying table:[/red] {e}")
            logger.error(
                f"Error displaying table in {func.__name__}: {e}", exc_info=True
            )

    return wrapper  # type: ignore
