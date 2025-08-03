"""
Unified error handling across all components.

This module consolidates the different error handling patterns that were
inconsistently implemented across tools, agents, evaluation, and other modules.
"""

import json
from typing import Any


class ErrorResponseHandler:
    """Unified error handling for consistent responses across all components."""

    @staticmethod
    def tool_error(
        message: str,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        """
        Create consistent JSON error response for tool functions.

        Consolidates the _error_response pattern from tools.py

        Args:
            message: Error message to display
            code: Optional error code for categorization
            details: Optional additional details about the error

        Returns:
            JSON-formatted error string
        """
        error_dict: dict[str, Any] = {
            "status": "error",
            "error": message,
            "message": message,
        }

        if code is not None:
            error_dict["code"] = code

        if details is not None:
            error_dict["details"] = details

        return json.dumps(error_dict)

    @staticmethod
    def agent_error(message: str, include_prefix: bool = True) -> str:
        """
        Create consistent string error response for agent functions.

        Consolidates the agent error pattern from agent.py

        Args:
            message: Error message to display
            include_prefix: Whether to include "I encountered an error" prefix

        Returns:
            Formatted error string for agent responses
        """
        if include_prefix:
            return f"I encountered an error while processing your question: {message}"
        else:
            return message

    @staticmethod
    def evaluation_error() -> bool:
        """
        Return consistent error response for evaluation functions.

        Consolidates the boolean error pattern from metrics.py

        Returns:
            False to indicate evaluation failure
        """
        return False

    @staticmethod
    def calculation_error(message: str, code: str = "CALCULATION_ERROR") -> str:
        """
        Create error response for calculation functions.

        Args:
            message: Error message describing the calculation failure
            code: Error code for the calculation error

        Returns:
            JSON-formatted error for calculation tools
        """
        return ErrorResponseHandler.tool_error(message, code)

    @staticmethod
    def validation_error(
        message: str, field: str | None = None, value: Any | None = None
    ) -> str:
        """
        Create error response for validation failures.

        Args:
            message: Validation error message
            field: Optional field name that failed validation
            value: Optional value that failed validation

        Returns:
            JSON-formatted validation error
        """
        details = {}
        if field is not None:
            details["field"] = field
        if value is not None:
            details["invalid_value"] = str(value)

        return ErrorResponseHandler.tool_error(
            message, code="VALIDATION_ERROR", details=details if details else None
        )

    @staticmethod
    def missing_context_error(context_type: str = "record") -> str:
        """
        Create error response for missing context.

        Args:
            context_type: Type of context that is missing

        Returns:
            JSON-formatted context error
        """
        return ErrorResponseHandler.tool_error(
            f"No {context_type} context set", code=f"NO_{context_type.upper()}_CONTEXT"
        )

    @staticmethod
    def parse_error(
        text: str, expected_format: str, parser_name: str = "parser"
    ) -> str:
        """
        Create error response for parsing failures.

        Args:
            text: Text that failed to parse
            expected_format: Expected format description
            parser_name: Name of the parser that failed

        Returns:
            JSON-formatted parse error
        """
        return ErrorResponseHandler.tool_error(
            f"{parser_name} failed to parse '{text[:50]}...' (expected: {expected_format})",
            code="PARSE_ERROR",
            details={
                "parser": parser_name,
                "expected_format": expected_format,
                "input_preview": text[:100] if text else None,
            },
        )


class ValidationError(Exception):
    """Custom exception for validation errors that provides structured error info."""

    def __init__(
        self,
        message: str,
        field: str | None = None,
        value: Any | None = None,
        code: str = "VALIDATION_ERROR",
    ):
        super().__init__(message)
        self.message = message
        self.field = field
        self.value = value
        self.code = code

    def to_json_error(self) -> str:
        """Convert to JSON error response."""
        return ErrorResponseHandler.validation_error(
            self.message, self.field, self.value
        )


class DSPyFallbackError(Exception):
    """Exception raised when DSPy operations fail and no fallback is available."""

    def __init__(self, operation_name: str, original_error: Exception):
        self.operation_name = operation_name
        self.original_error = original_error
        super().__init__(f"DSPy operation '{operation_name}' failed: {original_error}")

    def to_agent_error(self) -> str:
        """Convert to agent error response."""
        return ErrorResponseHandler.agent_error(
            f"Analysis system temporarily unavailable for {self.operation_name}"
        )
