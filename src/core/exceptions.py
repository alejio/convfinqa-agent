"""
Custom exception classes for ConvFinQA application.
"""

from typing import Any


class ConvFinQAError(Exception):
    """Base exception class for ConvFinQA application."""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class DataLoadingError(ConvFinQAError):
    """Exception raised when dataset loading fails."""

    def __init__(
        self,
        message: str,
        file_path: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.file_path = file_path


class TableValidationError(ConvFinQAError):
    """Exception raised when table data validation fails."""

    def __init__(
        self,
        message: str,
        table_id: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.table_id = table_id


class SchemaDetectionError(ConvFinQAError):
    """Exception raised when table schema detection fails."""

    def __init__(
        self,
        message: str,
        table_name: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.table_name = table_name


class RecordNotFoundError(ConvFinQAError):
    """Exception raised when a requested record is not found."""

    def __init__(self, record_id: str, available_records: list[str] | None = None):
        message = f"Record '{record_id}' not found"
        details: dict[str, Any] = {"record_id": record_id}
        if available_records:
            details["available_count"] = len(available_records)
            details["similar_records"] = [
                r for r in available_records if record_id.lower() in r.lower()
            ][:5]
        super().__init__(message, details)
        self.record_id = record_id
        self.available_records = available_records


class DataQualityError(ConvFinQAError):
    """Exception raised when data quality issues are detected."""

    def __init__(
        self,
        message: str,
        severity: str = "warning",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.severity = severity


class NumericConversionError(ConvFinQAError):
    """Exception raised when numeric conversion fails."""

    def __init__(
        self,
        message: str,
        column_name: str | None = None,
        value: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, details)
        self.column_name = column_name
        self.value = value
