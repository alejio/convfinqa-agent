"""Core components: models, exceptions, and logging."""

from .exceptions import ConvFinQAError, DataLoadingError, SchemaDetectionError
from .logger import get_logger
from .models import (
    ColumnType,
    ConvFinQADataset,
    Dialogue,
    Document,
    FinancialTable,
    Record,
    TableColumn,
    TableSchema,
)

__all__ = [
    # Models
    "ColumnType",
    "TableColumn",
    "TableSchema",
    "FinancialTable",
    "Document",
    "Dialogue",
    "Record",
    "ConvFinQADataset",
    # Exceptions
    "ConvFinQAError",
    "SchemaDetectionError",
    "DataLoadingError",
    # Logger
    "get_logger",
]
