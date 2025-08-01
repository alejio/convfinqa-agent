"""
Data models for ConvFinQA dataset and table schema management.
"""

from datetime import datetime
from enum import Enum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .exceptions import SchemaDetectionError


class ColumnType(str, Enum):
    """Enumeration of possible column types in financial tables."""

    NUMERIC = "numeric"
    TEXT = "text"
    DATE = "date"
    PERCENTAGE = "percentage"
    CURRENCY = "currency"


class TableColumn(BaseModel):
    """Metadata for a single table column."""

    name: str
    column_type: ColumnType
    nullable: bool = True
    description: str | None = None

    model_config = ConfigDict(use_enum_values=True)


class TableSchema(BaseModel):
    """Schema definition for a financial table."""

    name: str
    columns: list[TableColumn]
    row_count: int
    description: str | None = None

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, name: str, description: str | None = None
    ) -> "TableSchema":
        """Create a TableSchema from a pandas DataFrame with robust type detection."""
        if df.empty:
            raise SchemaDetectionError(
                "Cannot create schema from empty DataFrame",
                table_name=name,
                details={"shape": df.shape},
            )

        columns = []
        detection_warnings: list[str] = []

        for col_name in df.columns:
            col_data = df[col_name]
            assert isinstance(col_data, pd.Series)

            # Determine column type based on data with fallback logic
            try:
                col_type = cls._detect_column_type(
                    col_data, str(col_name), detection_warnings
                )
            except Exception as e:
                detection_warnings.append(
                    f"Column '{col_name}' type detection failed: {e}"
                )
                col_type = ColumnType.TEXT  # Fallback to TEXT

            columns.append(
                TableColumn(
                    name=str(col_name),
                    column_type=col_type,
                    nullable=bool(col_data.isnull().any()),
                    description=None,
                )
            )

        # Log warnings if any
        if detection_warnings:
            from .logger import get_logger

            logger = get_logger(__name__)
            logger.warning(f"Schema detection warnings for table '{name}':")
            for warning in detection_warnings:
                logger.warning(f"  - {warning}")

        return cls(
            name=name, columns=columns, row_count=len(df), description=description
        )

    @staticmethod
    def _detect_column_type(
        col_data: pd.Series, col_name: str, warnings: list[str]
    ) -> ColumnType:
        """Detect column type with robust fallback logic."""
        # Handle empty columns
        if col_data.empty or col_data.isnull().all():
            warnings.append(f"Column '{col_name}' is empty or all null")
            return ColumnType.TEXT

        # Check pandas detected types first
        if col_data.dtype in ["int64", "float64", "int32", "float32"]:
            return ColumnType.NUMERIC

        # For object columns, try multiple detection strategies
        if col_data.dtype == "object":
            non_null_data = col_data.dropna()

            if non_null_data.empty:
                warnings.append(f"Column '{col_name}' has no non-null values")
                return ColumnType.TEXT

            # Strategy 1: Try direct numeric conversion
            try:
                pd.to_numeric(non_null_data, errors="raise")
                return ColumnType.NUMERIC
            except (ValueError, TypeError):
                pass

            # Strategy 2: Try converting common numeric patterns
            # Handle parentheses for negative numbers: (123) -> -123
            # Handle currency symbols: $123.45 -> 123.45
            # Handle commas: 1,234.56 -> 1234.56
            try:
                cleaned_data = non_null_data.astype(str)
                # Remove common non-numeric characters
                cleaned_data = cleaned_data.str.replace(r"[\$,]", "", regex=True)
                # Convert (123) to -123
                cleaned_data = cleaned_data.str.replace(
                    r"\(([0-9.,]+)\)", r"-\1", regex=True
                )
                # Try conversion again
                pd.to_numeric(cleaned_data, errors="raise")
                warnings.append(
                    f"Column '{col_name}' contains formatted numeric values"
                )
                return ColumnType.NUMERIC
            except (ValueError, TypeError):
                pass

            # Strategy 3: Check for percentage patterns
            if non_null_data.astype(str).str.contains("%").any():
                try:
                    # Remove % and convert
                    cleaned_data = non_null_data.astype(str).str.replace("%", "")
                    pd.to_numeric(cleaned_data, errors="raise")
                    return ColumnType.PERCENTAGE
                except (ValueError, TypeError):
                    pass

            # Strategy 4: Check for date patterns
            if (
                non_null_data.astype(str)
                .str.contains(r"\d{4}|\d{2}/\d{2}|\d{2}-\d{2}", regex=True)
                .any()
            ):
                try:
                    pd.to_datetime(non_null_data, errors="raise")
                    return ColumnType.DATE
                except (ValueError, TypeError):
                    pass

        # Default to TEXT if all strategies fail
        return ColumnType.TEXT


class FinancialTable(BaseModel):
    """Represents a financial table with data and metadata."""

    data: dict[str, dict[str, float | int | str]]
    table_schema: TableSchema

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the table data to a pandas DataFrame with robust error handling."""
        if not self.data:
            raise SchemaDetectionError(
                "Cannot convert empty table data to DataFrame",
                details={"table_schema": self.table_schema.name},
            )

        try:
            # The data structure is: {column_name: {row_name: value}}
            # We need to transpose this to have rows as index and columns as columns
            df = pd.DataFrame(self.data)
        except Exception as e:
            raise SchemaDetectionError(
                f"Failed to create DataFrame from table data: {e}",
                table_name=self.table_schema.name,
                details={"error_type": type(e).__name__},
            ) from e

        # Ensure numeric types are properly converted with error handling
        conversion_errors = []

        for col_name in df.columns:
            column_info = next(
                (c for c in self.table_schema.columns if c.name == col_name), None
            )

            if column_info and column_info.column_type == ColumnType.NUMERIC:
                try:
                    # Try direct conversion first
                    df[col_name] = pd.to_numeric(df[col_name], errors="coerce")

                    # Check if conversion resulted in too many NaN values
                    nan_count = df[col_name].isnull().sum()
                    total_count = len(df[col_name])

                    if nan_count > 0 and nan_count / total_count > 0.5:
                        conversion_errors.append(
                            f"Column '{col_name}': {nan_count}/{total_count} values failed numeric conversion"
                        )

                except Exception as e:
                    conversion_errors.append(
                        f"Column '{col_name}' numeric conversion failed: {e}"
                    )
                    # Keep original data if conversion fails
                    pass

        # Log conversion errors as warnings
        if conversion_errors:
            from .logger import get_logger

            logger = get_logger(__name__)
            logger.warning(
                f"DataFrame conversion warnings for table '{self.table_schema.name}':"
            )
            for error in conversion_errors:
                logger.warning(f"  - {error}")

        return df


class Document(BaseModel):
    """Document containing pre/post text and financial table."""

    pre_text: str
    post_text: str
    table: dict[str, dict[str, float | int | str]]

    def get_financial_table(self, table_name: str = "financial_data") -> FinancialTable:
        """Convert the raw table data to a FinancialTable with schema."""
        if not self.table:
            raise SchemaDetectionError(
                "Cannot create financial table from empty table data",
                table_name=table_name,
            )

        try:
            # Create DataFrame to analyze schema
            df = pd.DataFrame(self.table)
        except Exception as e:
            raise SchemaDetectionError(
                f"Failed to create DataFrame from table data: {e}",
                table_name=table_name,
                details={"error_type": type(e).__name__},
            ) from e

        # Create schema with error handling
        try:
            schema = TableSchema.from_dataframe(
                df, table_name, "Financial data extracted from document"
            )
        except Exception as e:
            raise SchemaDetectionError(
                f"Failed to create schema for table: {e}",
                table_name=table_name,
                details={"error_type": type(e).__name__, "table_shape": df.shape},
            ) from e

        return FinancialTable(data=self.table, table_schema=schema)


class Dialogue(BaseModel):
    """Dialogue structure in ConvFinQA dataset."""

    conv_questions: list[str]
    conv_answers: list[str]
    turn_program: list[str]
    executed_answers: list[float | int | str]
    qa_split: list[bool]

    def get_conversation_turns(self) -> list[tuple[str, str]]:
        """Get conversation as list of (question, answer) tuples."""
        return list(zip(self.conv_questions, self.conv_answers, strict=False))


class Record(BaseModel):
    """Complete ConvFinQA record with document and dialogue."""

    id: str
    doc: Document
    dialogue: Dialogue
    features: dict[str, Any] | None = None

    def get_table_names(self) -> list[str]:
        """Get list of available table names in this record."""
        return [f"{self.id}_table"]

    def get_financial_table(self, table_name: str | None = None) -> FinancialTable:
        """Get the financial table for this record."""
        if table_name is None:
            table_name = f"{self.id}_table"

        try:
            return self.doc.get_financial_table(table_name)
        except Exception as e:
            raise SchemaDetectionError(
                f"Failed to get financial table for record '{self.id}': {e}",
                table_name=table_name,
                details={"record_id": self.id, "error_type": type(e).__name__},
            ) from e


class ConvFinQADataset(BaseModel):
    """Complete ConvFinQA dataset with train/dev splits."""

    train: list[Record]
    dev: list[Record]

    def get_record_by_id(self, record_id: str) -> Record | None:
        """Find a record by its ID across all splits."""
        for record in self.train + self.dev:
            if record.id == record_id:
                return record
        return None

    def get_all_records(self) -> list[Record]:
        """Get all records from both splits."""
        return self.train + self.dev


class ConversationTurn(BaseModel):
    """A single turn in a conversation with context tracking."""

    turn_id: str
    user_message: str
    assistant_response: str
    timestamp: datetime
    tool_calls: list[dict[str, Any]] = []
    referenced_entities: list[str] = []  # Tables, columns, values mentioned
    computation_results: list[dict[str, Any]] = []


class ConversationState(BaseModel):
    """Complete conversation state with multi-turn context."""

    session_id: str
    record_id: str
    turns: list[ConversationTurn] = []
    current_context: dict[str, Any] = {}  # Last referenced tables, values, etc.
    entity_references: dict[str, list[str]] = {}  # Maps entities to their mentions
    created_at: datetime
    updated_at: datetime

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a new conversation turn and update context."""
        self.turns.append(turn)
        self.updated_at = datetime.now()
        self._update_context_from_turn(turn)

    def _update_context_from_turn(self, turn: ConversationTurn) -> None:
        """Update conversation context based on the latest turn."""
        # Track referenced entities
        for entity in turn.referenced_entities:
            if entity not in self.entity_references:
                self.entity_references[entity] = []
            self.entity_references[entity].append(turn.turn_id)

        # Update current context with latest tool calls and results
        if turn.tool_calls:
            self.current_context["last_tool_calls"] = turn.tool_calls
        if turn.computation_results:
            self.current_context["last_computation_results"] = turn.computation_results

    def get_conversation_context(self, max_turns: int = 5) -> str:
        """Generate a context string for the LLM with recent conversation history."""
        if not self.turns:
            return "This is the start of a new conversation."

        recent_turns = self.turns[-max_turns:]
        context_parts = ["Recent conversation history:"]

        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_message}")
            context_parts.append(f"Assistant: {turn.assistant_response}")

        # Add current context information
        if self.current_context:
            context_parts.append("\nCurrent context:")
            for key, value in self.current_context.items():
                if key == "last_tool_calls" and value:
                    context_parts.append(
                        f"- Last tool calls: {[call.get('tool_name', 'unknown') for call in value]}"
                    )
                elif key == "last_computation_results" and value:
                    context_parts.append(
                        "- Recent computations available for reference"
                    )

        return "\n".join(context_parts)

    def find_entity_references(self, entity: str) -> list[str]:
        """Find all turn IDs where an entity was referenced."""
        return self.entity_references.get(entity, [])
