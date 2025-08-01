"""
Data loader for ConvFinQA dataset with structured table management.
"""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from ..core.exceptions import (
    DataLoadingError,
    RecordNotFoundError,
    TableValidationError,
)
from ..core.logger import get_logger
from ..core.models import ConvFinQADataset, FinancialTable, Record

logger = get_logger(__name__)


class DataLoader:
    """Handles loading and processing ConvFinQA dataset."""

    def __init__(self, dataset_path: str):
        """Initialize DataLoader with dataset path.

        Args:
            dataset_path: Path to the ConvFinQA JSON dataset
        """
        self.dataset_path = Path(dataset_path)
        self.dataset: ConvFinQADataset | None = None
        self._record_cache: dict[str, Record] = {}

    def load_dataset(self) -> ConvFinQADataset:
        """Load the complete ConvFinQA dataset from JSON."""
        if self.dataset is not None:
            return self.dataset

        logger.info(f"Loading dataset from {self.dataset_path}")

        try:
            # Validate file exists and is readable
            if not self.dataset_path.exists():
                raise DataLoadingError(
                    f"Dataset file not found: {self.dataset_path}",
                    file_path=str(self.dataset_path),
                )

            if not self.dataset_path.is_file():
                raise DataLoadingError(
                    f"Dataset path is not a file: {self.dataset_path}",
                    file_path=str(self.dataset_path),
                )

            # Check file size
            file_size = self.dataset_path.stat().st_size
            if file_size == 0:
                raise DataLoadingError(
                    "Dataset file is empty",
                    file_path=str(self.dataset_path),
                    details={"file_size": file_size},
                )

            logger.info(f"Reading dataset file ({file_size / 1024 / 1024:.2f} MB)")

            # Use sync file reading for now (async version available separately)
            with open(self.dataset_path, encoding="utf-8") as f:
                raw_data = json.load(f)

            # Validate basic structure
            if not isinstance(raw_data, dict):
                raise DataLoadingError(
                    "Dataset file does not contain a JSON object",
                    file_path=str(self.dataset_path),
                    details={"actual_type": type(raw_data).__name__},
                )

            required_keys = {"train", "dev"}
            missing_keys = required_keys - set(raw_data.keys())
            if missing_keys:
                raise DataLoadingError(
                    f"Dataset missing required keys: {missing_keys}",
                    file_path=str(self.dataset_path),
                    details={"available_keys": list(raw_data.keys())},
                )

            # Parse raw data into structured models
            try:
                self.dataset = ConvFinQADataset(**raw_data)
            except ValidationError as e:
                raise DataLoadingError(
                    f"Dataset validation failed: {e}",
                    file_path=str(self.dataset_path),
                    details={"validation_errors": str(e)},
                ) from e

            # Validate and cache records
            self._validate_and_cache_records()

            logger.info(
                f"Successfully loaded {len(self.dataset.train)} training records and {len(self.dataset.dev)} dev records"
            )

            return self.dataset

        except (FileNotFoundError, PermissionError) as e:
            raise DataLoadingError(
                f"Failed to access dataset file: {e}",
                file_path=str(self.dataset_path),
                details={"error_type": type(e).__name__},
            ) from e
        except json.JSONDecodeError as e:
            raise DataLoadingError(
                f"Invalid JSON in dataset file: {e}",
                file_path=str(self.dataset_path),
                details={"line": e.lineno, "column": e.colno},
            ) from e
        except DataLoadingError:
            raise
        except Exception as e:
            raise DataLoadingError(
                f"Unexpected error loading dataset: {e}",
                file_path=str(self.dataset_path),
                details={"error_type": type(e).__name__},
            ) from e

    def get_record(self, record_id: str) -> Record:
        """Get a specific record by ID.

        Args:
            record_id: The ID of the record to retrieve

        Returns:
            The requested record

        Raises:
            RecordNotFoundError: If the record is not found
        """
        if self.dataset is None:
            self.load_dataset()

        record = self._record_cache.get(record_id)
        if record is None:
            available_records = list(self._record_cache.keys())
            raise RecordNotFoundError(record_id, available_records)

        return record

    def get_record_table(self, record_id: str) -> FinancialTable:
        """Get the financial table for a specific record.

        Args:
            record_id: The ID of the record

        Returns:
            The financial table for the record

        Raises:
            RecordNotFoundError: If the record is not found
            TableValidationError: If the table is invalid
        """
        record = self.get_record(record_id)

        try:
            return record.get_financial_table()
        except Exception as e:
            raise TableValidationError(
                f"Failed to extract table from record: {e}",
                table_id=record_id,
                details={"error_type": type(e).__name__},
            ) from e

    def get_record_dataframe(self, record_id: str) -> pd.DataFrame:
        """Get the financial table as a pandas DataFrame.

        Args:
            record_id: The ID of the record

        Returns:
            The financial table as a DataFrame

        Raises:
            RecordNotFoundError: If the record is not found
            TableValidationError: If the table cannot be converted to DataFrame
        """
        table = self.get_record_table(record_id)

        try:
            return table.to_dataframe()
        except Exception as e:
            raise TableValidationError(
                f"Failed to convert table to DataFrame: {e}",
                table_id=record_id,
                details={"error_type": type(e).__name__},
            ) from e

    def list_available_tables(self) -> list[str]:
        """List all available tables in the dataset."""
        if self.dataset is None:
            self.load_dataset()

        assert self.dataset is not None  # For type checker
        return [record.id for record in self.dataset.get_all_records()]

    def get_dataset_statistics(self) -> dict[str, Any]:
        """Get basic statistics about the dataset."""
        if self.dataset is None:
            self.load_dataset()

        assert self.dataset is not None  # For type checker
        train_records = len(self.dataset.train)
        dev_records = len(self.dataset.dev)

        # Analyze table structures
        table_shapes = []
        column_types: dict[str, int] = {}

        for record in self.dataset.get_all_records()[
            :100
        ]:  # Sample first 100 for stats
            table = record.get_financial_table()
            df = table.to_dataframe()
            table_shapes.append(df.shape)

            for col in table.table_schema.columns:
                column_types[col.column_type] = column_types.get(col.column_type, 0) + 1

        return {
            "train_records": train_records,
            "dev_records": dev_records,
            "total_records": train_records + dev_records,
            "sample_table_shapes": table_shapes[:10],
            "column_type_distribution": column_types,
        }

    def _validate_and_cache_records(self) -> None:
        """Validate records and build cache with quality checks."""
        if self.dataset is None:
            return

        all_records = self.dataset.get_all_records()
        quality_issues = []

        for record in all_records:
            try:
                # Basic record validation
                if not record.id:
                    quality_issues.append("Record has empty ID")
                    continue

                # Table validation
                if not record.doc.table:
                    quality_issues.append(f"Record {record.id} has no table data")
                    continue

                # Check for duplicate columns
                if record.doc.table:
                    columns = list(record.doc.table.keys())
                    if len(columns) != len(set(columns)):
                        duplicate_cols = [
                            col for col in columns if columns.count(col) > 1
                        ]
                        quality_issues.append(
                            f"Record {record.id} has duplicate columns: {duplicate_cols}"
                        )

                # Cache the record
                self._record_cache[record.id] = record

            except Exception as e:
                quality_issues.append(f"Record {record.id} validation failed: {e}")

        # Log quality issues as warnings
        if quality_issues:
            logger.warning(f"Found {len(quality_issues)} data quality issues:")
            for issue in quality_issues[:10]:  # Log first 10 issues
                logger.warning(f"  - {issue}")
            if len(quality_issues) > 10:
                logger.warning(f"  ... and {len(quality_issues) - 10} more issues")

        logger.info(f"Cached {len(self._record_cache)} valid records")


def create_data_loader(dataset_path: str = "data/convfinqa_dataset.json") -> DataLoader:
    """Factory function to create a DataLoader instance."""
    return DataLoader(dataset_path)
