"""
Intelligent table structure analyzer for financial data.
Replaces hardcoded interpretation guides with dynamic analysis.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from ..core.models import FinancialTable

if TYPE_CHECKING:
    from .query_parser import EnhancedQueryParser


@dataclass
class TableStructureAnalysis:
    """Results of table structure analysis."""

    primary_structure: str  # "time_series", "metrics_by_categories", "mixed", "unknown"
    time_columns: list[str]
    metric_rows: list[str]
    data_types: dict[str, str]
    interpretation_guide: dict[str, str]
    confidence: float  # 0.0 to 1.0


class TableStructureAnalyzer:
    """Analyzes financial table structure and provides dynamic interpretation guides."""

    def __init__(self, query_parser: "EnhancedQueryParser | None" = None) -> None:
        self.query_parser = query_parser

    def analyze_structure(
        self, financial_table: FinancialTable, df: pd.DataFrame
    ) -> TableStructureAnalysis:
        """Analyze table structure and generate dynamic interpretation guide.

        Args:
            financial_table: Financial table metadata
            df: Pandas DataFrame with the table data

        Returns:
            TableStructureAnalysis with insights and interpretation guide
        """
        # Create table context for DSPy classification
        table_context = (
            f"Financial table with {df.shape[0]} rows and {df.shape[1]} columns"
        )

        # Analyze columns for time series patterns using DSPy
        time_columns = self._identify_time_columns(df.columns.tolist(), table_context)

        # Analyze rows for financial metrics using DSPy
        metric_rows = self._identify_metric_rows(
            df.index.tolist() if hasattr(df, "index") else [], table_context
        )

        # Determine primary structure
        primary_structure, confidence = self._determine_primary_structure(
            time_columns, metric_rows, df.shape
        )

        # Analyze data types
        data_types = self._analyze_data_types(financial_table)

        # Generate dynamic interpretation guide
        interpretation_guide = self._generate_interpretation_guide(
            primary_structure, time_columns, metric_rows, df.shape
        )

        return TableStructureAnalysis(
            primary_structure=primary_structure,
            time_columns=time_columns,
            metric_rows=metric_rows,
            data_types=data_types,
            interpretation_guide=interpretation_guide,
            confidence=confidence,
        )

    def _identify_time_columns(self, columns: list[str], context: str) -> list[str]:
        """Identify columns that represent time periods using DSPy classification."""
        time_columns = []

        for col in columns:
            if self.query_parser:
                # Use DSPy classification
                if self.query_parser.classify_time_column(str(col), context):
                    time_columns.append(col)
            else:
                # Fallback to pattern-based classification
                if self._fallback_time_column_check(str(col)):
                    time_columns.append(col)

        return time_columns

    def _identify_metric_rows(self, row_labels: list[str], context: str) -> list[str]:
        """Identify rows that represent financial metrics using DSPy classification."""
        metric_rows = []

        for row_label in row_labels:
            if self.query_parser:
                # Use DSPy classification
                if self.query_parser.classify_financial_metric(str(row_label), context):
                    metric_rows.append(row_label)
            else:
                # Fallback to pattern-based classification
                if self._fallback_metric_check(str(row_label)):
                    metric_rows.append(row_label)

        return metric_rows

    def _fallback_time_column_check(self, column_name: str) -> bool:
        """Pattern-based fallback for time column identification."""
        import re

        column_lower = column_name.lower()

        # Year patterns (more flexible than hardcoded list)
        if re.search(r"\b(19|20)[0-9]{2}\b", column_lower):
            return True

        # Quarter patterns
        if re.search(r"\b(q[1-4]|quarter)\b", column_lower):
            return True

        # Common time indicators (minimal set)
        time_words = ["year", "fy", "fiscal", "period", "month"]
        return any(word in column_lower for word in time_words)

    def _fallback_metric_check(self, label: str) -> bool:
        """Pattern-based fallback for financial metric identification."""
        # Use DSPy-powered financial terms recognition
        from ..core.financial_terms import get_financial_terms_instance

        return get_financial_terms_instance().has_financial_content(label)

    def _determine_primary_structure(
        self, time_columns: list[str], metric_rows: list[str], shape: tuple[int, int]
    ) -> tuple[str, float]:
        """Determine the primary structure of the table."""
        rows, cols = shape

        # Time series structure (years/periods as columns, metrics as rows)
        if len(time_columns) >= 2 and len(metric_rows) >= 1:
            confidence = min(
                0.9, 0.5 + (len(time_columns) * 0.1) + (len(metric_rows) * 0.05)
            )
            return "time_series", confidence

        # Metrics by categories (categories as columns, metrics as rows)
        elif rows > cols and len(metric_rows) > len(time_columns):
            confidence = 0.7
            return "metrics_by_categories", confidence

        # Mixed structure
        elif len(time_columns) > 0 or len(metric_rows) > 0:
            confidence = 0.5
            return "mixed", confidence

        # Unknown structure
        else:
            confidence = 0.2
            return "unknown", confidence

    def _analyze_data_types(self, financial_table: FinancialTable) -> dict[str, str]:
        """Analyze the data types of columns."""
        data_types: dict[str, str] = {}

        for col in financial_table.table_schema.columns:
            data_types[col.name] = str(col.column_type)

        return data_types

    def _generate_interpretation_guide(
        self,
        primary_structure: str,
        time_columns: list[str],
        metric_rows: list[str],
        shape: tuple[int, int],
    ) -> dict[str, str]:
        """Generate dynamic interpretation guide based on analysis."""
        rows, cols = shape

        guide = {}

        if primary_structure == "time_series":
            guide = {
                "structure": f"Time series financial data with {len(time_columns)} time periods and {len(metric_rows)} identified metrics",
                "columns": f"Columns represent time periods: {', '.join(time_columns[:3])}{'...' if len(time_columns) > 3 else ''}",
                "rows": f"Rows represent financial metrics: {', '.join(metric_rows[:3])}{'...' if len(metric_rows) > 3 else ''}",
                "reading_guide": "Each cell shows the value of a financial metric for a specific time period",
                "analysis_suggestion": "Use time-based comparisons and trend analysis for insights",
            }
        elif primary_structure == "metrics_by_categories":
            guide = {
                "structure": f"Categorical financial data with {cols} categories and {rows} data points",
                "organization": "Data is organized by categories rather than time periods",
                "rows": f"Contains {len(metric_rows)} identified financial metrics"
                if metric_rows
                else "Contains various data points",
                "reading_guide": "Each cell shows the value of a metric for a specific category",
                "analysis_suggestion": "Use category-based comparisons for insights",
            }
        elif primary_structure == "mixed":
            guide = {
                "structure": f"Mixed structure with both time and categorical elements ({rows}x{cols})",
                "complexity": "This table combines multiple organizational patterns",
                "time_elements": f"Time-related columns: {', '.join(time_columns)}"
                if time_columns
                else "No clear time columns identified",
                "metric_elements": f"Financial metrics: {', '.join(metric_rows)}"
                if metric_rows
                else "No clear financial metrics identified",
                "reading_guide": "Examine both row and column headers to understand data organization",
                "analysis_suggestion": "Consider the context provided in pre_text and post_text for interpretation",
            }
        else:
            guide = {
                "structure": f"Table structure not automatically recognized ({rows} rows, {cols} columns)",
                "organization": "Unable to determine standard financial table patterns",
                "reading_guide": "Refer to the document context (pre_text and post_text) to understand data organization",
                "analysis_suggestion": "Examine column headers and row labels manually for patterns",
            }

        return guide
