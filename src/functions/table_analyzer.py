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
        if self.query_parser is None:
            # Fallback to basic analysis if no query parser is available
            return self._fallback_analysis(financial_table, df)

        # Use the powerful DSPy-based table structure analyzer
        try:
            row_labels = df.index.tolist() if hasattr(df.index, "tolist") else []
            column_labels = df.columns.tolist()
            sample_data = df.head(3).to_string()
            doc_context = financial_table.table_schema.description or ""

            # Call the advanced DSPy analysis
            analysis_results = self.query_parser.analyze_table_structure(
                row_labels=row_labels,
                column_labels=column_labels,
                document_context=doc_context,
                sample_data=sample_data,
            )

            # Map results to the TableStructureAnalysis object
            primary_structure = analysis_results.get("table_type", "unknown")
            time_columns = self._identify_time_columns(
                df.columns.tolist(), analysis_results
            )
            metric_rows = self._identify_metric_rows(
                df.index.tolist(), analysis_results
            )
            data_types = self._analyze_data_types(financial_table)

            interpretation_guide = self._generate_interpretation_guide(
                primary_structure, time_columns, metric_rows, df.shape
            )

            confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            confidence = confidence_map.get(
                analysis_results.get("confidence", "low"), 0.3
            )

            return TableStructureAnalysis(
                primary_structure=primary_structure,
                time_columns=time_columns,
                metric_rows=metric_rows,
                data_types=data_types,
                interpretation_guide=interpretation_guide,
                confidence=confidence,
            )

        except Exception:
            # If DSPy analysis fails, revert to fallback
            return self._fallback_analysis(financial_table, df)

    def _identify_time_columns(
        self, columns: list[str], analysis_results: dict[str, str]
    ) -> list[str]:
        """Identify time columns based on DSPy analysis results."""
        time_axis = analysis_results.get("time_axis")
        if time_axis == "columns":
            # Heuristic: assume all columns are time columns if the axis is identified as such
            return columns
        return []

    def _identify_metric_rows(
        self, row_labels: list[str], analysis_results: dict[str, str]
    ) -> list[str]:
        """Identify metric rows based on DSPy analysis results."""
        metric_axis = analysis_results.get("metric_axis")
        if metric_axis == "rows":
            # Heuristic: assume all rows are metric rows if the axis is identified as such
            return row_labels
        return []

    def _fallback_analysis(
        self, financial_table: FinancialTable, df: pd.DataFrame
    ) -> TableStructureAnalysis:
        """A fallback analysis method that doesn't rely on the query parser."""
        time_columns = [
            col for col in df.columns if self._fallback_time_column_check(str(col))
        ]
        metric_rows = [row for row in df.index if self._fallback_metric_check(str(row))]

        primary_structure, confidence = self._determine_primary_structure(
            time_columns, metric_rows, df.shape
        )
        data_types = self._analyze_data_types(financial_table)
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

    def _fallback_time_column_check(self, column_name: str) -> bool:
        """Pattern-based fallback for time column identification."""
        import re

        column_lower = column_name.lower()

        if re.search(r"\b(19|20)[0-9]{2}\b", column_lower):
            return True
        if re.search(r"\b(q[1-4]|quarter)\b", column_lower):
            return True
        time_words = ["year", "fy", "fiscal", "period", "month"]
        return any(word in column_lower for word in time_words)

    def _fallback_metric_check(self, label: str) -> bool:
        """Pattern-based fallback for financial metric identification."""
        from ..core.financial_terms import get_financial_terms_instance

        return get_financial_terms_instance().has_financial_content(label)

    def _determine_primary_structure(
        self, time_columns: list[str], metric_rows: list[str], shape: tuple[int, int]
    ) -> tuple[str, float]:
        """Determine the primary structure of the table."""
        rows, cols = shape

        if len(time_columns) >= 2 and len(metric_rows) >= 1:
            confidence = min(
                0.9, 0.5 + (len(time_columns) * 0.1) + (len(metric_rows) * 0.05)
            )
            return "time_series", confidence
        elif rows > cols and len(metric_rows) > len(time_columns):
            confidence = 0.7
            return "metrics_by_categories", confidence
        elif len(time_columns) > 0 or len(metric_rows) > 0:
            confidence = 0.5
            return "mixed", confidence
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
