"""
DSPy-based query parser for financial table queries.
Replaces hardcoded string matching with structured interpretation.
"""

from typing import Any

import dspy
from pydantic import BaseModel

from ..core.logger import get_logger

logger = get_logger(__name__)


class QueryIntent(BaseModel):
    """Structured representation of a parsed query."""

    operation: str  # 'sum', 'show', 'filter', 'info'
    target_columns: list[str]
    parameters: dict[str, Any]
    query_type: str  # 'aggregation', 'selection', 'info'


class ExpressionIntent(BaseModel):
    """Structured representation of a parsed mathematical expression."""

    expression_type: str  # 'table_operation', 'mathematical', 'complex'
    operations: list[str]  # ['sum', 'multiply', etc.]
    target_columns: list[str]
    is_table_dependent: bool
    parameters: dict[str, Any]


class FinancialQueryClassifier(dspy.Signature):
    """Advanced classification of financial table queries into structured intents with enhanced context understanding."""

    query = dspy.InputField(desc="Natural language query about financial data")
    table_context = dspy.InputField(
        desc="Available table columns and structure context"
    )

    operation = dspy.OutputField(
        desc="Primary operation needed: 'lookup', 'calculate', 'compare', 'aggregate', 'analyze_change'"
    )
    target_columns = dspy.OutputField(
        desc="Specific column names or descriptions that should be accessed"
    )
    target_rows = dspy.OutputField(
        desc="Specific row identifiers or descriptions needed for the query"
    )
    query_complexity = dspy.OutputField(
        desc="Complexity level: 'simple_lookup', 'single_calculation', 'multi_step_analysis'"
    )
    expected_result_type = dspy.OutputField(
        desc="Expected result: 'single_value', 'comparison', 'percentage', 'change_amount'"
    )


class FinancialToolSelector(dspy.Signature):
    """Intelligent tool selection for financial analysis tasks based on query analysis."""

    query = dspy.InputField(desc="The financial question being asked")
    query_analysis = dspy.InputField(
        desc="Analysis of the query including operation type and complexity"
    )
    available_tools = dspy.InputField(
        desc="List of available tools and their capabilities"
    )

    recommended_tools = dspy.OutputField(
        desc="Ordered list of tools to use for this query"
    )
    analysis_strategy = dspy.OutputField(
        desc="Step-by-step strategy for answering the query"
    )
    confidence = dspy.OutputField(desc="Confidence level: 'high', 'medium', 'low'")


class MathExpressionClassifier(dspy.Signature):
    """Classify mathematical expressions and identify table operations."""

    expression = dspy.InputField(desc="Mathematical expression or computation request")
    expression_type = dspy.OutputField(
        desc="Expression category: table_operation, mathematical, or complex"
    )
    operations = dspy.OutputField(
        desc="Comma-separated list of operations: sum, multiply, divide, add, subtract"
    )
    target_columns = dspy.OutputField(
        desc="Comma-separated list of column names referenced"
    )
    is_table_dependent = dspy.OutputField(
        desc="true if expression needs table data, false for pure math"
    )


class TimeColumnClassifier(dspy.Signature):
    """Classify if a column name represents a time period in financial data."""

    column_name = dspy.InputField(desc="Column name from a financial table")
    context = dspy.InputField(desc="Additional context about the table structure")
    is_time_column = dspy.OutputField(
        desc="true if column represents time period, false otherwise"
    )
    time_type = dspy.OutputField(
        desc="Type of time period: year, quarter, month, fiscal_year, or none"
    )
    confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")


class FinancialMetricClassifier(dspy.Signature):
    """Classify if a row/column label represents a financial metric."""

    label = dspy.InputField(desc="Row or column label from a financial table")
    context = dspy.InputField(desc="Additional context about the table structure")
    is_financial_metric = dspy.OutputField(
        desc="true if label represents financial metric, false otherwise"
    )
    metric_category = dspy.OutputField(
        desc="Category: revenue, expense, asset, liability, equity, performance, or none"
    )
    confidence = dspy.OutputField(desc="Confidence level: high, medium, or low")


class EnhancedQueryParser:
    """Enhanced DSPy-based parser for financial table queries with intelligent tool selection."""

    def __init__(self) -> None:
        self.query_classifier = dspy.ChainOfThought(FinancialQueryClassifier)
        self.tool_selector = dspy.ChainOfThought(FinancialToolSelector)
        self.expression_classifier = dspy.Predict(MathExpressionClassifier)

    def parse_with_strategy(
        self,
        query: str,
        available_columns: list[str] | None = None,
        table_context: str = "",
    ) -> tuple[QueryIntent, str]:
        """Parse a query and return both intent and recommended analysis strategy.

        Args:
            query: Natural language query
            available_columns: List of available column names
            table_context: Description of table structure and data

        Returns:
            Tuple of (QueryIntent, analysis_strategy)
        """
        try:
            # Enhanced query classification with table context
            context_info = f"Available columns: {', '.join(available_columns or [])}. {table_context}"

            classification = self.query_classifier(
                query=query, table_context=context_info
            )

            # Get tool recommendations
            available_tools_desc = """
            Available tools:
            - show_table(): View complete table structure and data
            - get_table_value(): Extract specific values from identified row/column
            - calculate_change(): Compute changes between two values
            - compute(): Perform mathematical calculations
            - validate_data_selection(): Verify data selections are correct
            - final_answer(): Return the final numeric result
            """

            tool_recommendation = self.tool_selector(
                query=query,
                query_analysis=f"Operation: {classification.operation}, Complexity: {classification.query_complexity}, Expected: {classification.expected_result_type}",
                available_tools=available_tools_desc,
            )

            # Parse target columns with intelligent filtering
            target_columns = []
            if classification.target_columns:
                mentioned_columns = [
                    col.strip() for col in classification.target_columns.split(",")
                ]
                if available_columns:
                    # Use DSPy-enhanced column matching
                    target_columns = self._match_columns_intelligently(
                        mentioned_columns, available_columns
                    )
                else:
                    target_columns = mentioned_columns

            # Build enhanced parameters
            parameters = {
                "operation_type": classification.operation,
                "complexity": classification.query_complexity,
                "target_rows": classification.target_rows,
                "expected_result": classification.expected_result_type,
                "recommended_tools": tool_recommendation.recommended_tools,
                "confidence": tool_recommendation.confidence,
            }

            # Determine query type from operation
            query_type_mapping = {
                "lookup": "selection",
                "calculate": "calculation",
                "compare": "comparison",
                "aggregate": "aggregation",
                "analyze_change": "change_analysis",
            }

            intent = QueryIntent(
                operation=classification.operation,
                target_columns=target_columns,
                parameters=parameters,
                query_type=query_type_mapping.get(classification.operation, "analysis"),
            )

            return intent, tool_recommendation.analysis_strategy

        except Exception as e:
            logger.debug(f"Enhanced DSPy parsing failed: {e}")
            # Fallback to basic parsing
            basic_intent = self._create_basic_intent(query, available_columns)
            return (
                basic_intent,
                "Basic analysis: Use show_table() to explore, then extract relevant values.",
            )

    def parse_expression(
        self, expression: str, available_columns: list[str] | None = None
    ) -> ExpressionIntent:
        """Parse a mathematical expression into structured intent using DSPy.

        Args:
            expression: Mathematical expression to evaluate
            available_columns: List of available column names for validation

        Returns:
            ExpressionIntent with structured interpretation
        """
        try:
            # Use DSPy to classify the expression
            result = self.expression_classifier(expression=expression)

            # Parse operations
            operations = []
            if result.operations:
                operations = [op.strip() for op in result.operations.split(",")]

            # Parse target columns
            target_columns = []
            if result.target_columns:
                target_columns = [
                    col.strip() for col in result.target_columns.split(",")
                ]

            # Filter to only available columns if provided
            if available_columns:
                target_columns = [
                    col for col in target_columns if col in available_columns
                ]

            # Parse table dependency
            is_table_dependent = result.is_table_dependent.lower() == "true"

            # Build parameters based on expression type
            parameters = {}
            if result.expression_type == "table_operation" and operations:
                parameters = {
                    "primary_operation": operations[0] if operations else "sum"
                }

            return ExpressionIntent(
                expression_type=result.expression_type,
                operations=operations,
                target_columns=target_columns,
                is_table_dependent=is_table_dependent,
                parameters=parameters,
            )

        except Exception as e:
            logger.debug(f"DSPy expression parsing failed: {e}")
            # Fallback to basic parsing if DSPy fails
            return self._fallback_parse_expression(expression)

    def _fallback_parse_expression(self, expression: str) -> ExpressionIntent:
        """Simple fallback parser for expressions if DSPy fails."""
        expression_lower = expression.lower()

        # Check for table operations
        if any(word in expression_lower for word in ["sum", "total"]):
            return ExpressionIntent(
                expression_type="table_operation",
                operations=["sum"],
                target_columns=[],
                is_table_dependent=True,
                parameters={"primary_operation": "sum"},
            )
        elif any(word in expression_lower for word in ["+", "add"]):
            return ExpressionIntent(
                expression_type="mathematical",
                operations=["add"],
                target_columns=[],
                is_table_dependent=False,
                parameters={},
            )
        elif any(word in expression_lower for word in ["*", "multiply"]):
            return ExpressionIntent(
                expression_type="mathematical",
                operations=["multiply"],
                target_columns=[],
                is_table_dependent=False,
                parameters={},
            )
        else:
            return ExpressionIntent(
                expression_type="mathematical",
                operations=[],
                target_columns=[],
                is_table_dependent=False,
                parameters={},
            )

    def _match_columns_intelligently(
        self, mentioned_columns: list[str], available_columns: list[str]
    ) -> list[str]:
        """Intelligently match mentioned columns to available columns using DSPy-enhanced matching."""
        matched_columns = []

        for mentioned in mentioned_columns:
            mentioned_lower = mentioned.lower().strip()

            # Direct match first
            for available in available_columns:
                if mentioned_lower == available.lower():
                    matched_columns.append(available)
                    break
            else:
                # Fuzzy matching for partial matches
                for available in available_columns:
                    if (
                        mentioned_lower in available.lower()
                        or available.lower() in mentioned_lower
                    ):
                        matched_columns.append(available)
                        break

        return matched_columns

    def _create_basic_intent(
        self, query: str, available_columns: list[str] | None
    ) -> QueryIntent:
        """Create a basic intent when enhanced parsing fails."""
        query_lower = query.lower()

        # Basic operation detection
        if any(
            word in query_lower
            for word in ["change", "increase", "decrease", "difference"]
        ):
            operation = "analyze_change"
            query_type = "change_analysis"
        elif any(word in query_lower for word in ["calculate", "compute", "total"]):
            operation = "calculate"
            query_type = "calculation"
        elif any(word in query_lower for word in ["compare", "versus", "vs"]):
            operation = "compare"
            query_type = "comparison"
        else:
            operation = "lookup"
            query_type = "selection"

        return QueryIntent(
            operation=operation,
            target_columns=available_columns[:3] if available_columns else [],
            parameters={"complexity": "unknown", "confidence": "low"},
            query_type=query_type,
        )

    def classify_time_column(self, column_name: str, context: str = "") -> bool:
        """Classify if a column name represents a time period (compatibility method).

        Args:
            column_name: Name of the column to classify
            context: Additional context about the table

        Returns:
            True if the column represents a time period, False otherwise
        """
        try:
            # Simple pattern-based classification for time columns
            column_lower = str(column_name).lower()

            # Year patterns (2000-2030)
            import re

            if re.search(r"\b(19|20)[0-9]{2}\b", column_lower):
                return True

            # Quarter patterns
            if re.search(r"\b(q[1-4]|quarter)\b", column_lower):
                return True

            # Common time indicators
            time_words = ["year", "fy", "fiscal", "period", "month", "date"]
            return any(word in column_lower for word in time_words)

        except Exception:
            return False

    def classify_financial_metric(self, label: str, context: str = "") -> bool:
        """Classify if a label represents a financial metric (compatibility method).

        Args:
            label: Row or column label to classify
            context: Additional context about the table

        Returns:
            True if the label represents a financial metric, False otherwise
        """
        try:
            label_lower = str(label).lower()

            # Common financial terms
            financial_words = [
                "revenue",
                "sales",
                "income",
                "profit",
                "loss",
                "expense",
                "cost",
                "asset",
                "liability",
                "equity",
                "cash",
                "debt",
                "margin",
                "ebitda",
                "earnings",
                "dividend",
                "interest",
                "tax",
                "capital",
                "investment",
            ]

            return any(word in label_lower for word in financial_words)

        except Exception:
            return False
