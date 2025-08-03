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


class SmartRowColumnMatcher(dspy.Signature):
    """Intelligently match user queries to specific table rows and columns."""

    target_description = dspy.InputField(
        desc="Description of what the user is looking for (e.g., 'senior notes', 'equipment rents payable')"
    )
    available_options = dspy.InputField(
        desc="Comma-separated list of available row/column names from the table"
    )
    table_context = dspy.InputField(
        desc="Information about table structure and document context"
    )
    search_type = dspy.InputField(desc="Whether searching for 'row' or 'column'")

    best_match = dspy.OutputField(
        desc="The best matching row/column name from available options"
    )
    match_confidence = dspy.OutputField(desc="Confidence level: high, medium, low")
    reasoning = dspy.OutputField(desc="Brief explanation of why this match was chosen")
    alternative_matches = dspy.OutputField(
        desc="Comma-separated list of other possible matches"
    )


class TableStructureSignature(dspy.Signature):
    """Analyze table structure to understand data organization."""

    row_labels = dspy.InputField(desc="Comma-separated list of row labels/indices")
    column_labels = dspy.InputField(desc="Comma-separated list of column names")
    document_context = dspy.InputField(desc="Context from document pre/post text")
    sample_data = dspy.InputField(
        desc="Sample of the table data to understand patterns"
    )

    table_type = dspy.OutputField(
        desc="Table organization: time_series, cross_sectional, mixed, or unknown"
    )
    primary_dimension = dspy.OutputField(
        desc="Where main data varies: rows, columns, or both"
    )
    time_axis = dspy.OutputField(
        desc="Which axis contains time periods: rows, columns, or none"
    )
    metric_axis = dspy.OutputField(
        desc="Which axis contains financial metrics: rows, columns, or both"
    )
    extraction_strategy = dspy.OutputField(
        desc="Recommended approach for data extraction from this table"
    )


class ContextualValueExtractor(dspy.Signature):
    """Extract specific values using contextual understanding of financial questions."""

    question = dspy.InputField(desc="The financial question being asked")
    conversation_context = dspy.InputField(
        desc="Previous conversation turns for reference resolution"
    )
    table_structure = dspy.InputField(desc="Analysis of table organization and layout")
    available_data = dspy.InputField(
        desc="Summary of available rows, columns, and data points"
    )

    target_metric = dspy.OutputField(desc="The specific financial metric to find")
    target_timeframe = dspy.OutputField(
        desc="The time period or context for the metric"
    )
    row_strategy = dspy.OutputField(desc="How to identify the correct row")
    column_strategy = dspy.OutputField(desc="How to identify the correct column")
    extraction_confidence = dspy.OutputField(
        desc="Confidence in the extraction approach: high, medium, low"
    )


class EnhancedQueryParser:
    """Enhanced DSPy-based parser for financial table queries with intelligent tool selection."""

    def __init__(self) -> None:
        self.query_classifier = dspy.ChainOfThought(FinancialQueryClassifier)
        self.tool_selector = dspy.ChainOfThought(FinancialToolSelector)
        self.expression_classifier = dspy.Predict(MathExpressionClassifier)

        # New DSPy components for enhanced table understanding
        self.row_column_matcher = dspy.ChainOfThought(SmartRowColumnMatcher)
        self.structure_analyzer = dspy.ChainOfThought(TableStructureSignature)
        self.value_extractor = dspy.ChainOfThought(ContextualValueExtractor)

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
        """Classify if a label represents a financial metric using DSPy.

        Args:
            label: Row or column label to classify
            context: Additional context about the table

        Returns:
            True if the label represents a financial metric, False otherwise
        """
        try:
            # Use DSPy classifier
            classifier = dspy.Predict(FinancialMetricClassifier)
            result = classifier(label=label, context=context)
            return str(result.is_financial_metric).lower() == "true"
        except Exception:
            # Fallback to basic check
            from ..core.financial_terms import get_financial_terms_instance

            return get_financial_terms_instance().is_financial_term(label, context)

    def smart_match_row_column(
        self,
        target_description: str,
        available_options: list[str],
        table_context: str,
        search_type: str,
    ) -> tuple[str | None, float, str]:
        """Use DSPy to intelligently match target descriptions to table rows/columns.

        Args:
            target_description: What the user is looking for
            available_options: Available row/column names
            table_context: Context about the table structure
            search_type: "row" or "column"

        Returns:
            Tuple of (best_match, confidence_score, reasoning)
        """
        try:
            if not available_options:
                return None, 0.0, "No options available"

            result = self.row_column_matcher(
                target_description=target_description,
                available_options=", ".join(available_options),
                table_context=table_context,
                search_type=search_type,
            )

            # Convert confidence to numeric score
            confidence_map = {"high": 0.9, "medium": 0.6, "low": 0.3}
            confidence_score = confidence_map.get(result.match_confidence.lower(), 0.3)

            # Validate that the best match is actually in available options
            best_match = result.best_match
            if best_match not in available_options:
                # Try to find a close match
                best_match = self._find_closest_option(best_match, available_options)
                if best_match:
                    confidence_score *= 0.8  # Reduce confidence for fuzzy match

            return best_match, confidence_score, result.reasoning

        except Exception as e:
            logger.debug(f"DSPy row/column matching failed: {e}")
            # Fallback to simple string matching
            return self._fallback_match(target_description, available_options)

    def analyze_table_structure(
        self,
        row_labels: list[str],
        column_labels: list[str],
        document_context: str,
        sample_data: str,
    ) -> dict[str, str]:
        """Use DSPy to analyze table structure and recommend extraction strategies.

        Args:
            row_labels: List of row names/indices
            column_labels: List of column names
            document_context: Context from document
            sample_data: Sample data from the table

        Returns:
            Dictionary with structure analysis results
        """
        try:
            result = self.structure_analyzer(
                row_labels=", ".join(str(label) for label in row_labels),
                column_labels=", ".join(str(label) for label in column_labels),
                document_context=document_context[:500],  # Limit context length
                sample_data=sample_data[:300],  # Limit sample data length
            )

            return {
                "table_type": result.table_type,
                "primary_dimension": result.primary_dimension,
                "time_axis": result.time_axis,
                "metric_axis": result.metric_axis,
                "extraction_strategy": result.extraction_strategy,
            }

        except Exception as e:
            logger.debug(f"DSPy table structure analysis failed: {e}")
            return {
                "table_type": "unknown",
                "primary_dimension": "unknown",
                "time_axis": "unknown",
                "metric_axis": "unknown",
                "extraction_strategy": "Use semantic matching for row/column identification",
            }

    def extract_value_context(
        self,
        question: str,
        conversation_context: str,
        table_structure: dict[str, str],
        available_data: str,
    ) -> dict[str, str]:
        """Use DSPy to understand what value to extract from the question context.

        Args:
            question: The current financial question
            conversation_context: Previous conversation turns
            table_structure: Results from table structure analysis
            available_data: Summary of available data

        Returns:
            Dictionary with extraction guidance
        """
        try:
            result = self.value_extractor(
                question=question,
                conversation_context=conversation_context[-500:],  # Limit context
                table_structure=str(table_structure),
                available_data=available_data[:400],  # Limit data summary
            )

            return {
                "target_metric": result.target_metric,
                "target_timeframe": result.target_timeframe,
                "row_strategy": result.row_strategy,
                "column_strategy": result.column_strategy,
                "confidence": result.extraction_confidence,
            }

        except Exception as e:
            logger.debug(f"DSPy value extraction guidance failed: {e}")
            return {
                "target_metric": question,
                "target_timeframe": "unknown",
                "row_strategy": "semantic matching",
                "column_strategy": "semantic matching",
                "confidence": "low",
            }

    def _find_closest_option(self, target: str, options: list[str]) -> str | None:
        """Find the closest matching option using string similarity."""
        if not target or not options:
            return None

        target_lower = target.lower()
        best_match = None
        best_score = 0.0

        for option in options:
            option_lower = str(option).lower()

            # Exact match
            if target_lower == option_lower:
                return option

            # Substring match
            if target_lower in option_lower or option_lower in target_lower:
                score = min(len(target_lower), len(option_lower)) / max(
                    len(target_lower), len(option_lower)
                )
                if score > best_score:
                    best_score = score
                    best_match = option

        return best_match if best_score > 0.3 else None

    def _fallback_match(
        self, target: str, options: list[str]
    ) -> tuple[str | None, float, str]:
        """Fallback matching when DSPy fails."""
        best_match = self._find_closest_option(target, options)
        if best_match:
            return best_match, 0.5, "Fallback string matching"
        return None, 0.0, "No suitable match found"
