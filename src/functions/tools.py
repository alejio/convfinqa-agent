"""
LLM tool functions using smolagents for ConvFinQA.
"""

import json
import threading
from collections.abc import Callable
from typing import Any

import pandas as pd
from smolagents import tool

from ..core.logger import get_logger
from ..core.models import Record
from ..data.loader import DataLoader
from .query_parser import EnhancedQueryParser
from .table_analyzer import TableStructureAnalyzer

logger = get_logger(__name__)

# Thread-local storage for current record context (thread-safe for parallel evaluation)
_thread_local = threading.local()

# Initialize enhanced DSPy query parser and table analyzer
_query_parser = EnhancedQueryParser()
_table_analyzer = TableStructureAnalyzer(_query_parser)


def set_context(record: Record | None, data_loader: DataLoader | None) -> None:
    """Set the current record and data loader context for tool functions (thread-safe)."""
    _thread_local.current_record = record
    _thread_local.current_data_loader = data_loader


def _get_current_record() -> Record | None:
    """Get the current record from thread-local storage."""
    return getattr(_thread_local, "current_record", None)


def _get_current_data_loader() -> DataLoader | None:
    """Get the current data loader from thread-local storage."""
    return getattr(_thread_local, "current_data_loader", None)


def _clean_math_expression(expression: str) -> str | None:
    """Clean mathematical expression for safe evaluation."""
    try:
        # Remove common text and normalize
        cleaned = expression.strip()

        # Handle parentheses and basic operators
        cleaned = cleaned.replace("abs(", "abs(")  # Keep abs function

        # Ensure it looks like a valid mathematical expression
        import re

        if re.match(r"^[\d\s\+\-\*\/\(\)\.abs]+$", cleaned.replace("abs", "")):
            return cleaned

        return None
    except Exception:
        return None


@tool
def list_tables(compact: bool = False) -> str:
    """List all available tables in the current record.

    Args:
        compact: Return minimal response for token efficiency (default: False)
    """
    current_record = _get_current_record()
    if current_record is None:
        return json.dumps({"error": "No record context set"})

    try:
        table_names = current_record.get_table_names()
        financial_table = current_record.get_financial_table()

        # Define result type before the if/else block
        result: dict[str, Any]
        if compact:
            # Minimal response for token efficiency
            result = {
                "name": table_names[0] if table_names else "financial_data",
                "cols": [str(col.name) for col in financial_table.table_schema.columns],
            }
        else:
            # Full response
            result = {
                "tables": [
                    {
                        "name": table_names[0] if table_names else "financial_data",
                        "rows": financial_table.table_schema.row_count,
                        "columns": len(financial_table.table_schema.columns),
                        "column_names": [
                            str(col.name)
                            for col in financial_table.table_schema.columns
                        ],
                    }
                ]
            }

        logger.info(f"Listed tables for record {current_record.id}")
        return json.dumps(result)

    except Exception as e:
        error_msg = f"Error listing tables: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


@tool
def show_table(table_name: str = "financial_data", compact: bool = False) -> str:
    """
    Show the structure and data of a specific table including document context.

    Args:
        table_name: Name of the table to display (optional, defaults to 'financial_data')
        compact: Return minimal response for token efficiency (default: False)

    Returns:
        JSON with table schema, sample data, and document context that explains the data
    """
    current_record = _get_current_record()
    current_data_loader = _get_current_data_loader()

    if current_record is None:
        return json.dumps({"error": "No record context set"})

    if current_data_loader is None:
        return json.dumps({"error": "No data loader context set"})

    try:
        # Get the financial table
        financial_table = current_record.get_financial_table()
        df = financial_table.to_dataframe()

        if compact:
            # Minimal response for token efficiency
            result = {
                "cols": list(df.columns),
                "rows": list(df.index),
                "data": df.to_dict(orient="records"),  # type: ignore
            }
        else:
            # Get document context
            doc_context = {
                "pre_text": current_record.doc.pre_text[:500] + "..."
                if len(current_record.doc.pre_text) > 500
                else current_record.doc.pre_text,
                "post_text": current_record.doc.post_text[:500] + "..."
                if len(current_record.doc.post_text) > 500
                else current_record.doc.post_text,
            }

            # Return comprehensive table information with context
            result = {
                "table_schema": {
                    "name": financial_table.table_schema.name,
                    "rows": len(df),
                    "columns": [
                        {
                            "name": col.name,
                            "type": col.column_type,
                            "nullable": col.nullable,
                        }
                        for col in financial_table.table_schema.columns
                    ],
                },
                "document_context": doc_context,
                "sample_data": df.to_dict(orient="records"),  # type: ignore
                "total_rows": len(df),
                "analysis": _table_analyzer.analyze_structure(
                    financial_table, df
                ).__dict__,
            }

        logger.info(f"Showed table {table_name} for record {current_record.id}")
        return json.dumps(result)

    except Exception as e:
        return json.dumps(
            {
                "error": f"Failed to show table: {str(e)}",
                "available_tables": ["financial_data"],
            }
        )


@tool
def query_table(
    query: str, table_name: str = "financial_data", compact: bool = False
) -> str:
    """Query financial table data using enhanced DSPy-based natural language interpretation.

    This tool uses advanced DSPy analysis to understand financial queries and recommend
    the best analysis strategy for accurate results.

    Args:
        query: Natural language query about the financial data
        table_name: Name of the table to query (optional, defaults to 'financial_data')
        compact: Return minimal response for token efficiency (default: False)

    Returns:
        JSON with query results, analysis strategy, and relevant financial data
    """
    current_record = _get_current_record()
    if current_record is None:
        return json.dumps({"error": "No record context set"})

    try:
        financial_table = current_record.get_financial_table()
        df = financial_table.to_dataframe()

        # Get document context
        doc_context = f"Pre-text: {current_record.doc.pre_text[:500]}... Post-text: {current_record.doc.post_text[:500]}..."

        # Create enhanced table structure summary
        table_structure = {
            "columns": list(df.columns),
            "row_count": len(df),
            "row_labels": df.index.tolist()
            if hasattr(df.index, "tolist")
            else list(range(len(df))),
            "sample_data": df.head(3).to_dict(orient="records") if len(df) > 0 else [],  # type: ignore
        }

        # Use enhanced DSPy parser with strategy analysis
        available_columns = [col.name for col in financial_table.table_schema.columns]
        table_context = f"Financial table with {len(df)} rows and {len(df.columns)} columns. Document context: {doc_context[:200]}"

        try:
            intent, analysis_strategy = _query_parser.parse_with_strategy(
                query, available_columns, table_context
            )

            logger.info(
                f"Enhanced DSPy analysis - Operation: {intent.operation}, Strategy: {analysis_strategy[:100]}..."
            )

            # Process based on enhanced DSPy analysis
            result: dict[str, Any]
            if compact:
                result = {
                    "op": intent.operation,
                    "data": None,  # Will be filled below
                }
            else:
                result = {
                    "query": query,
                    "operation": intent.operation,
                    "query_type": intent.query_type,
                    "analysis_strategy": analysis_strategy,
                    "recommended_tools": str(
                        intent.parameters.get("recommended_tools", "")
                    ),
                    "confidence": str(intent.parameters.get("confidence", "medium")),
                    "table_structure": table_structure,
                    "document_context": doc_context,
                }

            # Handle different operation types with DSPy guidance
            if intent.operation == "lookup" and intent.target_columns:
                # Simple lookup operation
                selected_columns = [
                    col for col in intent.target_columns if col in df.columns
                ]
                if selected_columns:
                    result["data"] = df[selected_columns].to_dict(orient="records")  # type: ignore
                    if not compact:
                        result["selected_columns"] = selected_columns  # type: ignore
                else:
                    result["data"] = df.to_dict(orient="records")  # type: ignore
                    if not compact:
                        result["suggestion"] = (
                            f"Columns {intent.target_columns} not found. Showing all data."
                        )

            elif intent.operation == "aggregate" and intent.target_columns:
                # Aggregation operation - find numeric columns
                numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                target_numeric = [
                    col for col in intent.target_columns if col in numeric_cols
                ]

                if target_numeric:
                    aggregated_data = {}
                    for col in target_numeric:
                        try:
                            total = df[col].sum()
                            aggregated_data[f"{col}_total"] = total
                        except Exception:
                            continue

                    result["data"] = aggregated_data  # type: ignore
                    if not compact:
                        result["operation_details"] = (
                            f"Aggregated {len(target_numeric)} numeric columns"
                        )
                else:
                    result["data"] = {
                        "error": "No numeric columns found for aggregation"
                    }  # type: ignore
                    if not compact:
                        result["suggestion"] = (
                            f"Available numeric columns: {numeric_cols}"
                        )

            elif intent.operation in ["calculate", "analyze_change", "compare"]:
                # Complex operations - provide guidance for next steps
                if compact:
                    result["data"] = df.head(5).to_dict(orient="records")  # type: ignore
                else:
                    result["data"] = {
                        "status": "analysis_required",
                        "message": "Complex operation detected. Use the analysis strategy below.",
                        "next_steps": analysis_strategy,
                        "available_data": df.head(5).to_dict(orient="records"),  # type: ignore
                    }

            else:
                # Default: show relevant data based on columns
                if intent.target_columns:
                    relevant_columns = [
                        col for col in intent.target_columns if col in df.columns
                    ]
                    if relevant_columns:
                        result["data"] = df[relevant_columns].to_dict(orient="records")  # type: ignore
                    else:
                        result["data"] = df.to_dict(orient="records")  # type: ignore
                else:
                    result["data"] = df.to_dict(orient="records")  # type: ignore

            return json.dumps(result)

        except Exception as e:
            logger.debug(f"Enhanced DSPy parsing failed: {e}")
            # Fallback to basic processing
            if compact:
                return json.dumps(
                    {
                        "data": df.to_dict(orient="records"),  # type: ignore
                        "error": f"Enhanced analysis failed: {str(e)}",
                    }
                )
            else:
                return json.dumps(
                    {
                        "query": query,
                        "result_type": "basic_fallback",
                        "data": df.to_dict(orient="records"),  # type: ignore
                        "table_structure": table_structure,
                        "document_context": doc_context,
                        "error": f"Enhanced analysis failed: {str(e)}",
                    }
                )

    except Exception as e:
        error_msg = f"Error querying table: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})


@tool
def compute(expression: str) -> str:
    """Compute mathematical expressions using available functions and table data.

    Args:
        expression: The mathematical expression to evaluate.
    """
    current_record = _get_current_record()
    if current_record is None:
        return json.dumps({"error": "No record context set"})

    try:
        # Get table context for DSPy parsing
        financial_table = current_record.get_financial_table()
        df = financial_table.to_dataframe()
        available_columns = [col.name for col in financial_table.table_schema.columns]

        # Use DSPy to parse expression intent
        try:
            intent = _query_parser.parse_expression(expression, available_columns)
        except Exception as e:
            logger.debug(f"DSPy expression parsing failed: {e}")
            intent = None

        # Handle table-dependent expressions
        if (
            intent
            and intent.is_table_dependent
            and intent.expression_type == "table_operation"
        ):
            # Handle table operations identified by DSPy
            numeric_cols = [
                col.name
                for col in financial_table.table_schema.columns
                if col.column_type == "numeric"
            ]

            # Use DSPy-identified columns or fall back to numeric column detection
            target_columns = (
                intent.target_columns if intent.target_columns else numeric_cols
            )

            # Execute the primary operation
            primary_op = intent.parameters.get("primary_operation", "sum")

            for col_name in target_columns:
                if col_name in numeric_cols:
                    try:
                        if primary_op == "sum":
                            result_value = float(df[col_name].sum())
                            logger.info(
                                f"Computed DSPy-parsed table expression: {expression} = {result_value}"
                            )
                            return str(result_value)
                    except Exception as e:
                        logger.debug(
                            f"Table operation failed for column {col_name}: {e}"
                        )
                        continue

        # Handle pure mathematical expressions with improved safety
        try:
            # Clean the expression for safe evaluation
            cleaned_expr = _clean_math_expression(expression)
            if cleaned_expr:
                # Safe operators
                import ast
                import operator

                binary_ops: dict[type, Callable[[float, float], float]] = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                }
                unary_ops: dict[type, Callable[[float], float]] = {
                    ast.USub: operator.neg,
                    ast.UAdd: operator.pos,
                }

                # Safe functions
                import math

                safe_functions: dict[str, Callable[..., Any]] = {
                    "abs": abs,
                    "sqrt": math.sqrt,
                    "pow": pow,
                    "round": round,
                }

                def safe_eval(node: Any) -> float:
                    if isinstance(node, ast.Expression):
                        return safe_eval(node.body)
                    elif isinstance(node, ast.Constant):  # numbers
                        if isinstance(node.value, int | float):
                            return float(node.value)
                        else:
                            raise ValueError(
                                f"Unsupported constant type: {type(node.value)}"
                            )
                    elif isinstance(node, ast.Num):  # for older Python versions
                        if isinstance(node.n, int | float):
                            return float(node.n)
                        else:
                            raise ValueError(f"Unsupported num type: {type(node.n)}")
                    elif isinstance(node, ast.BinOp):
                        left = safe_eval(node.left)
                        right = safe_eval(node.right)
                        if type(node.op) in binary_ops:
                            result = binary_ops[type(node.op)](left, right)
                            if isinstance(result, int | float):
                                return float(result)
                            else:
                                raise ValueError(
                                    f"Unsupported result type: {type(result)}"
                                )
                        else:
                            raise ValueError(
                                f"Unsupported binary operation: {type(node.op)}"
                            )
                    elif isinstance(node, ast.UnaryOp):
                        operand = safe_eval(node.operand)
                        if type(node.op) in unary_ops:
                            result = unary_ops[type(node.op)](operand)
                            if isinstance(result, int | float):
                                return float(result)
                            else:
                                raise ValueError(
                                    f"Unsupported result type: {type(result)}"
                                )
                        else:
                            raise ValueError(
                                f"Unsupported unary operation: {type(node.op)}"
                            )
                    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                        if func_name in safe_functions:
                            args = [safe_eval(arg) for arg in node.args]
                            result = safe_functions[func_name](*args)
                            if isinstance(result, int | float):
                                return float(result)
                            else:
                                raise ValueError(
                                    f"Unsupported result type: {type(result)}"
                                )
                        else:
                            raise ValueError(f"Unsupported function: {func_name}")
                    else:
                        raise ValueError(f"Unsupported operation: {ast.dump(node)}")

                # Parse and evaluate with better error handling
                try:
                    tree = ast.parse(cleaned_expr, mode="eval")
                    result_value = safe_eval(tree)

                    logger.info(
                        f"Computed mathematical expression: {expression} = {result_value}"
                    )
                    return str(result_value)
                except Exception as e:
                    logger.debug(f"AST evaluation failed: {e}")
                    # Fall through to basic arithmetic

        except Exception as e:
            logger.debug(f"Safe evaluation setup failed: {e}")

        # Basic arithmetic fallback with improved parsing
        try:
            # Extract numbers from expression for basic operations
            import re

            # Find all numbers (including negative and decimal)
            numbers = re.findall(r"-?\d+\.?\d*", expression)
            if len(numbers) >= 2:
                # Convert to floats
                num1: float = float(numbers[0])
                num2: float = float(numbers[1])

                # Try common operations based on keywords/symbols
                expr_lower = expression.lower()

                result_val: float
                # Addition
                if any(op in expr_lower for op in ["+", "add", "plus", "sum"]):
                    result_val = num1 + num2
                # Subtraction
                elif any(
                    op in expr_lower for op in ["-", "subtract", "minus", "difference"]
                ):
                    result_val = num1 - num2
                # Multiplication
                elif any(op in expr_lower for op in ["*", "multiply", "times", "×"]):
                    result_val = num1 * num2
                # Division
                elif any(op in expr_lower for op in ["/", "divide", "divided by", "÷"]):
                    if num2 != 0:
                        result_val = num1 / num2
                    else:
                        return "ERROR: Division by zero"
                # Default to first number if no clear operation
                else:
                    result_val = num1

                logger.info(f"Computed basic arithmetic: {expression} = {result_val}")
                return str(result_val)

            elif len(numbers) == 1:
                # Single number, return as is
                return str(float(numbers[0]))

        except Exception as e:
            logger.debug(f"Basic arithmetic failed: {e}")

        # If all else fails, return a clear error
        return f"ERROR: Unable to compute expression '{expression}'. Please use simpler mathematical expressions or check syntax."

    except Exception as e:
        error_msg = f"ERROR: Error computing expression '{expression}': {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
def calculate_change(
    old_value: float | str, new_value: float | str, change_type: str = "auto"
) -> str:
    """Calculate change between two values with proper financial conventions.

    IMPORTANT: For senior notes or specific financial instruments, make sure you're using
    the correct values from the right table rows, not general/aggregate values.

    Args:
        old_value: The earlier/baseline value (can be float or string)
        new_value: The later/current value (can be float or string)
        change_type: Type of change calculation ("auto", "standard", "cost", "percentage", "simple")

    Returns:
        JSON string with different change calculations and data validation warnings,
        or simple numeric result if change_type="simple"
    """
    try:
        # Convert inputs to floats (handle string inputs from get_table_value)
        try:
            if isinstance(old_value, str):
                old_value = float(old_value.replace("$", "").replace(",", "").strip())
            if isinstance(new_value, str):
                new_value = float(new_value.replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError) as e:
            return json.dumps(
                {
                    "error": f"Invalid numeric values: old_value='{old_value}', new_value='{new_value}', error: {str(e)}"
                }
            )

        # Calculate all possible interpretations
        standard_change = new_value - old_value  # Standard: new - old
        cost_change = (
            old_value - new_value
        )  # For costs: old - new (decrease is positive)
        absolute_change = abs(new_value - old_value)
        percentage_change = (
            ((new_value - old_value) / old_value) * 100 if old_value != 0 else 0
        )

        # Simple mode: return just the recommended numeric value
        if change_type == "simple":
            # Auto-detect the most likely correct interpretation
            if old_value < 0 and new_value < 0:
                # For costs (negative values), use cost convention
                return str(cost_change)
            else:
                # Standard convention (new - old)
                return str(standard_change)

        # Full JSON mode (original behavior)
        result: dict[str, Any] = {
            "old_value": old_value,
            "new_value": new_value,
            "standard_change": standard_change,
            "cost_change": cost_change,
            "absolute_change": absolute_change,
            "percentage_change": percentage_change,
            "recommended": None,
            "data_validation_warning": None,
            "reasoning": None,
        }

        # Add data validation warnings for common errors
        if abs(standard_change) < 10 and abs(cost_change) < 10:
            result["data_validation_warning"] = (
                "⚠️  Small change detected. If question asks about 'senior notes' or specific instruments, verify you're using the right table row, not aggregate/general values."
            )

        # Auto-detect the most likely correct interpretation
        if change_type == "auto":
            # For costs (negative values), use cost convention
            if old_value < 0 and new_value < 0:
                result["recommended"] = cost_change
                result["reasoning"] = (
                    "Using cost convention (old - new) for negative values"
                )
            else:
                result["recommended"] = standard_change
                result["reasoning"] = (
                    "Using standard convention (new - old) for positive values"
                )
        elif change_type == "cost":
            result["recommended"] = cost_change
            result["reasoning"] = "Using cost convention (old - new)"
        elif change_type == "standard":
            result["recommended"] = standard_change
            result["reasoning"] = "Using standard convention (new - old)"
        elif change_type == "percentage":
            result["recommended"] = percentage_change
            result["reasoning"] = "Using percentage change"

        return json.dumps(result)

    except Exception as e:
        return json.dumps({"error": f"Error calculating change: {str(e)}"})


@tool
def validate_data_selection(
    question: str, selected_values: str, table_context: str
) -> str:
    """Validate if selected data matches the question requirements.

    Use this before calculations to ensure you're using the right data.

    Args:
        question: The original question asking for calculations
        selected_values: Description of the values you plan to use (e.g., "using -26 and -27 from general debt costs row")
        table_context: Brief description of available table data

    Returns:
        JSON validation result with warnings and suggestions
    """
    question_lower = question.lower()
    selected_lower = selected_values.lower()

    warnings = []
    suggestions = []

    # Check for specific instrument mentions
    if "senior notes" in question_lower:
        if "senior" not in selected_lower:
            warnings.append(
                "🚨 Question asks for 'senior notes' but selected data doesn't mention 'senior'"
            )
            suggestions.append(
                "Look for table rows specifically labeled with 'senior notes' or similar"
            )

    if "debt issuance costs" in question_lower:
        if "general" in selected_lower or "aggregate" in selected_lower:
            warnings.append(
                "⚠️  Using general/aggregate debt costs - consider if specific instrument costs are needed"
            )

    # Check for percentage questions
    if "percentage" in question_lower or "%" in question_lower:
        if "previous" in question_lower or "that" in question_lower:
            suggestions.append(
                "💡 This seems to be a follow-up question - use the same data from previous calculation"
            )

    return json.dumps(
        {
            "validation_status": "warnings_found" if warnings else "looks_good",
            "warnings": warnings,
            "suggestions": suggestions,
            "recommendation": "Proceed with calculation"
            if not warnings
            else "Review data selection before calculating",
        }
    )


@tool
def final_answer(answer: str) -> str:
    """Extract and return the final numeric answer from calculations.

    Use this tool when you have determined the final answer and want to return
    just the clean numeric value without any formatting or explanation.

    Args:
        answer: The final numeric answer (e.g., "60.94", "-4", "0.25")

    Returns:
        The clean numeric answer
    """
    try:
        # Clean the answer of common formatting
        cleaned = str(answer).strip()

        # Remove common prefixes/suffixes
        cleaned = cleaned.replace("$", "").replace("%", "").replace(",", "")

        # Try to parse as float to validate it's a number
        float(cleaned)

        return cleaned
    except (ValueError, TypeError):
        return str(answer).strip()


@tool
def get_table_value(
    row_identifier: str, column_identifier: str, table_name: str = "financial_data"
) -> str:
    """Get a specific value from the financial table by row and column identifiers.

    This tool helps extract specific numeric values for calculations.

    Args:
        row_identifier: Description of the row to find (e.g., "equipment rents payable", "2008", "senior notes")
        column_identifier: Description of the column to find (e.g., "2008", "dec 31 2008")
        table_name: Name of the table (optional)

    Returns:
        The numeric value as a clean string, or error message
    """
    current_record = _get_current_record()
    if current_record is None:
        return "ERROR: No record context set"

    try:
        financial_table = current_record.get_financial_table()
        df = financial_table.to_dataframe()

        # Input validation
        if not row_identifier or not column_identifier:
            return "ERROR: Both row_identifier and column_identifier are required"

        # Convert identifiers to lowercase for matching
        row_id_lower = row_identifier.lower().strip()
        col_id_lower = column_identifier.lower().strip()

        logger.info(
            f"Searching for row: '{row_identifier}' column: '{column_identifier}'"
        )

        # Find matching column with improved error handling
        matching_col = None
        available_cols = list(df.columns)

        # First try exact match (case insensitive)
        for col in available_cols:
            col_str = str(col).lower()
            if col_id_lower == col_str:
                matching_col = col
                break

        # Then try partial match
        if matching_col is None:
            for col in available_cols:
                col_str = str(col).lower()
                if col_id_lower in col_str or col_str in col_id_lower:
                    matching_col = col
                    break

        if matching_col is None:
            return f"ERROR: Column '{column_identifier}' not found. Available columns: {[str(col) for col in available_cols]}"

        # Enhanced fuzzy row matching with better error handling
        matching_row_idx = None
        best_match_score: float = 0.0
        available_rows = []

        try:
            available_rows = [str(idx) for idx in df.index]
        except Exception as e:
            logger.debug(f"Error getting row indices: {e}")
            available_rows = [f"row_{i}" for i in range(len(df))]

        # First try exact index match
        for idx in df.index:
            try:
                idx_text = str(idx).lower()
                if row_id_lower == idx_text:
                    matching_row_idx = idx
                    best_match_score = 1.0
                    break
                elif row_id_lower in idx_text or idx_text in row_id_lower:
                    if best_match_score < 0.8:
                        matching_row_idx = idx
                        best_match_score = 0.8
            except Exception as e:
                logger.debug(f"Error matching row index {idx}: {e}")
                continue

        # If no exact match, try fuzzy matching on index with improved logic
        if matching_row_idx is None or best_match_score < 0.8:
            for idx in df.index:
                try:
                    idx_text = str(idx).lower()
                    # Split both strings and check for word matches
                    row_words = set(row_id_lower.split())
                    idx_words = set(idx_text.split())

                    # Calculate word overlap score
                    if row_words and idx_words:
                        overlap = len(row_words.intersection(idx_words))
                        union_size = len(row_words.union(idx_words))
                        score: float = overlap / union_size if union_size > 0 else 0.0

                        # Boost score for important financial terms
                        financial_boost: float = 0.0
                        financial_terms = [
                            "credit",
                            "facility",
                            "debt",
                            "notes",
                            "senior",
                            "total",
                            "revenue",
                            "expense",
                            "cost",
                        ]
                        for term in financial_terms:
                            if term in row_id_lower and term in idx_text:
                                financial_boost += 0.3

                        final_score = min(1.0, score + financial_boost)

                        # Accept if good overlap (>40%) or strong financial term match
                        if final_score > 0.4 and final_score > best_match_score:
                            matching_row_idx = idx
                            best_match_score = final_score

                except Exception as e:
                    logger.debug(f"Error in fuzzy matching for row {idx}: {e}")
                    continue

        # If still no match, try finding by row content with safer iteration
        if matching_row_idx is None:
            for idx in df.index:
                try:
                    row = df.loc[idx]
                    # Safely convert row values to string for searching
                    row_values = []
                    for val in row.values:
                        try:
                            if pd.notna(val):
                                row_values.append(str(val).lower())
                        except Exception:
                            continue

                    row_text = " ".join(row_values)

                    # Check if row identifier appears in row content
                    if row_id_lower in row_text:
                        matching_row_idx = idx
                        break

                    # Also try word matching for financial terms
                    row_words = set(row_text.split())
                    search_words = set(row_id_lower.split())

                    if row_words and search_words:
                        overlap = len(row_words.intersection(search_words))
                        # Require at least 2 word matches or 1 strong financial term match
                        if overlap >= 2 or any(
                            term in row_text
                            for term in [
                                "credit",
                                "facility",
                                "debt",
                                "notes",
                                "senior",
                            ]
                        ):
                            matching_row_idx = idx
                            break

                except Exception as e:
                    logger.debug(f"Error searching row content for {idx}: {e}")
                    continue

        if matching_row_idx is None:
            return f"ERROR: Row '{row_identifier}' not found. Available rows: {available_rows[:10]}{'...' if len(available_rows) > 10 else ''}"

        # Get the value with better error handling
        try:
            value = df.loc[matching_row_idx, matching_col]
            logger.info(
                f"Found value at row '{matching_row_idx}' column '{matching_col}': {value}"
            )
        except Exception as e:
            return f"ERROR: Failed to retrieve value at row '{matching_row_idx}', column '{matching_col}': {str(e)}"

        # Clean and return as string
        if pd.isna(value):
            return "ERROR: Value is null/missing"

        # Convert to clean numeric string with better error handling
        try:
            # Try to convert to float first
            if isinstance(value, int | float):
                clean_value = float(value)
                return str(clean_value)

            # If it's a string, try to clean and convert
            value_str = str(value).strip()

            # Remove common formatting
            cleaned_str = (
                value_str.replace(",", "")
                .replace("$", "")
                .replace("%", "")
                .replace("(", "-")
                .replace(")", "")
            )

            # Try to convert to float
            clean_value = float(cleaned_str)
            return str(clean_value)

        except (ValueError, TypeError) as e:
            logger.debug(f"Could not convert value to numeric: {value}, error: {e}")
            # Return as string if can't convert to number
            return str(value).strip()

    except Exception as e:
        error_msg = f"ERROR: Unexpected error in get_table_value: {str(e)}"
        logger.error(error_msg)
        return error_msg
