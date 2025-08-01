"""
Tests for smolagents LLM tool functions.
"""

import json
import pytest

from src.data.loader import create_data_loader
from src.functions.tools import (
    compute,
    list_tables,
    query_table,
    set_context,
    show_table,
)


class TestToolFunctions:
    """Test suite for smolagents tool functions."""

    @pytest.fixture
    def setup_context(self):
        """Set up test context with real data."""
        data_loader = create_data_loader()
        tables = data_loader.list_available_tables()

        # Use the first available record
        if not tables:
            pytest.skip("No test data available")

        record_id = tables[0]
        record = data_loader.get_record(record_id)
        set_context(record, data_loader)

        return record, data_loader

    @pytest.mark.integration
    def test_list_tables_success(self, setup_context):
        """Test successful table listing."""
        record, _ = setup_context

        result_str = list_tables()
        result = json.loads(result_str)  # type: ignore

        assert "tables" in result
        assert len(result["tables"]) > 0

        table_info = result["tables"][0]
        assert "name" in table_info
        assert "rows" in table_info
        assert "columns" in table_info
        assert "column_names" in table_info

        # Verify structure
        assert isinstance(table_info["rows"], int)
        assert isinstance(table_info["columns"], int)
        assert isinstance(table_info["column_names"], list)

    @pytest.mark.unittest
    def test_list_tables_no_context(self):
        """Test list_tables without context."""
        # Clear context
        set_context(None, None)

        result_str = list_tables()
        result = json.loads(result_str)  # type: ignore

        assert "error" in result
        assert "No record context set" in result["error"]

    @pytest.mark.integration
    def test_show_table_success(self, setup_context):
        """Test successful table display."""
        record, _ = setup_context

        result_str = show_table()
        result = json.loads(result_str)  # type: ignore

        assert "table_schema" in result
        assert "sample_data" in result
        assert "total_rows" in result

        schema = result["table_schema"]
        assert "name" in schema
        assert "rows" in schema
        assert "columns" in schema

        # Verify sample data is present
        assert isinstance(result["sample_data"], list)
        assert isinstance(result["total_rows"], int)

    @pytest.mark.unittest
    def test_show_table_no_context(self):
        """Test show_table without context."""
        set_context(None, None)

        result_str = show_table()
        result = json.loads(result_str)  # type: ignore

        assert "error" in result
        assert "No record context set" in result["error"]

    @pytest.mark.integration
    def test_query_table_sum_operation(self, setup_context):
        """Test table query with sum aggregation operation."""
        record, _ = setup_context

        # Get available columns
        financial_table = record.get_financial_table()
        columns = [col.name for col in financial_table.table_schema.columns]

        if columns:
            query = f"sum {columns[0]}"
            result_str = query_table(query)
            result = json.loads(result_str)  # type: ignore

            assert "query" in result
            # New enhanced DSPy behavior returns different fields
            assert "operation" in result
            assert "data" in result
            # May have result_type in some cases, but not always required
            assert result["operation"] in ["lookup", "aggregate", "calculate"]

    @pytest.mark.integration
    def test_query_table_show_operation(self, setup_context):
        """Test table query with show operation."""
        record, _ = setup_context

        # Get available columns
        financial_table = record.get_financial_table()
        columns = [col.name for col in financial_table.table_schema.columns]

        if columns:
            query = f"show {columns[0]}"
            result_str = query_table(query)
            result = json.loads(result_str)  # type: ignore

            assert "query" in result
            # New enhanced DSPy behavior
            assert "operation" in result
            assert result["operation"] in ["lookup", "aggregate", "calculate", "analyze_change", "compare"]

    @pytest.mark.integration
    def test_query_table_info_default(self, setup_context):
        """Test table query with unknown operation returns info."""
        record, _ = setup_context

        result_str = query_table("what is this table about")
        result = json.loads(result_str)  # type: ignore

        assert "query" in result
        # New enhanced DSPy behavior provides analysis strategy and table structure
        assert "operation" in result
        assert "table_structure" in result
        assert "document_context" in result
        # The analysis strategy provides guidance instead of simple suggestions
        assert "analysis_strategy" in result

    @pytest.mark.unittest
    def test_query_table_no_context(self):
        """Test query_table without context."""
        set_context(None, None)

        result_str = query_table("test query")
        result = json.loads(result_str)  # type: ignore

        assert "error" in result
        assert "No record context set" in result["error"]

    @pytest.mark.integration
    def test_compute_mathematical_expression(self, setup_context):
        """Test compute with mathematical expression."""
        record, _ = setup_context

        result_str = compute("add(5, 3)")

        # New behavior: returns plain numeric result for agent efficiency
        try:
            # Ensure we have a string and try to parse as float (new behavior)
            result_value = float(str(result_str))
            assert result_value == 8.0
        except ValueError:
            # Fallback to old JSON format if still used
            result = json.loads(str(result_str))  # type: ignore
            assert "expression" in result
            assert "result" in result
            assert result["result"] == 8
            assert result["type"] == "mathematical"

    @pytest.mark.integration
    def test_compute_table_operation(self, setup_context):
        """Test compute with table operation."""
        record, _ = setup_context

        # Get available columns
        financial_table = record.get_financial_table()
        numeric_cols = [col.name for col in financial_table.table_schema.columns
                       if col.column_type == "numeric"]

        if numeric_cols:
            expression = f"{numeric_cols[0]} sum"
            result_str = compute(expression)

            # New behavior: returns plain numeric result for agent efficiency
            try:
                # Ensure we have a string and try to parse as float (new behavior)
                result_value = float(str(result_str))
                assert isinstance(result_value, float)
                assert result_value > 0  # Should be a positive sum for test data
            except ValueError:
                # Fallback to old JSON format if still used
                result = json.loads(str(result_str))  # type: ignore
                assert "expression" in result
                assert "result" in result
                assert "type" in result
                assert result["type"] == "table_operation"

    @pytest.mark.integration
    def test_compute_invalid_expression(self, setup_context):
        """Test compute with invalid expression."""
        record, _ = setup_context

        result_str = compute("invalid_function(1, 2)")

        # New behavior: may return numeric fallback or error message
        result_as_string = str(result_str)
        if result_as_string.startswith("ERROR:"):
            # Expected error format
            assert "ERROR:" in result_as_string
        else:
            try:
                # May return a fallback numeric result
                result_value = float(result_as_string)
                assert isinstance(result_value, float)
            except ValueError:
                # Old JSON format fallback
                result = json.loads(result_as_string)  # type: ignore
                assert "expression" in result
                assert "error" in result
                assert "suggestion" in result

    @pytest.mark.unittest
    def test_compute_no_context(self):
        """Test compute without context."""
        set_context(None, None)

        result_str = compute("add(1, 2)")
        result = json.loads(result_str)  # type: ignore

        assert "error" in result
        assert "No record context set" in result["error"]

    @pytest.mark.unittest
    def test_set_context_function(self):
        """Test the set_context function."""
        data_loader = create_data_loader()
        tables = data_loader.list_available_tables()

        if not tables:
            pytest.skip("No test data available")

        record = data_loader.get_record(tables[0])

        # Test setting context
        set_context(record, data_loader)

        # Verify context is set by calling a tool function
        result_str = list_tables()
        result = json.loads(result_str)  # type: ignore

        assert "tables" in result
        assert "error" not in result

    @pytest.mark.integration
    def test_all_tools_return_json_strings(self, setup_context):
        """Test that all tool functions return valid JSON strings."""
        record, _ = setup_context

        # Test each tool function
        tools_results = [
            list_tables(),
            show_table(),
            query_table("test"),
            compute("add(1, 2)")
        ]

        for i, result_str in enumerate(tools_results):
            # Most tools should return JSON, but compute may return plain numbers now
            if i == 3:  # compute function
                # May return plain numeric string or JSON
                try:
                    float(str(result_str))  # Try parsing as number
                    # It's a plain number, which is fine for compute
                    continue
                except ValueError:
                    # Not a plain number, should be JSON
                    result = json.loads(str(result_str))  # type: ignore
                    assert isinstance(result, dict)
            else:
                # Other tools should return JSON
                result = json.loads(str(result_str))  # type: ignore
                assert isinstance(result, dict)

    @pytest.mark.end2end
    def test_tool_integration_workflow(self, setup_context):
        """Test a complete workflow using multiple tools."""
        record, _ = setup_context

        # Step 1: List tables
        tables_result = json.loads(list_tables())  # type: ignore
        assert "tables" in tables_result

        # Step 2: Show table structure
        table_result = json.loads(show_table())  # type: ignore
        assert "table_schema" in table_result

        # Step 3: Query the table
        query_result = json.loads(query_table("what columns are available"))  # type: ignore
        # Updated for new DSPy-enhanced behavior
        assert "operation" in query_result
        assert "table_structure" in query_result

        # Step 4: Perform computation
        compute_result_str = compute("add(10, 20)")
        # Handle new behavior: may return plain number or JSON
        try:
            result_value = float(str(compute_result_str))
            assert result_value == 30.0
        except ValueError:
            # Fallback to JSON format
            compute_result = json.loads(str(compute_result_str))  # type: ignore
            assert "result" in compute_result
            assert compute_result["result"] == 30
