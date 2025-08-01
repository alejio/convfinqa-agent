"""
Tests for mathematical functions, constants, and function registry.
"""

import math
import pytest

from src.functions.math import (
    CONSTANTS,
    MATH_FUNCTIONS,
    add,
    divide,
    exp,
    get_constant,
    greater,
    multiply,
    subtract,
)


class TestMathematicalFunctions:
    """Test mathematical functions."""

    @pytest.mark.unittest
    def test_subtract(self):
        """Test subtract function."""
        assert subtract(10, 5) == 5
        assert subtract(0, 5) == -5
        assert subtract(-5, -3) == -2
        assert subtract(10.5, 2.5) == 8.0

    @pytest.mark.unittest
    def test_add(self):
        """Test add function."""
        assert add(10, 5) == 15
        assert add(0, 5) == 5
        assert add(-5, -3) == -8
        assert add(10.5, 2.5) == 13.0

    @pytest.mark.unittest
    def test_multiply(self):
        """Test multiply function."""
        assert multiply(10, 5) == 50
        assert multiply(0, 5) == 0
        assert multiply(-5, -3) == 15
        assert multiply(10.5, 2) == 21.0

    @pytest.mark.unittest
    def test_divide(self):
        """Test divide function."""
        assert divide(10, 5) == 2
        assert divide(0, 5) == 0
        assert divide(-10, -2) == 5
        assert divide(10.5, 2) == 5.25

    @pytest.mark.unittest
    def test_divide_by_zero(self):
        """Test divide by zero raises ValueError."""
        with pytest.raises(ValueError, match="Division by zero is not allowed"):
            divide(10, 0)

    @pytest.mark.unittest
    def test_greater(self):
        """Test greater function."""
        assert greater(10, 5) is True
        assert greater(5, 10) is False
        assert greater(5, 5) is False
        assert greater(10.5, 10) is True

    @pytest.mark.unittest
    def test_exp(self):
        """Test exp function."""
        assert exp(0) == 1
        assert exp(1) == math.e
        assert abs(exp(2) - math.e**2) < 1e-10


class TestConstants:
    """Test constants functionality."""

    @pytest.mark.unittest
    def test_get_constant(self):
        """Test get_constant function."""
        assert get_constant("const_100") == 100
        assert get_constant("const_1000") == 1000
        assert get_constant("const_m1") == -1
        assert get_constant("const_0") == 0

    @pytest.mark.unittest
    def test_get_constant_invalid(self):
        """Test get_constant with invalid name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown constant: invalid_const"):
            get_constant("invalid_const")

    @pytest.mark.unittest
    def test_constants_completeness(self):
        """Test that all expected constants are defined."""
        expected_constants = [
            "const_100", "const_1000", "const_10000", "const_100000", "const_1000000",
            "const_m1", "const_0", "const_1", "const_2", "const_3", "const_4", "const_5",
            "const_10", "const_12", "const_31", "const_365"
        ]
        for const_name in expected_constants:
            assert const_name in CONSTANTS


class TestFunctionRegistry:
    """Test function registry."""

    @pytest.mark.unittest
    def test_math_functions_registry(self):
        """Test that all math functions are registered."""
        expected_functions = ["subtract", "add", "multiply", "divide", "greater", "exp"]
        for func_name in expected_functions:
            assert func_name in MATH_FUNCTIONS
            assert callable(MATH_FUNCTIONS[func_name])

    @pytest.mark.unittest
    def test_function_registry_execution(self):
        """Test executing functions through registry."""
        assert MATH_FUNCTIONS["add"](5, 3) == 8
        assert MATH_FUNCTIONS["subtract"](10, 4) == 6
        assert MATH_FUNCTIONS["multiply"](6, 7) == 42
        assert MATH_FUNCTIONS["divide"](20, 4) == 5
        assert MATH_FUNCTIONS["greater"](10, 5) is True
        assert abs(MATH_FUNCTIONS["exp"](1) - math.e) < 1e-10
