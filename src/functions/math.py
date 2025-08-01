"""
Core mathematical functions for ConvFinQA program execution.
"""

import math

from ..core.logger import get_logger

logger = get_logger(__name__)

Number = int | float


def subtract(a: Number, b: Number) -> Number:
    """Subtract b from a."""
    result = a - b
    logger.debug(f"subtract({a}, {b}) = {result}")
    return result


def add(a: Number, b: Number) -> Number:
    """Add a and b."""
    result = a + b
    logger.debug(f"add({a}, {b}) = {result}")
    return result


def multiply(a: Number, b: Number) -> Number:
    """Multiply a and b."""
    result = a * b
    logger.debug(f"multiply({a}, {b}) = {result}")
    return result


def divide(a: Number, b: Number) -> Number:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Division by zero is not allowed")
    result = a / b
    logger.debug(f"divide({a}, {b}) = {result}")
    return result


def greater(a: Number, b: Number) -> bool:
    """Check if a is greater than b."""
    result = a > b
    logger.debug(f"greater({a}, {b}) = {result}")
    return result


def exp(a: Number) -> Number:
    """Calculate e^a."""
    result = math.exp(a)
    logger.debug(f"exp({a}) = {result}")
    return result


# Common constants used in ConvFinQA
CONSTANTS = {
    "const_100": 100,
    "const_1000": 1000,
    "const_10000": 10000,
    "const_100000": 100000,
    "const_1000000": 1000000,
    "const_m1": -1,
    "const_0": 0,
    "const_1": 1,
    "const_2": 2,
    "const_3": 3,
    "const_4": 4,
    "const_5": 5,
    "const_10": 10,
    "const_12": 12,
    "const_31": 31,
    "const_365": 365,
}


def get_constant(name: str) -> Number:
    """Get a constant value by name."""
    if name not in CONSTANTS:
        raise ValueError(f"Unknown constant: {name}")
    return CONSTANTS[name]


# Function registry for program execution
MATH_FUNCTIONS = {
    "subtract": subtract,
    "add": add,
    "multiply": multiply,
    "divide": divide,
    "greater": greater,
    "exp": exp,
}
