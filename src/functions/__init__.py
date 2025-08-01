"""ConvFinQA agent and tool functions with PEC+DSPy architecture."""

from .agent import ConvFinQAAgent
from .math import (
    CONSTANTS,
    MATH_FUNCTIONS,
    Number,
    add,
    divide,
    exp,
    get_constant,
    greater,
    multiply,
    subtract,
)
from .tools import (
    compute,
    final_answer,
    get_table_value,
    list_tables,
    query_table,
    set_context,
    show_table,
)

__all__ = [
    # Main agent
    "ConvFinQAAgent",
    # Math functions
    "Number",
    "add",
    "subtract",
    "multiply",
    "divide",
    "greater",
    "exp",
    "get_constant",
    "CONSTANTS",
    "MATH_FUNCTIONS",
    # LLM tools
    "set_context",
    "list_tables",
    "show_table",
    "query_table",
    "get_table_value",
    "compute",
    "final_answer",
]
