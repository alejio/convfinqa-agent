"""ConvFinQA agent and tool functions with token-optimized architecture."""

from .agent import ConvFinQAAgent
from .dspy_signatures import (
    ConversationalReferenceResolution,
    build_dspy_prompt,
    build_initial_dspy_prompt,
)
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
    calculate_change,
    compute,
    final_answer,
    get_table_value,
    list_tables,
    query_table,
    set_context,
    show_table,
    validate_data_selection,
)

__all__ = [
    # Main agent
    "ConvFinQAAgent",
    # DSPy Signatures
    "build_dspy_prompt",
    "build_initial_dspy_prompt",
    "ConversationalReferenceResolution",
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
    "calculate_change",
    "validate_data_selection",
]
