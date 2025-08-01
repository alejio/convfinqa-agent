"""
LLM agent for ConvFinQA using smolagents CodeAgent.
"""

import os
import random
import time
from functools import wraps
from typing import Any

from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

from ..core.conversation import ConversationManager
from ..core.logger import get_logger
from ..core.models import Record
from ..data.loader import DataLoader
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

load_dotenv()

logger = get_logger(__name__)


def with_rate_limiting(max_retries: int = 5, base_delay: float = 1.0) -> Any:
    """Decorator to add exponential backoff retry logic for rate limiting.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds before first retry
    """

    def decorator(func: Any) -> Any:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    # Check for rate limiting errors
                    if any(
                        keyword in error_str
                        for keyword in ["rate", "429", "too many requests", "quota"]
                    ):
                        if attempt == max_retries - 1:
                            logger.error(
                                f"Rate limiting failed after {max_retries} attempts: {e}"
                            )
                            raise

                        # Exponential backoff with jitter
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        # Non-rate-limiting error, don't retry
                        raise
            return None

        return wrapper

    return decorator


class ConvFinQAAgent:
    """LLM agent for conversational financial question answering using smolagents."""

    def __init__(self, model: str = "gpt-4.1") -> None:
        """Initialize the smolagents-based agent.

        Args:
            model: The model to use for the agent. Defaults to "gpt-4.1".
        """
        self.model = model
        self.tools = [
            list_tables,
            show_table,
            query_table,
            get_table_value,
            compute,
            calculate_change,
            validate_data_selection,
            final_answer,
        ]
        self.conversation_manager = ConversationManager()
        self.conversation_history: list[dict[str, str]] = []

        # Initialize smolagents CodeAgent with OpenAI model
        try:
            llm_model = LiteLLMModel(
                model=f"openai/{model}",
                model_id=f"openai/{model}",  # Explicitly set model_id to avoid default
            )
            self.agent = CodeAgent(
                tools=self.tools,
                model=llm_model,
            )
            logger.info(
                f"Initialized ConvFinQA agent with smolagents CodeAgent, model: {model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize smolagents CodeAgent: {e}")
            raise

        # Check for OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY not found in environment variables")

    def set_record_context(
        self, record: Record, data_loader: DataLoader, session_id: str | None = None
    ) -> None:
        """Set the current record context for tool functions and start conversation.

        Args:
            record: The record to set as current context.
            data_loader: The data loader to set as current context.
            session_id: Optional session ID to resume previous conversation.
        """
        set_context(record, data_loader)
        self.conversation_manager.start_conversation(record.id, session_id)

        session_id = (
            self.conversation_manager.current_state.session_id
            if self.conversation_manager.current_state
            else "unknown"
        )
        logger.info(f"Set record context to: {record.id}, session: {session_id}")

    @with_rate_limiting(max_retries=5, base_delay=1.0)
    def chat(self, message: str) -> str:
        """Process a chat message using smolagents CodeAgent with rate limiting protection.

        Args:
            message: The user's message/question.

        Returns:
            The agent's response.
        """
        try:
            # Build contextual message with conversation history
            contextual_message = self._build_contextual_message(message)

            # Pass contextual message to smolagents CodeAgent
            response = self.agent.run(contextual_message)
            response_str = str(response)

            # Extract entities from response for conversation tracking
            referenced_entities = self._extract_entities_from_response(response_str)

            # Store conversation turn
            self.conversation_manager.add_turn(
                user_message=message,
                assistant_response=response_str,
                tool_calls=[],  # smolagents handles tool calls internally
                referenced_entities=referenced_entities,
            )

            # Keep legacy conversation history for backward compatibility
            self.conversation_history.append(
                {"user": message, "assistant": response_str}
            )

            logger.info(
                f"Processed chat message with smolagents, "
                f"response length: {len(response_str)}, entities: {referenced_entities}"
            )
            return response_str

        except Exception as e:
            error_msg = f"Error processing chat message: {str(e)}"
            logger.error(error_msg)
            return f"I encountered an error while processing your question: {str(e)}"

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the conversation history.

        Returns:
            List of conversation turns with user messages and assistant responses.
        """
        return self.conversation_history.copy()

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history.clear()
        if self.conversation_manager.current_state:
            self.conversation_manager.current_state.turns.clear()
        logger.info("Cleared conversation history")

    def get_current_session_id(self) -> str | None:
        """Get the current conversation session ID."""
        if self.conversation_manager.current_state:
            return self.conversation_manager.current_state.session_id
        return None

    def _build_contextual_message(self, message: str) -> str:
        """Build a contextual message that includes conversation history.

        Args:
            message: The current user message

        Returns:
            Enhanced message with conversation context
        """
        if (
            not self.conversation_manager.current_state
            or not self.conversation_manager.current_state.turns
        ):
            return self._build_initial_message(message)

        # Get recent conversation turns for context
        recent_turns = self.conversation_manager.current_state.turns[
            -3:
        ]  # Back to 3 turns for better context

        # Build specific context about what was discussed
        context_parts = ["CONVERSATION CONTEXT:"]
        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"Previous Q{i}: {turn.user_message}")
            context_parts.append(f"Previous A{i}: {turn.assistant_response}")

        context_str = "\n".join(context_parts)

        # Build contextual message with comprehensive guidance (restored)
        contextual_message = f"""
{context_str}

CURRENT QUESTION: {message}

CRITICAL CONTEXT ANALYSIS:
- If the current question contains words like "it", "that", "the same", "also", "and what was", "in the previous year", etc., these refer to topics or values from the conversation above.
- Use the previous questions and answers to understand what these pronouns and references mean.
- Maintain consistency with previous calculations and data.
- If asking about a different year or time period of the same metric, use the same table rows/columns but different time periods.

SYSTEMATIC ANALYSIS APPROACH:
1. UNDERSTAND THE QUESTION:
   - Identify what financial metric is being asked for
   - Determine if it's a lookup, calculation, comparison, or change analysis
   - Note any time periods, specific items, or conditions mentioned

2. DATA EXPLORATION STRATEGY:
   - Start with show_table() to see all available data if unfamiliar with the table
   - Use validate_data_selection() to confirm you're looking at the right rows/columns
   - Use get_table_value() for precise single-cell extractions with SPECIFIC FINANCIAL TERMINOLOGY

3. CALCULATION METHODOLOGY:
   - For simple lookups: Use get_table_value() with specific financial line item names (e.g., "senior notes", "debt issuance costs", "total revenue")
   - For changes/differences: Use calculate_change() with "simple" mode for clean numeric results
   - For complex calculations: Break into steps, validate each component
   - For comparisons: Always validate you're comparing the same metrics from correct time periods

4. COMMON FINANCIAL PATTERNS:
   - Revenue questions: Look for revenue, sales, or income rows
   - Expense questions: Look for costs, expenses, or negative values
   - Change questions: Identify old vs new values, use calculate_change()
   - Ratio questions: Get individual components, then divide
   - Percentage questions: Calculate ratio, multiply by 100 if needed

CRITICAL - AVOID GENERIC REFERENCES:
❌ NEVER use vague terms like "first row", "second column", "top line", "bottom entry"
✅ ALWAYS use specific financial terms like "senior notes", "debt issuance costs", "total revenue", "operating expenses"
❌ NEVER say "the value in row 0" or "cell [1,2]"
✅ ALWAYS say "the senior notes value for 2008" or "debt issuance costs for the current year"

TOOL SELECTION GUIDANCE:
- show_table(): Use when you need to see the full table structure or are unsure about data layout
- get_table_value(): Use for extracting specific single values with precise FINANCIAL TERMINOLOGY for row/column references
- calculate_change(): Use for any "increase", "decrease", "change" questions
- compute(): Use for mathematical operations between extracted values
- validate_data_selection(): Use before calculations to verify you have the right data
- final_answer(): ALWAYS use this as your last step with just the numeric result

ANSWER EXTRACTION RULES:
- Extract only the final numeric answer, no explanations
- Format as clean number (e.g., "60.94", "-4", "25.14")
- For percentages, provide as decimal (e.g., "0.1083" not "10.83%") unless context clearly indicates percentage format expected
- Remove currency symbols, commas, and units
- For negative values, include the minus sign

QUALITY CHECKS:
- Always double-check your data sources (right financial line items, columns, time periods)
- Verify calculations make logical sense
- Ensure you're answering exactly what was asked
- Use validate_data_selection() to confirm data quality before final calculations
- Always reference financial line items by their proper names, not position

REQUIRED FINAL STEP: Always end by calling final_answer() with ONLY the clean numeric result.
"""
        return contextual_message

    def _build_initial_message(self, message: str) -> str:
        """Build message for initial conversation turn with comprehensive guidance."""
        return f"""
CURRENT QUESTION: {message}

You are a financial analysis assistant. Follow this systematic approach:

STEP 1 - UNDERSTAND THE QUESTION:
- Identify the specific financial metric being requested
- Note any time periods, conditions, or constraints
- Determine if this is a lookup, calculation, comparison, or trend analysis

STEP 2 - EXPLORE THE DATA:
- Use show_table() first to understand the table structure and available data
- Identify relevant financial line items and time period columns for your analysis
- Use validate_data_selection() to confirm data quality

STEP 3 - EXTRACT DATA SYSTEMATICALLY:
- Use get_table_value() for precise single-cell extractions
- Always specify exact financial line item names and time period column references
- Double-check you're getting data from correct time periods

CRITICAL - USE SPECIFIC FINANCIAL TERMINOLOGY:
❌ NEVER use vague terms like "first row", "second column", "top line", "bottom entry"
✅ ALWAYS use specific financial terms like "senior notes", "debt issuance costs", "total revenue", "operating expenses"
❌ NEVER say "the value in row 0" or "cell [1,2]"
✅ ALWAYS say "the senior notes value for 2008" or "debt issuance costs for the current year"

STEP 4 - PERFORM CALCULATIONS:
- For simple lookups: Extract the value directly using proper financial terminology
- For changes: Use calculate_change() with old_value, new_value, and "simple" mode
- For complex calculations: Break into steps, use compute() for math operations
- Always validate intermediate results

STEP 5 - FORMAT FINAL ANSWER:
- Use final_answer() as your last step
- Provide ONLY the clean numeric result
- Format: plain number (e.g., "60.94", "-4", "0.1083")
- Remove currency symbols, commas, explanatory text

TOOL USAGE PATTERNS:
- show_table() → validate_data_selection() → get_table_value() → calculate_change()/compute() → final_answer()
- Always end with final_answer() containing just the numeric result

COMMON QUESTION TYPES:
- "What is/was [metric] in [year]?" → Direct lookup with get_table_value() using specific financial terms
- "How much did [metric] change?" → Use calculate_change() with old vs new values
- "What's the difference between [A] and [B]?" → Extract both values using proper terminology, subtract
- "Calculate [formula]" → Break into components, use compute()

QUALITY CHECKS:
- Reference financial line items by their proper names, not position
- Verify you're using the correct time periods and financial metrics
- Ensure calculations make logical sense for the financial context

CRITICAL: Your response should end with final_answer() containing only the numeric result.
"""

    def _extract_entities_from_response(self, response: str) -> list[str]:
        """Extract financial entities mentioned in the response.

        Args:
            response: The agent's response

        Returns:
            List of entities found in the response
        """
        entities: list[str] = []

        # Use shared financial terms for entity extraction
        from ..core.financial_terms import FinancialTerms

        entities.extend(FinancialTerms.extract_financial_terms(response))

        return entities

    def get_agent_info(self) -> dict[str, Any]:
        """Get information about current agent configuration.

        Returns:
            Dictionary with agent configuration details.
        """
        return {
            "model": self.model,
            "architecture": "smolagents CodeAgent",
            "tools": [getattr(tool, "__name__", str(tool)) for tool in self.tools],
            "conversation_enabled": True,
        }
