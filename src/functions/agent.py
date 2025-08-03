"""
LLM agent for ConvFinQA using smolagents CodeAgent.
"""

import os
import random
import time
from functools import wraps
from typing import Any

import dspy
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

from ..core.conversation import ConversationManager
from ..core.logger import get_logger
from ..core.models import Record
from ..data.loader import DataLoader
from .dspy_signatures import (
    ConversationalReferenceResolution,
    build_dspy_prompt,
    build_initial_dspy_prompt,
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
                    if any(
                        keyword in error_str
                        for keyword in ["rate", "429", "too many requests", "quota"]
                    ):
                        if attempt == max_retries - 1:
                            logger.error(
                                f"Rate limiting failed after {max_retries} attempts: {e}"
                            )
                            raise

                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Rate limited (attempt {attempt + 1}/{max_retries}), retrying in {delay:.2f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        raise
            return None

        return wrapper

    return decorator


class ConvFinQAAgent:
    """LLM agent for conversational financial question answering using smolagents."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        """Initialize the smolagents-based agent with token optimization.

        Args:
            model: The model to use for the agent. Defaults to "gpt-4o-mini".
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

        try:
            llm_model = LiteLLMModel(
                model=f"openai/{model}",
                model_id=f"openai/{model}",
            )
            self.agent = CodeAgent(
                tools=self.tools,
                model=llm_model,
            )
            # Add reference resolution tool
            self.reference_resolver = dspy.ChainOfThought(
                ConversationalReferenceResolution
            )
            logger.info(
                f"Initialized ConvFinQA agent with smolagents CodeAgent, model: {model}"
            )
        except Exception as e:
            logger.error(f"Failed to initialize smolagents CodeAgent: {e}")
            raise

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
            contextual_message = self._build_contextual_message(message)
            response = self.agent.run(contextual_message)
            response_str = str(response)

            referenced_entities = self._extract_entities_from_response(response_str)
            self.conversation_manager.add_turn(
                user_message=message,
                assistant_response=response_str,
                tool_calls=[],
                referenced_entities=referenced_entities,
            )

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

    def _resolve_references(self, message: str, conversation_history: str) -> str:
        """Resolve references in the message using the conversation history."""
        try:
            if not conversation_history.strip():
                return message  # No history to resolve from

            # Use the DSPy signature to resolve references
            resolved = self.reference_resolver(
                current_question=message,
                conversation_history=conversation_history,
            )

            # Log the resolution
            if resolved.resolved_question.lower() != message.lower():
                logger.info(
                    f"Resolved query: '{message}' -> '{resolved.resolved_question}'"
                )

            return str(resolved.resolved_question)
        except Exception as e:
            logger.error(f"Failed to resolve references: {e}")
            return message  # Fallback to original message

    def get_conversation_history(self) -> list[dict[str, str]]:
        """Get the conversation history."""
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
        """Build a contextual message that includes resolved conversation history."""
        if (
            not self.conversation_manager.current_state
            or not self.conversation_manager.current_state.turns
        ):
            return self._build_initial_message(message)

        recent_turns = self.conversation_manager.current_state.turns[-3:]
        context_parts = []
        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(f"Q{i}: {turn.user_message}")
            context_parts.append(f"A{i}: {turn.assistant_response}")
        context_str = " | ".join(context_parts)

        # Resolve references in the current message before creating the prompt
        resolved_message = self._resolve_references(message, context_str)

        return build_dspy_prompt(resolved_message, context_str)

    def _build_initial_message(self, message: str) -> str:
        """Build message for initial conversation turn using token-optimized prompts."""
        return build_initial_dspy_prompt(message)

    def _extract_entities_from_response(self, response: str) -> list[str]:
        """Extract financial entities mentioned in the response."""
        from ..core.financial_terms import get_financial_terms_instance

        terms_instance = get_financial_terms_instance()
        conversation_context = getattr(self, "conversation_context", "")
        extracted_entities = terms_instance.extract_entities(
            response, conversation_context
        )
        return extracted_entities

    def get_agent_info(self) -> dict[str, Any]:
        """Get information about current agent configuration."""
        return {
            "model": self.model,
            "architecture": "smolagents CodeAgent",
            "tools": [getattr(tool, "__name__", str(tool)) for tool in self.tools],
            "conversation_enabled": True,
        }
