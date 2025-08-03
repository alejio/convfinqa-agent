"""
Unified DSPy component management and fallback handling.

This module consolidates all DSPy initialization, configuration, and fallback logic
that was previously scattered across 6+ modules.
"""

import os
from collections.abc import Callable
from typing import Any, TypeVar

import dspy

from .logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


class DSPyComponentManager:
    """Unified DSPy component initialization and fallback handling."""

    def __init__(self, model: str = "gpt-4o-mini"):
        """Initialize DSPy component manager with model configuration.

        Args:
            model: The model name to use for DSPy operations
        """
        self.model = model
        self._components: dict[str, Any] = {}
        self._configured = False
        self._configure_dspy()

    def _configure_dspy(self) -> None:
        """Configure DSPy with the specified model."""
        try:
            # Check if DSPy is already configured
            if hasattr(dspy.settings, "lm") and dspy.settings.lm is not None:
                self._configured = True
                logger.debug("DSPy already configured")
                return

            # Configure DSPy with OpenAI model
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                lm = dspy.LM(f"openai/{self.model}", api_key=api_key)
                dspy.configure(lm=lm)
                self._configured = True
                logger.info(f"DSPy configured with {self.model}")
            else:
                logger.warning(
                    "OPENAI_API_KEY not found - DSPy operations will use fallbacks"
                )
                self._configured = False

        except Exception as e:
            logger.warning(
                f"Failed to configure DSPy: {e} - DSPy operations will use fallbacks"
            )
            self._configured = False

    def get_component(
        self,
        component_key: str,
        signature_class: type[T],
        component_type: str = "ChainOfThought",
        fallback_func: Callable[..., Any] | None = None,
    ) -> Callable[..., Any]:
        """Get or create a DSPy component with automatic fallback handling.


        Args:
            component_key: Unique key for caching the component
            signature_class: DSPy signature class
            component_type: "ChainOfThought" or "Predict"
            fallback_func: Optional fallback function if DSPy fails

        Returns:
            A callable that executes DSPy operation with fallback
        """
        if component_key not in self._components:
            try:
                if component_type == "ChainOfThought":
                    component = dspy.ChainOfThought(signature_class)
                elif component_type == "Predict":
                    component = dspy.Predict(signature_class)
                else:
                    raise ValueError(f"Unknown component type: {component_type}")

                self._components[component_key] = component
                logger.debug(f"Created DSPy component: {component_key}")

            except Exception as e:
                logger.debug(f"Failed to create DSPy component {component_key}: {e}")
                self._components[component_key] = None

        component = self._components[component_key]

        def execute_with_fallback(*args: Any, **kwargs: Any) -> Any:
            """Execute DSPy operation with automatic fallback."""
            if not self._configured or component is None:
                if fallback_func:
                    logger.debug(f"Using fallback for {component_key}")
                    return fallback_func(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"DSPy not configured and no fallback provided for {component_key}"
                    )

            try:
                result = component(*args, **kwargs)
                logger.debug(f"DSPy component {component_key} executed successfully")
                return result
            except Exception as e:
                logger.debug(f"DSPy component {component_key} failed: {e}")
                if fallback_func:
                    logger.debug(f"Using fallback for {component_key}")
                    return fallback_func(*args, **kwargs)
                else:
                    raise

        return execute_with_fallback

    def with_fallback(
        self,
        operation: Callable[..., Any],
        fallback: Callable[..., Any],
        operation_name: str = "DSPy operation",
    ) -> Any:
        """Execute any DSPy operation with automatic fallback.

        Args:
            operation: The DSPy operation to execute
            fallback: The fallback function to use if DSPy fails
            operation_name: Name of the operation for logging

        Returns:
            Result from either DSPy operation or fallback
        """
        if not self._configured:
            logger.debug(f"DSPy not configured, using fallback for {operation_name}")
            return fallback()

        try:
            result = operation()
            logger.debug(f"{operation_name} executed successfully")
            return result
        except Exception as e:
            logger.debug(f"{operation_name} failed: {e}, using fallback")
            return fallback()

    def is_configured(self) -> bool:
        """Check if DSPy is properly configured."""
        return self._configured

    def reconfigure(self, model: str) -> None:
        """Reconfigure DSPy with a different model.

        Args:
            model: New model name to use
        """
        self.model = model
        self._components.clear()  # Clear cached components
        self._configure_dspy()


# Global instance for use across the codebase
_dspy_manager: DSPyComponentManager | None = None


def get_dspy_manager(model: str = "gpt-4o-mini") -> DSPyComponentManager:
    """Get the global DSPy component manager instance.

    Args:
        model: Model name to use (only used on first call)

    Returns:
        The global DSPyComponentManager instance
    """
    global _dspy_manager
    if _dspy_manager is None:
        _dspy_manager = DSPyComponentManager(model)
    return _dspy_manager


def reset_dspy_manager() -> None:
    """Reset the global DSPy manager (useful for testing)."""
    global _dspy_manager
    _dspy_manager = None
