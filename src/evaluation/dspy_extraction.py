"""
DSPy-based intelligent numeric answer extraction for ConvFinQA evaluation.

This module provides context-aware numeric extraction that understands financial questions
and can distinguish between incidental numbers and the actual answer.
"""

from typing import Any

import dspy

from ..core.logger import get_logger

logger = get_logger(__name__)


class NumericAnswerExtractor(dspy.Signature):
    """Extract the primary numeric answer from a verbose financial response.

    This signature is optimized for ConvFinQA evaluation accuracy by focusing on
    the main financial calculation result while ignoring contextual numbers.
    """

    question: str = dspy.InputField(
        desc="The original financial question that was asked"
    )
    verbose_answer: str = dspy.InputField(
        desc="The complete response from the financial agent containing calculations and reasoning"
    )
    expected_format: str = dspy.InputField(
        desc="Expected format hint based on ground truth (e.g., 'decimal number', 'percentage', 'currency')"
    )

    numeric_value: str = dspy.OutputField(
        desc="ONLY the final numeric answer as a clean number (e.g., '60.94', '-4.2', '0.1083'). No units, symbols, or explanations."
    )


class FinancialContextAnalyzer(dspy.Signature):
    """Analyze financial context to better understand what the question is asking for."""

    question: str = dspy.InputField(desc="The financial question being asked")
    table_context: str = dspy.InputField(desc="Available financial table data context")

    question_type: str = dspy.OutputField(
        desc="Type: 'lookup', 'calculation', 'comparison', 'change_analysis'"
    )
    expected_units: str = dspy.OutputField(
        desc="Expected units: 'currency', 'percentage', 'count', 'ratio', 'plain_number'"
    )
    confidence: str = dspy.OutputField(desc="Confidence level: 'high', 'medium', 'low'")


class FinancialAnswerNormalizer(dspy.Module):
    """Enhanced DSPy module for intelligent financial answer extraction and normalization."""

    def __init__(self) -> None:
        super().__init__()
        self.context_analyzer = dspy.ChainOfThought(FinancialContextAnalyzer)
        self.extract_answer = dspy.ChainOfThought(NumericAnswerExtractor)

    def forward(
        self,
        question: str,
        verbose_answer: str,
        expected_answer: str = "",
        table_context: str = "",
    ) -> str:
        """Extract the primary numeric value from a verbose financial answer using enhanced DSPy analysis.

        Args:
            question: The original question asked
            verbose_answer: The verbose response from the agent
            expected_answer: Expected answer format for context (optional)
            table_context: Financial table context for better understanding (optional)

        Returns:
            The extracted numeric value as a string
        """
        try:
            # First, analyze the context to understand what we're looking for
            if table_context:
                context_analysis = self.context_analyzer(
                    question=question,
                    table_context=table_context[:500],  # Limit context size
                )
                logger.debug(
                    f"DSPy context analysis - Type: {context_analysis.question_type}, Units: {context_analysis.expected_units}"
                )

            # Determine expected format from the expected answer or context analysis
            expected_format = self._infer_format_intelligently(
                expected_answer, question
            )

            # Use enhanced DSPy extraction
            result = self.extract_answer(
                question=question,
                verbose_answer=verbose_answer,
                expected_format=expected_format,
            )

            # Clean and validate the extracted value
            cleaned_value = self._clean_numeric_value(result.numeric_value)
            logger.debug(
                f"DSPy extracted: '{result.numeric_value}' -> cleaned: '{cleaned_value}'"
            )

            return cleaned_value

        except Exception as e:
            logger.debug(f"DSPy extraction failed: {e}")
            # Fallback to basic extraction
            return self._fallback_extraction(verbose_answer)

    def _infer_format_intelligently(self, expected_answer: str, question: str) -> str:
        """Intelligently infer the expected format using both expected answer and question context."""
        if not expected_answer and not question:
            return "numeric value"

        # Analyze expected answer format
        if expected_answer:
            expected_clean = expected_answer.strip().lower()

            if expected_clean.endswith("%") or "percent" in expected_clean:
                return "percentage as decimal (e.g., 0.1083 for 10.83%)"
            elif (
                expected_clean.startswith("$")
                or "dollar" in expected_clean
                or "million" in expected_clean
            ):
                return "currency amount as number"
            elif "." in expected_clean and len(expected_clean.split(".")[-1]) >= 2:
                return "decimal number with precision"
            elif expected_clean.startswith("-"):
                return "negative number"
            elif expected_clean.isdigit():
                return "whole number"

        # Analyze question context
        question_lower = question.lower()
        if any(word in question_lower for word in ["percent", "percentage", "rate"]):
            return "percentage as decimal"
        elif any(
            word in question_lower
            for word in ["increase", "decrease", "change", "difference"]
        ):
            return "change value (can be positive or negative)"
        elif any(word in question_lower for word in ["ratio", "times", "multiple"]):
            return "ratio as decimal"

        return "numeric value"

    def _fallback_extraction(self, verbose_answer: str) -> str:
        """Simple fallback extraction when DSPy fails."""
        import re

        numbers = re.findall(r"-?\d+\.?\d*", verbose_answer)
        if numbers:
            return str(numbers[-1])  # Ensure return type is str
        return "0"

    def _clean_numeric_value(self, value: str) -> str:
        """Clean and normalize the extracted numeric value."""
        if not value:
            return ""

        # Remove common prefixes/suffixes but preserve the core number
        cleaned = value.strip()

        # Remove currency symbols but keep the number
        cleaned = cleaned.replace("$", "")

        # Handle percentage - remove % but keep the number
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]

        # Remove commas
        cleaned = cleaned.replace(",", "")

        # Ensure it's a valid number format
        try:
            float(cleaned)
            return cleaned
        except ValueError:
            # If we can't parse it, return as-is and let downstream handle it
            return value


class DSPyExecutionAccuracyMetric:
    """Enhanced ExecutionAccuracyMetric using DSPy for intelligent answer extraction."""

    def __init__(self, threshold: float = 1.0, lm_model: dspy.LM | None = None) -> None:
        """Initialize the DSPy-enhanced metric.

        Args:
            threshold: Threshold for success (1.0 for exact match)
            lm_model: DSPy language model to use (if None, uses default)
        """
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False

        # Initialize DSPy components
        if lm_model:
            dspy.configure(lm=lm_model)

        self.answer_normalizer = FinancialAnswerNormalizer()

    @property
    def __name__(self) -> str:
        return "DSPy Execution Accuracy"

    def measure(self, test_case: Any, question: str = "") -> float:
        """Measure execution accuracy using DSPy-powered extraction.

        Args:
            test_case: The test case containing actual_output and expected_output
            question: Original question for context (optional)

        Returns:
            Score of 1.0 for exact match, 0.0 otherwise
        """
        if not test_case.actual_output or not test_case.expected_output:
            self.reason = "Missing actual_output or expected_output"
            self.score = 0.0
            self.success = False
            return self.score

        try:
            # Use DSPy to intelligently extract the numeric answer
            actual_extracted = self.answer_normalizer(
                question=question or getattr(test_case, "input", ""),
                verbose_answer=test_case.actual_output,
                expected_answer=test_case.expected_output,
            )

            # Clean the expected answer consistently
            expected_extracted = self.answer_normalizer(
                question=question or getattr(test_case, "input", ""),
                verbose_answer=test_case.expected_output,  # Treat expected as if it were a response
                expected_answer=test_case.expected_output,
            )

            # Compare the extracted values
            if self._numeric_match(actual_extracted, expected_extracted):
                self.score = 1.0
                self.reason = (
                    f"DSPy match: '{actual_extracted}' == '{expected_extracted}'"
                )
                self.success = True
            else:
                self.score = 0.0
                self.reason = (
                    f"DSPy no match: '{actual_extracted}' != '{expected_extracted}'"
                )
                self.success = False

        except Exception as e:
            self.score = 0.0
            self.reason = f"DSPy extraction failed: {str(e)}"
            self.success = False

        return self.score

    def _numeric_match(self, actual: str, expected: str) -> bool:
        """Check if two numeric values match exactly."""
        if not actual or not expected:
            return actual == expected

        try:
            # Convert to floats and compare with small tolerance
            actual_float = float(actual)
            expected_float = float(expected)
            return abs(actual_float - expected_float) < 1e-6
        except (ValueError, TypeError):
            # Fall back to string comparison if float conversion fails
            return actual.strip() == expected.strip()

    def is_successful(self) -> bool:
        """Check if the metric was successful."""
        return self.success


# Convenience function for testing
def test_dspy_extraction() -> None:
    """Test the DSPy extraction with sample cases."""
    # Initialize with a mock model for testing
    normalizer = FinancialAnswerNormalizer()

    test_cases = [
        {
            "question": "What was the weighted average exercise price per share in 2007?",
            "verbose_answer": "The weighted average exercise price per share in 2007 was $60.94.",
            "expected": "60.94",
        },
        {
            "question": "What was the change in the unamortized debt issuance costs between 2016 and 2017?",
            "verbose_answer": "The change in the unamortized debt issuance costs associated with the senior notes between 2016 and 2017 is 1.0.",
            "expected": "-4",
        },
        {
            "question": "What is the ratio?",
            "verbose_answer": "The ratio of discretionary company contributions to total expensed amounts for savings plans in 2009 is 0.1083",
            "expected": "0.1083",
        },
    ]

    logger.info("=== DSPy Extraction Test ===")
    for i, case in enumerate(test_cases, 1):
        try:
            extracted = normalizer(
                question=case["question"],
                verbose_answer=case["verbose_answer"],
                expected_answer=case["expected"],
            )
            logger.info(f"Test {i}:")
            logger.info(f"  Question: {case['question']}")
            logger.info(f"  Verbose: {case['verbose_answer']}")
            logger.info(f"  Expected: {case['expected']}")
            logger.info(f"  Extracted: {extracted}")
            logger.info(f"  Match: {extracted == case['expected']}")
            logger.info("")
        except Exception as e:
            logger.error(f"Test {i} failed: {e}")


if __name__ == "__main__":
    test_dspy_extraction()
