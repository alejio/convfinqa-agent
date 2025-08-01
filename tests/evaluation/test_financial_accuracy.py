"""
Financial accuracy evaluation tests using proper DeepEval custom metrics.
"""

import os
import pytest
from deepeval import assert_test, evaluate
from deepeval.metrics.answer_relevancy.answer_relevancy import AnswerRelevancyMetric
from deepeval.metrics.faithfulness.faithfulness import FaithfulnessMetric
from deepeval.test_case.conversational_test_case import ConversationalTestCase, Turn
from deepeval.test_case.llm_test_case import LLMTestCase

from src.evaluation.metrics import (
    create_financial_accuracy_metric,
    create_financial_relevance_metric,
    create_conversational_financial_metric,
    get_financial_metrics,
)


def _should_run_evaluation() -> bool:
    """Check if evaluation tests should run based on environment variables."""
    return os.environ.get("RUN_EVALUATION", "").lower() == "true"


def _has_valid_openai_key() -> bool:
    """Check if we have a valid OpenAI API key for evaluation tests."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return api_key and api_key != "test-key" and len(api_key) > 10


@pytest.mark.evaluation
@pytest.mark.slow
class TestFinancialAccuracy:
    """Test suite for financial evaluation using proper DeepEval custom metrics."""

    def test_financial_accuracy_custom_metric(self, financial_test_case):
        """Test custom financial accuracy metric."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        metric = create_financial_accuracy_metric(threshold=0.5)
        assert_test(financial_test_case, [metric])

    def test_financial_relevance_custom_metric(self, financial_test_case):
        """Test custom financial relevance metric."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        metric = create_financial_relevance_metric(threshold=0.5)
        assert_test(financial_test_case, [metric])

    def test_built_in_metrics_comparison(self, financial_test_case):
        """Test comparison between built-in and custom metrics."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        # Only use metrics that work with our test case structure
        built_in_metrics = [
            AnswerRelevancyMetric(threshold=0.5),
            # Skip FaithfulnessMetric - it requires retrieval_context
        ]

        custom_metrics = [
            create_financial_accuracy_metric(threshold=0.5),
            create_financial_relevance_metric(threshold=0.5),
        ]

        # Test both sets of metrics
        assert_test(financial_test_case, built_in_metrics)  # type: ignore
        assert_test(financial_test_case, custom_metrics)  # type: ignore

    def test_financial_evaluation_with_dataset(self, deepeval_dataset):
        """Test financial evaluation using DeepEval dataset with custom metrics."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        # Convert goldens to test cases
        for golden in deepeval_dataset.goldens:
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=golden.expected_output,
                expected_output=golden.expected_output,
                context=golden.context,
                retrieval_context=golden.context  # Add for FaithfulnessMetric
            )
            deepeval_dataset.add_test_case(test_case)

        if not deepeval_dataset.test_cases:
            pytest.skip("No test cases in dataset")

        # Test with our custom financial metrics
        metrics = get_financial_metrics(threshold=0.4)  # Lower threshold for mock responses

        # Evaluate using DeepEval's evaluate function
        evaluate(test_cases=deepeval_dataset.test_cases, metrics=metrics)  # type: ignore

    @pytest.mark.parametrize("threshold", [0.3, 0.5, 0.7])
    def test_custom_metric_thresholds(self, deepeval_dataset, threshold):
        """Test custom metrics with different thresholds."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        if not deepeval_dataset.goldens:
            pytest.skip("No goldens in dataset")

        golden = deepeval_dataset.goldens[0]
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=golden.expected_output,
            expected_output=golden.expected_output,
            context=golden.context,
            retrieval_context=golden.context
        )

        metric = create_financial_accuracy_metric(threshold=threshold)
        assert_test(test_case, [metric])

    def test_conversational_financial_evaluation(self, deepeval_dataset):
        """Test conversational financial evaluation - the key feature for ConvFinQA!"""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        if len(deepeval_dataset.goldens) < 2:
            pytest.skip("Need at least 2 goldens for conversational test")

        # Create realistic conversation turns alternating between user and assistant
        turns = []
        for i, golden in enumerate(deepeval_dataset.goldens[:3]):  # Max 3 turns (1-2 pairs)
            # Add user question (even indices)
            if i % 2 == 0:
                user_turn = Turn(
                    role="user",
                    content=golden.input,  # Actual financial question from dataset
                )
                turns.append(user_turn)

                # Add assistant response (odd indices)
                assistant_turn = Turn(
                    role="assistant",
                    content=golden.expected_output,  # Actual financial answer from dataset
                )
                turns.append(assistant_turn)

        # Ensure we have at least one complete user-assistant pair
        if len(turns) < 2:
            pytest.skip("Need at least one user-assistant turn pair")

        # Create conversational test case with proper chatbot_role as required by DeepEval
        convo_test_case = ConversationalTestCase(
            chatbot_role="Financial Analyst Assistant that provides accurate financial analysis based on company documents and tables",
            turns=turns
        )

        # Test with conversational financial metric (lower threshold for realistic evaluation)
        metric = create_conversational_financial_metric(threshold=0.3)
        assert_test(convo_test_case, [metric])

    def test_mixed_evaluation_approach(self, deepeval_dataset):
        """Test mixing built-in metrics with custom financial metrics."""
        if not _should_run_evaluation():
            pytest.skip("Set RUN_EVALUATION=true to run evaluation tests")

        if not _has_valid_openai_key():
            pytest.skip("Valid OpenAI API key required for evaluation tests")
        if not deepeval_dataset.goldens:
            pytest.skip("No goldens in dataset")

        golden = deepeval_dataset.goldens[0]
        test_case = LLMTestCase(
            input=golden.input,
            actual_output=golden.expected_output,
            expected_output=golden.expected_output,
            context=golden.context,
            # Add retrieval_context for FaithfulnessMetric
            retrieval_context=golden.context or ["Financial data from ConvFinQA dataset"]
        )

        # Mix built-in and custom metrics
        mixed_metrics = [
            AnswerRelevancyMetric(threshold=0.4),          # Built-in
            create_financial_accuracy_metric(threshold=0.4), # Custom
            FaithfulnessMetric(threshold=0.4),              # Built-in (now with retrieval_context)
            create_financial_relevance_metric(threshold=0.4), # Custom
        ]

        assert_test(test_case, mixed_metrics)
